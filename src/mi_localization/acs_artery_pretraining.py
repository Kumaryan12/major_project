"""Fit and replay matched ACS-ECG occlusion and occlusion-plus-artery encoders."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
import torch
from torch.nn import functional as F
import yaml

from . import acs_ecg_bank as bank, benchmark, neural, supervised_source_train as model_tools
from .contrastive import _progress, initial_state_hash
from .multiview_bank import array_hash
from .signal_audit import file_hash

OBJECTIVES = ('omi', 'omi_artery')


def objectives(logits, omi, artery):
    binary = F.binary_cross_entropy_with_logits(logits[:, 0], omi.to(logits.dtype))
    eligible = (artery >= 0).to(logits.dtype)
    losses = F.cross_entropy(logits[:, 1:], artery.clamp_min(0), reduction='none')
    auxiliary = (losses * eligible).sum() / eligible.sum().clamp_min(1)
    return binary, auxiliary


def source_schedule(c, arrays, rows):
    weights = benchmark.patient_balanced_weights(arrays['omi'][rows], arrays['groups'][rows])
    rng = np.random.default_rng(c['source_seed'])
    epochs = c['source_epochs']
    if c['source_samples_per_epoch'] % c['source_batch_size']:
        raise ValueError('Require complete fixed source batches')
    batches = c['source_samples_per_epoch'] // c['source_batch_size']
    return rng.choice(rows, size=(epochs, batches, c['source_batch_size']), p=weights / weights.sum())


def initial_model(c, target):
    neural._seed(c['source_seed'], target)
    return model_tools.RecordModel(1 + len(bank.AUXILIARY), target)


def predict(c, target, model, wave, mask):
    device = neural._device(target)
    model.eval()
    result = []
    with torch.no_grad():
        for x, beat_mask in zip(wave.split(c['source_batch_size']), mask.split(c['source_batch_size'])):
            logits = model(x.to(device), beat_mask.to(device))
            result.append(torch.cat([logits[:, :1].sigmoid(), logits[:, 1:].softmax(-1)], dim=1).cpu().numpy())
    p = np.concatenate(result)
    if not np.isfinite(p).all() or not np.allclose(p[:, 1:].sum(1), 1) or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid source validation probabilities')
    return p


def source_metrics(arrays, query, probabilities, objective):
    omi, artery, groups = (arrays[k][query] for k in ('omi', 'artery', 'groups'))
    weights = model_tools.patient_weights(groups)
    scores = dict(
        omi_average_precision=float(average_precision_score(omi, probabilities[:, 0], sample_weight=weights)),
        omi_prevalence=float(np.average(omi, weights=weights)),
    )
    if objective == 'omi_artery':
        valid = artery >= 0
        scores['artery_macro_f1'] = float(f1_score(
            artery[valid], probabilities[valid, 1:].argmax(1), average='macro',
            labels=np.arange(len(bank.AUXILIARY)), sample_weight=model_tools.patient_weights(groups[valid]),
            zero_division=0))
    return scores


def scope_for(c, target, bank_receipt, smoke):
    modules = (bank, benchmark, neural, model_tools)
    paths = [Path(m.__file__) for m in modules] + [Path(__file__)]
    return dict(config=c, target_settings=target, bank_receipt_sha256=neural._hash(bank_receipt), smoke=smoke,
                numpy=np.__version__, torch=str(torch.__version__), device=str(neural._device(target)),
                code={p.name: file_hash(p) for p in paths})


def train_or_replay(c, target, root, arrays, schedule, validation, prepared, scope, objective, verify):
    folder = root / 'review' / 'source' / objective
    folder.mkdir(parents=True, exist_ok=True)
    model_path, audit_path, probability_path = (folder / n for n in ('model.pt', 'audit.json', 'probabilities.npz'))
    wave, mask, omi, artery = prepared
    model = initial_model(c, target)
    context = dict(scope_sha256=neural._hash(scope), objective=objective, initial_sha256=initial_state_hash(model),
                   sampling_sha256=array_hash(schedule), validation_rows_sha256=array_hash(validation),
                   artery_weight=0.0 if objective == 'omi' else c['artery_weight'])
    device = neural._device(target)
    model.to(device)
    existing = audit_path.is_file()
    if verify and not existing:
        raise ValueError('Missing source model to replay')
    if existing:
        audit = bank.read(audit_path)
        if (audit['context'] != context or audit['model_sha256'] != file_hash(model_path)
                or audit['predictions_sha256'] != file_hash(probability_path)):
            raise ValueError('Source model provenance changed')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if checkpoint['context'] != context:
            raise ValueError('Source checkpoint context differs')
        model.load_state_dict(checkpoint['model'], strict=True)
        history = audit['history']
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=target['learning_rate'], weight_decay=target['weight_decay'])
        history = []
        for epoch, draws in enumerate(schedule, 1):
            model.train()
            totals = np.zeros(3)
            for indices in draws:
                optimizer.zero_grad(set_to_none=True)
                binary, auxiliary = objectives(model(wave[indices].to(device), mask[indices].to(device)),
                                               omi[indices].to(device), artery[indices].to(device))
                loss = binary + context['artery_weight'] * auxiliary
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite source objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                totals += torch.stack([loss, binary, auxiliary]).detach().cpu().numpy()
            loss, binary, auxiliary = (totals / len(draws)).tolist()
            history.append(dict(epoch=epoch, loss=loss, binary=binary, auxiliary=auxiliary,
                                rows_sha256=array_hash(draws), presentations=draws.size))
            _progress(root, state='source_training', objective=objective, epoch=epoch, epochs=len(schedule))
            print(f'ACS-ECG source {objective}, epoch {epoch}/{len(schedule)}', flush=True)
    if (len(history) != c['source_epochs'] or any(not torch.isfinite(v).all() for v in model.state_dict().values())):
        raise ValueError('Incomplete source epoch/model coverage')
    for epoch, (row, draws) in enumerate(zip(history, schedule), 1):
        if (row['epoch'] != epoch or row['rows_sha256'] != array_hash(draws) or row['presentations'] != draws.size
                or not np.isfinite([row['loss'], row['binary'], row['auxiliary']]).all()
                or not np.isclose(row['loss'], row['binary'] + context['artery_weight'] * row['auxiliary'], atol=1e-6)):
            raise ValueError('Source objective or budget did not replay')
    probabilities = predict(c, target, model, wave[validation], mask[validation])
    if existing:
        with np.load(probability_path, allow_pickle=False) as saved:
            if (not np.array_equal(saved['rows'], validation)
                    or not np.allclose(saved['probabilities'], probabilities, atol=2e-5, rtol=2e-5)):
                raise ValueError('Source checkpoint probabilities did not replay')
            probabilities = saved['probabilities']
    else:
        model_tools.save_checkpoint(model_path, dict(context=context, model=model.state_dict()))
        partial = probability_path.with_suffix('.partial.npz')
        np.savez_compressed(partial, rows=validation, probabilities=probabilities)
        partial.replace(probability_path)
        neural._save_json(audit_path, dict(context=context, history=history, model_sha256=file_hash(model_path),
                                          predictions_sha256=file_hash(probability_path)))
    encoder_path = folder / 'encoder.pt'
    encoder_hash = initial_state_hash(model.encoder)
    if not verify:
        return context, None
    encoder = dict(context=context, model=model.encoder.state_dict())
    if encoder_path.is_file():
        saved = torch.load(encoder_path, map_location='cpu', weights_only=True)
        if (saved['context'] != context or saved['model'].keys() != encoder['model'].keys()
                or any(not torch.equal(v, saved['model'][k].cpu()) for k, v in model.encoder.cpu().state_dict().items())):
            raise ValueError('Saved encoder export differs from restored source model')
    else:
        model_tools.save_checkpoint(encoder_path, encoder)
    return context, dict(
        metrics=source_metrics(arrays, validation, probabilities, objective),
        encoder_weights_sha256=encoder_hash, encoder_file_sha256=file_hash(encoder_path),
        audit_file_sha256=file_hash(audit_path),
    )


def run(config, smoke=False, verify=False):
    arrays, receipt = bank.load(config, smoke)
    target = yaml.safe_load(Path(config['target_config']).read_text())
    c = dict(config, source_epochs=1, source_samples_per_epoch=config['source_batch_size']) if smoke else config
    root = Path(c['output_dir']) / ('smoke' if smoke else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(arrays['roles'] == 'train')
    valid = np.flatnonzero(arrays['roles'] == 'validation')
    if len(train) + len(valid) != len(arrays['groups']) or set(arrays['groups'][train]) & set(arrays['groups'][valid]):
        raise ValueError('Source learning and validation patients overlap')
    norm = model_tools.fit_normalizer(arrays['waves'], arrays['counts'], arrays['groups'], train)
    mean, scale = np.asarray(norm['mean'], np.float32)[:, None], np.asarray(norm['scale'], np.float32)[:, None]
    wave = torch.from_numpy((arrays['waves'] - mean) / scale)
    mask = torch.from_numpy(np.arange(arrays['waves'].shape[1]) < arrays['counts'][:, None])
    prepared = wave, mask, torch.from_numpy(arrays['omi']), torch.from_numpy(arrays['artery'])
    schedule = source_schedule(c, arrays, train)
    scope = dict(scope_for(c, target, receipt, smoke), normalizer=norm)
    path = root / 'source_manifest.json'
    if (path.is_file() or verify) and bank.read(path) != scope:
        raise ValueError('Source learning scope changed')
    if not path.is_file():
        neural._save_json(path, scope)
    contexts, fits = {}, {}
    for objective in OBJECTIVES:
        contexts[objective], fits[objective] = train_or_replay(
            c, target, root, arrays, schedule, valid, prepared, scope, objective, verify)
    keys = ('scope_sha256', 'initial_sha256', 'sampling_sha256', 'validation_rows_sha256')
    if any(contexts['omi'][k] != contexts['omi_artery'][k] for k in keys):
        raise ValueError('Source arms do not match in exposure or initialization')
    if verify:
        result = dict(complete=True, bank_receipt_sha256=scope['bank_receipt_sha256'],
                      source_manifest_sha256=file_hash(path), source_bank=receipt['summary'],
                      smoke=smoke, source_epochs=c['source_epochs'], models=2, matched_sampling=True, fits=fits)
        evidence = root / 'source_summary.json'
        if evidence.is_file() and bank.read(evidence) != result:
            raise ValueError('Source verification summary did not reproduce')
        neural._save_json(evidence, result)
        _progress(root, state='source_verified', models=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/acs_artery_transfer_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.smoke, args.verify)
