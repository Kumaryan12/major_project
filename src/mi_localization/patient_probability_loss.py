"""Compare beat loss and patient probability loss on matched measured XYZ batches."""
import argparse
import json
import math
from pathlib import Path
import platform

import numpy as np
import sklearn
import torch
from torch.nn import functional as F
import yaml

from . import benchmark, contrastive, masked_pretraining, model_error_comparison, multiview_bank
from . import neural, patient_probability_report, preprocessing_study, regularization, residual_context_report, transfer
from .contrastive import _progress, initial_state_hash
from .multiview_bank import array_hash
from .neural import FoldNormalizer, _device, _hash, _save_json
from .signal_audit import file_hash

PATIENT_WEIGHTS = {'beat_ce': 0.0, 'patient_ce': 1.0, 'hybrid_ce': 0.5}


def cross_entropies(logits, targets):
    """One class per bag. Compare mean log-probability with log mean probability."""
    log_p = F.log_softmax(logits, dim=-1)
    true_log_p = log_p.gather(-1, targets[:, None, None].expand(-1, logits.shape[1], 1)).squeeze(-1)
    return -true_log_p.mean(), (math.log(logits.shape[1]) - torch.logsumexp(true_log_p, dim=1)).mean()


def sampling_plan(config, train, labels, groups, row_weights, seed):
    """Give classes equal mass, patients equal mass within class, and beats equal mass within patient."""
    training_groups = groups[train]
    pools = {patient: train[training_groups == patient] for patient in np.unique(training_groups)}
    rng = np.random.default_rng(seed)
    n_beats = config['beats_per_bag']
    batches = config['samples_per_epoch'] // config['batch_size']
    bags_per_batch = config['batch_size'] // n_beats
    shape = (config['target_epochs'], batches, bags_per_batch)
    # The anchor only selects a patient. Its aggregate probability is the sum of its training beat weights.
    anchors = rng.choice(train, size=shape, p=row_weights / row_weights.sum())
    rows = np.stack([rng.choice(pools[groups[anchor]], size=n_beats, replace=True)
                     for anchor in anchors.ravel()]).reshape(*shape, n_beats)
    if not (groups[rows] == groups[anchors][..., None]).all() or not (labels[rows] == labels[anchors][..., None]).all():
        raise ValueError('Expected one training patient and class per bag')
    return rows


def initial_model(config, seed):
    return transfer.initial_model(config, seed, 'scratch', None)


def fit(config, data, norm, sampling, context, arm, root):
    device = _device(config)
    model = initial_model(config, context['model_seed'])
    if initial_state_hash(model) != context['initial_model_sha256']:
        raise ValueError('ResNet initialization differs across arms')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    weight = PATIENT_WEIGHTS[arm]
    waves, labels = data['waves'], data['labels']
    mean, scale = norm.wave_mean_[:, None], norm.wave_scale_[:, None]
    history = []
    for epoch, batches in enumerate(sampling, 1):
        model.train()
        sums = np.zeros(3)
        for bags in batches:
            wave = (waves[bags.ravel()] - mean) / scale
            x = torch.from_numpy(wave).to(device)
            y = torch.from_numpy(labels[bags[:, 0]]).to(device)
            optimizer.zero_grad(set_to_none=True)
            beat_loss, patient_loss = cross_entropies(model(x).reshape(*bags.shape, -1), y)
            loss = torch.lerp(beat_loss, patient_loss, weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite patient probability loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            sums += torch.stack([beat_loss, patient_loss, loss]).detach().cpu().numpy()
        beat, patient, objective = (sums / len(batches)).tolist()
        row = dict(epoch=epoch, beat_loss=beat, patient_loss=patient, objective=objective,
                   sampling_sha256=array_hash(batches), beat_presentations=batches.size)
        history.append(row)
        _progress(root, state='training', fold=context['fold'], seed=context['seed'], arm=arm, epoch=epoch)
        print(f"Fold {context['fold']}, seed {context['seed']}, {arm}, epoch {epoch}/{len(sampling)}", flush=True)
    return model, history


def restore_or_fit(config, data, norm, sampling, context, arm, query, root, verify):
    run_name = f"fold{context['fold']:02d}_seed{context['seed']}_{arm}"
    folder = root / 'review' / run_name
    audit_path, model_path, prob_path = (folder / name for name in ('audit.json', 'model.pt', 'predictions.npz'))
    expected = dict(context, arm=arm, patient_loss_weight=PATIENT_WEIGHTS[arm])
    exists = audit_path.is_file()
    if verify and not exists:
        raise ValueError(f'Missing required completed fit {run_name}')
    if exists:
        receipt = json.loads(audit_path.read_text())
        if (receipt['context'] != expected or receipt['model_sha256'] != file_hash(model_path)
                or receipt['predictions_sha256'] != file_hash(prob_path)):
            raise ValueError(f'Checkpoint provenance changed for {run_name}')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if checkpoint['context'] != expected:
            raise ValueError(f'Checkpoint context changed for {run_name}')
        model = initial_model(config, context['model_seed'])
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(_device(config))
        history = receipt['history']
    else:
        model, history = fit(config, data, norm, sampling, context, arm, root)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError(f'Checkpoint parameters are nonfinite for {run_name}')
    if len(history) != config['target_epochs']:
        raise ValueError(f'Wrong epoch count for {run_name}')
    weight = PATIENT_WEIGHTS[arm]
    for epoch, (row, batches) in enumerate(zip(history, sampling), 1):
        loss = (1 - weight) * row['beat_loss'] + weight * row['patient_loss']
        if (row['epoch'] != epoch or row['sampling_sha256'] != array_hash(batches)
                or row['beat_presentations'] != batches.size or not np.isfinite(loss)
                or not np.isclose(loss, row['objective'], atol=1e-6)):
            raise ValueError(f'Objective, sampling, or exposure changed for {run_name}')
    p = preprocessing_study.predict_waves(model, norm, data['waves'], data['labels'], query, config)
    if exists:
        with np.load(prob_path, allow_pickle=False) as saved:
            if (not np.array_equal(saved['rows'], query)
                    or not np.allclose(saved['probabilities'], p, rtol=2e-5, atol=2e-5)):
                raise ValueError(f'Held-out inference does not replay for {run_name}')
            # Use the frozen probabilities after independently restoring and checking them.
            p = saved['probabilities']
    else:
        folder.mkdir(parents=True, exist_ok=True)
        masked_pretraining._save_checkpoint(model_path, dict(context=expected, model=model.state_dict()))
        temporary = prob_path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, rows=query, probabilities=p)
        temporary.replace(prob_path)
        _save_json(audit_path, dict(context=expected, history=history, model_sha256=file_hash(model_path),
                                    predictions_sha256=file_hash(prob_path)))
    return run_name, p, file_hash(audit_path)


def load_setup(config_path, smoke):
    study_config = yaml.safe_load(Path(config_path).read_text())
    reference_path = Path('configs/model_error_comparison_v1.yaml')
    references = yaml.safe_load(reference_path.read_text())
    baseline, baseline_scope, _ = model_error_comparison.read_source(
        Path(references['waveform_root']), references['waveform_summary_sha256'], 'fingerprint')
    base_config = yaml.safe_load(Path(study_config['base_config']).read_text())
    if dict(base_config, max_epochs=base_config['target_epochs']) != baseline_scope['config']:
        raise ValueError('Waveform optimizer and architecture must match the frozen reference')
    config = dict(base_config, output_dir=study_config['output_dir'], beats_per_bag=study_config['beats_per_bag'])
    if smoke:
        config.update(target_epochs=1, samples_per_epoch=256)
    if (config['beats_per_bag'] != 8 or config['batch_size'] % config['beats_per_bag']
            or config['samples_per_epoch'] % config['batch_size']):
        raise ValueError('Require fixed eight-beat bags and full matched batches')
    fold_path = Path(config['folds_path'])
    if not fold_path.is_file() or file_hash(fold_path) != baseline_scope['folds_sha256']:
        raise ValueError('Frozen target folds required')
    waves, labels, groups, bank = regularization.load_bank(config)
    if bank != baseline_scope['bank'] or (bank['n_patients'], bank['n_records'], bank['n_beats']) != (164, 391, 55863):
        raise ValueError('Original target bank membership or contents changed')
    assigned = benchmark.load_or_create_folds(labels, groups, config).set_index('patient_id').loc[groups, 'fold'].to_numpy()
    if set(assigned) != set(range(1, config['outer_splits'] + 1)):
        raise ValueError('Expected all frozen folds')
    modules = (benchmark, contrastive, masked_pretraining, model_error_comparison, multiview_bank, neural,
               patient_probability_report, preprocessing_study, regularization, residual_context_report, transfer)
    paths = [Path(m.__file__) for m in modules] + [Path(__file__), Path('scripts/run_patient_probability_loss.py')]
    scope = dict(config=config, smoke=smoke, weights=PATIENT_WEIGHTS, bank=bank, folds_sha256=file_hash(fold_path),
                 waveform_summary_sha256=references['waveform_summary_sha256'],
                 waveform_reference_macro_f1=baseline['ensemble_metrics']['scratch']['macro_f1'],
                 convolution_reference_macro_f1=0.603374,
                 runtime=dict(numpy=np.__version__, sklearn=sklearn.__version__, torch=str(torch.__version__),
                              device=str(_device(config)), platform=platform.platform()),
                 code={p.name: file_hash(p) for p in paths})
    scope['fingerprint'] = _hash(scope)
    return scope, dict(waves=waves, labels=labels, groups=groups, assigned=assigned)


def run(config_path, smoke=False, verify=False):
    scope, data = load_setup(config_path, smoke)
    config = scope['config']
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'study_manifest.json'
    if manifest.is_file() or verify:
        if json.loads(manifest.read_text()) != scope:
            raise ValueError('Study inputs, protocol, or runtime changed')
    else:
        _save_json(manifest, scope)
    assigned, labels, groups = data['assigned'], data['labels'], data['groups']
    seeds = config['training_seeds'][:1] if smoke else config['training_seeds']
    folds = [1] if smoke else range(1, config['outer_splits'] + 1)
    predictions = {a: {s: np.full((len(labels), len(config['classes'])), np.nan) for s in seeds}
                   for a in PATIENT_WEIGHTS} if verify and not smoke else None
    audits, matched = {}, 0
    total = len(folds) * len(seeds) * len(PATIENT_WEIGHTS)
    for fold in folds:
        train = np.flatnonzero(assigned != fold)
        if smoke:
            ti, vi = next(benchmark.inner_patient_splits(labels[train], groups[train], 3, config['seed'] + fold * 1000))
            train, query = train[ti], train[vi]
        else:
            query = np.flatnonzero(assigned == fold)
        if set(groups[train]) & set(groups[query]):
            raise ValueError('Training and query patients overlap')
        weights = benchmark.patient_balanced_weights(labels[train], groups[train])
        norm = FoldNormalizer().fit(data['waves'], np.empty((len(labels), 0), dtype=np.float32), train, weights)
        for seed in seeds:
            model_seed = seed + fold
            sampling = sampling_plan(config, train, labels, groups, weights, model_seed)
            context = dict(fingerprint=scope['fingerprint'], fold=fold, seed=seed, model_seed=model_seed,
                           training_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query),
                           normalization=norm.audit(), sampling_sha256=array_hash(sampling),
                           initial_model_sha256=initial_state_hash(initial_model(config, model_seed)))
            for arm in PATIENT_WEIGHTS:
                if verify:
                    _progress(root, state='restoring', fold=fold, seed=seed, arm=arm, done=len(audits), total=total)
                name, p, digest = restore_or_fit(config, data, norm, sampling, context, arm, query, root, verify)
                audits[name] = digest
                if predictions is not None:
                    if np.isfinite(predictions[arm][seed][query]).any():
                        raise ValueError('Duplicate outer query rows')
                    predictions[arm][seed][query] = p
                _progress(root, state='restoring' if verify else 'fit_saved', fold=fold, seed=seed, arm=arm,
                          done=len(audits), total=total)
            matched += 1
            print(f"{'Restored' if verify else 'Saved'} {len(audits)}/{total} fits, fold {fold}, seed {seed}", flush=True)
    if len(audits) != total or len(list((root / 'review').glob('*/audit.json'))) != total:
        raise ValueError('Fit audits differ from the declared experiment')
    if verify:
        evidence = root / 'verification_manifest.json'
        _save_json(evidence, dict(fingerprint=scope['fingerprint'], fit_audit_hashes=audits))
        common = dict(complete=True, study_fingerprint=scope['fingerprint'], restored_models=total,
                      matched_fold_seed_groups=matched, verification_manifest_sha256=file_hash(evidence),
                      outer_test_patients_used=not smoke)
        if smoke:
            _save_json(root / 'summary.json', common)
        else:
            patient_probability_report.summarize(root, scope, data, predictions, common)
    _progress(root, state='complete_verified' if verify else 'all_fits_saved', done=total, total=total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/patient_probability_loss_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    run(args.config, args.smoke, args.verify)
