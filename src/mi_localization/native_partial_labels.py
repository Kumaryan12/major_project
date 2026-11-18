"""Matched six-class training with positive-only auxiliary native location mentions."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import yaml

from . import benchmark, labels, masked_pretraining, native_partial_source, neural, preprocessing_study
from . import regularization, territory_study, vcg_expansion_bank
from .contrastive import _progress, initial_state_hash
from .multiview_bank import array_hash
from .neural import _device, _hash, _save_json, FoldNormalizer
from .native_partial_report import summarize
from .signal_audit import file_hash
from .vcg_pretraining_study import check_finite_state

ARMS = ('scratch', 'target_descriptors', 'shuffled_source', 'true_source')


def read(path):
    return json.loads(path.read_text())


def shuffle_mentions(codes, seed):
    n = len(codes)
    rng = np.random.default_rng(seed)
    same = np.all(codes[:, None, :] == codes[None, :, :], axis=2)
    costs = same.astype(float) * (n + 1) + rng.random((n, n))
    np.fill_diagonal(costs, np.inf)
    rows, partners = linear_sum_assignment(costs)
    if not np.array_equal(rows, np.arange(n)) or np.any(rows == partners):
        raise ValueError('Source control must permute patient code vectors with no self-pairs')
    return partners


def positive_loss(logits, observed):
    """Only named positives are in the objective; all other gradients are zero."""
    return (F.softplus(-logits) * observed).sum(dim=1).div(observed.sum(dim=1)).mean()


def make_plans(config, train, weights, pools, seed):
    torch_rng = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(weights, dtype=torch.double)
    batches = config['samples_per_epoch'] // config['batch_size']
    target, source = [], []
    rng = np.random.default_rng(seed + 2813)
    for _ in range(config['target_epochs']):
        draws = torch.multinomial(weights, config['samples_per_epoch'], replacement=True, generator=torch_rng)
        target.append(train[draws.numpy()].reshape(batches, config['batch_size']))
        source.append(np.stack([
            np.concatenate([rng.choice(pool, config['source_beats_per_patient_per_update'], replace=True)
                            for pool in pools])
            for _ in range(batches)
        ]))
    return np.stack(target), np.stack(source)


def loss_weights(config, arm):
    return (
        config['target_descriptor_weight'] if arm != 'scratch' else 0.,
        config['source_positive_weight'] if arm in ARMS[2:] else 0.,
    )


def fit(config, data, seed, arm, norm, target, source, member_codes, root, context):
    model = territory_study.initial_model(config, seed).to(_device(config))
    device = _device(config)
    initial = initial_state_hash(model)
    region_weight, source_weight = loss_weights(config, arm)
    codebook = torch.as_tensor(territory_study.codebook(config), device=device)
    masks = torch.as_tensor(member_codes, device=device)
    mean, scale = norm.wave_mean_[None, :, None], norm.wave_scale_[None, :, None]
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    wave, source_wave, labels_index = data['wave'], data['source_wave'], data['y']
    history = []
    for index, (target_epoch, source_epoch) in enumerate(zip(target, source), 1):
        model.train()
        sums = np.zeros(4, dtype=float)
        for target_rows, source_rows in zip(target_epoch, source_epoch):
            x = torch.from_numpy((wave[target_rows] - mean) / scale).to(device)
            y = torch.as_tensor(labels_index[target_rows], device=device)
            opt.zero_grad(set_to_none=True)
            logits, auxiliary = model(x, return_auxiliary=True)
            ce = F.cross_entropy(logits, y)
            target_aux = F.binary_cross_entropy_with_logits(auxiliary, codebook[y])
            source_aux = x.new_zeros(())
            if source_weight:
                sx = torch.from_numpy((source_wave[source_rows] - mean) / scale).to(device)
                source_aux = positive_loss(model.auxiliary(model.encoder(sx)), masks)
            loss = ce + region_weight * target_aux + source_weight * source_aux
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite partial-label objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            sums += [float(t.detach().cpu()) for t in (loss, ce, target_aux, source_aux)]
        total, ce, target_loss, source_loss = (sums / len(target_epoch)).tolist()
        row = dict(epoch=index, loss=total, target_ce=ce, target_descriptors=target_loss,
                   source_positive=source_loss, target_rows_sha256=array_hash(target_epoch),
                   source_rows_sha256=array_hash(source_epoch) if source_weight else None)
        history.append(row)
        _progress(root, state='training', fold=context['fold'], seed=context['seed'], arm=arm, **row)
    return model, dict(initial_sha256=initial, normalization=norm.audit(), history=history)


def run_arm(config, data, root, context, arm, train, query, norm, plans, observed, verify):
    target, source = plans
    base_seed = context['seed'] + context['fold']
    target_weight, source_weight = loss_weights(config, arm)
    if source_weight and not (observed.sum(axis=1) > 0).all():
        raise ValueError('Source batches require observed positive tags')
    scope = dict(context, arm=arm, target_weight=target_weight, source_weight=source_weight,
                 source_tags_sha256=array_hash(observed) if source_weight else None)
    path = root / 'review' / f"f{context['fold']:02d}_s{context['seed']}_{arm}"
    path.mkdir(parents=True, exist_ok=True)
    audit_path, model_path, p_path = (path / name for name in ('audit.json', 'model.pt', 'probabilities.npz'))
    existing = audit_path.exists()
    if verify and not existing:
        raise ValueError('Required completed fit is missing')
    if existing:
        saved = read(audit_path)
        if (saved['scope'] != scope or saved['model_sha256'] != file_hash(model_path)
                or saved['probability_sha256'] != file_hash(p_path)):
            raise ValueError('Frozen fit artifacts changed')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if checkpoint['scope'] != scope:
            raise ValueError('Checkpoint context changed')
        model = territory_study.initial_model(config, base_seed)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(_device(config))
        fit_audit = saved['fit']
    else:
        model, fit_audit = fit(config, data, base_seed, arm, norm, target, source, observed, root, context)
    check_finite_state(model.state_dict())
    initial = initial_state_hash(territory_study.initial_model(config, base_seed))
    if (fit_audit['initial_sha256'] != initial or fit_audit['normalization'] != norm.audit()
            or len(fit_audit['history']) != len(target)):
        raise ValueError('Mismatch in initialization, normalization or epoch count')
    for i, (row, target_epoch, source_epoch) in enumerate(zip(fit_audit['history'], target, source), 1):
        expected_loss = row['target_ce'] + target_weight * row['target_descriptors'] + source_weight * row['source_positive']
        if (row['epoch'] != i or not np.isfinite(expected_loss)
                or not np.isclose(row['loss'], expected_loss, atol=1e-6)
                or row['target_rows_sha256'] != array_hash(target_epoch)
                or row['source_rows_sha256'] != (array_hash(source_epoch) if source_weight else None)):
            raise ValueError('Partial-label objective or exact sampled rows changed')
    p = preprocessing_study.predict_waves(model, norm, data['wave'], data['y'], query, config)
    if not np.isfinite(p).all() or np.any(p < 0.) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Invalid model probabilities')
    if existing:
        with np.load(p_path, allow_pickle=False) as value:
            if not np.array_equal(value['rows'], query) or not np.allclose(value['probabilities'], p, rtol=2e-5, atol=2e-5):
                raise ValueError('Held-out probability replay differs')
    else:
        masked_pretraining._save_checkpoint(model_path, dict(scope=scope, model=model.state_dict()))
        temporary = p_path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, rows=query, probabilities=p)
        temporary.replace(p_path)
        _save_json(audit_path, dict(scope=scope, fit=fit_audit,
                                   model_sha256=file_hash(model_path), probability_sha256=file_hash(p_path)))
    return p, file_hash(audit_path)


def load_settings(path, smoke):
    partial = yaml.safe_load(Path(path).read_text())
    base = yaml.safe_load(Path(partial['base_config']).read_text())
    config = dict(base, **partial)
    if smoke:
        config.update(target_epochs=1, samples_per_epoch=32, batch_size=16, source_beats_per_patient_per_update=1)
    if (config['samples_per_epoch'] % config['batch_size'] or config['target_epochs'] < 1
            or config['samples_per_epoch'] < config['batch_size']
            or config['source_beats_per_patient_per_update'] < 1
            or not np.isfinite([config['target_descriptor_weight'], config['source_positive_weight']]).all()
            or min(config['target_descriptor_weight'], config['source_positive_weight']) <= 0):
        raise ValueError('Invalid fixed training budget or objective weights')
    return config, base


def run(config_path, smoke=False, verify=False, audit_only=False):
    config, base = load_settings(config_path, smoke)
    if not Path(config['folds_path']).is_file():
        raise ValueError('Frozen target folds required')
    wave, y, groups, bank = regularization.load_bank(base)
    if (bank['n_patients'], bank['n_records'], bank['n_beats']) != (164, 391, 55863):
        raise ValueError('Original target coverage changed')
    assigned = benchmark.load_or_create_folds(y, groups, base).set_index('patient_id').loc[groups, 'fold'].to_numpy()
    if set(assigned) != set(range(1, config['outer_splits'] + 1)):
        raise ValueError('Frozen target fold coverage changed')
    source_wave, pools, codes, source_audit = native_partial_source.audit(config, base, groups)
    root = Path(config['output_dir']) / ('audit' if audit_only else 'smoke' if smoke else 'complete')
    source_path = root / 'source_audit.json'
    if source_path.is_file() or verify:
        if read(source_path) != source_audit:
            raise ValueError('Audited source changed')
    else:
        root.mkdir(parents=True, exist_ok=True)
        _save_json(source_path, source_audit)
    if audit_only:
        print(json.dumps(source_audit, indent=2), flush=True)
        return
    module_paths = [Path(m.__file__) for m in (
        benchmark, labels, masked_pretraining, native_partial_source, neural, preprocessing_study,
        regularization, territory_study, vcg_expansion_bank)]
    module_paths.extend([Path(__file__), Path(__file__).with_name('native_partial_report.py'),
                         Path('scripts/run_native_partial_labels.py')])
    scope = dict(config=config, smoke=smoke, bank=bank, source_audit=source_audit,
                 folds_sha256=file_hash(config['folds_path']),
                 runtime=dict(numpy=np.__version__, torch=str(torch.__version__),
                              device=str(_device(config)), platform=platform.platform()),
                 code={str(path): file_hash(path) for path in module_paths})
    scope['fingerprint'] = _hash(scope)
    guard = root / 'study_manifest.json'
    if guard.exists() or verify:
        if read(guard) != scope:
            raise ValueError('The partial label study scope changed')
    else:
        _save_json(guard, scope)
    seeds = config['training_seeds'][:1] if smoke else config['training_seeds']
    folds = [1] if smoke else list(range(1, config['outer_splits'] + 1))
    predictions = {seed: {arm: np.full((len(y), len(config['classes'])), np.nan)
                          for arm in ARMS} for seed in seeds}
    audits, changed = {}, []
    data = dict(wave=wave, source_wave=source_wave, y=y, groups=groups, bank=bank)
    per_patient = config['source_beats_per_patient_per_update']
    for fold in folds:
        train = np.flatnonzero(assigned != fold)
        if smoke:
            ti, vi = next(benchmark.inner_patient_splits(y[train], groups[train], 3, config['seed'] + fold * 1000))
            train, query = train[ti], train[vi]
        else:
            query = np.flatnonzero(assigned == fold)
        if set(groups[train]) & set(groups[query]):
            raise ValueError('Target patients overlap')
        weights = benchmark.patient_balanced_weights(y[train], groups[train])
        norm = FoldNormalizer().fit(wave, np.empty((len(y), 0), np.float32), train, weights)
        for seed in seeds:
            sampling = make_plans(config, train, weights, pools, seed + fold)
            partners = shuffle_mentions(codes, seed + 101 * fold)
            shuffled, original = np.repeat(codes[partners], per_patient, axis=0), np.repeat(codes, per_patient, axis=0)
            if not np.array_equal(shuffled.sum(axis=0), original.sum(axis=0)):
                raise ValueError('True and shuffled source tag frequencies differ')
            changed.append(int(np.count_nonzero(np.any(codes != codes[partners], axis=1))))
            context = dict(fingerprint=scope['fingerprint'], fold=fold, seed=seed,
                           train_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query),
                           source_permutation_sha256=array_hash(partners),
                           train_patients=len(set(groups[train])), query_patients=len(set(groups[query])))
            for arm in ARMS:
                p, digest = run_arm(config, data, root, context, arm, train, query, norm,
                                    sampling, shuffled if arm == 'shuffled_source' else original, verify)
                if np.isfinite(predictions[seed][arm][query]).any():
                    raise ValueError('Duplicate held-out predictions')
                predictions[seed][arm][query] = p
                audits[f"f{fold:02d}_s{seed}_{arm}"] = digest
                _progress(root, state='replaying' if verify else 'fit_saved', fitted=len(audits),
                          total=len(folds) * len(seeds) * len(ARMS), fold=fold, seed=seed, arm=arm)
            print(f"{'Replayed' if verify else 'Fitted'} fold {fold}, seed {seed} ({len(audits)} models)", flush=True)
    if verify:
        evidence = root / 'verification_manifest.json'
        _save_json(evidence, dict(fingerprint=scope['fingerprint'], fit_audit_hashes=audits))
        if smoke:
            _save_json(root / 'summary.json', dict(
                complete=True, restored_models=len(audits), outer_test_patients_used=False,
                study_fingerprint=scope['fingerprint'], verification_sha256=file_hash(evidence)))
        else:
            summarize(root, scope, data, predictions, ARMS, len(audits), changed, file_hash(evidence))
        _progress(root, state='complete_independently_validated', fitted=len(audits), total=len(audits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/native_partial_labels_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    run(args.config, args.smoke, args.verify, args.audit)


if __name__ == '__main__':
    main()
