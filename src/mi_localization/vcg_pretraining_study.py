"""Fixed-budget native VCG pretraining and ECG-assisted training pilots."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
from scipy.fft import dct
import sklearn
import torch
from torch import nn
import yaml

from . import benchmark, masked_pretraining, neural, paired_bank, preprocessing_study
from . import regularization, transfer, vcg_expansion_bank
from .contrastive import _progress, initial_state_hash
from .multiview_bank import array_hash
from .neural import _device, _hash, _save_json, _seed, FoldNormalizer, VCGResNet
from .signal_audit import file_hash
from .supervised_transfer import restore_norm
from .vcg_pretraining_report import summarize

ARMS = {
    'native': ('scratch', 'target_only', 'target_plus_native'),
    'bridge': ('scratch', 'shuffled_ecg', 'paired_ecg'),
}


def read(path):
    return json.loads(path.read_text())


def row_groups(groups, rows):
    return [rows[groups[rows] == patient] for patient in np.unique(groups[rows])]


def sample_patient_beats(pools, generator, batch, unique=False):
    patients = generator.choice(len(pools), size=batch, replace=not unique)
    return np.array([generator.choice(pools[i]) for i in patients], dtype=np.int64)


def sampling_plan(config, mode, arm, train, groups, extra_groups, seed):
    """Pair arms on backbone exposure and give paired/shuffled ECG the same marginal."""
    rng = np.random.default_rng(seed + 18019)
    other = np.random.default_rng(seed + 18031)
    pools = row_groups(groups, train)
    count = config['batch_size']
    steps = config['pretrain_steps']
    if mode == 'native':
        extra_count = int(round(config['native_extra_fraction'] * count))
        first = np.stack([sample_patient_beats(pools, rng, count - extra_count) for _ in range(steps)])
        other_pools = row_groups(extra_groups, np.arange(len(extra_groups))) if arm == 'target_plus_native' else pools
        second = np.stack([sample_patient_beats(other_pools, other, extra_count) for _ in range(steps)])
        return first, second
    rows = np.stack([sample_patient_beats(pools, rng, count, unique=True) for _ in range(steps)])
    teacher = rows if arm == 'paired_ecg' else np.stack([
        np.roll(batch, int(other.integers(1, count))) for batch in rows
    ])
    if arm == 'shuffled_ecg' and np.any(groups[rows] == groups[teacher]):
        raise ValueError('ECG control must use a different training patient for every paired beat')
    return rows, teacher


def compact_ecg(ecg, train, groups, coefficients):
    """Compute signed low-frequency DCT descriptors on training rows only."""
    shape = (len(train), ecg.shape[1] * coefficients)
    features = np.empty(shape, dtype=np.float32)
    for offset in range(0, len(train), 256):
        rows = train[offset:offset + 256]
        values = dct(np.asarray(ecg[rows]), type=2, norm='ortho', axis=-1)[..., :coefficients]
        features[offset:offset + len(rows)] = values.reshape(len(rows), -1)
    _, inverse, counts = np.unique(groups[train], return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    mean = np.average(features.astype(np.float64), axis=0, weights=weights)
    variance = np.average((features - mean)**2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    return ((features - mean) / scale).astype(np.float32), array_hash(mean), array_hash(scale)


def fit_pretrain(config, mode, arm, inputs, train, initial_norm, seed, ecg_features, plan):
    _seed(seed, config)
    device = _device(config)
    first, second = plan
    waves, extra = inputs['waves'], inputs['extra']
    mask_rng = torch.Generator().manual_seed(seed + 18041)
    if mode == 'native':
        model = masked_pretraining.MaskedVCG(config).to(device)
    else:
        base = VCGResNet(len(config['classes']), tuple(config['widths']), config['pool_bins'], config['dropout'])
        model = nn.Sequential(base.encoder, nn.Linear(64, ecg_features.shape[1])).to(device)
    encoder = model.encoder if mode == 'native' else model[0]
    initial = initial_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['pretrain_learning_rate'],
                                  weight_decay=config['weight_decay'])
    mean, scale = initial_norm.wave_mean_[None, :, None], initial_norm.wave_scale_[None, :, None]
    losses = []
    for rows, paired in zip(first, second):
        if mode == 'native':
            source = extra if arm == 'target_plus_native' else waves
            block = np.concatenate([waves[rows], source[paired]])
        else:
            block = np.asarray(waves[rows])
        x = torch.from_numpy((block - mean) / scale).to(device)
        optimizer.zero_grad(set_to_none=True)
        if mode == 'native':
            mask = masked_pretraining.make_masks(
                len(x), x.shape[-1], config['patch_samples'], config['masked_patches'], mask_rng).to(device)
            loss = masked_pretraining.masked_mse(model(x.masked_fill(mask, 0)), x, mask)
        else:
            local = np.searchsorted(train, paired)
            if not np.array_equal(train[local], paired):
                raise ValueError('ECG supervision has rows outside the training partition')
            teacher = torch.from_numpy(ecg_features[local]).to(device)
            loss = nn.functional.mse_loss(model(x), teacher)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite pretraining loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    state = {name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()}
    audit = dict(initial_sha256=initial, encoder_sha256=initial_state_hash(encoder),
                 updates=len(losses), presentations=len(losses) * config['batch_size'],
                 first_loss=losses[0], last_loss=losses[-1])
    return state, audit


def expected_pretrainer_hash(config, mode, seed, teacher_dimensions):
    _seed(seed, config)
    if mode == 'native':
        return initial_state_hash(masked_pretraining.MaskedVCG(config))
    base = VCGResNet(len(config['classes']), tuple(config['widths']), config['pool_bins'], config['dropout'])
    return initial_state_hash(nn.Sequential(base.encoder, nn.Linear(64, teacher_dimensions)))


def load_inputs(config, mode):
    if not Path(config['folds_path']).is_file():
        raise ValueError('Require frozen target folds')
    waves, labels, groups, bank = regularization.load_bank(config)
    folds = benchmark.load_or_create_folds(labels, groups, config).set_index('patient_id')
    beat_folds = folds.loc[groups, 'fold'].to_numpy()
    if ((bank['n_patients'], bank['n_records'], bank['n_beats']) != (164, 391, 55863)
            or set(beat_folds) != set(range(1, config['outer_splits'] + 1))):
        raise ValueError('Target cohort or fold coverage changed')
    source, ecg, extra, extra_groups = {}, None, None, None
    if mode == 'native':
        audit = vcg_expansion_bank.audit(config)
        extra, extra_groups, manifest = vcg_expansion_bank.load(config)
        if set(groups) & set(extra_groups) or audit['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Invalid native source patient membership')
        source = dict(audit=audit, manifest_sha256=file_hash(Path(config['source_bank_dir']) / 'manifest.json'))
    else:
        if not (Path(config['ecg_bank_dir']) / 'manifest.json').is_file():
            raise FileNotFoundError('The independently aligned ECG cache is required')
        vcg, ecg, ey, eg, source = paired_bank.load_paired(config)
        if vcg.shape != waves.shape or not np.array_equal(ey, labels) or not np.array_equal(eg, groups):
            raise ValueError('The ECG cache does not align with original target rows')
    return dict(waves=waves, labels=labels, groups=groups, bank=bank,
                folds=beat_folds, extra=extra, extra_groups=extra_groups, ecg=ecg, source=source)


def check_finite_state(state):
    if not state or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint state is missing or nonfinite')


def fit_member(config, mode, arm, inputs, context, train, query, norm, teacher, root, verify):
    seed = context['seed']
    waves, y, groups = inputs['waves'], inputs['labels'], inputs['groups']
    path = root / f"fold{context['fold']:02d}_seed{context['declared_seed']}_{arm}"
    path.mkdir(parents=True, exist_ok=True)
    audit_path = path / 'audit.json'
    logical = 'scratch' if arm == 'scratch' else 'pretrained'
    plan = None if arm == 'scratch' else sampling_plan(
        config, mode, arm, train, groups, inputs['extra_groups'], seed)
    scope = dict(context, arm=arm, source_rows_sha256=None if plan is None else [array_hash(a) for a in plan])
    existing = audit_path.exists()
    if verify and not existing:
        raise ValueError('Required fit is missing')
    state = None
    if existing:
        receipt = read(audit_path)
        if (receipt['context'] != scope or any(file_hash(path / n) != digest
                                               for n, digest in receipt['files'].items())):
            raise ValueError('Saved model or training membership changed')
        audit = receipt['fit']
        pretrain = receipt['pretrain']
        checkpoint = torch.load(path / 'model.pt', map_location='cpu', weights_only=True)
        if checkpoint['context'] != scope:
            raise ValueError('Saved model context changed')
        state = checkpoint['pretrained_encoder']
        if state is not None:
            check_finite_state(state)
            expected_encoder = VCGResNet(len(config['classes']), tuple(config['widths']), config['pool_bins']).encoder
            expected_encoder.load_state_dict(state, strict=True)
            if initial_state_hash(expected_encoder) != pretrain['encoder_sha256']:
                raise ValueError('Pretrained encoder hash changed')
        model = transfer.initial_model(config, seed, logical, state)
        check_finite_state(checkpoint['model'])
        model.load_state_dict(checkpoint['model'], strict=True)
        model = model.to(_device(config))
        actual_norm = restore_norm(audit['normalization'], len(train))
    else:
        pretrain = None
        if plan is not None:
            state, pretrain = fit_pretrain(config, mode, arm, inputs, train, norm, seed, teacher, plan)
        model, actual_norm, audit = transfer.fit_transfer(
            waves, y, groups, train, None, config['target_epochs'], config, seed, logical, state,
            lambda row: _progress(root.parent, state='training', arm=arm,
                                  fold=context['fold'], seed=context['declared_seed'], **row))
    if arm == 'scratch':
        if pretrain is not None or state is not None:
            raise ValueError('Scratch received a source checkpoint')
    else:
        dims = teacher.shape[1] if teacher is not None else 0
        if (pretrain['initial_sha256'] != expected_pretrainer_hash(config, mode, seed, dims)
                or pretrain['updates'] != config['pretrain_steps']
                or pretrain['presentations'] != config['pretrain_steps'] * config['batch_size']
                or not np.isfinite([pretrain['first_loss'], pretrain['last_loss']]).all()):
            raise ValueError('Pretraining initialization, loss or budget differs')
    encoder_hash = None if pretrain is None else pretrain['encoder_sha256']
    transfer.validate_fit(audit, config, seed, logical, config['learning_rate'], len(train),
                          len(np.unique(groups[train])), 0, encoder_hash, config['target_epochs'])
    if norm.audit() != actual_norm.audit():
        raise ValueError('Target normalization differs from training-only reconstruction')
    probabilities = preprocessing_study.predict_waves(model, actual_norm, waves, y, query, config)
    if existing:
        with np.load(path / 'predictions.npz', allow_pickle=False) as saved:
            if (not np.array_equal(saved['rows'], query)
                    or not np.allclose(saved['probabilities'], probabilities, rtol=2e-5, atol=2e-5)):
                raise ValueError('Restored target predictions disagree')
    else:
        masked_pretraining._save_checkpoint(path / 'model.pt', {
            'context': scope, 'model': model.state_dict(), 'pretrained_encoder': state})
        temporary = path / 'predictions.partial.npz'
        np.savez_compressed(temporary, rows=query, probabilities=probabilities)
        temporary.replace(path / 'predictions.npz')
        _save_json(audit_path, dict(context=scope, fit=audit, pretrain=pretrain,
                                   files={n: file_hash(path / n) for n in ('model.pt', 'predictions.npz')}))
    return probabilities, audit, pretrain, plan, file_hash(audit_path)


def settings(config, smoke):
    configured = dict(config, max_epochs=config['target_epochs'])
    if smoke:
        configured.update(target_epochs=1, max_epochs=1, pretrain_steps=2,
                          samples_per_epoch=32, batch_size=16)
    return configured


def run(config, mode, smoke=False, verify=False):
    inputs = load_inputs(config, mode)
    config = settings(config, smoke)
    root = Path(config[f'{mode}_output_dir']) / ('smoke' if smoke else 'complete')
    directory = root / 'review'
    directory.mkdir(parents=True, exist_ok=True)
    modules = (neural, regularization, transfer, benchmark, preprocessing_study,
               vcg_expansion_bank, masked_pretraining, paired_bank)
    paths = [Path(m.__file__) for m in modules] + [Path(__file__),
                                                  Path(__file__).with_name('vcg_pretraining_report.py')]
    scope = dict(config=config, mode=mode, smoke=smoke, bank=inputs['bank'], source=inputs['source'],
                 folds_sha256=file_hash(config['folds_path']),
                 code={str(p): file_hash(p) for p in paths},
                 runtime=dict(torch=str(torch.__version__), numpy=np.__version__, sklearn=sklearn.__version__,
                              device=str(_device(config)), platform=platform.platform()))
    scope['fingerprint'] = _hash(scope)
    guard = root / 'study_manifest.json'
    if guard.exists():
        if read(guard) != scope:
            raise ValueError('Pilot scope changed. Use a separate namespace.')
    else:
        if verify:
            raise ValueError('A fit scope is required for verification')
        _save_json(guard, scope)
    y, groups = inputs['labels'], inputs['groups']
    folds = [1] if smoke else list(range(1, config['outer_splits'] + 1))
    seeds = config['training_seeds'][:1] if smoke else config['training_seeds']
    if not smoke and (len(seeds) != 3 or len(set(seeds)) != 3):
        raise ValueError('A complete pilot requires all three distinct seeds')
    predictions = {seed: {arm: np.full((len(y), len(config['classes'])), np.nan)
                          for arm in ARMS[mode]} for seed in seeds}
    audits = {}
    count = 0
    for fold in folds:
        train = np.flatnonzero(inputs['folds'] != fold)
        if smoke:
            ti, vi = next(benchmark.inner_patient_splits(y[train], groups[train], 3, config['seed'] + fold * 1000))
            train, query = np.sort(train[ti]), np.sort(train[vi])
        else:
            query = np.flatnonzero(inputs['folds'] == fold)
        if set(groups[train]) & set(groups[query]):
            raise ValueError('Fitting and query patients overlap')
        weights = benchmark.patient_balanced_weights(y[train], groups[train])
        norm = FoldNormalizer().fit(inputs['waves'], np.empty((len(y), 0), dtype=np.float32), train, weights)
        teacher, mean_hash, scale_hash = None, None, None
        if mode == 'bridge':
            teacher, mean_hash, scale_hash = compact_ecg(
                inputs['ecg'], train, groups, config['bridge_dct_coefficients'])
        for seed in seeds:
            fits, plans, pretrains = [], [], []
            context = dict(fingerprint=scope['fingerprint'], fold=fold, declared_seed=seed, seed=seed + fold,
                           train_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query),
                           teacher_mean_sha256=mean_hash, teacher_scale_sha256=scale_hash,
                           train_patient_count=len(np.unique(groups[train])),
                           query_patient_count=len(np.unique(groups[query])))
            for arm in ARMS[mode]:
                p, fit, pret, plan, digest = fit_member(
                    config, mode, arm, inputs, context, train, query, norm, teacher, directory, verify)
                predictions[seed][arm][query] = p
                fits.append(fit)
                if plan is not None:
                    plans.append(plan)
                    pretrains.append(pret)
                audits[f'fold{fold:02d}_seed{seed}_{arm}'] = digest
                count += 1
                _progress(root, state='restoring' if verify else 'fit_saved',
                          completed_fits=count, total_fits=len(folds) * len(seeds) * len(ARMS[mode]),
                          arm=arm, fold=fold, seed=seed)
            for fit in fits[1:]:
                transfer.check_pair(fits[0], fit)
            if (len(set(a['initial_sha256'] for a in pretrains)) != 1
                    or not np.array_equal(plans[0][0], plans[1][0])
                    or (mode == 'bridge' and not np.array_equal(np.sort(plans[0][1], axis=1),
                                                                np.sort(plans[1][1], axis=1)))):
                raise ValueError('Matched pretraining controls differ in initialization or shared samples')
            print(f"{mode}: {'replayed' if verify else 'fitted'} fold {fold}, seed {seed} ({count} models)", flush=True)
    if verify:
        manifest = root / 'verification_manifest.json'
        _save_json(manifest, dict(fingerprint=scope['fingerprint'], fit_audit_hashes=audits))
        if smoke:
            _save_json(root / 'summary.json', dict(
                complete=True, smoke=True, restored_models=count, outer_test_patients_used=False,
                study_fingerprint=scope['fingerprint'], verification_manifest_sha256=file_hash(manifest)))
        else:
            summarize(root, scope, inputs, predictions, ARMS[mode], count, file_hash(manifest))
        _progress(root, state='complete_independently_validated', completed_fits=count, total_fits=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=tuple(ARMS))
    parser.add_argument('--config', default='configs/vcg_pretraining_pilots_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    run(config, args.mode, args.smoke, args.verify)


if __name__ == '__main__':
    main()
