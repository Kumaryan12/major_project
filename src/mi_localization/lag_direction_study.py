"""Matched ridge readouts of signed within-beat direction with frozen baseline features."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import benchmark, chronology_bank, distribution_kernel_model as kernel
from . import distribution_kernel_source as source, distribution_score_correction as saved
from . import distribution_study as core, lag_direction_features as features, regularization
from .lag_direction_report import summarize
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

ARMS = ('baseline', 'magnitude', 'sign_scrambled', 'signed')


def validate(config):
    lags = config['lags_ms']
    if (len(lags) != 4 or any(type(x) is not int or x <= 0 for x in lags) or sorted(set(lags)) != lags
            or len(config['windows_ms']) != 4
            or any(len(w) != 2 or any(type(x) is not int for x in w) or w[1] - w[0] <= max(lags)
                   for w in config['windows_ms'])):
        raise ValueError('Require four fixed windows and four valid positive lags')
    if not all(np.isfinite(config[k]) and config[k] > 0
               for k in ('auxiliary_weight', 'scale_relative_floor', 'scale_absolute_floor')):
        raise ValueError('Require positive common scale settings')


def record_context(bank, scope, output, patient, seed, fold, inner, train, query, verify, choices=None):
    config = scope['config']
    values, source_path = source.context(bank, seed, fold, inner, train, query)
    base = values['baseline']
    aux = dict(magnitude=patient['magnitude'], sign_scrambled=patient[str(seed)], signed=patient['signed'])
    blocks, scale, gain = features.scaled_block(patient['magnitude'], aux, train, base.shape[1], config)
    n_classes = len(config['classes'])
    expected = core.membership(bank, scope, train, query)
    arrays = {'query_rows': query, 'column_scale': scale}
    scores, references, results, model_states = {}, {}, {}, {}
    for arm in ARMS:
        candidates = range(len(config['ridge_alphas'])) if inner else [choices[arm]]
        matrix = base if arm == 'baseline' else np.column_stack((base, blocks[arm][np.r_[train, query]]))
        data = None
        if arm != 'baseline':
            data = kernel.KernelData(matrix[:len(train)], bank.labels[train], matrix[len(train):], n_classes)
            model_states[arm] = data.kernel(0)[2]
        trials = []
        for candidate in candidates:
            if arm == 'baseline':
                p, audit, path = source.reference(bank, seed, fold, inner, 'baseline', candidate,
                                                  train, query, base, source_path)
                references[str(candidate)] = file_hash(path)
                if inner and saved.score(bank.labels[query], p, n_classes) != audit['validation_macro_f1']:
                    raise ValueError('Frozen inner baseline score changed')
            else:
                model = kernel.KernelModel.fit(data, config['ridge_alphas'][candidate], 0)
                p = model.predict(data)
                arrays[f'{arm}_{candidate}_dual'] = model.dual
                arrays[f'{arm}_{candidate}_intercept'] = model.intercept
            arrays[f'{arm}_{candidate}_p'] = p
            trials.append(p)
        if inner:
            scores[arm] = [saved.score(bank.labels[query], p, n_classes) for p in trials]
        else:
            results[arm] = f'{arm}_{choices[arm]}_p'
    audit = dict(expected, fold=fold, seed=seed, inner=inner, scales_sha256=array_hash(scale),
                 auxiliary_gain=float(gain), source_context_sha256=file_hash(source_path),
                 baseline_reference_audits=references, kernel_states=model_states,
                 inner_scores=scores, outer_choices=choices)
    path = output / f'fold{fold:02d}_s{seed}_inner{inner}.json'
    restored = saved.save_record(path, audit, arrays, verify)
    return scores if inner else {arm: restored[key] for arm, key in results.items()}


def evaluate(bank, scope, output, patient, seed, fold, verify, smoke):
    config = scope['config']
    train = np.flatnonzero(bank.assigned != fold)
    splits = benchmark.inner_patient_splits(bank.labels[train], bank.ids[train], config['inner_splits'],
                                            config['split_seed'] + fold * 1000)
    results = [
        record_context(bank, scope, output, patient, seed, fold, inner, train[ti], train[vi], verify)
        for inner, (ti, vi) in enumerate(splits, 1)
    ]
    scores = {arm: np.asarray([item[arm] for item in results]).T for arm in ARMS}
    choices = {arm: core.select(values)[0] for arm, values in scores.items()}
    historical = saved.read(bank.root / 'review' / f'selection_{fold:02d}_baseline_s{seed}.json')
    if historical['selected_candidate'] != choices['baseline'] or historical['inner_scores'] != scores['baseline'].tolist():
        raise ValueError('Original baseline alpha selection changed')
    if smoke:
        return None, None, choices
    query = np.flatnonzero(bank.assigned == fold)
    probabilities = record_context(bank, scope, output, patient, seed, fold, 0, train, query, verify, choices)
    return query, probabilities, choices


def run(config, smoke=False, verify=False):
    validate(config)
    bank, _ = source.validate_chain(config)
    records = features.load_records(config, bank)
    ids = bank.ids[bank.assigned != 1] if smoke else None
    patient, encoded = features.reduce_records(config, records, allowed=ids)
    print(f"{'Re-encoded' if verify else 'Encoded'} {encoded['beats']:,} beats across {encoded['records']} records", flush=True)
    paths = [Path(module.__file__) for module in (benchmark, source, kernel, saved, core, features,
                                                 chronology_bank, regularization)]
    paths += [Path(__file__), Path(__file__).with_name('lag_direction_report.py'),
              Path('scripts/run_lag_direction_tensor.py')]
    scope = dict(config=config, source=bank.provenance, encoded=encoded, smoke=smoke,
                 runtime=bank.source_scope['runtime'], code={str(path): file_hash(path) for path in paths})
    scope['study_fingerprint'] = _hash(scope)
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    output = root / 'review'
    output.mkdir(parents=True, exist_ok=True)
    manifest = root / 'study_manifest.json'
    if verify or manifest.exists():
        if saved.read(manifest) != scope:
            raise ValueError('Existing lag tensor scope differs')
    else:
        _save_json(manifest, scope)
    if smoke:
        evaluate(bank, scope, output, patient, config['projection_seeds'][0], 1, verify, True)
        if verify:
            _save_json(root / 'summary.json', dict(
                complete=True, study_fingerprint=scope['study_fingerprint'], outer_test_patients_used=False,
                inner_contexts=3, new_models_restored=18, baseline_models_restored=6,
                encoded_beats=encoded['beats']))
        print(f"Inner-only smoke {'replayed' if verify else 'fitted'}", flush=True)
        return
    members = {arm: {s: np.full((len(bank.ids), len(config['classes'])), np.nan)
                     for s in config['projection_seeds']} for arm in ARMS}
    counts = {arm: [0] * len(config['ridge_alphas']) for arm in ARMS}
    for fold in range(1, config['outer_splits'] + 1):
        for seed in config['projection_seeds']:
            rows, values, choices = evaluate(bank, scope, output, patient, seed, fold, verify, False)
            for arm in ARMS:
                if np.isfinite(members[arm][seed][rows]).any():
                    raise ValueError('Duplicate held-out patient predictions')
                members[arm][seed][rows] = values[arm]
                counts[arm][choices[arm]] += 1
        print(f"{'Replayed' if verify else 'Fitted'} fold {fold}/{config['outer_splits']}", flush=True)
    if verify:
        summarize(root, scope, bank, members, counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/lag_direction_tensor_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config['torch_threads'])
    with threadpool_limits(limits=1):
        run(config, args.smoke, args.verify)


if __name__ == '__main__':
    main()
