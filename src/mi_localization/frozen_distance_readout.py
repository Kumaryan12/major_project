"""Matched frozen64 ridge, centroid and shrinkage-distance patient readouts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
from threadpoolctl import threadpool_limits
import yaml

from . import benchmark, frozen_patient_readout as frozen, frozen_distance_model as model
from . import earliest_training_duration as duration, later_recording_distillation as source
from . import neural, rocket_study as rocket, model_error_comparison as replay
from .contrastive import _progress
from .earliest_record_convolution import write_or_verify
from .multiview_bank import array_hash
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash

CONFIG = Path('configs/frozen_distance_readout_v1.yaml')
PROTOCOL = Path('docs/frozen_distance_readout_plan.md')
ARMS = ('ridge', 'centroid', 'mahalanobis')
CONTRASTS = (('mahalanobis', 'ridge'), ('mahalanobis', 'centroid'), ('centroid', 'ridge'))


def custody(root):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name not in ('progress.json', 'runner.lock')}


def load_inputs(config):
    root = Path(config['source_root'])
    prior = replay.checked_json(root / 'summary.json', config['source_summary_sha256'])
    scope = replay.read(root / 'study_manifest.json')
    if (not prior['complete'] or prior['smoke'] or prior['ridge_fits'] != 420
            or prior['encoder_contexts'] != 120 or prior['encoder_updates'] != 0
            or prior['study_fingerprint'] != scope['study_fingerprint']
            or neural._hash({k: v for k, v in scope.items() if k != 'study_fingerprint'}) != scope['study_fingerprint']
            or scope['code_sha256'] != file_hash(Path(frozen.__file__))
            or scope['protocol_sha256'] != file_hash(frozen.PROTOCOL)):
        raise ValueError('Expected the pinned completed frozen readout source')
    for module, expected in scope['dependencies'].items():
        if file_hash(Path('src').joinpath(*module.split('.')).with_suffix('.py')) != expected:
            raise ValueError('Pinned source dependency changed')
    runtime = dict(numpy=np.__version__, torch=str(torch.__version__), sklearn=sklearn.__version__,
                   platform=platform.platform(), device='cpu')
    if scope['runtime'] != runtime:
        raise ValueError('Source frozen-feature runtime changed')
    source_config = yaml.safe_load(Path(config['source_config']).read_text())
    data, bank, _, settings, encoder_scope, encoder_hashes = frozen.load_inputs(source_config)
    if scope['config'] != dict(source_config, classes=settings['classes']):
        raise ValueError('Source configuration differs from its completed scope')
    hashes = custody(root)
    return data, bank, prior, settings, encoder_scope, encoder_hashes, source_config, scope, hashes


def model_state(fitted):
    return dict(scaler_mean=array_hash(fitted['scaler'].mean_), scaler_scale=array_hash(fitted['scaler'].scale_),
                centroids=array_hash(fitted['centroids']), covariance=array_hash(fitted['covariance']),
                precision=array_hash(fitted['precision']),
                **{k: fitted[k] for k in ('tau', 'shrinkage', 'classes', 'trace_floor',
                                          'eigenvalue_floor', 'score_divisor')})


def fit_model(x, y, config, shrinkage):
    return model.fit_distance(x, y, len(config['classes']), shrinkage,
                              config['trace_floor'], config['eigenvalue_floor'], config['score_divisor'])


def distance_context(bank, config, scope, review, seed, fold, inner, arm, candidate,
                     train, query, x, context_path, verify, receipts):
    shrinkage = 1. if arm == 'centroid' else config['shrinkages'][candidate]
    expected = dict(rocket.expected_membership(bank, scope, train, query), seed=seed, fold=fold,
                    inner=inner, arm=arm, candidate=candidate, shrinkage=shrinkage,
                    context_sha256=file_hash(context_path), training_features_sha256=array_hash(x[:len(train)]),
                    query_features_sha256=array_hash(x[len(train):]))
    path = review / f'head_f{fold:02d}_s{seed}_i{inner}_{arm}_c{candidate}.json'
    mpath, ppath = path.with_suffix('.joblib'), path.with_suffix('.npz')
    if verify or path.exists():
        audit = replay.read(path)
        replay.require_fields(audit, expected)
        if file_hash(mpath) != audit['model_sha256'] or file_hash(ppath) != audit['probabilities_sha256']:
            raise ValueError('Distance model or prediction file changed')
        fitted = joblib.load(mpath)
        with np.load(ppath) as saved:
            if not np.array_equal(saved['rows'], query):
                raise ValueError('Distance query rows changed')
            p = saved['probabilities']
        if model_state(fitted) != audit['fitted_state']:
            raise ValueError('Restored distance parameters changed')
        if verify:
            fresh = fit_model(x[:len(train)], bank.labels[train], config, shrinkage)
            if model_state(fresh) != audit['fitted_state']:
                raise ValueError('Training-only distance statistics failed reconstruction')
            if not np.allclose(model.predict_distance(fitted, x[len(train):]), p, atol=1e-12, rtol=1e-12):
                raise ValueError('Restored distance predictions disagree')
    else:
        fitted = fit_model(x[:len(train)], bank.labels[train], config, shrinkage)
        p = model.predict_distance(fitted, x[len(train):])
        rocket.dump_model(mpath, fitted)
        temporary = ppath.with_suffix('.partial.npz')
        np.savez_compressed(temporary, rows=query, probabilities=p)
        temporary.replace(ppath)
        audit = dict(expected, fitted_state=model_state(fitted), model_sha256=file_hash(mpath),
                     probabilities_sha256=file_hash(ppath))
        neural._save_json(path, audit)
    if (p.shape != (len(query), len(config['classes'])) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(1), 1)):
        raise ValueError('Invalid distance probabilities')
    key = f'{fold}_{seed}_{inner}_{arm}_{candidate}'
    receipts[key] = neural._hash(audit)
    _progress(review.parent, state='verifying' if verify else 'fitting', fold=fold, seed=seed,
              inner=inner, arm=arm, candidate=candidate, distance_heads=len(receipts))
    return p


def restore_ridge(bank, source_config, source_scope, source_root, source_hashes, seed, fold, inner,
                  candidate, train, query, x, context_path, receipts):
    suffix = f'_i{inner}_a{candidate}' if inner else ''
    path = source_root / 'review' / f"{'inner' if inner else 'outer'}_{fold:02d}_ridge64_s{seed}{suffix}.json"
    audit = replay.checked_json(path, source_hashes[str(path.relative_to(source_root))])
    expected = dict(rocket.expected_membership(bank, source_scope, train, query), transform_seed=seed,
                    outer_fold=fold, inner_fold=inner, arm='ridge64', candidate_id=candidate,
                    alpha=source_config['ridge_alphas'][candidate], context_sha256=file_hash(context_path),
                    training_features_sha256=array_hash(x[:len(train)]),
                    query_features_sha256=array_hash(x[len(train):]), n_features=64)
    replay.require_fields(audit, expected)
    if (file_hash(path.with_suffix('.joblib')) != audit['model_sha256']
            or file_hash(path.with_suffix('.npz')) != audit['probabilities_sha256']):
        raise ValueError('Protected ridge artifact changed')
    fitted = joblib.load(path.with_suffix('.joblib'))
    with np.load(path.with_suffix('.npz')) as saved:
        if not np.array_equal(saved['rows'], query):
            raise ValueError('Source ridge query rows changed')
        p = saved['probabilities']
    scaler = StandardScaler().fit(x[:len(train)])
    if (not np.array_equal(scaler.mean_, fitted[0].mean_) or not np.array_equal(scaler.scale_, fitted[0].scale_)
            or fitted[-1].alpha != expected['alpha'] or fitted[-1].class_weight != 'balanced'
            or fitted[-1].solver != 'cholesky'
            or not np.allclose(rocket.predict(fitted, x[len(train):], len(source_config['classes'])), p,
                               atol=1e-12, rtol=1e-10)):
        raise ValueError('Restored ridge settings or probabilities changed')
    if inner and audit['validation_macro_f1'] != float(f1_score(
            bank.labels[query], p.argmax(1), labels=np.arange(6), average='macro', zero_division=0)):
        raise ValueError('Source ridge inner score changed')
    receipts[f'{fold}_{seed}_{inner}_{candidate}'] = neural._hash(audit)
    return p


def features(data, bank, config, settings, encoder_scope, encoder_hashes, source_config,
             source_scope, own_scope, review, seed, fold, inner, train, query, verify, contexts):
    _progress(review.parent, state='verifying_features' if verify else 'encoding_features', fold=fold,
              seed=seed, inner=inner, feature_contexts=len(contexts))
    values, _, _, source_path = frozen.feature_context(
        data, bank, source_config, settings, encoder_scope, encoder_hashes, source_scope,
        Path(config['source_root']) / 'review', seed, fold, inner, train, query, True)
    x = values['ridge64']
    if x.shape != (len(train) + len(query), 64):
        raise ValueError('Expected exact 64-feature source matrix')
    path = review / f'features_f{fold:02d}_s{seed}_i{inner}.json'
    receipt = dict(rocket.expected_membership(bank, own_scope, train, query), seed=seed, fold=fold, inner=inner,
                   source_context_sha256=file_hash(source_path), feature_sha256=array_hash(x))
    npz = path.with_suffix('.npz')
    if verify or path.exists():
        old = replay.read(path)
        if file_hash(npz) != old['cache_sha256']:
            raise ValueError('Private feature cache changed')
        with np.load(npz) as saved:
            if not np.array_equal(saved['rows'], np.r_[train, query]) or not np.array_equal(saved['features'], x):
                raise ValueError('Independent source feature replay differs')
        receipt['cache_sha256'] = old['cache_sha256']
    else:
        temporary = npz.with_suffix('.partial.npz')
        np.savez_compressed(temporary, rows=np.r_[train, query], features=x)
        temporary.replace(npz)
        receipt['cache_sha256'] = file_hash(npz)
    write_or_verify(path, receipt, verify)
    contexts[f'{fold}_{seed}_{inner}'] = neural._hash(receipt)
    return x, source_path, path


def evaluate(data, bank, config, settings, encoder_scope, encoder_hashes, source_config, source_scope,
             source_hashes, scope, review, seed, fold, train, query, verify, heads, ridges, contexts, checks,
             smoke=False):
    ridge_oof = [np.full((len(bank.ids), 6), np.nan) for _ in source_config['ridge_alphas']]
    distance_oof = [np.full((len(bank.ids), 6), np.nan) for _ in config['shrinkages']]
    splits = benchmark.inner_patient_splits(bank.labels[train], bank.ids[train], config['inner_splits'],
                                           settings['seed'] + fold)
    for inner, (ti, vi) in enumerate(splits, 1):
        fitting, validation = train[ti], train[vi]
        x, old_path, path = features(data, bank, config, settings, encoder_scope, encoder_hashes,
                                     source_config, source_scope, scope, review, seed, fold, inner,
                                     fitting, validation, verify, contexts)
        for candidate in range(len(source_config['ridge_alphas'])):
            ridge_oof[candidate][validation] = restore_ridge(
                bank, source_config, source_scope, Path(config['source_root']), source_hashes,
                seed, fold, inner, candidate, fitting, validation, x, old_path, ridges)
        centroid = distance_context(bank, config, scope, review, seed, fold, inner, 'centroid', 0,
                                     fitting, validation, x, path, verify, heads)
        for candidate, shrinkage in enumerate(config['shrinkages']):
            p = distance_context(bank, config, scope, review, seed, fold, inner, 'mahalanobis', candidate,
                                 fitting, validation, x, path, verify, heads)
            distance_oof[candidate][validation] = p
            if shrinkage == 1.:
                if not np.allclose(p, centroid, atol=1e-12, rtol=1e-12):
                    raise ValueError('Full-shrinkage identity control failed')
                checks.append(f'{fold}_{seed}_{inner}')
        if smoke:
            return None
    dated = train[data['eligible'][train]]
    ridge_choice, ridge_scores = duration.select_epochs(
        bank.labels[dated], {i: p[dated] for i, p in enumerate(ridge_oof)}, 6)
    choice, scores = duration.select_epochs(
        bank.labels[dated], {i: p[dated] for i, p in enumerate(distance_oof)}, 6)
    source_selection = Path(config['source_root']) / 'review' / f'selection_{fold:02d}_s{seed}_ridge64.json'
    decision = replay.checked_json(source_selection, source_hashes[str(source_selection.relative_to(config['source_root']))])
    if decision['alpha_index'] != ridge_choice or decision['inner_macro_f1'] != ridge_scores:
        raise ValueError('Source ridge alpha selection did not reproduce')
    write_or_verify(review / f'selection_f{fold:02d}_s{seed}.json',
                    dict(study_fingerprint=scope['study_fingerprint'], seed=seed, fold=fold,
                         ridge_candidate=ridge_choice, ridge_scores=ridge_scores, distance_candidate=choice,
                         distance_scores=scores, shrinkage=config['shrinkages'][choice]), verify)
    x, old_path, path = features(data, bank, config, settings, encoder_scope, encoder_hashes,
                                 source_config, source_scope, scope, review, seed, fold, 0, train, query,
                                 verify, contexts)
    ridge = restore_ridge(bank, source_config, source_scope, Path(config['source_root']), source_hashes,
                          seed, fold, 0, ridge_choice, train, query, x, old_path, ridges)
    centroid = distance_context(bank, config, scope, review, seed, fold, 0, 'centroid', 0,
                                 train, query, x, path, verify, heads)
    p = distance_context(bank, config, scope, review, seed, fold, 0, 'mahalanobis', choice,
                         train, query, x, path, verify, heads)
    if config['shrinkages'][choice] == 1.:
        if not np.allclose(p, centroid, atol=1e-12, rtol=1e-12):
            raise ValueError('Selected outer identity control failed')
        checks.append(f'{fold}_{seed}_0')
    return dict(ridge=ridge, centroid=centroid, mahalanobis=p)


def summarize(data, config, predictions, prior):
    results = {}
    for name, mask in dict(dated=data['eligible'], all_patients=np.ones(len(data['ids']), dtype=bool)).items():
        truth = data['labels'][mask]
        weights = patient_weights(truth, np.arange(6), config['bootstrap_draws'], 42)
        scores, draws = {}, {}
        for arm, runs in predictions.items():
            members = {str(seed): p[mask] for seed, p in runs.items()}
            if any(not np.isfinite(p).all() for p in members.values()):
                raise ValueError('Missing patient predictions')
            p = np.mean(list(members.values()), axis=0)
            scores[arm] = dict(ensemble=benchmark.metric_bundle(truth, p, config['classes']),
                               members={s: benchmark.metric_bundle(truth, value, config['classes']) for s, value in members.items()})
            draws[arm] = weighted_f1(truth, p.argmax(1), np.arange(6), weights)
        if scores['ridge'] != prior['cohorts'][name]['scores']['ridge64']:
            raise ValueError('Source ridge baseline metrics differ')
        contrasts = {f'{a}_minus_{b}': dict(delta=scores[a]['ensemble']['macro_f1'] - scores[b]['ensemble']['macro_f1'],
                                            interval=np.quantile(draws[a] - draws[b], [.025, .975]).tolist())
                     for a, b in CONTRASTS}
        results[name] = dict(patients=int(mask.sum()), scores=scores, contrasts=contrasts)
    return results


def render(result):
    lines = ['# Frozen64 ridge and shared-covariance distance readouts', '',
             f"Verified {result['feature_contexts']} feature contexts, {result['distance_heads']} new distance heads "
             f"and {result['source_ridge_heads']} source ridge heads; {result['identity_controls']} identity checks.",
             'Neural encoder updates: zero. All source artifacts retained unchanged.', '']
    if result['smoke']:
        return '\n'.join(lines + ['Inner-training/validation smoke only; no outer endpoint score.', ''])
    lines += ['| Cohort | Arm | Ensemble macro F1 | Member F1 in seed order |', '|---|---|---:|---|']
    for name, cohort in result['cohorts'].items():
        for arm, score in cohort['scores'].items():
            members = ', '.join(f"{v['macro_f1']:.6f}" for v in score['members'].values())
            lines.append(f"| {name} ({cohort['patients']}) | {arm} | {score['ensemble']['macro_f1']:.6f} | {members} |")
    lines += ['', '| Cohort | Contrast | Difference | Descriptive paired 95% interval |', '|---|---|---:|---|']
    for name, cohort in result['cohorts'].items():
        for contrast, row in cohort['contrasts'].items():
            low, high = row['interval']
            lines.append(f"| {name} | {contrast} | {row['delta']:+.6f} | [{low:+.6f},{high:+.6f}] |")
    lines += ['', '| Primary cohort class | Patients | Ridge | Centroid | Mahalanobis |', '|---|---:|---:|---:|---:|']
    for label in result['classes']:
        metrics = [result['cohorts']['dated']['scores'][arm]['ensemble']['classification_report'][label] for arm in ARMS]
        lines.append(f"| {label} | {int(metrics[0]['support'])} | " + ' | '.join(f"{m['f1-score']:.4f}" for m in metrics) + ' |')
    lines += ['', 'Primary: Mahalanobis minus ridge. Covariance control: Mahalanobis minus centroid. '
              'All three seeds are equally weighted. Mahalanobis has three inner shrinkage candidates; centroid has one. '
              'Intervals condition on fixed predictions from repeatedly studied development patients. '
              'The 145-patient endpoint uses earliest dated recordings; the 164 sensitivity retains one selected record each. '
              'Historical all-record .60 results are separate.', '', '[Protocol](../../../docs/frozen_distance_readout_plan.md)', '']
    return '\n'.join(lines)


def run(config_path, smoke=False, verify=False):
    config = yaml.safe_load(config_path.read_text())
    if (config['shrinkages'] != [1., .9, .5] or config['score_divisor'] != 128.
            or config['device'] != 'cpu' or config['inner_splits'] != 3):
        raise ValueError('Expected the declared bounded distance scope')
    data, bank, prior, settings, encoder_scope, encoder_hashes, source_config, source_scope, source_hashes = load_inputs(config)
    config = dict(config, classes=settings['classes'])
    source_config = dict(source_config, classes=config['classes'])
    neural._seed(42, config)
    scope = dict(config=config, smoke=smoke, source_fingerprint=source_scope['study_fingerprint'],
                 source_custody_sha256=neural._hash(source_hashes), code_sha256=file_hash(Path(__file__)),
                 model_sha256=file_hash(Path(model.__file__)), protocol_sha256=file_hash(PROTOCOL),
                 folds_sha256=file_hash(settings['folds_path']), bank=data['bank'],
                 dependencies={m.__name__: file_hash(Path(m.__file__)) for m in
                               (frozen, source, duration, rocket, neural, benchmark, replay)},
                 runtime=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                              torch=str(torch.__version__), platform=platform.platform(), device='cpu'))
    scope['study_fingerprint'] = neural._hash(scope)
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    review = root / 'review'
    review.mkdir(exist_ok=True)
    seeds = settings['training_seeds'][:1] if smoke else settings['training_seeds']
    folds = [1] if smoke else list(range(1, 11))
    with source.runner_lock(root):
        write_or_verify(root / 'study_manifest.json', scope, verify)
        write_or_verify(review / 'source_custody.json', source_hashes, verify)
        heads, ridges, contexts, checks = {}, {}, {}, []
        predictions = {arm: {s: np.full((len(bank.ids), 6), np.nan) for s in seeds} for arm in ARMS}
        for fold in folds:
            train, query = np.flatnonzero(data['folds'] != fold), np.flatnonzero(data['folds'] == fold)
            for seed in seeds:
                p = evaluate(data, bank, config, settings, encoder_scope, encoder_hashes, source_config,
                             source_scope, source_hashes, scope, review, seed, fold, train, query,
                             verify, heads, ridges, contexts, checks, smoke)
                if not smoke:
                    for arm, value in p.items():
                        predictions[arm][seed][query] = value
            _progress(root, state='verifying' if verify else 'fitting', fold=fold,
                      feature_contexts=len(contexts), distance_heads=len(heads), source_ridge_heads=len(ridges))
            print(f"{'Verified' if verify else 'Saved'} fold {fold}/{len(folds)}; contexts {len(contexts)}, distance heads {len(heads)}", flush=True)
        if (len(heads), len(ridges), len(contexts)) != ((4, 2, 1) if smoke else (420, 210, 120)):
            raise ValueError('Incomplete declared distance inventory')
        if len(checks) < (1 if smoke else 90) or custody(Path(config['source_root'])) != source_hashes:
            raise ValueError('Identity controls incomplete or protected source files changed')
        if not verify:
            _progress(root, state='fits_saved', distance_heads=len(heads), feature_contexts=len(contexts))
            return
        result = dict(complete=True, smoke=smoke, classes=config['classes'], seeds=seeds, folds=folds,
                      completed_utc=datetime.now(timezone.utc).isoformat(), encoder_updates=0,
                      distance_heads=len(heads), source_ridge_heads=len(ridges), feature_contexts=len(contexts),
                      outer_distance_heads=0 if smoke else 60, identity_controls=len(checks),
                      source_files_unchanged=len(source_hashes), study_fingerprint=scope['study_fingerprint'],
                      verification_sha256=neural._hash(dict(heads=heads, ridges=ridges, contexts=contexts, controls=checks)),
                      cohorts=None if smoke else summarize(data, config, predictions, prior))
        # Completion timestamp is history; an identical replay retains it.
        if (root / 'summary.json').exists():
            result['completed_utc'] = replay.read(root / 'summary.json')['completed_utc']
        write_or_verify(root / 'summary.json', result, False)
        report = render(result)
        path = root / 'REPORT.md'
        if path.exists() and path.read_text() != report:
            raise ValueError('Completed distance report changed')
        if not path.exists():
            temporary = path.with_suffix('.partial')
            temporary.write_text(report)
            temporary.replace(path)
        _progress(root, state='complete_validated', distance_heads=len(heads), feature_contexts=len(contexts),
                  source_ridge_heads=len(ridges))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.smoke, args.verify)


if __name__ == '__main__':
    main()
