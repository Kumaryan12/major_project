"""Matched flat and probabilistic MI subtype logistic models on frozen features."""
import argparse
from pathlib import Path
import warnings

import numpy as np
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import yaml

from . import benchmark, distribution_kernel_source as source, distribution_score_correction as receipts
from . import model_error_comparison as historical, residual_context_report as bootstrap
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

ARMS = ('flat', 'hierarchical')


def combine_probabilities(mi, conditional, subtype_codes, healthy):
    """Map the factorized probabilities back to the frozen class positions."""
    codes = np.asarray(subtype_codes)
    n_classes = len(codes) + 1
    if (mi.ndim != 1 or conditional.shape != (len(mi), len(codes))
            or set(codes) != set(range(n_classes)) - {healthy}
            or not np.isfinite(mi).all() or not np.isfinite(conditional).all()
            or np.any((mi < 0) | (mi > 1)) or np.any(conditional < 0)
            or not np.allclose(conditional.sum(1), 1)):
        raise ValueError('Invalid conditional subtype probabilities or class order')
    out = np.empty((len(mi), n_classes), dtype=np.float64)
    out[:, healthy] = 1 - mi
    out[:, codes] = mi[:, None] * conditional
    return out


def fit_logit(x, labels, weights, query, setting, config):
    # A common per-patient regularization convention keeps C comparable across nodes.
    model = LogisticRegression(C=setting, solver='lbfgs', max_iter=config['max_iterations'], tol=config['tolerance'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, labels, sample_weight=weights / weights.mean())
    state = dict(coef=model.coef_, intercept=model.intercept_, classes=model.classes_, iterations=model.n_iter_)
    score = query @ state['coef'].T + state['intercept']
    if len(state['classes']) == 2:
        positive = expit(score[:, 0])
        probabilities = np.column_stack((1 - positive, positive))
    else:
        probabilities = softmax(score, axis=1)
    if not np.allclose(probabilities, model.predict_proba(query), atol=1e-12, rtol=1e-10):
        raise ValueError('Saved logistic coefficients do not reproduce the estimator')
    return probabilities, state


def predict_arms(x, y, weights, query, healthy, setting, config):
    flat, flat_state = fit_logit(x, y, weights, query, setting, config)
    mi = y != healthy
    binary, binary_state = fit_logit(x, mi.astype(np.int64), weights, query, setting, config)
    cond, conditional_state = fit_logit(x[mi], y[mi], weights[mi], query, setting, config)
    hierarchical = combine_probabilities(binary[:, 1], cond, conditional_state['classes'], healthy)
    return dict(flat=flat, hierarchical=hierarchical), dict(flat=flat_state, mi=binary_state, subtype=conditional_state)


def load_reference(config, bank):
    orig = yaml.safe_load(Path(config['historical_reference_config']).read_text())
    root = Path(orig['convolution_root'])
    summary, scope, evidence = historical.read_source(root, orig['convolution_summary_sha256'], 'study_fingerprint')
    historical.require_fields(scope['config'], {k: config[k] for k in (
        'classes', 'projection_seeds', 'representations', 'outer_splits', 'inner_splits', 'split_seed')})
    historical.require_fields(scope['source'], dict(
        patient_ids_sha256=array_hash(bank.ids), labels_sha256=array_hash(bank.labels),
        folds_sha256=file_hash(config['folds_path']), source_fingerprint=bank.source_scope['study_fingerprint']))
    return root, summary, scope, evidence


def representation(ref, fold, seed):
    root, _, scope, evidence = ref
    filename = f'review/paired_selection_{fold:02d}_s{seed}.json'
    digest = evidence['restored_audit_hashes'][filename]
    selected = historical.checked_json(root / filename, digest)
    historical.require_fields(selected, dict(study_fingerprint=scope['study_fingerprint'], outer_fold=fold,
                                             projection_seed=seed))
    reps = scope['config']['representations']
    scores = selected['symmetric_inner_scores']
    if (set(reps) != set(scores) or not np.isfinite(list(scores.values())).all()
            or selected['selected_representation'] != max(reps, key=scores.__getitem__)):
        raise ValueError('Historical train-only representation choice did not reproduce')
    return selected['selected_representation'], digest


def context(config, bank, root, scope, rep, selected_digest, fold, seed, inner, train, query, verify, choices=None):
    values, feature_path = source.context(bank, seed, fold, inner, train, query)
    n, k = len(train), len(config['classes'])
    x = values[rep]
    train_labels = bank.labels[train]
    if set(train_labels) != set(range(k)) or np.intersect1d(train, query).size:
        raise ValueError('Expected every training class and disjoint query patients')
    weights = benchmark.patient_balanced_weights(train_labels, bank.ids[train])
    arrays = dict(query_rows=query)
    scores = {a: [] for a in ARMS}
    if inner:
        for i, setting in enumerate(config['logistic_c']):
            probabilities, states = predict_arms(
                x[:n], train_labels, weights, x[n:], config['classes'].index('HC'), setting, config)
            for a, p in probabilities.items():
                arrays[f'{a}_{i}_probabilities'] = p
                scores[a].append(receipts.score(bank.labels[query], p, k))
            for head, state in states.items():
                for name, value in state.items():
                    arrays[f'{head}_{i}_{name}'] = value
    else:
        # Each arm uses the same inner candidate list. Its own winner is refit on all outer-train patients.
        healthy = config['classes'].index('HC')
        mi = train_labels != healthy
        flat, flat_state = fit_logit(x[:n], train_labels, weights, x[n:], config['logistic_c'][choices['flat']], config)
        setting = config['logistic_c'][choices['hierarchical']]
        binary, binary_state = fit_logit(x[:n], mi.astype(int), weights, x[n:], setting, config)
        cond, subtype_state = fit_logit(x[:n][mi], train_labels[mi], weights[mi], x[n:], setting, config)
        arrays['flat_probabilities'] = flat
        arrays['hierarchical_probabilities'] = combine_probabilities(binary[:, 1], cond, subtype_state['classes'], healthy)
        for head, state in (('flat', flat_state), ('mi', binary_state), ('subtype', subtype_state)):
            for name, value in state.items():
                arrays[f'{head}_{name}'] = value
    audit = dict(study_fingerprint=scope['fingerprint'], outer_fold=fold, inner_fold=inner, projection_seed=seed,
                 train_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query),
                 train_label_sha256=array_hash(train_labels), train_weight_sha256=array_hash(weights),
                 representation=rep, representation_selection_sha256=selected_digest,
                 feature_context_sha256=file_hash(feature_path), feature_matrix_sha256=array_hash(x),
                 inner_scores=scores if inner else None, choices=choices)
    stem = root / 'review' / f'fold{fold:02d}_seed{seed}_inner{inner}.json'
    restored = receipts.save_record(stem, audit, arrays, verify)
    return scores if inner else {a: restored[f'{a}_probabilities'] for a in ARMS}


def fold_seed(config, bank, ref, root, scope, fold, seed, verify, smoke):
    outer_train = np.flatnonzero(bank.assigned != fold)
    samples = benchmark.inner_patient_splits(bank.labels[outer_train], bank.ids[outer_train],
                                             config['inner_splits'], config['split_seed'] + fold * 1000)
    rep, digest = representation(ref, fold, seed)
    outcomes = [context(config, bank, root, scope, rep, digest, fold, seed, i,
                        outer_train[tr], outer_train[va], verify)
                for i, (tr, va) in enumerate(samples, 1)]
    matrix = {a: np.asarray([rows[a] for rows in outcomes]).T for a in ARMS}
    choices = {a: int(np.argmax(v.mean(axis=1))) for a, v in matrix.items()}
    if smoke:
        return None
    query = np.flatnonzero(bank.assigned == fold)
    return query, context(config, bank, root, scope, rep, digest, fold, seed, 0, outer_train, query, verify, choices)


def error_breakdown(labels, decisions, healthy):
    wrong = labels != decisions
    return dict(within_mi=int(np.count_nonzero(wrong & (labels != healthy) & (decisions != healthy))),
                mi_to_hc=int(np.count_nonzero((labels != healthy) & (decisions == healthy))),
                hc_to_mi=int(np.count_nonzero((labels == healthy) & (decisions != healthy))))


def summarize(config, bank, reference, root, scope, members):
    classes, seeds = config['classes'], config['projection_seeds']
    healthy = classes.index('HC')
    rr, summary, ss, verified = reference
    old, old_hashes = historical.convolution_predictions(rr, ss, verified, bank.ids, bank.labels, bank.assigned, seeds)
    members['historical_selector'] = old
    ensemble_metrics, seed_metrics, mean_seed, error_counts, predictions = {}, {}, {}, {}, {}
    for arm, cases in members.items():
        p = [cases[s] for s in seeds]
        if any(not np.isfinite(x).all() or np.any(x < 0) or not np.allclose(x.sum(1), 1) for x in p):
            raise ValueError('Uncovered held-out patients or invalid probabilities')
        seed_metrics[arm] = {str(s): benchmark.metric_bundle(bank.labels, x, classes) for s, x in zip(seeds, p)}
        mean_seed[arm] = float(np.mean([v['macro_f1'] for v in seed_metrics[arm].values()]))
        q = np.mean(p, axis=0)
        ensemble_metrics[arm] = benchmark.metric_bundle(bank.labels, q, classes)
        predictions[arm] = q.argmax(1)
        error_counts[arm] = error_breakdown(bank.labels, predictions[arm], healthy)
    if (ensemble_metrics['historical_selector'] != summary['ensemble_metrics']['selected_linear']
            or seed_metrics['historical_selector'] != summary['member_metrics']['selected_linear']):
        raise ValueError('Historical reference scores did not reproduce')
    labels = np.arange(len(classes))
    weights = bootstrap.patient_weights(bank.labels, labels, config['bootstrap_draws'], config['bootstrap_seed'])
    draws = {a: bootstrap.weighted_f1(bank.labels, p, labels, weights) for a, p in predictions.items()}
    contrasts = {}
    for a, b in (('hierarchical', 'flat'), ('hierarchical', 'historical_selector'), ('flat', 'historical_selector')):
        delta = ensemble_metrics[a]['macro_f1'] - ensemble_metrics[b]['macro_f1']
        contrasts[f'{a}_minus_{b}'] = dict(delta_macro_f1=delta,
                                          descriptive_patient_interval=np.quantile(draws[a] - draws[b], [.025, .975]).tolist())
    paths = sorted((root / 'review').glob('fold*_inner*.json'))
    groups = config['outer_splits'] * len(seeds)
    contexts = groups * (config['inner_splits'] + 1)
    procedures = 2 * groups * (config['inner_splits'] * len(config['logistic_c']) + 1)
    if len(paths) != contexts:
        raise ValueError('Required matched contexts are missing')
    evidence = root / 'verification_manifest.json'
    _save_json(evidence, dict(study_fingerprint=scope['fingerprint'],
                             contexts={str(p.relative_to(root)): file_hash(p) for p in paths},
                             historical_files=old_hashes))
    lines = [
        '# Probabilistic MI subtype readout on measured XYZ', '',
        f"Replayed {procedures} matched procedures across {contexts} training/query partitions, "
        f"164 patients, ten frozen folds and three feature seeds.", '',
        '| Readout | Mean seed macro F1 | Three-seed ensemble macro F1 |',
        '|---|---:|---:|',
    ]
    for a in members:
        lines.append(f"| {a} | {mean_seed[a]:.6f} | {ensemble_metrics[a]['macro_f1']:.6f} |")
    lines.extend(['', 'The primary comparison is hierarchical minus flat.', '',
                  '| Contrast | Ensemble difference | Descriptive paired patient 95% interval |', '|---|---:|---|'])
    for name, v in contrasts.items():
        lo, hi = v['descriptive_patient_interval']
        lines.append(f"| {name} | {v['delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(['', '| Class | Flat | Hierarchical | Historical selector |', '|---|---:|---:|---:|'])
    for c in classes:
        values = [ensemble_metrics[a]['classification_report'][c]['f1-score'] for a in members]
        lines.append(f'| {c} | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    lines.extend(['', '| Readout | Within-MI confusions | MI predicted HC | HC predicted MI |', '|---|---:|---:|---:|'])
    for a, errors in error_counts.items():
        lines.append(f"| {a} | {errors['within_mi']} | {errors['mi_to_hc']} | {errors['hc_to_mi']} |")
    lines.extend([
        '', 'Both new readouts use the same cached feature matrix in each partition. Their shared '
        'representation came from the historical train-only selector. Each readout uses the same '
        'three-value C grid and its own inner-score choice. The hierarchical model multiplies MI '
        'probability by conditional subtype probability and retains HC probability. It evaluates all '
        'six outcomes for every patient.',
        '', 'This is repeatedly inspected PTB development data. Paired intervals condition on these '
        'predictions and omit refitting and repeated-development effects. The historical selector is a '
        'read-only performance reference.',
        '', '[Protocol](../../../docs/probabilistic_subtype_plan.md)', '',
    ])
    report = root / 'REPORT.md'
    text = '\n'.join(lines)
    if report.is_file() and report.read_text() != text:
        raise ValueError('Previously verified report differs')
    temporary = report.with_suffix('.partial.md')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(report)
    result = dict(complete=True, study_fingerprint=scope['fingerprint'], restored_procedures=procedures,
                  matched_contexts=contexts, reconstructed_waveforms=0, n_patients=len(bank.ids),
                  n_records=bank.n_records, n_beats=bank.n_beats, seeds=seeds, folds=list(range(1, config['outer_splits']+1)),
                  ensemble_metrics=ensemble_metrics, seed_metrics=seed_metrics, mean_seed_macro_f1=mean_seed,
                  error_counts=error_counts, contrasts=contrasts, primary_comparison='hierarchical_minus_flat',
                  report_sha256=file_hash(report), verification_manifest_sha256=file_hash(evidence))
    path = root / 'summary.json'
    if path.is_file() and receipts.read(path) != result:
        raise ValueError('Previously verified cohort metrics differ')
    _save_json(path, result)


def run(config, smoke, verify):
    if sorted(set(config['logistic_c'])) != config['logistic_c'] or any(not np.isfinite(v) or v <= 0 for v in config['logistic_c']):
        raise ValueError('Require a fixed increasing positive C grid')
    bank, _ = source.validate_chain(config)
    if (len(bank.ids), bank.n_records, bank.n_beats) != (164, 391, 55863):
        raise ValueError('Original measurement cohort changed')
    ref = load_reference(config, bank)
    paths = [Path(__file__), Path('scripts/run_probabilistic_subtype.py')] + [
        Path(m.__file__) for m in (source, receipts, benchmark, historical, bootstrap)]
    scope = dict(config=config, source=bank.provenance, smoke=smoke,
                 historical_summary_sha256=file_hash(ref[0] / 'summary.json'),
                 historical_manifest_sha256=file_hash(ref[0] / 'study_manifest.json'),
                 code={p.name: file_hash(p) for p in paths})
    scope['fingerprint'] = _hash(scope)
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    (root / 'review').mkdir(parents=True, exist_ok=True)
    guard = root / 'study_manifest.json'
    if (guard.is_file() or verify) and receipts.read(guard) != scope:
        raise ValueError('Probabilistic subtype study scope changed')
    if not guard.exists():
        _save_json(guard, scope)
    seeds = config['projection_seeds']
    if smoke:
        fold_seed(config, bank, ref, root, scope, 1, seeds[0], verify, True)
        if verify:
            _save_json(root / 'summary.json', dict(complete=True, outer_test_patients_used=False,
                                                   restored_procedures=18, matched_contexts=3,
                                                   study_fingerprint=scope['fingerprint']))
        return
    members = {a: {s: np.full((len(bank.ids), len(config['classes'])), np.nan) for s in seeds} for a in ARMS}
    for fold in range(1, config['outer_splits'] + 1):
        for seed in seeds:
            rows, outcomes = fold_seed(config, bank, ref, root, scope, fold, seed, verify, False)
            for a, p in outcomes.items():
                if np.isfinite(members[a][seed][rows]).any():
                    raise ValueError('Duplicate outer prediction')
                members[a][seed][rows] = p
            _progress(root, state='replaying' if verify else 'fitting', fold=fold, seed=seed)
        print(f"{'Replayed' if verify else 'Fitted'} fold {fold}/{config['outer_splits']}", flush=True)
    if verify:
        summarize(config, bank, ref, root, scope, members)
        _progress(root, state='complete_verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/probabilistic_subtype_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    with threadpool_limits(limits=1):
        run(config, args.smoke, args.verify)
