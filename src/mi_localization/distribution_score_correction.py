"""Nested additive corrections to an unchanged VCG ridge decision function."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import benchmark, distribution_kernel_source as source
from .distribution_study import membership, select
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

BRANCHES = {
    'convolution_spread': ('convolution', 'plus_spread'),
    'convolution_quantiles': ('convolution', 'plus_quantiles'),
    'learned_spread': ('learned', 'plus_spread'),
    'learned_quantiles': ('learned', 'plus_quantiles'),
}


def read(path):
    return json.loads(path.read_text())


def load_inputs(config):
    ratios = config['penalty_ratios']
    if (len(ratios) != 3 or ratios[0] is not None or
            not np.isfinite(ratios[1:]).all() or not ratios[1] > ratios[2] > 0):
        raise ValueError('Require no correction and two positive penalties, strongest first')
    bank, _ = source.validate_chain(config)
    learned = yaml.safe_load(Path(config['learned_config']).read_text())
    root = Path(learned['output_dir']) / 'complete'
    summary = read(root / 'summary.json')
    scope = read(root / 'study_manifest.json')
    verification = read(root / 'verification_manifest.json')
    groups = config['outer_splits'] * len(config['projection_seeds'])
    contexts = groups * (1 + config['inner_splits'])
    source_models = groups * (1 + config['inner_splits'] * len(config['ridge_alphas']))
    fingerprint = config['expected_learned_fingerprint']
    if (
        file_hash(root / 'summary.json') != config['expected_learned_summary_sha256']
        or scope['config'] != learned
        or any(doc['study_fingerprint'] != fingerprint for doc in (summary, scope, verification))
        or _hash({k: v for k, v in scope.items() if k != 'study_fingerprint'}) != fingerprint
        or scope['source']['cached_features'] != bank.provenance
        or scope['runtime'] != bank.source_scope['runtime']
        or file_hash(root / 'REPORT.md') != summary['report_sha256']
        or file_hash(root / 'verification_manifest.json') != summary['verification_manifest_sha256']
        or summary['feature_contexts_reconstructed'] != contexts
        or summary['matched_context_checks'] != contexts
        or summary['teacher_query_predictions_reproduced'] != contexts
        or summary['all_models_restored'] != 4 * source_models
        or any(summary[k] is not True for k in (
            'complete', 'all_original_beats_replayed', 'original_cohort_unchanged',
            'source_files_unchanged', 'baseline_metrics_exact'))
        or any(summary[k] != bank.summary[k] for k in (
            'n_patients', 'n_records', 'n_beats', 'base_dimensions', 'auxiliary_dimensions'))
        or any(file_hash(Path(p)) != digest for p, digest in scope['code'].items())
    ):
        raise ValueError('Require the verified learned cache paired with this convolution source')
    code_modules = (
        'distribution_score_correction', 'distribution_score_correction_report',
        'distribution_kernel_source', 'distribution_study', 'benchmark',
        'multiview_bank', 'neural', 'signal_audit', 'residual_context_report',
    )
    paths = [Path(__file__).with_name(f'{module}.py') for module in code_modules]
    paths.append(Path('scripts/run_distribution_score_correction.py'))
    current = {
        'config': config,
        'runtime': scope['runtime'],
        'convolution': bank.provenance,
        'learned_fingerprint': fingerprint,
        'learned_verification_sha256': summary['verification_manifest_sha256'],
        'code': {str(path): file_hash(path) for path in paths},
    }
    current['study_fingerprint'] = _hash(current)
    return bank, root, scope, verification['restored_audit_hashes'], current


def checked_features(inputs, seed, fold, inner, train, query):
    bank, root, learned_scope, hashes, _ = inputs
    convolution, conv_path = source.context(bank, seed, fold, inner, train, query)
    path = root / 'review/contexts' / conv_path.name
    digest = file_hash(path)
    audit = read(path)
    expected = membership(bank, learned_scope, train, query)
    expected.update(projection_seed=seed, outer_fold=fold, inner_fold=inner,
                    coverage_mode='all_original_beats')
    data = path.with_suffix('.npz')
    if (digest != hashes[str(path.relative_to(root))]
            or any(audit.get(k) != value for k, value in expected.items())
            or file_hash(data) != audit['features_sha256']):
        raise ValueError('Learned feature audit or membership does not match')
    base = convolution['baseline']
    dimensions = bank.summary['base_dimensions']
    auxiliary = bank.summary['auxiliary_dimensions']
    features = {}
    with np.load(data, allow_pickle=False) as saved:
        if not np.array_equal(saved['rows'], np.r_[train, query]):
            raise ValueError('Learned patient order does not match')
        if not np.array_equal(saved['baseline'], base):
            raise ValueError('The cache families must share the exact baseline')
        for branch, (family, representation) in BRANCHES.items():
            matrix = (convolution if family == 'convolution' else saved)[representation]
            if (matrix.shape != (len(base), dimensions + auxiliary)
                    or not np.array_equal(matrix[:, :dimensions], base)
                    or not np.isfinite(matrix).all()):
                raise ValueError('Invalid matched auxiliary feature block')
            features[branch] = np.asarray(matrix[:, dimensions:], dtype=np.float64)
    return base, features, conv_path, digest


def fit_corrections(auxiliary, train_logits, labels, ratios):
    """Fit a weighted, centered branch in label-score space."""
    count, classes = train_logits.shape
    support = np.bincount(labels, minlength=classes)
    if np.any(support == 0):
        raise ValueError('Correction training requires all classes')
    weights = count / (classes * support[labels])
    mean = np.average(auxiliary, axis=0, weights=weights)
    centered = auxiliary - mean
    weighted = centered * np.sqrt(weights[:, None])
    gram = weighted.T @ weighted
    energy = float(np.trace(gram))
    residuals = (2 * np.eye(classes)[labels] - 1) - train_logits
    rhs = weighted.T @ (np.sqrt(weights[:, None]) * residuals)
    coefficients = []
    for ratio in ratios:
        if ratio is None or energy <= 1e-12:
            coefficients.append(np.zeros_like(rhs))
        else:
            penalized = gram + (ratio * energy) * np.eye(len(gram))
            coefficients.append(cho_solve(cho_factor(penalized, lower=True), rhs))
    return mean, np.stack(coefficients)


def score(labels, probabilities, n_classes):
    return float(f1_score(labels, probabilities.argmax(axis=1), labels=np.arange(n_classes),
                          average='macro', zero_division=0))


def save_record(path, audit, arrays, verify):
    data = path.with_suffix('.npz')
    if verify or path.exists():
        receipt = read(path)
        if receipt != dict(audit, data_sha256=file_hash(data)):
            raise ValueError(f'Correction receipt changed: {path}')
        with np.load(data, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[name], value) for name, value in arrays.items()
            ):
                raise ValueError(f'Rebuilt correction differs from saved model or queries: {path}')
            return {name: saved[name] for name in arrays}
    temporary = data.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(data)
    _save_json(path, dict(audit, data_sha256=file_hash(data)))
    return arrays


def evaluate_context(inputs, output, seed, fold, inner, train, query, verify, selection=None):
    bank, _, _, _, scope = inputs
    config = scope['config']
    n_classes = len(config['classes'])
    ratios = config['penalty_ratios']
    base, features, context, learned_hash = checked_features(inputs, seed, fold, inner, train, query)
    candidate_ids = range(len(config['ridge_alphas'])) if inner else [selection['baseline_candidate']]
    arrays = {'query_rows': query}
    source_hashes = {}
    baseline_scores = []
    scores = {branch: [] for branch in BRANCHES}
    values = {}
    for candidate in candidate_ids:
        baseline_p, baseline_audit, model_audit = source.reference(
            bank, seed, fold, inner, 'baseline', candidate, train, query, base, context)
        source_hashes[str(candidate)] = file_hash(model_audit)
        model = joblib.load(model_audit.with_suffix('.joblib'))[-1]
        if not np.array_equal(model.classes_, np.arange(n_classes)):
            raise ValueError('Baseline class order changed')
        logits = model.decision_function(base)
        if not np.isfinite(logits).all():
            raise ValueError('Invalid frozen baseline scores')
        arrays[f'baseline_{candidate}'] = baseline_p
        values['baseline'] = f'baseline_{candidate}'
        if inner:
            baseline_score = score(bank.labels[query], baseline_p, n_classes)
            if baseline_score != baseline_audit['validation_macro_f1']:
                raise ValueError('Restored baseline inner score changed')
            baseline_scores.append(baseline_score)
        for branch, auxiliary in features.items():
            picks = list(range(len(ratios))) if inner else [selection['corrections'][branch]]
            chosen_ratios = [ratios[i] for i in picks]
            mean, coefficients = fit_corrections(
                auxiliary[:len(train)], logits[:len(train)], bank.labels[train], chosen_ratios)
            prefix = f'{branch}_{candidate}'
            arrays[f'{prefix}_mean'] = mean
            arrays[f'{prefix}_coefficients'] = coefficients
            p = []
            for ratio, coef in zip(chosen_ratios, coefficients):
                if ratio is None or not np.any(coef):
                    p.append(baseline_p)
                else:
                    correction = (auxiliary[len(train):] - mean) @ coef
                    p.append(softmax(logits[len(train):] + correction, axis=1))
            arrays[prefix] = np.stack(p)
            if inner:
                scores[branch].append([score(bank.labels[query], trial, n_classes) for trial in p])
            else:
                values[branch] = prefix
    audit = dict(membership(bank, scope, train, query), fold=fold, seed=seed, inner=inner,
                 convolution_context_sha256=file_hash(context), learned_context_sha256=learned_hash,
                 baseline_models=source_hashes, baseline_inner_scores=baseline_scores,
                 branch_inner_scores=scores, outer_selection=selection)
    path = output / f'fold{fold:02d}_s{seed}_inner{inner}.json'
    restored = save_record(path, audit, arrays, verify)
    if inner:
        return baseline_scores, scores
    return {arm: restored[key] if arm == 'baseline' else restored[key][0] for arm, key in values.items()}


def choose(inner_scores):
    baseline, baseline_means = select(np.asarray([item[0] for item in inner_scores]).T)
    corrections, means = {}, {}
    for branch in BRANCHES:
        candidates = np.asarray([item[1][branch][baseline] for item in inner_scores]).T
        corrections[branch], means[branch] = select(candidates)
    return dict(baseline_candidate=baseline, baseline_inner_means=baseline_means,
                corrections=corrections, correction_inner_means=means)


def evaluate_fold(inputs, output, seed, fold, verify):
    bank, _, _, _, scope = inputs
    config = scope['config']
    training = np.flatnonzero(bank.assigned != fold)
    splits = benchmark.inner_patient_splits(bank.labels[training], bank.ids[training],
                                            config['inner_splits'], config['split_seed'] + 1000 * fold)
    results = []
    for inner, (ti, vi) in enumerate(splits, 1):
        results.append(evaluate_context(inputs, output, seed, fold, inner,
                                        training[ti], training[vi], verify))
    selection = choose(results)
    query = np.flatnonzero(bank.assigned == fold)
    values = evaluate_context(inputs, output, seed, fold, 0, training, query, verify, selection)
    return query, values, selection


def prepare_output(config, scope, smoke):
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    output = root / 'review'
    output.mkdir(parents=True, exist_ok=True)
    manifest = root / 'study_manifest.json'
    if manifest.exists():
        if read(manifest) != scope:
            raise ValueError('Study fingerprint changed. Use a fresh namespace.')
    else:
        _save_json(manifest, scope)
    return root, output


def run(config, smoke=False):
    inputs = load_inputs(config)
    bank, _, _, _, scope = inputs
    root, output = prepare_output(config, scope, smoke)
    if smoke:
        training = np.flatnonzero(bank.assigned != 1)
        ti, vi = next(iter(benchmark.inner_patient_splits(
            bank.labels[training], bank.ids[training], config['inner_splits'],
            config['split_seed'] + 1000)))
        seed = config['projection_seeds'][0]
        train, query = training[ti], training[vi]
        evaluate_context(inputs, output, seed, 1, 1, train, query, False)
        evaluate_context(inputs, output, seed, 1, 1, train, query, True)
        _save_json(root / 'summary.json', {
            'complete': True, 'study_fingerprint': scope['study_fingerprint'],
            'outer_test_patients_used': False,
            'query_count': len(query), 'baseline_models_restored': len(config['ridge_alphas']),
            'nonzero_correction_candidates_restored': len(BRANCHES) * len(config['ridge_alphas']) *
                                                       (len(config['penalty_ratios']) - 1),
        })
        print(f'Inner-only smoke restored both baseline candidates and all correction candidates', flush=True)
        return
    for fold in range(1, config['outer_splits'] + 1):
        for seed in config['projection_seeds']:
            evaluate_fold(inputs, output, seed, fold, False)
        print(f'Fitted fold {fold}/{config["outer_splits"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/distribution_score_correction_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config['torch_threads'])
    with threadpool_limits(limits=1):
        run(config, args.smoke)


if __name__ == '__main__':
    main()
