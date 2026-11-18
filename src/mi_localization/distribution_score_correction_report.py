"""Replay every correction fit and aggregate held-out patient predictions."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import benchmark, distribution_score_correction as study
from . import residual_context_report as uncertainty
from .neural import _save_json
from .signal_audit import file_hash

PAIRS = (
    ('learned_quantiles', 'baseline'),
    ('learned_spread', 'baseline'),
    ('convolution_quantiles', 'baseline'),
    ('convolution_spread', 'baseline'),
    ('learned_quantiles', 'learned_spread'),
    ('convolution_quantiles', 'convolution_spread'),
    ('learned_quantiles', 'convolution_quantiles'),
)


def report(config):
    inputs = study.load_inputs(config)
    bank, _, _, _, scope = inputs
    root, output = study.prepare_output(config, scope, False)
    seeds = config['projection_seeds']
    classes = config['classes']
    labels = np.arange(len(classes))
    arms = ('baseline', *study.BRANCHES)
    folds = range(1, config['outer_splits'] + 1)
    members = {arm: {seed: np.full((len(bank.ids), len(classes)), np.nan)
                     for seed in seeds} for arm in arms}
    counts = {branch: [0] * len(config['penalty_ratios']) for branch in study.BRANCHES}
    baseline_counts = [0] * len(config['ridge_alphas'])
    for fold in folds:
        for seed in seeds:
            patients, probabilities, selected = study.evaluate_fold(inputs, output, seed, fold, True)
            baseline_counts[selected['baseline_candidate']] += 1
            for branch, index in selected['corrections'].items():
                counts[branch][index] += 1
                if index == 0 and not np.array_equal(probabilities[branch], probabilities['baseline']):
                    raise ValueError('Disabled correction must reproduce baseline probabilities exactly')
            for arm in arms:
                if np.isfinite(members[arm][seed][patients]).any():
                    raise ValueError('Duplicate held-out patient predictions')
                members[arm][seed][patients] = probabilities[arm]
        print(f'Replayed fold {fold}/{len(folds)}', flush=True)
    contexts = sorted(output.glob('fold*_s*_inner*.json'))
    groups = len(folds) * len(seeds)
    if (len(contexts) != groups * (1 + config['inner_splits'])
            or any(not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1)
                   for runs in members.values() for p in runs.values())):
        raise ValueError('Incomplete correction replay')
    results, ensemble, members_metrics = {}, {}, {}
    weights = uncertainty.patient_weights(bank.labels, labels, config['bootstrap_draws'], config['seed'])
    paired_draws, member_draws = {}, {}
    for arm, runs in members.items():
        probabilities = np.stack([runs[seed] for seed in seeds])
        ensemble_p = probabilities.mean(axis=0)
        ensemble[arm] = benchmark.metric_bundle(bank.labels, ensemble_p, classes)
        members_metrics[arm] = {
            str(seed): benchmark.metric_bundle(bank.labels, runs[seed], classes)
            for seed in seeds
        }
        scores = [metrics['macro_f1'] for metrics in members_metrics[arm].values()]
        results[arm] = {'mean': float(np.mean(scores)), 'seed_sd': float(np.std(scores, ddof=1))}
        paired_draws[arm] = uncertainty.weighted_f1(bank.labels, ensemble_p.argmax(1), labels, weights)
        member_draws[arm] = np.mean([
            uncertainty.weighted_f1(bank.labels, p.argmax(1), labels, weights)
            for p in probabilities
        ], axis=0)
    if (ensemble['baseline'] != bank.summary['ensemble_metrics']['baseline']
            or members_metrics['baseline'] != bank.summary['member_metrics']['baseline']):
        raise ValueError('Frozen baseline metrics do not reproduce the source')
    contrasts = {}
    for left, right in PAIRS:
        contrasts[f'{left}_minus_{right}'] = {
            'ensemble_delta_macro_f1': ensemble[left]['macro_f1'] - ensemble[right]['macro_f1'],
            'descriptive_patient_interval': np.quantile(
                paired_draws[left] - paired_draws[right], [.025, .975]).tolist(),
            'mean_member_delta_macro_f1': results[left]['mean'] - results[right]['mean'],
            'mean_member_descriptive_interval': np.quantile(
                member_draws[left] - member_draws[right], [.025, .975]).tolist(),
        }
    inner_contexts = groups * config['inner_splits']
    source_restores = inner_contexts * len(config['ridge_alphas']) + groups
    inner_fits = (inner_contexts * len(config['ridge_alphas']) * len(study.BRANCHES)
                  * (len(config['penalty_ratios']) - 1))
    outer_fits = sum(sum(branch[1:]) for branch in counts.values())
    manifest = root / 'verification_manifest.json'
    _save_json(manifest, {
        'study_fingerprint': scope['study_fingerprint'],
        'context_audit_hashes': {str(p.relative_to(root)): file_hash(p) for p in contexts},
    })
    lines = [
        '# Frozen baseline score correction', '',
        f'All {len(bank.ids)} patients, {bank.n_records} recordings and {bank.n_beats:,} source beats '
        f'are represented. All {len(contexts)} partition/seed contexts replayed, with {source_restores} '
        f'baseline restores and {inner_fits} inner + {outer_fits} outer active correction fits reproduced. '
        'The original baseline metrics reproduced exactly.', '',
        '| Procedure | Mean member macro F1 ± seed SD | Three-seed ensemble macro F1 |',
        '|---|---:|---:|',
    ]
    for arm in arms:
        stats = results[arm]
        lines.append(f"| {arm} | {stats['mean']:.6f} ± {stats['seed_sd']:.6f} | "
                     f"{ensemble[arm]['macro_f1']:.6f} |")
    lines.extend([
        '', 'The primary comparison is learned quantiles against the frozen baseline.', '',
        '| Paired comparison | Ensemble difference | Descriptive patient 95% interval |',
        '|---|---:|---:|',
    ])
    for name, contrast in contrasts.items():
        lo, hi = contrast['descriptive_patient_interval']
        lines.append(f"| {name} | {contrast['ensemble_delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend([
        '', '| Procedure | Correction off | Strong penalty | Weaker penalty |',
        '|---|---:|---:|---:|',
    ])
    for arm, (off, strong, weak) in counts.items():
        lines.append(f'| {arm} | {off} | {strong} | {weak} |')
    lines.extend([
        '', '| Class | Baseline | Convolution spread | Convolution quantiles | Learned spread | Learned quantiles |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for name in classes:
        values = [ensemble[arm]['classification_report'][name]['f1-score'] for arm in arms]
        lines.append(f"| {name} | " + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    lines.extend([
        '', 'The frozen baseline alpha and correction penalty are chosen with inner patient macro F1. '
        'Penalty ties favor the disabled branch and then the stronger penalty. The baseline uses its '
        'original coefficients. Each auxiliary branch learns a centered linear correction from the '
        'frozen baseline’s in-sample training residuals. Those residuals reflect the baseline’s degree '
        'of fit in the training partition. Softmax converts scores to probabilities for equal-seed averaging.',
        '', 'This is a cache-based follow-up. Numerical replay checked source context membership, '
        'cached feature hashes, matching baseline columns, baseline probabilities, saved correction '
        'coefficients and inner decisions. The upstream studies provide the waveform replay evidence. '
        'Inherited inner-validation early stopping of the frozen learned encoder still applies.',
        '', 'PTB remains an inspected development cohort. The paired intervals use shared, '
        'class-stratified patient resampling of fixed predictions. They omit refits and adjustments '
        'for overlapping folds, repeated development or multiple comparisons. Selection counts '
        'describe overlapping partition decisions.', '',
    ])
    report_path = root / 'REPORT.md'
    temporary = report_path.with_suffix('.partial.md')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(report_path)
    summary = {
        'complete': True, 'study_fingerprint': scope['study_fingerprint'],
        'evaluated_outer_folds': list(folds), 'seeds': seeds,
        'n_patients': len(bank.ids), 'n_records': bank.n_records, 'n_beats': bank.n_beats,
        'contexts_restored': len(contexts), 'baseline_models_restored': source_restores,
        'outer_groups_restored': groups, 'inner_active_fits_restored': inner_fits,
        'outer_active_fits_restored': outer_fits, 'baseline_metrics_exact': True,
        'primary_contrast': 'learned_quantiles_minus_baseline', 'contrasts': contrasts,
        'ensemble_metrics': ensemble, 'mean_member_metrics': results, 'member_metrics': members_metrics,
        'baseline_alpha_counts': baseline_counts, 'penalty_ratios': config['penalty_ratios'],
        'correction_selection_counts': counts, 'waveform_extractions': 0,
        'report_sha256': file_hash(report_path), 'verification_manifest_sha256': file_hash(manifest),
    }
    _save_json(root / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/distribution_score_correction_v1.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config['torch_threads'])
    with threadpool_limits(limits=1):
        result = report(config)
    print(f"Independent replay completed for {result['contexts_restored']} contexts", flush=True)
