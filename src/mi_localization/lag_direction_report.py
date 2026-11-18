"""Independent replay report for the within-beat direction tensor."""
import numpy as np

from .benchmark import metric_bundle
from .neural import _save_json
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash

CONTRASTS = (
    ('signed', 'magnitude'),
    ('signed', 'sign_scrambled'),
    ('signed', 'baseline'),
    ('magnitude', 'baseline'),
    ('sign_scrambled', 'baseline'),
)


def summarize(root, scope, bank, members, counts):
    config = scope['config']
    classes, seeds = config['classes'], config['projection_seeds']
    groups = config['outer_splits'] * len(seeds)
    n_inner = groups * config['inner_splits']
    paths = sorted((root / 'review').glob('fold*_inner*.json'))
    if (len(paths) != n_inner + groups or any(sum(c) != groups for c in counts.values())
            or scope['encoded']['beats'] != bank.n_beats or scope['encoded']['records'] != bank.n_records):
        raise ValueError('Require complete declared fit and source beat coverage')
    labels = np.arange(len(classes))
    weights = patient_weights(bank.labels, labels, config['bootstrap_draws'], config['seed'])
    means, member_metrics, ensemble, projected, member_draws = {}, {}, {}, {}, {}
    for arm, runs in members.items():
        ordered = [runs[seed] for seed in seeds]
        if any(not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1) for p in ordered):
            raise ValueError('Incomplete patient probabilities')
        member_metrics[arm] = {str(seed): metric_bundle(bank.labels, p, classes) for seed, p in zip(seeds, ordered)}
        scores = [row['macro_f1'] for row in member_metrics[arm].values()]
        means[arm] = dict(mean=float(np.mean(scores)), seed_sd=float(np.std(scores, ddof=1)))
        p = np.mean(ordered, axis=0)
        ensemble[arm] = metric_bundle(bank.labels, p, classes)
        projected[arm] = weighted_f1(bank.labels, p.argmax(1), labels, weights)
        member_draws[arm] = np.mean([weighted_f1(bank.labels, m.argmax(1), labels, weights) for m in ordered], axis=0)
    if (ensemble['baseline'] != bank.summary['ensemble_metrics']['baseline'] or
            member_metrics['baseline'] != bank.summary['member_metrics']['baseline']):
        raise ValueError('Baseline patient-level results differ from the frozen source')
    comparisons = {}
    for a, b in CONTRASTS:
        comparisons[f'{a}_minus_{b}'] = dict(
            ensemble_delta=ensemble[a]['macro_f1'] - ensemble[b]['macro_f1'],
            ensemble_interval=np.quantile(projected[a] - projected[b], [.025, .975]).tolist(),
            mean_member_delta=means[a]['mean'] - means[b]['mean'],
            mean_member_interval=np.quantile(member_draws[a] - member_draws[b], [.025, .975]).tolist(),
        )
    baseline = n_inner * len(config['ridge_alphas']) + groups
    new = baseline * (len(members) - 1)
    verification = root / 'verification_manifest.json'
    _save_json(verification, dict(study_fingerprint=scope['study_fingerprint'],
                                 context_audits={str(p.relative_to(root)): file_hash(p) for p in paths}))
    report = root / 'REPORT.md'
    lines = [
        '# Within-beat lead direction tensor results', '',
        f"Re-encoded all {bank.n_beats:,} measured XYZ beats, reconstructed all {len(paths)} "
        f"matched contexts, restored {baseline} source ridge references and replayed {new} "
        f"new classifiers across all {len(bank.ids)} target patients.", '',
        '| Arm | Mean member macro F1 ± seed SD | Three-seed ensemble macro F1 |',
        '|---|---:|---:|',
    ]
    for arm in members:
        lines.append(f"| {arm} | {means[arm]['mean']:.6f} ± {means[arm]['seed_sd']:.6f} | {ensemble[arm]['macro_f1']:.6f} |")
    lines.extend([
        '', 'The primary comparison is signed minus magnitude.', '',
        '| Paired comparison | Ensemble difference | Descriptive patient 95% interval |',
        '|---|---:|---:|',
    ])
    for name, contrast in comparisons.items():
        lo, hi = contrast['ensemble_interval']
        lines.append(f"| {name} | {contrast['ensemble_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(['', '| Class | ' + ' | '.join(members) + ' |', '|---|' + '---:|' * len(members)])
    for label in classes:
        f1 = [ensemble[arm]['classification_report'][label]['f1-score'] for arm in members]
        lines.append(f'| {label} | ' + ' | '.join(f'{x:.6f}' for x in f1) + ' |')
    lines.extend([
        '', 'The 144 directional tensor columns summarize within-window lag covariance. The three added arms '
        'share the symmetric block, the original convolution base, training-only scales and the same '
        'alpha budget. The scrambled arm flips the antisymmetric entries per beat while preserving their '
        'magnitudes and the symmetric entries. All beats contribute and the patient summaries weight '
        'recordings equally. Reversal acts on the time index, and the lead coordinates retain the measured frame.',
        '', 'The post-R windows are fixed relative to the original R indices. The sign statistics are '
        'observed lead interactions. The paired patient intervals condition on fixed predictions of this '
        'repeatedly inspected PTB development cohort. They omit refits and repeated-development effects.', '',
        '[Protocol](../../../docs/lag_direction_tensor_plan.md)', '',
    ])
    temporary = report.with_suffix('.partial.md')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(report)
    _save_json(root / 'summary.json', dict(
        complete=True, study_fingerprint=scope['study_fingerprint'], seed_values=seeds,
        outer_folds=list(range(1, config['outer_splits'] + 1)), n_patients=len(bank.ids),
        n_records=bank.n_records, n_beats=bank.n_beats, reconstructed_contexts=len(paths),
        source_classifiers_restored=baseline, new_classifiers_replayed=new, baseline_metrics_exact=True,
        primary_comparison='signed_minus_magnitude', ensemble_metrics=ensemble, member_metrics=member_metrics,
        mean_member_metrics=means, comparisons=comparisons, alpha_choice_counts=counts,
        report_sha256=file_hash(report), verification_manifest_sha256=file_hash(verification),
    ))
