"""Report full patient-level comparisons from the independently replayed fits."""
import numpy as np

from .benchmark import metric_bundle
from .neural import _save_json
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash

PAIRS = (
    ('subtype_graph', 'ridge'),
    ('subtype_graph', 'relabeled_graph'),
    ('relabeled_graph', 'ridge'),
)


def summarize(root, scope, bank, members, counts):
    config = scope['config']
    classes = config['classes']
    seeds = config['projection_seeds']
    groups = config['outer_splits'] * len(seeds)
    inner = groups * config['inner_splits']
    contexts = sorted((root / 'review').glob('fold*_inner*.json'))
    if len(contexts) != inner + groups or any(sum(n) != groups for n in counts.values()):
        raise ValueError('Incomplete verified fit coverage')
    if any(
        not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1)
        for runs in members.values() for p in runs.values()
    ):
        raise ValueError('Missing or invalid held-out patient probabilities')
    labels = np.arange(len(classes))
    weights = patient_weights(bank.labels, labels, config['bootstrap_draws'], config['seed'])
    ensemble_metrics, member_metrics, means, ensemble_draws, mean_draws = {}, {}, {}, {}, {}
    label_predictions = {}
    for arm, runs in members.items():
        metrics = {str(seed): metric_bundle(bank.labels, runs[seed], classes) for seed in seeds}
        member_metrics[arm] = metrics
        scores = [metrics[str(seed)]['macro_f1'] for seed in seeds]
        means[arm] = dict(mean=float(np.mean(scores)), seed_sd=float(np.std(scores, ddof=1)))
        p = np.mean([runs[seed] for seed in seeds], axis=0)
        ensemble_metrics[arm] = metric_bundle(bank.labels, p, classes)
        label_predictions[arm] = p.argmax(1)
        ensemble_draws[arm] = weighted_f1(bank.labels, label_predictions[arm], labels, weights)
        mean_draws[arm] = np.mean(
            [weighted_f1(bank.labels, runs[seed].argmax(1), labels, weights) for seed in seeds], axis=0)
    if (ensemble_metrics['ridge'] != bank.summary['ensemble_metrics']['baseline'] or
            member_metrics['ridge'] != bank.summary['member_metrics']['baseline']):
        raise ValueError('Frozen source ridge metrics changed')
    comparisons = {}
    for a, b in PAIRS:
        comparisons[f'{a}_minus_{b}'] = dict(
            ensemble_delta=ensemble_metrics[a]['macro_f1'] - ensemble_metrics[b]['macro_f1'],
            ensemble_interval=np.quantile(ensemble_draws[a] - ensemble_draws[b], [.025, .975]).tolist(),
            mean_member_delta=means[a]['mean'] - means[b]['mean'],
            mean_member_interval=np.quantile(mean_draws[a] - mean_draws[b], [.025, .975]).tolist(),
            ensemble_label_changes=int(np.count_nonzero(label_predictions[a] != label_predictions[b])),
        )
    baselines = inner * len(config['ridge_alphas']) + groups
    inner_fits = inner * len(counts) * (len(config['strengths']) - 1)
    outer_fits = sum(sum(choices[1:]) for choices in counts.values())
    manifest = root / 'verification_manifest.json'
    _save_json(manifest, dict(study_fingerprint=scope['study_fingerprint'],
                             fit_context_hashes={str(p.relative_to(root)): file_hash(p) for p in contexts}))
    lines = [
        '# Subtype graph regularization', '',
        f'All {len(bank.ids)} patients, {bank.n_records} recordings and {bank.n_beats:,} original source beats. '
        f'Replayed all {len(contexts)} feature contexts, {baselines} ordinary ridge references, '
        f'{inner_fits} positive inner candidates and {outer_fits} positive outer fits. '
        'The original convolution ridge metrics reproduced exactly.', '',
        '| Arm | Mean member macro F1 ± seed SD | Equal-seed ensemble macro F1 |',
        '|---|---:|---:|',
    ]
    for arm in members:
        lines.append(f"| {arm} | {means[arm]['mean']:.6f} ± {means[arm]['seed_sd']:.6f} | "
                     f"{ensemble_metrics[arm]['macro_f1']:.6f} |")
    lines.extend([
        '', 'The primary contrast is subtype graph versus ordinary ridge. The relabeled graph checks '
        'whether the declared subtype links improve on the same amount of graph structure.', '',
        '| Paired comparison | Ensemble macro F1 difference | Descriptive patient 95% interval |',
        '|---|---:|---:|',
    ])
    for name, comparison in comparisons.items():
        lo, hi = comparison['ensemble_interval']
        lines.append(f"| {name} | {comparison['ensemble_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend([
        '', '| Arm | Sharing off | Strength 0.5 | Strength 2 |', '|---|---:|---:|---:|',
    ])
    for arm, (off, weak, strong) in counts.items():
        lines.append(f'| {arm} | {off} | {weak} | {strong} |')
    lines.extend([
        '', '| Class | Ordinary ridge F1 | Subtype graph F1 | Relabeled graph F1 |',
        '|---|---:|---:|---:|',
    ])
    for name in classes:
        values = [ensemble_metrics[arm]['classification_report'][name]['f1-score'] for arm in members]
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines.extend([
        '', 'Both graphs preserve six independent score columns while penalizing differences between linked '
        'weight vectors. Both use the same inner-selected ordinary ridge alpha and the same three graph '
        'strengths. Only the MI label assignments on the edges differ. The control preserves topology and '
        'keeps HC isolated, as does the subtype graph. Zero strength uses the original ridge probabilities.',
        '', 'All ten original patient folds and all three feature seeds contribute to the results. The study '
        'uses verified convolution feature caches with independent classifier replay. The graph reflects '
        'relationships proposed from the label names. Patient bootstrap intervals describe fixed predictions '
        'on this repeatedly inspected development cohort and omit refitting and repeated-development effects.', '',
    ])
    report = root / 'REPORT.md'
    temporary = report.with_suffix('.partial.md')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(report)
    _save_json(root / 'summary.json', dict(
        complete=True, study_fingerprint=scope['study_fingerprint'], n_patients=len(bank.ids),
        n_records=bank.n_records, n_beats=bank.n_beats,
        evaluated_folds=list(range(1, config['outer_splits'] + 1)), seeds=seeds,
        feature_contexts_replayed=len(contexts), baseline_models_replayed=baselines,
        positive_inner_graph_fits_replayed=inner_fits, positive_outer_graph_fits_replayed=outer_fits,
        baseline_metrics_exact=True, primary_contrast='subtype_graph_minus_ridge',
        ensemble_metrics=ensemble_metrics, member_metrics=member_metrics,
        mean_member_metrics=means, comparisons=comparisons, strength_selection_counts=counts,
        report_sha256=file_hash(report), verification_manifest_sha256=file_hash(manifest),
    ))
