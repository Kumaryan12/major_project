"""Publish patient-level metrics for the independently restored native mention study."""
import numpy as np

from .benchmark import aggregate_patients, metric_bundle
from .neural import _save_json
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash

PAIRS = (
    ('true_source', 'shuffled_source'),
    ('true_source', 'target_descriptors'),
    ('true_source', 'scratch'),
    ('target_descriptors', 'scratch'),
    ('shuffled_source', 'target_descriptors'),
)


def summarize(root, scope, data, predictions, arms, count, changes, verification_hash):
    config = scope['config']
    classes, seeds = config['classes'], config['training_seeds']
    groups = config['outer_splits'] * len(seeds)
    if len(seeds) != 3 or count != groups * len(arms) or len(changes) != groups:
        raise ValueError('Full replay requires all declared folds and seeds')
    members = {a: [] for a in arms}
    truth, ids = None, np.unique(data['groups'])
    for seed in seeds:
        for arm in arms:
            p = predictions[seed][arm]
            if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
                raise ValueError('Incomplete outer beat predictions')
            patient_ids, labels, pooled = aggregate_patients(p, data['y'], data['groups'])
            if not np.array_equal(ids, patient_ids) or (truth is not None and not np.array_equal(labels, truth)):
                raise ValueError('Invalid patient pairing')
            truth = labels
            members[arm].append(pooled)
    labels = np.arange(len(classes))
    weights = patient_weights(truth, labels, config['bootstrap_draws'], config['seed'])
    ensemble_metrics, member_metrics, means, ensemble_draws, member_draws = {}, {}, {}, {}, {}
    for arm, runs in members.items():
        member_metrics[arm] = {str(seed): metric_bundle(truth, p, classes) for seed, p in zip(seeds, runs)}
        scores = [m['macro_f1'] for m in member_metrics[arm].values()]
        means[arm] = dict(mean=float(np.mean(scores)), seed_sd=float(np.std(scores, ddof=1)))
        ensemble = np.mean(runs, axis=0)
        ensemble_metrics[arm] = metric_bundle(truth, ensemble, classes)
        ensemble_draws[arm] = weighted_f1(truth, ensemble.argmax(1), labels, weights)
        member_draws[arm] = np.mean([weighted_f1(truth, p.argmax(1), labels, weights) for p in runs], axis=0)
    contrasts = {}
    for left, right in PAIRS:
        contrasts[f'{left}_minus_{right}'] = dict(
            ensemble_delta=ensemble_metrics[left]['macro_f1'] - ensemble_metrics[right]['macro_f1'],
            descriptive_ensemble_interval=np.quantile(ensemble_draws[left] - ensemble_draws[right], [.025, .975]).tolist(),
            mean_member_delta=means[left]['mean'] - means[right]['mean'],
            descriptive_mean_member_interval=np.quantile(member_draws[left] - member_draws[right], [.025, .975]).tolist(),
        )
    source = scope['source_audit']
    lines = [
        '# Native measured XYZ with partial region mentions', '',
        f"Replayed {count} models and {groups} matched fold/seed groups for all {len(ids)} target patients. "
        f"Audited {source['patients_with_consistent_location']} outside-target patients and used "
        f"{source['patients_with_usable_mentions']} with at least one shared region mention.", '',
        '| Procedure | Mean member macro F1 ± seed SD | Three-seed ensemble macro F1 |',
        '|---|---:|---:|',
    ]
    for arm in arms:
        lines.append(f"| {arm} | {means[arm]['mean']:.6f} ± {means[arm]['seed_sd']:.6f} | {ensemble_metrics[arm]['macro_f1']:.6f} |")
    lines.extend([
        '', 'The primary contrast is true versus shuffled source mentions.', '',
        '| Contrast | Ensemble difference | Descriptive paired patient 95% interval |',
        '|---|---:|---:|',
    ])
    for name, result in contrasts.items():
        lo, hi = result['descriptive_ensemble_interval']
        lines.append(f"| {name} | {result['ensemble_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend([
        '', '| Class | ' + ' | '.join(arms) + ' |', '|---|' + '---:|' * len(arms),
    ])
    for name in classes:
        lines.append(f'| {name} | ' + ' | '.join(
            f"{ensemble_metrics[arm]['classification_report'][name]['f1-score']:.6f}" for arm in arms) + ' |')
    lines.extend([
        '', f"The shuffle reassigned all patient code vectors and changed the observed tag set for "
        f"{min(changes)}–{max(changes)} of the {source['patients_with_usable_mentions']} patients per fold/seed group. "
        'It preserves exact per-update tag frequencies and excludes self-pairs. The repeated tag patterns '
        'limit how many vectors can differ after shuffling. Source loss only uses recorded positive mentions.',
        '', 'All procedures use the same fixed target recipe. Source and target auxiliary heads do not '
        'participate in held-out six-class inference. The source review confirmed the headers and measured '
        'waveform membership. The reported regions are mentions and have no new independent clinical adjudication.',
        '', 'The comparison uses repeatedly inspected PTB development patients. Paired conditionally '
        'stratified patient intervals describe fixed predictions, with no adjustments for refits, shared '
        'training folds, repeated development or multiple comparisons.', '',
    ])
    report = root / 'REPORT.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    _save_json(root / 'summary.json', dict(
        complete=True, study_fingerprint=scope['fingerprint'], restored_models=count, matched_groups=groups,
        n_patients=len(ids), n_records=data['bank']['n_records'], n_beats=data['bank']['n_beats'],
        evaluated_folds=list(range(1, config['outer_splits'] + 1)), seeds=seeds, source_audit=source,
        source_shuffled_patients_with_changed_tag_set_range=[min(changes), max(changes)],
        primary_contrast='true_source_minus_shuffled_source', ensemble_metrics=ensemble_metrics,
        member_metrics=member_metrics, mean_member_metrics=means, contrasts=contrasts,
        report_sha256=file_hash(report), verification_manifest_sha256=verification_hash,
    ))
