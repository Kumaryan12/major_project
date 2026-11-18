"""Aggregate independently restored XYZ-only predictions at the patient level."""
from pathlib import Path

import numpy as np

from .benchmark import aggregate_patients, metric_bundle
from .neural import _save_json
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash


def summarize(root, scope, inputs, predictions, arms, count, verification_hash):
    config = scope['config']
    seeds = config['training_seeds']
    classes = config['classes']
    members = {arm: [] for arm in arms}
    metrics = {arm: {} for arm in arms}
    ids = np.unique(inputs['groups'])
    truth = None
    if count != config['outer_splits'] * len(seeds) * len(arms):
        raise ValueError('Only report complete patient folds and seeds')
    for seed, by_arm in predictions.items():
        for arm, probabilities in by_arm.items():
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
                raise ValueError('Missing or invalid beat predictions')
            patient_ids, labels, pooled = aggregate_patients(probabilities, inputs['labels'], inputs['groups'])
            if not np.array_equal(ids, patient_ids) or (truth is not None and not np.array_equal(labels, truth)):
                raise ValueError('Held-out patient alignment differs across arms')
            truth = labels
            members[arm].append(pooled)
            metrics[arm][str(seed)] = metric_bundle(truth, pooled, classes)
    p = {arm: np.mean(members[arm], axis=0) for arm in arms}
    ensembles = {arm: metric_bundle(truth, p[arm], classes) for arm in arms}
    means = {}
    for arm in arms:
        values = [metrics[arm][str(seed)]['macro_f1'] for seed in seeds]
        means[arm] = dict(mean=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1)))
    labels = np.arange(len(classes))
    weights = patient_weights(truth, labels, config['bootstrap_draws'], config['seed'])
    draws = {arm: weighted_f1(truth, p[arm].argmax(axis=1), labels, weights) for arm in arms}
    contrasts = {}
    for left, right in ((arms[2], arms[1]), (arms[2], arms[0]), (arms[1], arms[0])):
        contrasts[f'{left}_minus_{right}'] = dict(
            ensemble_delta=ensembles[left]['macro_f1'] - ensembles[right]['macro_f1'],
            mean_member_delta=means[left]['mean'] - means[right]['mean'],
            descriptive_patient_interval=np.quantile(draws[left] - draws[right], [.025, .975]).tolist())
    title = ('Native measured XYZ pretraining' if scope['mode'] == 'native'
             else 'Paired ECG-assisted training with XYZ-only inference')
    lines = [
        f'# {title}', '',
        f"All {len(ids)} patients, {inputs['bank']['n_records']} recordings and "
        f"{inputs['bank']['n_beats']:,} original beats. "
        f"Replayed all {count} models and {count // len(arms)} matched fold/seed groups.", '',
        '| Arm | Mean member macro F1 ± seed SD | Equal-seed ensemble macro F1 |',
        '|---|---:|---:|',
    ]
    for arm in arms:
        lines.append(f"| {arm} | {means[arm]['mean']:.6f} ± {means[arm]['seed_sd']:.6f} | "
                     f"{ensembles[arm]['macro_f1']:.6f} |")
    primary = f'{arms[2]}_minus_{arms[1]}'
    lines.extend([
        '', f'The primary contrast is {arms[2]} versus {arms[1]}.', '',
        '| Comparison | Ensemble difference | Paired patient 95% interval |',
        '|---|---:|---:|',
    ])
    for name, comparison in contrasts.items():
        lo, hi = comparison['descriptive_patient_interval']
        lines.append(f"| {name} | {comparison['ensemble_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(['', '| Class | ' + ' | '.join(arms) + ' |', '|---|' + '---:|' * len(arms)])
    for name in classes:
        values = [ensembles[arm]['classification_report'][name]['f1-score'] for arm in arms]
        lines.append(f'| {name} | ' + ' | '.join(f'{x:.6f}' for x in values) + ' |')
    lines.extend([
        '', f"Target fitting used a fixed {config['target_epochs']}-epoch budget, with no validation "
        'monitoring or target hyperparameter selection. All three declared seeds and ten patient folds '
        'contribute to each score. The scratch control uses the same classification schedule. '
        'Pretrained arms add the same number of pretraining steps. Exact-partition scaling preserves '
        'signed physical XYZ orientation. Independent replay restores the encoder and final model '
        'checkpoints, checks sample plans and normalizers, and reproduces held-out probabilities.',
        '', 'These results concern the inspected PTB development cohort. Conditional bootstrap '
        'intervals use class-stratified patient resampling of fixed predictions. They do not include '
        'refitting or adjust for shared training folds and repeated development.', '',
    ])
    if scope['mode'] == 'native':
        lines.append('The additional patients supply measured XYZ waveforms without using their diagnostic '
                     'labels. The native source audit documents recording checks and their scope.')
    else:
        lines.append('Only training patients contribute measured ECG targets. Within each matched batch, '
                     'the shuffled arm uses the identical set of ECG descriptors with cross-patient '
                     'pairings. Both arms use measured XYZ alone for classification and held-out inference.')
    report = Path(root) / 'REPORT.md'
    partial = report.with_suffix('.partial.md')
    partial.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    partial.replace(report)
    _save_json(root / 'summary.json', dict(
        complete=True, study_fingerprint=scope['fingerprint'], mode=scope['mode'],
        n_patients=len(ids), n_records=inputs['bank']['n_records'], n_beats=inputs['bank']['n_beats'],
        seeds=seeds, evaluated_folds=list(range(1, config['outer_splits'] + 1)),
        restored_models=count, matched_groups=count // len(arms),
        primary_contrast=primary, ensemble_metrics=ensembles, member_metrics=metrics,
        mean_member_metrics=means, contrasts=contrasts, report_sha256=file_hash(report),
        verification_manifest_sha256=verification_hash))
