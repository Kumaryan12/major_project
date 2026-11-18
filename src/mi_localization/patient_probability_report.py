"""Report the matched patient probability loss comparison after full checkpoint replay."""
import numpy as np

from .benchmark import aggregate_patients, metric_bundle
from .neural import _save_json
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash


def summarize(root, scope, data, predictions, evidence):
    config = scope['config']
    classes, seeds = config['classes'], config['training_seeds']
    arms = list(predictions)
    paired = {'member_metrics': {}, 'ensemble_metrics': {}, 'mean_member_metrics': {}}
    patient_ids, truth, _ = aggregate_patients(
        np.zeros((len(data['labels']), 1)), data['labels'], data['groups'])
    codes = np.arange(len(classes))
    weights = patient_weights(truth, codes, config['bootstrap_draws'], config['seed'])
    draws = {}
    for arm, runs in predictions.items():
        pooled = []
        for seed in seeds:
            p = runs[seed]
            if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
                raise ValueError('Incomplete held-out beat coverage')
            ids, labels, probabilities = aggregate_patients(p, data['labels'], data['groups'])
            if not np.array_equal(ids, patient_ids) or not np.array_equal(labels, truth):
                raise ValueError('Patient pooling differs across arms')
            pooled.append(probabilities)
        members = {str(s): metric_bundle(truth, p, classes) for s, p in zip(seeds, pooled)}
        member_f1 = [m['macro_f1'] for m in members.values()]
        ensemble = np.mean(pooled, axis=0)
        paired['member_metrics'][arm] = members
        paired['mean_member_metrics'][arm] = dict(mean=float(np.mean(member_f1)), seed_sd=float(np.std(member_f1, ddof=1)))
        paired['ensemble_metrics'][arm] = metric_bundle(truth, ensemble, classes)
        draws[arm] = weighted_f1(truth, ensemble.argmax(1), codes, weights)

    contrasts = {}
    for arm in ('patient_ce', 'hybrid_ce'):
        diff = paired['ensemble_metrics'][arm]['macro_f1'] - paired['ensemble_metrics']['beat_ce']['macro_f1']
        contrasts[f'{arm}_minus_beat_ce'] = dict(
            ensemble_delta=diff, paired_patient_interval=np.quantile(draws[arm] - draws['beat_ce'], [.025, .975]).tolist())
    lines = ['# Beat and patient probability loss on measured XYZ', '',
             f"Restored all {evidence['restored_models']} models and audited {evidence['matched_fold_seed_groups']} "
             f"matched fold and seed groups. Test inference averages every original beat probability per patient "
             f"for all {len(patient_ids)} patients.", '',
             '| Objective | Mean member macro F1 ± seed SD | Three-seed ensemble macro F1 |', '|---|---:|---:|']
    for arm in arms:
        mean = paired['mean_member_metrics'][arm]
        f1 = paired['ensemble_metrics'][arm]['macro_f1']
        lines.append(f"| {arm} | {mean['mean']:.6f} ± {mean['seed_sd']:.6f} | {f1:.6f} |")
    lines.extend(['', 'Primary comparison is patient-level cross-entropy versus per-beat cross-entropy.', '',
                  '| Comparison | Ensemble difference | Descriptive paired patient 95% interval |', '|---|---:|---|'])
    for name, row in contrasts.items():
        lo, hi = row['paired_patient_interval']
        lines.append(f"| {name} | {row['ensemble_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(['', '| Class | ' + ' | '.join(arms) + ' |', '|---|' + '---:|' * len(arms)])
    for name in classes:
        scores = [paired['ensemble_metrics'][a]['classification_report'][name]['f1-score'] for a in arms]
        lines.append(f"| {name} | " + ' | '.join(f'{score:.6f}' for score in scores) + ' |')
    lines.extend([
        '', f"Historical development references are {scope['waveform_reference_macro_f1']:.6f} from the waveform "
        f"pilot and {scope['convolution_reference_macro_f1']:.6f} from selected linear convolution. "
        'The matched comparison uses grouped batches for every arm. The historical waveform pilot used individual beat draws.',
        '', 'All arms used the same signed XYZ waveforms, class- and patient-balanced eight-beat bags, training '
        'normalization, model initialization, optimizer, eight epochs, and all-beat inference. No epoch or '
        'objective weights were selected from these outer folds.',
        '', 'These PTB patients have been used repeatedly for development. The paired bootstrap describes uncertainty '
        'conditional on the saved predictions and does not account for refitting or repeated development.',
        '', '[Protocol](../../../docs/patient_probability_loss_plan.md)', '',
    ])
    report = root / 'REPORT.md'
    temporary = report.with_suffix('.partial.md')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(report)
    _save_json(root / 'summary.json', dict(
        **evidence, **paired, contrasts=contrasts, primary_contrast='patient_ce_minus_beat_ce',
        n_patients=len(patient_ids), n_records=scope['bank']['n_records'], n_beats=len(data['labels']),
        evaluated_folds=list(range(1, config['outer_splits'] + 1)), seeds=seeds,
        waveform_reference_macro_f1=scope['waveform_reference_macro_f1'],
        convolution_reference_macro_f1=scope['convolution_reference_macro_f1'], report_sha256=file_hash(report),
    ))
