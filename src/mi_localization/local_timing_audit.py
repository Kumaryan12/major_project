"""Label-blind feasibility audit of local timing relative to the cached R anchor."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from . import benchmark, regularization
from .neural import _hash, _save_json
from .signal_audit import file_hash


@dataclass
class Fits:
    scores: np.ndarray
    lag_index: np.ndarray
    accepted: np.ndarray


def lag_grid(config):
    maximum, step = config['max_lag_ms'], config['lag_step_ms']
    if type(maximum) is not int or type(step) is not int or not 0 < step < maximum or maximum % step:
        raise ValueError('Require an integer, symmetric lag grid with interior candidates')
    return np.array([0, *(lag for k in range(step, maximum + 1, step) for lag in (-k, k))])


def spatial_unit(x):
    """Center each lead in the fixed window, then normalize XYZ jointly for scoring."""
    centered = x - x.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(centered), axis=(-2, -1)))
    divisor = np.maximum(rms * np.sqrt(centered.shape[-2] * centered.shape[-1]), 1e-15)
    return centered / divisor[..., None, None], rms


def fit_lags(beats, references, bounds, grid, config):
    start, stop = bounds
    if start + min(grid) < 0 or stop + max(grid) > beats.shape[-1] or start >= stop:
        raise ValueError('Lag candidates must have the same full in-crop support')
    templates, ref_rms = spatial_unit(references[:, :, start:stop])
    scores = np.empty((len(references), len(beats), len(grid)))
    rms = np.empty((len(beats), len(grid)))
    for index, lag in enumerate(grid):
        observed, rms[:, index] = spatial_unit(beats[:, :, start + lag:stop + lag])
        scores[:, :, index] = np.einsum('bct,rct->rb', observed, templates)
    scores = np.clip(scores, -1, 1)
    best = scores.argmax(axis=-1)
    values = np.take_along_axis(scores, best[..., None], axis=-1)[..., 0]
    far_apart = abs(grid[best, None] - grid) >= config['peak_separation_ms']
    alternatives = np.where(far_apart, scores, -np.inf).max(axis=-1)
    minimum = config['minimum_centered_rms_mv']
    valid = (np.abs(grid[best]) < np.max(np.abs(grid))) & (values >= config['minimum_score'])
    valid &= (ref_rms[:, None] >= minimum) & (rms[np.arange(len(beats)), best] >= minimum)
    valid &= values - alternatives >= config['minimum_peak_gap']
    return Fits(scores, best, valid)


def cross_gain(current, baseline_indices):
    """Apply A's choice to B's query scores and B's choice to A's."""
    query = np.arange(current.scores.shape[1])
    pair = np.array([1, 0])[:, None]
    return (current.scores[pair, query, current.lag_index]
            - current.scores[pair, query, baseline_indices])


def audit_record(beats, windows, grid, config):
    n = len(beats)
    shape = (n, len(windows))
    output = {
        'accepted_a': np.zeros(shape, dtype=bool),
        'accepted_b': np.zeros(shape, dtype=bool),
        'repeatable': np.zeros(shape, dtype=bool),
        'comparable_to_anchor': np.zeros(shape, dtype=bool),
        'supported_local': np.zeros(shape, dtype=bool),
        'mean_lag_ms': np.zeros(shape, dtype=np.float32),
        'absolute_reference_disagreement_ms': np.zeros(shape, dtype=np.float32),
        'mean_relative_lag_ms': np.zeros(shape, dtype=np.float32),
        'independent_gain_vs_zero': np.zeros(shape, dtype=np.float32),
        'independent_gain_vs_anchor': np.zeros(shape, dtype=np.float32),
    }
    if n < 4:
        return output
    phase = np.arange(n) % 4
    for g in range(4):
        query = np.flatnonzero(phase == g)
        refs = np.stack((np.median(beats[phase == (g + 1) % 4], axis=0),
                         np.median(beats[phase == (g + 2) % 4], axis=0)))
        fits = [fit_lags(beats[query], refs, window, grid, config) for window in windows]
        anchor = fits[0]
        anchor_lags = grid[anchor.lag_index]
        anchor_agrees = anchor.accepted.all(axis=0) & (abs(anchor_lags[0] - anchor_lags[1]) <= config['agreement_ms'])
        for j, current in enumerate(fits):
            lags = grid[current.lag_index]
            disagreement = abs(lags[0] - lags[1])
            repeatable = current.accepted.all(axis=0) & (disagreement <= config['agreement_ms'])
            comparable = repeatable & anchor_agrees
            relative = lags - anchor_lags
            nonzero = (abs(relative) >= config['minimum_local_ms']).all(axis=0)
            same_direction = relative[0] * relative[1] > 0
            gains = cross_gain(current, anchor.lag_index)
            output['accepted_a'][query, j] = current.accepted[0]
            output['accepted_b'][query, j] = current.accepted[1]
            output['repeatable'][query, j] = repeatable
            output['comparable_to_anchor'][query, j] = comparable
            output['supported_local'][query, j] = (
                comparable & nonzero & same_direction & (gains >= config['minimum_cross_reference_gain']).all(axis=0))
            output['mean_lag_ms'][query, j] = lags.mean(axis=0)
            output['absolute_reference_disagreement_ms'][query, j] = disagreement
            output['mean_relative_lag_ms'][query, j] = relative.mean(axis=0)
            output['independent_gain_vs_zero'][query, j] = cross_gain(
                current, np.zeros_like(current.lag_index)).mean(axis=0)
            output['independent_gain_vs_anchor'][query, j] = gains.mean(axis=0)
    return output


def delay_trace(template, delay):
    """Positive delay shifts the template to the right, with endpoint padding."""
    rows = np.clip(np.arange(template.shape[-1]) - delay, 0, template.shape[-1] - 1)
    return template[:, rows]


def record_controls(template, windows, grid, config, rng):
    delays = np.array(config['control_delays_ms'])
    shifted = np.stack([delay_trace(template, shift) for shift in delays])
    clean_controls = np.concatenate((shifted, template[None]), axis=0)
    altered = 1.13 * clean_controls + rng.normal(0., config['control_noise_mv'], clean_controls.shape)
    output = dict(positive_accepted=[], positive_correct=[], estimated_delays=[],
                  null_apparent_shift=[], shape_apparent_shift=[])
    for window in windows:
        start, stop = window
        time = np.arange(template.shape[-1])
        center = start + (stop - start) / 3
        envelope = 1. + config['control_shape_change'] * np.exp(-0.5 * ((time - center) / ((stop - start) / 7))**2)
        deformation = 1.13 * template * envelope + rng.normal(0., config['control_noise_mv'], template.shape)
        test = np.concatenate((altered, deformation[None]), axis=0)
        fit = fit_lags(test, template[None], window, grid, config)
        estimates = grid[fit.lag_index[0]]
        accepted = fit.accepted[0]
        n = len(delays)
        output['positive_accepted'].append(accepted[:n])
        output['positive_correct'].append(accepted[:n] & (abs(estimates[:n] - delays) <= config['lag_step_ms']))
        output['estimated_delays'].append(estimates[:n])
        output['null_apparent_shift'].append(accepted[n] & (abs(estimates[n]) >= config['minimum_local_ms']))
        output['shape_apparent_shift'].append(accepted[n + 1] & (abs(estimates[n + 1]) >= config['minimum_local_ms']))
    return {key: np.asarray(values) for key, values in output.items()}


def median_or_none(values):
    return float(np.median(values)) if len(values) else None


def equal_record_patient_values(values, rows, owners, mask=None):
    """Average record summaries equally within patient. Masked means skip empty records."""
    patient_values = {}
    for indices, owner in zip(rows, owners):
        usable = indices if mask is None else indices[mask[indices]]
        if len(usable):
            patient_values.setdefault(owner, []).append(float(np.mean(values[usable])))
    return np.array([np.mean(records) for records in patient_values.values()])


def aggregate(results, controls, windows, rows, owners, config):
    output = {}
    for j, name in enumerate(windows):
        accepted = controls['positive_accepted'][:, j]
        correct = controls['positive_correct'][:, j]
        predicted = controls['estimated_delays'][:, j]
        anchor_lags = controls['estimated_delays'][:, 0]
        valid_both = accepted & controls['positive_accepted'][:, 0]
        repeat = results['repeatable'][:, j]
        comparable = results['comparable_to_anchor'][:, j]
        support = results['supported_local'][:, j]
        both = results['accepted_a'][:, j] & results['accepted_b'][:, j]
        control_coverage = float(accepted.mean())
        control_recovery = float(correct.sum() / accepted.sum()) if accepted.any() else None
        relative = results['mean_relative_lag_ms'][:, j]
        lags = results['mean_lag_ms'][:, j]
        patient_support = equal_record_patient_values(support, rows, owners)
        prevalence = float(np.mean(patient_support >= config['minimum_patient_support_fraction']))
        passes = (j > 0 and control_coverage >= config['minimum_control_accepted_fraction']
                  and control_recovery is not None
                  and control_recovery >= config['minimum_control_accuracy_accepted']
                  and prevalence >= config['minimum_supported_patient_fraction'])
        output[name] = dict(
            offsets_ms=config['windows'][name],
            beat_count=len(support), both_identifiable_beats=int(both.sum()),
            repeatable_beats=int(repeat.sum()), comparable_to_anchor_beats=int(comparable.sum()),
            supported_local_beats=int(support.sum()),
            median_reference_disagreement_if_both_ms=median_or_none(
                results['absolute_reference_disagreement_ms'][both, j]),
            median_patient_repeatable_fraction=median_or_none(equal_record_patient_values(repeat, rows, owners)),
            median_patient_mean_lag_if_repeatable_ms=median_or_none(equal_record_patient_values(lags, rows, owners, repeat)),
            median_patient_mean_absolute_lag_if_repeatable_ms=median_or_none(
                equal_record_patient_values(abs(lags), rows, owners, repeat)),
            median_patient_mean_relative_lag_if_comparable_ms=median_or_none(
                equal_record_patient_values(relative, rows, owners, comparable)),
            median_patient_mean_absolute_relative_lag_if_comparable_ms=median_or_none(
                equal_record_patient_values(abs(relative), rows, owners, comparable)),
            median_patient_cosine_gain_vs_zero_if_repeatable=median_or_none(
                equal_record_patient_values(results['independent_gain_vs_zero'][:, j], rows, owners, repeat)),
            median_patient_cosine_gain_vs_anchor_if_comparable=median_or_none(
                equal_record_patient_values(results['independent_gain_vs_anchor'][:, j], rows, owners, comparable)),
            median_patient_supported_local_fraction=median_or_none(patient_support),
            supported_patients=int((patient_support >= config['minimum_patient_support_fraction']).sum()),
            patient_support_prevalence=prevalence,
            positive_controls=int(accepted.size), accepted_positive_controls=int(accepted.sum()),
            correctly_recovered_positive_controls=int(correct.sum()),
            positive_control_acceptance=control_coverage, recovery_among_accepted=control_recovery,
            joint_recovery_fraction=float(correct.mean()),
            global_delay_controls_accepted_at_anchor_and_window=int(valid_both.sum()),
            apparent_local_global_controls=int((valid_both & (abs(predicted - anchor_lags) >= config['minimum_local_ms'])).sum()),
            zero_delay_gain_noise_apparent_shifts=int(controls['null_apparent_shift'][:, j].sum()),
            zero_delay_shape_apparent_shifts=int(controls['shape_apparent_shift'][:, j].sum()),
            passes_feasibility_gate=bool(passes),
        )
    return output


def render(summary):
    lines = [
        '# Local timing feasibility in measured XYZ', '',
        f"All {summary['patients']} patients, {summary['recordings']} recordings and {summary['beats']:,} original beats. "
        'Two disjoint same-record reference templates per query group, with all beats in the denominators.', '',
        '| R-relative window (ms) | Repeatable beats | Supported local beats | Patients above support threshold | Control acceptance | Correct among accepted | Gate |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for name, row in summary['windows'].items():
        lo, hi = row['offsets_ms']
        recovery = row['recovery_among_accepted']
        formatted = '—' if recovery is None else f'{recovery:.1%}'
        gate = 'Reference' if name == 'r_anchor' else 'Pass' if row['passes_feasibility_gate'] else 'Below gate'
        lines.append(f"| [{lo}, {hi}) | {row['repeatable_beats']:,} | {row['supported_local_beats']:,} | "
                     f"{row['supported_patients']}/{summary['patients']} | {row['positive_control_acceptance']:.1%} | "
                     f"{formatted} | {gate} |")
    lines.extend(['', '| R-relative window (ms) | Zero-delay gain/noise cases with apparent shift | Zero-delay shape-change cases with apparent shift |',
                  '|---|---:|---:|'])
    for row in summary['windows'].values():
        lo, hi = row['offsets_ms']
        lines.append(f"| [{lo}, {hi}) | {row['zero_delay_gain_noise_apparent_shifts']}/{summary['recordings']} | "
                     f"{row['zero_delay_shape_apparent_shifts']}/{summary['recordings']} |")
    lines.extend([
        '', 'Positive lag means the observed signal occurs later. Supported local shifts require agreement '
        'between disjoint references, departure from the near-R anchor and a cosine improvement over the '
        'near-R-only shift against the other reference in both directions.',
        '', 'Signal morphology and local timing can change the same score. The synthetic controls measure '
        'recovery of known shifts and sensitivity to one fixed shape deformation. Disjoint templates within '
        'a recording can still be correlated. The windows are fixed relative to the cached detector index '
        'and have no clinical annotations. This label-blind feasibility report gives no localization score.',
        '', 'See the [protocol](../../docs/local_timing_audit_plan.md) for the exact thresholds and denominators.', '',
    ])
    return '\n'.join(lines)


def load(config):
    waves, y, groups, source = regularization.load_bank(config)
    paper = yaml.safe_load(Path(config['paper_config']).read_text())
    if (config['sampling_rate'] != 1000 or paper['data']['before_r'] != config['r_index']
            or not Path(config['folds_path']).is_file()):
        raise ValueError('Require original R index, 1000 Hz and frozen folds')
    benchmark.load_or_create_folds(y, groups, config)
    table = pd.read_csv(config['beats_path'])
    table = table[table.label.isin(config['classes'])].reset_index(drop=True)
    records, owners = [], []
    for _, part in table.groupby('record', sort=True):
        part = part.sort_values('r_peak', kind='stable')
        if (part.patient_id.nunique() != 1 or not np.isfinite(part.r_peak).all()
                or np.any(np.diff(part.r_peak) <= 0) or np.any(np.diff(part.beat_index) != 1)):
            raise ValueError('Invalid original recording chronology')
        records.append(part.index.to_numpy())
        owners.append(str(part.patient_id.iloc[0]))
    if ((len(set(owners)), len(records), len(waves)) != (164, 391, 55863)
            or not np.array_equal(np.sort(np.concatenate(records)), np.arange(len(waves)))):
        raise ValueError('Original beat/record/patient coverage changed')
    return waves, records, owners, source


def run(config, verify):
    grid = lag_grid(config)
    window_names = list(config['windows'])
    if not window_names or window_names[0] != 'r_anchor':
        raise ValueError('The near-R reference window must be first')
    windows = [tuple(config['r_index'] + index for index in bounds)
               for bounds in config['windows'].values()]
    waves, records, owners, source = load(config)
    scope = dict(config=config, source=source, folds_sha256=file_hash(config['folds_path']),
                 numpy=np.__version__, code_sha256=file_hash(__file__),
                 bank_loader_sha256=file_hash(regularization.__file__), record_count=len(records))
    scope['fingerprint'] = _hash(scope)
    root = Path(config['output_dir'])
    review = root / 'review'
    if not verify:
        review.mkdir(parents=True, exist_ok=True)
    manifest = root / 'study_manifest.json'
    if manifest.is_file() or verify:
        if json.loads(manifest.read_text()) != scope:
            raise ValueError('Audit provenance changed. Use a separate output namespace.')
    else:
        _save_json(manifest, scope)
    rng = np.random.default_rng(config['seed'])
    results, controls = {}, {}
    for i, rows in enumerate(records, 1):
        beats = np.asarray(waves[rows], dtype=np.float64)
        record = audit_record(beats, windows, grid, config)
        for name, values in record.items():
            if name not in results:
                results[name] = np.empty((len(waves), len(windows)), dtype=values.dtype)
            results[name][rows] = values
        control = record_controls(np.median(beats, axis=0), windows, grid, config, rng)
        for name, values in control.items():
            controls.setdefault(name, []).append(values)
        if i % 100 == 0:
            print(f"{'Replayed' if verify else 'Audited'} {i}/{len(records)} recordings", flush=True)
    controls = {name: np.stack(values) for name, values in controls.items()}
    summaries = aggregate(results, controls, window_names, records, owners, config)
    arrays = dict(results, **controls)
    cache = review / 'diagnostics.npz'
    completed = verify or (root / 'summary.json').exists()
    if completed:
        with np.load(cache, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(x, saved[key]) for key, x in arrays.items()):
                raise ValueError('Independent timing arrays differ from saved diagnostics')
    else:
        temp = cache.with_suffix('.partial.npz')
        np.savez_compressed(temp, **arrays)
        temp.replace(cache)
    summary = dict(
        complete=True, study_fingerprint=scope['fingerprint'], patients=len(set(owners)),
        recordings=len(records), beats=len(waves), cohort_fits=0, new_r_positions=0,
        shape_only_controls_per_window=len(records), diagnostic_arrays_sha256=file_hash(cache),
        windows=summaries, passed_windows=[name for name, row in summaries.items() if row['passes_feasibility_gate']])
    report = root / 'REPORT.md'
    report_text = render(summary)
    if not completed:
        report.write_text(report_text)
    summary['report_sha256'] = file_hash(report)
    if completed:
        if summary != json.loads((root / 'summary.json').read_text()) or report_text != report.read_text():
            raise ValueError('Published timing aggregate changed')
    else:
        _save_json(root / 'summary.json', summary)
    if verify:
        _save_json(root / 'verification.json', dict(
            verified=True, study_fingerprint=scope['fingerprint'], replayed_all_beats=len(waves),
            replayed_all_recordings=len(records), replayed_arrays=len(arrays),
            summary_sha256=file_hash(root / 'summary.json'), report_sha256=summary['report_sha256'],
            diagnostic_arrays_sha256=summary['diagnostic_arrays_sha256']))
    print(json.dumps({'records': len(records), 'passed_windows': summary['passed_windows'], 'verified': verify}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local_timing_audit_v1.yaml')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    with threadpool_limits(limits=1):
        run(config, args.verify)


if __name__ == '__main__':
    main()
