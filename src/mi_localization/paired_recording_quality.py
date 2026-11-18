"""Audit signal quality and timing for the frozen matched recording comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
import wfdb
import yaml

from . import index_recording_audit as index
from . import matched_recording_beats as matched
from . import model_error_comparison as replay
from .blinded_recording_review import evidence
from .multiview_bank import array_hash
from .signal import denoise_vcg, detect_r_peaks, extract_beats
from .signal_audit import file_hash, signal_metrics


PROTOCOL = Path('docs/paired_recording_quality_plan.md')
SOURCE = Path('results/matched_recording_beats_v1/summary.json')
SOURCE_HASH = '64f09179ab7060a238d6f16c271e3f1578ad7c45827bdadd8b2b2232b7f777dc'
OUTPUT = Path('results/paired_recording_quality_v1')
MEASURES = ('drift_mv', 'adc_rail_fraction', 'r_disagreement_fraction',
            'relative_denoising_change', 'unusual_rr_fraction')
PRIMARY = ('drift_mv', 'r_disagreement_fraction')


def raw_peak_disagreement(cached, detected, tolerance_samples):
    if len(detected) == 0:
        return np.ones(len(cached), dtype=bool)
    right = np.searchsorted(detected, cached)
    past = detected[np.maximum(right - 1, 0)]
    future = detected[np.minimum(right, len(detected) - 1)]
    distance = np.minimum(np.abs(cached - past), np.abs(cached - future))
    return distance > tolerance_samples


def sampled_means(features, samples, rows):
    local = pd.Index(rows).get_indexer(samples.ravel()).reshape(samples.shape)
    if np.any(local < 0):
        raise ValueError('Sampled beat is outside the declared recording')
    return np.nanmean(features[local], axis=1).mean(axis=0)


def recording_features(raw_root, record, rows, beats, paper, cached):
    data, signal = paper['data'], paper['signal']
    path = raw_root / record
    decoded = wfdb.rdrecord(str(path), channel_names=data['leads'], physical=False)
    digital = decoded.d_signal
    if (decoded.fs != data['sampling_rate'] or decoded.sig_name != data['leads']
            or decoded.fmt != ['16'] * 3 or decoded.units != ['mV'] * 3
            or digital.ndim != 2 or digital.shape[1] != 3
            or np.any(np.asarray(decoded.adc_gain) <= 0)
            or np.any(digital == -32768)):
        raise ValueError('Expected usable signed 16-bit measured XYZ in mV')
    raw = (digital - np.asarray(decoded.baseline)) / np.asarray(decoded.adc_gain)
    peaks = beats.loc[rows, 'r_peak'].to_numpy(dtype=int)
    if np.any(np.diff(peaks) <= 0):
        raise ValueError('Expected ordered, distinct cached R positions')
    before, after = data['before_r'], data['after_r']
    raw_beats, used = extract_beats(raw, peaks, before, after)
    clean = denoise_vcg(raw, signal['denoise_wavelet'], signal['denoise_level'])
    clean_beats, _ = extract_beats(clean, peaks, before, after)
    if not np.array_equal(used, peaks) or not np.array_equal(clean_beats, cached[rows]):
        raise ValueError('Physical units or cached denoised beat reconstruction changed')

    drift = sosfiltfilt(butter(4, .5, fs=decoded.fs, output='sos'), raw, axis=0)
    drift_beats, _ = extract_beats(drift, peaks, before, after)
    drift_mv = np.linalg.norm(np.ptp(drift_beats.astype(float), axis=-1), axis=1)

    # WFDB reserves -32768 as invalid data for format 16. Both usable endpoints are explicit.
    rail_samples = np.any((digital == 32767) | (digital == -32767), axis=1)
    rail_beats, _ = extract_beats(rail_samples[:, None], peaks, before, after)
    rail_hits = rail_beats.any(axis=(1, 2))

    detected = detect_r_peaks(raw, decoded.fs,
                              refractory_ms=signal['rpeak_refractory_ms'],
                              integration_ms=signal['rpeak_integration_ms'],
                              threshold_scale=signal['rpeak_threshold_scale'])
    disagreement = raw_peak_disagreement(peaks, detected, round(.04 * decoded.fs))
    denoising_change = signal_metrics(raw_beats, clean_beats)['centered_relative_l2_change']
    rr = np.diff(peaks) * 1000 / decoded.fs
    unusual_rr = np.r_[np.nan, (rr < 300) | (rr > 2000)]
    features = np.column_stack([drift_mv, rail_hits, disagreement, denoising_change, unusual_rr])
    return features, bool(rail_samples.any()), file_hash(path.with_suffix('.xyz'))


def stratified_indices(labels, draws, seed):
    rng = np.random.default_rng(seed)
    partitions = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return np.concatenate([rng.choice(group, size=(draws, len(group))) for group in partitions], axis=1)


def rank_correlation(left, right):
    x = rankdata(left, axis=-1)
    y = rankdata(right, axis=-1)
    x -= x.mean(axis=-1, keepdims=True)
    y -= y.mean(axis=-1, keepdims=True)
    denom = np.sqrt(np.square(x).sum(axis=-1) * np.square(y).sum(axis=-1))
    return np.divide((x * y).sum(axis=-1), denom, out=np.full(denom.shape, np.nan), where=denom != 0)


def association(x, outcome, truth, draws, seed):
    valid = np.isfinite(x) & np.isfinite(outcome)
    x, outcome, truth = x[valid], outcome[valid], truth[valid]
    if len(x) < 2:
        return dict(patients=len(x), rho=None, interval=None, finite_draws=0)
    point = rank_correlation(x, outcome)
    sample = stratified_indices(truth, draws, seed)
    bootstrap = rank_correlation(x[sample], outcome[sample])
    finite = bootstrap[np.isfinite(bootstrap)]
    return dict(patients=len(x), rho=float(point) if np.isfinite(point) else None,
                interval=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                finite_draws=len(finite))


def summarize(quality, outcomes, truth, classes, timing, rails, draws, seed):
    sample = stratified_indices(truth, draws, seed)
    changes, associations, companion = {}, {}, {}
    correct_change = outcomes[:, 1, 0] - outcomes[:, 0, 0]
    probability_change = outcomes[:, 1, 1] - outcomes[:, 0, 1]
    for i, metric in enumerate(MEASURES):
        early, later = quality[:, :, i].T
        delta = later - early
        changes[metric] = dict(
            earliest_median=float(np.median(early)), later_median=float(np.median(later)),
            later_minus_earliest_mean=float(delta.mean()),
            interval=np.quantile(delta[sample].mean(axis=1), [.025, .975]).tolist(),
            higher=int(np.count_nonzero(delta > 0)), lower=int(np.count_nonzero(delta < 0)),
            equal=int(np.count_nonzero(delta == 0)))
        associations[f'decrease_{metric}'] = association(-delta, correct_change, truth, draws, seed)
        if metric in PRIMARY:
            companion[f'decrease_{metric}'] = association(-delta, probability_change, truth, draws, seed)
    associations.update({metric: association(values, correct_change, truth, draws, seed)
                         for metric, values in timing.items()})
    return dict(
        changes=changes, associations=associations, companion=companion,
        class_results={label: dict(patients=int(np.count_nonzero(truth == i)),
                                  later_minus_earliest_accuracy=float(correct_change[truth == i].mean()))
                       for i, label in enumerate(classes)},
        later_minus_earliest_accuracy=float(correct_change.mean()),
        records_with_rail_samples=dict(earliest=int(rails[:, 0].sum()), later=int(rails[:, 1].sum())),
    )


def paired_outcomes(ensemble, pair, early, later, label):
    # Combine predictions within a draw before measuring correctness.
    decisions = matched.decisions(ensemble, pair, early, later)
    accuracy = (decisions[:2] == label).mean(axis=1)
    mean_probability = [float(ensemble[chosen, label].mean()) for chosen in (early, later)]
    return decisions, np.column_stack([accuracy, mean_probability])


def render(result):
    lines = [
        '# Paired measured-XYZ signal quality and timing audit', '',
        f"Replayed all {result['verified_prediction_blocks']} saved prediction blocks, "
        f"{result['patients']} patients and {result['draws']} matched beat draws. "
        f"Raw decoding reproduced cached denoised beats in all {2 * result['patients']} paired records. "
        'Zero new fits.', '',
        '| Measure | Earliest median | Later median | Mean later − earliest [95% interval] | Later ↑ / ↓ / equal |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, row in result['changes'].items():
        low, high = row['interval']
        lines.append(f"| {name} | {row['earliest_median']:.5f} | {row['later_median']:.5f} "
                     f"| {row['later_minus_earliest_mean']:+.5f} [{low:+.5f}, {high:+.5f}] "
                     f"| {row['higher']} / {row['lower']} / {row['equal']} |")
    lines += ['', f"Records with at least one usable ADC rail sample. Earliest "
              f"{result['records_with_rail_samples']['earliest']} of {result['patients']}, later "
              f"{result['records_with_rail_samples']['later']} of {result['patients']}.", '',
              '| Association with later-minus-earliest outcome | Patients | Spearman rho | 95% interval |',
              '|---|---:|---:|---:|']
    for outcome, rows in (('correctness', result['associations']), ('correct-class probability', result['companion'])):
        for name, item in rows.items():
            interval = item['interval']
            formatted_interval = f'[{interval[0]:+.3f}, {interval[1]:+.3f}]' if interval else 'Unavailable'
            rho = f"{item['rho']:+.3f}" if item['rho'] is not None else 'Unavailable'
            lines.append(f"| {name} / {outcome} | {item['patients']} | {rho} | {formatted_interval} |")
    lines += ['', '| Class | Patients | Later − earliest accuracy |', '|---|---:|---:|']
    for name, row in result['class_results'].items():
        lines.append(f"| {name} | {row['patients']} | {row['later_minus_earliest_accuracy']:+.5f} |")
    lines += [
        '', f"Overall later-minus-earliest accuracy was {result['later_minus_earliest_accuracy']:+.5f}. "
        f"The fixed paired macro F1 comparison reproduced {result['earliest_macro_f1']:.6f} earliest "
        f"and {result['later_macro_f1']:.6f} later.",
        '', 'The two primary associations are decreases in baseline drift and raw-detector disagreement '
        'with the change in held-out correctness. ADC rail hits detect exact digital extrema. Cached R '
        'disagreements and RR outliers are descriptive proxies. PTB headers give archived date fields, and '
        'marked infarct-date coverage is reported in the association table. Class-stratified patient intervals '
        'condition on the previously inspected cohort and frozen predictions.',
        '', 'Clinical review remains pending. [Protocol](../../docs/paired_recording_quality_plan.md)', '',
    ]
    return '\n'.join(lines)


def run(output, verify):
    source = replay.checked_json(SOURCE, SOURCE_HASH)
    prior = replay.checked_json(matched.PRIOR, matched.PRIOR_HASH)
    cfg = yaml.safe_load(index.SOURCE_CONFIG.read_text())
    paper_path = Path('configs/paper.yaml')
    paper = yaml.safe_load(paper_path.read_text())
    fold_file, beat_file = Path(cfg['folds_path']), Path(cfg['beats_path'])
    if (file_hash(fold_file) != prior['provenance']['folds_sha256']
            or file_hash(beat_file) != prior['provenance']['beats_sha256']
            or file_hash(Path(matched.__file__)) != source['provenance']['code_sha256']):
        raise ValueError('Expected the pinned folds, beat table and matched sampler')
    patients = pd.read_csv(fold_file).sort_values('patient_id')
    ids = patients.patient_id.to_numpy(dtype=str)
    classes = cfg['classes']
    labels = pd.Index(classes).get_indexer(patients.label)
    beats = pd.read_csv(beat_file)
    beats = beats[beats.label.isin(classes)].reset_index(drop=True)
    raw_root = Path(paper['data']['raw_dir'])
    pairs, header_hash = matched.find_pairs(beats, raw_root)
    membership = array_hash(np.asarray([(p.patient, p.earliest, p.later) for p in pairs]))
    if (membership != source['provenance']['paired_membership_sha256']
            or header_hash != source['provenance']['headers_sha256']):
        raise ValueError('Original paired archive membership changed')
    root = Path(cfg['waveform_root'])
    _, scope, verification = replay.read_source(root, cfg['waveform_summary_sha256'], 'fingerprint')
    if file_hash(paper_path) != scope['bank']['paper_sha256']:
        raise ValueError('Original signal preparation configuration changed')
    saved, artifacts = index.load_predictions(root, scope, verification, ids, labels, patients.fold.to_numpy(), beats)
    artifact_hash = array_hash(np.asarray(sorted((name, *hashes) for name, hashes in artifacts.items())))
    if artifact_hash != source['provenance']['verified_artifacts_sha256']:
        raise ValueError('Original held-out prediction artifacts changed')
    ensemble = np.mean(list(saved.values()), axis=0)
    truth = labels[pd.Index(ids).get_indexer([pair.patient for pair in pairs])]
    cache_path = Path(scope['config']['bank_dir']) / 'waveforms.npy'
    cached = np.load(cache_path, mmap_mode='r')
    if cached.shape != (len(beats), 3, paper['data']['before_r'] + paper['data']['after_r'] + 1):
        raise ValueError('Denoised waveform cache shape changed')

    rng = np.random.default_rng(matched.SAMPLE_SEED)
    quality, outcomes, calls, rails, time_from_infarct, raw_hashes = [], [], [], [], [], []
    for pair, label in zip(pairs, truth):
        early, later = matched.sample_indices(pair, rng)
        pair_features, pair_rails, infarct_offsets = [], [], []
        for record, rows, draw, day in (
                (pair.earliest, pair.early_rows, early, pair.earliest_day),
                (pair.later, pair.later_rows, later, pair.later_day)):
            features, hit, digest = recording_features(raw_root, record, rows, beats, paper, cached)
            pair_features.append(sampled_means(features, draw, rows))
            pair_rails.append(hit)
            infarct_offsets.append(evidence(raw_root, record, day)[0])
            raw_hashes.append((record, digest))
        a, b = infarct_offsets
        time_from_infarct.append(a if a is not None and b is not None and a >= 0 and b - a == pair.days else np.nan)
        paired_calls, outcome = paired_outcomes(ensemble, pair, early, later, label)
        calls.append(paired_calls)
        quality.append(pair_features)
        outcomes.append(outcome)
        rails.append(pair_rails)
    draw_count, seed = cfg['bootstrap_draws'], cfg['bootstrap_seed']
    previous, contrasts = matched.summarize(truth, np.stack(calls, axis=1), classes, draw_count, seed)
    if previous != source['arms'] or contrasts != source['contrasts']:
        raise ValueError('Matched beat result did not reproduce')
    timing = dict(days_between_recordings=np.asarray([pair.days for pair in pairs]),
                  days_from_documented_infarct_to_earliest=np.asarray(time_from_infarct))
    result = dict(complete=True, diagnostic_only=True, model_fits=0, patients=len(pairs),
                  verified_prediction_blocks=len(artifacts), draws=matched.DRAWS, seeds=source['seeds'],
                  folds=source['folds'], earliest_macro_f1=previous['earliest']['macro_f1'],
                  later_macro_f1=previous['later']['macro_f1'],
                  provenance=dict(matched_summary_sha256=SOURCE_HASH, verified_artifacts_sha256=artifact_hash,
                                  raw_records_sha256=array_hash(np.asarray(sorted(raw_hashes))),
                                  code_sha256=file_hash(Path(__file__)), protocol_sha256=file_hash(PROTOCOL)),
                  **summarize(np.asarray(quality), np.asarray(outcomes), truth, classes, timing,
                              np.asarray(rails), draw_count, seed))
    payloads = {'summary.json': json.dumps(result, indent=2, allow_nan=False) + '\n', 'REPORT.md': render(result)}
    if verify:
        for name, payload in payloads.items():
            if (output / name).read_text() != payload:
                raise ValueError(f'{name} did not reproduce')
    else:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temp:
            stage = Path(temp) / 'audit'
            stage.mkdir()
            for name, payload in payloads.items():
                (stage / name).write_text(payload)
            stage.rename(output)
    print(payloads['REPORT.md'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)


if __name__ == '__main__':
    main()
