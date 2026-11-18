"""Compare recording-level inference policies using frozen held-out ResNet predictions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from . import benchmark, model_error_comparison as replay
from .multiview_bank import array_hash
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash


DATE_FIELD = re.compile(r'^\s*#\s*ECG date\s*:\s*(.*?)\s*$', re.IGNORECASE)
DATE_FORMAT = re.compile(r'(\d{2})/(\d{2})/(\d{4})')
ARMS = ('all_beats', 'equal_records', 'one_record')
OUTPUT = Path('results/index_recording_audit_v1')
PROTOCOL = Path('docs/index_recording_audit_plan.md')
SOURCE_CONFIG = Path('configs/model_error_comparison_v1.yaml')


def recording_date(header):
    fields = [match[1] for line in header.splitlines() if (match := DATE_FIELD.fullmatch(line))]
    match = DATE_FORMAT.fullmatch(fields[0]) if len(fields) == 1 else None
    if match is None:
        return None
    day, month, year = map(int, match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def select_record(dates):
    if any(value is None for value in dates.values()):
        return min(dates), 'missing_or_invalid_date'
    first_day = min(dates.values())
    first_records = sorted(record for record, value in dates.items() if value == first_day)
    status = 'eligible' if len(first_records) == 1 else 'earliest_date_tie'
    return first_records[0], status


def select_records(beats, ids, raw_root):
    by_patient = {}
    header_hashes = {}
    dated_count = 0
    for patient, rows in beats.groupby('patient_id'):
        dates = {}
        for record in sorted(rows.record.unique()):
            path = raw_root / f'{record}.hea'
            raw = path.read_bytes()
            header_hashes[record] = file_hash(path)
            dates[record] = recording_date(raw.decode('latin1'))
            dated_count += int(dates[record] is not None)
        selected, status = select_record(dates)
        by_patient[patient] = (selected, status, len(dates))
    ordered = [by_patient[patient] for patient in ids]
    choices, reasons, sizes = (np.asarray(column) for column in zip(*ordered))
    return choices, reasons, sizes, dated_count, header_hashes


def load_predictions(root, scope, verification, ids, labels, folds, beats):
    config = scope['config']
    beat_patient_index = pd.Index(ids).get_indexer(beats.patient_id)
    beat_labels = pd.Index(config['classes']).get_indexer(beats.label)
    if (np.any(beat_patient_index < 0) or not np.array_equal(beat_labels, labels[beat_patient_index])
            or np.unique(beat_patient_index).size != len(ids)):
        raise ValueError('Original beat rows do not match frozen patient labels')
    beat_folds = folds[beat_patient_index]
    hashes = verification['fit_audit_hashes']
    results = {}
    verified_artifacts = {}
    for seed in config['training_seeds']:
        probabilities = np.full((len(beats), len(config['classes'])), np.nan)
        for fold in range(1, config['outer_splits'] + 1):
            train = np.flatnonzero(beat_folds != fold)
            query = np.flatnonzero(beat_folds == fold)
            name = f'fold{fold:02d}_seed{seed}_scratch'
            folder = root / 'review' / name
            audit = replay.checked_json(folder / 'audit.json', hashes[name])
            replay.require_fields(audit['context'], dict(
                fingerprint=scope['fingerprint'], arm='scratch', fold=fold, declared_seed=seed,
                train_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query),
                train_patient_count=int((folds != fold).sum()), query_patient_count=int((folds == fold).sum()),
                source_rows_sha256=None))
            if audit['pretrain'] is not None:
                raise ValueError('Expected independently verified scratch')
            digest = audit['files']['predictions.npz']
            probabilities[query] = replay.checked_probabilities(
                folder / 'predictions.npz', digest, query, len(config['classes']))
            verified_artifacts[name] = [hashes[name], digest]
        if not np.isfinite(probabilities).all():
            raise ValueError('Incomplete held-out beat coverage')
        results[seed] = probabilities
    return results, verified_artifacts


def pool_predictions(probabilities, beats, ids, labels, choices):
    beat_labels = labels[pd.Index(ids).get_indexer(beats.patient_id)]
    patient_ids, _, all_beats = benchmark.aggregate_patients(probabilities, beat_labels, beats.patient_id)
    if not np.array_equal(patient_ids, ids):
        raise ValueError('All-beat patient membership changed')
    records = pd.DataFrame(probabilities).groupby([beats.patient_id, beats.record]).mean()
    equal_records = records.groupby(level='patient_id').mean().loc[ids].to_numpy()
    chosen_records = records.loc[list(zip(ids, choices))].to_numpy()
    return dict(all_beats=all_beats, equal_records=equal_records, one_record=chosen_records)


def evaluate(labels, pooled, classes, cohorts, draws, seed):
    seeds = list(pooled)
    rows = {}
    for cohort, (mask, arms) in cohorts.items():
        truth = labels[mask]
        class_indices = np.arange(len(classes))
        weights = patient_weights(truth, class_indices, draws, seed)
        metrics = {}
        samples = {}
        decisions = {}
        for arm in arms:
            members = {str(s): pooled[s][arm][mask] for s in seeds}
            ensemble = np.mean(list(members.values()), axis=0)
            metrics[arm] = dict(
                ensemble=benchmark.metric_bundle(truth, ensemble, classes),
                members={s: benchmark.metric_bundle(truth, p, classes) for s, p in members.items()})
            decisions[arm] = ensemble.argmax(1)
            samples[arm] = weighted_f1(truth, decisions[arm], class_indices, weights)
        baseline = metrics['all_beats']['ensemble']['macro_f1']
        contrasts = {arm: dict(
            delta=metrics[arm]['ensemble']['macro_f1'] - baseline,
            interval=np.quantile(samples[arm] - samples['all_beats'], [.025, .975]).tolist(),
            changed_top_label=int(np.count_nonzero(decisions[arm] != decisions['all_beats'])),
        ) for arm in arms if arm != 'all_beats'}
        rows[cohort] = dict(patients=int(mask.sum()), metrics=metrics, contrasts=contrasts)
    return rows


def render(result):
    lines = [
        '# Earliest-recording sensitivity for the frozen measured-XYZ ResNet', '',
        f"Replayed {result['verified_prediction_blocks']} held-out prediction blocks. The "
        f"{result['n_patients']} patients, {result['n_records']} recordings and {result['n_beats']:,} "
        f"beats retain their original labels and folds. All {result['n_valid_dates']} usable "
        'dates come from PTB ECG headers. Zero new fits.', '',
        '| Date eligibility | All patients | Patients with multiple recordings |',
        '|---|---:|---:|',
    ]
    for reason, counts in result['eligibility'].items():
        lines.append(f"| {reason} | {counts['all']} | {counts['multiple_records']} |")
    lines += [
        '', '| Cohort and inference | Patient macro F1 | Accuracy | Member F1 in seed order |',
        '|---|---:|---:|---|',
    ]
    for cohort, row in result['cohorts'].items():
        for arm, metric in row['metrics'].items():
            member = ', '.join(f"{m['macro_f1']:.6f}" for m in metric['members'].values())
            score = metric['ensemble']
            lines.append(f"| {cohort} ({row['patients']}) / {arm} | {score['macro_f1']:.6f} "
                         f"| {score['accuracy']:.6f} | {member} |")
    lines += ['', '| Paired contrast versus all beats | Macro F1 difference | 95% interval | Top label changed |',
              '|---|---:|---|---:|']
    for cohort, row in result['cohorts'].items():
        for arm, metric in row['contrasts'].items():
            low, high = metric['interval']
            lines.append(f"| {cohort} / {arm} | {metric['delta']:+.6f} | "
                         f"[{low:+.6f}, {high:+.6f}] | {metric['changed_top_label']} |")
    lines += ['', '## Primary cohort class performance', '',
              '| Class | Eligible / original support | All beats P / R / F1 | Earliest record P / R / F1 |',
              '|---|---:|---|---|']
    primary = result['cohorts']['dated_unique_earliest']['metrics']
    for label in result['classes']:
        baseline, earliest = (primary[arm]['ensemble']['classification_report'][label]
                              for arm in ('all_beats', 'one_record'))
        scores = [' / '.join(f'{row[k]:.3f}' for k in ('precision', 'recall', 'f1-score'))
                  for row in (baseline, earliest)]
        lines.append(f"| {label} | {int(baseline['support'])} / {result['class_support'][label]} "
                     f"| {scores[0]} | {scores[1]} |")
    lines += [
        '', 'The primary contrast is one earliest dated record minus all beats on the same eligible patients. '
        'The multiple-record subset is descriptive. The all-patient one-record policy uses a deterministic fallback '
        'when dates are missing or tied. Equal-record inference gives each available record the same weight.',
        '', 'The dates describe available archival recordings. Clinical presentation time and the effects of elapsed '
        'time, treatment and recording quality remain unresolved. Intervals condition on the saved predictions and '
        'observed class counts from repeatedly inspected PTB development data.',
        '', '[Registered protocol](../../docs/index_recording_audit_plan.md)', '',
    ]
    return '\n'.join(lines)


def run(output, verify):
    config = yaml.safe_load(SOURCE_CONFIG.read_text())
    paper = yaml.safe_load(Path('configs/paper.yaml').read_text())
    fold_path, beat_path = Path(config['folds_path']), Path(config['beats_path'])
    classes = config['classes']
    patients = pd.read_csv(fold_path).sort_values('patient_id')
    ids = patients.patient_id.to_numpy(dtype=str)
    labels = pd.Index(classes).get_indexer(patients.label)
    folds = patients.fold.to_numpy()
    beats = pd.read_csv(beat_path)
    beats = beats[beats.label.isin(classes)].reset_index(drop=True)
    if (len(np.unique(ids)) != len(ids) or np.any(labels < 0)
            or set(folds) != set(range(1, config['outer_folds'] + 1))
            or (len(ids), beats.record.nunique(), len(beats)) != (164, 391, 55863)
            or beats.groupby('record').patient_id.nunique().ne(1).any()):
        raise ValueError('Original target scope changed')
    choices, reasons, sizes, n_valid_dates, header_hashes = select_records(
        beats, ids, Path(paper['data']['raw_dir']))
    eligible = reasons == 'eligible'
    cohorts = {
        'dated_unique_earliest': (eligible, ('all_beats', 'one_record')),
        'dated_multiple_records': (eligible & (sizes > 1), ('all_beats', 'one_record')),
        'all_patients': (np.ones(len(ids), dtype=bool), ARMS),
    }
    root = Path(config['waveform_root'])
    source, scope, verification = replay.read_source(root, config['waveform_summary_sha256'], 'fingerprint')
    fold_hash, beat_hash = file_hash(fold_path), file_hash(beat_path)
    if scope['folds_sha256'] != fold_hash or scope['bank']['beats_sha256'] != beat_hash:
        raise ValueError('Source fold file or beat table changed')
    replay.require_fields(scope['config'], dict(
        classes=classes, outer_splits=config['outer_folds'], training_seeds=config['seeds']))
    predictions, verified_artifacts = load_predictions(root, scope, verification, ids, labels, folds, beats)
    pooled = {seed: pool_predictions(p, beats, ids, labels, choices)
              for seed, p in predictions.items()}
    evaluation = evaluate(labels, pooled, classes, cohorts, config['bootstrap_draws'], config['bootstrap_seed'])
    reference = evaluation['all_patients']['metrics']['all_beats']
    if (reference['ensemble'] != source['ensemble_metrics']['scratch']
            or reference['members'] != source['member_metrics']['scratch']):
        raise ValueError('All-beat control did not reproduce original waveform metrics')
    result = dict(
        complete=True, diagnostic_only=True, model_fits=0,
        n_patients=len(ids), n_records=len(header_hashes), n_beats=len(beats), n_valid_dates=n_valid_dates,
        verified_prediction_blocks=len(verified_artifacts), classes=classes, seeds=config['seeds'],
        folds=sorted(map(int, set(folds))),
        class_support={label: int(np.count_nonzero(labels == i)) for i, label in enumerate(classes)},
        eligibility={reason: dict(all=count, multiple_records=int(np.count_nonzero((reasons == reason) & (sizes > 1))))
                     for reason, count in sorted(Counter(reasons).items())},
        provenance=dict(source_study=scope['fingerprint'], source_summary_sha256=config['waveform_summary_sha256'],
                        audit_code_sha256=file_hash(Path(__file__)), protocol_sha256=file_hash(PROTOCOL),
                        folds_sha256=fold_hash, beats_sha256=beat_hash, source_config_sha256=file_hash(SOURCE_CONFIG),
                        headers_sha256=array_hash(np.asarray(sorted(header_hashes.items()))),
                        verified_artifacts_sha256=array_hash(np.asarray(sorted(
                            (name, *hashes) for name, hashes in verified_artifacts.items())))),
        cohorts=evaluation,
    )
    payloads = {'summary.json': json.dumps(result, indent=2, allow_nan=False) + '\n', 'REPORT.md': render(result)}
    if verify:
        for name, contents in payloads.items():
            if (output / name).read_text() != contents:
                raise ValueError(f'Saved {name} did not reproduce')
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, contents in payloads.items():
            (output / name).write_text(contents)
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
