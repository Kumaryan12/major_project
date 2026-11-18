"""Compare fixed beat budgets from one or two archived recordings."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import yaml

from . import index_recording_audit as index
from . import model_error_comparison as replay
from .multiview_bank import array_hash
from .residual_context_report import patient_weights
from .signal_audit import file_hash


PROTOCOL = Path('docs/matched_recording_beats_plan.md')
PRIOR = Path('results/index_recording_audit_v1/summary.json')
PRIOR_HASH = '200850565c27afdd8f3a9738371f6b1fbd729648278871dc7547e39bc876d0fc'
OUTPUT = Path('results/matched_recording_beats_v1')
MAX_BEATS = 64
DRAWS = 256
SAMPLE_SEED = 42
ARMS = ('earliest', 'later', 'half_from_each', 'full_earliest', 'full_pair')
CONTRASTS = (('half_from_each', 'earliest'), ('later', 'earliest'),
             ('half_from_each', 'later'), ('full_pair', 'half_from_each'),
             ('full_earliest', 'earliest'))


@dataclass(frozen=True)
class RecordPair:
    patient: str
    earliest: str
    later: str
    earliest_day: date
    later_day: date
    early_rows: np.ndarray
    later_rows: np.ndarray

    @property
    def days(self):
        return (self.later_day - self.earliest_day).days


def order_pair(dates):
    earliest, status = index.select_record(dates)
    if status != 'eligible' or len(dates) < 2:
        return None
    later = min((record for record in dates if record != earliest),
                key=lambda record: (dates[record], record))
    return earliest, later


def find_pairs(beats, raw_root):
    pairs, headers = [], {}
    for patient, rows in beats.groupby('patient_id', sort=True):
        records = {record: frame.index.to_numpy() for record, frame in rows.groupby('record')}
        dates = {}
        for record in records:
            path = raw_root / f'{record}.hea'
            headers[record] = file_hash(path)
            dates[record] = index.recording_date(path.read_text(encoding='latin1'))
        choice = order_pair(dates)
        if choice is not None:
            early, later = choice
            pairs.append(RecordPair(patient, early, later, dates[early], dates[later],
                                    records[early], records[later]))
    return pairs, array_hash(np.asarray(sorted(headers.items())))


def sample_indices(pair, rng):
    budget = 2 * (min(MAX_BEATS, len(pair.early_rows), len(pair.later_rows)) // 2)
    if budget < 2:
        raise ValueError('Both recordings must have at least two beats')
    early = np.stack([rng.choice(pair.early_rows, budget, replace=False) for _ in range(DRAWS)])
    later = np.stack([rng.choice(pair.later_rows, budget, replace=False) for _ in range(DRAWS)])
    return early, later


def decisions(probabilities, pair, early, later):
    half = early.shape[1] // 2
    one = probabilities[early]
    two = probabilities[later]
    pair_rows = np.concatenate([pair.early_rows, pair.later_rows])
    return np.stack([
        one.mean(1).argmax(1),
        two.mean(1).argmax(1),
        (one[:, :half].mean(1) + two[:, :half].mean(1)).argmax(1),
        np.full(len(early), probabilities[pair.early_rows].mean(0).argmax()),
        np.full(len(early), probabilities[pair_rows].mean(0).argmax()),
    ])


def metrics(confusion, classes):
    support = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    correct = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = support + predicted
    scores = np.divide(2 * correct, denominator, out=np.zeros_like(correct), where=denominator != 0)
    return {
        'macro_f1': float(scores.mean()),
        'accuracy': float(correct.sum() / support.sum()),
        'class_f1': dict(zip(classes, scores.tolist())),
        'confusion': confusion.tolist(),
    }


def summarize(labels, predictions, classes, bootstrap_draws, bootstrap_seed):
    targets = np.eye(len(classes))[labels]
    labels_for_bootstrap = np.arange(len(classes))
    weights = patient_weights(labels, labels_for_bootstrap, bootstrap_draws, bootstrap_seed)
    arm_results, boot = {}, {}
    for arm, calls in zip(ARMS, predictions):
        one_hot = np.eye(len(classes))[calls]
        expected_calls = one_hot.mean(axis=1)
        expected_confusion = targets.T @ expected_calls
        draws = np.einsum('pc,pdk->dck', targets, one_hot)
        diag = np.diagonal(draws, axis1=1, axis2=2)
        denominator = draws.sum(axis=1) + draws.sum(axis=2)
        per_draw = np.divide(2 * diag, denominator, out=np.zeros_like(diag), where=denominator != 0).mean(1)
        arm_results[arm] = dict(metrics(expected_confusion, classes),
                                mean_draw_macro_f1=float(per_draw.mean()),
                                sd_draw_macro_f1=float(per_draw.std(ddof=1)))
        tp = weights @ (targets * expected_calls)
        denom = weights @ (targets + expected_calls)
        boot[arm] = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0).mean(1)
    differences = {}
    for arm, control in CONTRASTS:
        differences[f'{arm}_minus_{control}'] = dict(
            delta=arm_results[arm]['macro_f1'] - arm_results[control]['macro_f1'],
            interval=np.quantile(boot[arm] - boot[control], [.025, .975]).tolist())
    return arm_results, differences


def render(result):
    lines = [
        '# Measured-XYZ localization with matched beat counts', '',
        f"Replayed {result['verified_prediction_blocks']} original held-out blocks. The dated paired-record "
        f"cohort contains {result['patients']} patients. Median spacing is {result['median_days']:g} days. "
        f"Matched budgets are {result['budget_counts']} beats per patient with {result['draws']} draws. Zero new fits.",
        '', '| Arm | Macro F1 of expected counts | Accuracy | Mean draw macro F1 ± SD |',
        '|---|---:|---:|---:|',
    ]
    for arm, row in result['arms'].items():
        lines.append(f"| {arm} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | "
                     f"{row['mean_draw_macro_f1']:.6f} ± {row['sd_draw_macro_f1']:.6f} |")
    lines += ['', '| Paired contrast | Macro F1 difference | Descriptive 95% interval |',
              '|---|---:|---:|']
    for name, contrast in result['contrasts'].items():
        low, high = contrast['interval']
        lines.append(f"| {name} | {contrast['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ['', '| Class | Patients | Earliest F1 | Later F1 | Half from each F1 |',
              '|---|---:|---:|---:|---:|']
    for label, support in result['class_support'].items():
        scores = [result['arms'][arm]['class_f1'][label] for arm in ARMS[:3]]
        lines.append(f"| {label} | {support} | {scores[0]:.3f} | {scores[1]:.3f} | {scores[2]:.3f} |")
    lines += ['', '| Arm | Member F1 in declared seed order |', '|---|---|']
    for arm in ARMS:
        member = ', '.join(f'{scores[arm]:.6f}' for scores in result['member_macro_f1'].values())
        lines.append(f'| {arm} | {member} |')
    lines += [
        '', 'The primary contrast is half from each minus earliest only. The full-pair and full-earliest '
        'references use their naturally available beat counts. Intervals condition on the frozen predictions '
        'and date-complete repeat-record subset. Differences can reflect changes in injury, treatment, '
        'recording quality or other factors across archived visits.',
        '', '[Registered protocol](../../docs/matched_recording_beats_plan.md)', '',
    ]
    return '\n'.join(lines)


def run(output, verify, review_output):
    prior = replay.checked_json(PRIOR, PRIOR_HASH)
    config = yaml.safe_load(index.SOURCE_CONFIG.read_text())
    paper = yaml.safe_load(Path('configs/paper.yaml').read_text())
    fold_path, beat_path = Path(config['folds_path']), Path(config['beats_path'])
    if (file_hash(fold_path) != prior['provenance']['folds_sha256']
            or file_hash(beat_path) != prior['provenance']['beats_sha256']):
        raise ValueError('Expected the original fold and beat table')
    classes = config['classes']
    patients = pd.read_csv(fold_path).sort_values('patient_id')
    ids = patients.patient_id.to_numpy(dtype=str)
    labels = pd.Index(classes).get_indexer(patients.label)
    beats = pd.read_csv(beat_path)
    beats = beats[beats.label.isin(classes)].reset_index(drop=True)
    raw_root = Path(paper['data']['raw_dir'])
    pairs, header_hash = find_pairs(beats, raw_root)
    if (header_hash != prior['provenance']['headers_sha256']
            or len(pairs) != prior['cohorts']['dated_multiple_records']['patients']):
        raise ValueError('Paired examination membership or headers changed')
    root = Path(config['waveform_root'])
    _, scope, verification = replay.read_source(root, config['waveform_summary_sha256'], 'fingerprint')
    saved, artifacts = index.load_predictions(root, scope, verification, ids, labels, patients.fold.to_numpy(), beats)
    artifact_hash = array_hash(np.asarray(sorted((name, *hashes) for name, hashes in artifacts.items())))
    if artifact_hash != prior['provenance']['verified_artifacts_sha256']:
        raise ValueError('Prediction artifacts do not match the completed recording audit')
    ensemble = np.mean(list(saved.values()), axis=0)
    paired_rows = pd.Index(ids).get_indexer([pair.patient for pair in pairs])
    truth = labels[paired_rows]
    rng = np.random.default_rng(SAMPLE_SEED)
    samples = [sample_indices(pair, rng) for pair in pairs]
    choices = [pair.earliest for pair in pairs]
    subset = beats[beats.patient_id.isin([pair.patient for pair in pairs])]
    baseline = index.pool_predictions(ensemble[subset.index], subset.reset_index(drop=True),
                                      ids[paired_rows], truth, choices)
    prev = prior['cohorts']['dated_multiple_records']['metrics']
    for arm in ('all_beats', 'one_record'):
        score = f1_score(truth, baseline[arm].argmax(1), average='macro')
        if score != prev[arm]['ensemble']['macro_f1']:
            raise ValueError('Prior paired-cohort metrics did not reproduce')
    calls = [decisions(ensemble, pair, *sample) for pair, sample in zip(pairs, samples)]
    arms, contrasts = summarize(truth, np.stack(calls, axis=1), classes,
                                config['bootstrap_draws'], config['bootstrap_seed'])
    member_macro_f1 = {}
    for seed, probabilities in saved.items():
        member_calls = np.stack([decisions(probabilities, p, *s) for p, s in zip(pairs, samples)], axis=1)
        member_macro_f1[str(seed)] = {}
        for arm, matrix in zip(ARMS, member_calls):
            confusion = np.eye(len(classes))[truth].T @ np.eye(len(classes))[matrix].mean(1)
            member_macro_f1[str(seed)][arm] = metrics(confusion, classes)['macro_f1']
    result = dict(
        complete=True, model_fits=0, patients=len(pairs), folds=sorted(set(map(int, patients.fold))),
        seeds=config['seeds'], draws=DRAWS, sampling_seed=SAMPLE_SEED,
        bootstrap_draws=config['bootstrap_draws'],
        median_days=float(np.median([pair.days for pair in pairs])),
        budget_counts=dict(sorted(Counter(early.shape[1] for early, _ in samples).items())),
        class_support=dict(zip(classes, np.bincount(truth, minlength=len(classes)).tolist())),
        verified_prediction_blocks=len(artifacts), arms=arms, contrasts=contrasts,
        member_macro_f1=member_macro_f1,
        provenance=dict(prior_summary_sha256=PRIOR_HASH, code_sha256=file_hash(Path(__file__)),
                        protocol_sha256=file_hash(PROTOCOL), verified_artifacts_sha256=artifact_hash,
                        headers_sha256=header_hash, paired_membership_sha256=array_hash(np.asarray(
                            [(pair.patient, pair.earliest, pair.later) for pair in pairs]))),
    )
    payloads = {'summary.json': json.dumps(result, indent=2, allow_nan=False) + '\n', 'REPORT.md': render(result)}
    if verify:
        for name, content in payloads.items():
            if (output / name).read_text() != content:
                raise ValueError(f'{name} did not reproduce')
    else:
        if output.exists():
            raise FileExistsError(f'{output} already exists')
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temporary:
            staged = Path(temporary) / 'audit'
            staged.mkdir()
            for name, content in payloads.items():
                (staged / name).write_text(content)
            staged.rename(output)
    print(payloads['REPORT.md'])
    if review_output is not None:
        from . import blinded_recording_review
        blinded_recording_review.prepare(review_output, pairs, classes, truth,
                                         baseline['one_record'].argmax(1), baseline['all_beats'].argmax(1),
                                         raw_root, beats, paper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--prepare-review', action='store_true')
    args = parser.parse_args()
    review_output = Path('tmp/matched_recording_review_v1') if args.prepare_review else None
    with threadpool_limits(limits=1):
        run(args.output, args.verify, review_output)


if __name__ == '__main__':
    main()
