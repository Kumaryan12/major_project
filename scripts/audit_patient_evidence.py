"""Read-only replay of the November research review. Outputs aggregates only.

Run from the repository root with the project environment and original local banks.
"""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, log_loss
from threadpoolctl import threadpool_limits

from mi_localization import benchmark, model_error_comparison as replay
from mi_localization import residual_context_report as bootstrap
from mi_localization.labels import clinical_fields, label_from_comments
from mi_localization.multiview_bank import array_hash
from mi_localization.signal_audit import file_hash


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_raw_records(beats):
    raw = Path('data/raw/ptbdb')
    records = beats[['record', 'patient_id', 'label']].drop_duplicates()
    check(records.record.is_unique, 'record assigned to multiple patients or labels')
    check(not beats.duplicated(['record', 'r_peak']).any(), 'duplicate R index in a record')
    digests = set()
    exam_dates = defaultdict(list)
    for row in records.itertuples():
        prefix = raw / row.record
        header = prefix.with_suffix('.hea').read_text(encoding='latin1')
        comments = [line for line in header.splitlines() if line.startswith('#')]
        check(prefix.parent.name == row.patient_id and label_from_comments(comments) == row.label,
              'header mapping or patient directory mismatch')
        exam_dates[row.patient_id].append(clinical_fields(comments).get('ecg date', '').strip())
        digests.add(file_hash(prefix.with_suffix('.xyz')))
    sizes = beats.groupby(['label', 'patient_id']).agg(records=('record', 'nunique'), beats=('record', 'size'))
    print('records, distinct encoded XYZ hashes:', len(records), len(digests))
    formats = Counter()
    for dates in exam_dates.values():
        for date in dates:
            fields = date.split('/')
            if len(fields) == 3 and all(field.isdigit() for field in fields):
                first, second = map(int, fields[:2])
                if first > 12:
                    formats['first > 12'] += 1
                elif second > 12:
                    formats['second > 12'] += 1
                else:
                    formats['both <= 12'] += 1
            else:
                formats['missing or other format'] += 1
    print('ECG date fields:', dict(formats), 'multi-record patients with different date text:',
          sum(len(set(dates)) > 1 for dates in exam_dates.values()))
    print('per-class recordings, single-record patients and range of beats per patient')
    print(sizes.groupby(level='label').agg(records=('records', 'sum'),
          single_record=('records', lambda n: int((n == 1).sum())),
          min_beats=('beats', 'min'), max_beats=('beats', 'max')).to_string())


def check_fits(root, study, verified, ids, y, folds):
    partitions = {}
    for fold in sorted(set(folds)):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        partitions[fold, 0] = (train, test)
        inner = benchmark.inner_patient_splits(y[train], ids[train], 3, study['config']['split_seed'] + fold * 1000)
        held = []
        for j, (a, b) in enumerate(inner, 1):
            partitions[fold, j] = train[a], train[b]
            held.extend(train[b])
        check(np.array_equal(np.sort(held), train), 'inner query coverage differs')
    counts = Counter()
    for name, digest in verified['restored_audit_hashes'].items():
        if not name.startswith('review/fits/') or not name.endswith('.json'):
            continue
        receipt = replay.checked_json(root / name, digest)
        fold, inner = receipt['outer_fold'], receipt['inner_fold']
        a, b = partitions[fold, inner]
        check(not np.intersect1d(ids[a], ids[b]).size, 'patient overlap')
        replay.require_fields(receipt, dict(train_membership_sha256=array_hash(ids[a]),
                              query_membership_sha256=array_hash(ids[b]), training_labels_sha256=array_hash(y[a]),
                              n_train=len(a), n_query=len(b), patient_overlap=0))
        counts['inner' if inner else 'outer'] += 1
    check(counts == {'outer': 240, 'inner': 2160}, 'incomplete fit receipt coverage')
    print('hash-verified memberships:', dict(counts))


def check_single_record(beats, ids, truth, root, c, reference):
    """Average errors over each patient's possible exams without choosing an exam by its outcome."""
    beat_groups = beats.patient_id.to_numpy()
    idx = pd.Index(ids).get_indexer(beat_groups)
    total = np.zeros((len(beats), len(c['classes'])))
    seen = np.zeros(len(beats), dtype=int)
    loss_history = []
    for seed in c['seeds']:
        for fold in range(1, c['outer_folds'] + 1):
            directory = root / 'review' / f'fold{fold:02d}_seed{seed}_scratch'
            fit = replay.read(directory / 'audit.json')
            check(file_hash(directory / 'model.pt') == fit['files']['model.pt'], 'waveform checkpoint differs')
            loss_history.append([entry['training_loss'] for entry in fit['fit']['history']])
            path = directory / 'predictions.npz'
            with np.load(path, allow_pickle=False) as saved:
                rows = saved['rows']
            total[rows] += replay.checked_probabilities(path, fit['files']['predictions.npz'], rows, total.shape[1])
            seen[rows] += 1
    check(np.all(seen == len(c['seeds'])), 'missing held-out predictions')
    losses = np.asarray(loss_history)
    check(losses.shape == (30, 8) and np.isfinite(losses).all(), 'scratch loss history differs')
    print('30 model hashes matched; mean first/last epoch training CE:', losses[:, [0, -1]].mean(axis=0))
    total /= len(c['seeds'])
    _, _, pooled = benchmark.aggregate_patients(total, truth[idx], beat_groups)
    check(np.allclose(pooled, reference), 'all-beat aggregation mismatch')
    scores = pd.DataFrame(total)
    per_record = scores.groupby([beats.patient_id, beats.record]).mean()
    confusion = np.zeros((total.shape[1], total.shape[1]))
    disagreements = 0
    for patient, y in zip(ids, truth):
        pred = per_record.loc[patient].to_numpy().argmax(1)
        disagreements += int(np.unique(pred).size > 1)
        confusion[y] += np.bincount(pred, minlength=total.shape[1]) / len(pred)
    denom = confusion.sum(0) + confusion.sum(1)
    print('uniform single-record expected confusion: accuracy, macro F1 of expected counts, '
          'patients with differing record predictions',
          confusion.trace()/len(ids), np.mean(2*confusion.diagonal()/denom), disagreements)


def run():
    c = yaml.safe_load(Path('configs/model_error_comparison_v1.yaml').read_text())
    replay.run(c, verify=True)
    table = pd.read_csv(c['folds_path']).sort_values('patient_id')
    classes, seeds = c['classes'], c['seeds']
    ids = table.patient_id.to_numpy(dtype=str)
    y, folds = pd.Index(classes).get_indexer(table.label), table.fold.to_numpy()
    beats = pd.read_csv(c['beats_path'])
    beats = beats[beats.label.isin(classes)].reset_index(drop=True)
    check_raw_records(beats)
    root = Path(c['convolution_root'])
    _, scope, verified = replay.read_source(root, c['convolution_summary_sha256'], 'study_fingerprint')
    check_fits(root, scope, verified, ids, y, folds)
    conv, _ = replay.convolution_predictions(root, scope, verified, ids, y, folds, seeds)
    root = Path(c['waveform_root'])
    _, scope, verified = replay.read_source(root, c['waveform_summary_sha256'], 'fingerprint')
    wave, _ = replay.waveform_predictions(root, scope, verified, beats, pd.Index(classes).get_indexer(beats.label),
                                          ids, y, folds, seeds)
    check_single_record(beats, ids, y, root, c, np.mean(list(wave.values()), axis=0))
    old = pd.read_csv('results/patient_benchmark/patient_predictions_xgboost.csv').set_index('patient_id')
    check(old.index.is_unique and set(old.index) == set(ids), 'baseline membership differs')
    old = old.loc[ids]
    check(np.array_equal(old.truth, table.label) and np.array_equal(old.fold, folds), 'baseline targets differ')
    pred_old = pd.Index(classes).get_indexer(old.prediction)
    check((pred_old >= 0).all(), 'unknown baseline label')
    labels = np.arange(len(classes))
    draws = bootstrap.patient_weights(y, labels, 5000, 42)
    original = bootstrap.weighted_f1(y, pred_old, labels, draws)
    for name, members in [('convolution', conv), ('waveform', wave)]:
        p = np.mean([members[s] for s in seeds], axis=0)
        pred = p.argmax(axis=1)
        def f1(a, b):
            return f1_score(a, b, labels=labels, average='macro', zero_division=0)
        samples = bootstrap.weighted_f1(y, pred, labels, draws)
        print(name, 'seed F1', [round(f1(y, members[s].argmax(1)), 4) for s in seeds])
        print('fold F1', [round(f1(y[folds == i], pred[folds == i]), 4) for i in sorted(set(folds))])
        print('ensemble F1, interval, delta from original XGB, paired interval', round(f1(y, pred), 6),
              np.quantile(samples, [.025, .975]), round(f1(y, pred)-f1(y, pred_old), 6),
              np.quantile(samples-original, [.025, .975]))
        print('mean max probability, accuracy, log loss, multiclass Brier sum',
              np.mean(p.max(1)), np.mean(y == pred), log_loss(y, p, labels=labels),
              np.square(p-np.eye(len(labels))[y]).sum(1).mean())
        print('class P/R/F1', {k: [round(v[m], 3) for m in ('precision', 'recall', 'f1-score')]
              for k, v in benchmark.metric_bundle(y, p, classes)['classification_report'].items() if k in classes})


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
