"""Cross-fit later-recording teachers for a measured-XYZ earliest-recording student."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F
import yaml

from . import benchmark, index_recording_audit as index, matched_recording_beats as matched
from . import model_error_comparison as replay, neural, preprocessing_study, regularization
from .contrastive import initial_state_hash, _progress
from .multiview_bank import array_hash
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash
from .supervised_source_train import save_checkpoint


CONFIG = Path('configs/later_recording_distillation_v1.yaml')
PROTOCOL = Path('docs/later_recording_distillation_plan.md')
MATCHED_HASH = '64f09179ab7060a238d6f16c271e3f1578ad7c45827bdadd8b2b2232b7f777dc'
ARMS = ('supervised', 'paired', 'shuffled')
CONTRASTS = (('paired', 'supervised'), ('paired', 'shuffled'), ('shuffled', 'supervised'))


@contextmanager
def runner_lock(root):
    with (root / 'runner.lock').open('a+b') as stream:
        if platform.system() == 'Windows':
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def shuffled_targets(patients, labels, seed):
    rng = np.random.default_rng(seed)
    donors = np.arange(len(patients))
    for label in np.unique(labels[patients]):
        group = np.flatnonzero(labels[patients] == label)
        if len(group) < 2:
            raise ValueError('Each paired training class needs at least two patients for shuffling')
        order = rng.permutation(group)
        donors[order] = np.roll(order, 1)
    return donors


def teacher_partitions(patients, labels, splits, seed):
    """Each class is held out in blocks of at least two, so a donor can share the same teacher."""
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in range(splits)]
    for i, label in enumerate(np.unique(labels[patients])):
        group = rng.permutation(patients[labels[patients] == label])
        blocks = min(splits, len(group) // 2)
        if blocks < 2:
            raise ValueError('Each paired class requires two held-out blocks')
        for offset, block in enumerate(np.array_split(group, blocks)):
            partitions[(i + offset) % splits].extend(block.tolist())
    return [np.asarray(sorted(block), dtype=int) for block in partitions]


def load_data(settings):
    prior = replay.checked_json(matched.PRIOR, matched.PRIOR_HASH)
    timing = replay.checked_json(matched.OUTPUT / 'summary.json', MATCHED_HASH)
    wave_config = yaml.safe_load(index.SOURCE_CONFIG.read_text())
    _, reference, _ = replay.read_source(
        Path(wave_config['waveform_root']), wave_config['waveform_summary_sha256'], 'fingerprint')
    waves, beat_labels, beat_groups, bank = regularization.load_bank(settings)
    if (bank != reference['bank'] or file_hash(settings['folds_path']) != reference['folds_sha256']
            or settings['classes'] != reference['config']['classes']
            or settings['training_seeds'] != prior['seeds']):
        raise ValueError('Original waveforms, patient folds, classes or seeds changed')
    patients = pd.read_csv(settings['folds_path']).sort_values('patient_id')
    ids = patients.patient_id.to_numpy(dtype=str)
    labels = pd.Index(settings['classes']).get_indexer(patients.label)
    groups = pd.Index(ids).get_indexer(beat_groups)
    if np.any(groups < 0) or not np.array_equal(labels[groups], beat_labels):
        raise ValueError('Beat and frozen patient labels disagree')
    beats = pd.read_csv(settings['beats_path'])
    beats = beats[beats.label.isin(settings['classes'])].reset_index(drop=True)
    paper = yaml.safe_load(Path(settings['paper_config']).read_text())
    raw = Path(paper['data']['raw_dir'])
    chosen, reasons, _, _, hashes = index.select_records(beats, ids, raw)
    if array_hash(np.asarray(sorted(hashes.items()))) != prior['provenance']['headers_sha256']:
        raise ValueError('PTB archival header set changed')
    pairs, _ = matched.find_pairs(beats, raw)
    digest = array_hash(np.asarray([(p.patient, p.earliest, p.later) for p in pairs]))
    eligible = reasons == 'eligible'
    if (digest != timing['provenance']['paired_membership_sha256'] or int(eligible.sum()) != 145
            or len(pairs) != 79 or not all(chosen[np.searchsorted(ids, p.patient)] == p.earliest for p in pairs)):
        raise ValueError('Original date eligibility or later-record pair changed')
    earliest = np.flatnonzero(beats.record.to_numpy() == chosen[groups])
    later = {int(np.searchsorted(ids, p.patient)): p.later_rows for p in pairs}
    return dict(waves=waves, beat_labels=beat_labels, labels=labels, groups=groups, ids=ids, bank=bank,
                folds=patients.fold.to_numpy(), earliest=earliest, later=later, eligible=eligible,
                source_headers=prior['provenance']['headers_sha256'], pair_sha256=digest)


def sampling_plan(data, rows, cfg, seed):
    weights = benchmark.patient_balanced_weights(data['beat_labels'][rows], data['groups'][rows])
    norm = neural.FoldNormalizer().fit(
        data['waves'], np.empty((len(data['waves']), 0), np.float32), rows, weights)
    rng = np.random.default_rng(seed)
    samples = rng.choice(rows, size=(cfg['target_epochs'], cfg['samples_per_epoch']),
                         p=weights / weights.sum())
    return norm, samples


def objective(logits, labels, target, mask, temperature, weight):
    supervised = F.cross_entropy(logits, labels)
    if not weight:
        return supervised
    soft = F.kl_div(F.log_softmax(logits / temperature, dim=1), target, reduction='none').sum(1)
    return supervised + weight * temperature ** 2 * (soft * mask).mean()


def new_model(cfg, seed):
    neural._seed(seed, cfg)
    return neural.VCGResNet(len(cfg['classes']), tuple(cfg['widths']), cfg['pool_bins'], cfg['dropout'])


def run_fit(data, cfg, root, name, context, query, norm, samples, targets, mask, weight, verify):
    folder = root / 'review' / name
    audit_path, model_path, probabilities_path = (folder / f for f in ('audit.json', 'model.pt', 'probabilities.npz'))
    device = neural._device(cfg)
    model = new_model(cfg, context['model_seed'])
    context = dict(context, initial_sha256=initial_state_hash(model), slots_sha256=array_hash(samples),
                   normalization=norm.audit(), loss_weight=weight)
    closed = audit_path.is_file()
    if closed or verify:
        audit = replay.read(audit_path)
        if audit['context'] != context or file_hash(model_path) != audit['model_sha256']:
            raise ValueError(f'Changed model or context in {name}')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if checkpoint['context'] != context:
            raise ValueError(f'Changed checkpoint context in {name}')
        model.load_state_dict(checkpoint['model'], strict=True)
        history = audit['history']
        model.to(device)
    else:
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        mean, scale = norm.wave_mean_[:, None], norm.wave_scale_[:, None]
        y, groups, waves = data['beat_labels'], data['groups'], data['waves']
        history = []
        for epoch, epoch_rows in enumerate(samples, 1):
            model.train()
            losses = []
            for rows in np.array_split(epoch_rows, len(epoch_rows) // cfg['batch_size']):
                x = torch.from_numpy((waves[rows] - mean) / scale).to(device)
                labels = torch.from_numpy(y[rows]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                target_p = torch.from_numpy(targets[groups[rows]]).to(device) if weight else None
                eligible = torch.from_numpy(mask[groups[rows]]).to(device) if weight else None
                loss = objective(logits, labels, target_p, eligible, cfg['temperature'], weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite objective in {name}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append(dict(epoch=epoch, loss=float(np.mean(losses)),
                                beat_presentations=len(epoch_rows), sample_sha256=array_hash(epoch_rows)))
            _progress(root, state='training', member=name, epoch=epoch)
        folder.mkdir(parents=True, exist_ok=True)
    if any(not torch.isfinite(p).all() for p in model.state_dict().values()):
        raise ValueError(f'Nonfinite checkpoint in {name}')
    if (len(history) != len(samples) or any(
            row['epoch'] != e or not np.isfinite(row['loss'])
            or row['beat_presentations'] != len(s) or row['sample_sha256'] != array_hash(s)
            for e, (row, s) in enumerate(zip(history, samples), 1))):
        raise ValueError(f'Exposure or loss changed in {name}')
    probabilities = preprocessing_study.predict_waves(
        model, norm, data['waves'], data['beat_labels'], query, cfg)
    if closed:
        snapshot = replay.checked_probabilities(probabilities_path, audit['probabilities_sha256'],
                                                 query, len(cfg['classes']))
        if not np.allclose(probabilities, snapshot, rtol=2e-5, atol=2e-5):
            raise ValueError(f'Prediction replay changed in {name}')
        probabilities = snapshot.astype(probabilities.dtype, copy=False)
    else:
        save_checkpoint(model_path, dict(context=context, model=model.state_dict()))
        partial = probabilities_path.with_suffix('.partial.npz')
        np.savez_compressed(partial, rows=query, probabilities=probabilities)
        partial.replace(probabilities_path)
        neural._save_json(audit_path, dict(context=context, history=history, model_sha256=file_hash(model_path),
                                          probabilities_sha256=file_hash(probabilities_path)))
    return probabilities, file_hash(audit_path)


def make_targets(data, cfg, patients, fold, root, fingerprint, verify, receipts):
    groups, labels = data['groups'], data['labels']
    paired = np.array(sorted(set(patients) & data['later'].keys()), dtype=int)
    teacher = np.zeros((len(labels), len(cfg['classes'])), np.float32)
    received = np.zeros(len(labels), dtype=bool)
    shuffled = teacher.copy()
    donor_ids = np.full(len(labels), -1, dtype=int)
    partitions = teacher_partitions(paired, labels, cfg['teacher_splits'], cfg['teacher_seed'] + fold)
    for split, targets in enumerate(partitions, 1):
        fit = np.setdiff1d(paired, targets, assume_unique=True)
        train = np.sort(np.concatenate([data['later'][p] for p in fit]))
        query = np.sort(np.concatenate([data['later'][p] for p in targets]))
        if (np.intersect1d(groups[train], groups[query]).size or received[targets].any()
                or set(fit) | set(targets) != set(paired)
                or np.unique(labels[fit]).size != len(cfg['classes'])):
            raise ValueError('Invalid cross-fitting partition')
        model_seed = cfg['teacher_seed'] + 100 * fold + split
        norm, samples = sampling_plan(data, train, cfg, model_seed)
        context = dict(fingerprint=fingerprint, fold=fold, split=split, model_seed=model_seed,
                       fit_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query))
        name = f'fold{fold:02d}_teacher{split}'
        probabilities, receipts[name] = run_fit(data, cfg, root, name, context, query, norm, samples,
                                                 None, None, 0., verify)
        for target in targets:
            teacher[target] = probabilities[groups[query] == target].mean(axis=0)
        received[targets] = True
        donors = shuffled_targets(targets, labels, cfg['shuffle_seed'] + 10 * fold + split)
        donor_ids[targets] = targets[donors]
    if not np.array_equal(np.flatnonzero(received), paired):
        raise ValueError('Incomplete teacher coverage')
    softened = np.power(teacher[paired], 1 / cfg['temperature'])
    teacher[paired] = softened / softened.sum(axis=1, keepdims=True)
    shuffled[paired] = teacher[donor_ids[paired]]
    return teacher, shuffled, received, paired, donor_ids[paired]


def fold_fit(data, cfg, fold, train_patients, query_patients, root, fingerprint, verify, receipts):
    groups = data['groups']
    if np.intersect1d(train_patients, query_patients).size:
        raise ValueError('Student training and evaluation patients overlap')
    train = data['earliest'][np.isin(groups[data['earliest']], train_patients)]
    query = data['earliest'][np.isin(groups[data['earliest']], query_patients)]
    paired, shuffled, mask, target_patients, donors = make_targets(
        data, cfg, train_patients, fold, root, fingerprint, verify, receipts)
    predictions = {}
    for seed in cfg['training_seeds']:
        model_seed = seed + fold
        norm, samples = sampling_plan(data, train, cfg, model_seed)
        shared = dict(fingerprint=fingerprint, fold=fold, seed=seed, model_seed=model_seed,
                      fit_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query),
                      matched_target_mask_sha256=array_hash(mask))
        for arm, targets in zip(ARMS, (None, paired, shuffled)):
            name = f'fold{fold:02d}_seed{seed}_{arm}'
            context = dict(shared, arm=arm, targets_sha256=array_hash(targets) if targets is not None else None,
                           donors_sha256=array_hash(donors) if arm == 'shuffled' else None)
            weight = cfg['distillation_weight'] if targets is not None else 0.
            probabilities, receipts[name] = run_fit(data, cfg, root, name, context, query, norm, samples,
                                                     targets, mask, weight, verify)
            out_ids, _, pooled = benchmark.aggregate_patients(probabilities, data['beat_labels'][query], groups[query])
            if not np.array_equal(out_ids, query_patients):
                raise ValueError('Student inference did not cover the declared query')
            predictions[arm, seed] = pooled
        comparison = [replay.read(root / 'review' / f'fold{fold:02d}_seed{seed}_{arm}' / 'audit.json')
                      for arm in ARMS]
        for field in ('initial_sha256', 'slots_sha256', 'normalization'):
            if any(item['context'][field] != comparison[0]['context'][field] for item in comparison[1:]):
                raise ValueError(f'Matched student field changed: {field}')
    changed = int(np.count_nonzero(np.any(paired[target_patients] != shuffled[target_patients], axis=1)))
    return predictions, dict(teacher_patients=len(target_patients), shuffled_targets_changed=changed,
                             evaluated_patients=len(query_patients))


def cohorts(data, cfg, predictions):
    masks = dict(dated=data['eligible'], repeat_record=np.isin(np.arange(len(data['ids'])), list(data['later'])),
                 all_patients=np.ones(len(data['ids']), dtype=bool))
    results = {}
    class_ids = np.arange(len(cfg['classes']))
    for name, mask in masks.items():
        truth = data['labels'][mask]
        weights = patient_weights(truth, class_ids, cfg['bootstrap_draws'], cfg['seed'])
        scores, bootstrap = {}, {}
        for arm in ARMS:
            members = {str(seed): predictions[arm, seed][mask] for seed in cfg['training_seeds']}
            if any(not np.isfinite(p).all() for p in members.values()):
                raise ValueError('Missing held-out patient predictions')
            ensemble = np.mean(list(members.values()), axis=0)
            scores[arm] = dict(ensemble=benchmark.metric_bundle(truth, ensemble, cfg['classes']),
                               members={s: benchmark.metric_bundle(truth, p, cfg['classes']) for s, p in members.items()})
            bootstrap[arm] = weighted_f1(truth, ensemble.argmax(1), class_ids, weights)
        contrasts = {f'{arm}_minus_{control}': dict(
            delta=scores[arm]['ensemble']['macro_f1'] - scores[control]['ensemble']['macro_f1'],
            interval=np.quantile(bootstrap[arm] - bootstrap[control], [.025, .975]).tolist())
            for arm, control in CONTRASTS}
        results[name] = dict(patients=int(mask.sum()), scores=scores, contrasts=contrasts)
    return results


def report(result):
    lines = ['# Later-recording distillation for earliest-record VCG', '',
             f"Verified {result['restored_fits']} saved fits across {len(result['folds'])} folds. "
             f"All {result['student_matched_groups']} student fold/seed groups matched.", '']
    if result['smoke']:
        return '\n'.join(lines + ['Inner patients only. No outer performance evaluation.', ''])
    lines += ['| Cohort | Arm | Ensemble macro F1 | Member macro F1 in seed order |',
              '|---|---|---:|---|']
    for name, rows in result['cohorts'].items():
        for arm, scores in rows['scores'].items():
            members = ', '.join(f"{m['macro_f1']:.6f}" for m in scores['members'].values())
            lines.append(f"| {name} ({rows['patients']}) | {arm} | {scores['ensemble']['macro_f1']:.6f} | {members} |")
    lines += ['', '| Cohort | Paired contrast | Macro F1 difference | Descriptive 95% interval |',
              '|---|---|---:|---|']
    for name, rows in result['cohorts'].items():
        for contrast, item in rows['contrasts'].items():
            low, high = item['interval']
            lines.append(f"| {name} | {contrast} | {item['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ['', '| Class on primary dated cohort | Patients | Supervised F1 | Paired F1 | Shuffled F1 |',
              '|---|---:|---:|---:|---:|']
    score = result['cohorts']['dated']['scores']
    for label in result['classes']:
        rows = [score[arm]['ensemble']['classification_report'][label] for arm in ARMS]
        lines.append(f"| {label} | {int(rows[0]['support'])} | " + ' | '.join(f"{x['f1-score']:.4f}" for x in rows) + ' |')
    lines += [
        '', 'Teachers used later recordings of outer-training patients and generated patient targets by cross-fitting. '
        'Students used the same waveforms, sampled rows and training budget. The dated cohort and repeat-record '
        'cohort use the unique earliest archived recording. The all-patient sensitivity includes deterministic fallbacks.',
        '', 'The primary comparisons are paired versus supervised and paired versus shuffled in the 145-patient '
        'dated cohort. Intervals are conditional on the saved held-out predictions, and PTB remains development data.',
        '', '[Protocol](../../../docs/later_recording_distillation_plan.md)', '',
    ]
    return '\n'.join(lines)


def run(config_path, smoke, verify):
    declared = yaml.safe_load(config_path.read_text())
    cfg = dict(yaml.safe_load(Path(declared['base_config']).read_text()), **declared)
    data = load_data(cfg)
    if smoke:
        cfg.update(target_epochs=cfg['smoke_epochs'], samples_per_epoch=cfg['smoke_samples_per_epoch'],
                   training_seeds=cfg['training_seeds'][:1])
    folds = [1] if smoke else list(range(1, cfg['outer_splits'] + 1))
    if (cfg['samples_per_epoch'] < cfg['batch_size'] or cfg['samples_per_epoch'] % cfg['batch_size'] != 0
            or set(data['folds']) != set(range(1, cfg['outer_splits'] + 1))):
        raise ValueError('Invalid exposure or frozen folds')
    scope = dict(cfg=cfg, smoke=smoke, folds=folds, bank=data['bank'], headers_sha256=data['source_headers'],
                 paired_membership_sha256=data['pair_sha256'], folds_sha256=file_hash(cfg['folds_path']),
                 code_sha256=file_hash(Path(__file__)), protocol_sha256=file_hash(PROTOCOL),
                 dependency_sha256={m.__name__: file_hash(Path(m.__file__)) for m in
                                    (matched, index, neural, preprocessing_study, regularization, benchmark)},
                 runtime=dict(numpy=np.__version__, torch=str(torch.__version__), sklearn=sklearn.__version__,
                              platform=platform.platform(), device=str(neural._device(cfg))))
    fingerprint = neural._hash(scope)
    scope['fingerprint'] = fingerprint
    root = Path(cfg['output_dir']) / ('smoke' if smoke else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    with runner_lock(root):
        manifest = root / 'study_manifest.json'
        if manifest.exists() or verify:
            if replay.read(manifest) != scope:
                raise ValueError('Run scope changed. Use a new study namespace.')
        else:
            neural._save_json(manifest, scope)
        receipts, counts = {}, {}
        predictions = {(arm, seed): np.full((len(data['ids']), len(cfg['classes'])), np.nan)
                       for arm in ARMS for seed in cfg['training_seeds']}
        for fold in folds:
            train, query = np.flatnonzero(data['folds'] != fold), np.flatnonzero(data['folds'] == fold)
            if smoke:
                unpaired = np.setdiff1d(train, list(data['later']))
                query = np.sort([unpaired[data['labels'][unpaired] == label][0]
                                 for label in range(len(cfg['classes']))])
                train = np.setdiff1d(train, query, assume_unique=True)
            values, counts[str(fold)] = fold_fit(data, cfg, fold, train, query, root, fingerprint, verify, receipts)
            for key, p in values.items():
                predictions[key][query] = p
            _progress(root, state='verifying' if verify else 'fold_saved', fold=fold, completed_fits=len(receipts))
            print(f"{'Verified' if verify else 'Saved'} fold {fold}/{len(folds)} with {len(receipts)} completed fits",
                  flush=True)
        expected = len(folds) * (cfg['teacher_splits'] + len(ARMS) * len(cfg['training_seeds']))
        if len(receipts) != expected:
            raise ValueError('Fit inventory incomplete')
        if not verify:
            _progress(root, state='fits_saved', completed_fits=len(receipts))
            return
        result = dict(complete=True, smoke=smoke, fingerprint=fingerprint, folds=folds,
                      classes=cfg['classes'], seeds=cfg['training_seeds'], restored_fits=len(receipts),
                      student_matched_groups=len(folds) * len(cfg['training_seeds']), fold_audit=counts,
                      verification_sha256=neural._hash(receipts),
                      cohorts=None if smoke else cohorts(data, cfg, predictions))
        for file, content in {'summary.json': json.dumps(result, indent=2, allow_nan=False) + '\n',
                              'REPORT.md': report(result)}.items():
            path = root / file
            if path.exists() and path.read_text() != content:
                raise ValueError(f'Completed {file} changed during replay')
            if not path.exists():
                temporary = path.with_suffix('.partial')
                temporary.write_text(content)
                temporary.replace(path)
        _progress(root, state='complete_validated', completed_fits=len(receipts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.smoke, args.verify)


if __name__ == '__main__':
    main()
