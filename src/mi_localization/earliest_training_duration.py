"""Nested patient-level selection of a supervised earliest-record training duration."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import benchmark, later_recording_distillation as source, neural
from . import model_error_comparison as replay, preprocessing_study
from .contrastive import _progress, initial_state_hash
from .multiview_bank import array_hash
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash
from .supervised_source_train import save_checkpoint

CONFIG = Path('configs/earliest_training_duration_v1.yaml')
PROTOCOL = Path('docs/earliest_training_duration_plan.md')


def read_reference(config):
    root = Path(config['source_root'])
    summary = replay.checked_json(root / 'summary.json', config['source_summary_sha256'])
    manifest = replay.checked_json(root / 'study_manifest.json', config['source_manifest_sha256'])
    if (not summary['complete'] or summary['smoke'] or summary['restored_fits'] != 120
            or summary['fingerprint'] != manifest['fingerprint']
            or file_hash(Path(source.__file__)) != manifest['code_sha256']):
        raise ValueError('Expected the verified earliest-record supervised source')
    audit_files = (root / 'review').glob('*/audit.json')
    hashes = {p.parent.name: file_hash(p) for p in audit_files}
    if len(hashes) != 120 or neural._hash(hashes) != summary['verification_sha256']:
        raise ValueError('Original study fit receipts differ from verified evidence')
    return summary, manifest, hashes


def train_checkpoints(data, cfg, root, folder, name, context, samples, norm, model, epochs):
    device = neural._device(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    mean, scale = norm.wave_mean_[:, None], norm.wave_scale_[:, None]
    waves, labels = data['waves'], data['beat_labels']
    history, hashes = [], {}
    for epoch, epoch_rows in enumerate(samples, 1):
        model.train()
        losses = []
        for rows in np.array_split(epoch_rows, len(epoch_rows) // cfg['batch_size']):
            x = torch.from_numpy((waves[rows] - mean) / scale).to(device)
            y = torch.from_numpy(labels[rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = source.objective(model(x), y, None, None, cfg['temperature'], 0.)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite objective in {name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)),
                            sample_sha256=array_hash(epoch_rows), presentations=len(epoch_rows)))
        if epoch in epochs:
            path = folder / f'epoch{epoch:02d}.pt'
            save_checkpoint(path, dict(context=context, epoch=epoch, model=model.state_dict()))
            hashes[str(epoch)] = file_hash(path)
            _progress(root, state='training', trajectory=name, epoch=epoch)
    return history, hashes


def restore_predictions(folder, model, data, cfg, context, query, norm, epochs, hashes, saved_hashes):
    blocks, prob_hashes = {}, {}
    for epoch in epochs:
        key = str(epoch)
        path = folder / f'epoch{epoch:02d}.pt'
        if file_hash(path) != hashes[key]:
            raise ValueError('Saved checkpoint hash changed')
        values = torch.load(path, map_location='cpu', weights_only=True)
        if values['context'] != context or values['epoch'] != epoch:
            raise ValueError('Saved checkpoint belongs to another trajectory or epoch')
        model.load_state_dict(values['model'], strict=True)
        model.to(neural._device(cfg))
        probabilities = preprocessing_study.predict_waves(
            model, norm, data['waves'], data['beat_labels'], query, cfg)
        prob_path = folder / f'epoch{epoch:02d}_probabilities.npz'
        if saved_hashes is not None:
            expected = replay.checked_probabilities(prob_path, saved_hashes[key], query, len(cfg['classes']))
            if not np.allclose(probabilities, expected, rtol=2e-5, atol=2e-5):
                raise ValueError('Saved probabilities changed on checkpoint replay')
            probabilities = expected.astype(probabilities.dtype, copy=False)
        else:
            temporary = prob_path.with_suffix('.partial.npz')
            np.savez_compressed(temporary, rows=query, probabilities=probabilities)
            temporary.replace(prob_path)
        blocks[epoch] = probabilities
        prob_hashes[key] = file_hash(prob_path)
    return blocks, prob_hashes


def trajectory(data, cfg, root, name, context, train_patients, query_patients, epochs, verify):
    if np.intersect1d(train_patients, query_patients).size:
        raise ValueError('Training patients overlap the query')
    indices = data['earliest']
    groups = data['groups'][indices]
    train, query = (indices[np.isin(groups, patients)] for patients in (train_patients, query_patients))
    seed = context['model_seed']
    norm, samples = source.sampling_plan(data, train, cfg, seed)
    model = source.new_model(cfg, seed)
    context = dict(context, train_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query),
                   sample_sha256=array_hash(samples), initial_sha256=initial_state_hash(model),
                   normalization=norm.audit())
    folder = root / 'review' / name
    folder.mkdir(parents=True, exist_ok=True)
    audit_path = folder / 'audit.json'
    if audit_path.exists() or verify:
        audit = replay.read(audit_path)
        if audit['context'] != context:
            raise ValueError('Saved trajectory belongs to another context')
        history, hashes, saved = audit['history'], audit['checkpoints'], audit['probabilities']
    else:
        history, hashes = train_checkpoints(data, cfg, root, folder, name, context, samples, norm, model, epochs)
        saved = None
    if (len(history) != len(samples) or any(
            row['epoch'] != epoch or row['presentations'] != len(samples[epoch - 1])
            or row['sample_sha256'] != array_hash(samples[epoch - 1]) or not np.isfinite(row['loss'])
            for epoch, row in enumerate(history, 1))):
        raise ValueError('Training exposure, sampling or loss history changed')
    probabilities, probability_hashes = restore_predictions(
        folder, model, data, cfg, context, query, norm, epochs, hashes, saved)
    if saved is None:
        neural._save_json(audit_path, dict(context=context, history=history, checkpoints=hashes,
                                          probabilities=probability_hashes))
    pooled = {}
    for epoch, values in probabilities.items():
        ids, _, p = benchmark.aggregate_patients(values, data['beat_labels'][query], data['groups'][query])
        if not np.array_equal(ids, query_patients):
            raise ValueError('Inference covers unexpected patients')
        pooled[epoch] = p
    return dict(rows=query, blocks=probabilities, pooled=pooled,
                losses={str(n): history[n - 1]['loss'] for n in epochs},
                context=context, history=history, receipt=file_hash(audit_path))


def select_epochs(labels, probabilities, n_classes):
    codes = np.arange(n_classes)
    scores = {}
    for epoch, p in probabilities.items():
        if np.any(~np.isfinite(p)):
            raise ValueError('Incomplete inner validation coverage')
        scores[epoch] = float(f1_score(labels, p.argmax(1), labels=codes, average='macro', zero_division=0))
    chosen = min(scores, key=lambda epoch: (-scores[epoch], epoch))
    return chosen, {str(epoch): score for epoch, score in scores.items()}


def innermost_choice(data, cfg, root, fold, seed, outer_train, epochs, fingerprint, verify, receipts):
    inner_predictions = {epoch: np.full((len(data['labels']), len(cfg['classes'])), np.nan) for epoch in epochs}
    splits = benchmark.inner_patient_splits(data['labels'][outer_train], data['ids'][outer_train],
                                           cfg['inner_splits'], cfg['seed'] + fold)
    for split, (fit, hold) in enumerate(splits, 1):
        train, query = outer_train[fit], outer_train[hold]
        name = f'fold{fold:02d}_seed{seed}_inner{split}'
        context = dict(fingerprint=fingerprint, fold=fold, seed=seed, split=split,
                       model_seed=seed + 100 * fold + split)
        values = trajectory(data, cfg, root, name, context, train, query, epochs, verify)
        receipts[name] = values['receipt']
        for epoch, p in values['pooled'].items():
            if np.isfinite(inner_predictions[epoch][query]).any():
                raise ValueError('A patient appeared in multiple inner validation folds')
            inner_predictions[epoch][query] = p
    dated = outer_train[data['eligible'][outer_train]]
    selected, scores = select_epochs(data['labels'][dated],
                                     {epoch: p[dated] for epoch, p in inner_predictions.items()},
                                     len(cfg['classes']))
    return selected, scores


def check_baseline(data, config, fold, seed, trajectory_result, hashes):
    name = f'fold{fold:02d}_seed{seed}_supervised'
    folder = Path(config['source_root']) / 'review' / name
    audit = replay.checked_json(folder / 'audit.json', hashes[name])
    expected, actual = audit['context'], trajectory_result['context']
    for key, alias in (('fit_rows_sha256', 'train_rows_sha256'), ('query_rows_sha256', 'query_rows_sha256'),
                       ('initial_sha256', 'initial_sha256'), ('normalization', 'normalization')):
        if expected[key] != actual[alias]:
            raise ValueError(f'Baseline {key} differs in {name}')
    for before, now in zip(audit['history'], trajectory_result['history'][:8]):
        if before['sample_sha256'] != now['sample_sha256'] or not np.isclose(before['loss'], now['loss'], atol=2e-5):
            raise ValueError(f'Baseline training prefix changed in {name}')
    rows, p = trajectory_result['rows'], trajectory_result['blocks'][8]
    ref = replay.checked_probabilities(folder / 'probabilities.npz', audit['probabilities_sha256'], rows, p.shape[1])
    _, _, expected_patients = benchmark.aggregate_patients(ref.astype(p.dtype), data['beat_labels'][rows], data['groups'][rows])
    if not np.allclose(ref, p, rtol=2e-5, atol=2e-5) or not np.array_equal(
            expected_patients.argmax(1), trajectory_result['pooled'][8].argmax(1)):
        raise ValueError(f'Epoch-eight predictions differ from the frozen baseline in {name}')


def summarize(data, cfg, predictions, baseline):
    masks = dict(dated=data['eligible'], all_patients=np.ones(len(data['ids']), dtype=bool))
    classes = cfg['classes']
    result = {}
    for name, mask in masks.items():
        truth = data['labels'][mask]
        codes = np.arange(len(classes))
        draws = patient_weights(truth, codes, cfg['bootstrap_draws'], cfg['seed'])
        scores, distributions = {}, {}
        for arm in predictions:
            runs = {str(seed): p[mask] for seed, p in predictions[arm].items()}
            if any(not np.isfinite(p).all() for p in runs.values()):
                raise ValueError('Incomplete held-out prediction coverage')
            ensemble = np.mean(list(runs.values()), axis=0)
            scores[arm] = dict(ensemble=benchmark.metric_bundle(truth, ensemble, classes),
                               members={s: benchmark.metric_bundle(truth, p, classes) for s, p in runs.items()})
            distributions[arm] = weighted_f1(truth, ensemble.argmax(1), codes, draws)
        old = baseline['cohorts'][name]['scores']['supervised']
        new = scores['fixed_8']
        if (new['ensemble']['macro_f1'] != old['ensemble']['macro_f1']
                or any(rows['macro_f1'] != old['members'][s]['macro_f1'] for s, rows in new['members'].items())):
            raise ValueError('Eight-epoch ensemble or member macro F1 did not reproduce')
        contrasts = {arm: dict(delta=row['ensemble']['macro_f1'] - new['ensemble']['macro_f1'],
                               interval=np.quantile(distributions[arm] - distributions['fixed_8'],
                                                    [.025, .975]).tolist())
                     for arm, row in scores.items() if arm != 'fixed_8'}
        result[name] = dict(patients=int(mask.sum()), scores=scores, contrasts=contrasts)
    return result


def render(result, epochs):
    lines = ['# Earliest-record measured-XYZ training duration', '',
             f"Replayed {result['trajectories']} trajectories and {result['checkpoints']} model checkpoints.", '']
    if result['smoke']:
        return '\n'.join(lines + ['Inner-patient smoke only. No performance estimate.', ''])
    lines += [f"Matched all {result['original_baseline_blocks']} original eight-epoch probability blocks.",
              '', '| Cohort | Arm | Ensemble macro F1 | Member macro F1 in seed order |',
              '|---|---|---:|---|']
    for cohort, section in result['cohorts'].items():
        for arm, score in section['scores'].items():
            members = ', '.join(f"{m['macro_f1']:.6f}" for m in score['members'].values())
            lines.append(f"| {cohort} ({section['patients']}) | {arm} | {score['ensemble']['macro_f1']:.6f} | {members} |")
    lines += ['', '| Cohort | Comparison against fixed 8 | Macro F1 difference | Descriptive 95% interval |',
              '|---|---|---:|---|']
    for name, section in result['cohorts'].items():
        for arm, delta in section['contrasts'].items():
            low, high = delta['interval']
            lines.append(f"| {name} | {arm} | {delta['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ['', '| Primary cohort class | Patients | Fixed 8 F1 | Inner-selected F1 |',
              '|---|---:|---:|---:|']
    scores = result['cohorts']['dated']['scores']
    for label in result['classes']:
        fixed = scores['fixed_8']['ensemble']['classification_report'][label]
        selected = scores['inner_selected']['ensemble']['classification_report'][label]
        lines.append(f"| {label} | {int(fixed['support'])} | {fixed['f1-score']:.4f} | {selected['f1-score']:.4f} |")
    lines += ['', '| Epoch | Inner selections across 30 decisions | Mean outer training loss at checkpoint |',
              '|---|---:|---:|']
    for epoch in epochs:
        key = str(epoch)
        lines.append(f"| {epoch} | {result['selection_counts'].get(key, 0)} | {result['outer_losses'][key]:.6f} |")
    lines += ['', 'Primary comparison uses the 145 dated patients. The fixed 16- and 32-epoch checkpoints are '
              'descriptive. Every displayed ensemble gives each of the three seeds equal probability weight. '
              'The paired intervals condition on saved PTB development predictions. Inner selections share '
              'overlapping training patients.', '', '[Protocol](../../../docs/earliest_training_duration_plan.md)', '']
    return '\n'.join(lines)


def run(config_path, smoke, verify):
    config = yaml.safe_load(config_path.read_text())
    baseline, manifest, hashes = read_reference(config)
    cfg = dict(manifest['cfg'], inner_splits=config['inner_splits'])
    data = source.load_data(cfg)
    epochs = config['smoke_epochs'] if smoke else config['epochs']
    if sorted(set(epochs)) != epochs or not epochs or (not smoke and epochs[0] != cfg['target_epochs']):
        raise ValueError('Invalid fixed training durations')
    cfg['target_epochs'] = epochs[-1]
    if smoke:
        cfg.update(training_seeds=cfg['training_seeds'][:1], samples_per_epoch=config['smoke_samples_per_epoch'])
    folds = [1] if smoke else manifest['folds']
    scope = dict(config=config, settings=cfg, folds=folds, smoke=smoke, bank=data['bank'],
                 headers_sha256=data['source_headers'], paired_membership_sha256=data['pair_sha256'],
                 original_verification_sha256=baseline['verification_sha256'],
                 code_sha256=file_hash(Path(__file__)), protocol_sha256=file_hash(PROTOCOL),
                 dependencies={m.__name__: file_hash(Path(m.__file__)) for m in
                               (source, benchmark, preprocessing_study, neural)},
                 runtime=dict(numpy=np.__version__, torch=str(torch.__version__), sklearn=sklearn.__version__,
                              platform=platform.platform(), device=str(neural._device(cfg))))
    scope['fingerprint'] = neural._hash(scope)
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    with source.runner_lock(root):
        manifest_path = root / 'study_manifest.json'
        if manifest_path.exists() or verify:
            if replay.read(manifest_path) != scope:
                raise ValueError('Saved run scope changed')
        else:
            neural._save_json(manifest_path, scope)
        counts, choices, receipts, losses = Counter(), {}, {}, []
        arm_names = [f'fixed_{e}' for e in epochs] + ['inner_selected']
        predictions = {arm: {s: np.full((len(data['ids']), len(cfg['classes'])), np.nan)
                             for s in cfg['training_seeds']} for arm in arm_names}
        for fold in folds:
            train, query = np.flatnonzero(data['folds'] != fold), np.flatnonzero(data['folds'] == fold)
            if smoke:
                # Exercise selection and the final refit inside the same outer-training partition.
                fit, hold = next(benchmark.inner_patient_splits(
                    data['labels'][train], data['ids'][train], 3, cfg['seed']))
                train, query = train[fit], train[hold]
            for seed in cfg['training_seeds']:
                selected, scores = innermost_choice(data, cfg, root, fold, seed, train, epochs,
                                                    scope['fingerprint'], verify, receipts)
                name = f'fold{fold:02d}_seed{seed}_outer'
                context = dict(fingerprint=scope['fingerprint'], fold=fold, seed=seed, model_seed=seed + fold,
                               selected=selected, inner_dated_oof_macro_f1=scores)
                current = trajectory(data, cfg, root, name, context, train, query, epochs, verify)
                receipts[name] = current['receipt']
                if not smoke:
                    check_baseline(data, config, fold, seed, current, hashes)
                choices[name] = dict(selected=selected, inner_dated_oof_macro_f1=scores)
                counts[str(selected)] += 1
                losses.append(current['losses'])
                for epoch, scores in current['pooled'].items():
                    predictions[f'fixed_{epoch}'][seed][query] = scores
                predictions['inner_selected'][seed][query] = current['pooled'][selected]
            _progress(root, state='verifying' if verify else 'fold_saved', fold=fold, trajectories=len(receipts))
            print(f"{'Verified' if verify else 'Saved'} fold {fold}/{len(folds)}. {len(receipts)} trajectories.", flush=True)
        expected = len(folds) * len(cfg['training_seeds']) * (cfg['inner_splits'] + 1)
        if len(receipts) != expected:
            raise ValueError('Incomplete trajectory inventory')
        if not verify:
            _progress(root, state='fits_saved', trajectories=expected)
            return
        result = dict(complete=True, smoke=smoke, folds=folds, seeds=cfg['training_seeds'], classes=cfg['classes'],
                      trajectories=expected, checkpoints=expected * len(epochs),
                      original_baseline_blocks=0 if smoke else len(folds) * len(cfg['training_seeds']),
                      selection_counts=dict(counts), selections=choices,
                      outer_losses={str(e): float(np.mean([row[str(e)] for row in losses])) for e in epochs},
                      verification_sha256=neural._hash(receipts), fingerprint=scope['fingerprint'],
                      cohorts=None if smoke else summarize(data, cfg, predictions, baseline))
        for filename, content in {'summary.json': json.dumps(result, indent=2, allow_nan=False) + '\n',
                                  'REPORT.md': render(result, epochs)}.items():
            path = root / filename
            if path.exists() and path.read_text() != content:
                raise ValueError(f'{filename} differs from the completed result')
            if not path.exists():
                temporary = path.with_suffix('.partial')
                temporary.write_text(content)
                temporary.replace(path)
        _progress(root, state='complete_validated', trajectories=expected, checkpoints=result['checkpoints'])


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
