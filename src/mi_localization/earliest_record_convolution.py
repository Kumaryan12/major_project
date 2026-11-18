"""Test convolution features against the frozen ResNet on the same earliest records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import benchmark, earliest_training_duration as duration, later_recording_distillation as dates
from . import model_error_comparison as replay, neural, rocket_study as rocket
from .rocket_bank import TemplateBank
from .rocket_features import FeatureTransform
from .contrastive import _progress
from .multiview_bank import array_hash
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash

CONFIG = Path('configs/earliest_record_convolution_v1.yaml')
PROTOCOL = Path('docs/earliest_record_convolution_plan.md')
ARMS = ('dct', 'combined')
CONTRASTS = (('combined', 'resnet_8'), ('combined', 'dct'), ('dct', 'resnet_8'))


def write_or_verify(path, expected, verify):
    if path.exists() or verify:
        if replay.read(path) != expected:
            raise ValueError(f'Saved {path.name} changed')
    else:
        neural._save_json(path, expected)


def prepare(config, cfg):
    duration_config = yaml.safe_load(Path(config['duration_config']).read_text())
    prior, source_scope, hashes = duration.read_reference(duration_config)
    reference = replay.checked_json(
        Path(duration_config['output_dir']) / 'complete/summary.json', config['duration_summary_sha256'])
    if (not reference['complete'] or reference['smoke'] or reference['original_baseline_blocks'] != 30
            or source_scope['cfg']['training_seeds'] != cfg['transform_seeds']
            or source_scope['cfg']['classes'] != cfg['classes']
            or source_scope['cfg']['folds_path'] != cfg['folds_path']
            or source_scope['cfg']['seed'] != cfg['split_seed']):
        raise ValueError('Expected the original earliest-record folds, classes, seeds and reference')
    data = dates.load_data(source_scope['cfg'])
    train_groups = data['groups'][data['earliest']]
    templates = np.stack([
        np.median(data['waves'][data['earliest'][train_groups == i]], axis=0)
        for i in range(len(data['ids']))
    ])
    bank = TemplateBank(templates, data['ids'], data['ids'], data['labels'],
                        dict(bank=data['bank'], earliest_rows_sha256=array_hash(data['earliest']),
                             templates_sha256=array_hash(templates), headers_sha256=data['source_headers']))
    return bank, data, prior, hashes, Path(duration_config['source_root'])


def feature_context(bank, cfg, scope, root, seed, fold, inner, train, query, verify):
    fields = rocket.expected_membership(bank, scope, train, query)
    fit_seed = seed + fold * 1000 + inner
    transform = FeatureTransform(cfg, fit_seed).fit(bank.templates[train], bank.ids[train])
    extracted = transform.transform(bank.templates[np.r_[train, query]])
    values = {arm: extracted[arm].astype(np.float64) for arm in ARMS}
    path = root / f'context_{fold:02d}_s{seed}_i{inner}.json'
    receipt = dict(fields, outer_fold=fold, inner_fold=inner, transform_seed=seed, fit_seed=fit_seed,
                   fit=transform.audit(), features_sha256={arm: array_hash(x) for arm, x in values.items()})
    write_or_verify(path, receipt, verify)
    return values, path


def classify(bank, cfg, scope, root, seed, fold, inner, arm, alpha, train, query, values, path, verify, receipts):
    p, audit = rocket.classifier_fit(bank, cfg, scope, root, seed, fold, inner, arm, alpha,
                                     train, query, values, path, verify)
    receipts[f'{fold:02d}_{seed}_{inner}_{arm}_{alpha}'] = neural._hash(audit)
    return p


def fit_fold(bank, cfg, scope, root, seed, fold, train, query, eligible, verify, receipts):
    candidates = cfg['ridge_alphas']
    inner_oof = {arm: [np.full((len(bank.ids), len(cfg['classes'])), np.nan) for _ in candidates]
                 for arm in ARMS}
    splits = benchmark.inner_patient_splits(bank.labels[train], bank.ids[train],
                                           cfg['inner_splits'], cfg['split_seed'] + fold)
    for inner, (ti, vi) in enumerate(splits, 1):
        train_idx, valid_idx = train[ti], train[vi]
        values, path = feature_context(bank, cfg, scope, root, seed, fold, inner, train_idx, valid_idx, verify)
        for arm in ARMS:
            for alpha in range(len(candidates)):
                if np.isfinite(inner_oof[arm][alpha][valid_idx]).any():
                    raise ValueError('Repeated inner held-out patient')
                inner_oof[arm][alpha][valid_idx] = classify(
                    bank, cfg, scope, root, seed, fold, inner, arm, alpha, train_idx, valid_idx,
                    values, path, verify, receipts)
    dated = train[eligible[train]]
    selections = {}
    for arm in ARMS:
        chosen, scores = duration.select_epochs(
            bank.labels[dated], {alpha: p[dated] for alpha, p in enumerate(inner_oof[arm])}, len(cfg['classes']))
        selection = dict(study_fingerprint=scope['study_fingerprint'], fold=fold, seed=seed,
                         arm=arm, inner_dated_oof_macro_f1=scores, alpha_index=chosen, alpha=candidates[chosen])
        write_or_verify(root / f'selection_{fold:02d}_s{seed}_{arm}.json', selection, verify)
        selections[arm] = chosen
    values, path = feature_context(bank, cfg, scope, root, seed, fold, 0, train, query, verify)
    return {arm: classify(bank, cfg, scope, root, seed, fold, 0, arm, selections[arm],
                          train, query, values, path, verify, receipts) for arm in ARMS}


def waveform_reference(root, data, fold, seed, train, query, hashes):
    name = f'fold{fold:02d}_seed{seed}_supervised'
    folder = root / 'review' / name
    audit = replay.checked_json(folder / 'audit.json', hashes[name])
    selected = data['earliest']
    groups = data['groups'][selected]
    fit_rows, query_rows = (selected[np.isin(groups, patients)] for patients in (train, query))
    replay.require_fields(audit['context'], dict(fit_rows_sha256=array_hash(fit_rows),
                                                 query_rows_sha256=array_hash(query_rows)))
    p = replay.checked_probabilities(folder / 'probabilities.npz', audit['probabilities_sha256'],
                                     query_rows, 6).astype(np.float32)
    ids, _, pooled = benchmark.aggregate_patients(p, data['beat_labels'][query_rows], data['groups'][query_rows])
    if not np.array_equal(ids, query):
        raise ValueError('Waveform and convolution query differs')
    return pooled


def summarize(data, cfg, predictions, prior):
    masks = dict(dated=data['eligible'], all_patients=np.ones(len(data['ids']), dtype=bool))
    classes, seeds = cfg['classes'], cfg['transform_seeds']
    codes = np.arange(len(classes))
    output = {}
    for name, mask in masks.items():
        truth = data['labels'][mask]
        weights = patient_weights(truth, codes, cfg['bootstrap_draws'], cfg['seed'])
        scores, distributions = {}, {}
        for arm, runs in predictions.items():
            members = {str(s): runs[s][mask] for s in seeds}
            if any(not np.isfinite(p).all() for p in members.values()):
                raise ValueError('Missing predictions')
            ensemble = np.mean(list(members.values()), axis=0)
            scores[arm] = dict(ensemble=benchmark.metric_bundle(truth, ensemble, classes),
                               members={s: benchmark.metric_bundle(truth, p, classes) for s, p in members.items()})
            distributions[arm] = weighted_f1(truth, ensemble.argmax(1), codes, weights)
        if scores['resnet_8'] != prior['cohorts'][name]['scores']['supervised']:
            raise ValueError('Verified waveform baseline did not reproduce')
        contrasts = {f'{a}_minus_{b}': dict(
            delta=scores[a]['ensemble']['macro_f1'] - scores[b]['ensemble']['macro_f1'],
            interval=np.quantile(distributions[a] - distributions[b], [.025, .975]).tolist())
            for a, b in CONTRASTS}
        output[name] = dict(patients=int(mask.sum()), scores=scores, contrasts=contrasts)
    return output


def report(result):
    lines = ['# Convolution features on earliest-record measured XYZ', '',
             f"Verified {result['fits']} ridge fits and {result['feature_contexts']} regenerated feature contexts.", '']
    if result['smoke']:
        return '\n'.join(lines + ['Inner-training smoke. Outer performance remains unevaluated.', ''])
    lines += ['| Population | Arm | Ensemble macro F1 | Member macro F1 in seed order |',
              '|---|---|---:|---|']
    for name, group in result['cohorts'].items():
        for arm, row in group['scores'].items():
            members = ', '.join(f"{value['macro_f1']:.6f}" for value in row['members'].values())
            lines.append(f"| {name} ({group['patients']}) | {arm} | {row['ensemble']['macro_f1']:.6f} | {members} |")
    lines += ['', '| Population | Contrast | Macro F1 difference | Descriptive 95% interval |',
              '|---|---|---:|---|']
    for name, group in result['cohorts'].items():
        for contrast, row in group['contrasts'].items():
            low, high = row['interval']
            lines.append(f"| {name} | {contrast} | {row['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ['', '| Class on primary dated cohort | Patients | ResNet F1 | DCT F1 | Combined F1 |',
              '|---|---:|---:|---:|---:|']
    stats = result['cohorts']['dated']['scores']
    for label in result['classes']:
        metrics = [stats[a]['ensemble']['classification_report'][label] for a in ('resnet_8', *ARMS)]
        scores = ' | '.join(f"{row['f1-score']:.4f}" for row in metrics)
        lines.append(f"| {label} | {int(metrics[0]['support'])} | {scores} |")
    lines += [
        '', 'The primary contrast is combined convolution versus fixed eight-epoch ResNet on the same 145 '
        'earliest-record patients. The DCT contrast controls the ridge readout and record template. '
        'No new waveforms enter these results. The 164-patient sensitivity includes deterministic fallbacks. '
        'Intervals condition on the saved predictions in repeatedly studied PTB.',
        '', '[Protocol](../../../docs/earliest_record_convolution_plan.md)', '',
    ]
    return '\n'.join(lines)


def run(path, smoke, verify):
    config = yaml.safe_load(path.read_text())
    cfg = dict(yaml.safe_load(Path(config['base_config']).read_text()), **config)
    if cfg['arms'] != list(ARMS) or cfg['ridge_alphas'] != [1.0, 100.0]:
        raise ValueError('Require the declared matched ridge contrast')
    bank, data, prior, references, source_root = prepare(config, cfg)
    torch.set_num_threads(cfg['torch_threads'])
    folds = [1] if smoke else list(range(1, cfg['outer_splits'] + 1))
    seeds = cfg['transform_seeds'][:1] if smoke else cfg['transform_seeds']
    root = Path(cfg['output_dir']) / ('smoke' if smoke else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    artifact_root = root / 'review'
    artifact_root.mkdir(exist_ok=True)
    scope = dict(config=cfg, folds=folds, seeds=seeds, smoke=smoke, bank=bank.provenance,
                 folds_sha256=file_hash(cfg['folds_path']), protocol_sha256=file_hash(PROTOCOL),
                 source_verification_sha256=prior['verification_sha256'],
                 code_sha256={m: file_hash(Path(p)) for m, p in (
                     ('study', __file__), ('rocket', rocket.__file__), ('features', rocket.features.__file__),
                     ('dates', dates.__file__))},
                 runtime=dict(torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__,
                              sklearn=sklearn.__version__, platform=platform.platform(), blas_threads=1))
    scope['study_fingerprint'] = neural._hash(scope)
    with dates.runner_lock(root):
        write_or_verify(root / 'study_manifest.json', scope, verify)
        predictions = {arm: {seed: np.full((len(bank.ids), len(cfg['classes'])), np.nan) for seed in seeds}
                       for arm in ('resnet_8', *ARMS)}
        receipts = {}
        for fold in folds:
            train, query = np.flatnonzero(data['folds'] != fold), np.flatnonzero(data['folds'] == fold)
            if smoke:
                ti, vi = next(benchmark.inner_patient_splits(
                    bank.labels[train], bank.ids[train], cfg['inner_splits'], cfg['split_seed']))
                train, query = train[ti], train[vi]
            for seed in seeds:
                outputs = fit_fold(bank, cfg, scope, artifact_root, seed, fold, train, query,
                                   data['eligible'], verify, receipts)
                for arm, p in outputs.items():
                    predictions[arm][seed][query] = p
                if not smoke:
                    predictions['resnet_8'][seed][query] = waveform_reference(
                        source_root, data, fold, seed, train, query, references)
            _progress(root, state='verifying' if verify else 'fold_saved', fold=fold, fits=len(receipts))
            print(f"{'Verified' if verify else 'Saved'} fold {fold}/{len(folds)}, {len(receipts)} ridge fits",
                  flush=True)
        groups = len(folds) * len(seeds)
        expected = groups * len(ARMS) * (cfg['inner_splits'] * len(cfg['ridge_alphas']) + 1)
        if len(receipts) != expected:
            raise ValueError('Missing declared classifier fits')
        if not verify:
            _progress(root, state='fits_saved', fits=expected)
            return
        result = dict(complete=True, smoke=smoke, folds=folds, seeds=seeds, fits=expected,
                      classes=cfg['classes'], feature_contexts=groups * (cfg['inner_splits'] + 1),
                      outer_models=groups * len(ARMS), source_resnet_blocks=0 if smoke else groups,
                      verification_sha256=neural._hash(receipts), study_fingerprint=scope['study_fingerprint'],
                      cohorts=None if smoke else summarize(data, cfg, predictions, prior))
        write_or_verify(root / 'summary.json', result, False)
        report_path = root / 'REPORT.md'
        content = report(result)
        if report_path.exists() and report_path.read_text() != content:
            raise ValueError('Completed convolution report changed')
        if not report_path.exists():
            tmp = report_path.with_suffix('.partial')
            tmp.write_text(content)
            tmp.replace(report_path)
        _progress(root, state='complete_validated', fits=expected)


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
