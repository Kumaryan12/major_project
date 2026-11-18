"""Evaluate frozen pre- and post-compression ResNet features with patient ridge heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.special import softmax
import sklearn
import torch
import yaml
from threadpoolctl import threadpool_limits

from . import benchmark, earliest_training_duration as duration, later_recording_distillation as source
from . import model_error_comparison as replay, neural, rocket_study as rocket
from .contrastive import _progress, initial_state_hash
from .earliest_record_convolution import write_or_verify
from .multiview_bank import array_hash
from .residual_context_report import patient_weights, weighted_f1
from .rocket_bank import TemplateBank
from .signal_audit import file_hash

CONFIG = Path('configs/frozen_patient_readout_v1.yaml')
PROTOCOL = Path('docs/frozen_patient_readout_plan.md')
HEADS = ('ridge64', 'ridge256')
ARMS = ('original', 'mean_logits', *HEADS)
CONTRASTS = (('ridge256', 'ridge64'), ('ridge256', 'original'),
             ('mean_logits', 'original'), ('ridge64', 'original'))


def encode_patient(model, waves, rows, normalization, batch_size, expected_probabilities=None):
    probability_sum = np.zeros(6, dtype=np.float64)
    logits_sum = np.zeros(6, dtype=np.float64)
    sums = {64: np.zeros(64), 256: np.zeros(256)}
    mean = np.asarray(normalization['wave_mean_'], np.float32)[:, None]
    scale = np.asarray(normalization['wave_scale_'], np.float32)[:, None]
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            chosen = rows[offset:offset + batch_size]
            x = torch.from_numpy((waves[chosen] - mean) / scale)
            bins = model.encoder[:8](x)
            embedding = model.encoder[8:](bins)
            logits = model.classifier(embedding)
            probability = logits.softmax(1).numpy()
            if (bins.shape != (len(chosen), 256) or embedding.shape != (len(chosen), 64)
                    or not torch.isfinite(bins).all() or not torch.isfinite(logits).all()):
                raise ValueError('Invalid frozen encoder output')
            sums[256] += bins.numpy().sum(axis=0, dtype=np.float64)
            sums[64] += embedding.numpy().sum(axis=0, dtype=np.float64)
            if expected_probabilities is not None and not np.allclose(
                    probability, expected_probabilities[offset:offset + len(chosen)], atol=2e-5, rtol=2e-5):
                raise ValueError('Source per-beat query probabilities failed replay')
            probability_sum += probability.sum(axis=0, dtype=np.float64)
            logits_sum += logits.numpy().sum(axis=0, dtype=np.float64)
    features = {size: value / len(rows) for size, value in sums.items()}
    head = model.classifier[-1]
    direct = features[64] @ head.weight.detach().numpy().T + head.bias.detach().numpy()
    if not np.allclose(direct, logits_sum / len(rows), atol=2e-5, rtol=2e-5):
        raise ValueError('Mean original classifier logits do not match mean feature logits')
    return features, probability_sum / len(rows), softmax(direct)


def load_inputs(config):
    declared = yaml.safe_load(Path(config['duration_config']).read_text())
    baseline, _, _ = duration.read_reference(declared)
    root = Path(config['source_root'])
    summary = replay.checked_json(root / 'summary.json', config['source_summary_sha256'])
    scope = replay.read(root / 'study_manifest.json')
    if (not summary['complete'] or summary['smoke'] or summary['trajectories'] != 120
            or summary['fingerprint'] != scope['fingerprint']
            or neural._hash({key: value for key, value in scope.items() if key != 'fingerprint'}) != scope['fingerprint']
            or scope['code_sha256'] != file_hash(Path(duration.__file__))):
        raise ValueError('Expected the pinned complete duration source')
    hashes = {p.parent.name: file_hash(p) for p in (root / 'review').glob('*/audit.json')}
    if len(hashes) != 120 or neural._hash(hashes) != summary['verification_sha256']:
        raise ValueError('Duration source receipts changed')
    settings = scope['settings']
    if settings['target_epochs'] != 32 or scope['config']['epochs'] != [8, 16, 32]:
        raise ValueError('Expected the original fixed-duration checkpoints')
    data = source.load_data(settings)
    bank = TemplateBank(None, data['ids'], data['ids'], data['labels'], data['bank'])
    return data, bank, baseline, settings, scope, hashes


def feature_context(data, bank, config, settings, source_scope, hashes, scope, root,
                    seed, fold, inner, train, query, verify):
    source_name = f'fold{fold:02d}_seed{seed}_' + (f'inner{inner}' if inner else 'outer')
    folder = Path(config['source_root']) / 'review' / source_name
    audit = replay.checked_json(folder / 'audit.json', hashes[source_name])
    selected = data['earliest']
    groups = data['groups'][selected]
    train_rows, query_rows = (selected[np.isin(groups, people)] for people in (train, query))
    if np.intersect1d(train, query).size:
        raise ValueError('Encoder fitting and query patients overlap')
    model_seed = seed + 100 * fold + inner if inner else seed + fold
    expected = dict(fingerprint=source_scope['fingerprint'], fold=fold, seed=seed, model_seed=model_seed,
                    train_rows_sha256=array_hash(train_rows), query_rows_sha256=array_hash(query_rows))
    if inner:
        expected['split'] = inner
    replay.require_fields(audit['context'], expected)
    norm, sampled = source.sampling_plan(data, train_rows, settings, model_seed)
    if norm.audit() != audit['context']['normalization'] or array_hash(sampled) != audit['context']['sample_sha256']:
        raise ValueError('Source training normalization or sampled rows changed')
    for epoch, row in enumerate(audit['history'][:8]):
        if row['sample_sha256'] != array_hash(sampled[epoch]):
            raise ValueError('Source epoch-eight sampling prefix changed')
    path = folder / 'epoch08.pt'
    if file_hash(path) != audit['checkpoints']['8']:
        raise ValueError('Pinned epoch-eight checkpoint changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['context'] != audit['context'] or checkpoint['epoch'] != 8:
        raise ValueError('Encoder checkpoint context differs')
    model = source.new_model(dict(settings, device=config['device']), model_seed)
    if initial_state_hash(model) != audit['context']['initial_sha256']:
        raise ValueError('Source initialization changed')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    saved_path = folder / 'epoch08_probabilities.npz'
    frozen = replay.checked_probabilities(saved_path, audit['probabilities']['8'], query_rows, 6).astype(np.float32)
    query_lookup = pd.Index(query_rows)
    rows = np.r_[train, query]
    before = initial_state_hash(model)
    values = {head: [] for head in HEADS}
    original, mean_logits = [], []
    for patient in rows:
        beat_rows = selected[groups == patient]
        expected_p = frozen[query_lookup.get_indexer(beat_rows)] if patient in query else None
        features, p, logits = encode_patient(
            model, data['waves'], beat_rows, norm.audit(), config['batch_size'], expected_p)
        values['ridge64'].append(features[64])
        values['ridge256'].append(features[256])
        original.append(p)
        mean_logits.append(logits)
    values = {key: np.stack(value) for key, value in values.items()}
    original, mean_logits = np.stack(original), np.stack(mean_logits)
    ids, _, pooled = benchmark.aggregate_patients(frozen, data['beat_labels'][query_rows], data['groups'][query_rows])
    if (not np.array_equal(ids, query) or not np.allclose(original[len(train):], pooled, atol=2e-5, rtol=2e-5)
            or initial_state_hash(model) != before):
        raise ValueError('Frozen encoder predictions or parameters changed')
    original[len(train):] = pooled
    expected = rocket.expected_membership(bank, scope, train, query)
    receipt = dict(expected, seed=seed, fold=fold, inner=inner,
                   source_audit_sha256=hashes[source_name], source_checkpoint_sha256=audit['checkpoints']['8'],
                   source_query_probability_sha256=audit['probabilities']['8'], encoder_state_sha256=before,
                   feature_sha256={head: array_hash(x) for head, x in values.items()},
                   original_sha256=array_hash(original), mean_logits_sha256=array_hash(mean_logits))
    context_path = root / f'features_{fold:02d}_s{seed}_i{inner}.json'
    write_or_verify(context_path, receipt, verify)
    return values, original[len(train):], mean_logits[len(train):], context_path


def classifier(bank, config, scope, root, seed, fold, inner, head, alpha, train, query,
               values, context, verify, receipts):
    p, audit = rocket.classifier_fit(bank, config, scope, root, seed, fold, inner, head, alpha,
                                     train, query, values, context, verify)
    receipts[f'{fold}_{seed}_{inner}_{head}_{alpha}'] = neural._hash(audit)
    return p


def evaluate_fold(data, bank, config, settings, source_scope, source_hashes, scope, root,
                  seed, fold, train, query, verify, receipts):
    inner_values = {head: [np.full((len(bank.ids), 6), np.nan) for _ in config['ridge_alphas']]
                    for head in HEADS}
    splits = benchmark.inner_patient_splits(bank.labels[train], bank.ids[train], config['inner_splits'],
                                           settings['seed'] + fold)
    for inner, (ti, vi) in enumerate(splits, 1):
        fitting, validation = train[ti], train[vi]
        values, _, _, path = feature_context(data, bank, config, settings, source_scope, source_hashes,
                                             scope, root, seed, fold, inner, fitting, validation, verify)
        for head in HEADS:
            for alpha in range(len(config['ridge_alphas'])):
                if np.isfinite(inner_values[head][alpha][validation]).any():
                    raise ValueError('Repeated inner validation patients')
                inner_values[head][alpha][validation] = classifier(
                    bank, config, scope, root, seed, fold, inner, head, alpha, fitting, validation,
                    values, path, verify, receipts)
    dated = train[data['eligible'][train]]
    choices = {}
    for head in HEADS:
        chosen, scores = duration.select_epochs(
            bank.labels[dated], {i: p[dated] for i, p in enumerate(inner_values[head])}, 6)
        decision = dict(study_fingerprint=scope['study_fingerprint'], fold=fold, seed=seed,
                        head=head, inner_macro_f1=scores, alpha_index=chosen)
        write_or_verify(root / f'selection_{fold:02d}_s{seed}_{head}.json', decision, verify)
        choices[head] = chosen
    values, original, logits, path = feature_context(
        data, bank, config, settings, source_scope, source_hashes,
        scope, root, seed, fold, 0, train, query, verify)
    outputs = dict(original=original, mean_logits=logits)
    for head in HEADS:
        outputs[head] = classifier(bank, config, scope, root, seed, fold, 0, head, choices[head],
                                   train, query, values, path, verify, receipts)
    return outputs


def summarize(data, config, predictions, baseline):
    masks = dict(dated=data['eligible'], all_patients=np.ones(len(data['ids']), dtype=bool))
    results = {}
    for name, mask in masks.items():
        truth = data['labels'][mask]
        weights = patient_weights(truth, np.arange(6), config['bootstrap_draws'], 42)
        scores, draws = {}, {}
        for arm, runs in predictions.items():
            members = {str(seed): p[mask] for seed, p in runs.items()}
            if any(not np.isfinite(p).all() for p in members.values()):
                raise ValueError('Missing held-out predictions')
            ensemble = np.mean(list(members.values()), axis=0)
            scores[arm] = dict(ensemble=benchmark.metric_bundle(truth, ensemble, config['classes']),
                               members={s: benchmark.metric_bundle(truth, p, config['classes']) for s, p in members.items()})
            draws[arm] = weighted_f1(truth, ensemble.argmax(1), np.arange(6), weights)
        if scores['original'] != baseline['cohorts'][name]['scores']['supervised']:
            raise ValueError('Original waveform reference did not reproduce exactly')
        comparisons = {f'{a}_minus_{b}': dict(
            delta=scores[a]['ensemble']['macro_f1'] - scores[b]['ensemble']['macro_f1'],
            interval=np.quantile(draws[a] - draws[b], [.025, .975]).tolist()) for a, b in CONTRASTS}
        results[name] = dict(patients=int(mask.sum()), scores=scores, comparisons=comparisons)
    return results


def render(result):
    lines = ['# Frozen patient readouts before and after ResNet compression', '',
             f"Restored {result['encoder_contexts']} epoch-eight encoders and {result['ridge_fits']} ridge heads. "
             'All feature summaries and source query predictions replayed. Encoder updates were zero.', '']
    if result['smoke']:
        return '\n'.join(lines + ['The smoke used only outer-training patients.', ''])
    lines += ['| Cohort | Readout | Ensemble patient macro F1 | Member F1 in seed order |',
              '|---|---|---:|---|']
    for name, cohort in result['cohorts'].items():
        for arm, score in cohort['scores'].items():
            members = ', '.join(f"{v['macro_f1']:.6f}" for v in score['members'].values())
            lines.append(f"| {name} ({cohort['patients']}) | {arm} | {score['ensemble']['macro_f1']:.6f} | {members} |")
    lines += ['', '| Cohort | Contrast | Macro F1 difference | Descriptive 95% interval |',
              '|---|---|---:|---|']
    for name, cohort in result['cohorts'].items():
        for contrast, row in cohort['comparisons'].items():
            low, high = row['interval']
            lines.append(f"| {name} | {contrast} | {row['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ['', '| Class in 145-patient primary cohort | Patients | Original | Mean logits | Ridge 64 | Ridge 256 |',
              '|---|---:|---:|---:|---:|---:|']
    scores = result['cohorts']['dated']['scores']
    for label in result['classes']:
        metrics = [scores[a]['ensemble']['classification_report'][label] for a in ARMS]
        row = ' | '.join(f"{m['f1-score']:.4f}" for m in metrics)
        lines.append(f"| {label} | {int(metrics[0]['support'])} | {row} |")
    lines += ['', 'The two main contrasts test 256 versus 64 ridge features and 256 ridge versus the original '
              'probability average. The original head on the mean feature controls aggregation. '
              'Every head uses the encoder for its own fitting partition, fixed at eight epochs. '
              'Three transform-seed members receive equal probability weights. The intervals condition on fixed '
              'predictions for repeatedly studied PTB patients. The primary cohort uses earliest dated records.',
              '', '[Protocol](../../../docs/frozen_patient_readout_plan.md)', '']
    return '\n'.join(lines)


def run(config_path, smoke, verify):
    config = yaml.safe_load(config_path.read_text())
    data, bank, baseline, settings, source_scope, source_hashes = load_inputs(config)
    config = dict(config, classes=settings['classes'])
    if config['device'] != 'cpu' or config['ridge_alphas'] != [1.0, 100.0]:
        raise ValueError('Expected the declared CPU head study')
    neural._seed(42, dict(config, device='cpu'))
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    review = root / 'review'
    review.mkdir(exist_ok=True)
    folds = [1] if smoke else list(range(1, 11))
    seeds = settings['training_seeds'][:1] if smoke else settings['training_seeds']
    scope = dict(config=config, smoke=smoke, folds=folds, seeds=seeds, bank=data['bank'],
                 source_fingerprint=source_scope['fingerprint'], source_verification_sha256=neural._hash(source_hashes),
                 headers_sha256=data['source_headers'], code_sha256=file_hash(Path(__file__)),
                 protocol_sha256=file_hash(PROTOCOL), folds_sha256=file_hash(settings['folds_path']),
                 dependencies={m.__name__: file_hash(Path(m.__file__)) for m in (source, duration, rocket, neural)},
                 runtime=dict(numpy=np.__version__, torch=str(torch.__version__), sklearn=sklearn.__version__,
                              platform=platform.platform(), device='cpu'))
    scope['study_fingerprint'] = neural._hash(scope)
    with source.runner_lock(root):
        write_or_verify(root / 'study_manifest.json', scope, verify)
        receipts = {}
        predictions = {arm: {s: np.full((len(bank.ids), 6), np.nan) for s in seeds} for arm in ARMS}
        contexts = 0
        for fold in folds:
            train, query = np.flatnonzero(data['folds'] != fold), np.flatnonzero(data['folds'] == fold)
            for seed in seeds:
                if smoke:
                    ti, vi = next(benchmark.inner_patient_splits(
                        bank.labels[train], bank.ids[train], config['inner_splits'], settings['seed'] + fold))
                    ti, vi = train[ti], train[vi]
                    values, _, _, path = feature_context(
                        data, bank, config, settings, source_scope, source_hashes, scope, review,
                        seed, fold, 1, ti, vi, verify)
                    for head in HEADS:
                        for alpha in range(2):
                            classifier(bank, config, scope, review, seed, fold, 1, head, alpha,
                                       ti, vi, values, path, verify, receipts)
                    contexts += 1
                else:
                    values = evaluate_fold(data, bank, config, settings, source_scope, source_hashes,
                                           scope, review, seed, fold, train, query, verify, receipts)
                    for arm, p in values.items():
                        predictions[arm][seed][query] = p
                    contexts += config['inner_splits'] + 1
            _progress(root, state='verifying' if verify else 'fitting', fold=fold, encoder_contexts=contexts,
                      ridge_fits=len(receipts))
            print(f"{'Verified' if verify else 'Saved'} fold {fold}/{len(folds)}, {contexts} encoder contexts",
                  flush=True)
        expected = 4 if smoke else 420
        if len(receipts) != expected or contexts != (1 if smoke else 120):
            raise ValueError('Incomplete study inventory')
        if not verify:
            _progress(root, state='fits_saved', encoder_contexts=contexts, ridge_fits=expected)
            return
        if neural._hash({p.parent.name: file_hash(p) for p in
                         (Path(config['source_root']) / 'review').glob('*/audit.json')}) != scope['source_verification_sha256']:
            raise ValueError('Protected source receipts changed')
        result = dict(complete=True, smoke=smoke, classes=config['classes'], folds=folds, seeds=seeds,
                      encoder_updates=0, encoder_contexts=contexts, ridge_fits=expected,
                      outer_ridge_models=0 if smoke else 60,
                      verification_sha256=neural._hash(receipts), study_fingerprint=scope['study_fingerprint'],
                      cohorts=None if smoke else summarize(data, config, predictions, baseline))
        write_or_verify(root / 'summary.json', result, False)
        text = render(result)
        report = root / 'REPORT.md'
        if report.exists() and report.read_text() != text:
            raise ValueError('Completed report changed during replay')
        if not report.exists():
            tmp = report.with_suffix('.partial')
            tmp.write_text(text)
            tmp.replace(report)
        _progress(root, state='complete_validated', encoder_contexts=contexts, ridge_fits=expected)


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
