"""Compare matched whole-beat, position-shuffled and R-aligned convolution pooling."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import benchmark, convolution_distribution_source as source, distribution_kernel_model as kernel
from . import distribution_score_correction as records, model_error_comparison as historical
from . import r_relative_convolution_features as features, residual_context_report as bootstrap
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

ARMS = ('baseline', *features.ARMS)
PRIMARY = ('aligned', 'whole_beat')


def context(bank, config, scope, root, seed, fold, inner, train, query, verify, choices=None):
    original = source.context(bank, seed, fold, inner, train, query)
    transform, base, original_path = original
    rows = np.r_[train, query]
    blocks, record_rows, kernel_ids, permutations = features.patient_blocks(bank, transform, rows, seed, config)
    n_train, n_classes = len(train), len(config['classes'])
    expected_width = 2 * len(config['dilations']) * len(config['windows_ms']) * config['kernels_per_window'] * 2
    if (base.shape != (len(rows), 3584) or
            any(x.shape != (len(rows), expected_width) for x in blocks.values())):
        raise ValueError('Baseline or matched augmentation budgets differ')
    base_scaled = StandardScaler().fit(base[:n_train]).transform(base)
    source_fits, probabilities, matrix_hashes, states = {}, {}, {}, {}
    arrays = dict(query=query)
    for arm in ARMS:
        if arm != 'baseline':
            aux = blocks[arm]
            aux_scaled = StandardScaler().fit(aux[:n_train]).transform(aux)
            x = np.column_stack((base_scaled, aux_scaled))
            data = kernel.KernelData(x[:n_train], bank.labels[train], x[n_train:], n_classes)
            matrix_hashes[arm] = array_hash(x)
            states[arm] = data.kernel(0)[2]
        trials = range(len(config['ridge_alphas'])) if inner else [choices[arm]]
        for index in trials:
            if arm == 'baseline':
                p, _, path = source.ridge(bank, seed, fold, inner, index, train, query, base)
                source_fits[str(index)] = file_hash(path)
            else:
                model = kernel.KernelModel.fit(data, config['ridge_alphas'][index], 0)
                p = model.predict(data)
                arrays[f'{arm}_{index}_dual'] = model.dual
                arrays[f'{arm}_{index}_intercept'] = model.intercept
            arrays[f'{arm}_{index}_p'] = p
        probabilities[arm] = [arrays[f'{arm}_{index}_p'] for index in trials]
    scores = ({arm: [records.score(bank.labels[query], p, n_classes) for p in probabilities[arm]]
               for arm in ARMS} if inner else None)
    evidence = dict(fingerprint=scope['study_fingerprint'], seed=seed, outer_fold=fold, inner_fold=inner,
                    train_rows_sha256=array_hash(train), query_rows_sha256=array_hash(query),
                    record_rows_sha256=array_hash(record_rows), kernel_ids=kernel_ids,
                    window_permutations_sha256=array_hash(permutations), source_context_sha256=file_hash(original_path),
                    baseline_restorations=source_fits, augmented_matrix_sha256=matrix_hashes, kernel_states=states,
                    inner_scores=scores, selected_alpha_indices=choices, auxiliary_width=expected_width)
    path = root / 'review' / f'fold{fold:02d}_seed{seed}_inner{inner}.json'
    restored = records.save_record(path, evidence, arrays, verify)
    return scores if inner else {arm: restored[f'{arm}_{choices[arm]}_p'] for arm in ARMS}


def evaluate(bank, assigned, config, scope, root, seed, fold, verify, smoke):
    outer_train = np.flatnonzero(assigned != fold)
    splits = benchmark.inner_patient_splits(bank.labels[outer_train], bank.ids[outer_train],
                                            config['inner_splits'], config['split_seed'] + fold * 1000)
    trials = [context(bank, config, scope, root, seed, fold, inner,
                      outer_train[ti], outer_train[vi], verify)
              for inner, (ti, vi) in enumerate(splits, 1)]
    scores = {arm: np.asarray([t[arm] for t in trials]).T for arm in ARMS}
    choices = {arm: int(np.argmax(values.mean(axis=1))) for arm, values in scores.items()}
    source_choice = records.read(bank.source_root / f'selection_{fold:02d}_combined_s{seed}.json')
    if source_choice['selected_candidate'] != choices['baseline'] or source_choice['inner_scores'] != scores['baseline'].tolist():
        raise ValueError('Baseline inner selection did not reproduce')
    if smoke:
        return None
    query = np.flatnonzero(assigned == fold)
    return query, context(bank, config, scope, root, seed, fold, 0, outer_train, query, verify, choices)


def historical_predictions(config, bank, assigned):
    old = yaml.safe_load(Path(config['historical_reference_config']).read_text())
    root = Path(old['convolution_root'])
    summary, scope, verified = historical.read_source(root, old['convolution_summary_sha256'], 'study_fingerprint')
    if scope['source']['folds_sha256'] != file_hash(config['folds_path']) or scope['config']['projection_seeds'] != config['projection_seeds']:
        raise ValueError('Historical reference scope differs')
    historical.require_fields(scope['source'], dict(patient_ids_sha256=array_hash(bank.ids),
                                                     labels_sha256=array_hash(bank.labels)))
    p, protected = historical.convolution_predictions(
        root, scope, verified, bank.ids, bank.labels, assigned, config['projection_seeds'])
    return p, protected, summary['ensemble_metrics']['selected_linear'], summary['member_metrics']['selected_linear']


def report(root, bank, scope, assigned, members, baseline_summary):
    cfg = scope['config']
    groups = cfg['outer_splits'] * len(cfg['projection_seeds'])
    paths = sorted((root / 'review').glob('fold*_inner*.json'))
    if len(paths) != groups * (cfg['inner_splits'] + 1):
        raise ValueError('Incomplete matched context coverage')
    old_p, old_hashes, old_ensemble, old_members = historical_predictions(cfg, bank, assigned)
    members['historical_selector'] = old_p
    classes, seeds = cfg['classes'], cfg['projection_seeds']
    ensemble_metrics, seed_metrics, mean_seed, predictions = {}, {}, {}, {}
    for arm, cases in members.items():
        p = [cases[seed] for seed in seeds]
        if any(not np.isfinite(x).all() or np.any(x < 0) or not np.allclose(x.sum(1), 1) for x in p):
            raise ValueError('Incomplete held-out patient predictions')
        seed_metrics[arm] = {str(s): benchmark.metric_bundle(bank.labels, x, classes) for s, x in zip(seeds, p)}
        mean_seed[arm] = float(np.mean([v['macro_f1'] for v in seed_metrics[arm].values()]))
        q = np.mean(p, axis=0)
        ensemble_metrics[arm] = benchmark.metric_bundle(bank.labels, q, classes)
        predictions[arm] = q.argmax(1)
    if (ensemble_metrics['baseline'] != baseline_summary['ensemble_metrics']['combined']
            or seed_metrics['baseline'] != baseline_summary['member_metrics']['combined']
            or ensemble_metrics['historical_selector'] != old_ensemble or seed_metrics['historical_selector'] != old_members):
        raise ValueError('Frozen baseline or historical selector metrics did not reproduce')
    codes = np.arange(len(classes))
    weights = bootstrap.patient_weights(bank.labels, codes, cfg['bootstrap_draws'], cfg['seed'])
    draws = {a: bootstrap.weighted_f1(bank.labels, p, codes, weights) for a, p in predictions.items()}
    comparisons = (PRIMARY, ('aligned', 'shuffled'), ('aligned', 'baseline'),
                   ('aligned', 'historical_selector'), ('whole_beat', 'baseline'))
    contrasts = {}
    for a, b in comparisons:
        delta = ensemble_metrics[a]['macro_f1'] - ensemble_metrics[b]['macro_f1']
        interval = np.quantile(draws[a] - draws[b], [.025, .975]).tolist()
        contrasts[f'{a}_minus_{b}'] = dict(delta_macro_f1=delta, descriptive_patient_interval=interval)
    primary = '_minus_'.join(PRIMARY)
    evidence = root / 'verification_manifest.json'
    _save_json(evidence, dict(study_fingerprint=scope['study_fingerprint'],
                             context_audits={str(p.relative_to(root)): file_hash(p) for p in paths},
                             historical_source_artifacts=old_hashes))
    n_baseline = groups * (cfg['inner_splits'] * len(cfg['ridge_alphas']) + 1)
    beats, count = scope['n_source_beats'], scope['n_record_templates']
    lines = [
        '# R-relative convolution pooling on measured XYZ', '',
        f"Reconstructed {len(paths)} matched feature contexts, restored {n_baseline} frozen baseline "
        f"models and replayed {3*n_baseline} augmented ridge fits. "
        f"All {count} record templates from the original {beats:,} beats contribute across {len(bank.ids)} patients.", '',
        '| Procedure | Mean seed macro F1 | Three-seed ensemble macro F1 |', '|---|---:|---:|',
    ]
    for a in members:
        lines.append(f"| {a} | {mean_seed[a]:.6f} | {ensemble_metrics[a]['macro_f1']:.6f} |")
    lines.extend(['', 'The primary comparison is aligned minus the equal-width whole-beat control.', '',
                  '| Comparison | Macro F1 difference | Descriptive paired patient 95% interval |',
                  '|---|---:|---|'])
    for name, value in contrasts.items():
        lo, hi = value['descriptive_patient_interval']
        lines.append(f"| {name} | {value['delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(['', '| Class | Baseline | Whole beat | Shuffled | Aligned | Historical selector |',
                  '|---|' + '---:|' * len(members)])
    for name in classes:
        scores = [ensemble_metrics[a]['classification_report'][name]['f1-score'] for a in members]
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.6f}' for v in scores) + ' |')
    lines.extend([
        '', 'All augmented arms retain the exact 3,584-column convolution base and add 512 columns. '
        'The whole-beat control uses additional distinct short kernels and the same signed response summaries. '
        'The shuffled control permutes the same four window slots within each record. The original record '
        'templates and training-fitted convolution thresholds are reused. Each procedure standardizes its '
        'patient-level columns on training patients and selects ridge alpha in the inner folds.',
        '', 'Fixed windows are relative to the original R location. The cohort is repeatedly used PTB '
        'development data. Descriptive intervals condition on predictions and omit refits and repeated-development '
        'effects. The historical selector uses a different, previously declared recipe.',
        '', '[Protocol](../../../docs/r_relative_convolution_plan.md)', '',
    ])
    text = '\n'.join(lines)
    dest = root / 'REPORT.md'
    if dest.is_file() and dest.read_text() != text:
        raise ValueError('Completed R-relative report did not reproduce')
    temporary = dest.with_suffix('.partial.md')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(dest)
    result = dict(complete=True, study_fingerprint=scope['study_fingerprint'], restored_baseline_models=n_baseline,
                  replayed_new_models=3*n_baseline, reconstructed_contexts=len(paths), n_patients=len(bank.ids),
                  seeds=seeds, folds=list(range(1, cfg['outer_splits']+1)), n_record_templates=count, n_source_beats=beats,
                  primary_comparison=primary, ensemble_metrics=ensemble_metrics, seed_metrics=seed_metrics,
                  mean_seed_macro_f1=mean_seed, contrasts=contrasts, report_sha256=file_hash(dest),
                  verification_manifest_sha256=file_hash(evidence))
    if (root / 'summary.json').is_file() and records.read(root / 'summary.json') != result:
        raise ValueError('Completed R-relative metrics did not reproduce')
    _save_json(root / 'summary.json', result)


def run(config, smoke=False, verify=False):
    bank, assigned = source.load(config)
    features.validate_windows(config, bank.templates.shape[-1])
    original_summary = records.read(bank.source_root / 'summary.json')
    manifest = bank.source_scope['bank']['bank_manifest']
    cohort = len(bank.ids), len(bank.templates), manifest['n_beats']
    if cohort != (164, 391, 55863):
        raise ValueError('Original PTB template bank scope differs')
    module_paths = [Path(__file__), Path('scripts/run_r_relative_convolution.py')] + [
        Path(m.__file__) for m in (features, source, kernel, records, historical, benchmark, bootstrap)]
    scope = dict(config=config, protected_source=bank.provenance, n_record_templates=cohort[1],
                 n_source_beats=cohort[2], smoke=smoke, code={p.name: file_hash(p) for p in module_paths})
    scope['study_fingerprint'] = _hash(scope)
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    (root / 'review').mkdir(parents=True, exist_ok=True)
    path = root / 'study_manifest.json'
    if (path.is_file() or verify) and records.read(path) != scope:
        raise ValueError('R-relative convolution study scope changed')
    if not path.is_file():
        _save_json(path, scope)
    seeds, folds = config['projection_seeds'], range(1, config['outer_splits'] + 1)
    if smoke:
        evaluate(bank, assigned, config, scope, root, seeds[0], 1, verify, True)
        if verify:
            _save_json(root / 'summary.json', dict(
                complete=True, outer_test_patients_used=False, inner_contexts=3,
                replayed_new_models=18, restored_baseline_models=6, study_fingerprint=scope['study_fingerprint']))
        return
    members = {a: {s: np.full((len(bank.ids), len(config['classes'])), np.nan) for s in seeds} for a in ARMS}
    for fold in folds:
        for seed in seeds:
            rows, values = evaluate(bank, assigned, config, scope, root, seed, fold, verify, False)
            for a, p in values.items():
                if np.isfinite(members[a][seed][rows]).any():
                    raise ValueError('Repeated held-out patients')
                members[a][seed][rows] = p
            _progress(root, state='replaying' if verify else 'fitting', fold=fold, seed=seed)
        print(f"{'Replayed' if verify else 'Fitted'} fold {fold}/{len(folds)}", flush=True)
    if verify:
        report(root, bank, scope, assigned, members, original_summary)
        _progress(root, state='complete_verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/r_relative_convolution_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config['torch_threads'])
    with threadpool_limits(limits=1):
        run(config, args.smoke, args.verify)
