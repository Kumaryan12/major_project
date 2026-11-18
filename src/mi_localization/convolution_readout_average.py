"""Compare a fixed equal average of four saved linear readouts with their selector."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from . import benchmark, model_error_comparison as source, residual_context_report as intervals
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

REPRESENTATIONS = ('baseline', 'plus_mean', 'plus_spread', 'plus_quantiles')
OUTPUT = Path('results/convolution_readout_average_v1')


def fixed_predictions(root, scope, verification, ids, labels, folds, seeds):
    config = scope['config']
    hashes = verification['restored_audit_hashes']
    members = {r: {s: np.full((len(ids), len(config['classes'])), np.nan) for s in seeds}
               for r in REPRESENTATIONS}
    protected = {}
    for fold in range(1, config['outer_splits'] + 1):
        train, query = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        for seed in seeds:
            context = dict(study_fingerprint=scope['study_fingerprint'], outer_fold=fold, projection_seed=seed)
            for rep in REPRESENTATIONS:
                arm = f'{rep}__linear'
                selection_path = f'review/selection_{fold:02d}_{arm}_s{seed}.json'
                selection = source.checked_json(root / selection_path, hashes[selection_path])
                scores = np.asarray(selection['inner_scores'])
                if scores.shape != (len(config['ridge_alphas']), config['inner_splits']) or not np.isfinite(scores).all():
                    raise ValueError('Fixed readout requires all original inner scores')
                index = int(np.argmax(scores.mean(axis=1)))
                alpha = config['ridge_alphas'][index]
                source.require_fields(selection, dict(context, arm=arm, selected_candidate=index,
                                                      selected_settings=dict(alpha=alpha, weight=0.0)))
                audit_path = f'review/fits/outer_{fold:02d}_{arm}_s{seed}.json'
                audit = source.checked_json(root / audit_path, hashes[audit_path])
                source.require_fields(audit, dict(
                    context, representation=rep, family='linear', inner_fold=0, alpha=alpha,
                    candidate_id=index, weight=0.0, train_membership_sha256=array_hash(ids[train]),
                    query_membership_sha256=array_hash(ids[query]), training_labels_sha256=array_hash(labels[train]),
                    n_train=len(train), n_query=len(query), patient_overlap=0))
                probability_path = Path(audit_path).with_suffix('.npz')
                digest = audit['probabilities_sha256']
                members[rep][seed][query] = source.checked_probabilities(
                    root / probability_path, digest, query, len(config['classes']))
                protected.update({selection_path: hashes[selection_path], audit_path: hashes[audit_path],
                                  probability_path.as_posix(): digest})
    return members, protected


def render(result):
    metrics = result['ensemble_metrics']
    procedures = list(metrics)
    lines = ['# Fixed average of four linear convolution readouts', '',
             'Replayed the saved, independently verified outer probabilities for all 164 patients, '
             'ten folds and three seeds. Each readout contributes a fixed quarter within each seed. '
             'The three seed probability vectors are then averaged. This comparison uses zero new fits.', '',
             '| Procedure | Mean seed macro F1 | Ensemble macro F1 |', '|---|---:|---:|']
    for arm in procedures:
        lines.append(f"| {arm} | {result['mean_seed_macro_f1'][arm]:.6f} | {metrics[arm]['macro_f1']:.6f} |")
    lo, hi = result['descriptive_patient_interval']
    lines.extend(['', f"Equal average minus inner selector was **{result['macro_f1_difference']:+.6f}** "
                      f"with descriptive paired patient 95% interval [{lo:+.6f}, {hi:+.6f}].", '',
                  '| Class | Inner selector F1 | Equal average F1 |', '|---|---:|---:|'])
    for name in result['classes']:
        values = [metrics[a]['classification_report'][name]['f1-score'] for a in ('selected_linear', 'equal_average')]
        lines.append(f'| {name} | {values[0]:.6f} | {values[1]:.6f} |')
    lines.extend(['', 'This is a development comparison on repeatedly studied PTB patients. The interval '
                  'describes fixed predictions on this cohort and does not account for refitting or repeated study. '
                  'The source regularization and shared representation selection were reproduced from their '
                  'saved inner scores. Artifact verification covers probabilities and receipts from the earlier '
                  'model restoration. The current comparison does not refit or reload the original classifiers.', ''])
    return '\n'.join(lines)


def run(verify=False):
    config_path = Path('configs/model_error_comparison_v1.yaml')
    config = yaml.safe_load(config_path.read_text())
    root = Path(config['convolution_root'])
    summary, scope, verification = source.read_source(root, config['convolution_summary_sha256'], 'study_fingerprint')
    classes, seeds = config['classes'], config['seeds']
    source.require_fields(scope['config'], dict(classes=classes, projection_seeds=seeds,
                                                outer_splits=config['outer_folds'], representations=list(REPRESENTATIONS)))
    folds_path = Path(config['folds_path'])
    if file_hash(folds_path) != scope['source']['folds_sha256']:
        raise ValueError('Frozen fold hash changed')
    records = pd.read_csv(folds_path).sort_values('patient_id')
    ids, folds = records.patient_id.to_numpy(dtype=str), records.fold.to_numpy()
    labels = pd.Index(classes).get_indexer(records.label)
    if len(ids) != 164 or len(set(ids)) != len(ids) or (labels < 0).any() or set(folds) != set(range(1, 11)):
        raise ValueError('Original patient coverage changed')
    source.require_fields(scope['source'], dict(patient_ids_sha256=array_hash(ids), labels_sha256=array_hash(labels)))
    members, fixed_hashes = fixed_predictions(root, scope, verification, ids, labels, folds, seeds)
    selected, selected_hashes = source.convolution_predictions(root, scope, verification, ids, labels, folds, seeds)
    members['selected_linear'] = selected
    members['equal_average'] = {s: np.mean([members[r][s] for r in REPRESENTATIONS], axis=0) for s in seeds}
    if any(not np.isfinite(p).all() for runs in members.values() for p in runs.values()):
        raise ValueError('Held-out probabilities are incomplete')
    ensembles = {a: np.mean([runs[s] for s in seeds], axis=0) for a, runs in members.items()}
    metrics = {a: benchmark.metric_bundle(labels, p, classes) for a, p in ensembles.items()}
    seed_metrics = {a: {str(s): benchmark.metric_bundle(labels, p, classes) for s, p in runs.items()}
                    for a, runs in members.items()}
    for arm in (*REPRESENTATIONS, 'selected_linear'):
        key = arm + '__linear' if arm in REPRESENTATIONS else arm
        if metrics[arm] != summary['ensemble_metrics'][key] or seed_metrics[arm] != summary['member_metrics'][key]:
            raise ValueError(f'Source scores changed for {arm}')
    left, right = 'equal_average', 'selected_linear'
    codes = np.arange(len(classes))
    weights = intervals.patient_weights(labels, codes, config['bootstrap_draws'], config['bootstrap_seed'])
    draws = [intervals.weighted_f1(labels, ensembles[a].argmax(1), codes, weights) for a in (left, right)]
    manifest = dict(config_sha256=file_hash(config_path), folds_sha256=file_hash(folds_path),
                    source_fingerprint=scope['study_fingerprint'], source_summary_sha256=config['convolution_summary_sha256'],
                    protected_artifacts={**fixed_hashes, **selected_hashes},
                    code_sha256={Path(x.__file__).name: file_hash(x.__file__) for x in (source, intervals, benchmark)},
                    implementation_sha256=file_hash(__file__))
    result = dict(complete=True, model_fits=0, n_patients=len(ids), classes=classes, seeds=seeds,
                  folds=sorted(map(int, set(folds))), n_fixed_outer_models=len(seeds) * len(REPRESENTATIONS) * len(set(folds)),
                  primary_contrast='equal_average_minus_selected_linear', fingerprint=_hash(manifest),
                  ensemble_metrics=metrics, seed_metrics=seed_metrics,
                  mean_seed_macro_f1={a: float(np.mean([m['macro_f1'] for m in runs.values()]))
                                      for a, runs in seed_metrics.items()},
                  macro_f1_difference=metrics[left]['macro_f1'] - metrics[right]['macro_f1'],
                  descriptive_patient_interval=np.quantile(draws[0] - draws[1], [.025, .975]).tolist())
    manifest_path, report_path, summary_path = (OUTPUT / n for n in ('study_manifest.json', 'REPORT.md', 'summary.json'))
    report = render(result)
    if verify or summary_path.exists():
        if (source.read(manifest_path) != manifest or source.read(summary_path) != result
                or report_path.read_text() != report):
            raise ValueError('Fixed average report did not reproduce')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        _save_json(manifest_path, manifest)
        report_path.write_text(report, encoding='utf-8')
        _save_json(summary_path, result)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    with threadpool_limits(limits=1):
        run(parser.parse_args().verify)
