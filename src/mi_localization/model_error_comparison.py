"""Read-only reconstruction of two completed prediction banks and their error overlap."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from . import benchmark, multiview_bank, residual_context_report
from .neural import _hash, _save_json
from .signal_audit import file_hash


def read(path):
    return json.loads(path.read_text())


def checked_json(path, digest):
    if file_hash(path) != digest:
        raise ValueError(f'Artifact hash differs from its verified receipt: {path.name}')
    return read(path)


def read_source(root, expected_summary, fingerprint_key):
    summary = checked_json(root / 'summary.json', expected_summary)
    scope = read(root / 'study_manifest.json')
    verification = checked_json(root / 'verification_manifest.json', summary['verification_manifest_sha256'])
    fingerprint = scope[fingerprint_key]
    if (summary.get('complete') is not True or summary['study_fingerprint'] != fingerprint
            or verification[fingerprint_key] != fingerprint
            or _hash({k: v for k, v in scope.items() if k != fingerprint_key}) != fingerprint
            or file_hash(root / 'REPORT.md') != summary['report_sha256']):
        raise ValueError('The source study is missing verified completion evidence')
    return summary, scope, verification


def require_fields(document, expected):
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError('Prediction or selection context does not match the declared study')


def checked_probabilities(path, digest, rows, n_classes):
    if file_hash(path) != digest:
        raise ValueError(f'Probability file hash changed: {path.name}')
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved['rows'], rows):
            raise ValueError('Probability rows differ from held-out membership')
        p = saved['probabilities']
    if (p.shape != (len(rows), n_classes) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1)):
        raise ValueError('Invalid saved probabilities')
    return np.asarray(p, dtype=np.float64)


def convolution_predictions(root, scope, verification, ids, labels, folds, seeds):
    config = scope['config']
    n_classes = len(config['classes'])
    fingerprint = scope['study_fingerprint']
    hashes = verification['restored_audit_hashes']
    predictions = {seed: np.full((len(ids), n_classes), np.nan) for seed in seeds}
    protected = {}
    arr_hash = multiview_bank.array_hash
    for fold in range(1, config['outer_splits'] + 1):
        train, query = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        for seed in seeds:
            shared_name = f'review/paired_selection_{fold:02d}_s{seed}.json'
            shared = checked_json(root / shared_name, hashes[shared_name])
            expected = dict(study_fingerprint=fingerprint, outer_fold=fold, projection_seed=seed)
            require_fields(shared, expected)
            representation = max(config['representations'], key=shared['symmetric_inner_scores'].__getitem__)
            if representation != shared['selected_representation']:
                raise ValueError('Saved shared representation selection changed')
            arm = f'{representation}__linear'
            selected_name = f'review/selection_{fold:02d}_{arm}_s{seed}.json'
            selected = checked_json(root / selected_name, hashes[selected_name])
            require_fields(selected, dict(expected, arm=arm))
            scores = np.asarray(selected['inner_scores'])
            if scores.shape != (len(config['ridge_alphas']), config['inner_splits']):
                raise ValueError('Inner score coverage changed')
            candidate = int(np.argmax(scores.mean(axis=1)))
            alpha = config['ridge_alphas'][candidate]
            require_fields(selected, dict(selected_candidate=candidate, selected_settings=dict(alpha=alpha, weight=0.0)))
            fit_name = f'review/fits/outer_{fold:02d}_{arm}_s{seed}.json'
            audit = checked_json(root / fit_name, hashes[fit_name])
            require_fields(audit, dict(
                expected, family='linear', representation=representation, inner_fold=0, weight=0.0,
                alpha=alpha, candidate_id=candidate, train_membership_sha256=arr_hash(ids[train]),
                query_membership_sha256=arr_hash(ids[query]), training_labels_sha256=arr_hash(labels[train]),
                n_train=len(train), n_query=len(query), patient_overlap=0))
            path = (root / fit_name).with_suffix('.npz')
            predictions[seed][query] = checked_probabilities(path, audit['probabilities_sha256'], query, n_classes)
            protected.update({shared_name: hashes[shared_name], selected_name: hashes[selected_name],
                              fit_name: hashes[fit_name], str(path.relative_to(root)): audit['probabilities_sha256']})
    return predictions, protected


def waveform_predictions(root, scope, verification, table, beat_labels, ids, labels, folds, seeds):
    config = scope['config']
    fingerprint = scope['fingerprint']
    hashes = verification['fit_audit_hashes']
    n_classes = len(config['classes'])
    beat_ids = table.patient_id.to_numpy(dtype=str)
    index = pd.Index(ids).get_indexer(beat_ids)
    if np.any(index < 0) or not np.array_equal(labels[index], beat_labels):
        raise ValueError('Original beat table and frozen patient labels disagree')
    beat_folds = folds[index]
    predictions = {}
    protected = {}
    arr_hash = multiview_bank.array_hash
    for seed in seeds:
        probabilities = np.full((len(beat_ids), n_classes), np.nan)
        for fold in range(1, config['outer_splits'] + 1):
            train, query = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            name = f'fold{fold:02d}_seed{seed}_scratch'
            directory = root / 'review' / name
            audit_path = directory / 'audit.json'
            audit = checked_json(audit_path, hashes[name])
            require_fields(audit['context'], dict(
                fingerprint=fingerprint, arm='scratch', fold=fold, declared_seed=seed,
                train_rows_sha256=arr_hash(train), query_rows_sha256=arr_hash(query),
                train_patient_count=int((folds != fold).sum()), query_patient_count=int((folds == fold).sum()),
                source_rows_sha256=None))
            if audit['pretrain'] is not None:
                raise ValueError('Expected the waveform-only scratch control')
            p_path = directory / 'predictions.npz'
            probabilities[query] = checked_probabilities(
                p_path, audit['files']['predictions.npz'], query, n_classes)
            protected.update({str(audit_path.relative_to(root)): hashes[name],
                              str(p_path.relative_to(root)): audit['files']['predictions.npz']})
        if not np.isfinite(probabilities).all():
            raise ValueError('Waveform predictions do not cover every target beat')
        pooled_ids, pooled_truth, pooled = benchmark.aggregate_patients(probabilities, beat_labels, beat_ids)
        if not np.array_equal(ids, pooled_ids) or not np.array_equal(labels, pooled_truth):
            raise ValueError('Waveform and convolution patient membership differs')
        predictions[seed] = pooled
    return predictions, protected


def error_overlap(truth, left, right):
    l_ok, r_ok = left == truth, right == truth
    both_wrong = ~l_ok & ~r_ok
    return {
        'patients': len(truth),
        'both_correct': int(np.count_nonzero(l_ok & r_ok)),
        'convolution_only_correct': int(np.count_nonzero(l_ok & ~r_ok)),
        'waveform_only_correct': int(np.count_nonzero(~l_ok & r_ok)),
        'both_wrong_same_prediction': int(np.count_nonzero(both_wrong & (left == right))),
        'both_wrong_different_predictions': int(np.count_nonzero(both_wrong & (left != right))),
        'both_wrong': int(np.count_nonzero(both_wrong)),
    }


def analyze(config, truth, members, expected_metrics):
    classes, seeds = config['classes'], config['seeds']
    metrics, member_metrics, components = {}, {}, {}
    for name, runs in members.items():
        if any(not np.isfinite(runs[s]).all() for s in seeds):
            raise ValueError('Incomplete held-out patient coverage')
        member_metrics[name] = {str(s): benchmark.metric_bundle(truth, runs[s], classes) for s in seeds}
        components[name] = np.mean([runs[s] for s in seeds], axis=0)
        metrics[name] = benchmark.metric_bundle(truth, components[name], classes)
        expected_ensemble, expected_members = expected_metrics[name]
        if metrics[name] != expected_ensemble or member_metrics[name] != expected_members:
            raise ValueError(f'Source metrics do not reproduce for {name}')
    p = dict(components, equal_blend=(components['convolution'] + components['waveform']) * 0.5)
    metrics['equal_blend'] = benchmark.metric_bundle(truth, p['equal_blend'], classes)
    predictions = {name: q.argmax(1) for name, q in p.items()}
    conv, wave = predictions['convolution'], predictions['waveform']
    overlap = error_overlap(truth, conv, wave)
    by_class = {
        label: error_overlap(truth[truth == i], conv[truth == i], wave[truth == i])
        for i, label in enumerate(classes)
    }
    labels = np.arange(len(classes))
    weights = residual_context_report.patient_weights(truth, labels, config['bootstrap_draws'], config['bootstrap_seed'])
    samples = {arm: residual_context_report.weighted_f1(truth, pred, labels, weights)
               for arm, pred in predictions.items()}
    contrasts = {}
    for left, right in (('waveform', 'convolution'), ('equal_blend', 'convolution'), ('equal_blend', 'waveform')):
        contrasts[f'{left}_minus_{right}'] = dict(
            macro_f1_difference=metrics[left]['macro_f1'] - metrics[right]['macro_f1'],
            descriptive_patient_interval=np.quantile(samples[left] - samples[right], [.025, .975]).tolist())
    return dict(overall_overlap=overlap, overlap_by_class=by_class, ensemble_metrics=metrics,
                component_member_metrics=member_metrics, comparisons=contrasts, blend_weights=[.5, .5])


def render(result):
    lines = [
        '# Saved convolution and waveform prediction comparison', '',
        f"Rebuilt both source ensembles across {result['n_patients']} patients, {len(result['seeds'])} seeds "
        'and all ten original held-out folds from verified prediction files.', '',
        '| Model | Patient macro F1 | Accuracy |', '|---|---:|---:|',
    ]
    for arm, metrics in result['ensemble_metrics'].items():
        lines.append(f"| {arm} | {metrics['macro_f1']:.6f} | {metrics['accuracy']:.1%} |")
    lines.extend([
        '', '| Contrast | Macro F1 difference | Descriptive paired patient 95% interval |',
        '|---|---:|---:|',
    ])
    for name, row in result['comparisons'].items():
        lo, hi = row['descriptive_patient_interval']
        lines.append(f"| {name} | {row['macro_f1_difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend([
        '', '| True class | Patients | Both correct | Convolution only | Waveform only | Both wrong |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for name, row in dict(result['overlap_by_class'], total=result['overall_overlap']).items():
        lines.append(f"| {name} | {row['patients']} | {row['both_correct']} | {row['convolution_only_correct']} | "
                     f"{row['waveform_only_correct']} | {row['both_wrong']} |")
    overlap = result['overall_overlap']
    lines.extend([
        '', f"Among the {overlap['both_wrong']} shared errors, both predicted the same wrong label for "
        f"{overlap['both_wrong_same_prediction']} patients.", '',
        'The equal blend gives each of the two seed-averaged probability vectors a fixed weight of one half. '
        'The two component procedures were chosen retrospectively from inspected PTB development results. '
        'Bootstrap intervals describe fixed predictions on these patients. This diagnostic fits no model, '
        'selects no mixture and uses no new waveform inputs. The per-class breakdown is descriptive.',
        '', '[Comparison protocol](../../docs/model_error_comparison_plan.md)', '',
    ])
    return '\n'.join(lines)


def run(config, verify):
    conv_root, wave_root = Path(config['convolution_root']), Path(config['waveform_root'])
    cs, c_scope, cv = read_source(conv_root, config['convolution_summary_sha256'], 'study_fingerprint')
    ws, w_scope, wv = read_source(wave_root, config['waveform_summary_sha256'], 'fingerprint')
    folds_file, beat_file = Path(config['folds_path']), Path(config['beats_path'])
    fold_hash, beat_hash = file_hash(folds_file), file_hash(beat_file)
    if (c_scope['source']['folds_sha256'] != fold_hash or w_scope['folds_sha256'] != fold_hash
            or w_scope['bank']['beats_sha256'] != beat_hash):
        raise ValueError('The saved predictions require the original target folds and beat table')
    classes, seeds = config['classes'], config['seeds']
    for model, seed_key in ((c_scope, 'projection_seeds'), (w_scope, 'training_seeds')):
        require_fields(model['config'], dict(classes=classes, outer_splits=config['outer_folds'], **{seed_key: seeds}))
    patients = pd.read_csv(folds_file).sort_values('patient_id')
    ids = patients.patient_id.to_numpy(dtype=str)
    labels = pd.Index(classes).get_indexer(patients.label)
    folds = patients.fold.to_numpy()
    table = pd.read_csv(beat_file)
    table = table[table.label.isin(classes)].reset_index(drop=True)
    if (len(ids) != len(np.unique(ids)) or np.any(labels < 0)
            or set(folds) != set(range(1, config['outer_folds'] + 1))
            or (len(ids), table.record.nunique(), len(table)) != (164, 391, 55863)):
        raise ValueError('Original patient, label, fold or waveform coverage differs')
    require_fields(c_scope['source'], dict(patient_ids_sha256=multiview_bank.array_hash(ids),
                                           labels_sha256=multiview_bank.array_hash(labels)))
    beat_labels = pd.Index(classes).get_indexer(table.label)
    conv, conv_hashes = convolution_predictions(conv_root, c_scope, cv, ids, labels, folds, seeds)
    wave, wave_hashes = waveform_predictions(wave_root, w_scope, wv, table, beat_labels, ids, labels, folds, seeds)
    checked = dict(convolution=conv_hashes, waveform=wave_hashes)
    findings = analyze(config, labels, dict(convolution=conv, waveform=wave), {
        'convolution': (cs['ensemble_metrics']['selected_linear'], cs['member_metrics']['selected_linear']),
        'waveform': (ws['ensemble_metrics']['scratch'], ws['member_metrics']['scratch']),
    })
    scope = dict(config=config, folds_sha256=fold_hash, beats_sha256=beat_hash,
                 source_studies=[cs['study_fingerprint'], ws['study_fingerprint']],
                 checked_source_artifact_hashes=checked,
                 code={str(Path(path)): file_hash(path) for path in (
                     __file__, benchmark.__file__, residual_context_report.__file__)})
    result = dict(complete=True, diagnostic_only=True, model_fits=0, n_patients=len(ids),
                  n_records=int(table.record.nunique()), n_beats=len(table),
                  seeds=seeds, evaluated_folds=sorted(map(int, set(folds))),
                  comparison_fingerprint=_hash(scope), **findings)
    root = Path(config['output_dir'])
    summary, report, manifest = (root / name for name in ('summary.json', 'REPORT.md', 'study_manifest.json'))
    text = render(result)
    if verify or summary.exists():
        if read(summary) != result or read(manifest) != scope or report.read_text() != text:
            raise ValueError('Saved diagnostic does not reproduce')
    else:
        root.mkdir(parents=True, exist_ok=True)
        _save_json(manifest, scope)
        report.write_text(text, encoding='utf-8')
        _save_json(summary, result)
    print(json.dumps({'verified': verify, 'patients': len(ids), 'model_fits': 0,
                      'checked_source_artifacts': sum(map(len, checked.values()))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/model_error_comparison_v1.yaml')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(yaml.safe_load(Path(args.config).read_text()), args.verify)


if __name__ == '__main__':
    main()
