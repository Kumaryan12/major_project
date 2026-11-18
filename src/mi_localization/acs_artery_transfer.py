"""Evaluate fixed ACS-ECG source encoders on the original measured-XYZ patient folds."""
import argparse
from pathlib import Path

import numpy as np
import sklearn
import torch
import yaml

from . import acs_artery_pretraining as source, acs_ecg_bank as bank, benchmark, model_error_comparison
from . import neural, preprocessing_study, regularization, residual_context_report as intervals
from . import supervised_source_train, supervised_transfer, transfer
from .contrastive import _progress, initial_state_hash
from .signal_audit import file_hash

ARMS = ('scratch', 'omi', 'omi_artery')


def source_encoders(config, root, smoke, target):
    _, receipt = bank.load(config, smoke)
    summary = bank.read(root / 'source_summary.json')
    manifest = bank.read(root / 'source_manifest.json')
    if (summary['complete'] is not True or summary['matched_sampling'] is not True or summary['models'] != 2
            or summary['smoke'] is not smoke or summary['bank_receipt_sha256'] != neural._hash(receipt)
            or summary['source_manifest_sha256'] != file_hash(root / 'source_manifest.json')
            or manifest['target_settings'] != target or set(summary['fits']) != set(source.OBJECTIVES)):
        raise ValueError('Verified matched source encoders required')
    encoders = {'scratch': None}
    for objective in source.OBJECTIVES:
        evidence = summary['fits'][objective]
        folder = root / 'review' / 'source' / objective
        if (file_hash(folder / 'encoder.pt') != evidence['encoder_file_sha256']
                or file_hash(folder / 'audit.json') != evidence['audit_file_sha256']):
            raise ValueError('Source export differs from the verified pair')
        state = torch.load(folder / 'encoder.pt', map_location='cpu', weights_only=True)
        audit = bank.read(folder / 'audit.json')
        if (state['context'] != audit['context'] or state['context']['scope_sha256'] != neural._hash(manifest)
                or any(file_hash(folder / filename) != audit[key] for filename, key in
                       [('model.pt', 'model_sha256'), ('probabilities.npz', 'predictions_sha256')])):
            raise ValueError('Source model, predictions or export context changed')
        model = source.initial_model(config, target)
        model.encoder.load_state_dict(state['model'], strict=True)
        if initial_state_hash(model.encoder) != evidence['encoder_weights_sha256']:
            raise ValueError('Exported source encoder weights differ')
        encoders[objective] = state['model']
    return encoders, summary


def target_fit(settings, data, context, arm, encoder, train, query, root, verify):
    name = f"fold{context['fold']:02d}_seed{context['seed']}_{arm}"
    folder = root / 'review' / 'target' / name
    folder.mkdir(parents=True, exist_ok=True)
    audit_path, model_path, p_path = (folder / n for n in ('audit.json', 'model.pt', 'predictions.npz'))
    logical = 'scratch' if arm == 'scratch' else 'pretrained'
    waves, labels, groups = data['waves'], data['y'], data['groups']
    epochs, seed = settings['target_epochs'], context['model_seed']
    expected_hash = None if arm == 'scratch' else data['source_evidence']['fits'][arm]['encoder_weights_sha256']
    fit_scope = dict(context, arm=arm, source_encoder_weights_sha256=expected_hash)
    existing = audit_path.is_file()
    if verify and not existing:
        raise ValueError(f'Required closed target fit is missing: {name}')
    if existing:
        audit = bank.read(audit_path)
        if (audit['scope'] != fit_scope or file_hash(model_path) != audit['model_sha256']
                or file_hash(p_path) != audit['probabilities_sha256']):
            raise ValueError('Target checkpoint provenance changed')
        saved = torch.load(model_path, map_location='cpu', weights_only=True)
        if saved['scope'] != fit_scope:
            raise ValueError('Target checkpoint context changed')
        model = transfer.initial_model(settings, seed, logical, encoder)
        model.load_state_dict(saved['state'], strict=True)
        model.to(neural._device(settings))
        fit = audit['fit']
        norm = supervised_transfer.restore_norm(fit['normalization'], len(train))
    else:
        def progress(row):
            _progress(root, state='target_training', arm=arm, fold=context['fold'], seed=context['seed'], epoch=row['epoch'])

        model, norm, fit = transfer.fit_transfer(
            waves, labels, groups, train, None, epochs, settings, seed, logical, encoder, progress)
    transfer.validate_fit(fit, settings, seed, logical, settings['learning_rate'], len(train),
                          len(np.unique(groups[train])), 0, expected_hash, epochs)
    if fit['normalization'] != supervised_transfer.expected_norm(data, train):
        raise ValueError('Measured XYZ normalizer differs from the exact target train partition')
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Nonfinite target model checkpoint')
    p = preprocessing_study.predict_waves(model, norm, waves, labels, query, settings)
    if existing:
        with np.load(p_path, allow_pickle=False) as saved_p:
            if (not np.array_equal(saved_p['rows'], query)
                    or not np.allclose(saved_p['probabilities'], p, atol=2e-5, rtol=2e-5)):
                raise ValueError('Held-out measured XYZ predictions differ during replay')
            p = saved_p['probabilities']
    else:
        supervised_source_train.save_checkpoint(model_path, dict(scope=fit_scope, state=model.state_dict()))
        partial = p_path.with_suffix('.partial.npz')
        np.savez_compressed(partial, rows=query, probabilities=p)
        partial.replace(p_path)
        neural._save_json(audit_path, dict(scope=fit_scope, fit=fit, model_sha256=file_hash(model_path),
                                          probabilities_sha256=file_hash(p_path)))
    return fit, p, name, file_hash(audit_path)


def summarize(root, scope, data, predictions, common):
    cfg = scope['target_settings']
    seeds, classes = cfg['training_seeds'], cfg['classes']
    members, ensembles, means, draws = {}, {}, {}, {}
    truth, ids = None, None
    for arm, runs in predictions.items():
        pooled = []
        for seed in seeds:
            p = runs[seed]
            if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
                raise ValueError('Held-out beat predictions are incomplete')
            got_ids, got_truth, patient_p = benchmark.aggregate_patients(p, data['y'], data['groups'])
            if ids is not None and (not np.array_equal(ids, got_ids) or not np.array_equal(truth, got_truth)):
                raise ValueError('Patient pairing changed')
            ids, truth = got_ids, got_truth
            pooled.append(patient_p)
        members[arm] = {str(s): benchmark.metric_bundle(truth, p, classes) for s, p in zip(seeds, pooled)}
        means[arm] = float(np.mean([m['macro_f1'] for m in members[arm].values()]))
        ensembles[arm] = np.mean(pooled, axis=0)
    codes = np.arange(len(classes))
    weights = intervals.patient_weights(truth, codes, cfg['bootstrap_draws'], cfg['seed'])
    metrics = {arm: benchmark.metric_bundle(truth, p, classes) for arm, p in ensembles.items()}
    draws = {arm: intervals.weighted_f1(truth, p.argmax(1), codes, weights) for arm, p in ensembles.items()}
    pairs = (('omi_artery', 'omi'), ('omi_artery', 'scratch'), ('omi', 'scratch'))
    contrasts = {f'{a}_minus_{b}': dict(delta_macro_f1=metrics[a]['macro_f1'] - metrics[b]['macro_f1'],
                                       descriptive_interval=np.quantile(draws[a] - draws[b], [.025, .975]).tolist())
                 for a, b in pairs}
    baseline = scope['waveform_reference_macro_f1']
    lines = [
        '# ACS-ECG artery-supervised transfer to measured Frank XYZ', '',
        f"All {common['restored_target_models']} models across the ten frozen folds and three seeds restored. "
        f"{len(ids)} patients, {data['bank']['n_records']} recordings and {len(data['y'])} beats in all arms.", '',
        '| Arm | Mean over seeds | Three-seed ensemble macro F1 |', '|---|---:|---:|',
    ]
    for arm in ARMS:
        lines.append(f"| {arm} | {means[arm]:.6f} | {metrics[arm]['macro_f1']:.6f} |")
    lines.extend(['', 'Primary contrast is OMI plus artery versus OMI-only pretraining.', '',
                  '| Contrast | Ensemble difference | Descriptive paired patient 95% interval |',
                  '|---|---:|---|'])
    for name, result in contrasts.items():
        lo, hi = result['descriptive_interval']
        lines.append(f"| {name} | {result['delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(['', '| Class | Scratch F1 | OMI F1 | OMI + artery F1 |', '|---|---:|---:|---:|'])
    for c in classes:
        values = [metrics[arm]['classification_report'][c]['f1-score'] for arm in ARMS]
        lines.append(f'| {c} | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    lines.extend([
        '', f"Historical PTB development references are {baseline:.6f} for the original waveform and "
        f"{scope['convolution_reference_macro_f1']:.6f} for selected linear convolution.",
        '', 'Both source arms use identical ACS-ECG inputs, initialization and training exposures. Only the '
        'encoder weights transfer. Each target arm uses the same new six-class head, exact target-train '
        'normalization and fixed eight-epoch budget. Source artery labels stay in ACS-ECG, and the target receives '
        'measured XYZ alone at inference.', '',
        'PTB is repeatedly studied development data. These conditional intervals describe predictions from '
        'two fixed source fits and three target seeds. The recording-copy screen describes waveform overlap '
        'evidence. Cross-dataset patient identity remains unresolved.', '',
        '[Protocol](../../../docs/acs_artery_transfer_plan.md)', '',
    ])
    report = root / 'REPORT.md'
    temporary = report.with_suffix('.partial.md')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    if report.exists() and report.read_bytes() != temporary.read_bytes():
        raise ValueError('Verified target report does not reproduce')
    temporary.replace(report)
    result = dict(**common, n_patients=len(ids), n_records=data['bank']['n_records'], n_beats=len(data['y']),
                  seeds=seeds, folds=list(range(1, cfg['outer_splits']+1)), ensemble_metrics=metrics,
                  member_metrics=members, mean_member_macro_f1=means, contrasts=contrasts,
                  primary_contrast='omi_artery_minus_omi', source_evidence=data['source_evidence'],
                  report_sha256=file_hash(report))
    summary_path = root / 'summary.json'
    if summary_path.exists() and bank.read(summary_path) != result:
        raise ValueError('Verified target summary does not reproduce')
    neural._save_json(summary_path, result)


def run(config, smoke=False, verify=False):
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    cfg = yaml.safe_load(Path(config['target_config']).read_text())
    encoders, source_evidence = source_encoders(config, root, smoke, cfg)
    reference_cfg = yaml.safe_load(Path(config['target_reference']).read_text())
    old_summary, reference, _ = model_error_comparison.read_source(
        Path(reference_cfg['waveform_root']), reference_cfg['waveform_summary_sha256'], 'fingerprint')
    conv, _, _ = model_error_comparison.read_source(
        Path(reference_cfg['convolution_root']), reference_cfg['convolution_summary_sha256'], 'study_fingerprint')
    cfg = dict(cfg, max_epochs=cfg['target_epochs'])
    if cfg != reference['config'] or file_hash(cfg['folds_path']) != reference['folds_sha256']:
        raise ValueError('Require the historical measured waveform protocol and original frozen folds')
    waves, labels, groups, target_bank = regularization.load_bank(cfg)
    if target_bank != reference['bank'] or (target_bank['n_patients'], target_bank['n_records'], target_bank['n_beats']) != (164, 391, 55863):
        raise ValueError('Original target waveform membership or contents changed')
    assigned = benchmark.load_or_create_folds(labels, groups, cfg).set_index('patient_id').loc[groups, 'fold'].to_numpy()
    if set(assigned) != set(range(1, cfg['outer_splits'] + 1)):
        raise ValueError('Frozen fold coverage differs')
    if smoke:
        cfg.update(target_epochs=1, max_epochs=1, samples_per_epoch=32, batch_size=16)
    paths = [Path(__file__)] + [Path(m.__file__) for m in (
        bank, source, benchmark, model_error_comparison, transfer, supervised_transfer,
        supervised_source_train, neural, regularization, preprocessing_study, intervals)]
    scope = dict(config=config, target_settings=cfg, data=target_bank, smoke=smoke,
                 source_summary_sha256=file_hash(root / 'source_summary.json'), target_reference_sha256=file_hash(config['target_reference']),
                 folds_sha256=reference['folds_sha256'],
                 waveform_reference_macro_f1=old_summary['ensemble_metrics']['scratch']['macro_f1'],
                 convolution_reference_macro_f1=conv['ensemble_metrics']['selected_linear']['macro_f1'],
                 runtime=dict(numpy=np.__version__, torch=str(torch.__version__), sklearn=sklearn.__version__,
                              device=str(neural._device(cfg))),
                 code={p.name: file_hash(p) for p in paths})
    scope['fingerprint'] = neural._hash(scope)
    path = root / 'study_manifest.json'
    if (path.is_file() or verify) and bank.read(path) != scope:
        raise ValueError('Frozen target study scope changed')
    if not path.is_file():
        neural._save_json(path, scope)
    data = dict(waves=waves, y=labels, groups=groups, bank=target_bank, norm_memo={}, source_evidence=source_evidence)
    seeds = cfg['training_seeds'][:1] if smoke else cfg['training_seeds']
    folds = [1] if smoke else range(1, cfg['outer_splits'] + 1)
    predictions = {a: {s: np.full((len(labels), len(cfg['classes'])), np.nan) for s in seeds}
                   for a in ARMS} if verify and not smoke else None
    receipts = {}
    total = len(seeds) * len(folds) * len(ARMS)
    matched = 0
    for fold in folds:
        train = np.flatnonzero(assigned != fold)
        if smoke:
            ti, vi = next(benchmark.inner_patient_splits(labels[train], groups[train], 3, cfg['seed'] + fold * 1000))
            train, query = train[ti], train[vi]
        else:
            query = np.flatnonzero(assigned == fold)
        if set(groups[train]) & set(groups[query]):
            raise ValueError('Target fit and query patients overlap')
        for seed in seeds:
            context = dict(study_fingerprint=scope['fingerprint'], fold=fold, seed=seed, model_seed=seed+fold,
                           train_rows_sha256=supervised_transfer.rows_hash(train),
                           query_rows_sha256=supervised_transfer.rows_hash(query))
            fits = []
            for arm in ARMS:
                fit, p, name, receipt = target_fit(cfg, data, context, arm, encoders[arm], train, query, root, verify)
                fits.append(fit)
                receipts[name] = receipt
                if predictions is not None:
                    if np.isfinite(predictions[arm][seed][query]).any():
                        raise ValueError('Repeated target query rows')
                    predictions[arm][seed][query] = p
                _progress(root, state='target_replaying' if verify else 'target_fit_saved', fold=fold, seed=seed,
                          arm=arm, done=len(receipts), total=total)
            supervised_transfer.check_matched(fits)
            matched += 1
            print(f"{'Replayed' if verify else 'Saved'} target models {len(receipts)}/{total}, fold {fold}, seed {seed}", flush=True)
    if len(receipts) != total or len(list((root / 'review' / 'target').glob('*/audit.json'))) != total:
        raise ValueError('Target fit coverage differs from the declared scope')
    if verify:
        evidence = root / 'verification_manifest.json'
        neural._save_json(evidence, dict(fingerprint=scope['fingerprint'], fit_audit_hashes=receipts))
        common = dict(complete=True, restored_target_models=total, matched_fold_seed_groups=matched,
                      outer_test_patients_used=not smoke, study_fingerprint=scope['fingerprint'],
                      verification_manifest_sha256=file_hash(evidence))
        if smoke:
            neural._save_json(root / 'summary.json', common)
        else:
            summarize(root, scope, data, predictions, common)
        _progress(root, state='complete_verified', done=total, total=total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/acs_artery_transfer_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.smoke, args.verify)
