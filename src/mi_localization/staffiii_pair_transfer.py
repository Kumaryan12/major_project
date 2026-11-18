"""Transfer both frozen STAFF III encoders to measured XYZ and replay every fit."""
import argparse
from pathlib import Path

import numpy as np
import sklearn
import torch
import yaml

from . import acs_artery_transfer as reusable, benchmark, model_error_comparison as historical, neural
from . import regularization, residual_context_report as intervals, staffiii_pair_bank as bank
from . import staffiii_pair_pretraining as source, supervised_transfer
from .contrastive import _progress, initial_state_hash
from .signal_audit import file_hash

ARMS = ('scratch', *source.ARMS)


def encoders(config, cfg, root, smoke):
    _, receipt = bank.load(config)
    manifest = root / 'source_manifest.json'
    summary = bank.read(root / 'source_summary.json')
    if (summary['complete'] is not True or summary['models'] != 2 or summary['smoke'] is not smoke
            or summary['bank_sha256'] != neural._hash(receipt)
            or summary['source_scope_sha256'] != file_hash(manifest)
            or set(summary['fits']) != set(source.ARMS)
            or not receipt['source_support_passed'] or not receipt['recording_copy_screen']['passed']):
        raise ValueError('Require matched verified source models and bank')
    result = {'scratch': None}
    for arm, evidence in summary['fits'].items():
        folder = root / 'review' / 'source' / arm
        audit = bank.read(folder / 'audit.json')
        if (file_hash(folder / 'audit.json') != evidence['audit_sha256']
                or file_hash(folder / 'model.pt') != audit['model_sha256']
                or file_hash(folder / 'probabilities.npz') != audit['probability_sha256']
                or file_hash(folder / 'encoder.pt') != evidence['encoder_file_sha256']):
            raise ValueError('Paired source evidence changed')
        state = torch.load(folder / 'encoder.pt', map_location='cpu', weights_only=True)
        if state['context'] != audit['context'] or audit['context']['fingerprint'] != neural._hash(bank.read(manifest)):
            raise ValueError('Wrong source encoder scope')
        model = source.initial_model(cfg, config['source_seed'])
        model.encoder.load_state_dict(state['state'], strict=True)
        if initial_state_hash(model.encoder) != evidence['encoder_weights_sha256']:
            raise ValueError('Source encoder hash differs')
        result[arm] = state['state']
    return result, summary


def report(root, scope, data, predictions, common):
    settings = scope['target_settings']
    seeds, classes = settings['training_seeds'], settings['classes']
    metrics, members, decisions = {}, {}, {}
    truth, ids = None, None
    for arm, seeds_p in predictions.items():
        patient_p = []
        for seed in seeds:
            p = seeds_p[seed]
            if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
                raise ValueError('Incomplete held-out beat probabilities')
            patients, y, q = benchmark.aggregate_patients(p, data['y'], data['groups'])
            if ids is not None and (not np.array_equal(patients, ids) or not np.array_equal(y, truth)):
                raise ValueError('Unexpected patient alignment')
            ids, truth = patients, y
            patient_p.append(q)
        members[arm] = {str(s): benchmark.metric_bundle(truth, p, classes) for s,p in zip(seeds,patient_p)}
        combined = np.mean(patient_p, axis=0)
        metrics[arm] = benchmark.metric_bundle(truth, combined, classes)
        decisions[arm] = combined.argmax(1)
    means = {a: float(np.mean([m['macro_f1'] for m in fits.values()])) for a,fits in members.items()}
    labels = np.arange(len(classes))
    weights = intervals.patient_weights(truth, labels, settings['bootstrap_draws'], settings['seed'])
    draws = {a: intervals.weighted_f1(truth, p, labels, weights) for a,p in decisions.items()}
    comparisons = {}
    for a,b in (('change','mean'),('change','scratch'),('mean','scratch')):
        comparisons[f'{a}_minus_{b}'] = dict(delta_macro_f1=metrics[a]['macro_f1']-metrics[b]['macro_f1'],
            descriptive_patient_interval=np.quantile(draws[a]-draws[b],[.025,.975]).tolist())
    lines = ['# Paired STAFF III source transfer to measured Frank XYZ','',
             f"Replayed {common['restored_models']} outer fits on {len(ids)} PTB patients, ten frozen outer folds and three seeds.",'',
             '| Method | Mean seed macro F1 | Three-seed ensemble macro F1 |','|---|---:|---:|']
    for a in ARMS:
        lines.append(f"| {a} | {means[a]:.6f} | {metrics[a]['macro_f1']:.6f} |")
    lines.extend(['','The primary contrast is change minus mean.','',
                  '| Contrast | Macro F1 difference | Descriptive paired 95% interval |','|---|---:|---|'])
    for name,c in comparisons.items():
        lo,hi = c['descriptive_patient_interval']
        lines.append(f"| {name} | {c['delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(['','| Class | Scratch | Mean | Change |','|---|---:|---:|---:|'])
    for c in classes:
        lines.append('| ' + ' | '.join([c] + [f"{metrics[a]['classification_report'][c]['f1-score']:.6f}" for a in ARMS])+' |')
    lines.extend(['',
                  'STAFF III native labels describe the vessel involved in an annotated balloon occlusion. '
                  'Both source arms see identical before/during ECG windows. Source ECG uses Mason–Likar limbs '
                  'and Kors-derived XYZ. The target is measured Frank XYZ with a fresh six-class head and all original beats. '
                  'This is an ECG-assisted training extension on repeatedly inspected PTB development data. '
                  'Paired intervals condition on the saved predictions. Recording-level screening leaves cross-dataset patient identity unresolved.',
                  '',f"Historical measured-XYZ references were {scope['waveform_reference_macro_f1']:.6f} for waveform "
                  f"scratch and {scope['convolution_reference_macro_f1']:.6f} for selected linear convolution.",
                  '', '[Protocol](../../../docs/staffiii_pair_transfer_plan.md)', ''])
    dest = root / 'REPORT.md'
    text = '\n'.join(lines)
    if dest.exists() and dest.read_text() != text:
        raise ValueError('Verified STAFF III target report differs')
    temp = dest.with_suffix('.partial.md')
    temp.write_text(text)
    temp.replace(dest)
    summary = dict(**common,n_patients=len(ids),n_records=data['bank']['n_records'],n_beats=len(data['y']),
                   folds=list(range(1,settings['outer_splits']+1)),seeds=seeds,ensemble_metrics=metrics,
                   member_metrics=members,mean_seed_macro_f1=means,contrasts=comparisons,primary='change_minus_mean',
                   report_sha256=file_hash(dest))
    path = root/'summary.json'
    if path.is_file() and bank.read(path) != summary:
        raise ValueError('Verified summary differs')
    neural._save_json(path, summary)


def run(config, smoke, verify):
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    settings = yaml.safe_load(Path(config['target_config']).read_text())
    fitted, sources = encoders(config, settings, root, smoke)
    ref = yaml.safe_load(Path(config['target_reference']).read_text())
    original, reference, _ = historical.read_source(Path(ref['waveform_root']),ref['waveform_summary_sha256'],'fingerprint')
    conv, _, _ = historical.read_source(Path(ref['convolution_root']),ref['convolution_summary_sha256'],'study_fingerprint')
    settings = dict(settings, max_epochs=settings['target_epochs'])
    if settings != reference['config'] or file_hash(settings['folds_path']) != reference['folds_sha256']:
        raise ValueError('Frozen target protocol changed')
    waves, labels, patients, target = regularization.load_bank(settings)
    if target != reference['bank'] or (target['n_patients'],target['n_records'],target['n_beats']) != (164,391,55863):
        raise ValueError('Original measured target bank changed')
    assignment = benchmark.load_or_create_folds(labels, patients, settings).set_index('patient_id').loc[patients,'fold'].to_numpy()
    if set(assignment) != set(range(1,settings['outer_splits']+1)):
        raise ValueError('Unexpected frozen fold coverage')
    if smoke:
        settings.update(target_epochs=1,max_epochs=1,samples_per_epoch=32,batch_size=16)
    modules = (bank, source, reusable, benchmark, neural, regularization, historical, intervals,
               supervised_transfer, reusable.preprocessing_study, reusable.transfer)
    scope = dict(config=config, target_settings=settings, smoke=smoke, data=target,
                 folds_sha256=reference['folds_sha256'], source_summary_sha256=file_hash(root/'source_summary.json'),
                 waveform_reference_macro_f1=original['ensemble_metrics']['scratch']['macro_f1'],
                 convolution_reference_macro_f1=conv['ensemble_metrics']['selected_linear']['macro_f1'],
                 code={Path(p).name:file_hash(p) for p in [__file__,*(m.__file__ for m in modules)]},
                 runtime=dict(numpy=np.__version__,sklearn=sklearn.__version__,torch=str(torch.__version__),
                              device=str(neural._device(settings))))
    scope['fingerprint'] = neural._hash(scope)
    manifest = root/'study_manifest.json'
    if (manifest.exists() or verify) and bank.read(manifest) != scope:
        raise ValueError('Paired target scope changed')
    if not manifest.exists():
        neural._save_json(manifest,scope)
    data = dict(waves=waves,y=labels,groups=patients,norm_memo={},bank=target,source_evidence=sources)
    seeds = settings['training_seeds'][:1] if smoke else settings['training_seeds']
    folds = [1] if smoke else range(1,settings['outer_splits']+1)
    p = {a:{s:np.full((len(labels),len(settings['classes'])),np.nan) for s in seeds} for a in ARMS} if verify and not smoke else None
    audits = {}
    for fold in folds:
        train = np.flatnonzero(assignment != fold)
        if smoke:
            tr,va = next(benchmark.inner_patient_splits(labels[train],patients[train],3,settings['seed']+fold*1000))
            train,query = train[tr],train[va]
        else:
            query = np.flatnonzero(assignment == fold)
        if set(patients[train]) & set(patients[query]):
            raise ValueError('Target train and query patients overlap')
        for seed in seeds:
            context = dict(study_fingerprint=scope['fingerprint'],fold=fold,seed=seed,model_seed=seed+fold,
                           train_rows_sha256=supervised_transfer.rows_hash(train),query_rows_sha256=supervised_transfer.rows_hash(query))
            fits = []
            for arm in ARMS:
                fit, values, name, audit = reusable.target_fit(settings,data,context,arm,fitted[arm],train,query,root,verify)
                fits.append(fit)
                audits[name] = audit
                if p is not None:
                    if np.isfinite(p[arm][seed][query]).any():
                        raise ValueError('Duplicate held-out prediction')
                    p[arm][seed][query] = values
            supervised_transfer.check_matched(fits)
            print(f"{'Replayed' if verify else 'Saved'} {len(audits)} target models, fold {fold}, seed {seed}",flush=True)
    expected = len(folds)*len(seeds)*len(ARMS)
    if len(audits) != expected or len(list((root/'review'/'target').glob('*/audit.json'))) != expected:
        raise ValueError('Declared target fit coverage differs')
    if verify:
        manifest = root/'verification_manifest.json'
        neural._save_json(manifest,dict(study_fingerprint=scope['fingerprint'],audits=audits))
        common = dict(complete=True,restored_models=expected,matched_groups=len(seeds)*len(folds),
                      outer_test_patients_used=not smoke,study_fingerprint=scope['fingerprint'],
                      verification_manifest_sha256=file_hash(manifest))
        if smoke:
            neural._save_json(root/'summary.json',common)
        else:
            report(root,scope,data,p,common)
        _progress(root,state='complete_verified',done=expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/staffiii_pair_transfer_v1.yaml')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()),args.smoke,args.verify)
