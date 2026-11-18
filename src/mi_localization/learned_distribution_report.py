"""Independent all-beat replay and restoration of every new and reused classifier."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits
from . import learned_distribution_study as study, learned_distribution_source as source
from . import residual_context_report as uncertainty
from .benchmark import metric_bundle, save_confusion
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _save_json
from .signal_audit import file_hash


def summarize(config):
    bank, assigned, _ = study.load_inputs(config, verify=True)
    scope = study.make_scope(config, bank); root = Path(config['output_dir'])/'complete'
    if json.loads((root/'study_manifest.json').read_text()) != scope: raise ValueError('Report scope changed')
    study.require_audit(config,scope)
    members = {(a,s): np.full((len(bank.ids),len(config['classes'])),np.nan) for a in config['arms'] for s in config['projection_seeds']}
    matched = 0
    for fold in range(1,config['outer_splits']+1):
        for seed in config['projection_seeds']:
            rows, values = study.evaluate_fold(bank,assigned,config,scope,root,seed,fold,verify=True)
            for arm,p in values.items():
                if np.isfinite(members[arm,seed][rows]).any(): raise ValueError('Duplicate outer patients')
                members[arm,seed][rows] = p
            for inner in range(config['inner_splits']+1):
                ctx = root/'review/contexts'/f'context_{fold:02d}_s{seed}_i{inner}.json'
                audit = json.loads(ctx.read_text()); state = audit['state']; d = state['base_dimensions']
                if (audit['coverage_mode'] != 'all_original_beats' or state['coverage_mode'] != 'all_original_beats'
                        or audit['dimensions'] != {a:d+(0 if a=='baseline' else config['projections']*len(config['quantiles'])) for a in config['arms']}):
                    raise ValueError('Unmatched/full-coverage feature context')
                # Audit every declared original row, including records of validation/query patients.
                record_audits = [json.loads((root/name).read_text()) for name in state['record_audit_hashes']]
                if state['source_teacher_query_labels_reproduced'] is not True or state['teacher']['encoder_frozen'] is not True:raise ValueError('Teacher restoration/freeze failed')
                record_indices = np.array([a['record_index'] for a in record_audits])
                if (len(set(record_indices)) != state['n_records']
                        or sum(a['n_beats'] for a in record_audits) != state['original_beats_covered']
                        or state['original_beats_covered'] != sum(len(bank.rows(r)) for r in record_indices)):
                    raise ValueError('Original beat coverage differs')
                if not inner and (state['n_records']!=len(bank.record_patients) or state['original_beats_covered']!=len(bank.records.waves)):
                    raise ValueError('Outer context does not cover full cohort')
                for arm in config['arms']:
                    suffixes = [f'_i{inner}_a{i}' for i in range(len(config['ridge_alphas']))] if inner else ['']
                    for suffix in suffixes:
                        name = f"{'inner' if inner else 'outer'}_{fold:02d}_{'' if arm=='baseline' else arm+'_'}s{seed}{suffix}.json"
                        directory = 'baseline_refs' if arm=='baseline' else 'fits'
                        fit = json.loads((root/'review'/directory/name).read_text())
                        if fit['context_sha256'] != file_hash(ctx): raise ValueError('Unmatched arm context')
                matched += 1
    expected = study.budget(config)
    fits = sorted((root/'review/fits').glob('*.json')); refs = sorted((root/'review/baseline_refs').glob('*.json'))
    contexts = sorted((root/'review/contexts').glob('*.json'))
    if (len(fits)!=expected['new_fits'] or len(refs)!=expected['source_ridge_contexts']
            or len(contexts)!=expected['feature_contexts'] or matched!=len(contexts)):
        raise ValueError('Incomplete/unexpected audit counts')
    metrics, means, ensemble_metrics, ensemble, counts = {}, {}, {}, {}, {}
    for arm in config['arms']:
        metrics[arm]={}; counts[arm]={str(a):0 for a in config['ridge_alphas']}
        for seed in config['projection_seeds']:
            p=members[arm,seed]
            if not np.isfinite(p).all(): raise ValueError('Missing patient predictions')
            metrics[arm][str(seed)]=metric_bundle(bank.labels,p,config['classes'])
            for fold in range(1,config['outer_splits']+1):
                decision=json.loads((root/'review'/f'selection_{fold:02d}_{arm}_s{seed}.json').read_text())
                counts[arm][str(decision['selected_alpha'])]+=1
        vals=[m['macro_f1'] for m in metrics[arm].values()]
        means[arm]=dict(macro_f1_mean=float(np.mean(vals)),macro_f1_sample_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.)
        ensemble[arm]=np.mean([members[arm,s] for s in config['projection_seeds']],axis=0)
        ensemble_metrics[arm]=metric_bundle(bank.labels,ensemble[arm],config['classes'])
        save_confusion(bank.labels,ensemble[arm].argmax(1),config['classes'],arm,root)
        for suffix,p in [(f's{s}',members[arm,s]) for s in config['projection_seeds']]+[('ensemble',ensemble[arm])]:
            frame=pd.DataFrame(dict(patient_id=bank.ids,truth=np.asarray(config['classes'])[bank.labels],prediction=np.asarray(config['classes'])[p.argmax(1)],fold=assigned))
            for i,c in enumerate(config['classes']):frame[f'probability_{c}']=p[:,i]
            frame.to_csv(root/'review'/f'patient_predictions_{arm}_{suffix}.csv',index=False)
    if ensemble_metrics['baseline']!=bank.cache.summary['ensemble_metrics']['baseline']:raise ValueError('Original baseline metrics changed')
    for seed in config['projection_seeds']:
        if metrics['baseline'][str(seed)]!=bank.cache.summary['member_metrics']['baseline'][str(seed)]:raise ValueError('Original baseline member metrics changed')
    labels=np.arange(len(config['classes']))
    weights=uncertainty.patient_weights(bank.labels,labels,config['bootstrap_draws'],config['seed'])
    draws={a:uncertainty.weighted_f1(bank.labels,p.argmax(1),labels,weights) for a,p in ensemble.items()}
    mean_draws={a:np.mean([uncertainty.weighted_f1(bank.labels,members[a,s].argmax(1),labels,weights) for s in config['projection_seeds']],axis=0) for a in config['arms']}
    pairs=[('plus_quantiles','plus_spread'),('plus_quantiles','plus_mean'),('plus_quantiles','baseline'),('plus_spread','plus_mean'),('plus_mean','baseline')]
    contrasts={f'{a}_minus_{b}':dict(primary=(a,b)==pairs[0],ensemble_delta_macro_f1=ensemble_metrics[a]['macro_f1']-ensemble_metrics[b]['macro_f1'],
                descriptive_patient_interval=np.quantile(draws[a]-draws[b],[.025,.975]).tolist(),
                mean_member_delta_macro_f1=means[a]['macro_f1_mean']-means[b]['macro_f1_mean'],
                mean_member_descriptive_interval=np.quantile(mean_draws[a]-mean_draws[b],[.025,.975]).tolist()) for a,b in pairs}
    current,_,_=study.load_inputs(config)
    if current.provenance!=scope['source']:raise ValueError('Frozen upstream source changed during verification')
    audits={str(p.relative_to(root)):file_hash(p) for p in fits+refs+contexts}
    _save_json(root/'verification_manifest.json',dict(study_fingerprint=scope['study_fingerprint'],restored_audit_hashes=audits,teacher_audit_sha256=file_hash(Path(config['output_dir'])/'teacher_audit/summary.json')))
    lines=['# Frozen learned morphology distributions','',study.CLAIM,'',
           f"All {len(bank.ids)} patients, {len(bank.record_patients)} recordings and {len(bank.records.waves):,} original beats retained. Re-encoded every original beat in all {len(contexts)} exact teacher/feature contexts, restored {len(fits)} new and {len(refs)} source classifiers including {expected['outer_models']} outer models. All {matched} matched checks passed; {len(contexts)} frozen teacher query predictions reproduced. No encoder training.",'',
           '| Procedure | Mean member macro F1 ± seed SD | Uniform seed ensemble macro F1 |','|---|---:|---:|']
    for a in config['arms']:lines.append(f"| {a} | {means[a]['macro_f1_mean']:.6f} ± {means[a]['macro_f1_sample_sd']:.6f} | {ensemble_metrics[a]['macro_f1']:.6f} |")
    lines+=['','## Prespecified contrasts','','Primary: learned quantiles minus learned mean/spread. All comparisons retained; descriptive conditional patient intervals.','','| Comparison | Ensemble difference | Paired patient 95% interval |','|---|---:|---:|']
    for name,v in contrasts.items():
        lo,hi=v['descriptive_patient_interval'];lines.append(f"| {name} | {v['ensemble_delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## Class F1','','| Class | '+' | '.join(config['arms'])+' |','|---|'+'---:|'*len(config['arms'])]
    for c in config['classes']:lines.append(f"| {c} | "+' | '.join(f"{ensemble_metrics[a]['classification_report'][c]['f1-score']:.6f}" for a in config['arms'])+' |')
    lines+=['','## Method and limits','',
            'The complete frozen recording-median combined-convolution ridge baseline is retained, including its entire standardized 3,584-column vector. Three learned-summary arms add144columns; the base prefix remains bit-identical. Plain morphology teachers are fixed in advance from the completed measured-VCG staged study, using each exact inner/outer training partition and seed. Inner teachers were early-stopped using their original inner validation; this reused selection is disclosed, not an assertion of validation-independent encoders. Outer teachers used original inner-selected epoch counts and no outer-label monitoring. No source/seed winner selection. Every stored teacher state, scope and freshly reconstructed training normalizer is checked; CPU forward inference only, no retraining.', '',
            'For every recording, encode all original beats and its own coordinate-median waveform into64learned features. Subtract the median embedding, divide by embedding-column scales fitted only from training-patient equal-record median embeddings, and project into16fixed feature-space directions. One training-only residual RMS uses equal patient, recording and within-record beat weights. Pool mean, mean/population-SD normal-quantile grids or empirical quantiles at q=.1,...,.9. Every grid is divided by sqrt(9) and receives the same fixed sqrt(base_dimension/16) factor; no second per-arm scaler. The moment basis does not assert Gaussian physiology. Equal-record patient pooling remains; feature ranks/capacities differ.', '',
            'Class-balanced ridge with intercept/Cholesky, alpha[1,100], three inner patient folds and configured-order ties. All three declared feature/teacher seeds remain in uniform probability ensembles; member means/SD and paired differences remain separate. Teacher predictions from the new all-beat embeddings reproduce the saved query probabilities/labels within fixed CPU-versus-source MPS tolerances. Independent reporting re-encodes every waveform, checks every record, rebuilds training scales/projections/summaries, restores all classifiers and repeats selections. Source raw metadata/cache/checkpoint hashes are checked read-only; no old completed scope or report is edited.', '',
            'This tests frozen learned residual summaries while preserving the full convolution base. It differs from earlier jointly trained16-beat attention bags and from the current fixed-convolution pooling; it is not a claim of a first-ever method, exact PSWE/FSW, physiological identification or guaranteed neural superiority. Supervised feature training adds historical compute; reusing its weights does not erase that cost. These64learned coordinates are not new sensors or patients, and feature-space directions do not rotate physical XYZ. Input normalization is training-only; signed/amplitude-bearing signal information and offline recording medians retain their declared interpretation.', '',
            'Repeatedly inspected PTB patients are development data. Conditional class-stratified shared patient-bootstrap intervals use fixed predictions without refits or adjustment for repeated development/overlapping training folds/multiple comparisons. No best seed, arm, class routing, threshold or outer-feature-budget selection. More beats or latent features do not add independent target patients. No independent clinical confirmation, established SOTA, demonstrated ceiling or guaranteed0.80–0.90 score. Historical .603374 selected-linear, .594098 moment ensemble and .584570 neural ensemble remain separately labeled references, not interchangeable matched controls.', '',
            'Protocol: docs/learned_distribution_plan.md. Distribution pooling motivation: https://proceedings.neurips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html. Source HYDRA attribution/license remain in src/mi_localization/_vendor/hydra. Runner publishes verified aggregates and stops; no next automatic experiment.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    result=dict(complete=True,study_fingerprint=scope['study_fingerprint'],claim_scope=study.CLAIM,**expected,
                all_models_restored=len(fits)+len(refs),new_models_restored=len(fits),source_models_restored=len(refs),outer_models_restored=expected['outer_models'],
                feature_contexts_reconstructed=len(contexts),matched_context_checks=matched,all_original_beats_replayed=True,original_cohort_unchanged=True,
                source_files_unchanged=True,baseline_metrics_exact=True,teacher_models_restored=len(contexts),teacher_query_predictions_reproduced=len(contexts),encoder_training_steps=0,n_patients=len(bank.ids),n_records=len(bank.record_patients),n_beats=len(bank.records.waves),
                projection_seeds=config['projection_seeds'],base_dimensions=d,auxiliary_dimensions=config['projections']*len(config['quantiles']),
                mean_member_metrics=means,member_metrics=metrics,ensemble_metrics=ensemble_metrics,contrasts=contrasts,selection_counts=counts,
                report_sha256=file_hash(root/'REPORT.md'),verification_manifest_sha256=file_hash(root/'verification_manifest.json'))
    _save_json(root/'summary.json',result);_progress(root,state='complete_independently_validated',**expected)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/learned_distribution_v1.yaml');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(config['torch_threads'])
    try:
        with threadpool_limits(limits=1):result=summarize(config)
        print(json.dumps({k:result[k] for k in ('complete','fits','new_models_restored','all_models_restored')}))
    except Exception as error:
        _progress(Path(config['output_dir'])/'complete',state='failed',error=str(error));raise
