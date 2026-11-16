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
from . import convolution_distribution_study as study, convolution_distribution_source as source
from . import residual_context_report as uncertainty
from .benchmark import metric_bundle, save_confusion
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _save_json
from .signal_audit import file_hash


def summarize(config):
    bank, assigned = study.load_inputs(config, verify=True)
    scope = study.make_scope(config, bank); root = Path(config['output_dir'])/'complete'
    if json.loads((root/'study_manifest.json').read_text()) != scope: raise ValueError('Report scope changed')
    receipt = source.verify_baseline(bank, assigned, config, write=False)
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
                record_indices = np.array([a['record_index'] for a in record_audits])
                if (len(set(record_indices)) != state['n_records'] or any(a['mode']!='all_original_beats' for a in record_audits)
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
    if (any(array_hash(members['baseline',s])!=receipt['member_probability_hashes'][str(s)] for s in config['projection_seeds'])
            or array_hash(ensemble['baseline'])!=receipt['ensemble_probability_sha256']
            or ensemble_metrics['baseline']['macro_f1']!=receipt['ensemble_macro_f1']):raise ValueError('Frozen baseline not reproduced exactly')
    labels=np.arange(len(config['classes']))
    weights=uncertainty.patient_weights(bank.labels,labels,config['bootstrap_draws'],config['seed'])
    draws={a:uncertainty.weighted_f1(bank.labels,p.argmax(1),labels,weights) for a,p in ensemble.items()}
    mean_draws={a:np.mean([uncertainty.weighted_f1(bank.labels,members[a,s].argmax(1),labels,weights) for s in config['projection_seeds']],axis=0) for a in config['arms']}
    pairs=[('plus_quantiles','plus_spread'),('plus_quantiles','plus_mean'),('plus_quantiles','baseline'),('plus_spread','plus_mean'),('plus_mean','baseline')]
    contrasts={f'{a}_minus_{b}':dict(primary=(a,b)==pairs[0],ensemble_delta_macro_f1=ensemble_metrics[a]['macro_f1']-ensemble_metrics[b]['macro_f1'],
                descriptive_patient_interval=np.quantile(draws[a]-draws[b],[.025,.975]).tolist(),
                mean_member_delta_macro_f1=means[a]['macro_f1_mean']-means[b]['macro_f1_mean'],
                mean_member_descriptive_interval=np.quantile(mean_draws[a]-mean_draws[b],[.025,.975]).tolist()) for a,b in pairs}
    if source.protected_files(bank.source_config)!=bank.provenance['protected_files']:raise ValueError('Frozen source changed')
    audits={str(p.relative_to(root)):file_hash(p) for p in fits+refs+contexts}
    _save_json(root/'verification_manifest.json',dict(study_fingerprint=scope['study_fingerprint'],restored_audit_hashes=audits,source_preflight_sha256=file_hash(Path(config['output_dir'])/'source_preflight/summary.json')))
    lines=['# Intact-base all-beat convolution distributions','',study.CLAIM,'',
           f"Retained {len(bank.ids)} patients, {len(bank.record_patients)} recordings and {len(bank.records.waves):,} original beats. Replayed all original beats in {len(contexts)} exact training contexts, restored {len(fits)} new classifiers and {len(refs)} frozen baseline classifiers, including {expected['outer_models']} outer models. All {matched} matched checks passed. The baseline probabilities reproduce the frozen source exactly.",'',
           '| Procedure | Mean member macro F1 ± seed SD | Uniform seed-ensemble macro F1 |','|---|---:|---:|']
    for a in config['arms']:lines.append(f"| {a} | {means[a]['macro_f1_mean']:.6f} ± {means[a]['macro_f1_sample_sd']:.6f} | {ensemble_metrics[a]['macro_f1']:.6f} |")
    lines+=['','## Prespecified contrasts','','Primary: plus_quantiles minus plus_spread. All comparisons retained; intervals are descriptive.','','| Comparison | Ensemble difference | Paired patient 95% interval |','|---|---:|---:|']
    for name,v in contrasts.items():
        lo,hi=v['descriptive_patient_interval'];lines.append(f"| {name} | {v['ensemble_delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## Class F1','','| Class | '+' | '.join(config['arms'])+' |','|---|'+'---:|'*len(config['arms'])]
    for c in config['classes']:lines.append(f"| {c} | "+' | '.join(f"{ensemble_metrics[a]['classification_report'][c]['f1-score']:.6f}" for a in config['arms'])+' |')
    lines+=['','## Method and limits','',
            'The complete frozen recording-median MultiRocket-style + experimental multivariate HYDRA ridge procedure is the matched baseline. Every augmented arm preserves its full standardized vector bit-for-bit. Frozen kernels, feature scales and exact patient partitions are reused read-only; HYDRA attribution and GPL license remain in third_party/hydra. Historical MultiRocket conventions differ from this project implementation. Source feature transforms are independently refitted on training templates; all original source classifiers are restored without refitting.', '',
            'For every original beat, subtract its own recording-median convolution vector, divide by the source training-column scales, and apply 16 seeded unit-norm feature-space projections. This does not rotate physical XYZ or unit-normalize beat amplitudes. A common residual RMS uses only training patients, with equal patient, recording and within-record beat weights. The added grid represents mean only, mean plus population SD on fixed normal quantiles, or empirical quantiles. The normal basis does not assert Gaussian physiology. Nine quantiles are divided by sqrt(9); all auxiliary grids receive the same fixed sqrt(base_dimension/projection_count) factor. No per-arm scaler undoes it. Full base and auxiliary grids pool equally across recordings per patient; stored dimensions and effective ranks differ.', '',
            'Class-balanced ridge with intercept/Cholesky receives the same two alpha candidates, selected by mean inner patient macro F1 and configured-order ties. All three declared seeds remain in uniform probability ensembles. Mean member results are separate. Fixed-temperature ridge softmax is an aggregation interface, not calibrated clinical risk. All fitted preprocessing stays in its exact training partition. Full-record medians make this offline inference.', '',
            'The primary comparison tests quantile shape beyond mean and spread; additions versus the intact baseline test preservation of beat variation. No outer-based arm, budget, source, seed, class routing or threshold selection. The repeated PTB cohort is development data. Conditional class-stratified patient bootstrap intervals share draws, with no refits or correction for repeated development, overlapping training folds or multiple comparisons. More beats do not add independent patients. No independent confirmation, clinical validity, SOTA or guaranteed 0.80–0.90 macro F1 follows. Historical neural 0.584570 remains a separate reference.', '',
            'This is a compact projected-quantile adaptation, not an exact implementation of FSW or PSWE and not a claim to their theoretical guarantees. No ECG, new patients, source pretraining, learned encoder or tensor decomposition is added. See docs/convolution_distribution_plan.md. After independent publication this runner stops; no further study is queued.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    result=dict(complete=True,study_fingerprint=scope['study_fingerprint'],claim_scope=study.CLAIM,**expected,
                all_models_restored=len(fits)+len(refs),new_models_restored=len(fits),source_models_restored=len(refs),outer_models_restored=expected['outer_models'],
                feature_contexts_reconstructed=len(contexts),matched_context_checks=matched,all_original_beats_replayed=True,original_cohort_unchanged=True,
                source_files_unchanged=True,baseline_probabilities_exact=True,n_patients=len(bank.ids),n_records=len(bank.record_patients),n_beats=len(bank.records.waves),
                projection_seeds=config['projection_seeds'],base_dimensions=d,auxiliary_dimensions=config['projections']*len(config['quantiles']),
                mean_member_metrics=means,member_metrics=metrics,ensemble_metrics=ensemble_metrics,contrasts=contrasts,selection_counts=counts,
                report_sha256=file_hash(root/'REPORT.md'),verification_manifest_sha256=file_hash(root/'verification_manifest.json'))
    _save_json(root/'summary.json',result);_progress(root,state='complete_independently_validated',**expected)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/convolution_distribution_v1.yaml');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(config['torch_threads'])
    try:
        with threadpool_limits(limits=1):result=summarize(config)
        print(json.dumps({k:result[k] for k in ('complete','fits','new_models_restored','all_models_restored')}))
    except Exception as error:
        _progress(Path(config['output_dir'])/'complete',state='failed',error=str(error));raise
