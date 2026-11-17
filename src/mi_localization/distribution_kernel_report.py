"""Rebuild each kernel, restore every classifier, repeat all inner decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits
from . import distribution_kernel_study as study,residual_context_report as uncertainty
from .benchmark import metric_bundle,save_confusion
from .contrastive import _progress
from .neural import _save_json
from .signal_audit import file_hash


def summarize(config):
    bank,assigned,_=study.load_inputs(config);root,scope=study.guard(config,bank)
    procedures=study.names(config)+['selected_linear','selected_nonlinear']
    members={(a,s):np.full((len(bank.ids),len(config['classes'])),np.nan) for a in procedures for s in config['projection_seeds']}
    counts={r:0 for r in config['representations']}
    for fold in range(1,config['outer_splits']+1):
        for seed in config['projection_seeds']:
            rows,values=study.evaluate_fold(bank,assigned,config,scope,root,seed,fold,verify=True)
            for arm,p in values.items():
                if np.isfinite(members[arm,seed][rows]).any():raise ValueError('Duplicate outer patients')
                members[arm,seed][rows]=p
            chosen=json.loads((root/'review'/f'paired_selection_{fold:02d}_s{seed}.json').read_text())['selected_representation'];counts[chosen]+=1
    expected=study.budget(config);fits=sorted((root/'review/fits').glob('*.json'));contexts=sorted((root/'review/contexts').glob('*.json'));eq=sorted((root/'review/equivalence').glob('*.json'))
    if (len(fits)!=expected['fits'] or len(contexts)!=expected['kernel_contexts'] or len(eq)!=expected['source_equivalence_checks']
            or sum(p.name.startswith('outer_') for p in fits)!=expected['outer_models']):raise ValueError('Kernel coverage incomplete')
    # Pairing is represented by the same frozen feature and kernel-statistic context.
    matched=0
    for ctx in contexts:
        a=json.loads(ctx.read_text());fold,seed,inner,rep=a['outer_fold'],a['projection_seed'],a['inner_fold'],a['representation']
        for family in ('linear','nonlinear'):
            suffixes=[f'_i{inner}_a{i}' for i in range(len(study.candidates(config,family)))] if inner else ['']
            for suffix in suffixes:
                fit=json.loads((root/'review/fits'/f"{'inner' if inner else 'outer'}_{fold:02d}_{rep}__{family}_s{seed}{suffix}.json").read_text())
                if fit['context_sha256']!=file_hash(ctx) or any(fit['kernel_state'][k]!=v for k,v in a['state'].items()):raise ValueError('Unmatched kernel context')
        matched+=1
    metrics,means,ensemble_metrics,ensemble={},{},{},{}
    for arm in procedures:
        metrics[arm]={}
        for seed in config['projection_seeds']:
            p=members[arm,seed]
            if not np.isfinite(p).all():raise ValueError('Missing outer predictions')
            metrics[arm][str(seed)]=metric_bundle(bank.labels,p,config['classes'])
        vals=[v['macro_f1'] for v in metrics[arm].values()];means[arm]=dict(macro_f1_mean=float(np.mean(vals)),macro_f1_sample_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.)
        ensemble[arm]=np.mean([members[arm,s] for s in config['projection_seeds']],axis=0);ensemble_metrics[arm]=metric_bundle(bank.labels,ensemble[arm],config['classes'])
        save_confusion(bank.labels,ensemble[arm].argmax(1),config['classes'],arm,root)
        for suffix,p in [(f's{s}',members[arm,s]) for s in config['projection_seeds']]+[('ensemble',ensemble[arm])]:
            frame=pd.DataFrame(dict(patient_id=bank.ids,truth=np.asarray(config['classes'])[bank.labels],prediction=np.asarray(config['classes'])[p.argmax(1)],fold=assigned))
            for i,c in enumerate(config['classes']):frame[f'probability_{c}']=p[:,i]
            frame.to_csv(root/'review'/f'patient_predictions_{arm}_{suffix}.csv',index=False)
    for rep in config['representations']:
        if ensemble_metrics[f'{rep}__linear']!=bank.summary['ensemble_metrics'][rep]:raise ValueError('Linear ensemble no longer reproduces source')
        for seed in config['projection_seeds']:
            if metrics[f'{rep}__linear'][str(seed)]!=bank.summary['member_metrics'][rep][str(seed)]:raise ValueError('Linear member no longer reproduces source')
    labels=np.arange(len(config['classes']));weights=uncertainty.patient_weights(bank.labels,labels,config['bootstrap_draws'],config['seed'])
    draws={a:uncertainty.weighted_f1(bank.labels,p.argmax(1),labels,weights) for a,p in ensemble.items()}
    mean_draws={a:np.mean([uncertainty.weighted_f1(bank.labels,members[a,s].argmax(1),labels,weights) for s in config['projection_seeds']],axis=0) for a in procedures}
    pairs=[('selected_nonlinear','selected_linear')]+[(f'{r}__nonlinear',f'{r}__linear') for r in config['representations']]+[('plus_quantiles__nonlinear','plus_spread__nonlinear')]
    contrasts={f'{a}_minus_{b}':dict(primary=(a,b)==pairs[0],ensemble_delta_macro_f1=ensemble_metrics[a]['macro_f1']-ensemble_metrics[b]['macro_f1'],
                descriptive_patient_interval=np.quantile(draws[a]-draws[b],[.025,.975]).tolist(),mean_member_delta_macro_f1=means[a]['macro_f1_mean']-means[b]['macro_f1_mean'],
                mean_member_descriptive_interval=np.quantile(mean_draws[a]-mean_draws[b],[.025,.975]).tolist()) for a,b in pairs}
    audit_files=fits+contexts+eq+sorted((root/'review').glob('*selection*.json'))
    _save_json(root/'verification_manifest.json',dict(study_fingerprint=scope['study_fingerprint'],restored_audit_hashes={str(p.relative_to(root)):file_hash(p) for p in audit_files}))
    lines=['# Linear and nonlinear distribution readouts','',study.CLAIM,'',
           f"All {len(bank.ids)} patients retained. Restored {len(fits)} classifiers including {expected['outer_models']} outer models; rebuilt {len(contexts)} representation kernels from {expected['feature_contexts']} independently verified source feature contexts; {matched} matching checks and {len(eq)} original-ridge equivalence checks passed. No waveform re-extraction or new feature fitting.",'',
           '| Procedure | Mean member F1 ± seed SD | Uniform seed ensemble F1 |','|---|---:|---:|']
    for a in procedures:lines.append(f"| {a} | {means[a]['macro_f1_mean']:.6f} ± {means[a]['macro_f1_sample_sd']:.6f} | {ensemble_metrics[a]['macro_f1']:.6f} |")
    lines+=['','## Prespecified paired comparisons','','Primary: selected_nonlinear minus selected_linear. Both families use the SAME inner-selected representation in each fold/seed. This is not an outer-score winner.','','| Comparison | Ensemble difference | Descriptive paired 95% interval |','|---|---:|---:|']
    for name,c in contrasts.items():
        lo,hi=c['descriptive_patient_interval'];lines.append(f"| {name} | {c['ensemble_delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## Interpretation and scope','',
            'Each exact training partition supplies class-balanced weights, an unpenalized weighted intercept, Gaussian bandwidth and kernel energy scale. Kernel weights and ridge alpha use inner patients only. Feature representations, all beats, labels, R positions, folds and source preprocessing remain unchanged. The linear dual solution reproduces the original primal/dual RidgeClassifier probabilities within declared numerical tolerances and its labels/inner selections/metrics exactly. No new physical rotation, amplitude normalization or raw-signal encoding.', '',
            'Nonlinear kernel is (1−lambda) XX^T + lambda * training_energy * exp(−squared_distance/(2*training_median_positive_squared_distance)). Each X is the ENTIRE verified feature vector for that representation, including the complete morphology base and any auxiliary summary. It tests nonlinear readout of fixed summaries and their interactions; it is not an exact sliced-Wasserstein distance, separate physiology kernel, full optimal-transport solver or import of theoretical guarantees. Degenerate training scale uses a declared fallback. Every kernel is checked for positive semidefiniteness and centered using training weights only.', '',
            'Linear family has two alpha candidates; nonlinear has two alphas times two positive mixture weights. The nonlinear search has twice the candidate budget, which is disclosed, not treated as a pure architecture-only causal intervention. For each representation/family, highest mean inner macro F1 selects a candidate, with configured-order ties. The shared representation maximizes the average of the two family best inner scores; both outer readouts use that same chosen representation. All four fixed-representation paired comparisons are retained. All three seeds remain in equal probability ensembles, with mean member scores separate. Softmax is a fixed-temperature aggregation interface, not calibrated clinical risk.', '',
            'The old cached feature study independently re-encoded all original beats before completion. This follow-up rechecks its pinned hashes, complete chain of source artifacts/record caches, memberships and restored source predictions. It rebuilds every new kernel and restores every new classifier; it does not rerun the expensive waveform feature extraction. Hash failures block rather than trigger a hidden source rebuild. Cache-only chain of custody is not a new independent raw-signal validation.', '',
            'Source outcomes were already inspected before this development hypothesis. Patient-separated fitting does not make the cohort untouched. Conditional stratified patient intervals use shared draws and fixed predictions, with no correction for repeated development, overlapping training sets, multiple comparisons or fitting variability. Small gains do not establish independent generalization, clinical validity, novel physiology, SOTA or guaranteed 0.80–0.90 macro F1. The source spread ensemble .594098 and earlier neural ensemble .584570 remain labeled references, not freely substituted matched controls.', '',
            'Research motivation: Meunier et al., Distribution Regression with Sliced Wasserstein Kernels, ICML 2022, https://proceedings.mlr.press/v162/meunier22b.html. Protocol: docs/distribution_kernel_plan.md. Source HYDRA attribution/license remain in src/mi_localization/_vendor/hydra. The runner publishes only verified aggregate outputs and then stops.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    result=dict(complete=True,study_fingerprint=scope['study_fingerprint'],claim_scope=study.CLAIM,**expected,all_models_restored=len(fits),outer_models_restored=expected['outer_models'],
                kernel_contexts_reconstructed=len(contexts),matched_context_checks=matched,source_all_arm_metrics_reproduced=True,
                cache_only_source_chain_verified=True,new_waveform_extractions=0,original_cohort_unchanged=True,n_patients=len(bank.ids),n_records=bank.n_records,n_beats=bank.n_beats,
                member_metrics=metrics,mean_member_metrics=means,ensemble_metrics=ensemble_metrics,contrasts=contrasts,shared_representation_counts=counts,
                report_sha256=file_hash(root/'REPORT.md'),verification_manifest_sha256=file_hash(root/'verification_manifest.json'))
    _save_json(root/'summary.json',result);_progress(root,state='complete_independently_validated',**expected);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/distribution_kernel_v1.yaml');args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(config['torch_threads'])
    try:
        with threadpool_limits(limits=1):result=summarize(config)
        print(json.dumps({k:result[k] for k in ('complete','fits','all_models_restored','matched_context_checks')}))
    except Exception as error:
        _progress(Path(config['output_dir'])/'complete',state='failed',error=str(error));raise
