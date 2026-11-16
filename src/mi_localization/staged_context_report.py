"""Replay every staged fit, frozen-weight invariant and prespecified seed ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import staged_context_study as study,residual_context_report as bootstrap
from .benchmark import metric_bundle,save_confusion
from .contrastive import _progress
from .neural import _save_json

CONTRASTS=(('frozen_context','morphology'),('frozen_context','continued_morphology'),('frozen_context','joint_context'),
           ('frozen_context','frozen_shuffled'),('joint_context','continued_morphology'),('continued_morphology','morphology'))


def matched_checks(root,config,seed,fold):
    checked=0
    for inner in range(config['inner_splits']+1):
        teacher=json.loads(study.path_for(root,fold,seed,inner,'morphology').read_text())
        audits=[json.loads(study.path_for(root,fold,seed,inner,a).read_text()) for a in study.models.ADAPTER_ARMS]
        for audit in audits:
            if (audit['teacher_model_sha256']!=teacher['model_sha256'] or audit['fit']['normalization']!=teacher['fit']['normalization']
                    or any(audit['fit'][k]!=audits[0]['fit'][k] for k in ['initial_state_sha256','initial_morphology_sha256','parameter_count'])):
                raise ValueError('Staged controls have different teachers/initialization')
        for epoch in range(min(len(a['fit']['history']) for a in audits)):
            rows=[a['fit']['history'][epoch] for a in audits]
            if any(any(row[k]!=rows[0][k] for k in ['window_plan_sha256','direct_token_order_sha256','valid_beat_presentations','nominal_beat_slots']) for row in rows):
                raise ValueError('Staged controls saw different windows/direct beats')
        checked+=1
    return checked


def metrics(y,p,classes):
    result=metric_bundle(y,p,classes);mi=y!=classes.index('HC')
    result['mi_only_macro_f1_six_way_predictions']=float(f1_score(y[mi],p[mi].argmax(1),labels=[i for i,c in enumerate(classes) if c!='HC'],average='macro',zero_division=0))
    return result


def summarize(config):
    bank,assigned=study.load_inputs(config);scope=study.make_scope(config,bank)
    root=Path(config['output_dir'])/'complete'
    if json.loads((root/'study_manifest.json').read_text())!=scope:raise ValueError('Staged report source/config/runtime changed')
    members={(a,s):np.full((len(bank.ids),len(config['classes'])),np.nan) for a in config['arms'] for s in config['training_seeds']}
    checks=0
    for fold in range(1,config['outer_splits']+1):
        for seed in config['training_seeds']:
            rows,values=study.evaluate_fold(bank,assigned,config,scope,root,seed,fold,verify=True)
            for arm,p in values.items():
                if np.isfinite(members[arm,seed][rows]).any():raise ValueError('Duplicate staged patient predictions')
                members[arm,seed][rows]=p
            checks+=matched_checks(root,config,seed,fold)
    member_metrics={a:{} for a in config['arms']};ensembles={};ensemble_metrics={}
    for (arm,seed),p in members.items():
        study.reference.cache.check_p(p,len(bank.ids),len(config['classes']))
        member_metrics[arm][str(seed)]=metrics(bank.labels,p,config['classes'])
    for arm in config['arms']:
        p=np.mean([members[arm,s] for s in config['training_seeds']],axis=0);ensembles[arm]=p
        ensemble_metrics[arm]=metrics(bank.labels,p,config['classes'])
        for suffix,value in [(f's{seed}',members[arm,seed]) for seed in config['training_seeds']]+[('ensemble',p)]:
            frame=pd.DataFrame(dict(patient_id=bank.ids,truth=np.asarray(config['classes'])[bank.labels],prediction=np.asarray(config['classes'])[value.argmax(1)],fold=assigned))
            for i,c in enumerate(config['classes']):frame[f'probability_{c}']=value[:,i]
            frame.to_csv(root/f'patient_predictions_{arm}_{suffix}.csv',index=False)
        save_confusion(bank.labels,p.argmax(1),config['classes'],arm,root)
    contrasts={}
    for name,labels in [('six_class',np.arange(len(config['classes']))),('mi_only',np.array([i for i,c in enumerate(config['classes']) if c!='HC']))]:
        weights=bootstrap.patient_weights(bank.labels,labels,config['bootstrap_draws'],config['seed']);selected=np.isin(bank.labels,labels)
        draws={a:bootstrap.weighted_f1(bank.labels,p.argmax(1),labels,weights) for a,p in ensembles.items()}
        points={a:float(f1_score(bank.labels[selected],p[selected].argmax(1),labels=labels,average='macro',zero_division=0)) for a,p in ensembles.items()}
        for a,b in CONTRASTS:
            contrasts.setdefault(f'{a}_minus_{b}',{})[name]=dict(ensemble_delta_macro_f1=points[a]-points[b],descriptive_patient_interval=np.quantile(draws[a]-draws[b],[.025,.975]).tolist())
    paths=list(root.glob('inner_*.json'))+list(root.glob('outer_*.json'))
    if len(paths)!=config['outer_splits']*len(config['training_seeds'])*len(config['arms'])*(config['inner_splits']+1):
        raise ValueError('Unexpected staged phase-fit count')
    zero_inner=zero_outer=0
    for path in paths:
        audit=json.loads(path.read_text())
        if audit['phase']=='adaptation' and audit['fit']['selected_epoch']==0:
            if audit['inner_fold']:zero_inner+=1
            else:zero_outer+=1
    outer=config['outer_splits']*len(config['training_seeds'])*len(config['arms'])
    summary=dict(complete=True,study_fingerprint=scope['study_fingerprint'],claim_scope=study.CLAIM,n_patients=len(bank.ids),
                 classes=config['classes'],training_seeds=config['training_seeds'],phase_fits=len(paths),
                 all_phase_models_restored=len(paths),outer_models_restored=outer,matched_adaptation_partition_checks=checks,
                 epoch_zero_inner_choices=zero_inner,zero_update_outer_adaptations=zero_outer,all_query_beats_covered=True,
                 member_metrics=member_metrics,ensemble_metrics=ensemble_metrics,contrasts=contrasts)
    lines=['# Staged morphology and frozen-context learning','',study.CLAIM,'',
           f"All {len(bank.ids)} original patients. {len(paths)} phase-fit/checkpoint contexts verified and restored (including {outer} outer models); {checks} matched adaptation-partition checks. {zero_outer} outer adaptation models selected zero updates from inner epochs. These remain phase checkpoints, not gradient-trained fits.",'',
           '| Procedure | Mean member macro F1 ± seed SD | Prespecified seed-ensemble F1 | Ensemble MI-only F1 |','|---|---:|---:|---:|']
    for arm in config['arms']:
        values=[v['macro_f1'] for v in member_metrics[arm].values()];sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        m=ensemble_metrics[arm]
        lines.append(f"| {arm} | {np.mean(values):.3f} ± {sd:.3f} | {m['macro_f1']:.3f} | {m['mi_only_macro_f1_six_way_predictions']:.3f} |")
    lines+=['','Seed ensembles were declared before this study, and use all declared seeds with equal probabilities. Ensemble F1 is not the mean member F1 or an extra independent cohort. All individual-seed metrics are in summary.json.','',
            '| Ensemble comparison | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |','|---|---:|---:|']
    for name,value in contrasts.items():
        cells=[]
        for k in ['six_class','mi_only']:
            r=value[k];lo,hi=r['descriptive_patient_interval'];cells.append(f"{r['ensemble_delta_macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines+=['','## Ensemble class F1','','| Class | '+' | '.join(config['arms'])+' |','|---|'+'---:|'*len(config['arms'])]
    for c in config['classes']:lines.append(f"| {c} | "+' | '.join(f"{ensemble_metrics[a]['classification_report'][c]['f1-score']:.3f}" for a in config['arms'])+' |')
    lines+=['','## Exact protocol and interpretation','',
            'Morphology is learned from scratch separately within each inner-training or outer-training partition. Its inner best epoch sets a fresh outer refit duration. Every adaptation arm receives only that exact partition teacher’s encoder/direct head, plus a freshly initialized contextual branch whose final logit layer is zero. Thus adaptation starts at the unchanged teacher prediction. No previously fitted outer encoder is reused across inner validation patients.','',
            'Continued morphology trains only the direct predictor; joint context trains base and context; frozen context and its shuffled control train only context/RR/gate parameters. All adaptation controls disable direct-head dropout and retain context-head dropout during training. Equal registered counts do not imply equal trainable counts. Common positive epochs have identical sampled windows and direct beats; shuffling changes only contextual waveform/RR order. Frozen base weights are hashed every epoch and in selected checkpoints.','',
            'Adaptation inner selection includes epoch zero, with earliest-best ties and the same 25-epoch cap / patience six. Each arm’s rounded median best inner epoch (including zero) sets the outer refit, with no query monitoring. Outer zero-update adaptations reproduce the teacher. This is an inner selection fallback, not a guarantee of improved outer F1. Baseline morphology has no second phase; continued morphology controls extra optimization. Total exposure and selected duration can differ by arm.','',
            'Both phases use the original class/patient-balanced, uniform-record contiguous four-beat sampling, 8,192 nominal beat slots and 64 updates/epoch at the default configuration. All original patients, six labels, measured XYZ, R positions and frozen folds are retained. Query inference covers every beat once, masks tails, weights windows by real-beat count within records and averages records equally.','',
            f"Intervals use {config['bootstrap_draws']:,} class-stratified paired patient draws, conditional on fixed ensemble predictions. No refits or correction for repeated development, overlapping CV training folds or multiple comparisons. Seed SD is optimization variation, not a generalization interval. MI-only scores retain unchanged six-way predictions, including MI→HC errors.",'',
            'The frozen historical residual-context ensemble 0.584570 is a different procedure, not the matched teacher or joint control here. Side-tuning and feature-preserving adaptation are prior art; this is a VCG-specific controlled hypothesis, not demonstrated novelty, SOTA or a guaranteed 0.80–0.90 result. These repeatedly inspected PTB patients remain development data, requiring independent label-compatible confirmation.','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');_save_json(root/'summary.json',summary);_progress(root,state='complete_validated')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/staged_context_v1.yaml')
    args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    try:
        s=summarize(config);print(json.dumps(dict(complete=s['complete'],phase_fits=s['phase_fits'])))
    except Exception as error:
        _progress(Path(config['output_dir'])/'complete',state='failed',error_type=type(error).__name__,error=str(error));raise
