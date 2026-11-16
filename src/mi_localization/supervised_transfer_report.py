"""Restore every target fit, repeat nested choices and report all declared ensembles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from . import supervised_transfer as study
from .benchmark import aggregate_patients,metric_bundle
from .contrastive_report import paired_seed_bootstrap
from .supervised_source_audit import file_hash,save_json

PUBLIC=['summary.json','REPORT.md','per_seed.csv','ensemble_metrics.csv','selection_summary.csv','paired_bootstrap.json','comparison.png']


def ensemble_intervals(y,probabilities,classes,draws,seed):
    rng=np.random.default_rng(seed);n=len(classes)
    indices=np.column_stack([rng.choice(np.flatnonzero(y==j),(draws,int((y==j).sum()))) for j in range(n)])
    offsets=np.arange(draws)[:,None]*n*n;distributions={}
    for arm,p in probabilities.items():
        encoded=n*y[indices]+p.argmax(1)[indices]+offsets
        counts=np.bincount(encoded.ravel(),minlength=draws*n*n).reshape(draws,n,n)
        numerator=2*np.diagonal(counts,axis1=1,axis2=2);denominator=counts.sum(1)+counts.sum(2)
        distributions[arm]=np.divide(numerator,denominator,out=np.zeros_like(numerator,float),where=denominator!=0).mean(1)
    return dict(draws=draws,unique_patients=len(y),seed=seed,
        arm_macro_f1_intervals={a:np.quantile(v,[.025,.975]).tolist() for a,v in distributions.items()},
        paired_difference_intervals={a+'_minus_scratch':np.quantile(distributions[a]-distributions['scratch'],[.025,.975]).tolist() for a in study.ARMS[1:]},
        method='Class-stratified paired patient bootstrap; identical patient draws across all fixed three-seed probability ensembles',
        limits='Conditional on fixed predictions and two fixed source encoders; no refitting/seed resampling, independent cohort or correction for shared training folds, repeated development or multiple comparisons')


def summarize(config):
    if len(config['training_seeds'])<3: raise ValueError('Final report requires every fold and at least three declared target seeds')
    data,result,scope,root,partial=study.run_study(config,verify=True)
    if partial: raise ValueError('Cannot report a partial target study')
    predictions={};rows=[];scores={};truth=None;patient_ids=None
    for seed,arms in result['predictions'].items():
        predictions[seed]={};scores[seed]={}
        for arm,p in arms.items():
            ids,y,pooled=aggregate_patients(p,data['y'],data['groups'])
            if truth is not None and (not np.array_equal(y,truth) or not np.array_equal(ids,patient_ids)):
                raise ValueError('Target patient alignment differs across arms/seeds')
            truth,patient_ids=y,ids;predictions[seed][arm]=pooled;scores[seed][arm]=metric_bundle(y,pooled,config['classes'])
            rows.append(dict(training_seed=seed,arm=arm,patient_macro_f1=scores[seed][arm]['macro_f1']))
    ensemble_p={arm:np.mean([predictions[s][arm] for s in config['training_seeds']],axis=0) for arm in study.ARMS}
    ensembles={a:metric_bundle(truth,p,config['classes']) for a,p in ensemble_p.items()}
    means={arm:dict(macro_f1_mean=float(np.mean([scores[s][arm]['macro_f1'] for s in config['training_seeds']])),
                    macro_f1_sample_std=float(np.std([scores[s][arm]['macro_f1'] for s in config['training_seeds']],ddof=1))) for arm in study.ARMS}
    mean_ci={}
    for arm in study.ARMS[1:]:
        labels={s:{a:predictions[s][a].argmax(1) for a in ['scratch',arm]} for s in config['training_seeds']}
        mean_ci[arm]=paired_seed_bootstrap(truth,labels,len(config['classes']),config['bootstrap_draws'],config['seed'],arms=('scratch',arm))
    ensemble_ci=ensemble_intervals(truth,ensemble_p,config['classes'],config['bootstrap_draws'],config['seed'])
    uncertainty=dict(mean_seed_comparisons=mean_ci,equal_seed_ensemble_comparison=ensemble_ci)
    summary=dict(complete=True,study_fingerprint=scope['study_fingerprint'],claim_scope=study.CLAIM,
        n_patients=len(truth),n_records=data['bank']['n_records'],n_beats=data['bank']['n_beats'],
        training_seeds=config['training_seeds'],source_training_seeds=[config['source_training_seed']],
        source=data['source'],completed_fit_count=result['completed_fit_count'],restored_fit_count=result['completed_fit_count'],
        outer_model_restorations=len(study.ARMS)*config['outer_splits']*len(config['training_seeds']),
        matched_context_checks=result['matched_context_checks'],mean_seed_metrics=means,per_seed_metrics={str(s):v for s,v in scores.items()},
        equal_probability_seed_ensembles=ensembles,
        primary_comparison='source_certain minus matched scratch, prespecified equal-three-seed probability ensembles',
        sensitivity_comparison='source_positive_intermediate minus matched scratch; no source-objective winner selection',
        ensemble_differences={a:ensembles[a]['macro_f1']-ensembles['scratch']['macro_f1'] for a in study.ARMS[1:]},
        mean_seed_differences={a:means[a]['macro_f1_mean']-means['scratch']['macro_f1_mean'] for a in study.ARMS[1:]},
        primary_ensemble_difference_interval=ensemble_ci['paired_difference_intervals']['source_certain_minus_scratch'],
        patient_identity_independence_cleared=False,independent_confirmation=False,next_experiment_automatically_launched=False)
    artifacts={str(p.relative_to(root)):file_hash(p) for p in sorted((root/'review/fits').glob('*/*')) if p.is_file()}
    artifacts.update({str(p.relative_to(root)):file_hash(p) for p in sorted((root/'review/selections').glob('*.json'))})
    manifest=root/'report_manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if (saved['study_fingerprint']!=scope['study_fingerprint'] or saved['private_fit_files']!=artifacts
                or any(file_hash(root/n)!=h for n,h in saved['files'].items())
                or json.loads((root/'summary.json').read_text())!=summary
                or json.loads((root/'paired_bootstrap.json').read_text())!=uncertainty
                or file_hash(root/'review/patient_probabilities.npz')!=saved['patient_probability_sha256']):
            raise ValueError('Supervised-transfer report/metrics/artifacts do not reproduce')
        with np.load(root/'review/patient_probabilities.npz') as saved_p:
            if (not np.array_equal(saved_p['patient_ids'],patient_ids) or not np.array_equal(saved_p['truth'],truth)
                    or any(not np.allclose(saved_p[f'{s}_{a}'],p,atol=2e-5,rtol=2e-5) for s,arms in predictions.items() for a,p in arms.items())):
                raise ValueError('Reported patient probabilities do not reproduce')
        return summary
    save_json(root/'summary.json',summary);save_json(root/'paired_bootstrap.json',uncertainty)
    pd.DataFrame(rows).to_csv(root/'per_seed.csv',index=False)
    pd.DataFrame([dict(arm=a,patient_macro_f1=m['macro_f1'],accuracy=m['accuracy']) for a,m in ensembles.items()]).to_csv(root/'ensemble_metrics.csv',index=False)
    pd.DataFrame(result['selections']).to_csv(root/'selection_summary.csv',index=False)
    private=root/'review/patient_probabilities.npz'
    np.savez_compressed(private,patient_ids=patient_ids,truth=truth,
        **{f'{seed}_{arm}':p for seed,arms in predictions.items() for arm,p in arms.items()})
    fig,ax=plt.subplots(figsize=(7.5,4))
    x=np.arange(len(study.ARMS));ax.bar(x,[ensembles[a]['macro_f1'] for a in study.ARMS],color=['.7','.25','.5'])
    for i,a in enumerate(study.ARMS):
        ax.scatter(np.full(len(config['training_seeds']),i),[scores[s][a]['macro_f1'] for s in config['training_seeds']],color='black',marker='x',s=25)
    ax.set_xticks(x,['Scratch','Source: certain','Source: intermediate positive']);ax.set(ylabel='Patient macro F1',ylim=(0,1),title='Bars: equal seed ensembles; crosses: individual seeds')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    lines=['# Matched supervised-source transfer to measured Frank VCG','',study.CLAIM,'',
        f"{len(truth)} target patients, {data['bank']['n_records']} recordings and {data['bank']['n_beats']:,} original beats; {config['outer_splits']} frozen outer folds, {config['inner_splits']} inner folds, target seeds {config['training_seeds']}. Both source objectives use the same single source training seed {config['source_training_seed']}; target restarts are not independent source pretraining replications.",'',
        '| Procedure | Mean seed macro F1 ± sample SD | Equal-probability seed ensemble macro F1 |', '|---|---:|---:|']
    for arm in study.ARMS:
        m=means[arm];lines.append(f"| {arm} | {m['macro_f1_mean']:.6f} ± {m['macro_f1_sample_std']:.6f} | {ensembles[arm]['macro_f1']:.6f} |")
    lines+=['','## Prespecified comparisons','']
    for arm in study.ARMS[1:]:
        lo,hi=ensemble_ci['paired_difference_intervals'][arm+'_minus_scratch']
        lines.append(f"{arm}−scratch ensemble difference **{summary['ensemble_differences'][arm]:+.6f}**, descriptive paired patient-bootstrap interval **[{lo:+.6f},{hi:+.6f}]**.")
    lines+=['','The certain source is the primary comparison; intermediate-positive is sensitivity. Both and all target seeds are retained, without choosing a source objective, seed subset or ensemble weights from outer scores. Mean member F1 and probability-ensemble F1 are different endpoints. The historical neural ensemble0.584570 and full-feature ridge0.584266 remain separate development references, not newly trained matched controls.','',
        '![Comparison](comparison.png)','','## Ensemble class results','','| Class | Target patients | Scratch F1 | Certain-source F1 | Intermediate-source F1 |','|---|---:|---:|---:|---:|']
    for j,label in enumerate(config['classes']):
        values=[ensembles[a]['classification_report'][label]['f1-score'] for a in study.ARMS]
        lines.append(f"| {label} | {int((truth==j).sum())} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    lines+=['','## Training and independent verification','',
        f"All **{result['completed_fit_count']}** declared fits were restored: **{summary['outer_model_restorations']} outer models**, remaining fits from every inner rate/partition/seed/arm. All **{result['matched_context_checks']} matched context checks** passed. Every saved model is re-inferred on its exact query rows; inner best-checkpoint F1 is reproduced, every learning-rate/epoch choice is repeated from inner scores, and outer test patients never set stopping or normalization. Model/prediction/selection hashes and source/provenance scopes are checked.",'',
        f"The existing same ResNet/CE/AdamW trainer is used for all arms. Only the declared source encoder weights transfer; the source diagnostic head, normalization, optimizer and positive-loss weights are discarded. Every target classifier is freshly initialized with matching paired weights. Target normalization is refitted within each target training partition, using the same class/patient weights; no per-beat unit-amplitude normalization or rotation is introduced. Full encoders fine-tune in both source arms.",'',
        f"All procedures share the target learning-rate candidates {config['learning_rates']}, batch{config['batch_size']}, {config['samples_per_epoch']:,} sampled beats/epoch, at most{config['max_epochs']} epochs and patience{config['patience']}. Matching initial heads, sample sequences for shared epochs, normalization, parameter counts and memberships are checked. Early-stopped durations and separately inner-selected settings can differ. Rate ties use configured order; refit epochs are the rounded median of the chosen candidate's best inner epochs. Outer refits receive no validation/test monitoring. Completed-fit resume is supported; interrupting an unfinished fit repeats that fit. Source pretraining adds compute beyond the equal target budgets.",'',
        'All held-out original beat probabilities are averaged per patient. Uniform probability averaging across all three target seeds is prespecified per procedure; no learned combiner, seed selection, threshold tuning or source-target class-label mapping is fitted. Native source overlapping diagnoses and the source NORM definition do not relabel target classes.','',
        '## Provenance and interpretation limits','',
        f"Both supervised source models and the expanded recording audit were independently verified before launch. The expanded screen covers {data['source']['expanded_recording_screen']['source_records']} source training/validation recordings against {data['source']['expanded_recording_screen']['target_records']} target recordings, with {data['source']['expanded_recording_screen']['continuous_pairs_checked']} retrieved continuous comparisons and {data['source']['expanded_recording_screen']['positive_controls_retrieved_and_confirmed']}/{data['source']['expanded_recording_screen']['positive_controls']} known-copy controls. Its passed recording flag does not resolve different visits by the same person, shorter/modified copies, retrieval misses or clinical source-target equivalence. No target patients were deleted or relabeled.",'',
        'This is repeatedly inspected PTB development data, with ECG-derived source XYZ versus measured Frank target XYZ. The paired bootstrap resamples each patient once per draw, shared across fixed predictions/arms/seeds. It does not refit models, resample source pretraining, correct correlated training folds/repeated development/multiple comparisons, or establish independent confirmation, clinical validity, SOTA or a guaranteed0.80–0.90 result. Full-record preprocessing is offline/noncausal; seeded MPS is not bitwise-deterministic training.','',
        'Raw source/target files, models, fit-level query predictions and patient probabilities stay local in ignored review/data locations; public results contain aggregate metrics and hashed provenance. No subsequent experiment is launched.','',
        'Protocol: [matched supervised transfer](../../../docs/supervised_transfer_plan.md). Source evidence: [pilot](../../supervised_source_pilot_v1/REPORT.md), [expanded overlap audit](../../supervised_source_overlap_v1/REPORT.md).','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    save_json(manifest,dict(study_fingerprint=scope['study_fingerprint'],private_fit_files=artifacts,
        files={n:file_hash(root/n) for n in PUBLIC},patient_probability_sha256=file_hash(private)))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/supervised_transfer_v1.yaml')
    args=p.parse_args();summarize(yaml.safe_load(Path(args.config).read_text()))
