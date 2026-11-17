"""Bounded paired linear/nonlinear readouts on immutable verified patient features."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import scipy
import sklearn
import torch
import yaml
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from . import benchmark,distribution_study as core
from . import distribution_kernel_model as kernel,distribution_kernel_source as source
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash,_save_json
from .signal_audit import file_hash

CLAIM='Nonlinear readout PTB development using already verified fixed feature caches; repeated cohort, no independent confirmation, clinical validity or established SOTA.'


def validate(config):
    if config['representations']!=['baseline','plus_mean','plus_spread','plus_quantiles']:raise ValueError('Require all four representations')
    for key in ('ridge_alphas','nonlinear_weights','projection_seeds'):
        if not config[key] or len(config[key])!=len(set(config[key])):raise ValueError('Missing/duplicate candidates')
    if any(not np.isfinite(x) or x<=0 for x in config['ridge_alphas']):raise ValueError('Invalid alpha')
    if any(not np.isfinite(x) or not 0<x<=1 for x in config['nonlinear_weights']):raise ValueError('Invalid nonlinear mixture')
    for key in ('outer_splits','inner_splits','torch_threads','bootstrap_draws'):
        if type(config[key])!=int or config[key]<1:raise ValueError('Invalid study budget')
    if min(config['outer_splits'],config['inner_splits'])<2:raise ValueError('Invalid folds')
    for key in ('bandwidth_multiplier','scale_floor','probability_rtol','probability_atol'):
        if not np.isfinite(config[key]) or config[key]<=0:raise ValueError('Invalid numerical setting')


def load_inputs(config):
    validate(config);bank,protected=source.validate_chain(config)
    return bank,bank.assigned,protected


def names(config):return [f'{rep}__{family}' for rep in config['representations'] for family in ('linear','nonlinear')]


def candidates(config,family):
    weights=[0.] if family=='linear' else config['nonlinear_weights']
    return [dict(alpha=alpha,weight=weight) for weight in weights for alpha in config['ridge_alphas']]


def budget(config):
    groups=config['outer_splits']*len(config['projection_seeds']);n=len(config['representations'])
    linear=groups*n*(config['inner_splits']*len(candidates(config,'linear'))+1)
    nonlinear=groups*n*(config['inner_splits']*len(candidates(config,'nonlinear'))+1)
    return dict(fits=linear+nonlinear,linear_fits=linear,nonlinear_fits=nonlinear,outer_models=groups*n*2,
                feature_contexts=groups*(config['inner_splits']+1),kernel_contexts=groups*(config['inner_splits']+1)*n,
                matched_checks=groups*(config['inner_splits']+1)*n,source_equivalence_checks=linear,paired_representation_selections=groups)


def make_scope(config,bank):
    from . import residual_context_report
    files=[Path(__file__),Path(__file__).with_name('distribution_kernel_report.py')]+[Path(m.__file__) for m in (source,kernel,core,benchmark,residual_context_report)]
    scope=dict(config=config,source=bank.provenance,code={str(p):file_hash(p) for p in files},claim_scope=CLAIM,
               runtime=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=str(torch.__version__),joblib=joblib.__version__,platform=platform.platform(),device='cpu',blas_threads=1))
    return dict(scope,study_fingerprint=_hash(scope))


def prepare_context(bank,config,scope,root,seed,fold,inner,train,query,verify=False):
    values,path=source.context(bank,seed,fold,inner,train,query);datas={}
    for rep,x in values.items():
        data=kernel.KernelData(x[:len(train)],bank.labels[train],x[len(train):],len(config['classes']),config['bandwidth_multiplier'],config['scale_floor'])
        expected=core.membership(bank,scope,train,query)
        expected.update(projection_seed=seed,outer_fold=fold,inner_fold=inner,representation=rep,source_context_sha256=file_hash(path),
                        training_features_sha256=array_hash(x[:len(train)]),query_features_sha256=array_hash(x[len(train):]),state=data.state)
        own=root/'review/contexts'/f'context_{fold:02d}_{rep}_s{seed}_i{inner}.json';own.parent.mkdir(parents=True,exist_ok=True)
        if verify or own.exists():
            if json.loads(own.read_text())!=expected:raise ValueError('Kernel context membership/scales changed')
        else:_save_json(own,expected)
        datas[rep]=(data,own)
    return values,datas,path


def classifier_fit(bank,config,scope,root,seed,fold,inner,rep,family,candidate,train,query,values,data,context,source_context,verify=False):
    settings=candidates(config,family)[candidate];alpha,weight=settings['alpha'],settings['weight']
    # Rebuild and PSD-check kernels even on resume, before checking fitted state.
    _,_,state=data.kernel(weight)
    expected=core.membership(bank,scope,train,query)
    expected.update(projection_seed=seed,outer_fold=fold,inner_fold=inner,representation=rep,family=family,candidate_id=candidate,
                    alpha=alpha,weight=weight,context_sha256=file_hash(context),kernel_state=state,query_features_sha256=array_hash(values[rep][len(train):]))
    suffix=f'_i{inner}_a{candidate}' if inner else ''
    path=root/'review/fits'/f"{'inner' if inner else 'outer'}_{fold:02d}_{rep}__{family}_s{seed}{suffix}.json";path.parent.mkdir(parents=True,exist_ok=True)
    if verify or path.exists():
        audit=json.loads(path.read_text())
        if any(audit.get(k)!=v for k,v in expected.items()) or file_hash(path.with_suffix('.joblib'))!=audit['model_sha256'] or file_hash(path.with_suffix('.npz'))!=audit['probabilities_sha256']:raise ValueError('Saved kernel model scope/hash changed')
        model=joblib.load(path.with_suffix('.joblib'))
        if model.alpha!=alpha or model.weight!=weight or model.state!=state:raise ValueError('Restored model settings changed')
        p=model.predict(data)
        with np.load(path.with_suffix('.npz')) as f:
            if not np.array_equal(f['rows'],query) or not np.allclose(p,f['probabilities'],rtol=1e-10,atol=1e-12):raise ValueError('Kernel prediction replay disagrees')
    else:
        model=kernel.KernelModel.fit(data,alpha,weight);p=model.predict(data)
        core.dump_model(path.with_suffix('.joblib'),model)
        temp=path.with_suffix('.partial.npz');np.savez_compressed(temp,rows=query,probabilities=p);temp.replace(path.with_suffix('.npz'))
        audit=dict(expected,model_sha256=file_hash(path.with_suffix('.joblib')),probabilities_sha256=file_hash(path.with_suffix('.npz')))
        if inner:audit['validation_macro_f1']=float(f1_score(bank.labels[query],p.argmax(1),labels=np.arange(len(config['classes'])),average='macro',zero_division=0))
        _save_json(path,audit)
    if p.shape!=(len(query),len(config['classes'])) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1):raise ValueError('Invalid kernel probabilities')
    if inner:
        score=float(f1_score(bank.labels[query],p.argmax(1),labels=np.arange(len(config['classes'])),average='macro',zero_division=0))
        if score!=audit['validation_macro_f1']:raise ValueError('Inner F1 changed')
    if family=='linear':
        old,old_audit,old_path=source.reference(bank,seed,fold,inner,rep,candidate,train,query,values[rep],source_context)
        if (not np.allclose(p,old,rtol=config['probability_rtol'],atol=config['probability_atol'])
                or not np.array_equal(p.argmax(1),old.argmax(1)) or (inner and audit['validation_macro_f1']!=old_audit['validation_macro_f1'])):
            raise ValueError('Linear kernel fails original ridge equivalence')
        eq=dict(study_fingerprint=scope['study_fingerprint'],new_audit_sha256=file_hash(path),source_audit=str(old_path),source_audit_sha256=file_hash(old_path),
                probability_rtol=config['probability_rtol'],probability_atol=config['probability_atol'],max_absolute_probability_error=float(np.max(np.abs(p-old))),predicted_labels_equal=True)
        eq_path=root/'review/equivalence'/path.name;eq_path.parent.mkdir(parents=True,exist_ok=True)
        if verify or eq_path.exists():
            if json.loads(eq_path.read_text())!=eq:raise ValueError('Linear equivalence receipt changed')
        else:_save_json(eq_path,eq)
    _progress(root,state='verifying_kernel_models' if verify else 'fitting_kernel_models',outer_fold=fold,projection_seed=seed,inner_fold=inner,representation=rep,family=family,candidate_id=candidate)
    return p,audit


def evaluate_fold(bank,assigned,config,scope,root,seed,fold,verify=False,linear_only=False):
    train,test=np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
    splits=[(train[t],train[v]) for t,v in benchmark.inner_patient_splits(bank.labels[train],bank.ids[train],config['inner_splits'],config['split_seed']+fold*1000)]
    families=('linear',) if linear_only else ('linear','nonlinear')
    arms=[f'{rep}__{f}' for rep in config['representations'] for f in families]
    scores={a:[[] for _ in candidates(config,a.split('__')[1])] for a in arms}
    for inner,(ti,vi) in enumerate(splits,1):
        values,datas,source_ctx=prepare_context(bank,config,scope,root,seed,fold,inner,ti,vi,verify)
        for arm in arms:
            rep,family=arm.split('__');data,ctx=datas[rep]
            for candidate in range(len(scores[arm])):
                _,audit=classifier_fit(bank,config,scope,root,seed,fold,inner,rep,family,candidate,ti,vi,values,data,ctx,source_ctx,verify)
                scores[arm][candidate].append(audit['validation_macro_f1'])
    values,datas,source_ctx=prepare_context(bank,config,scope,root,seed,fold,0,train,test,verify);out={};best={}
    for arm in arms:
        rep,family=arm.split('__');data,ctx=datas[rep];candidate,means=core.select(scores[arm]);best[arm]=means[candidate]
        decision=dict(study_fingerprint=scope['study_fingerprint'],arm=arm,projection_seed=seed,outer_fold=fold,inner_scores=scores[arm],mean_inner_macro_f1=means,
                      selected_candidate=candidate,selected_settings=candidates(config,family)[candidate],tie_rule='first configured candidate')
        p=root/'review'/f'selection_{fold:02d}_{arm}_s{seed}.json'
        if verify or p.exists():
            if json.loads(p.read_text())!=decision:raise ValueError('Kernel inner choice changed')
        else:_save_json(p,decision)
        if family=='linear':
            previous=json.loads((bank.root/'review'/f'selection_{fold:02d}_{rep}_s{seed}.json').read_text())
            if previous['selected_candidate']!=candidate or previous['inner_scores']!=scores[arm]:raise ValueError('Original linear selection not reproduced')
        out[arm],_=classifier_fit(bank,config,scope,root,seed,fold,0,rep,family,candidate,train,test,values,data,ctx,source_ctx,verify)
    if not linear_only:
        rep_scores={r:float(np.mean([best[f'{r}__{f}'] for f in families])) for r in config['representations']}
        chosen=max(config['representations'],key=rep_scores.__getitem__)
        decision=dict(study_fingerprint=scope['study_fingerprint'],outer_fold=fold,projection_seed=seed,symmetric_inner_scores=rep_scores,
                      selected_representation=chosen,tie_rule='first configured representation; mean of each family best mean-inner F1')
        path=root/'review'/f'paired_selection_{fold:02d}_s{seed}.json'
        if verify or path.exists():
            if json.loads(path.read_text())!=decision:raise ValueError('Shared representation selection changed')
        else:_save_json(path,decision)
        for family in families:out[f'selected_{family}']=out[f'{chosen}__{family}']
    return test,out


def guard(config,bank):
    root=Path(config['output_dir'])/'complete';root.mkdir(parents=True,exist_ok=True);scope=make_scope(config,bank);p=root/'study_manifest.json'
    if p.exists() and json.loads(p.read_text())!=scope:raise ValueError('Kernel scope changed; use new namespace')
    if not p.exists():_save_json(p,scope)
    return root,scope


def preflight(config):
    bank,assigned,_=load_inputs(config);root,scope=guard(config,bank);start=time.perf_counter()
    members={(r,s):np.full((len(bank.ids),len(config['classes'])),np.nan) for r in config['representations'] for s in config['projection_seeds']}
    for fold in range(1,config['outer_splits']+1):
        for seed in config['projection_seeds']:
            rows,values=evaluate_fold(bank,assigned,config,scope,root,seed,fold,linear_only=True)
            for r in config['representations']:members[r,seed][rows]=values[f'{r}__linear']
    reproduced={}
    for rep in config['representations']:
        for seed in config['projection_seeds']:
            if benchmark.metric_bundle(bank.labels,members[rep,seed],config['classes'])!=bank.summary['member_metrics'][rep][str(seed)]:raise ValueError('Linear member metrics differ')
        p=np.mean([members[rep,s] for s in config['projection_seeds']],axis=0);m=benchmark.metric_bundle(bank.labels,p,config['classes'])
        if m!=bank.summary['ensemble_metrics'][rep]:raise ValueError('Linear ensemble metrics differ')
        reproduced[rep]=m['macro_f1']
    eq=list((root/'review/equivalence').glob('*.json'))
    if len(eq)!=budget(config)['linear_fits']:raise ValueError('Incomplete linear equivalence gate')
    elapsed=time.perf_counter()-start
    result=dict(complete=True,study_fingerprint=scope['study_fingerprint'],source_equivalence_checks=len(eq),original_all_arm_metrics_reproduced=True,
                reproduced_ensemble_macro_f1=reproduced,seconds=elapsed,claim='Linear equivalence only; not nonlinear performance')
    target=Path(config['output_dir'])/'preflight';target.mkdir(parents=True,exist_ok=True);_save_json(target/'summary.json',result)
    _progress(root,state='linear_preflight_complete',equivalence_checks=len(eq));return result


def run(config):
    bank,assigned,_=load_inputs(config);root,scope=guard(config,bank)
    smoke_receipt=json.loads((Path(config['output_dir'])/'smoke/summary.json').read_text())
    if smoke_receipt.get('complete') is not True or smoke_receipt['study_fingerprint']!=scope['study_fingerprint'] or smoke_receipt.get('outer_test_patients_used') is not False:raise ValueError('Require matching inner-only engineering smoke')
    receipt=json.loads((Path(config['output_dir'])/'preflight/summary.json').read_text())
    if receipt.get('complete') is not True or receipt['study_fingerprint']!=scope['study_fingerprint'] or receipt['source_equivalence_checks']!=budget(config)['linear_fits']:raise ValueError('Require linear equivalence gate')
    for fold in range(1,config['outer_splits']+1):
        for seed in config['projection_seeds']:
            evaluate_fold(bank,assigned,config,scope,root,seed,fold)
            print(f'Kernel fold {fold}, seed {seed}: saved, not final performance',flush=True)
    _progress(root,state='awaiting_independent_kernel_verification',**budget(config))


def smoke(config):
    start=time.perf_counter();bank,assigned,_=load_inputs(config);root=Path(config['output_dir'])/'smoke';root.mkdir(parents=True,exist_ok=True);scope=make_scope(config,bank)
    train=np.flatnonzero(assigned!=1);ti,vi=next(benchmark.inner_patient_splits(bank.labels[train],bank.ids[train],config['inner_splits'],config['split_seed']+1000));train,query=train[ti],train[vi]
    values,datas,source_ctx=prepare_context(bank,config,scope,root,config['projection_seeds'][0],1,1,train,query);solve_start=time.perf_counter();count=0
    for rep in config['representations']:
        data,ctx=datas[rep]
        for family in ('linear','nonlinear'):
            for candidate in range(len(candidates(config,family))):
                classifier_fit(bank,config,scope,root,config['projection_seeds'][0],1,1,rep,family,candidate,train,query,values,data,ctx,source_ctx)
                classifier_fit(bank,config,scope,root,config['projection_seeds'][0],1,1,rep,family,candidate,train,query,values,data,ctx,source_ctx,True);count+=1
    result=dict(complete=True,study_fingerprint=scope['study_fingerprint'],outer_test_patients_used=False,classifiers_fitted_and_restored=count,
                total_seconds=time.perf_counter()-start,kernel_fit_restore_seconds=time.perf_counter()-solve_start,
                new_waveform_extractions=0,claim='Inner-only engineering smoke and timing, not final model performance')
    _save_json(root/'summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/distribution_kernel_v1.yaml');g=p.add_mutually_exclusive_group();g.add_argument('--preflight',action='store_true');g.add_argument('--smoke',action='store_true');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(config['torch_threads'])
    try:
        with threadpool_limits(limits=1):result=preflight(config) if args.preflight else smoke(config) if args.smoke else run(config)
        if result:print(json.dumps(result))
    except Exception as error:
        target=Path(config['output_dir'])/'complete';target.mkdir(parents=True,exist_ok=True);_progress(target,state='failed',error=str(error));raise
