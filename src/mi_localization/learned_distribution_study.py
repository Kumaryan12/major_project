"""Frozen learned morphology distributions with an intact convolution baseline."""
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
from threadpoolctl import threadpool_limits
from . import benchmark,distribution_study as core,convolution_distribution_features as pooling
from . import learned_distribution_source as source,learned_distribution_features as features
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash,_save_json
from .signal_audit import file_hash

CLAIM='Frozen partition-local learned morphology distributions on repeatedly inspected PTB development patients; no independent confirmation, physiological identification, clinical validity or established SOTA.'


def validate(config):
    if config['arms']!=list(pooling.ARMS) or config['representations']!=list(pooling.ARMS):raise ValueError('Require all four pooling controls')
    for key in ('projection_seeds','ridge_alphas'):
        if not config[key] or len(config[key])!=len(set(config[key])):raise ValueError('Missing/duplicate budget')
    if any(not np.isfinite(x) or x<=0 for x in config['ridge_alphas']):raise ValueError('Invalid alpha')
    for key in ('projections','embedding_batch_size','torch_threads','bootstrap_draws'):
        if type(config[key])!=int or config[key]<1:raise ValueError('Invalid dimension/budget')
    q=np.asarray(config['quantiles'])
    if len(q)<3 or np.any(q<=0) or np.any(q>=1) or not np.isfinite(q).all() or np.any(np.diff(q)<=0):raise ValueError('Invalid quantiles')
    if config['device']!='cpu':raise ValueError('This scope pins deterministic CPU embedding inference')
    for k in ('branch_scale_floor','teacher_probability_rtol','teacher_probability_atol','embedding_replay_rtol','embedding_replay_atol'):
        if not np.isfinite(config[k]) or config[k]<=0:raise ValueError('Invalid tolerance/floor')


def load_inputs(config,verify=False):validate(config);return source.load(config)


def budget(config):
    groups=config['outer_splits']*len(config['projection_seeds']);source_fits=groups*(config['inner_splits']*len(config['ridge_alphas'])+1)
    return dict(fits=4*source_fits,new_fits=3*source_fits,source_ridge_contexts=source_fits,outer_models=4*groups,new_outer_models=3*groups,
                source_outer_models=groups,feature_contexts=groups*(config['inner_splits']+1),teacher_models=groups*(config['inner_splits']+1))


def make_scope(config,bank):
    from . import residual_context_model,chronology_model,residual_context_report,distribution_features
    modules=(source,features,source.cached,core,pooling,benchmark,residual_context_model,chronology_model,residual_context_report,distribution_features)
    paths=[Path(m.__file__) for m in modules]+[Path(__file__),Path(__file__).with_name('learned_distribution_report.py')]
    scope=dict(config=config,source=bank.provenance,code={str(p):file_hash(p) for p in paths},claim_scope=CLAIM,
               runtime=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=str(torch.__version__),joblib=joblib.__version__,platform=platform.platform(),device=config['device'],blas_threads=1))
    return dict(scope,study_fingerprint=_hash(scope))


def guard(config,bank):
    root=Path(config['output_dir'])/'complete';root.mkdir(parents=True,exist_ok=True);scope=make_scope(config,bank);path=root/'study_manifest.json'
    if path.exists() and json.loads(path.read_text())!=scope:raise ValueError('Learned distribution scope changed; use new namespace')
    if not path.exists():_save_json(path,scope)
    return root,scope


def feature_context(bank,config,scope,root,seed,fold,inner,train,query,verify=False):
    expected=core.membership(bank,scope,train,query);expected.update(projection_seed=seed,outer_fold=fold,inner_fold=inner,coverage_mode='all_original_beats')
    directory=root/'review/contexts';directory.mkdir(parents=True,exist_ok=True);path=directory/f'context_{fold:02d}_s{seed}_i{inner}.json'
    if verify or path.exists():
        a=json.loads(path.read_text())
        if any(a.get(k)!=v for k,v in expected.items()) or file_hash(path.with_suffix('.npz'))!=a['features_sha256'] or file_hash(path.with_suffix('.joblib'))!=a['parameters_sha256']:raise ValueError('Learned feature context changed')
        with np.load(path.with_suffix('.npz')) as f:
            if not np.array_equal(f['rows'],np.r_[train,query]):raise ValueError('Pooled patient order changed')
            values={arm:f[arm] for arm in config['arms']}
        params=joblib.load(path.with_suffix('.joblib'))
        for key in ('scaler_mean','scaler_scale','projection'):
            if array_hash(params[key])!=a['state'][key+'_sha256']:raise ValueError('Fitted embedding scales/projections changed')
        if params['residual_rms']!=a['state']['residual_rms']:raise ValueError('Fitted embedding gain changed')
        for name,h in a['state']['record_audit_hashes'].items():
            p=root/name
            if file_hash(p)!=h or file_hash(p.with_suffix('.npz'))!=json.loads(p.read_text())['data_sha256']:raise ValueError('Completed embedding bank changed')
        _,base_context=source.cached.context(bank.cache,seed,fold,inner,train,query)
        if verify:
            rebuilt,state,_,_=features.make_features(bank,config,scope,root,seed,fold,inner,train,query,True)
            if state!=a['state'] or any(not np.allclose(rebuilt[arm],values[arm],rtol=1e-10,atol=1e-12) for arm in config['arms']):raise ValueError('Learned summary reconstruction disagrees')
    else:
        values,state,params,base_context=features.make_features(bank,config,scope,root,seed,fold,inner,train,query)
        core.dump_model(path.with_suffix('.joblib'),params)
        temp=path.with_suffix('.partial.npz');np.savez_compressed(temp,rows=np.r_[train,query],**values);temp.replace(path.with_suffix('.npz'))
        state['coverage_mode']='all_original_beats'
        a=dict(expected,state=state,features_sha256=file_hash(path.with_suffix('.npz')),parameters_sha256=file_hash(path.with_suffix('.joblib')),dimensions={arm:x.shape[1] for arm,x in values.items()});_save_json(path,a)
    d=a['state']['base_dimensions'];extra=config['projections']*len(config['quantiles'])
    if any(x.shape!=(len(train)+len(query),d+(0 if arm=='baseline' else extra)) or not np.isfinite(x).all() for arm,x in values.items()):raise ValueError('Invalid learned pooled dimensions')
    if any(not np.array_equal(values[arm][:,:d],values['baseline']) for arm in config['arms'][1:]):raise ValueError('Intact morphology base changed')
    return values,path,base_context


def classifier_fit(bank,config,scope,root,seed,fold,inner,arm,candidate,train,query,values,context,base_context,verify=False):
    if arm!='baseline':return core.classifier_fit(bank,config,scope,root,seed,fold,inner,arm,candidate,train,query,values,context,verify)
    p,old,old_path=source.cached.reference(bank.cache,seed,fold,inner,'baseline',candidate,train,query,values['baseline'],base_context)
    a=core.membership(bank,scope,train,query);a.update(source_audit=str(old_path),source_audit_sha256=file_hash(old_path),context_sha256=file_hash(context),candidate_id=candidate,
            projection_seed=seed,outer_fold=fold,inner_fold=inner,arm=arm,validation_macro_f1=old.get('validation_macro_f1'))
    suffix=f'_i{inner}_a{candidate}' if inner else '';path=root/'review/baseline_refs'/f"{'inner' if inner else 'outer'}_{fold:02d}_s{seed}{suffix}.json";path.parent.mkdir(parents=True,exist_ok=True)
    if verify or path.exists():
        if json.loads(path.read_text())!=a:raise ValueError('Baseline reuse context changed')
    else:_save_json(path,a)
    return p,a


def evaluate_fold(bank,assigned,config,scope,root,seed,fold,verify=False):
    train,test=np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
    scores={arm:[[] for _ in config['ridge_alphas']] for arm in config['arms']}
    splits=[(train[t],train[v]) for t,v in benchmark.inner_patient_splits(bank.labels[train],bank.ids[train],config['inner_splits'],config['split_seed']+fold*1000)]
    for inner,(ti,vi) in enumerate(splits,1):
        values,ctx,base_ctx=feature_context(bank,config,scope,root,seed,fold,inner,ti,vi,verify)
        for arm in config['arms']:
            for candidate in range(len(config['ridge_alphas'])):
                _,a=classifier_fit(bank,config,scope,root,seed,fold,inner,arm,candidate,ti,vi,values,ctx,base_ctx,verify);scores[arm][candidate].append(a['validation_macro_f1'])
    values,ctx,base_ctx=feature_context(bank,config,scope,root,seed,fold,0,train,test,verify);result={}
    for arm in config['arms']:
        candidate,means=core.select(scores[arm]);decision=dict(study_fingerprint=scope['study_fingerprint'],arm=arm,projection_seed=seed,outer_fold=fold,
                inner_scores=scores[arm],mean_inner_macro_f1=means,selected_candidate=candidate,selected_alpha=config['ridge_alphas'][candidate],tie_rule='first configured alpha')
        path=root/'review'/f'selection_{fold:02d}_{arm}_s{seed}.json'
        if verify or path.exists():
            if json.loads(path.read_text())!=decision:raise ValueError('Learned ridge selection changed')
        else:_save_json(path,decision)
        result[arm],_=classifier_fit(bank,config,scope,root,seed,fold,0,arm,candidate,train,test,values,ctx,base_ctx,verify)
    return test,result


def audit(config):
    bank,assigned,_=load_inputs(config);scope=make_scope(config,bank);start=time.perf_counter();count=restored=0
    members={seed:np.full((len(bank.ids),len(config['classes'])),np.nan) for seed in config['projection_seeds']}
    for fold in range(1,config['outer_splits']+1):
        train,test=np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
        splits=[(train[t],train[v]) for t,v in benchmark.inner_patient_splits(bank.labels[train],bank.ids[train],config['inner_splits'],config['split_seed']+fold*1000)]
        for seed in config['projection_seeds']:
            scores=[[] for _ in config['ridge_alphas']]
            for inner,(tr,qu) in enumerate(splits+[(train,test)],1):
                actual_inner=inner if inner<=config['inner_splits'] else 0
                model,_,_,_,_=source.restore_teacher(bank,config,seed,fold,actual_inner,tr,qu);del model;count+=1
                values,ctx=source.cached.context(bank.cache,seed,fold,actual_inner,tr,qu)
                if actual_inner:
                    for candidate in range(len(scores)):
                        _,a,_=source.cached.reference(bank.cache,seed,fold,actual_inner,'baseline',candidate,tr,qu,values['baseline'],ctx);scores[candidate].append(a['validation_macro_f1']);restored+=1
                else:
                    candidate,_=core.select(scores);p,_,_=source.cached.reference(bank.cache,seed,fold,0,'baseline',candidate,tr,qu,values['baseline'],ctx);members[seed][test]=p;restored+=1
                target=Path(config['output_dir'])/'teacher_audit';target.mkdir(parents=True,exist_ok=True);_progress(target,state='auditing_teacher_partitions',teacher_models=count,source_ridge_restorations=restored,outer_fold=fold,projection_seed=seed,inner_fold=actual_inner)
    ensemble=np.mean(list(members.values()),axis=0);m=benchmark.metric_bundle(bank.labels,ensemble,config['classes'])
    if m!=bank.cache.summary['ensemble_metrics']['baseline']:raise ValueError('Source baseline metric reproduction failed')
    result=dict(complete=True,study_fingerprint=scope['study_fingerprint'],teacher_models_restored=count,source_ridge_restorations=restored,training_normalizations_reconstructed=len(bank.normalizers),
                baseline_ensemble_macro_f1=m['macro_f1'],seconds=time.perf_counter()-start,claim='Checkpoint partitions/normalization/metadata and cached baseline gate; full teacher inference verified in embedding contexts, not a new localization result')
    _save_json(target/'summary.json',result);_progress(target,state='complete_verified',teacher_models=count,source_ridge_restorations=restored);return result


def require_audit(config,scope):
    a=json.loads((Path(config['output_dir'])/'teacher_audit/summary.json').read_text());b=budget(config)
    if a.get('complete') is not True or a['study_fingerprint']!=scope['study_fingerprint'] or a['teacher_models_restored']!=b['teacher_models'] or a['source_ridge_restorations']!=b['source_ridge_contexts']:raise ValueError('Require matching complete teacher audit')


def run(config):
    bank,assigned,_=load_inputs(config);root,scope=guard(config,bank);require_audit(config,scope)
    smoke=json.loads((Path(config['output_dir'])/'smoke/summary.json').read_text())
    if smoke.get('complete') is not True or smoke['study_fingerprint']!=scope['study_fingerprint'] or smoke['outer_test_patients_used'] is not False:raise ValueError('Require matching inner-only smoke')
    for fold in range(1,config['outer_splits']+1):
        for seed in config['projection_seeds']:
            evaluate_fold(bank,assigned,config,scope,root,seed,fold);print(f'Learned distributions fold {fold}, seed {seed}: saved, not final performance',flush=True)
    _progress(root,state='awaiting_independent_embedding_replay',**budget(config))


def smoke(config):
    bank,assigned,_=load_inputs(config);scope=make_scope(config,bank);root=Path(config['output_dir'])/'smoke';root.mkdir(parents=True,exist_ok=True)
    tr=np.flatnonzero(assigned!=1);ti,vi=next(benchmark.inner_patient_splits(bank.labels[tr],bank.ids[tr],config['inner_splits'],config['split_seed']+1000));train,query=tr[ti],tr[vi]
    seed=config['projection_seeds'][0];values,ctx,base=feature_context(bank,config,scope,root,seed,1,1,train,query)
    for arm in config['arms']:
        classifier_fit(bank,config,scope,root,seed,1,1,arm,0,train,query,values,ctx,base)
        classifier_fit(bank,config,scope,root,seed,1,1,arm,0,train,query,values,ctx,base,True)
    feature_context(bank,config,scope,root,seed,1,1,train,query,True)
    result=dict(complete=True,study_fingerprint=scope['study_fingerprint'],outer_test_patients_used=False,new_classifiers_restored=3,source_classifier_restored=True,
                all_inner_partition_beats_encoded=True,source_teacher_query_predictions_reproduced=True,dimensions={a:x.shape[1] for a,x in values.items()},claim='Inner-only full-beat engineering smoke, not final performance')
    _save_json(root/'summary.json',result);return result


def benchmark_cost(config):
    start=time.perf_counter();bank,assigned,_=load_inputs(config);tr=np.flatnonzero(assigned!=1)
    ti,vi=next(benchmark.inner_patient_splits(bank.labels[tr],bank.ids[tr],config['inner_splits'],config['split_seed']+1000));train,query=tr[ti],tr[vi]
    model,norm,_,_,_=source.restore_teacher(bank,config,config['projection_seeds'][0],1,1,train,query)
    records=bank.records_for(bank.ids[train])[:4];t=time.perf_counter();beats=0
    for r in records:
        rows=bank.rows(r);source.encode(model,norm,np.asarray(bank.records.waves[rows]),config['embedding_batch_size']);beats+=len(rows)
    elapsed=time.perf_counter()-t;result=dict(training_only=True,outer_test_patients_used=False,beats=beats,records=len(records),seconds=elapsed,seconds_per_beat=elapsed/beats,
                rough_two_pass_hours=2*budget(config)['feature_contexts']*len(bank.records.waves)*elapsed/beats/3600,total_seconds=time.perf_counter()-start,claim='Training-only timing extrapolation, not final performance or guaranteed ETA')
    target=Path(config['output_dir']);target.mkdir(parents=True,exist_ok=True);_save_json(target/'benchmark.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/learned_distribution_v1.yaml');g=p.add_mutually_exclusive_group();g.add_argument('--audit',action='store_true');g.add_argument('--smoke',action='store_true');g.add_argument('--benchmark',action='store_true');args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(config['torch_threads'])
    try:
        with threadpool_limits(limits=1):result=audit(config) if args.audit else smoke(config) if args.smoke else benchmark_cost(config) if args.benchmark else run(config)
        if result:print(json.dumps(result))
    except Exception as error:
        target=Path(config['output_dir'])/('smoke' if args.smoke else 'teacher_audit' if args.audit else 'complete');target.mkdir(parents=True,exist_ok=True);_progress(target,state='failed',error=str(error));raise
