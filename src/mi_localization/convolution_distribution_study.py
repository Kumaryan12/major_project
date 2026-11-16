"""Four-arm intact-base convolution distributions, with exact source partitions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import joblib
import numpy as np
import scipy
import sklearn
import torch
import yaml
from threadpoolctl import threadpool_limits

from . import benchmark, distribution_study as core, distribution_features
from . import convolution_distribution_source as source, convolution_distribution_features as features
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash,_save_json
from .signal_audit import file_hash

CLAIM='Intact-base all-beat convolution-distribution PTB development; repeated patient cohort, no independent confirmation, physiological distribution model or established SOTA.'
membership=core.membership
select=core.select
dump_model=core.dump_model


def validate(config):
    if config['arms']!=list(features.ARMS):raise ValueError('Require all four arms')
    for key in ('projection_seeds','ridge_alphas'):
        if not config[key] or len(config[key])!=len(set(config[key])):raise ValueError('Duplicate/missing seeds or candidates')
    if any(type(s)!=int or s<0 for s in config['projection_seeds']):raise ValueError('Invalid seed')
    if any(not np.isfinite(a) or a<=0 for a in config['ridge_alphas']):raise ValueError('Invalid alpha')
    for key in ('projections','smoke_beats_per_record','benchmark_records','torch_threads','bootstrap_draws'):
        if type(config[key])!=int or config[key]<1:raise ValueError('Invalid budget')
    q=np.asarray(config['quantiles'],dtype=float)
    if len(q)<3 or not np.isfinite(q).all() or np.any(q<=0) or np.any(q>=1) or np.any(np.diff(q)<=0):raise ValueError('Invalid quantile grid')
    if config['outer_splits']<2 or config['inner_splits']<2 or config['branch_scale_floor']<=0 or not np.isfinite(config['branch_scale_floor']):raise ValueError('Invalid partitions/gain floor')


def load_inputs(config,verify=False):
    validate(config)
    return source.load(config)


def budget(config):
    groups=config['outer_splits']*len(config['projection_seeds'])
    contexts=groups*(config['inner_splits']+1)
    source_fits=groups*(config['inner_splits']*len(config['ridge_alphas'])+1)
    return dict(feature_contexts=contexts,fits=4*source_fits,new_fits=3*source_fits,source_ridge_contexts=source_fits,
                outer_models=groups*4,new_outer_models=groups*3,source_outer_models=groups)


def make_scope(config,bank):
    from . import rocket_features,rocket_study,rocket_bank,chronology_bank,residual_context_report
    import inspect
    modules=[source,features,core,distribution_features,benchmark,rocket_features,rocket_study,rocket_bank,chronology_bank,residual_context_report]
    files=[Path(m.__file__) for m in modules]+[Path(__file__),Path(__file__).with_name('convolution_distribution_report.py'),Path(inspect.getfile(rocket_features.HydraMultivariate))]
    scope=dict(config=config,source=bank.provenance,code={str(p):file_hash(p) for p in files},
               folds_sha256=file_hash(config['folds_path']),claim_scope=CLAIM,
               runtime=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=str(torch.__version__),
                            joblib=joblib.__version__,platform=platform.platform(),device='cpu',blas_threads=1))
    return dict(scope,study_fingerprint=_hash(scope))


def require_preflight(config,bank):
    path=Path(config['output_dir'])/'source_preflight/summary.json'
    value=json.loads(path.read_text());b=budget(config)
    if (value.get('complete') is not True or value.get('source_fingerprint')!=_hash(bank.provenance)
            or value.get('verification_code_sha256')!=file_hash(source.__file__)
            or value.get('source_models_restored')!=b['source_ridge_contexts']
            or value.get('source_contexts_reconstructed')!=b['feature_contexts']
            or value.get('source_files_unchanged') is not True):raise ValueError('Require matching independently replayed source baseline')
    return value


def feature_context(bank,config,scope,root,seed,fold,inner,train,query,verify=False,smoke=False):
    transform,source_x,source_path=source.context(bank,seed,fold,inner,train,query,reconstruct=verify)
    expected=membership(bank,scope,train,query)
    expected.update(projection_seed=seed,outer_fold=fold,inner_fold=inner,source_context_sha256=file_hash(source_path),
                    coverage_mode='smoke_first_beats' if smoke else 'all_original_beats')
    directory=root/'review/contexts';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'context_{fold:02d}_s{seed}_i{inner}.json';params_path=path.with_suffix('.joblib');values_path=path.with_suffix('.npz')
    rows=np.r_[train,query]
    if verify or path.exists():
        audit=json.loads(path.read_text())
        if (any(audit.get(k)!=v for k,v in expected.items()) or file_hash(values_path)!=audit['features_sha256']
                or file_hash(params_path)!=audit['parameters_sha256']):raise ValueError('Convolution-distribution context changed')
        with np.load(values_path) as f:
            if not np.array_equal(f['rows'],rows):raise ValueError('Context row order changed')
            values={a:f[a] for a in config['arms']}
        params=joblib.load(params_path)
        for name,hkey in [('projection','projection_sha256'),('scaler_mean','scaler_mean_sha256'),('scaler_scale','scaler_scale_sha256')]:
            if array_hash(params[name])!=audit['state'][hkey]:raise ValueError('Restored auxiliary parameters changed')
        if params['residual_rms']!=audit['state']['residual_rms']:raise ValueError('Restored training gain changed')
        for name,expected_hash in audit['state']['record_audit_hashes'].items():
            record=root/name
            if file_hash(record)!=expected_hash or file_hash(record.with_suffix('.npz'))!=json.loads(record.read_text())['data_sha256']:
                raise ValueError('Record cache changed after context completion')
        if verify:
            rebuilt,state,_=features.make_features(bank,config,root,scope,transform,source_x,source_path,train,query,seed,fold,inner,True,smoke)
            if state!=audit['state'] or any(not np.allclose(rebuilt[a],values[a],rtol=1e-10,atol=1e-12) for a in config['arms']):
                raise ValueError('All-beat pooling reconstruction differs')
    else:
        values,state,params=features.make_features(bank,config,root,scope,transform,source_x,source_path,train,query,seed,fold,inner,False,smoke)
        dump_model(params_path,params)
        temp=values_path.with_suffix('.partial.npz');np.savez_compressed(temp,rows=rows,**values);temp.replace(values_path)
        audit=dict(expected,state=state,parameters_sha256=file_hash(params_path),features_sha256=file_hash(values_path),dimensions={a:int(v.shape[1]) for a,v in values.items()})
        _save_json(path,audit)
    d=source_x.shape[1];extra=config['projections']*len(config['quantiles'])
    if any(v.shape!=(len(rows),d+(extra if a!='baseline' else 0)) or not np.isfinite(v).all() for a,v in values.items()):raise ValueError('Invalid pooled feature dimensions')
    # The base block stays bit-identical in all augmented arms.
    if any(not np.array_equal(values[a][:,:d],values['baseline']) for a in config['arms'][1:]):raise ValueError('Full baseline vector was altered')
    values['_source']=source_x
    return values,path


def classifier_fit(bank,config,scope,root,seed,fold,inner,arm,candidate,train,query,values,context,verify=False):
    if arm!='baseline':return core.classifier_fit(bank,config,scope,root,seed,fold,inner,arm,candidate,train,query,values,context,verify)
    p,old,old_path=source.ridge(bank,seed,fold,inner,candidate,train,query,values['_source'])
    expected=membership(bank,scope,train,query)
    expected.update(source_audit=str(old_path),source_audit_sha256=file_hash(old_path),context_sha256=file_hash(context),candidate_id=candidate,
                    projection_seed=seed,outer_fold=fold,inner_fold=inner,arm=arm,validation_macro_f1=old.get('validation_macro_f1'))
    suffix=f'_i{inner}_a{candidate}' if inner else ''
    path=root/'review/baseline_refs'/f"{'inner' if inner else 'outer'}_{fold:02d}_s{seed}{suffix}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    if verify or path.exists():
        if json.loads(path.read_text())!=expected:raise ValueError('Frozen baseline reuse changed')
    else:_save_json(path,expected)
    return p,expected


def evaluate_fold(bank,assigned,config,scope,root,seed,fold,verify=False):
    train,test=np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
    splits=[(train[t],train[v]) for t,v in benchmark.inner_patient_splits(bank.labels[train],bank.ids[train],config['inner_splits'],config['split_seed']+fold*1000)]
    scores={a:[[] for _ in config['ridge_alphas']] for a in config['arms']}
    for inner,(ti,vi) in enumerate(splits,1):
        values,ctx=feature_context(bank,config,scope,root,seed,fold,inner,ti,vi,verify)
        for arm in config['arms']:
            for candidate in range(len(config['ridge_alphas'])):
                _,audit=classifier_fit(bank,config,scope,root,seed,fold,inner,arm,candidate,ti,vi,values,ctx,verify)
                scores[arm][candidate].append(audit['validation_macro_f1'])
    values,ctx=feature_context(bank,config,scope,root,seed,fold,0,train,test,verify)
    result={}
    for arm in config['arms']:
        candidate,means=select(scores[arm])
        decision=dict(study_fingerprint=scope['study_fingerprint'],arm=arm,projection_seed=seed,outer_fold=fold,
                      inner_scores=scores[arm],mean_inner_macro_f1=means,selected_candidate=candidate,selected_alpha=config['ridge_alphas'][candidate],tie_rule='first configured alpha')
        path=root/'review'/f'selection_{fold:02d}_{arm}_s{seed}.json'
        if verify or path.exists():
            if json.loads(path.read_text())!=decision:raise ValueError('Nested choices changed')
        else:_save_json(path,decision)
        result[arm],_=classifier_fit(bank,config,scope,root,seed,fold,0,arm,candidate,train,test,values,ctx,verify)
    return test,result


def run(config):
    root=Path(config['output_dir'])/'complete';root.mkdir(parents=True,exist_ok=True)
    _progress(root,state='checking_frozen_source')
    bank,assigned=load_inputs(config);require_preflight(config,bank);scope=make_scope(config,bank)
    path=root/'study_manifest.json'
    if path.exists() and json.loads(path.read_text())!=scope:raise ValueError('Scope changed; use a new namespace')
    _save_json(path,scope)
    for fold in range(1,config['outer_splits']+1):
        for seed in config['projection_seeds']:
            evaluate_fold(bank,assigned,config,scope,root,seed,fold)
            print(f'Convolution distribution seed {seed}, fold {fold}: saved, not final performance',flush=True)
    _progress(root,state='awaiting_independent_verification',**budget(config))


def smoke(config):
    bank,assigned=load_inputs(config);scope=make_scope(config,bank);root=Path(config['output_dir'])/'smoke';root.mkdir(parents=True,exist_ok=True)
    tr=np.flatnonzero(assigned!=1);ti,vi=next(benchmark.inner_patient_splits(bank.labels[tr],bank.ids[tr],config['inner_splits'],config['split_seed']+1000))
    train,query=tr[ti],tr[vi];seed=config['projection_seeds'][0]
    values,ctx=feature_context(bank,config,scope,root,seed,1,1,train,query,smoke=True)
    for arm in config['arms']:
        classifier_fit(bank,config,scope,root,seed,1,1,arm,0,train,query,values,ctx)
        classifier_fit(bank,config,scope,root,seed,1,1,arm,0,train,query,values,ctx,True)
    feature_context(bank,config,scope,root,seed,1,1,train,query,True,True)
    result=dict(complete=True,study_fingerprint=scope['study_fingerprint'],outer_test_patients_used=False,
                original_base_unchanged=True,new_classifiers_restored=3,source_classifier_restored=True,
                dimensions={a:v.shape[1] for a,v in values.items() if a in config['arms']},
                claim='Engineering inner-partition smoke, only first two beats/record for auxiliary features; no final performance or full coverage claim')
    _save_json(root/'summary.json',result);return result


def benchmark_cost(config):
    bank,assigned=load_inputs(config);tr=np.flatnonzero(assigned!=1)
    ti,vi=next(benchmark.inner_patient_splits(bank.labels[tr],bank.ids[tr],config['inner_splits'],config['split_seed']+1000))
    transform,_,_=source.context(bank,config['projection_seeds'][0],1,1,tr[ti],tr[vi])
    records=bank.records_for(bank.ids[tr[ti]])[:config['benchmark_records']];beats=0;start=time.perf_counter()
    for record in records:
        rows=bank.rows(record);x=transform.transform(np.asarray(bank.records.waves[rows]))['combined'];beats+=len(rows)
        if not np.isfinite(x).all():raise ValueError('Nonfinite benchmark')
    elapsed=time.perf_counter()-start
    result=dict(training_only=True,outer_test_patients_used=False,records=len(records),beats=beats,seconds=elapsed,seconds_per_beat=elapsed/beats,
                conservative_two_pass_hours=2*budget(config)['feature_contexts']*len(bank.records.waves)*elapsed/beats/3600,
                claim='Training-only timing probe, rough extrapolation, not a performance result or guaranteed ETA')
    try:
        import resource
        result['peak_resident_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:result['peak_resident_bytes']=None
    _save_json(Path(config['output_dir'])/'benchmark.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/convolution_distribution_v1.yaml')
    g=p.add_mutually_exclusive_group();g.add_argument('--preflight',action='store_true');g.add_argument('--smoke',action='store_true');g.add_argument('--benchmark',action='store_true')
    args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(config['torch_threads'])
    with threadpool_limits(limits=1):
        if args.preflight:
            bank,assigned=load_inputs(config);result=source.verify_baseline(bank,assigned,config)
        elif args.smoke:result=smoke(config)
        elif args.benchmark:result=benchmark_cost(config)
        else:result=run(config)
    if result:print(json.dumps(result))
