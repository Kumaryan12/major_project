"""Partition-local morphology pretraining followed by controlled context adaptation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from . import staged_context_model as models,residual_context_study as reference
from .benchmark import inner_patient_splits,load_or_create_folds,metric_bundle
from .contrastive import _progress,initial_state_hash
from .neural import _device,_hash,_save_json
from .signal_audit import file_hash

CLAIM="Staged morphology/context development on repeatedly inspected PTB patient folds; fold-local teachers, prespecified seed ensembles, no independent confirmation or established SOTA."


def load_inputs(config):
    if config['arms']!=list(models.ARMS) or not config['training_seeds'] or len(set(config['training_seeds']))!=len(config['training_seeds']):
        raise ValueError('Require all staged controls and unique seeds')
    for key in ['sequence_length','windows_per_epoch','batch_size','inference_batch_size','max_epochs','patience','stage2_seed_offset']:
        if type(config[key])!=int or config[key]<1:raise ValueError('Invalid staged budget')
    if config['windows_per_epoch']%config['batch_size']:raise ValueError('Require full optimizer batches')
    bank=reference.chronology_bank.prepare_bank(config)
    if not Path(config['folds_path']).is_file():raise ValueError('Require frozen patient folds')
    folds=load_or_create_folds(bank.y,bank.groups,config)
    if sorted(folds.fold.unique())!=list(range(1,config['outer_splits']+1)):raise ValueError('Staged patient folds changed')
    return bank,folds.set_index('patient_id').loc[bank.ids,'fold'].to_numpy()


def make_scope(config,bank):
    original=reference.make_scope(config,bank)
    scope=dict(config=config,source_protocol=original,claim_scope=CLAIM,code_sha256=file_hash(__file__),
               model_code_sha256=file_hash(models.__file__),report_code_sha256=file_hash(Path(__file__).with_name('staged_context_report.py')))
    return dict(scope,study_fingerprint=_hash(scope))


def path_for(root,fold,seed,inner,arm):
    return root/(f'inner_{fold:02d}_{arm}_s{seed}_i{inner}.json' if inner else f'outer_{fold:02d}_{arm}_s{seed}.json')


def progress_callback(root,config,seed,fold,inner,phase,arm,epochs):
    def callback(row):
        values=dict(epoch=row['epoch'],training_loss=row['training_loss'])
        if 'gate_mean' in row:values['gate_mean']=row['gate_mean']
        if inner:values.update(inner_validation_macro_f1=row['validation_patient_macro_f1'],max_epochs=config['max_epochs'])
        else:values['final_epochs']=epochs
        _progress(root,state='inner_training' if inner else 'outer_training',phase=phase,arm=arm,training_seed=seed,
                  outer_fold=fold,inner_fold=inner,sequence_length=config['sequence_length'],**values)
    return callback


def save_fit(path,expected,model,norm,fit,ids,p,coverage):
    temporary=path.with_suffix('.partial.pt')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},normalization=norm.audit(),expected=expected),temporary)
    temporary.replace(path.with_suffix('.pt'))
    audit=dict(expected,fit=fit,coverage=coverage,model_sha256=file_hash(path.with_suffix('.pt')),
               probabilities_sha256=reference.cache.save_result(path,ids,p))
    _save_json(path,audit)
    return audit


def verify_adapter(fit,model,teacher,bank,train,valid,config,seed,arm,norm):
    initial=models.initialize(config,seed,arm,teacher['state'])
    initial_hash=initial_state_hash(initial);base_hash=models.morphology_hash(initial)
    if (fit['initial_state_sha256']!=initial_hash or fit['initial_morphology_sha256']!=base_hash
            or fit['parameter_count']!=sum(p.numel() for p in initial.parameters())
            or fit['trainable_parameter_count']!=sum(p.numel() for p in initial.parameters() if p.requires_grad)
            or fit['normalization']!=norm.audit() or fit['seed']!=seed or fit['arm']!=arm or fit['phase']!='adaptation'
            or fit['patient_overlap'] or fit['train_patients']!=len(train) or fit['validation_patients']!=(len(valid) if valid is not None else 0)
            or fit['direct_dropout_enabled'] is not False or fit['fresh_optimizer'] is not True):
        raise ValueError('Staged initialization/trainability/normalization changed')
    del initial
    history=fit['history']
    if [r['epoch'] for r in history]!=list(range(1,len(history)+1)) or len(history)>config['max_epochs']:
        raise ValueError('Staged adaptation history changed')
    best=teacher['fit']['best_validation_macro_f1'] if valid is not None else None;epoch=0
    if fit['initial_validation_macro_f1']!=best:raise ValueError('Epoch-zero teacher validation changed')
    for row in history:
        plan=reference.chronology_model.training_schedule(bank,train,config['sequence_length'],config['windows_per_epoch'],seed,row['epoch'])
        expected=reference.models.ResidualDataset(bank,norm,plan,config['sequence_length'],models.mapped_arm(arm),seed,row['epoch']).audit()
        if any(row.get(k)!=v for k,v in expected.items()) or not np.isfinite(row['training_loss']) or not 0<=row['gate_mean']<=1:
            raise ValueError('Staged windows/loss/permutation changed')
        if arm.startswith('frozen_') and row['morphology_sha256']!=base_hash:raise ValueError('Frozen morphology drifted')
        if valid is not None:
            score=row['validation_patient_macro_f1']
            if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid staged inner score')
            if score>best+1e-8:best,epoch=score,row['epoch']
        elif 'validation_patient_macro_f1' in row:raise ValueError('Staged outer fit monitored validation')
    if valid is not None:
        if (fit['selected_epoch']!=epoch or fit['best_validation_macro_f1']!=best
                or (len(history)<config['max_epochs'] and len(history)-epoch!=config['patience'])):
            raise ValueError('Staged epoch-zero/early-stopping decision changed')
    elif fit['selected_epoch']!=len(history) or fit['best_validation_macro_f1'] is not None:
        raise ValueError('Staged outer refit duration changed')
    if fit['optimizer_steps']!=len(history)*(config['windows_per_epoch']//config['batch_size']):raise ValueError('Staged optimizer exposure changed')
    if fit['final_morphology_sha256']!=models.morphology_hash(model):raise ValueError('Saved morphology hash changed')
    if arm.startswith('frozen_') and fit['final_morphology_sha256']!=base_hash:raise ValueError('Saved frozen morphology changed')


def phase_fit(bank,config,scope,root,seed,fold,inner,arm,train,query,epochs,teacher=None,choices=None,verify=False,norm_memo=None):
    if norm_memo is None:norm_memo={}
    phase='morphology' if teacher is None else 'adaptation';valid=query if inner else None
    fit_seed=seed+fold*1000+inner if inner else seed+fold
    if teacher is not None:fit_seed+=config['stage2_seed_offset']
    expected=reference.cache.expected_fit(bank,scope,arm,seed,fold,inner,train,query)
    expected.update(phase=phase,sequence_length=config['sequence_length'])
    if not inner:expected.update(final_epochs=epochs,inner_choices=choices)
    if teacher is not None:
        if (teacher['phase']!='morphology' or teacher['study_fingerprint']!=scope['study_fingerprint']
                or teacher['train_membership_sha256']!=expected['train_membership_sha256']
                or teacher['query_membership_sha256']!=expected['query_membership_sha256']):
            raise ValueError('Teacher was not fitted for this exact patient partition')
        expected.update(teacher_model_sha256=teacher['model_sha256'],teacher_audit_sha256=teacher['audit_sha256'],teacher_selected_epoch=teacher['fit']['selected_epoch'])
    path=path_for(root,fold,seed,inner,arm)
    if verify or path.is_file():
        audit,p=reference.cache.read_result(path,expected,query,len(config['classes']),outer=True)
        state=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
        if state['expected']!=expected or state['normalization']!=audit['fit']['normalization']:
            raise ValueError('Staged saved checkpoint metadata changed')
    else:
        callback=progress_callback(root,config,seed,fold,inner,phase,arm,epochs)
        if teacher is None:
            model,norm,fit,p,coverage=reference.models.fit(bank,train,valid,config,fit_seed,'beat_only',epochs,callback)
        else:
            model,norm,fit,p,coverage=models.fit_adapter(bank,train,valid,config,fit_seed,arm,epochs,teacher['state'],teacher['normalization'],callback)
        if not inner:p,coverage=reference.models.predict(model,norm,bank,query,config,fit_seed)
        audit=save_fit(path,expected,model,norm,fit,query,p,coverage)
        state=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},normalization=norm.audit(),expected=expected)
        del model
    fit=audit['fit']
    if inner:
        score=metric_bundle(bank.labels[np.searchsorted(bank.ids,query)],p,config['classes'])['macro_f1']
        if abs(score-fit['best_validation_macro_f1'])>1e-12:raise ValueError('Staged selected inner predictions disagree with score')
    if verify:
        if teacher is None:
            norm=reference.verify_fit(fit,bank,train,config,'beat_only',fit_seed,valid,norm_memo)
            model=reference.models.initial_model(config,fit_seed,'beat_only')
        else:
            key=_hash(train.tolist())
            if key not in norm_memo:norm_memo[key]=reference.chronology_model.SequenceNormalizer().fit(bank,train)
            norm=norm_memo[key]
            if teacher['normalization']!=norm.audit():raise ValueError('Teacher normalization is not training-only')
            model=models.initialize(config,fit_seed,arm,teacher['state'])
        model.load_state_dict(state['state_dict']);model.to(_device(config))
        if teacher is not None:verify_adapter(fit,model,teacher,bank,train,valid,config,fit_seed,arm,norm)
        if not inner and (len(fit['history'])!=epochs or fit['selected_epoch']!=epochs):raise ValueError('Staged outer duration not inner-selected')
        reproduced,coverage=reference.models.predict(model,norm,bank,query,config,fit_seed)
        if coverage!=audit['coverage'] or not np.array_equal(p.argmax(1),reproduced.argmax(1)) or not np.allclose(p,reproduced,rtol=1e-5,atol=1e-6):
            raise ValueError('Staged restored predictions/coverage changed')
        del model
    return dict(state=state['state_dict'],normalization=state['normalization'],fit=fit,p=p,coverage=audit['coverage'],
                model_sha256=audit['model_sha256'],audit_sha256=file_hash(path),phase=phase,study_fingerprint=scope['study_fingerprint'],
                train_membership_sha256=expected['train_membership_sha256'],query_membership_sha256=expected['query_membership_sha256'])


def epoch_choices(rows):
    return [dict(inner_fold=i,selected_epoch=r['fit']['selected_epoch'],audit_sha256=r['audit_sha256']) for i,r in enumerate(rows,1)]


def rounded_epochs(rows):return int(np.floor(np.median([r['fit']['selected_epoch'] for r in rows])+.5))


def evaluate_fold(bank,assigned,config,scope,root,seed,fold,verify=False):
    train,query=bank.ids[assigned!=fold],bank.ids[assigned==fold]
    truth=bank.labels[assigned!=fold]
    splits=[(train[a],train[b]) for a,b in inner_patient_splits(truth,train,config['inner_splits'],config['split_seed']+fold*1000)]
    teachers=[];inner={a:[] for a in models.ADAPTER_ARMS};memo={}
    for i,(fit_ids,valid) in enumerate(splits,1):
        teacher=phase_fit(bank,config,scope,root,seed,fold,i,'morphology',fit_ids,valid,config['max_epochs'],verify=verify,norm_memo=memo)
        teachers.append(teacher)
        for arm in models.ADAPTER_ARMS:
            inner[arm].append(phase_fit(bank,config,scope,root,seed,fold,i,arm,fit_ids,valid,config['max_epochs'],teacher=teacher,verify=verify,norm_memo=memo))
    teacher=phase_fit(bank,config,scope,root,seed,fold,0,'morphology',train,query,rounded_epochs(teachers),choices=epoch_choices(teachers),verify=verify,norm_memo=memo)
    values={'morphology':teacher['p']}
    for arm in models.ADAPTER_ARMS:
        result=phase_fit(bank,config,scope,root,seed,fold,0,arm,train,query,rounded_epochs(inner[arm]),teacher=teacher,choices=epoch_choices(inner[arm]),verify=verify,norm_memo=memo)
        values[arm]=result['p']
    _progress(root,state='verifying' if verify else 'outer_partition_complete',outer_fold=fold,training_seed=seed)
    if not verify:print(f'Staged study: seed {seed}, outer {fold}/{config["outer_splits"]}: complete; partial scores are not final performance',flush=True)
    return np.flatnonzero(assigned==fold),values


def run(config):
    bank,assigned=load_inputs(config);scope=make_scope(config,bank)
    root=Path(config['output_dir'])/'complete';root.mkdir(parents=True,exist_ok=True)
    path=root/'study_manifest.json'
    if path.exists() and json.loads(path.read_text())!=scope:raise ValueError('Staged source/config/runtime changed; use a new namespace')
    _save_json(path,scope)
    for fold in range(1,config['outer_splits']+1):
        for seed in config['training_seeds']:evaluate_fold(bank,assigned,config,scope,root,seed,fold)
    _progress(root,state='fits_complete_report_pending')
    return root


def smoke(config,ensure=False):
    bank,assigned=load_inputs(config);scope=make_scope(config,bank)
    root=Path(config['output_dir'])/'smoke';root.mkdir(parents=True,exist_ok=True);path=root/'summary.json'
    if ensure and path.is_file():
        s=json.loads(path.read_text())
        if (s.get('complete') and s.get('scope')==scope and s.get('zero_residual_verified') is True
                and set(s.get('arms',{}))==set(models.ADAPTER_ARMS)
                and all(v.get('restore_verified') is True for v in s['arms'].values())):
            print('Matching staged GPU smoke already passed.',flush=True);return
    small=dict(config,windows_per_epoch=8,batch_size=4,max_epochs=1)
    train,query=bank.ids[assigned!=1],bank.ids[assigned==1][:2]
    teacher=phase_fit(bank,small,scope,root,42,1,0,'morphology',train,query,1,choices=[],verify=False)
    audits={}
    for arm in models.ADAPTER_ARMS:
        item=phase_fit(bank,small,scope,root,42,1,0,arm,train,query,1,teacher=teacher,choices=[],verify=False)
        repeated=phase_fit(bank,small,scope,root,42,1,0,arm,train,query,1,teacher=teacher,choices=[],verify=True)
        audits[arm]=dict(fit=item['fit'],restore_verified=True)
        if not np.allclose(item['p'],repeated['p']):raise ValueError('Staged smoke failed restoration')
    # An outer phase with zero selected epochs must preserve teacher predictions.
    zero=models.initialize(small,42+1+config['stage2_seed_offset'],'frozen_context',teacher['state']).to(_device(small))
    norm=reference.chronology_model.SequenceNormalizer.restore(teacher['normalization'])
    p,_=reference.models.predict(zero,norm,bank,query,small,42+1+config['stage2_seed_offset'])
    if not np.allclose(p,teacher['p'],rtol=1e-5,atol=1e-6):raise ValueError('Zero context does not preserve the teacher')
    _save_json(path,dict(complete=True,scope=scope,claim='Finite staged device/gradient/freeze/restore smoke, not classification performance',arms=audits,zero_residual_verified=True))
    print('Actual-data staged smoke passed for all adaptation controls.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/staged_context_v1.yaml');p.add_argument('--smoke',action='store_true');p.add_argument('--ensure-smoke',action='store_true')
    args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    try:
        if args.smoke or args.ensure_smoke:smoke(config,args.ensure_smoke)
        else:print(run(config))
    except Exception as error:
        root=Path(config['output_dir'])/('smoke' if args.smoke or args.ensure_smoke else 'complete')
        _progress(root,state='failed',error_type=type(error).__name__,error=str(error));raise
