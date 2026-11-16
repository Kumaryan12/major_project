"""Matched scratch and two diagnostic-source encoder procedures on frozen PTB folds."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.metrics import f1_score

from . import transfer as backend, preprocessing_study, neural, regularization
from . import supervised_source_overlap as overlap, supervised_source_train as source_train
from .supervised_source_audit import canonical,file_hash,save_json
from .benchmark import aggregate_patients,inner_patient_splits,load_or_create_folds,metric_bundle,patient_balanced_weights
from .contrastive import initial_state_hash

ARMS=('scratch','source_certain','source_positive_intermediate')
OBJECTIVES={'source_certain':'certain','source_positive_intermediate':'positive_intermediate'}
CLAIM='Exploratory measured-VCG PTB development; source-target patient identity remains unresolved; no independent confirmation, clinical validity or SOTA claim.'


def progress(root,state,**extra):
    save_json(Path(root)/'progress.json',dict(state=state,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**extra))


def validate_config(c):
    # Reuse the historical budget validation while declaring three distinct new arms.
    backend.validate_config(dict(c,arms=list(backend.ARMS)))
    if c['arms']!=list(ARMS) or type(c['source_training_seed']) is not int or c['source_training_seed']<0:
        raise ValueError('Require scratch and both declared diagnostic-source arms')
    if len(c['classes'])<2 or len(set(c['classes']))!=len(c['classes']): raise ValueError('Invalid target classes')
    if type(c['bootstrap_draws']) is not int or c['bootstrap_draws']<100: raise ValueError('Invalid patient bootstrap budget')


def checked_exports(config,pilot,summary,training):
    """Load encoder-only exports from both declared source objectives, never select a winner."""
    root=Path(pilot['output_dir']);states={};evidence={}
    if config['widths']!=pilot['widths'] or config['pool_bins']!=pilot['pool_bins']:
        raise ValueError('Source/target encoder architecture mismatch')
    for arm,objective in OBJECTIVES.items():
        seed=config['source_training_seed'];name=f'{objective}_{seed}'
        matches=[r for r in summary['runs'] if r['objective']==objective and r['seed']==seed]
        if len(matches)!=1 or not matches[0]['complete']: raise ValueError('Missing declared source encoder')
        result=matches[0];path=root/'runs'/name/'encoder.pt'
        member=json.loads((root/'runs'/name/'manifest.json').read_text())
        if member!=dict(training,objective=objective,seed=seed): raise ValueError('Source member scope mismatch')
        fingerprint=canonical(member)
        if result['fingerprint']!=fingerprint or file_hash(path)!=result['files']['encoder.pt']:
            raise ValueError('Supervised source export changed')
        export=torch.load(path,map_location='cpu',weights_only=True)
        if (export['fingerprint']!=fingerprint or export['widths']!=config['widths'] or export['pool_bins']!=config['pool_bins']
                or export['source_only'] is not True or export['ptb_patient_independence_cleared'] is not False
                or export['best_epoch']!=result['best_epoch']):
            raise ValueError('Source encoder provenance/endpoint mismatch')
        state=export['encoder_state_dict']
        if not state or any(not torch.isfinite(v).all() for v in state.values()): raise ValueError('Invalid source encoder weights')
        model=backend.initial_model(config,config['seed'],'pretrained',state)
        states[arm]=state
        evidence[arm]=dict(objective=objective,source_seed=seed,best_source_epoch=result['best_epoch'],
            export_sha256=file_hash(path),encoder_state_sha256=initial_state_hash(model.encoder),
            member_summary_sha256=file_hash(root/'runs'/name/'summary.json'),source_only_normalization_transferred=False)
    return states,evidence


def load_sources(config,target_bank):
    pilot=yaml.safe_load(Path(config['source_config']).read_text())
    summary,source_evidence=overlap.completed_pilot(pilot)
    training=json.loads((Path(pilot['output_dir'])/'training_manifest.json').read_text())
    sc=yaml.safe_load(Path(config['overlap_config']).read_text());root=Path(sc['output_dir'])
    if sc['source_config']!=config['source_config']: raise ValueError('Expanded screen used another source study')
    manifest=json.loads((root/'screen_manifest.json').read_text());fp=manifest['screen_fingerprint']
    if canonical({k:v for k,v in manifest.items() if k!='screen_fingerprint'})!=fp or manifest['config']!=sc:
        raise ValueError('Expanded recording-screen scope changed')
    output=json.loads((root/'output_manifest.json').read_text())
    if output['screen_fingerprint']!=fp or any(file_hash(root/n)!=h for n,h in output['files'].items()):
        raise ValueError('Expanded recording-screen artifact changed')
    if any(file_hash(Path(sc['cache_dir'])/n)!=h for n,h in output['cache_files'].items()):
        raise ValueError('Expanded waveform-screen cache changed')
    screened=json.loads((root/'summary.json').read_text())
    if (screened['complete'] is not True or screened['screen_fingerprint']!=fp or screened['recording_screen_passed'] is not True
            or screened['patient_identity_independence_cleared'] is not False or screened['target_classification_launched'] is not False
            or any(screened[k] for k in ['near_recording_candidate_pairs','exact_encoded_data_matches','incomplete_active_lead_pairs'])
            or screened['positive_controls']<1 or screened['positive_controls_retrieved_and_confirmed']!=screened['positive_controls']):
        raise ValueError('Expanded recording screen did not pass; review before transfer')
    for field,module in [('code_sha256',overlap),('numerical_code_sha256',overlap.numeric),
                         ('signal_code_sha256',overlap.numeric.signal),('kors_code_sha256',overlap.numeric.external)]:
        if manifest[field]!=file_hash(module.__file__): raise ValueError('Expanded-screen numerical code changed')
    sources,targets,provenance=overlap.load_context(sc)
    if (manifest['provenance']!=provenance or provenance['pilot_evidence']!=source_evidence
            or canonical(overlap.inventory(sources))!=manifest['source_inventory_sha256']
            or canonical(overlap.inventory(targets))!=manifest['target_inventory_sha256']):
        raise ValueError('Expanded screen cohort/waveform provenance changed')
    protected=provenance['target']
    if (protected['beats_sha256']!=target_bank['beats_sha256'] or protected['folds_sha256']!=file_hash(config['folds_path'])
            or any(protected[k]!=target_bank['n_'+k] for k in ['patients','records','beats'])):
        raise ValueError('Target cohort/folds differ from screened original cohort')
    ac=yaml.safe_load(Path(pilot['audit_config']).read_text())
    if config['classes']!=ac['target_classes']: raise ValueError('Target class definitions changed')
    states,exports=checked_exports(config,pilot,summary,training)
    return states,dict(source=source_evidence,exports=exports,source_training_seed=config['source_training_seed'],
        source_training_patients=provenance['source_training_patients'],source_validation_patients=provenance['source_validation_patients'],
        expanded_screen_fingerprint=fp,expanded_screen_manifest_sha256=file_hash(root/'screen_manifest.json'),
        expanded_screen_output_manifest_sha256=file_hash(root/'output_manifest.json'),
        expanded_screen_summary_sha256=file_hash(root/'summary.json'),
        expanded_recording_screen={k:screened[k] for k in ['source_records','target_records','continuous_pairs_checked','positive_controls','positive_controls_retrieved_and_confirmed']},
        patient_identity_independence_cleared=False,claim_scope=CLAIM)


def load_inputs(config):
    validate_config(config)
    if not Path(config['folds_path']).is_file(): raise ValueError('Frozen patient folds required; never generate new target splits')
    waves,y,groups,bank=regularization.load_bank(config)
    encoders,source=load_sources(config,bank)
    folds=load_or_create_folds(y,groups,config)
    if sorted(folds.fold.unique())!=list(range(1,config['outer_splits']+1)): raise ValueError('Frozen fold IDs changed')
    lookup=folds.set_index('patient_id').fold.to_dict()
    return dict(waves=waves,y=y,groups=groups,bank=bank,encoders=encoders,source=source,folds=folds,
                beat_folds=np.asarray([lookup[str(p)] for p in groups]),norm_memo={})


def make_scope(config,data,chosen):
    return dict(config=config,bank=data['bank'],source=data['source'],folds_sha256=file_hash(config['folds_path']),
        evaluated_folds=chosen,claim_scope=CLAIM,
        code_sha256=file_hash(__file__),backend_code_sha256=file_hash(backend.__file__),neural_code_sha256=file_hash(neural.__file__),
        scratch_code_sha256=file_hash(preprocessing_study.__file__),bank_loader_code_sha256=file_hash(regularization.__file__),
        report_code_sha256=file_hash(Path(__file__).with_name('supervised_transfer_report.py')),
        runtime=dict(torch=str(torch.__version__),numpy=np.__version__,sklearn=sklearn.__version__,pandas=pd.__version__,
                     platform=platform.platform(),device=str(neural._device(config)),deterministic_kernels=neural._device(config).type!='mps'))


def rows_hash(rows): return hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest()


def expected_norm(data,train):
    key=rows_hash(train)
    if key not in data['norm_memo']:
        weights=patient_balanced_weights(data['y'][train],data['groups'][train])
        norm=neural.FoldNormalizer().fit(data['waves'],np.empty((len(data['y']),0),np.float32),train,weights)
        data['norm_memo'][key]=norm.audit()
    return data['norm_memo'][key]


def restore_norm(values,count):
    norm=neural.FoldNormalizer()
    for key,value in values.items(): setattr(norm,key,np.asarray(value,np.float32))
    norm.n_fit_samples_=count
    return norm


def member(config,data,root,fingerprint,arm,train,query,seed,rate,epochs,context,inner,verify=False):
    if arm not in ARMS or set(data['groups'][train])&set(data['groups'][query]): raise ValueError('Invalid arm or overlapping target patients')
    logical='scratch' if arm=='scratch' else 'pretrained'
    encoder=None if arm=='scratch' else data['encoders'][arm]
    encoder_hash=None if arm=='scratch' else data['source']['exports'][arm]['encoder_state_sha256']
    settings=dict(config,learning_rate=rate)
    fit_scope=dict(context,study_fingerprint=fingerprint,arm=arm,fit_seed=seed,learning_rate=rate,epoch_budget=epochs,
                   train_rows_sha256=rows_hash(train),query_rows_sha256=rows_hash(query),inner_validation=inner)
    directory=Path(root)/'review/fits'/f"s{context['training_seed']}_f{context['outer_fold']:02d}_{arm}_{context['fit_id']}"
    directory.mkdir(parents=True,exist_ok=True);audit_path=directory/'audit.json'
    existing=audit_path.exists()
    if verify and not existing: raise ValueError('Incomplete supervised-transfer fit')
    if existing:
        saved=json.loads(audit_path.read_text())
        if saved['context']!=fit_scope or any(file_hash(directory/n)!=h for n,h in saved['files'].items()):
            raise ValueError('Supervised-transfer fit scope/artifact changed')
        fit=saved['fit']
    else:
        callback=lambda row:progress(root,'training',arm=arm,training_seed=context['training_seed'],outer_fold=context['outer_fold'],
                                    fit_id=context['fit_id'],learning_rate=rate,**row)
        model,norm,fit=backend.fit_transfer(data['waves'],data['y'],data['groups'],train,query if inner else None,
                                          epochs,settings,seed,logical,encoder,callback)
    backend.validate_fit(fit,config,seed,logical,rate,len(train),len(np.unique(data['groups'][train])),
                         len(np.unique(data['groups'][query])) if inner else 0,encoder_hash,None if inner else epochs)
    if fit['normalization']!=expected_norm(data,train): raise ValueError('Target normalization must use exact target-training partition only')
    if existing:
        checkpoint=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
        if checkpoint['context']!=fit_scope or checkpoint['normalization']!=fit['normalization']:
            raise ValueError('Supervised-transfer model checkpoint context changed')
        model=backend.initial_model(config,seed,logical,encoder)
        model.load_state_dict(checkpoint['state_dict'],strict=True);model=model.to(neural._device(config))
        norm=restore_norm(fit['normalization'],len(train))
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()): raise ValueError('Nonfinite target checkpoint')
    p=preprocessing_study.predict_waves(model,norm,data['waves'],data['y'],query,config)
    _,truth,pooled=aggregate_patients(p,data['y'][query],data['groups'][query])
    score=float(f1_score(truth,pooled.argmax(1),labels=np.arange(len(config['classes'])),average='macro',zero_division=0))
    if inner and not np.isclose(score,fit['best_validation_macro_f1'],atol=1e-8): raise ValueError('Restored inner best model/score does not reproduce')
    if existing:
        with np.load(directory/'predictions.npz') as saved_p:
            if not np.array_equal(saved_p['indices'],query) or not np.allclose(saved_p['p'],p,atol=2e-5,rtol=2e-5):
                raise ValueError('Restored target probabilities do not reproduce')
    else:
        source_train.save_checkpoint(directory/'model.pt',dict(context=fit_scope,state_dict=model.state_dict(),normalization=norm.audit()))
        temp=directory/'predictions.partial.npz';np.savez_compressed(temp,indices=query,p=p);temp.replace(directory/'predictions.npz')
        save_json(audit_path,dict(context=fit_scope,fit=fit,files={n:file_hash(directory/n) for n in ['model.pt','predictions.npz']}))
    del model,norm
    return fit,p


def check_matched(fits):
    for fit in fits[1:]: backend.check_pair(fits[0],fit)


def evaluate(config,data,root,fingerprint,chosen,verify=False):
    predictions={};selections=[];fit_count=0;matched=0
    for training_seed in config['training_seeds']:
        predictions[training_seed]={arm:np.full((len(data['y']),len(config['classes'])),np.nan) for arm in ARMS}
        for fold in chosen:
            train=np.flatnonzero(data['beat_folds']!=fold);test=np.flatnonzero(data['beat_folds']==fold)
            splits=[(train[ti],train[vi]) for ti,vi in inner_patient_splits(data['y'][train],data['groups'][train],
                                                                        config['inner_splits'],config['split_seed']+fold*1000)]
            candidates={arm:{} for arm in ARMS}
            for i,rate in enumerate(config['learning_rates']):
                key=f'rate_{i}'
                for arm in ARMS: candidates[arm][key]=[]
                for j,(ti,vi) in enumerate(splits,1):
                    if set(data['groups'][np.r_[ti,vi]])&set(data['groups'][test]): raise ValueError('Outer query leaked into inner fit')
                    fits=[]
                    for arm in ARMS:
                        fit,_=member(config,data,root,fingerprint,arm,ti,vi,training_seed+fold*1000+j,rate,config['max_epochs'],
                            dict(training_seed=training_seed,outer_fold=fold,fit_id=f'{key}_inner{j}'),True,verify)
                        candidates[arm][key].append(fit);fits.append(fit);fit_count+=1
                        progress(root,'restoring_fits' if verify else 'fit_complete',completed_fits=fit_count,arm=arm,
                                 training_seed=training_seed,outer_fold=fold,fit_id=f'{key}_inner{j}')
                    check_matched(fits);matched+=1
            for arm in ARMS:
                for j in range(config['inner_splits']):
                    a,b=[candidates[arm][f'rate_{i}'][j] for i in range(2)]
                    backend.check_pair(a,b)
                    if a['initial_state_sha256']!=b['initial_state_sha256']: raise ValueError('Learning-rate candidates changed initialization')
            outer_fits=[]
            for arm in ARMS:
                selected=backend.choose_rate(candidates[arm],config)
                path=Path(root)/'review/selections'/f's{training_seed}_f{fold:02d}_{arm}.json'
                record=dict(study_fingerprint=fingerprint,training_seed=training_seed,outer_fold=fold,arm=arm,selection=selected)
                if path.exists() and json.loads(path.read_text())!=record: raise ValueError('Inner-only learning-rate/epoch selection changed')
                if verify and not path.exists(): raise ValueError('Missing target selection audit')
                if not path.exists(): save_json(path,record)
                choice=selected['selected']
                fit,p=member(config,data,root,fingerprint,arm,train,test,training_seed+fold,choice['learning_rate'],choice['final_epochs'],
                    dict(training_seed=training_seed,outer_fold=fold,fit_id='outer'),False,verify)
                predictions[training_seed][arm][test]=p;outer_fits.append(fit);fit_count+=1
                selections.append(dict(training_seed=training_seed,outer_fold=fold,arm=arm,learning_rate=choice['learning_rate'],
                                       refit_epochs=choice['final_epochs'],mean_inner_f1=choice['mean_best_validation_macro_f1']))
                progress(root,'restoring_fits' if verify else 'outer_complete',completed_fits=fit_count,training_seed=training_seed,outer_fold=fold,arm=arm)
            check_matched(outer_fits);matched+=1
            print(f"{'Restored' if verify else 'Completed'} all matched arms: seed {training_seed}, outer fold {fold}",flush=True)
    return dict(predictions=predictions,selections=selections,completed_fit_count=fit_count,matched_context_checks=matched)


def run_study(config,outer_folds=None,verify=False):
    root,chosen,partial=preprocessing_study.directory(config,outer_folds)
    data=load_inputs(config)
    preflight_path=Path(config['output_dir'])/'complete/review/preflight.json'
    if not preflight_path.exists(): raise ValueError('Verified source/expanded-screen preflight required; use the study launcher')
    receipt=json.loads(preflight_path.read_text())
    full_scope=make_scope(config,data,list(range(1,config['outer_splits']+1)))
    if (receipt['scope']!=full_scope or receipt.get('source_models_independently_verified') is not True
            or receipt.get('expanded_screen_independently_verified') is not True):
        raise ValueError('Supervised-transfer preflight scope changed; use a new namespace')
    scope=make_scope(config,data,chosen);fingerprint=canonical(scope)
    manifest=dict(scope,study_fingerprint=fingerprint);root.mkdir(parents=True,exist_ok=True);guard=root/'study_manifest.json'
    if guard.exists() and json.loads(guard.read_text())!=manifest: raise ValueError('Supervised-transfer inputs/code/runtime changed; use a new namespace')
    if verify and not guard.exists(): raise ValueError('Missing supervised-transfer study')
    if not guard.exists(): save_json(guard,manifest)
    result=evaluate(config,data,root,fingerprint,chosen,verify)
    expected=len(config['training_seeds'])*len(chosen)*len(ARMS)*(len(config['learning_rates'])*config['inner_splits']+1)
    if result['completed_fit_count']!=expected: raise ValueError('Incomplete target fits')
    progress(root,'partial_fits_complete' if partial else 'all_fits_restored' if verify else 'all_fits_complete_pending_report',
             completed_fits=expected,target_patient_identity_cleared=False)
    return data,result,manifest,root,partial


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/supervised_transfer_v1.yaml')
    p.add_argument('--outer-folds',nargs='+',type=int);p.add_argument('--verify',action='store_true')
    args=p.parse_args();run_study(yaml.safe_load(Path(args.config).read_text()),args.outer_folds,args.verify)
