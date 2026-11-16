"""Two prespecified source-only multilabel objectives; no PTB training/evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import sklearn
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yaml

from . import neural, supervised_source_bank as source
from .supervised_source_audit import file_hash, canonical, save_json


def targets(states, objective):
    states=np.asarray(states)
    if not np.isin(states,[0,1,2,3]).all() or objective not in {'certain','positive_intermediate'}:
        raise ValueError('Invalid native states/objective')
    # Unlisted is an explicitly declared annotation-negative convention, not clinical truth.
    y=(states==3) if objective=='certain' else (states>=2)
    mask=(states==0)|(states==3)
    if objective=='positive_intermediate': mask|=states==2
    return y.astype(np.float32),mask.astype(np.float32)


def patient_weights(groups):
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    return 1/counts[inverse].astype(float)


def fit_normalizer(waves,counts,groups,train):
    weights=patient_weights(groups[train]);weights/=weights.sum()
    mean=np.zeros(3);second=np.zeros(3)
    for row,w in zip(train,weights):
        x=np.asarray(waves[row,:counts[row]],np.float64)
        if x.shape[0]==0 or not np.isfinite(x).all(): raise ValueError('Invalid training crops')
        mean+=w*x.mean(axis=(0,2));second+=w*np.square(x).mean(axis=(0,2))
    return dict(mean=mean.astype(np.float32).tolist(),scale=np.sqrt(np.maximum(second-mean**2,1e-12)).astype(np.float32).tolist(),
                fit_records=len(train),fit_patients=len(np.unique(groups[train])))


def active_columns(states,groups,train,minimum):
    return np.asarray([j for j in range(states.shape[1]) if len(np.unique(groups[train][states[train,j]==3]))>=minimum],int)


def positive_weights(y,mask,groups,maximum):
    w=patient_weights(groups)[:,None]
    positive=(w*mask*y).sum(0);negative=(w*mask*(1-y)).sum(0)
    if np.any(positive<=0): raise ValueError('Source head has no training positives')
    return np.clip(negative/positive,1,maximum).astype(np.float32)


def masked_loss(logits,y,mask,pos_weight):
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=pos_weight)
    return ((loss*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()


class RecordRows(Dataset):
    def __init__(self,arrays,indices,normalization,y,mask):
        self.arrays,self.indices,self.normalization,self.y,self.mask=arrays,np.asarray(indices),normalization,y,mask
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        row=self.indices[i]
        x=np.asarray(self.arrays['waves'][row],np.float32).copy()
        x=(x-np.asarray(self.normalization['mean'],np.float32)[None,:,None])/np.asarray(self.normalization['scale'],np.float32)[None,:,None]
        beat_mask=np.arange(x.shape[0])<self.arrays['counts'][row]
        return torch.from_numpy(x),torch.from_numpy(beat_mask),torch.from_numpy(self.y[row]),torch.from_numpy(self.mask[row])


class RecordModel(nn.Module):
    def __init__(self,heads,config):
        super().__init__()
        base=neural.VCGResNet(heads,config['widths'],config['pool_bins'],config['dropout'])
        self.encoder,self.classifier=base.encoder,base.classifier
    def forward(self,x,mask):
        features=self.encoder(x.flatten(0,1)).reshape(x.shape[0],x.shape[1],-1)
        weights=mask.to(features.dtype).unsqueeze(-1)
        return self.classifier((features*weights).sum(1)/weights.sum(1).clamp_min(1))


def predict(model,arrays,indices,norm,y,mask,config):
    loader=DataLoader(RecordRows(arrays,indices,norm,y,mask),batch_size=config['batch_size'],shuffle=False,num_workers=0)
    result=[];model.eval();device=neural._device(config)
    with torch.no_grad():
        for x,bm,_,_ in loader:
            p=torch.sigmoid(model(x.to(device),bm.to(device))).cpu().numpy()
            if not np.isfinite(p).all(): raise ValueError('Nonfinite source probabilities')
            result.append(p)
    return np.concatenate(result)


def metrics(y,mask,p,groups,labels):
    if p.shape!=y.shape or mask.shape!=y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid source metric inputs')
    weights=patient_weights(groups);rows=[]
    for j,label in enumerate(labels):
        selected=mask[:,j].astype(bool);truth=y[selected,j].astype(int);prob=p[selected,j];w=weights[selected]
        positive=int((truth==1).sum());negative=int((truth==0).sum());eligible=positive>0 and negative>0
        rows.append(dict(code=str(label),positive_records=positive,annotation_negative_records=negative,
            positive_patients=len(np.unique(groups[selected][truth==1])),annotation_negative_patients=len(np.unique(groups[selected][truth==0])),
            masked_records=int((~selected).sum()),eligible_for_macro=eligible,
            average_precision=float(average_precision_score(truth,prob,sample_weight=w)) if eligible else None,
            auroc=float(roc_auc_score(truth,prob,sample_weight=w)) if eligible else None,
            f1_at_05=float(f1_score(truth,prob>=.5,sample_weight=w,zero_division=0)) if eligible else None))
    good=[r for r in rows if r['eligible_for_macro']]
    if not good: raise ValueError('No evaluable source labels')
    return dict(macro_average_precision=float(np.mean([r['average_precision'] for r in good])),
        macro_auroc=float(np.mean([r['auroc'] for r in good])),macro_f1_at_05=float(np.mean([r['f1_at_05'] for r in good])),
        evaluable_codes=len(good),total_trained_codes=len(labels),per_code=rows,
        unit='record annotation; each patient has total record weight one before per-code masks',
        clinical_negatives=False,target_six_class_metric=False)


def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return type(value)(cpu(v) for v in value)
    return value


def save_checkpoint(path,value):
    temporary=path.with_suffix('.partial');torch.save(cpu(value),temporary);temporary.replace(path)


def state_hash(model):
    return canonical({k:source.array_hash(v.detach().cpu().numpy()) for k,v in model.state_dict().items()})


def metrics_close(left,right):
    if isinstance(left,dict) and isinstance(right,dict):
        return left.keys()==right.keys() and all(metrics_close(left[k],right[k]) for k in left)
    if isinstance(left,list) and isinstance(right,list):
        return len(left)==len(right) and all(metrics_close(a,b) for a,b in zip(left,right))
    if isinstance(left,float) and isinstance(right,(float,int)):
        return bool(np.isclose(left,right,atol=2e-5,rtol=2e-5))
    return left==right


def validate(config):
    if config['objectives']!=['certain','positive_intermediate'] or not config['training_seeds'] or len(set(config['training_seeds']))!=len(config['training_seeds']):
        raise ValueError('Require both prespecified objectives and distinct seeds')
    for name in ['max_epochs','patience','batch_size','samples_per_epoch','torch_threads']:
        if type(config[name]) is not int or config[name]<1: raise ValueError('Invalid source training budget')
    if any(type(s) is not int or s<0 for s in config['training_seeds']): raise ValueError('Invalid training seeds')
    if (not np.isfinite([config['learning_rate'],config['weight_decay'],config['maximum_positive_weight'],config['dropout']]).all()
            or config['learning_rate']<=0 or config['weight_decay']<0 or config['maximum_positive_weight']<1 or not 0<=config['dropout']<1):
        raise ValueError('Invalid source optimizer/loss settings')


def load_arrays(bank):
    names=['waves','counts','states','patient_ids','roles','ecg_ids','labels']
    a={name:np.load(bank/(name+'.npy'),mmap_mode='r') for name in names}
    n=len(a['waves'])
    if (a['waves'].ndim!=4 or a['waves'].shape[2:]!=(3,651) or len(a['counts'])!=n
            or np.any((a['counts']<1)|(a['counts']>a['waves'].shape[1]))
            or a['states'].shape!=(n,len(a['labels'])) or not np.isin(a['states'],[0,1,2,3]).all()
            or any(len(a[k])!=n for k in ['patient_ids','roles','ecg_ids']) or len(np.unique(a['ecg_ids']))!=n
            or set(a['roles'])!={'train','validation'}):
        raise ValueError('Invalid supervised source arrays')
    train=np.flatnonzero(a['roles']=='train');valid=np.flatnonzero(a['roles']=='validation')
    if set(a['patient_ids'][train])&set(a['patient_ids'][valid]): raise ValueError('Source patient leakage')
    return a,train,valid


def fit_one(config,a,train,valid,columns,norm,scope,objective,seed,verify=False):
    root=Path(config['output_dir'])/'runs'/f'{objective}_{seed}';root.mkdir(parents=True,exist_ok=True)
    guard=root/'manifest.json'
    local=dict(scope,objective=objective,seed=seed);fingerprint=canonical(local)
    if guard.exists() and json.loads(guard.read_text())!=local: raise ValueError('Source fit scope changed')
    if not guard.exists():
        if verify: raise ValueError('Missing source fit')
        save_json(guard,local)
    neural._seed(seed,config);device=neural._device(config)
    model=RecordModel(len(columns),config).to(device);initial_hash=state_hash(model)
    states=np.asarray(a['states'][:,columns]);labels=np.asarray(a['labels'][columns])
    y,mask=targets(states,objective);ey,em=targets(states,'certain')
    weight=positive_weights(y[train],mask[train],a['patient_ids'][train],config['maximum_positive_weight'])
    pw=torch.tensor(weight,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    sampler_rng=torch.Generator().manual_seed(seed+17);loader_rng=torch.Generator().manual_seed(seed+19)
    sampler=WeightedRandomSampler(torch.tensor(patient_weights(a['patient_ids'][train]),dtype=torch.double),
                                 config['samples_per_epoch'],replacement=True,generator=sampler_rng)
    loader=DataLoader(RecordRows(a,train,norm,y,mask),batch_size=config['batch_size'],sampler=sampler,num_workers=0,generator=loader_rng)
    checkpoint=root/'checkpoint.pt';summary_path=root/'summary.json'
    if verify and not summary_path.exists(): raise ValueError('Source training incomplete')
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['fingerprint']!=fingerprint or saved['initial_hash']!=initial_hash or saved['normalization']!=norm:
            raise ValueError('Source checkpoint settings/data/initialization changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        sampler_rng.set_state(saved['sampler_rng']);loader_rng.set_state(saved['loader_rng']);torch.set_rng_state(saved['torch_rng'])
        if device.type=='mps': torch.mps.set_rng_state(saved['mps_rng'])
        history,best,best_epoch,best_state,initial_metrics=saved['history'],saved['best'],saved['best_epoch'],saved['best_state'],saved['initial_metrics']
    else:
        if verify: raise ValueError('Missing source checkpoint')
        initial_p=predict(model,a,valid,norm,ey,em,config)
        initial_metrics=metrics(ey[valid],em[valid],initial_p,a['patient_ids'][valid],labels)
        history=[];best=initial_metrics['macro_average_precision'];best_epoch=0;best_state=cpu(model.state_dict())
    completed=summary_path.exists()
    if completed:
        saved_summary=json.loads(summary_path.read_text())
        for name,digest in saved_summary['files'].items():
            if file_hash(root/name)!=digest: raise ValueError('Source training artifact changed')
    for epoch in ([] if completed or verify else range(len(history)+1,config['max_epochs']+1)):
        if history and len(history)-best_epoch>=config['patience']: break
        model.train();total=0.;seen=0
        for x,bm,yy,mm in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=masked_loss(model(x.to(device),bm.to(device)),yy.to(device),mm.to(device),pw)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite source training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total+=float(loss.detach().cpu())*len(x);seen+=len(x)
        p=predict(model,a,valid,norm,ey,em,config)
        score=metrics(ey[valid],em[valid],p,a['patient_ids'][valid],labels)
        value=score['macro_average_precision']
        if value>best+1e-8: best,best_epoch,best_state=value,epoch,cpu(model.state_dict())
        row=dict(epoch=epoch,training_loss=total/seen,validation_macro_ap=value,
                 validation_macro_auroc=score['macro_auroc'],validation_macro_f1_at_05=score['macro_f1_at_05'],presentations=seen)
        history.append(row)
        save_checkpoint(checkpoint,dict(fingerprint=fingerprint,initial_hash=initial_hash,normalization=norm,
            model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best,best_epoch=best_epoch,best_state=best_state,
            initial_metrics=initial_metrics,sampler_rng=sampler_rng.get_state(),loader_rng=loader_rng.get_state(),
            torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if device.type=='mps' else None))
        source.progress(config,'training_source_only',objective=objective,seed=seed,**row)
        print(f"{objective} seed {seed} epoch {epoch}: AP={value:.5f}, F1@.5={score['macro_f1_at_05']:.5f}",flush=True)
    model.load_state_dict(best_state)
    p=predict(model,a,valid,norm,ey,em,config)
    final=metrics(ey[valid],em[valid],p,a['patient_ids'][valid],labels)
    if not np.isclose(final['macro_average_precision'],best,rtol=1e-5,atol=1e-6): raise ValueError('Restored source AP differs')
    prior=(patient_weights(a['patient_ids'][train])[:,None]*em[train]*ey[train]).sum(0)/(patient_weights(a['patient_ids'][train])[:,None]*em[train]).sum(0)
    baseline=metrics(ey[valid],em[valid],np.broadcast_to(prior,p.shape),a['patient_ids'][valid],labels)
    if completed:
        with np.load(root/'validation.npz') as saved:
            if (not np.allclose(saved['p'],p,atol=2e-5,rtol=2e-5) or not np.array_equal(saved['ecg_ids'],a['ecg_ids'][valid])
                    or not np.array_equal(saved['labels'],labels)):
                raise ValueError('Source validation probabilities do not reproduce')
        if saved_summary['history']!=history or saved_summary['best_epoch']!=best_epoch:
            raise ValueError('Source history changed')
        if (saved_summary['fingerprint']!=fingerprint or saved_summary['objective']!=objective or saved_summary['seed']!=seed
                or saved_summary['labels']!=labels.tolist() or saved_summary['normalization']!=norm
                or saved_summary['positive_weights']!=weight.tolist()
                or not metrics_close(saved_summary['metrics'],final)
                or not metrics_close(saved_summary['training_prior_baseline'],baseline)):
            raise ValueError('Restored source metrics/metadata do not reproduce')
        return saved_summary
    np.savez_compressed(root/'validation.npz',p=p,labels=labels,ecg_ids=a['ecg_ids'][valid])
    save_checkpoint(root/'encoder.pt',dict(encoder_state_dict=model.encoder.state_dict(),normalization=norm,
        fingerprint=fingerprint,widths=config['widths'],pool_bins=config['pool_bins'],best_epoch=best_epoch,
        source_only=True,ptb_patient_independence_cleared=False,claim_scope=source.CLAIM))
    pd.DataFrame(history).to_csv(root/'history.csv',index=False)
    result=dict(complete=True,fingerprint=fingerprint,objective=objective,seed=seed,initial_hash=initial_hash,
        best_epoch=best_epoch,epochs_completed=len(history),normalization=norm,positive_weights=weight.tolist(),labels=labels.tolist(),
        history=history,initial_metrics=initial_metrics,metrics=final,training_prior_baseline=baseline,
        training_presentations=sum(r['presentations'] for r in history),target_evaluated=False,
        cross_dataset_identity_cleared=False,files={name:file_hash(root/name) for name in ['checkpoint.pt','encoder.pt','validation.npz','history.csv']})
    save_json(summary_path,result)
    return result


def run(config,verify=False):
    validate(config);bank=source.prepare(config);a,train,valid=load_arrays(bank)
    columns=active_columns(a['states'],a['patient_ids'],train,config['minimum_training_positive_patients'])
    if not len(columns): raise ValueError('No source labels meet prespecified training support')
    norm=fit_normalizer(a['waves'],a['counts'],a['patient_ids'],train)
    scope=dict(config=config,bank_manifest_sha256=file_hash(bank/'manifest.json'),normalization=norm,
        labels=a['labels'][columns].tolist(),column_indices=columns.tolist(),code_sha256=file_hash(__file__),
        neural_code_sha256=file_hash(neural.__file__),report_code_sha256=file_hash(Path(__file__).with_name('supervised_source_pilot_report.py')),
        runtime=dict(torch=str(torch.__version__),numpy=np.__version__,sklearn=sklearn.__version__,platform=platform.platform(),
                     device=str(neural._device(config)),deterministic_kernels=neural._device(config).type!='mps'),
        objective_convention='Unlisted=annotation-negative; likelihood0 masked; certain masks intermediate, sensitivity marks intermediate positive; common validation uses only100/unlisted',
        claim_scope=source.CLAIM)
    root=Path(config['output_dir']);guard=root/'training_manifest.json'
    if guard.exists() and json.loads(guard.read_text())!=scope: raise ValueError('Source training scope changed; new namespace required')
    if not guard.exists():
        if verify: raise ValueError('Missing source training scope')
        save_json(guard,scope)
    results=[fit_one(config,a,train,valid,columns,norm,scope,obj,seed,verify) for obj in config['objectives'] for seed in config['training_seeds']]
    from .supervised_source_pilot_report import render
    render(config,results,scope,verify)
    source.progress(config,'complete_source_only_validated',fits=len(results),target_evaluated=False)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/supervised_source_pilot_v1.yaml');p.add_argument('--verify',action='store_true')
    args=p.parse_args();run(yaml.safe_load(Path(args.config).read_text()),args.verify)
