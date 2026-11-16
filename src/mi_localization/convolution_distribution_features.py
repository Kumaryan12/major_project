"""Streaming, resumable all-beat convolution residuals with an intact base path."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from .distribution_features import summaries
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash
from .contrastive import _progress

ARMS=('baseline','plus_mean','plus_spread','plus_quantiles')


def projection(config, dimensions, seed):
    rng=np.random.default_rng(seed+config['projection_seed_offset'])
    matrix=rng.normal(size=(dimensions,config['projections']))
    return matrix/np.linalg.norm(matrix,axis=0,keepdims=True)


def record_projection(transform,waves,median_features,scale,matrix):
    values=transform.transform(waves)['combined'].astype(np.float64)
    if values.shape[1:]!=scale.shape or not np.isfinite(values).all():raise ValueError('Invalid all-beat convolution features')
    return ((values-np.asarray(median_features,dtype=np.float64))/scale)@matrix


def projected_records(bank,config,root,scope,transform,source_context,scale,matrix,records,seed,fold,inner,verify=False,smoke=False):
    """Only the small projected residuals persist; never an all-beat x3584 bank."""
    mode='smoke_first_beats' if smoke else 'all_original_beats'
    directory=root/'review/records'/f'f{fold:02d}_s{seed}_i{inner}';directory.mkdir(parents=True,exist_ok=True)
    expected=dict(study_fingerprint=scope['study_fingerprint'],source_context_sha256=file_hash(source_context),
                  column_scale_sha256=array_hash(scale),projection_sha256=array_hash(matrix),mode=mode,
                  smoke_beats_per_record=config['smoke_beats_per_record'] if smoke else None)
    median=transform.transform(bank.templates[records])['combined']
    result=[];covered=0;audits=[]
    for j,record in enumerate(records):
        rows=bank.rows(record)
        if smoke:rows=rows[:config['smoke_beats_per_record']]
        record_expected=dict(expected,record_index=int(record),beat_rows_sha256=array_hash(rows),n_beats=len(rows),
                             median_features_sha256=array_hash(median[j]))
        path=directory/f'record_{record:04d}.json';data_path=path.with_suffix('.npz')
        if verify or path.exists():
            audit=json.loads(path.read_text())
            if any(audit.get(k)!=v for k,v in record_expected.items()) or file_hash(data_path)!=audit['data_sha256']:
                raise ValueError('Projected record cache changed')
            with np.load(data_path) as f:
                if not np.array_equal(f['rows'],rows):raise ValueError('Beat membership changed')
                z=f['projected']
            if verify:
                fresh=record_projection(transform,np.asarray(bank.records.waves[rows]),median[j],scale,matrix)
                if not np.allclose(fresh,z,rtol=1e-10,atol=1e-12):raise ValueError('Original beat convolution replay disagrees')
        else:
            z=record_projection(transform,np.asarray(bank.records.waves[rows]),median[j],scale,matrix)
            temporary=data_path.with_suffix('.partial.npz');np.savez_compressed(temporary,rows=rows,projected=z);temporary.replace(data_path)
            audit=dict(record_expected,data_sha256=file_hash(data_path));_save_json(path,audit)
        if z.shape!=(len(rows),config['projections']) or not np.isfinite(z).all():raise ValueError('Invalid projected residual shape')
        result.append(z);audits.append(path);covered+=len(rows)
        _progress(root,state='replaying_original_beats' if verify else 'extracting_original_beats',projection_seed=seed,
                  outer_fold=fold,inner_fold=inner,records_complete=j+1,context_records=len(records),context_beats_complete=covered,coverage_mode=mode)
    return result,audits


def fit_gain(projected,record_patients,train_patients,floor):
    train_patients=np.asarray(train_patients);record_patients=np.asarray(record_patients)
    if len(np.unique(train_patients))!=len(train_patients) or not len(train_patients):raise ValueError('Invalid gain training membership')
    energies=[]
    for patient in train_patients:
        selected=np.flatnonzero(record_patients==patient)
        if not len(selected):raise ValueError('Training patient has no record')
        energies.append(np.mean([np.mean(projected[i]**2) for i in selected]))
    value=float(np.sqrt(np.mean(energies)))
    if not np.isfinite(value):raise ValueError('Nonfinite training residual RMS')
    return value if value>floor else 1.


def pool(projected,record_patients,ids,gain,config,base_features):
    """Same 144-column auxiliary grid for B/C/D, with a fixed energy convention."""
    record_patients=np.asarray(record_patients);ids=np.asarray(ids)
    if set(record_patients)!=set(ids) or len(set(ids))!=len(ids):raise ValueError('Patient/record pooling membership differs')
    record_values={a:[] for a in ARMS[1:]}
    for z in projected:
        values=summaries((np.asarray(z)/gain)[:,None,:],np.zeros((1,config['projections'])),config['quantiles'])
        for arm,key in zip(ARMS[1:],('mean_embedding','mean_spread','quantiles')):record_values[arm].append(values[key])
    # Common training-RMS-normalized auxiliary block has comparable energy to
    # the standardized full base vector. Fixed by dimensions, never outer scores.
    balance=np.sqrt(base_features.shape[1]/config['projections'])
    result={'baseline':np.asarray(base_features)}
    for arm in ARMS[1:]:
        record=np.asarray(record_values[arm])
        branch=np.stack([record[record_patients==p].mean(0) for p in ids])*balance
        result[arm]=np.concatenate([base_features,branch],axis=1)
    if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite pooled tensors')
    return result


def make_features(bank,config,root,scope,transform,source_x,source_context,train,query,seed,fold,inner,verify=False,smoke=False):
    rows=np.r_[train,query];records=bank.records_for(bank.ids[rows]);owner=bank.record_patients[records]
    scaler=StandardScaler().fit(source_x[:len(train)])
    base=scaler.transform(source_x)
    matrix=projection(config,source_x.shape[1],seed)
    projected,record_audits=projected_records(bank,config,root,scope,transform,source_context,scaler.scale_,matrix,records,seed,fold,inner,verify,smoke)
    gain=fit_gain(projected,owner,bank.ids[train],config['branch_scale_floor'])
    values=pool(projected,owner,bank.ids[rows],gain,config,base)
    state=dict(scaler_mean_sha256=array_hash(scaler.mean_),scaler_scale_sha256=array_hash(scaler.scale_),
               projection_sha256=array_hash(matrix),residual_rms=gain,train_membership_sha256=array_hash(bank.ids[train]),
               record_audit_hashes={str(p.relative_to(root)):file_hash(p) for p in record_audits},
               coverage_mode='smoke_first_beats' if smoke else 'all_original_beats',
               original_beats_covered=sum(len(z) for z in projected),n_records=len(records),base_dimensions=source_x.shape[1],
               auxiliary_dimensions=config['projections']*len(config['quantiles']))
    parameters=dict(scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,projection=matrix,residual_rms=gain)
    return values,state,parameters
