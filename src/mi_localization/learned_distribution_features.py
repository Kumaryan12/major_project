"""Every original beat through frozen partition-local encoders, with record resume."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from . import learned_distribution_source as source,convolution_distribution_features as pooling
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash,_save_json
from .signal_audit import file_hash


def record_embeddings(bank,config,scope,root,model,norm,receipt,records,seed,fold,inner,verify=False):
    directory=root/'review/embeddings'/f'f{fold:02d}_s{seed}_i{inner}';directory.mkdir(parents=True,exist_ok=True)
    results=[];probabilities=[];paths=[];covered=0
    expected=dict(study_fingerprint=scope['study_fingerprint'],teacher_model_sha256=receipt['teacher_model_sha256'],normalization_sha256=receipt['normalization_sha256'],
                  device=config['device'],batch_size=config['embedding_batch_size'])
    for index,record in enumerate(records):
        rows=bank.rows(record);path=directory/f'record_{record:04d}.json';data=path.with_suffix('.npz')
        row_expected=dict(expected,record_index=int(record),rows_sha256=array_hash(rows),n_beats=len(rows))
        if verify or path.exists():
            a=json.loads(path.read_text())
            if any(a.get(k)!=v for k,v in row_expected.items()) or file_hash(data)!=a['data_sha256']:raise ValueError('Learned record cache changed')
            with np.load(data) as f:
                if not np.array_equal(f['rows'],rows):raise ValueError('Embedding beat membership changed')
                e,p=f['embeddings'],f['probabilities']
            if verify:
                fresh,fp=source.encode(model,norm,np.asarray(bank.records.waves[rows]),config['embedding_batch_size'])
                if any(not np.allclose(x,y,rtol=config['embedding_replay_rtol'],atol=config['embedding_replay_atol']) for x,y in [(fresh,e),(fp,p)]):raise ValueError('Original waveform embedding replay disagrees')
        else:
            e,p=source.encode(model,norm,np.asarray(bank.records.waves[rows]),config['embedding_batch_size'])
            temp=data.with_suffix('.partial.npz');np.savez_compressed(temp,rows=rows,embeddings=e,probabilities=p);temp.replace(data)
            _save_json(path,dict(row_expected,data_sha256=file_hash(data)))
        if e.shape!=(len(rows),64) or p.shape!=(len(rows),len(config['classes'])) or not np.isfinite(e).all() or not np.isfinite(p).all():raise ValueError('Invalid learned record features')
        results.append(e);probabilities.append(p);paths.append(path);covered+=len(rows)
        _progress(root,state='replaying_original_embeddings' if verify else 'extracting_original_embeddings',projection_seed=seed,outer_fold=fold,inner_fold=inner,
                  records_complete=index+1,context_records=len(records),context_beats_complete=covered)
    return results,probabilities,paths


def teacher_probabilities(probabilities,owners,patients):
    owners=np.asarray(owners)
    return np.stack([np.mean([np.asarray(probabilities[r],dtype=np.float64).mean(0) for r in np.flatnonzero(owners==p)],axis=0) for p in patients])


def make_features(bank,config,scope,root,seed,fold,inner,train,query,verify=False):
    values,base_context=source.cached.context(bank.cache,seed,fold,inner,train,query);base=values['baseline']
    model,norm,old_prob,receipt,path=source.restore_teacher(bank,config,seed,fold,inner,train,query)
    rows=np.r_[train,query];patients=bank.ids[rows];records=bank.records_for(patients);owners=bank.record_patients[records]
    medians,_=source.encode(model,norm,bank.templates[records],config['embedding_batch_size'])
    median_patient=np.stack([medians[owners==p].astype(np.float64).mean(0) for p in patients])
    scaler=StandardScaler().fit(median_patient[:len(train)])
    matrix=pooling.projection(config,64,seed)
    embeddings,probabilities,audits=record_embeddings(bank,config,scope,root,model,norm,receipt,records,seed,fold,inner,verify)
    # Verify the original teacher's declared query predictor, using every beat and equal-record aggregation.
    restored=teacher_probabilities(probabilities,owners,bank.ids[query])
    if (not np.allclose(restored,old_prob,rtol=config['teacher_probability_rtol'],atol=config['teacher_probability_atol'])
            or not np.array_equal(restored.argmax(1),old_prob.argmax(1))):raise ValueError('Frozen source teacher prediction does not reproduce')
    projected=[((np.asarray(e,dtype=np.float64)-medians[i].astype(np.float64))/scaler.scale_)@matrix for i,e in enumerate(embeddings)]
    gain=pooling.fit_gain(projected,owners,bank.ids[train],config['branch_scale_floor'])
    result=pooling.pool(projected,owners,patients,gain,config,base)
    state=dict(coverage_mode='all_original_beats',teacher=receipt,source_base_context_sha256=file_hash(base_context),median_embeddings_sha256=array_hash(medians),
               scaler_mean_sha256=array_hash(scaler.mean_),scaler_scale_sha256=array_hash(scaler.scale_),projection_sha256=array_hash(matrix),residual_rms=gain,
               record_audit_hashes={str(p.relative_to(root)):file_hash(p) for p in audits},original_beats_covered=sum(map(len,embeddings)),n_records=len(records),
               base_dimensions=base.shape[1],auxiliary_dimensions=config['projections']*len(config['quantiles']),embedding_dimensions=64,source_teacher_query_labels_reproduced=True,
               max_teacher_probability_error=float(np.max(np.abs(restored-old_prob))))
    params=dict(scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,projection=matrix,residual_rms=gain)
    return result,state,params,base_context
