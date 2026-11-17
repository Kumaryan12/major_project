"""Read-only chain of custody for already independently replayed feature caches."""
from __future__ import annotations
from dataclasses import dataclass
import json
import platform
from pathlib import Path
import joblib
import scipy
import sklearn
import torch
import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax
from .distribution_study import membership
from .multiview_bank import array_hash
from .neural import _hash
from .signal_audit import file_hash


@dataclass
class CachedBank:
    ids: np.ndarray
    labels: np.ndarray
    assigned: np.ndarray
    root: Path
    source_config: dict
    source_scope: dict
    summary: dict
    provenance: dict
    n_records: int
    n_beats: int


def encode_labels(values,classes):
    values=np.asarray(values)
    if values.ndim!=1:raise ValueError('Invalid source label dimensions')
    if values.dtype.kind in ('U','S','O'):
        text=values.astype(str);lookup={name:i for i,name in enumerate(classes)}
        if not set(text).issubset(lookup):raise ValueError('Unknown source label')
        return np.array([lookup[v] for v in text],dtype=np.int64)
    if not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all() or not set(values).issubset(range(len(classes))):raise ValueError('Invalid numeric source label')
    return values.astype(np.int64)


def validate_chain(config):
    source_config=yaml.safe_load(Path(config['source_config']).read_text());root=Path(source_config['output_dir'])/'complete'
    scope=json.loads((root/'study_manifest.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if (scope['config']!=source_config or scope['study_fingerprint']!=config['expected_source_fingerprint']
            or file_hash(root/'summary.json')!=config['expected_source_summary_sha256']
            or summary.get('complete') is not True or summary['study_fingerprint']!=scope['study_fingerprint']
            or any(summary.get(k) is not True for k in ('all_original_beats_replayed','original_cohort_unchanged','source_files_unchanged','baseline_probabilities_exact'))):
        raise ValueError('Require pinned independently completed source features')
    for key in ('classes','folds_path','outer_splits','inner_splits','split_seed','projection_seeds','ridge_alphas','torch_threads'):
        if config[key]!=source_config[key]:raise ValueError('Source partitions/settings differ')
    if config['representations']!=source_config['arms']:raise ValueError('Require all source representations')
    runtime=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=str(torch.__version__),joblib=joblib.__version__,platform=platform.platform(),device='cpu',blas_threads=1)
    if scope['runtime']!=runtime:raise ValueError('Pinned source numerical runtime changed')
    expected_groups=config['outer_splits']*len(config['projection_seeds']);expected_fits=4*expected_groups*(config['inner_splits']*len(config['ridge_alphas'])+1)
    if (summary['all_models_restored']!=expected_fits or summary['matched_context_checks']!=expected_groups*(config['inner_splits']+1)
            or summary['report_sha256']!=file_hash(root/'REPORT.md') or summary['verification_manifest_sha256']!=file_hash(root/'verification_manifest.json')):
        raise ValueError('Incomplete/changed upstream coverage')
    protected={}
    def check(path,expected=None):
        path=Path(path);h=file_hash(path)
        if expected is not None and h!=expected:raise ValueError(f'Pinned source artifact changed: {path}')
        protected[str(path)]=h
    for name in ('study_manifest.json','summary.json','REPORT.md','verification_manifest.json'):check(root/name)
    for paths in (scope['code'],scope['source']['protected_files']):
        for path,h in paths.items():check(path,h)
    verification=json.loads((root/'verification_manifest.json').read_text())
    for name,h in verification['restored_audit_hashes'].items():
        path=root/name;check(path,h);audit=json.loads(path.read_text())
        for key,ext in [('model_sha256','.joblib'),('probabilities_sha256','.npz'),('parameters_sha256','.joblib'),('features_sha256','.npz')]:
            if key in audit:check(path.with_suffix(ext),audit[key])
        for record,h in audit.get('state',{}).get('record_audit_hashes',{}).items():
            rp=root/record;check(rp,h);check(rp.with_suffix('.npz'),json.loads(rp.read_text())['data_sha256'])
    # Keep the original declared metadata/labels/R-position provenance, with no signal decoding or cache writes.
    original=yaml.safe_load(Path(source_config['source_config']).read_text())
    check(source_config['source_config'],scope['source']['source_config_sha256'])
    inherited=scope['source']['source_scope']['bank']['bank_manifest']['scope']['source']
    check(original['beats_path'],inherited['source']['beats_sha256'])
    check(original['paper_config'],inherited['source']['paper_sha256'])
    for name in ('y.npy','groups.npy'):check(Path(original['bank_dir'])/name,inherited['source']['files'][name])
    check(Path(original['sequence_bank_dir'])/'metadata.npz',inherited['metadata_sha256'])
    check(config['folds_path'],scope['folds_sha256'])
    groups=np.load(Path(original['bank_dir'])/'groups.npy');y=encode_labels(np.load(Path(original['bank_dir'])/'y.npy'),config['classes']);ids=np.unique(groups)
    if len(groups)!=len(y) or any(len(np.unique(y[groups==p]))!=1 for p in ids):raise ValueError('Patient labels inconsistent')
    labels=np.array([y[groups==p][0] for p in ids]);folds=pd.read_csv(config['folds_path'])
    if folds.patient_id.duplicated().any() or set(folds.patient_id)!=set(ids):raise ValueError('Frozen patient folds differ')
    assigned=folds.set_index('patient_id').loc[ids,'fold'].to_numpy()
    if set(assigned)!=set(range(1,config['outer_splits']+1)):raise ValueError('Invalid frozen fold coverage')
    with np.load(Path(original['sequence_bank_dir'])/'metadata.npz') as f:n_records=len(f['record_patients'])
    if len(ids)!=summary['n_patients'] or len(y)!=summary['n_beats'] or n_records!=summary['n_records']:raise ValueError('Cohort changed')
    for fold in range(1,config['outer_splits']+1):
        for seed in config['projection_seeds']:
            for rep in config['representations']:check(root/'review'/f'selection_{fold:02d}_{rep}_s{seed}.json')
    provenance=dict(source_fingerprint=scope['study_fingerprint'],source_summary_sha256=file_hash(root/'summary.json'),
                    verified_snapshot_sha256=_hash(protected),protected_file_count=len(protected),source_config_sha256=file_hash(config['source_config']),
                    patient_ids_sha256=array_hash(ids),labels_sha256=array_hash(labels),folds_sha256=file_hash(config['folds_path']),
                    claim='Reuse independently verified upstream caches; no waveform re-extraction or new raw-signal validation')
    return CachedBank(ids,labels,assigned,root,source_config,scope,summary,provenance,n_records,len(y)),protected


def context(bank,seed,fold,inner,train,query):
    path=bank.root/'review/contexts'/f'context_{fold:02d}_s{seed}_i{inner}.json';a=json.loads(path.read_text())
    expected=membership(bank,bank.source_scope,train,query);expected.update(projection_seed=seed,outer_fold=fold,inner_fold=inner,coverage_mode='all_original_beats')
    if (any(a.get(k)!=v for k,v in expected.items()) or file_hash(path.with_suffix('.npz'))!=a['features_sha256']
            or file_hash(path.with_suffix('.joblib'))!=a['parameters_sha256']):raise ValueError('Source cache membership/hash changed')
    with np.load(path.with_suffix('.npz')) as f:
        if not np.array_equal(f['rows'],np.r_[train,query]):raise ValueError('Source cached row order changed')
        values={rep:np.asarray(f[rep],dtype=np.float64) for rep in bank.source_config['arms']}
    d=a['state']['base_dimensions']
    if any(not np.isfinite(v).all() or len(v)!=len(train)+len(query) for v in values.values()):raise ValueError('Invalid cached features')
    if any(not np.array_equal(values[rep][:,:d],values['baseline']) for rep in bank.source_config['arms'][1:]):raise ValueError('Full source base differs')
    return values,path


def reference(bank,seed,fold,inner,rep,candidate,train,query,features,context_path):
    suffix=f'_i{inner}_a{candidate}' if inner else ''
    if rep=='baseline':
        ref=bank.root/'review/baseline_refs'/f"{'inner' if inner else 'outer'}_{fold:02d}_s{seed}{suffix}.json"
        ref_a=json.loads(ref.read_text());path=Path(ref_a['source_audit'])
        if ref_a['context_sha256']!=file_hash(context_path) or ref_a['source_audit_sha256']!=file_hash(path):raise ValueError('Source baseline reference changed')
    else:path=bank.root/'review/fits'/f"{'inner' if inner else 'outer'}_{fold:02d}_{rep}_s{seed}{suffix}.json"
    audit=json.loads(path.read_text())
    if (audit['train_membership_sha256']!=array_hash(bank.ids[train]) or audit['query_membership_sha256']!=array_hash(bank.ids[query])
            or audit['training_labels_sha256']!=array_hash(bank.labels[train])):raise ValueError('Source ridge patient membership changed')
    if audit['candidate_id']!=candidate or audit['alpha']!=bank.source_config['ridge_alphas'][candidate]:raise ValueError('Source candidate selection differs')
    if file_hash(path.with_suffix('.joblib'))!=audit['model_sha256'] or file_hash(path.with_suffix('.npz'))!=audit['probabilities_sha256']:raise ValueError('Source classifier cache changed')
    if rep!='baseline' and audit['context_sha256']!=file_hash(context_path):raise ValueError('Source classifier uses another context')
    model=joblib.load(path.with_suffix('.joblib'))
    if rep=='baseline':model=model[-1]
    p=softmax(model.decision_function(features[len(train):]),axis=1)
    with np.load(path.with_suffix('.npz')) as f:
        if not np.array_equal(f['rows'],query) or not np.allclose(p,f['probabilities'],rtol=1e-10,atol=1e-12):raise ValueError('Restored upstream ridge disagrees')
    return p,audit,path
