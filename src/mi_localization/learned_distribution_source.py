"""Read-only frozen morphology teachers, exact partitions and original signals."""
from __future__ import annotations
from dataclasses import dataclass,field
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from . import distribution_kernel_source as cached,chronology_bank,chronology_model,chronology_study
from . import residual_context_model,staged_context_study
from .contrastive import initial_state_hash
from .neural import _hash
from .signal_audit import file_hash


@dataclass
class LearnedBank:
    cache: object
    records: object
    templates: np.ndarray
    teacher_config: dict
    teacher_scope: dict
    teacher_root: Path
    provenance: dict
    normalizers: dict=field(default_factory=dict)
    @property
    def ids(self):return self.cache.ids
    @property
    def labels(self):return self.cache.labels
    @property
    def record_patients(self):return self.records.record_patients
    def rows(self,r):return self.records.rows(r)
    def records_for(self,ids):return self.records.records_for(ids)


def load(config):
    base,protected=cached.validate_chain(config)
    tc=yaml.safe_load(Path(config['teacher_config']).read_text());root=Path(tc['output_dir'])/'complete'
    scope=json.loads((root/'study_manifest.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if (scope['config']!=tc or scope['study_fingerprint']!=config['expected_teacher_fingerprint']
            or file_hash(root/'summary.json')!=config['expected_teacher_summary_sha256'] or summary.get('complete') is not True
            or summary.get('all_phase_models_restored')!=summary.get('phase_fits') or summary.get('all_query_beats_covered') is not True):
        raise ValueError('Require pinned independently completed teacher source')
    for key in ('classes','folds_path','outer_splits','inner_splits','split_seed','torch_threads'):
        if config[key]!=tc[key]:raise ValueError('Teacher partitions/settings differ')
    if config['projection_seeds']!=tc['training_seeds']:raise ValueError('Teacher seed set differs')
    def check(p,h=None):
        p=Path(p);actual=file_hash(p)
        if h is not None and actual!=h:raise ValueError(f'Teacher source file changed: {p}')
        protected[str(p)]=actual
    for name in ('study_manifest.json','summary.json','REPORT.md'):check(root/name)
    check(config['teacher_config'])
    original=scope['source_protocol'];signals=original['source'];raw=signals['source']
    for name,h in raw['files'].items():check(Path(tc['bank_dir'])/name,h)
    check(tc['beats_path'],raw['beats_sha256']);check(tc['paper_config'],raw['paper_sha256'])
    check(Path(tc['sequence_bank_dir'])/'metadata.npz',signals['metadata_sha256'])
    waves=np.load(Path(tc['bank_dir'])/'waveforms.npy',mmap_mode='r');groups=np.load(Path(tc['bank_dir'])/'groups.npy');y=cached.encode_labels(np.load(Path(tc['bank_dir'])/'y.npy'),tc['classes'])
    with np.load(Path(tc['sequence_bank_dir'])/'metadata.npz') as f:arrays={k:f[k] for k in f.files}
    records=chronology_bank.RecordBank(waves,y,groups,provenance=signals,**arrays)
    if not np.array_equal(records.ids,base.ids) or not np.array_equal(records.labels,base.labels):raise ValueError('Teacher and feature cohort differ')
    # This re-creates only the numerical scope, never old reports or progress.
    if staged_context_study.make_scope(tc,records)!=scope:raise ValueError('Teacher code/config/runtime changed')
    code_paths={'code_sha256':'staged_context_study.py','model_code_sha256':'staged_context_model.py','report_code_sha256':'staged_context_report.py'}
    for key,name in code_paths.items():check(Path(__file__).with_name(name),scope[key])
    for key,name in [('code_sha256','residual_context_study.py'),('model_code_sha256','residual_context_model.py'),('metadata_code_sha256','chronology_bank.py'),('shared_model_code_sha256','chronology_model.py'),('cache_code_sha256','chronology_study.py'),('neural_code_sha256','neural.py'),('metric_code_sha256','benchmark.py'),('report_code_sha256','residual_context_report.py')]:check(Path(__file__).with_name(name),original[key])
    template_manifest=base.source_scope['source']['source_scope']['bank']['bank_manifest']
    template_cfg=yaml.safe_load(Path(base.source_config['source_config']).read_text());p=Path(template_cfg['template_bank_dir'])/'templates.npz';check(p,template_manifest['templates_sha256'])
    with np.load(p) as f:templates=f['templates']
    for fold in range(1,config['outer_splits']+1):
        for seed in config['projection_seeds']:
            for inner in range(config['inner_splits']+1):
                path=teacher_path(root,seed,fold,inner);a=json.loads(path.read_text())
                check(path);check(path.with_suffix('.pt'),a['model_sha256']);check(path.with_suffix('.npz'),a['probabilities_sha256'])
                if a['arm']!='morphology' or a['phase']!='morphology':raise ValueError('Require plain measured-VCG morphology teachers')
    provenance=dict(cached_features=base.provenance,teacher_fingerprint=scope['study_fingerprint'],teacher_config_sha256=file_hash(config['teacher_config']),
                    teacher_summary_sha256=file_hash(root/'summary.json'),protected_snapshot_sha256=_hash(protected),protected_files=len(protected),
                    waveform_sha256=file_hash(Path(tc['bank_dir'])/'waveforms.npy'),embedding_device=config['device'])
    return LearnedBank(base,records,templates,tc,scope,root,provenance),base.assigned,protected


def teacher_path(root,seed,fold,inner):return root/(f'inner_{fold:02d}_morphology_s{seed}_i{inner}.json' if inner else f'outer_{fold:02d}_morphology_s{seed}.json')


def restore_teacher(bank,config,seed,fold,inner,train,query):
    tr,qu=bank.ids[train],bank.ids[query];path=teacher_path(bank.teacher_root,seed,fold,inner);a=json.loads(path.read_text());tc=bank.teacher_config
    expected=chronology_study.expected_fit(bank.records,bank.teacher_scope,'morphology',seed,fold,inner,tr,qu);expected.update(phase='morphology',sequence_length=tc['sequence_length'])
    if not inner:
        choices=[]
        for i in range(1,tc['inner_splits']+1):
            p=teacher_path(bank.teacher_root,seed,fold,i);v=json.loads(p.read_text());choices.append(dict(inner_fold=i,selected_epoch=v['fit']['selected_epoch'],audit_sha256=file_hash(p)))
        epochs=int(np.floor(np.median([x['selected_epoch'] for x in choices])+.5));expected.update(final_epochs=epochs,inner_choices=choices)
    if (any(a.get(k)!=v for k,v in expected.items()) or file_hash(path.with_suffix('.pt'))!=a['model_sha256']
            or file_hash(path.with_suffix('.npz'))!=a['probabilities_sha256']):raise ValueError('Teacher membership/checkpoint differs')
    state=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
    if state['expected']!=expected or state['normalization']!=a['fit']['normalization']:raise ValueError('Teacher saved scope differs')
    key=_hash(tr.tolist())
    if key not in bank.normalizers:bank.normalizers[key]=chronology_model.SequenceNormalizer().fit(bank.records,tr)
    norm=bank.normalizers[key]
    if norm.audit()!=state['normalization']:raise ValueError('Teacher normalization is not from exact training patients')
    fit_seed=seed+fold*1000+inner if inner else seed+fold
    model=residual_context_model.initial_model(tc,fit_seed,'beat_only')
    if a['fit']['initial_state_sha256']!=initial_state_hash(model) or a['fit']['seed']!=fit_seed:raise ValueError('Teacher initialization differs')
    history=a['fit']['history'];best=-np.inf;epoch=0
    for i,row in enumerate(history,1):
        if row['epoch']!=i:raise ValueError('Teacher epoch sequence differs')
        if inner:
            score=row['validation_patient_macro_f1']
            if score>best+1e-8:best,epoch=score,i
        elif 'validation_patient_macro_f1' in row:raise ValueError('Outer teacher monitored query labels')
    if inner:
        if a['fit']['selected_epoch']!=epoch or a['fit']['best_validation_macro_f1']!=best:raise ValueError('Teacher inner stopping changed')
    elif len(history)!=epochs or a['fit']['selected_epoch']!=epochs:raise ValueError('Teacher outer duration changed')
    model.load_state_dict(state['state_dict']);model.to(config['device']);model.eval();model.requires_grad_(False)
    with np.load(path.with_suffix('.npz')) as f:
        if not np.array_equal(f['patient_ids'],qu):raise ValueError('Teacher query patient order differs')
        probabilities=f['probabilities']
    receipt=dict(teacher_audit_sha256=file_hash(path),teacher_model_sha256=a['model_sha256'],teacher_predictions_sha256=a['probabilities_sha256'],
                 train_membership_sha256=expected['train_membership_sha256'],query_membership_sha256=expected['query_membership_sha256'],
                 normalization_sha256=_hash(norm.audit()),selected_epoch=a['fit']['selected_epoch'],source_inference_device=bank.teacher_scope['source_protocol']['runtime']['device'],new_inference_device=config['device'],encoder_frozen=True)
    return model,norm,probabilities,receipt,path


def encode(model,norm,waves,batch_size):
    embeddings=[];probabilities=[]
    with torch.inference_mode():
        for start in range(0,len(waves),batch_size):
            part=np.asarray(waves[start:start+batch_size],dtype=np.float32)
            x,_=norm.wave.transform(part,np.empty((len(part),0),np.float32));t=torch.from_numpy(x).to(next(model.parameters()).device)
            e=model.encoder(t);p=model.beat_head(e).softmax(-1)
            embeddings.append(e.cpu().numpy());probabilities.append(p.cpu().numpy())
    e=np.concatenate(embeddings);p=np.concatenate(probabilities)
    if e.shape!=(len(waves),64) or not np.isfinite(e).all() or not np.isfinite(p).all():raise ValueError('Invalid learned embedding')
    return e,p
