"""Read-only exact-partition reuse and independent replay of the frozen baseline."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import yaml

from . import benchmark, chronology_bank, rocket_bank, rocket_features, rocket_study
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash


@dataclass
class SourceBank:
    records: object
    templates: np.ndarray
    source_config: dict
    source_scope: dict
    source_root: Path
    provenance: dict

    @property
    def ids(self): return self.records.ids
    @property
    def labels(self): return self.records.labels
    @property
    def record_patients(self): return self.records.record_patients
    def rows(self, record): return self.records.rows(record)
    def records_for(self, patients): return self.records.records_for(patients)


def protected_files(config):
    root = Path(config['output_dir'])/'complete'
    files = [root/n for n in ('study_manifest.json', 'summary.json', 'REPORT.md')]
    for fold in range(1, config['outer_splits']+1):
        for seed in config['transform_seeds']:
            for inner in range(config['inner_splits']+1):
                p = root/f'context_{fold:02d}_s{seed}_i{inner}.json'
                files += [p, p.with_suffix('.joblib'), p.with_suffix('.npz')]
                for candidate in (range(len(config['ridge_alphas'])) if inner else [None]):
                    suffix = f'_i{inner}_a{candidate}' if inner else ''
                    p = root/f"{'inner' if inner else 'outer'}_{fold:02d}_combined_s{seed}{suffix}.json"
                    files += [p, p.with_suffix('.joblib'), p.with_suffix('.npz')]
            files.append(root/f'selection_{fold:02d}_combined_s{seed}.json')
    return {str(p): file_hash(p) for p in files}


def load(config):
    original = yaml.safe_load(Path(config['source_config']).read_text())
    for new, old in [('classes','classes'), ('folds_path','folds_path'), ('outer_splits','outer_splits'),
                     ('inner_splits','inner_splits'), ('split_seed','split_seed'), ('ridge_alphas','ridge_alphas'),
                     ('projection_seeds','transform_seeds'), ('torch_threads','torch_threads')]:
        if config[new] != original[old]: raise ValueError('New controls do not match frozen source protocol')
    records = chronology_bank.prepare_bank(original)
    # Redirect only the nonnumerical audit output. Frozen result files stay read only.
    templates = rocket_bank.prepare(dict(original, output_dir=str(Path(config['output_dir'])/'source_bank_audit')))
    if not np.array_equal(records.ids, templates.ids) or not np.array_equal(records.record_patients, templates.record_patients):
        raise ValueError('Measured-record bank and frozen templates differ')
    scope = rocket_study.make_scope(original, templates)
    root = Path(original['output_dir'])/'complete'
    if scope != json.loads((root/'study_manifest.json').read_text()): raise ValueError('Pinned source/config/runtime changed')
    summary = json.loads((root/'summary.json').read_text())
    budget = rocket_study.budget(original)
    if (summary.get('complete') is not True or summary.get('study_fingerprint') != scope['study_fingerprint']
            or summary.get('all_models_restored') != budget['fits']
            or summary.get('feature_contexts_reconstructed') != budget['feature_contexts']):
        raise ValueError('Require completed independently verified convolution source')
    snapshot = protected_files(original)
    bank = SourceBank(records, templates.templates, original, scope, root,
                      dict(source_scope=scope, protected_files=snapshot, source_config_sha256=file_hash(config['source_config'])))
    folds = benchmark.load_or_create_folds(bank.labels, bank.ids, original)
    assigned = folds.set_index('patient_id').loc[bank.ids, 'fold'].to_numpy()
    return bank, assigned


def context(bank, seed, fold, inner, train, query, reconstruct=False):
    path = bank.source_root/f'context_{fold:02d}_s{seed}_i{inner}.json'
    audit = json.loads(path.read_text())
    expected = rocket_study.expected_membership(bank, bank.source_scope, train, query)
    expected.update(transform_seed=seed, fit_seed=seed+fold*1000+inner, outer_fold=fold, inner_fold=inner,
                    train_records_sha256=array_hash(bank.records_for(bank.ids[train])),
                    n_train_records=len(bank.records_for(bank.ids[train])))
    if (any(audit.get(k) != v for k,v in expected.items()) or file_hash(path.with_suffix('.joblib')) != audit['transform_sha256']
            or file_hash(path.with_suffix('.npz')) != audit['features_sha256']):
        raise ValueError('Frozen source feature state/membership changed')
    transform = joblib.load(path.with_suffix('.joblib'))
    if transform.audit() != audit['fit']: raise ValueError('Frozen convolution state changed')
    with np.load(path.with_suffix('.npz')) as f:
        if not np.array_equal(f['rows'], np.r_[train,query]): raise ValueError('Source row order changed')
        values = {arm:f[arm] for arm in bank.source_config['arms']}
    if reconstruct:
        tr = bank.records_for(bank.ids[train]);selected = bank.records_for(bank.ids[np.r_[train,query]])
        fresh = rocket_features.FeatureTransform(bank.source_config, expected['fit_seed']).fit(bank.templates[tr],bank.record_patients[tr])
        if fresh.audit() != audit['fit']: raise ValueError('Source transform cannot reconstruct from training patients')
        rebuilt = rocket_features.aggregate(fresh.transform(bank.templates[selected]),bank.record_patients[selected],bank.ids[np.r_[train,query]])
        if any(not np.allclose(rebuilt[a],values[a],rtol=1e-6,atol=1e-6) for a in values):
            raise ValueError('Reconstructed source template features disagree')
    return transform, values['combined'], path


def ridge(bank, seed, fold, inner, candidate, train, query, x):
    suffix=f'_i{inner}_a{candidate}' if inner else ''
    path=bank.source_root/f"{'inner' if inner else 'outer'}_{fold:02d}_combined_s{seed}{suffix}.json"
    audit=json.loads(path.read_text());alpha=bank.source_config['ridge_alphas'][candidate]
    expected=rocket_study.expected_membership(bank,bank.source_scope,train,query)
    ctx=bank.source_root/f'context_{fold:02d}_s{seed}_i{inner}.json'
    expected.update(transform_seed=seed,outer_fold=fold,inner_fold=inner,arm='combined',candidate_id=candidate,alpha=alpha,
                    context_sha256=file_hash(ctx),training_features_sha256=array_hash(x[:len(train)]),
                    query_features_sha256=array_hash(x[len(train):]),n_features=x.shape[1])
    if (any(audit.get(k)!=v for k,v in expected.items()) or file_hash(path.with_suffix('.joblib'))!=audit['model_sha256']
            or file_hash(path.with_suffix('.npz'))!=audit['probabilities_sha256']):raise ValueError('Source ridge context changed')
    model=joblib.load(path.with_suffix('.joblib'));scaler=StandardScaler().fit(x[:len(train)])
    if (not np.array_equal(scaler.mean_,model[0].mean_) or not np.array_equal(scaler.scale_,model[0].scale_)
            or model[-1].alpha!=alpha or model[-1].class_weight!='balanced' or model[-1].solver!='cholesky'):
        raise ValueError('Source ridge scaling/settings changed')
    with np.load(path.with_suffix('.npz')) as f:
        if not np.array_equal(f['rows'],query):raise ValueError('Source ridge query rows changed')
        saved=f['probabilities']
    p=rocket_study.predict(model,x[len(train):],len(bank.source_config['classes']))
    if not np.allclose(p,saved,rtol=1e-10,atol=1e-12):raise ValueError('Source ridge restore disagrees')
    if inner:
        score=float(f1_score(bank.labels[query],p.argmax(1),labels=np.arange(len(bank.source_config['classes'])),average='macro',zero_division=0))
        if score!=audit['validation_macro_f1']:raise ValueError('Source inner F1 changed')
    return p,audit,path


def verify_baseline(bank,assigned,config,write=True):
    root=Path(config['output_dir'])/'source_preflight';root.mkdir(parents=True,exist_ok=True)
    members={s:np.full((len(bank.ids),len(config['classes'])),np.nan) for s in config['projection_seeds']}
    restored=contexts=0
    for fold in range(1,config['outer_splits']+1):
        train,test=np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
        splits=[(train[t],train[v]) for t,v in benchmark.inner_patient_splits(bank.labels[train],bank.ids[train],config['inner_splits'],config['split_seed']+fold*1000)]
        for seed in config['projection_seeds']:
            scores=[[] for _ in config['ridge_alphas']]
            for inner,(ti,vi) in enumerate(splits,1):
                _,x,_=context(bank,seed,fold,inner,ti,vi,True);contexts+=1
                for candidate in range(len(scores)):
                    _,audit,_=ridge(bank,seed,fold,inner,candidate,ti,vi,x);restored+=1
                    scores[candidate].append(audit['validation_macro_f1'])
            _,x,_=context(bank,seed,fold,0,train,test,True);contexts+=1
            means=[float(np.mean(v)) for v in scores];selected=max(range(len(means)),key=means.__getitem__)
            decision=json.loads((bank.source_root/f'selection_{fold:02d}_combined_s{seed}.json').read_text())
            if decision['inner_scores']!=scores or decision['selected_candidate']!=selected:raise ValueError('Source alpha selection changed')
            p,_,_=ridge(bank,seed,fold,0,selected,train,test,x);restored+=1;members[seed][test]=p
            _progress(root,state='replaying_baseline',projection_seed=seed,outer_fold=fold,models_restored=restored,contexts_reconstructed=contexts)
    if any(not np.isfinite(v).all() for v in members.values()):raise ValueError('Incomplete baseline cohort')
    ensemble=np.mean(list(members.values()),axis=0)
    metric=benchmark.metric_bundle(bank.labels,ensemble,config['classes'])
    old=json.loads((bank.source_root/'summary.json').read_text())['ensemble_metrics']['combined']
    if metric!=old:raise ValueError('Full source baseline reproduction differs')
    result=dict(complete=True,source_fingerprint=_hash(bank.provenance),verification_code_sha256=file_hash(__file__),source_models_restored=restored,
                source_contexts_reconstructed=contexts,source_outer_models=len(members)*config['outer_splits'],
                ensemble_macro_f1=metric['macro_f1'],member_probability_hashes={str(s):array_hash(p) for s,p in members.items()},
                ensemble_probability_sha256=array_hash(ensemble),source_files_unchanged=protected_files(bank.source_config)==bank.provenance['protected_files'])
    if not result['source_files_unchanged']:raise ValueError('Pinned source artifacts changed during replay')
    path=root/'summary.json'
    if write:_save_json(path,result)
    elif json.loads(path.read_text())!=result:raise ValueError('Source preflight receipt changed')
    _progress(root,state='complete_verified',models_restored=restored,contexts_reconstructed=contexts)
    return result
