"""Frozen teacher isolation, full-beat learned pooling and independent replay."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from mi_localization import learned_distribution_source as source,learned_distribution_features as features,learned_distribution_study as study,learned_distribution_report as report
from mi_localization import convolution_distribution_study as old_study,convolution_distribution_report as old_report,distribution_kernel_source as cached
from mi_localization import chronology_bank,chronology_model,chronology_study,residual_context_model,benchmark
from mi_localization.contrastive import initial_state_hash
from mi_localization.signal_audit import file_hash
from test_convolution_distribution import small as convolution_small


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    with threadpool_limits(limits=1):yield
    torch.set_num_threads(old)


@pytest.fixture
def small(convolution_small,monkeypatch):
    config,old,assigned=convolution_small;old_study.run(config);summary=old_report.summarize(config);root=Path(config['output_dir'])/'complete';scope=json.loads((root/'study_manifest.json').read_text())
    base=cached.CachedBank(old.ids,old.labels,assigned,root,config,scope,summary,{'synthetic_cache':True},len(old.record_patients),len(old.records.waves))
    owners=old.record_patients;rows=[old.rows(r) for r in range(len(owners))];lengths=np.array([len(x) for x in rows]);flat=np.concatenate(rows);offsets=np.r_[0,lengths.cumsum()]
    groups=np.repeat(owners,lengths);labels=np.array([old.labels[np.flatnonzero(old.ids==p)[0]] for p in groups]);rr=np.ones(len(groups),np.float32);valid=np.ones(len(groups),bool)
    for r in rows:rr[r[0]]=0;valid[r[0]]=False
    records=chronology_bank.RecordBank(old.records.waves,labels,groups,flat,offsets,owners,rr,valid,{'synthetic_signals':True})
    tc=yaml.safe_load(Path('configs/staged_context_v1.yaml').read_text());tc.update(device='cpu',output_dir=str(Path(config['output_dir'])/'teachers'),outer_splits=2,inner_splits=2,training_seeds=[42],widths=[4,8,16],hidden_size=8,rr_embedding=4,folds_path=config['folds_path'])
    ts={'study_fingerprint':'synthetic_teachers','source_protocol':{'runtime':{'device':'cpu'}}};teacher_root=Path(tc['output_dir']);teacher_root.mkdir()
    def save_teacher(fold,inner,tr,qu,choices=None):
        seed=42;fit_seed=seed+fold*1000+inner if inner else seed+fold;norm=chronology_model.SequenceNormalizer().fit(records,old.ids[tr]);model=residual_context_model.initial_model(tc,fit_seed,'beat_only');initial=initial_state_hash(model);model.eval()
        p,_=residual_context_model.predict(model,norm,records,old.ids[qu],tc,fit_seed)
        expected=chronology_study.expected_fit(records,ts,'morphology',seed,fold,inner,old.ids[tr],old.ids[qu]);expected.update(phase='morphology',sequence_length=tc['sequence_length'])
        if not inner:expected.update(final_epochs=1,inner_choices=choices)
        score=float(f1_score(old.labels[qu],p.argmax(1),labels=np.arange(6),average='macro',zero_division=0)) if inner else None
        history=[dict(epoch=1,validation_patient_macro_f1=score)] if inner else [dict(epoch=1)]
        fit=dict(seed=fit_seed,arm='beat_only',initial_state_sha256=initial,normalization=norm.audit(),selected_epoch=1,best_validation_macro_f1=score,history=history)
        path=source.teacher_path(teacher_root,seed,fold,inner);torch.save(dict(state_dict=model.state_dict(),normalization=norm.audit(),expected=expected),path.with_suffix('.pt'));np.savez_compressed(path.with_suffix('.npz'),patient_ids=old.ids[qu],probabilities=p)
        path.write_text(json.dumps(dict(expected,fit=fit,model_sha256=file_hash(path.with_suffix('.pt')),probabilities_sha256=file_hash(path.with_suffix('.npz')))))
        return dict(inner_fold=inner,selected_epoch=1,audit_sha256=file_hash(path))
    for fold in [1,2]:
        train,test=np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold);choices=[]
        for inner,(a,b) in enumerate(benchmark.inner_patient_splits(old.labels[train],old.ids[train],2,42+fold*1000),1):choices.append(save_teacher(fold,inner,train[a],train[b]))
        save_teacher(fold,0,train,test,choices)
    c=yaml.safe_load(Path('configs/learned_distribution_v1.yaml').read_text());c.update(output_dir=str(Path(config['output_dir'])/'learned'),folds_path=config['folds_path'],outer_splits=2,inner_splits=2,projection_seeds=[42],projections=3,quantiles=[.1,.5,.9],bootstrap_draws=20)
    bank=source.LearnedBank(base,records,old.templates,tc,ts,teacher_root,{'synthetic_teacher_checkpoints':True,'source_summary_sha256':file_hash(root/'summary.json')})
    monkeypatch.setattr(study,'load_inputs',lambda config,verify=False:(bank,assigned,{}))
    return c,bank,assigned


def test_encoder_frozen_eval_and_record_prediction_reproduction(small):
    c,b,a=small;train,query=np.flatnonzero(a!=1),np.flatnonzero(a==1);model,norm,old,receipt,_=source.restore_teacher(b,c,42,1,0,train,query)
    assert not model.training and not any(p.requires_grad for p in model.parameters()) and receipt['encoder_frozen']
    records=b.records_for(b.ids[query]);probs=[]
    for r in records:
        e,p=source.encode(model,norm,np.asarray(b.records.waves[b.rows(r)]),c['embedding_batch_size']);assert e.shape[1]==64;probs.append(p)
    np.testing.assert_allclose(features.teacher_probabilities(probs,b.record_patients[records],b.ids[query]),old,rtol=1e-5,atol=1e-6)
    assert set(b.normalizers)=={source._hash(b.ids[train].tolist())}


def test_teacher_partition_and_normalizer_tampering_rejected(small):
    c,b,a=small;train,query=np.flatnonzero(a!=1),np.flatnonzero(a==1)
    with pytest.raises(ValueError):source.restore_teacher(b,c,42,1,0,query,train)
    model,norm,_,_,path=source.restore_teacher(b,c,42,1,0,train,query)
    norm.wave.wave_mean_[0]+=1
    with pytest.raises(ValueError,match='normalization'):source.restore_teacher(b,c,42,1,0,train,query)
    path.with_suffix('.pt').write_bytes(path.with_suffix('.pt').read_bytes()+b'bad')
    with pytest.raises(ValueError):source.restore_teacher(b,c,42,1,0,train,query)


def test_teacher_features_query_independent_and_batch_invariant(small):
    c,b,a=small;train,query=np.flatnonzero(a!=1),np.flatnonzero(a==1);model,norm,_,receipt,_=source.restore_teacher(b,c,42,1,0,train,query)
    waves=np.asarray(b.records.waves[b.rows(0)]);ea,pa=source.encode(model,norm,waves,128);eb,pb=source.encode(model,norm,waves,1)
    np.testing.assert_allclose(ea,eb,rtol=1e-5,atol=1e-6);np.testing.assert_allclose(pa,pb,rtol=1e-5,atol=1e-6)
    original=norm.audit()
    for r in b.records_for(b.ids[query]):b.records.waves[b.rows(r)]*=10
    _,norm2,_,receipt2,_=source.restore_teacher(b,c,42,1,0,train,query)
    assert norm2.audit()==original and receipt2==receipt


def test_full_context_base_preservation_resume_and_raw_replay(small):
    c,b,a=small;root,scope=study.guard(c,b);train,query=np.flatnonzero(a!=1),np.flatnonzero(a==1)
    values,ctx,_=study.feature_context(b,c,scope,root,42,1,0,train,query);audit=json.loads(ctx.read_text());assert audit['state']['original_beats_covered']==len(b.records.waves)
    assert audit['state']['source_teacher_query_labels_reproduced']
    for arm in c['arms'][1:]:np.testing.assert_array_equal(values[arm][:,:values['baseline'].shape[1]],values['baseline'])
    study.feature_context(b,c,scope,root,42,1,0,train,query,True)
    paths=list((root/'review/embeddings').rglob('*.json'));before={p:p.read_bytes() for p in paths}
    ctx.unlink();ctx.with_suffix('.joblib').unlink();ctx.with_suffix('.npz').unlink();study.feature_context(b,c,scope,root,42,1,0,train,query)
    assert before=={p:p.read_bytes() for p in paths}
    b.records.waves[0,0,0]+=.1
    with pytest.raises(ValueError,match='replay'):study.feature_context(b,c,scope,root,42,1,0,train,query,True)


def test_record_cache_tamper_rejected(small):
    c,b,a=small;root,scope=study.guard(c,b);train,query=np.flatnonzero(a!=1),np.flatnonzero(a==1)
    study.feature_context(b,c,scope,root,42,1,0,train,query)
    p=next((root/'review/embeddings').rglob('*.npz'));p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError):study.feature_context(b,c,scope,root,42,1,0,train,query)


def test_outer_labels_do_not_select_new_settings(small):
    c,b,a=small;root,scope=study.guard(c,b);rows,first=study.evaluate_fold(b,a,c,scope,root,42,1)
    choices={p.name:p.read_bytes() for p in (root/'review').glob('selection*.json')}
    b.cache.labels[rows]=(b.cache.labels[rows]+1)%6
    # The actual raw-label metadata is immutable; query labels are not read by a fitted context.
    _,second=study.evaluate_fold(b,a,c,scope,root,42,1)
    for arm in c['arms']:np.testing.assert_array_equal(first[arm],second[arm])
    assert choices=={p.name:p.read_bytes() for p in (root/'review').glob('selection*.json')}


def test_complete_audit_run_and_independent_report_without_refitting(small,monkeypatch):
    c,b,a=small;gate=study.audit(c);assert gate['teacher_models_restored']==6 and gate['source_ridge_restorations']==10
    study.smoke(c);study.run(c)
    def forbid(*args,**kw):raise AssertionError('No model fitting during learned-feature report')
    monkeypatch.setattr(RidgeClassifier,'fit',forbid);monkeypatch.setattr(residual_context_model,'fit',forbid)
    result=report.summarize(c)
    assert result['all_models_restored']==40 and result['new_models_restored']==30 and result['outer_models_restored']==8
    assert result['feature_contexts_reconstructed']==result['teacher_models_restored']==result['teacher_query_predictions_reproduced']==6
    assert result['encoder_training_steps']==0 and result['all_original_beats_replayed']
    assert result['contrasts']['plus_quantiles_minus_plus_spread']['primary']
    spec=importlib.util.spec_from_file_location('learned_runner','scripts/run_learned_distribution.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    assert runner.verified_ready(c)
    p=Path(c['output_dir'])/'complete/REPORT.md';p.write_text('bad');assert not runner.verified_ready(c)


def test_scope_audit_and_configuration_guards(small):
    c,b,a=small
    with pytest.raises(FileNotFoundError):study.run(c)
    for k,v in [('arms',['baseline']),('ridge_alphas',[0]),('projection_seeds',[42,42]),('quantiles',[.5,.1,.9]),('device','mps')]:
        with pytest.raises(ValueError):study.validate(dict(c,**{k:v}))
    root,scope=study.guard(c,b);(root/'study_manifest.json').write_text('{}')
    with pytest.raises(ValueError):study.guard(c,b)


def test_real_budgets_and_private_embedding_outputs():
    c=yaml.safe_load(Path('configs/learned_distribution_v1.yaml').read_text());study.validate(c)
    assert study.budget(c)==dict(fits=840,new_fits=630,source_ridge_contexts=210,outer_models=120,new_outer_models=90,source_outer_models=30,feature_contexts=120,teacher_models=120)
    assert c['projections']*len(c['quantiles'])==144
    assert 'results/learned_distribution_v1/**/review/' in Path('.gitignore').read_text()
