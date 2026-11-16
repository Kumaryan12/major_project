"""Matched-base, training isolation, stream resume and nested restoration oracles."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from threadpoolctl import threadpool_limits
from sklearn.linear_model import RidgeClassifier
from mi_localization import convolution_distribution_features as features, convolution_distribution_source as source
from mi_localization import convolution_distribution_study as study, convolution_distribution_report as report
from mi_localization import rocket_study, rocket_report, rocket_bank
from mi_localization.signal_audit import file_hash


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    with threadpool_limits(limits=1):yield
    torch.set_num_threads(old)


@pytest.fixture
def small(tmp_path,monkeypatch):
    original=yaml.safe_load(Path('configs/rocket_vcg_v1.yaml').read_text())
    original.update(output_dir=str(tmp_path/'source'),folds_path=str(tmp_path/'folds.csv'),outer_splits=2,inner_splits=2,
                    transform_seeds=[42],dct_coefficients=8,multirocket_dilations=2,hydra_groups=2,hydra_kernels=2,chunk_size=8,bootstrap_draws=20)
    ids=np.array([f'p{i:02}' for i in range(24)]);labels=np.repeat(np.arange(6),4);owners=np.repeat(ids,2)
    rng=np.random.default_rng(79);lengths=rng.integers(3,6,len(owners));offsets=np.r_[0,lengths.cumsum()]
    waves=rng.normal(size=(offsets[-1],3,33)).astype(np.float32)
    waves[:,0]+=np.repeat(np.repeat(labels,2),lengths)[:,None]/3
    templates=np.stack([np.median(waves[offsets[i]:offsets[i+1]],axis=0) for i in range(len(owners))])
    template_bank=rocket_bank.TemplateBank(templates,owners,ids,labels,{'synthetic':True})
    assigned=np.tile([1,2,1,2],6);Path(original['folds_path']).write_text('synthetic frozen folds\n')
    monkeypatch.setattr(rocket_study,'load_inputs',lambda _:(template_bank,assigned))
    rocket_study.run(original);rocket_report.summarize(original)
    config=yaml.safe_load(Path('configs/convolution_distribution_v1.yaml').read_text())
    config.update(source_config=str(tmp_path/'source.yaml'),output_dir=str(tmp_path/'new'),folds_path=original['folds_path'],
                  outer_splits=2,inner_splits=2,projection_seeds=[42],projections=3,quantiles=[.1,.5,.9],bootstrap_draws=20)
    Path(config['source_config']).write_text(yaml.safe_dump(original))
    records=SimpleNamespace(ids=ids,labels=labels,record_patients=owners,waves=waves,
                            rows=lambda r:np.arange(offsets[r],offsets[r+1]),
                            records_for=lambda patients:np.flatnonzero(np.isin(owners,patients)))
    scope=rocket_study.make_scope(original,template_bank)
    bank=source.SourceBank(records,templates,original,scope,Path(original['output_dir'])/'complete',
                           dict(source_scope=scope,protected_files=source.protected_files(original),source_config_sha256=file_hash(config['source_config'])))
    monkeypatch.setattr(study,'load_inputs',lambda config,verify=False:(bank,assigned))
    source.verify_baseline(bank,assigned,config)
    return config,bank,assigned


def test_projection_residual_exact_oracle():
    config=dict(projections=2,projection_seed_offset=8123)
    matrix=features.projection(config,4,42)
    np.testing.assert_allclose(np.linalg.norm(matrix,axis=0),1)
    np.testing.assert_array_equal(matrix,features.projection(config,4,42))
    assert not np.array_equal(matrix,features.projection(config,4,123))
    x=np.arange(12,dtype=float).reshape(3,4);median=x[1];scale=np.arange(1,5)
    transform=SimpleNamespace(transform=lambda waves:{'combined':waves})
    np.testing.assert_allclose(features.record_projection(transform,x,median,scale,matrix),((x-median)/scale)@matrix)


def test_gain_equal_patient_record_weights_and_query_exclusion():
    z=[np.array([[1.],[1.]]),np.array([[3.]]),np.array([[5.]]),np.array([[100000.]])]
    owners=np.array(['a','a','b','query'])
    assert features.fit_gain(z,owners,['a','b'],1e-8)==pytest.approx(np.sqrt(((1+9)/2+25)/2))
    z[-1]*=1000
    assert features.fit_gain(z,owners,['a','b'],1e-8)==pytest.approx(np.sqrt(15))
    assert features.fit_gain([np.zeros((2,1))],['a'],['a'],1e-8)==1
    with pytest.raises(ValueError):features.fit_gain(z,owners,['a','a'],1e-8)


def test_quantile_shape_full_base_and_equal_record_pooling():
    config=dict(projections=1,quantiles=[.1,.5,.9]);base=np.array([[9.,8.],[7.,6.]])
    a=np.array([-np.sqrt(2),0,0,np.sqrt(2)])[:,None];b=np.array([-1.,-1.,1.,1.])[:,None]
    v=features.pool([a,b],['a','b'],['a','b'],1,config,base)
    np.testing.assert_array_equal(v['baseline'],base)
    for arm in features.ARMS[1:]:np.testing.assert_array_equal(v[arm][:,:2],base)
    np.testing.assert_allclose(v['plus_mean'][0,2:],v['plus_mean'][1,2:])
    np.testing.assert_allclose(v['plus_spread'][0,2:],v['plus_spread'][1,2:])
    assert not np.allclose(v['plus_quantiles'][0,2:],v['plus_quantiles'][1,2:])
    pooled=features.pool([a,b,b],['a','a','b'],['a','b'],1,config,base)
    np.testing.assert_allclose(pooled['plus_quantiles'][0,2:],v['plus_quantiles'][:,2:].mean(0))


def test_context_resume_independent_beat_replay_and_tamper(small):
    config,bank,assigned=small;scope=study.make_scope(config,bank);root=Path(config['output_dir'])/'complete'
    train,query=np.flatnonzero(assigned!=1),np.flatnonzero(assigned==1)
    values,ctx=study.feature_context(bank,config,scope,root,42,1,0,train,query)
    study.feature_context(bank,config,scope,root,42,1,0,train,query,True)
    paths=list((root/'review/records').rglob('*.json'));assert len(paths)==48
    state=json.loads(ctx.read_text())['state'];assert state['original_beats_covered']==len(bank.records.waves)
    # A context interrupted before its final audit resumes completed per-record work.
    ctx.unlink();ctx.with_suffix('.npz').unlink();ctx.with_suffix('.joblib').unlink()
    first={p:p.read_bytes() for p in paths}
    restored,_=study.feature_context(bank,config,scope,root,42,1,0,train,query)
    assert first=={p:p.read_bytes() for p in paths}
    for arm in config['arms']:np.testing.assert_array_equal(values[arm],restored[arm])
    bank.records.waves[0,0,0]+=.5
    with pytest.raises(ValueError,match='replay'):study.feature_context(bank,config,scope,root,42,1,0,train,query,True)
    bank.records.waves[0,0,0]-=.5
    p=paths[0].with_suffix('.npz');p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):study.feature_context(bank,config,scope,root,42,1,0,train,query)


def test_exact_source_baseline_and_membership_guards(small):
    config,bank,assigned=small
    receipt=source.verify_baseline(bank,assigned,config,write=False)
    assert receipt['source_contexts_reconstructed']==6 and receipt['source_models_restored']==10
    assert source.protected_files(bank.source_config)==bank.provenance['protected_files']
    train,query=np.flatnonzero(assigned!=1),np.flatnonzero(assigned==1)
    with pytest.raises(ValueError):source.context(bank,42,1,0,query,train)
    path=bank.source_root/'context_01_s42_i0.joblib';path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError):source.context(bank,42,1,0,train,query)


def test_training_gain_unchanged_by_query_waveforms(small):
    config,bank,assigned=small;scope=study.make_scope(config,bank)
    train,query=np.flatnonzero(assigned!=1),np.flatnonzero(assigned==1)
    a,ctx=study.feature_context(bank,config,scope,Path(config['output_dir'])/'first',42,1,0,train,query)
    before=json.loads(ctx.read_text())['state']
    for r in bank.records_for(bank.ids[query]):bank.records.waves[bank.rows(r)]*=1000
    b,ctx2=study.feature_context(bank,config,scope,Path(config['output_dir'])/'changed_query',42,1,0,train,query)
    after=json.loads(ctx2.read_text())['state']
    for key in ('residual_rms','projection_sha256','scaler_mean_sha256','scaler_scale_sha256'):assert before[key]==after[key]
    for arm in config['arms']:np.testing.assert_array_equal(a[arm][:len(train)],b[arm][:len(train)])


def test_outer_labels_cannot_choose_model(small):
    config,bank,assigned=small;scope=study.make_scope(config,bank);root=Path(config['output_dir'])/'complete'
    rows,a=study.evaluate_fold(bank,assigned,config,scope,root,42,1)
    decisions={p.name:p.read_bytes() for p in (root/'review').glob('selection*.json')}
    bank.labels[rows]=(bank.labels[rows]+1)%6
    _,b=study.evaluate_fold(bank,assigned,config,scope,root,42,1)
    for arm in config['arms']:np.testing.assert_array_equal(a[arm],b[arm])
    assert decisions=={p.name:p.read_bytes() for p in (root/'review').glob('selection*.json')}
    assert study.select([[.5,.5],[.5,.5]])[0]==0


def test_full_report_restores_all_without_classifier_fit(small,monkeypatch):
    config,bank,assigned=small;study.run(config)
    def forbid(*args,**kwargs):raise AssertionError('Classifier refit forbidden during independent replay')
    monkeypatch.setattr(RidgeClassifier,'fit',forbid)
    result=report.summarize(config)
    assert result['new_models_restored']==30 and result['source_models_restored']==10
    assert result['all_models_restored']==40 and result['outer_models_restored']==8
    assert result['feature_contexts_reconstructed']==result['matched_context_checks']==6
    assert result['baseline_probabilities_exact'] and result['all_original_beats_replayed']
    assert result['contrasts']['plus_quantiles_minus_plus_spread']['primary']
    assert source.protected_files(bank.source_config)==bank.provenance['protected_files']
    spec=importlib.util.spec_from_file_location('convolution_runner','scripts/run_convolution_distribution.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    assert runner.verified_ready(config)
    (Path(config['output_dir'])/'complete/REPORT.md').write_text('bad')
    assert not runner.verified_ready(config)


def test_preflight_scope_and_config_guards(small):
    config,bank,assigned=small
    p=Path(config['output_dir'])/'source_preflight/summary.json';v=json.loads(p.read_text());v['complete']=False;p.write_text(json.dumps(v))
    with pytest.raises(ValueError):study.run(config)
    for k,v in [('projections',0),('quantiles',[.5,.1,.9]),('ridge_alphas',[0]),('projection_seeds',[42,42]),('arms',['baseline'])]:
        with pytest.raises(ValueError):study.validate(dict(config,**{k:v}))


def test_real_budget_and_private_record_cache():
    c=yaml.safe_load(Path('configs/convolution_distribution_v1.yaml').read_text());study.validate(c)
    assert study.budget(c)==dict(feature_contexts=120,fits=840,new_fits=630,source_ridge_contexts=210,outer_models=120,new_outer_models=90,source_outer_models=30)
    assert c['projections']*len(c['quantiles'])==144
    assert 'results/convolution_distribution_v1/**/review/' in Path('.gitignore').read_text()
