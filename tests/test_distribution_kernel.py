"""Weighted-intercept oracle, PSD kernels, cache isolation and full nested replay."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from mi_localization import distribution_kernel_model as model,distribution_kernel_source as source,distribution_kernel_study as study,distribution_kernel_report as report
from mi_localization import convolution_distribution_study as previous,convolution_distribution_report as previous_report
from mi_localization.signal_audit import file_hash
from test_convolution_distribution import small as convolution_small


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    with threadpool_limits(limits=1):yield
    torch.set_num_threads(old)


@pytest.fixture
def small(convolution_small,monkeypatch):
    old,bank,assigned=convolution_small;previous.run(old);summary=previous_report.summarize(old)
    root=Path(old['output_dir'])/'complete';scope=json.loads((root/'study_manifest.json').read_text())
    config=yaml.safe_load(Path('configs/distribution_kernel_v1.yaml').read_text());config.update(source_config=str(Path(old['output_dir'])/'source_config.yaml'),output_dir=str(Path(old['output_dir'])/'kernel'),folds_path=old['folds_path'],outer_splits=2,inner_splits=2,projection_seeds=[42],bootstrap_draws=20,expected_source_fingerprint=scope['study_fingerprint'],expected_source_summary_sha256=file_hash(root/'summary.json'))
    Path(config['source_config']).write_text(yaml.safe_dump(old))
    cached=source.CachedBank(bank.ids,bank.labels,assigned,root,old,scope,summary,dict(synthetic_source_summary_sha256=file_hash(root/'summary.json')),len(bank.record_patients),len(bank.records.waves))
    monkeypatch.setattr(study,'load_inputs',lambda c:(cached,assigned,{}))
    return config,cached,assigned


def test_weighted_linear_equivalence_imbalanced_and_high_dimensional():
    rng=np.random.default_rng(11);y=np.repeat(np.arange(6),[4,5,6,7,8,15]);x=rng.normal(size=(len(y),93))+5;q=rng.normal(size=(9,93))+3
    for alpha in [1.,100.]:
        data=model.KernelData(x,y,q,6);fit=model.KernelModel.fit(data,alpha,0)
        original=RidgeClassifier(alpha=alpha,class_weight='balanced',solver='cholesky').fit(x,y)
        np.testing.assert_allclose(fit.scores(data),original.decision_function(q),rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(fit.predict(data),softmax(original.decision_function(q),axis=1),rtol=1e-10,atol=1e-10)
        assert np.allclose(data.r@data.kernel(0)[0],0,atol=1e-11)


def test_nonlinear_kernel_psd_and_weighted_krr_oracle():
    rng=np.random.default_rng(9);x=rng.normal(size=(36,8));y=np.repeat(np.arange(6),6);q=rng.normal(size=(5,8));data=model.KernelData(x,y,q,6)
    for weight in [.25,.75]:
        fit=model.KernelModel.fit(data,1,weight);k,cross,_=data.kernel(weight)
        assert np.linalg.eigvalsh(k).min()>-1e-10
        # Direct nonsymmetric weighted normal equations are an independent oracle.
        dual=np.linalg.solve(k+np.diag(1/data.weights),data.y_centered)
        np.testing.assert_allclose(fit.scores(data),cross@dual+data.y_mean,rtol=1e-10,atol=1e-10)
        assert not np.allclose(fit.predict(data),model.KernelModel.fit(data,1,0).predict(data))


def test_training_only_bandwidth_energy_and_query_batch_invariance():
    rng=np.random.default_rng(21);x=rng.normal(size=(36,8));y=np.repeat(np.arange(6),6);q=rng.normal(size=(4,8))
    a=model.KernelData(x,y,q,6);b=model.KernelData(x,y,q[:1],6);c=model.KernelData(x,y,q*1000,6)
    assert a.state==b.state==c.state
    f=model.KernelModel.fit(a,1,.25)
    np.testing.assert_allclose(f.predict(b),f.predict(a)[:1],atol=1e-12)
    changed=x.copy();changed[0]*=2
    with pytest.raises(ValueError):f.predict(model.KernelData(changed,y,q,6))


def test_constant_features_intercept_fallback_and_invalid_settings():
    x=np.ones((36,9))*17;y=np.repeat(np.arange(6),6);data=model.KernelData(x,y,np.zeros((3,9)),6)
    assert data.bandwidth_squared==data.energy==1
    for weight in [0,.25,.75]:np.testing.assert_allclose(model.KernelModel.fit(data,1,weight).predict(data),np.full((3,6),1/6))
    with pytest.raises(ValueError):model.KernelModel.fit(data,0,0)
    with pytest.raises(ValueError):data.kernel(-.5)
    with pytest.raises(ValueError):model.KernelData(x,y+1,np.zeros((3,9)),6)


def test_cached_membership_feature_tamper_and_equivalence(small):
    config,bank,assigned=small;root,scope=study.guard(config,bank);train,query=np.flatnonzero(assigned!=1),np.flatnonzero(assigned==1)
    values,datas,ctx=study.prepare_context(bank,config,scope,root,42,1,0,train,query)
    data,own=datas['baseline'];candidate=json.loads((bank.root/'review/selection_01_baseline_s42.json').read_text())['selected_candidate']
    p,a=study.classifier_fit(bank,config,scope,root,42,1,0,'baseline','linear',candidate,train,query,values,data,own,ctx)
    study.classifier_fit(bank,config,scope,root,42,1,0,'baseline','linear',candidate,train,query,values,data,own,ctx,True)
    with pytest.raises(ValueError):source.context(bank,42,1,0,query,train)
    path=next((root/'review/fits').glob('*.joblib'));path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError):study.classifier_fit(bank,config,scope,root,42,1,0,'baseline','linear',candidate,train,query,values,data,own,ctx,True)
    ctx.with_suffix('.npz').write_bytes(ctx.with_suffix('.npz').read_bytes()+b'corrupt')
    with pytest.raises(ValueError):source.context(bank,42,1,0,train,query)


def test_outer_labels_cannot_change_any_choice_or_shared_representation(small):
    config,bank,assigned=small;root,scope=study.guard(config,bank)
    rows,a=study.evaluate_fold(bank,assigned,config,scope,root,42,1)
    choices={p.name:p.read_bytes() for p in (root/'review').glob('*selection*.json')}
    bank.labels[rows]=(bank.labels[rows]+1)%6
    _,b=study.evaluate_fold(bank,assigned,config,scope,root,42,1)
    for arm in a:np.testing.assert_array_equal(a[arm],b[arm])
    assert choices=={p.name:p.read_bytes() for p in (root/'review').glob('*selection*.json')}
    selected=json.loads((root/'review/paired_selection_01_s42.json').read_text())['selected_representation']
    for family in ('linear','nonlinear'):np.testing.assert_array_equal(a[f'selected_{family}'],a[f'{selected}__{family}'])


def test_full_gate_run_report_restoration_without_refitting(small,monkeypatch):
    config,bank,assigned=small
    smoke=study.smoke(config);assert smoke['outer_test_patients_used'] is False
    gate=study.preflight(config);assert gate['source_equivalence_checks']==40 and gate['original_all_arm_metrics_reproduced']
    study.run(config)
    def forbid(*args,**kw):raise AssertionError('Report cannot refit a classifier')
    monkeypatch.setattr(model.KernelModel,'fit',forbid);monkeypatch.setattr(RidgeClassifier,'fit',forbid)
    result=report.summarize(config)
    assert result['fits']==result['all_models_restored']==112
    assert result['outer_models_restored']==16
    assert result['kernel_contexts_reconstructed']==result['matched_context_checks']==24
    assert result['source_equivalence_checks']==40 and result['new_waveform_extractions']==0
    assert result['contrasts']['selected_nonlinear_minus_selected_linear']['primary']
    assert len(result['ensemble_metrics'])==10 and sum(result['shared_representation_counts'].values())==2
    spec=importlib.util.spec_from_file_location('kernel_runner','scripts/run_distribution_kernel.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    assert runner.verified_ready(config)
    p=Path(config['output_dir'])/'complete/REPORT.md';p.write_text('bad')
    assert not runner.verified_ready(config)


def test_scope_config_and_missing_equivalence_gate(small):
    config,bank,assigned=small
    with pytest.raises(FileNotFoundError):study.run(config)
    for k,v in [('nonlinear_weights',[0]),('ridge_alphas',[-1]),('representations',['baseline']),('bandwidth_multiplier',0),('projection_seeds',[42,42])]:
        with pytest.raises(ValueError):study.validate(dict(config,**{k:v}))
    root,scope=study.guard(config,bank);(root/'study_manifest.json').write_text('{}')
    with pytest.raises(ValueError):study.guard(config,bank)


def test_source_chain_rejects_unverified_scope_before_loading_data(small):
    config,bank,assigned=small
    # Public fingerprint/summary gate cannot be bypassed by naming a folder complete.
    bad=dict(config,expected_source_fingerprint='bad')
    with pytest.raises(ValueError,match='pinned independently'):source.validate_chain(bad)
    bad=dict(config,expected_source_summary_sha256='bad')
    with pytest.raises(ValueError,match='pinned independently'):source.validate_chain(bad)


def test_real_budget_compact_model_and_ignored_private_outputs():
    c=yaml.safe_load(Path('configs/distribution_kernel_v1.yaml').read_text());study.validate(c)
    assert study.budget(c)==dict(fits=2400,linear_fits=840,nonlinear_fits=1560,outer_models=240,feature_contexts=120,kernel_contexts=480,matched_checks=480,source_equivalence_checks=840,paired_representation_selections=30)
    assert 'results/distribution_kernel_v1/**/review/' in Path('.gitignore').read_text()
    rng=np.random.default_rng(3);data=model.KernelData(rng.normal(size=(36,20)),np.repeat(np.arange(6),6),rng.normal(size=(3,20)),6);fit=model.KernelModel.fit(data,1,.25)
    assert fit.dual.shape==(36,6) and not hasattr(fit,'training_features')


def test_native_text_label_cache_preserves_configured_class_order():
    classes=['AMI','ALMI','ASMI','IMI','ILMI','HC']
    text=np.array(['HC','AMI','ILMI','ASMI']);expected=np.array([5,0,4,2],dtype=np.int64)
    np.testing.assert_array_equal(source.encode_labels(text,classes),expected)
    np.testing.assert_array_equal(source.encode_labels(expected,classes),expected)
    with pytest.raises(ValueError):source.encode_labels(['unknown'],classes)
    with pytest.raises(ValueError):source.encode_labels([.5],classes)
    with pytest.raises(ValueError):source.encode_labels([6],classes)
