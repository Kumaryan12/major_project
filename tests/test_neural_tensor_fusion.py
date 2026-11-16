import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import f1_score

from mi_localization import neural_tensor_fusion as fusion,neural_tensor_fusion_report as report
from mi_localization import residual_context_study as ns,residual_context_report as nr
from mi_localization import tabpfn_study as ts
from mi_localization.benchmark import inner_patient_splits,metric_bundle
from mi_localization.multiview_bank import array_hash
from mi_localization.neural import _hash,_save_json
from mi_localization.signal_audit import file_hash
from test_residual_context import setup
from test_tabpfn import FakeContext,fake_backend_audit


class Selector:
    def __init__(self,k,cache):self.selected=np.arange(k);self.cache=cache
    def audit(self):return {"n_selected":len(self.selected),"bootstrap_cache_key":"fixture","bootstrap_cache_sha256":file_hash(self.cache)}


def test_probability_blend_endpoints_and_no_mutation():
    a=np.array([[.8,.2],[.1,.9]]);b=np.array([[.2,.8],[.7,.3]])
    np.testing.assert_array_equal(fusion.blend(a,b,1),a)
    np.testing.assert_array_equal(fusion.blend(a,b,0),b)
    np.testing.assert_allclose(fusion.blend(a,b,.5),[[.5,.5],[.4,.6]])
    np.testing.assert_array_equal(a,[[.8,.2],[.1,.9]])
    for alpha in [-.1,1.1,np.nan]:
        with pytest.raises(ValueError):fusion.blend(a,b,alpha)
    with pytest.raises(ValueError):fusion.blend(a,b*2,.5)


def test_weight_selection_has_only_inner_inputs_and_stable_neural_ties():
    y=[np.array([0,1,0,1]),np.array([0,1,0,1])]
    good=np.array([[.9,.1],[.1,.9],[.9,.1],[.1,.9]])
    bad=good[:,::-1]
    alpha,rows=fusion.choose_weight([good,good],[bad,bad],y,['A','B'],[1.,.75,.5,.25,0.])
    assert alpha==1. and rows[0]['mean_inner_macro_f1']==1.
    alpha,_=fusion.choose_weight([bad,bad],[good,good],y,['A','B'],[1.,.75,.5,.25,0.])
    assert alpha in [.25,0.]
    assert fusion.choose_weight([good,good],[good,good],y,['A','B'],[1.,.75,.5,.25,0.])[0]==1.


def test_partition_rejects_outer_query_in_inner_training_or_validation():
    ids=np.array(['a','b','c','d'])
    fusion.check_partition(np.array([0,1]),np.array([2]),np.array([3]),ids)
    for a,b in [([0,3],[2]),([0,1],[3]),([0,1],[1])]:
        with pytest.raises(ValueError,match='overlap'):
            fusion.check_partition(np.array(a),np.array(b),np.array([3]),ids)


def fixture(tmp_path,monkeypatch):
    bank,assigned,nc=setup(tmp_path);nc['training_seeds']=[42,123,2026]
    monkeypatch.setattr(ns,'load_inputs',lambda config:(bank,assigned))
    nroot=ns.run_study(nc);nr.summarize_study(nc)
    npath=tmp_path/'neural.yaml';npath.write_text(yaml.safe_dump(nc))
    ec=dict(source_config=str(npath),output_dir=str(tmp_path/'ensemble'),operation='uniform_probability_mean',bootstrap_draws=100,bootstrap_seed=42)
    epath=tmp_path/'ensemble.yaml';epath.write_text(yaml.safe_dump(ec))
    eroot=Path(ec['output_dir'])/'complete';eroot.mkdir(parents=True)
    frames=[pd.read_csv(nroot/f'patient_predictions_ordered_context_aux_s{s}.csv') for s in nc['training_seeds']]
    columns=[f'probability_{c}' for c in nc['classes']];p=np.mean([f[columns].to_numpy() for f in frames],axis=0)
    f=frames[0].copy();f[columns]=p;f['prediction']=np.asarray(nc['classes'])[p.argmax(1)]
    f.to_csv(eroot/'patient_predictions_ordered_context_aux.csv',index=False)
    em={'config':ec,'fixture':True};_save_json(eroot/'analysis_manifest.json',em)
    nscope=ns.make_scope(nc,bank)
    _save_json(eroot/'summary.json',dict(complete=True,analysis_fingerprint=_hash(em),source_fingerprint=nscope['study_fingerprint'],
        n_patients=len(bank.ids),classes=nc['classes'],training_seeds=nc['training_seeds'],new_training_fits=0,
        metrics={'ordered_context_aux':metric_bundle(bank.labels,p,nc['classes'])}))
    tc=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/tabpfn_v2.yaml').read_text())
    tc.update(output_dir=str(tmp_path/'tensor'),bootstrap_draws=100)
    tpath=tmp_path/'tensor.yaml';tpath.write_text(yaml.safe_dump(tc))
    troot=Path(tc['output_dir'])/'complete';troot.mkdir(parents=True)
    prior=tmp_path/'prior';(prior/'bootstrap_cache').mkdir(parents=True)
    cache=prior/'bootstrap_cache/fixture.npz';cache.write_bytes(b'Only a synthetic selector cache; real selector replay is tested separately.')
    rng=np.random.default_rng(42);X=rng.normal(size=(len(bank.ids),6));X[:,0]+=bank.labels
    features={'morphology':X,'combined':np.c_[X,rng.normal(size=(len(X),2))]}
    folds=pd.read_csv(nc['folds_path'])
    sc={'families':['morphology','combined']};source=dict(classes=nc['classes'],outer_splits=2,inner_splits=2,feature_counts=[2,4])
    provenance={'prior_root':str(prior),'fixture':True}
    data=(sc,source,{},features,bank.labels,bank.ids,{k:[str(i) for i in range(v.shape[1])] for k,v in features.items()},folds,provenance)
    monkeypatch.setattr(ts,'load_inputs',lambda config:data)
    scope=ts.make_scope(tc,provenance);_save_json(troot/'study_manifest.json',scope)
    def prepare(data,config,candidate,train,evaluate,seed,memo,verify=False):
        assert set(bank.ids[train]).isdisjoint(bank.ids[evaluate])
        fold=(seed-42)//1000 if seed>=1000 else seed-42
        assert not np.any(assigned[train]==fold)
        if seed>=1000:assert not np.any(assigned[evaluate]==fold)
        selector=Selector(candidate['k'],cache);values=features[candidate['family']]
        audit=dict(patient_overlap=0,train_membership_sha256=_hash(bank.ids[train].tolist()),
            validation_membership_sha256=_hash(bank.ids[evaluate].tolist()),training_features_sha256=array_hash(values[train]),
            training_labels_sha256=array_hash(bank.labels[train]),selector=selector.audit())
        memo[(seed,candidate['family'])]=True
        return selector,None,audit
    monkeypatch.setattr(ts,'prepare',prepare)
    pooled=np.full((len(bank.ids),6),np.nan)
    for fold in [1,2]:
        train,query=np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
        splits=[(train[a],train[b]) for a,b in inner_patient_splits(bank.labels[train],bank.ids[train],2,42+fold*1000)]
        candidates=ts.pfn_candidates(sc,source);rows=[]
        for cid,candidate in enumerate(candidates):
            scores=[];hashes=[]
            for inner,(a,b) in enumerate(splits,1):
                seed=42+fold*1000+inner;sel,_,audit=prepare(data,tc,candidate,a,b,seed,{})
                values=features[candidate['family']];model=FakeContext(tc,seed,values[a][:,sel.selected],bank.labels[a])
                p=model.estimator.predict_proba(values[b][:,sel.selected]);balanced=ts.backend.balance(p,bank.labels[a],6)
                path=troot/f'inner_{fold:02d}_tabpfn_{cid}_{inner}.json'
                score=float(f1_score(bank.labels[b],p.argmax(1),labels=np.arange(6),average='macro',zero_division=0))
                audit.update(study_fingerprint=scope['study_fingerprint'],outer_fold=fold,inner_fold=inner,candidate_id=cid,candidate=candidate,
                    seed=seed,backend=fake_backend_audit(model,tc),checkpoint_sha256=ts.save_probabilities(path.with_suffix('.npz'),b,tabpfn=p,tabpfn_balanced=balanced),validation_macro_f1={'tabpfn':score})
                _save_json(path,audit);scores.append(score);hashes.append(file_hash(path))
            rows.append(dict(candidate_id=cid,candidate=candidate,mean_inner_macro_f1=float(np.mean(scores)),inner_audits_sha256=hashes))
        selection=troot/f'selection_{fold:02d}_tabpfn.json';_save_json(selection,dict(study_fingerprint=scope['study_fingerprint'],candidates=rows))
        chosen=max(range(len(rows)),key=lambda i:rows[i]['mean_inner_macro_f1']);candidate=candidates[chosen];seed=42+fold
        sel,_,audit=prepare(data,tc,candidate,train,query,seed,{})
        values=features[candidate['family']];context=FakeContext(tc,seed,values[train][:,sel.selected],bank.labels[train])
        p=context.estimator.predict_proba(values[query][:,sel.selected]);pooled[query]=p
        path=troot/f'fold_{fold:02d}_tabpfn.json';state=troot/f'model_{fold:02d}_tabpfn.tabpfn_fit';modelpath=troot/f'model_{fold:02d}_tabpfn.joblib'
        joblib.dump(context,state)
        actual=fake_backend_audit(context,tc)
        joblib.dump(dict(selector=sel,candidate=candidate,backend=actual,arm='tabpfn',fold=fold,study_fingerprint=scope['study_fingerprint']),modelpath)
        audit.update(study_fingerprint=scope['study_fingerprint'],fold=fold,arm='tabpfn',candidate_id=chosen,candidate=candidate,seed=seed,
            selection_sha256=file_hash(selection),backend=actual,model_sha256=file_hash(modelpath),fit_state_sha256=file_hash(state),
            checkpoint_sha256=ts.save_probabilities(path.with_suffix('.npz'),query,probabilities=p),patient_metrics=metric_bundle(bank.labels[query],p,nc['classes']))
        _save_json(path,audit)
    metric=metric_bundle(bank.labels,pooled,nc['classes'])
    _save_json(troot/'metrics_tabpfn.json',dict(complete=True,study_fingerprint=scope['study_fingerprint'],classes=nc['classes'],n_patients=len(bank.ids),patient_metrics=metric))
    _save_json(troot/'summary.json',dict(complete=True,study_fingerprint=scope['study_fingerprint'],n_patients=len(bank.ids)))
    (troot/'REPORT.md').write_text('Synthetic fitted-context fixture, not real TabPFN performance.')
    tf=frames[0].copy();tf[columns]=pooled;tf['prediction']=np.asarray(nc['classes'])[pooled.argmax(1)];tf.to_csv(troot/'patient_predictions_tabpfn.csv',index=False)
    def predict(config,state,query):
        model=joblib.load(state);return model.estimator.predict_proba(query),fake_backend_audit(model,config)
    monkeypatch.setattr(ts.backend,'run_predict',predict)
    monkeypatch.setattr(ts.backend,'run_fit',lambda *a,**k:pytest.fail('Fusion must not fit a tensor context'))
    monkeypatch.setattr(ns.models,'fit',lambda *a,**k:pytest.fail('Fusion must not train a neural model'))
    cfg=dict(neural_ensemble_config=str(epath),neural_arm='ordered_context_aux',tensor_config=str(tpath),tensor_arm='tabpfn',
             output_dir=str(tmp_path/'fusion'),procedures=list(fusion.PROCEDURES),neural_weight_grid=[1.,.75,.5,.25,0.],bootstrap_draws=100,bootstrap_seed=42)
    return cfg,nroot,troot


def test_complete_nested_fusion_restoration_resume_tamper_and_source_immutability(tmp_path,monkeypatch):
    cfg,nroot,troot=fixture(tmp_path,monkeypatch)
    hashes={str(p):file_hash(p) for r in [nroot,troot] for p in r.iterdir() if p.is_file()}
    root=fusion.run_study(cfg);summary=report.summarize(cfg)
    assert summary['total_source_contexts_verified']==36 and summary['new_model_fits']==0
    assert summary['neural_outer_members_restored']==6 and summary['tensor_outer_contexts_restored']==2
    assert summary['frozen_references_reproduced'] and len(summary['metrics'])==4
    assert hashes=={str(p):file_hash(p) for r in [nroot,troot] for p in r.iterdir() if p.is_file()}
    assert fusion.run_study(cfg)==root and report.summarize(cfg)==summary
    checkpoint=root/'fold_01.npz';original=checkpoint.read_bytes();checkpoint.write_bytes(original+b'tamper')
    with pytest.raises(ValueError,match='audit changed'):report.summarize(cfg)
    checkpoint.write_bytes(original)
    with pytest.raises(ValueError,match='namespace'):fusion.run_study(dict(cfg,bootstrap_seed=123))
    data=fusion.load_inputs(cfg);scope=fusion.make_scope(cfg,data)
    class StopAfterSelection(Exception):pass
    observed=[]
    def stop(*args):
        observed.append(json.loads((root/'selection_01.json').read_text())['selected_neural_weight'])
        raise StopAfterSelection
    monkeypatch.setattr(fusion,'neural_outer',stop)
    with pytest.raises(StopAfterSelection):fusion.evaluate_fold(data,cfg,scope,root,1,verify=True)
    altered=dict(data,y=data['y'].copy());query=data['assigned']==1;altered['y'][query]=(altered['y'][query]+1)%6
    with pytest.raises(StopAfterSelection):fusion.evaluate_fold(altered,cfg,scope,root,1,verify=True)
    assert observed[0]==observed[1]
    monkeypatch.undo()


def test_changed_grid_is_rejected_before_cache_use(tmp_path):
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/neural_tensor_fusion_v1.yaml').read_text())
    with pytest.raises(ValueError,match='grid'):
        fusion.load_inputs(dict(cfg,neural_weight_grid=[.5]))


def test_live_monitor_and_runner_use_reused_contexts_not_new_training(tmp_path,monkeypatch):
    from test_chronology_lengths import script
    live=script('live_status.py');runner=script('run_neural_tensor_fusion.py')
    p=tmp_path/'results/neural_tensor_fusion_v1/complete';p.mkdir(parents=True)
    (p/'progress.json').write_text(json.dumps(dict(pid=12,state='verifying_inner_tensor',outer_fold=1)))
    monkeypatch.setattr(live,'process_command',lambda pid:('python -m mi_localization.neural_tensor_fusion' if pid==12 else '', 'checked'))
    row=live.snapshot(tmp_path)['neural_tensor_fusion']
    assert row['phase']=='verifying_inner_tensor' and row['new_model_fits']==0 and row['planned_source_contexts']==250
    assert not runner.verified_ready(tmp_path)
    (p/'REPORT.md').write_text('verified')
    s=dict(complete=True,new_model_fits=0,total_source_contexts_verified=250,neural_outer_members_restored=30,
           tensor_outer_contexts_restored=10,completed_outer_folds=10,frozen_references_reproduced=True)
    _save_json(p/'summary.json',s)
    assert runner.verified_ready(tmp_path)
    assert live.snapshot(tmp_path)['neural_tensor_fusion']['report_present']
