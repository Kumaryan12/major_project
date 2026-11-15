import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml
from sklearn.naive_bayes import GaussianNB

from mi_localization import tabpfn_backend as backend, tabpfn_study as study, tabpfn_report as report
from mi_localization import tensor_feature_screen, tensor_refinement_study, tensor_refinement_report, stability_study, stability_report
from test_feature_stability import fixture as stability_fixture


class FakeContext:
    def __init__(self,config,seed,X,y):
        self.estimator=GaussianNB().fit(X,y)
        self.classes_=self.estimator.classes_
        self.n_features_in_=X.shape[1]
        self.seed=seed


def fake_backend_audit(model,config):
    return {"architecture":"TabPFN-v2","package_version":config["package_version"],"n_estimators":config["n_estimators"],
        "fitted_ensemble_members":config["n_estimators"],"classes":model.classes_.tolist(),"n_features_in":model.n_features_in_,
        "backend_device":config["device"],"random_state":model.seed,"balance_probabilities":False,
        "softmax_temperature":config["softmax_temperature"],"fit_mode":"fit_preprocessors","n_preprocessing_jobs":1}


def test_public_checkpoint_integrity_guard_without_network(tmp_path,monkeypatch):
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/tabpfn_v2.yaml").read_text())
    payload=b"test checkpoint, never executed"
    digest=hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(backend,"SHA256",digest);monkeypatch.setattr(backend,"BYTES",len(payload))
    monkeypatch.setattr(backend.importlib.metadata,"version",lambda name:backend.VERSION)
    path=tmp_path/"tabpfn-v2-classifier-v2_default.ckpt"
    path.write_bytes(payload)
    cfg.update(checkpoint_path=str(path),checkpoint_sha256=digest,checkpoint_bytes=len(payload))
    assert backend.prepare_checkpoint(cfg)["checkpoint_sha256"]==digest
    path.write_bytes(b"broken")
    with pytest.raises(ValueError,match="checkpoint changed"):
        backend.prepare_checkpoint(cfg)
    with pytest.raises(ValueError,match="pinned public"):
        backend.prepare_checkpoint(dict(cfg,package_version="different"))


def test_prior_correction_uses_training_counts_and_keeps_all_classes():
    p=np.array([[.75,.25],[.6,.4]])
    result=backend.balance(p,np.array([0,0,0,1]),2)
    np.testing.assert_allclose(result[0],[.5,.5])
    np.testing.assert_allclose(result[1],[1/3,2/3])
    np.testing.assert_allclose(result.sum(1),1)
    np.testing.assert_array_equal(p,np.array([[.75,.25],[.6,.4]]))
    with pytest.raises(ValueError,match="Every TabPFN"):
        backend.balance(p,np.array([0,0]),2)


def test_nested_benchmark_raw_contexts_reuse_resume_and_complete_report(tmp_path,monkeypatch):
    stability_cfg,ref_cfg,source=stability_fixture(tmp_path)
    tensor_feature_screen.run_screen(source)
    tensor_refinement_study.run_study(ref_cfg)
    tensor_refinement_report.summarize_study(ref_cfg)
    stability_study.run_study(stability_cfg)
    stability_report.summarize_study(stability_cfg)
    stability_path=tmp_path/"stability.yaml"
    stability_path.write_text(yaml.safe_dump(stability_cfg,sort_keys=False))
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/tabpfn_v2.yaml").read_text())
    cfg.update(stability_config=str(stability_path),output_dir=str(tmp_path/"pfn"),bootstrap_draws=100)
    monkeypatch.setattr(backend,"prepare_checkpoint",lambda config:{"test_checkpoint":True})
    monkeypatch.setattr(backend,"runtime_info",lambda:{"test_runtime":True})
    monkeypatch.setattr(backend,"settings_audit",fake_backend_audit)
    monkeypatch.setattr(backend,"save",lambda model,path:joblib.dump(model,path))
    monkeypatch.setattr(backend,"load",lambda path,config:joblib.load(path))
    monkeypatch.setattr(backend,"predict",lambda model,X,config:model.estimator.predict_proba(X))
    expected_contexts={}
    calls=[]
    real_prepare=study.prepare
    def tracked_prepare(data,config,candidate,train,evaluate,seed,memo,verify=False):
        result=real_prepare(data,config,candidate,train,evaluate,seed,memo,verify)
        _,source,_,features,y,ids,_,folds,_=data
        lookup=folds.set_index("patient_id").fold.to_dict()
        fold=(seed-config["seed"])//1000 if seed>=1000 else seed-config["seed"]
        assert set(ids[train]).isdisjoint(ids[evaluate]) and all(lookup[str(patient)]!=fold for patient in ids[train])
        if seed>=1000:
            assert all(lookup[str(patient)]!=fold for patient in ids[evaluate])
        X=features[candidate["family"]][train][:,result[0].selected]
        expected_contexts[(seed,hashlib.sha256(X.tobytes()).hexdigest())]=y[train].copy()
        return result
    def fake_fit(config,seed,X,y):
        key=(seed,hashlib.sha256(X.tobytes()).hexdigest())
        assert key in expected_contexts
        np.testing.assert_array_equal(y,expected_contexts[key])
        calls.append(key)
        return FakeContext(config,seed,X,y)
    monkeypatch.setattr(study,"prepare",tracked_prepare)
    monkeypatch.setattr(backend,"fit_context",fake_fit)
    root=study.run_study(cfg)
    summary=report.summarize_study(cfg)
    assert summary["total_fit_context_slots"]==57 and summary["reused_xgboost_inner_fits"]==24
    assert summary["new_tabpfn_inner_context_fits"]==24 and summary["new_tabpfn_outer_context_fits"]==6
    assert summary["new_xgboost_outer_fits"]==3 and summary["matching_reference_outer_predictions"]==3
    assert len(calls)==30 and len(summary["metrics"])==3 and len(summary["contrasts"])==3
    monkeypatch.setattr(backend,"fit_context",lambda *a,**k:pytest.fail("Completed contexts must resume"))
    monkeypatch.setattr(study.prior,"fit_selected",lambda *a,**k:pytest.fail("Completed tree control must resume"))
    assert study.run_study(cfg)==root and report.summarize_study(cfg)==summary
    with pytest.raises(ValueError,match="changed"):
        study.run_study(dict(cfg,n_estimators=9))
    path=root/"fold_01_tabpfn_balanced.json"
    audit=json.loads(path.read_text());audit["training_class_counts"][0]+=1
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="outer selector/context choice changed"):
        report.summarize_study(cfg)
