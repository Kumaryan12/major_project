import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mi_localization import feature_stability as fs, stability_study as study, stability_report as report
from mi_localization import tensor_feature_screen as old, tensor_refinement_study as reference, tensor_refinement_report
from test_tensor_refinement import fixture as reference_fixture


def fixture(tmp_path):
    ref_config,source = reference_fixture(tmp_path)
    ref_path = tmp_path/"reference.yaml"
    ref_path.write_text(yaml.safe_dump(ref_config,sort_keys=False))
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/feature_stability.yaml").read_text())
    cfg.update(reference_config=str(ref_path),output_dir=str(tmp_path/"stability"),bootstrap_replicates=4,audit_replicates=3,bootstrap_draws=100)
    return cfg,ref_config,source


def test_semantic_groups_merge_view_only_and_preserve_provenance():
    names = ["record_median/raw/X/dct_00","record_median/denoised/X/dct_00","record_iqr/raw/X/dct_00",
        "record_median/raw/Y/dct_00","record_median/raw/X/dct_01"]
    groups,_ = fs.semantic_groups(names,"morphology")
    assert groups[0]==groups[1] and len(set(groups))==4
    combined = ["morphology/"+name for name in names]+["statistics/record_median/baseline/X/original/mean"]
    g,_ = fs.semantic_groups(combined,"combined")
    assert g[0]==g[1] and len(set(g))==5


def test_draws_keep_class_counts_and_consensus_group_cap_is_explicit():
    y = np.repeat(np.arange(3),8)
    draws = fs.bootstrap_draws(y,16,42,0)
    np.testing.assert_array_equal(draws,fs.bootstrap_draws(y,16,42,0))
    assert not np.array_equal(draws,fs.bootstrap_draws(y,16,42,1))
    assert all(np.array_equal(np.bincount(y[row]),np.bincount(y)) for row in draws)
    assert any(len(np.unique(row))<len(y) for row in draws)
    rng = np.random.default_rng(12)
    X = np.column_stack([y+rng.normal(0,.5,len(y)),y+rng.normal(0,.5,len(y)),rng.normal(size=len(y))])
    names = ["record_median/raw/X/dct_00","record_median/denoised/X/dct_00","record_median/raw/Y/dct_00"]
    payload = {"training":np.tile([0,1,2],(4,1)),"audit":np.tile([0,2,-1],(3,1)),"cache_key":"key","cache_sha256":"hash","draws_sha256":"draws"}
    selectors = {method:fs.ConsensusSelector().fit(X,y,np.ones(len(y)),3,.99,names,"morphology",method,payload) for method in ("original","bootstrap","group_bootstrap")}
    assert len(selectors["original"].selected)==3 and len(selectors["bootstrap"].selected)==3
    grouped = selectors["group_bootstrap"]
    assert len(grouped.selected)==2 and len(set(grouped.groups[grouped.selected]))==2
    before = copy.deepcopy(grouped.audit())
    grouped.transform(rng.normal(100,2,size=(7,3)))
    assert grouped.audit()==before


def test_training_only_cache_replay_corruption_and_original_numeric_control(tmp_path):
    cfg,ref_cfg,source = fixture(tmp_path)
    _,features,y,ids,names,folds,_ = reference.load_inputs(ref_cfg)
    train = np.flatnonzero(folds.fold.to_numpy()!=1)
    valid = np.flatnonzero(folds.fold.to_numpy()==1)
    candidate = [c for c in reference.candidates(ref_cfg,source,"linear") if c["family"]=="morphology"][0]
    X = features["morphology"]
    root = Path(cfg["output_dir"])/"complete"
    original,weights,audit = study.prepare_selector(X,y,ids,train,valid,names["morphology"],candidate,"original",cfg,source,1043,root,{})
    model,p = study.fit_selected(original,X,y,ids,train,valid,weights,candidate,"linear",cfg,source,ref_cfg,1043)
    old_model,old_p,_ = reference.fit_classifier(X,y,ids,train,valid,candidate,ref_cfg,source,"linear",1043)
    np.testing.assert_array_equal(p,old_p)
    other = X.copy();other[valid]+=10000
    other_y = y.copy();other_y[valid]=0
    selector,_,other_audit = study.prepare_selector(other,other_y,ids,train,valid,names["morphology"],candidate,"original",cfg,source,1043,root,{},verify=True)
    assert other_audit==audit and selector.audit()==original.audit()
    path = next((root/"bootstrap_cache").glob("*.npz"))
    with np.load(path) as saved:
        payload = {key:saved[key] for key in saved.files}
    payload["training"][0,0]=-1
    np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match="selection content changed"):
        study.prepare_selector(X,y,ids,train,valid,names["morphology"],candidate,"bootstrap",cfg,source,1043,root,{})


def test_full_nested_stability_models_replay_and_scope_guards(tmp_path,monkeypatch):
    cfg,ref_cfg,source = fixture(tmp_path)
    old.run_screen(source)
    reference.run_study(ref_cfg)
    tensor_refinement_report.summarize_study(ref_cfg)
    inputs = study.load_inputs(cfg)
    lookup = inputs[6].set_index("patient_id").fold.to_dict()
    calls = []
    real = study.fit_selected
    def tracked(selector,X,y,ids,train,evaluate,weights,candidate,learner,config,source,ref_config,seed):
        fold = (seed-config["seed"])//1000 if seed>=1000 else seed-config["seed"]
        assert set(ids[train]).isdisjoint(ids[evaluate])
        assert all(lookup[str(patient)]!=fold for patient in ids[train])
        if seed>=1000:
            assert all(lookup[str(patient)]!=fold for patient in ids[evaluate])
        calls.append(selector.method)
        return real(selector,X,y,ids,train,evaluate,weights,candidate,learner,config,source,ref_config,seed)
    monkeypatch.setattr(study,"fit_selected",tracked)
    root = study.run_study(cfg)
    summary = report.summarize_study(cfg)
    assert summary["total_fit_slots"]==162 and summary["reused_original_inner_fits"]==48
    assert summary["new_inner_fits"]==96 and summary["outer_refits"]==18 and len(calls)==114
    assert summary["bootstrap_caches_fully_replayed"]==18 and summary["resampled_base_selector_fits"]==126
    assert len(summary["metrics"])==6 and len(summary["matched_support_means"])==12
    monkeypatch.setattr(study,"fit_selected",lambda *a,**k:pytest.fail("Completed models must resume"))
    assert study.run_study(cfg)==root and report.summarize_study(cfg)==summary
    with pytest.raises(ValueError,match="changed"):
        study.run_study(dict(cfg,bootstrap_replicates=5))
    path = root/"fold_01_linear_bootstrap.json"
    audit = json.loads(path.read_text())
    audit["selector"]["selected_indices"][0]+=1
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="outer selector/feature choice changed"):
        report.summarize_study(cfg)
