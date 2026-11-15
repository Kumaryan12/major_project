import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mi_localization import tensor_feature_bank as bank, tensor_feature_screen as screen
from test_multiview import fixture as view_fixture


def fixture(tmp_path):
    source = view_fixture(tmp_path)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/tensor_features.yaml").read_text())
    for key in ("seed","split_seed","paper_config","beats_path","bank_dir","raw_bank_dir","template_bank_dir","output_dir","folds_path","classes","outer_splits","inner_splits"):
        cfg[key] = source[key]
    cfg.update(feature_bank_dir=str(tmp_path/"features"),wavelet="sym2",level=2,dct_coefficients=4,
        windows={"pre":[0,16],"central":[16,40],"post":[40,65]},chunk_size=2,feature_counts=[3,6],linear_c=[.1])
    count = len(np.load(Path(cfg["bank_dir"])/"waveforms.npy",mmap_mode="r"))
    np.save(Path(cfg["bank_dir"])/"statistics.npy",np.random.default_rng(42).normal(size=(count,8)).astype(np.float32))
    return cfg


def test_tensor_is_additive_with_declared_axes_and_feature_schema(tmp_path):
    cfg = fixture(tmp_path)
    x = np.random.default_rng(12).normal(size=(3,5,3,65))
    tensor,error = bank.tensorize(x,cfg)
    assert tensor.shape==(3,5,3,3,65) and error<1e-8
    np.testing.assert_allclose(tensor.sum(-2),x,atol=1e-8)
    values,_ = bank.extract(x,cfg)
    names = bank.feature_names(cfg,192)
    for family,value in values.items():
        assert value.shape==(3,len(names[family]))
    # Check named order against direct calculations; especially band/window axes.
    name = "raw/X/A2/pre/mean"
    np.testing.assert_allclose(values["multiscale"][:,names["multiscale"].index(name)],tensor[:,0,0,0,:16].mean(-1))
    part = tensor[:,0,:3,0,:16]
    cov = np.mean((part[:,0]-part[:,0].mean(-1,keepdims=True))*(part[:,1]-part[:,1].mean(-1,keepdims=True)),axis=-1)
    np.testing.assert_allclose(values["relations"][:,names["relations"].index("raw/A2/pre/cov_XY")],cov)
    amplified,_ = bank.extract(2*x,cfg)
    np.testing.assert_allclose(amplified["morphology"],2*values["morphology"],atol=1e-8)
    relation_names = names["relations"]
    cov_cols = [i for i,name in enumerate(relation_names) if "/cov_" in name]
    corr_cols = [i for i,name in enumerate(relation_names) if "/corr_" in name]
    np.testing.assert_allclose(amplified["relations"][:,cov_cols],4*values["relations"][:,cov_cols],atol=1e-8)
    np.testing.assert_allclose(amplified["relations"][:,corr_cols],values["relations"][:,corr_cols],atol=1e-8)
    zeros,_ = bank.extract(np.zeros_like(x),cfg)
    assert all(np.isfinite(value).all() and not value.any() for value in zeros.values())
    with pytest.raises(ValueError,match="level"):
        bank.tensorize(x,dict(cfg,level=10))


def test_record_first_aggregation_and_bank_integrity(tmp_path):
    cfg = fixture(tmp_path)
    beats = pd.read_csv(cfg["beats_path"])
    beats.loc[2,"record"] = "p00/other"
    beats.to_csv(cfg["beats_path"],index=False)
    bank.multiview_bank.prepare_bank(cfg)
    cfg["output_dir"] = str(tmp_path/"new_audit_output")
    root = bank.prepare_bank(cfg)
    assert bank.prepare_bank(cfg)==root
    stats = np.load(Path(cfg["bank_dir"])/"statistics.npy")
    expected = np.median(np.stack([bank.robust_record_summary(stats[:2]),bank.robust_record_summary(stats[2:3])]),axis=0)
    with np.load(root/"patient_features.npz") as saved:
        np.testing.assert_allclose(saved["statistics"][0],expected)
        assert len(saved["patient_id"])==27
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["n_records"]==28 and manifest["nonfinite_features"]==0
    with pytest.raises(ValueError,match="construction/source changed"):
        bank.prepare_bank(dict(cfg,dct_coefficients=3))
    (root/"feature_names.json").write_text("{}")
    with pytest.raises(ValueError,match="cache changed"):
        bank.prepare_bank(cfg)


def test_selector_discards_training_constants_and_duplicates_without_using_other_rows():
    rng = np.random.default_rng(22)
    y = np.repeat(np.arange(3),8)
    signal = y+rng.normal(0,.05,len(y))
    X = np.column_stack([signal,signal,signal*3,np.ones(len(y)),rng.normal(size=len(y))])
    first = screen.TrainingSelector().fit(X,y,np.ones(len(y)),3,.95)
    assert len(set(first.selected)&{0,1,2})==1 and 3 not in first.selected
    assert len(first.selected)==2 and first.dropped_redundant==2
    # The learned state does not change when a different validation table is transformed.
    audit = copy.deepcopy(first.audit())
    first.transform(rng.normal(100,5,size=(7,5)))
    assert first.audit()==audit
    all_features = screen.TrainingSelector().fit(X,y,np.ones(len(y)),"all",.95)
    assert len(all_features.selected)==4


def test_complete_inner_only_screen_packages_resume_and_tamper_checks(tmp_path,monkeypatch):
    cfg = fixture(tmp_path)
    calls = []
    original = screen.fit_probe
    frozen = pd.read_csv(cfg["folds_path"]).set_index("patient_id").fold.to_dict()
    per_fold = len(screen.candidates(cfg))*cfg["inner_splits"]
    def tracked(X,y,ids,train,valid,*args):
        fold = len(calls)//per_fold+1
        assert set(ids[train]).isdisjoint(ids[valid])
        assert all(frozen[str(patient)]!=fold for patient in np.concatenate([ids[train],ids[valid]]))
        calls.append(1)
        return original(X,y,ids,train,valid,*args)
    monkeypatch.setattr(screen,"fit_probe",tracked)
    summary = screen.run_screen(cfg)
    assert summary["outer_test_predictions"]==0 and summary["probe_fit_count"]==66 and len(calls)==66
    root = Path(cfg["output_dir"])/"screen"
    assert len(list(root.glob("features_outer_*.npz")))==3
    assert not list(root.glob("patient_predictions*"))
    for fold in range(1,4):
        with np.load(root/f"features_outer_{fold:02d}.npz") as package:
            assert all(frozen[str(patient)]!=fold for patient in package["train_patient_ids"])
    monkeypatch.setattr(screen,"fit_probe",lambda *a,**k:pytest.fail("Completed probes should resume"))
    assert screen.run_screen(cfg)==summary
    with pytest.raises(ValueError,match="settings changed"):
        screen.run_screen(dict(cfg,correlation_threshold=.9))
    path = root/"inner_01_00_1.json"
    audit = json.loads(path.read_text())
    audit["selected_indices"][0] += 1
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="cached training selector changed"):
        screen.run_screen(cfg)
