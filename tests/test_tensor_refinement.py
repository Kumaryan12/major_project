import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.fft import dct

from mi_localization import tensor_refinement_bank as bank, tensor_refinement_study as study, tensor_refinement_report as report
from mi_localization import tensor_feature_screen as original
from test_tensor_features import fixture as source_fixture


def fixture(tmp_path):
    source = source_fixture(tmp_path)
    path = tmp_path/"source.yaml"
    path.write_text(yaml.safe_dump(source,sort_keys=False))
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/tensor_refinement.yaml").read_text())
    cfg.update(source_config=str(path),refined_bank_dir=str(tmp_path/"refined"),output_dir=str(tmp_path/"results_refined"),
        global_ranks=[8,16],wavelet_ranks=[2,4],chunk_size=2,tree_depths=[1],tree_estimators=10,bootstrap_draws=100)
    return cfg,source


def test_signed_core_shapes_names_amplitude_and_global_fidelity(tmp_path):
    cfg,source = fixture(tmp_path)
    x = np.random.default_rng(42).normal(size=(4,5,3,65))
    features,baseline,diagnostics = bank.extract(x,cfg,source)
    names = bank.feature_names(cfg,source)
    np.testing.assert_array_equal(baseline,dct(x,type=2,norm="ortho",axis=-1)[...,:4].reshape(4,-1))
    for family,value in features.items():
        assert value.shape==(4,len(names[family]))
    scaled,_,_ = bank.extract(2*x,cfg,source)
    for family,value in features.items():
        np.testing.assert_allclose(scaled[family],2*value,atol=1e-12)
    for view in bank.original.VIEWS:
        def relative(method):
            row = diagnostics[method][view]
            return row["relative_squared_error_sum"]/row["active_lead_beats"]
        assert relative("global16")<=relative("global8")<=relative("global4")
    cols = [names["all_dct8"].index(f"denoised/{lead}/dct_{k:02d}") for lead in ("X","Y","Z") for k in range(8)]
    np.testing.assert_array_equal(features["denoised_dct8"],features["all_dct8"][:,cols])
    values,_,diag = bank.extract(np.zeros_like(x),cfg,source)
    assert all(not value.any() for value in values.values())
    assert diag["global8"]["raw"]["active_lead_beats"]==0
    with pytest.raises(ValueError,match="rank"):
        bank.extract(x,dict(cfg,wavelet_ranks=[100]),source)


def test_refined_bank_exact_original_control_and_cache_guards(tmp_path):
    cfg,source = fixture(tmp_path)
    root = bank.prepare_bank(cfg)
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["original_dct_exact_match"] and manifest["n_patients"]==27 and manifest["n_records"]==27 and manifest["n_beats"]==81
    assert bank.prepare_bank(cfg)==root
    with pytest.raises(ValueError,match="construction changed"):
        bank.prepare_bank(dict(cfg,global_ranks=[7,16]))
    (root/"feature_names.json").write_text("{}")
    with pytest.raises(ValueError,match="content changed"):
        bank.prepare_bank(cfg)


def test_new_linear_trainer_matches_original_probe_exactly(tmp_path):
    cfg,source = fixture(tmp_path)
    source,features,y,ids,names,folds,_ = study.load_inputs(cfg)
    train,valid = np.flatnonzero(folds.fold.to_numpy()!=1),np.flatnonzero(folds.fold.to_numpy()==1)
    previous = original.candidates(source)[1]
    candidate = study.candidates(cfg,source,"linear")[1]
    p,audit = original.fit_probe(features[previous["family"]],y,ids,train,valid,previous,source)
    model,new,expected = study.fit_classifier(features[previous["family"]],y,ids,train,valid,candidate,cfg,source,"linear",1043)
    np.testing.assert_array_equal(p,new)
    assert expected==audit
    assert all(expected[key]==value for key,value in model["selector"].audit().items())


def test_nested_final_models_reuse_inner_probes_separate_patients_and_resume(tmp_path,monkeypatch):
    cfg,source = fixture(tmp_path)
    old_summary = original.run_screen(source)
    assert old_summary["probe_fit_count"]==66
    inputs = study.load_inputs(cfg)
    lookup = inputs[5].set_index("patient_id").fold.to_dict()
    calls = []
    real = study.fit_classifier
    def tracked(X,y,ids,train,evaluate,candidate,config,source,learner,seed):
        assert set(ids[train]).isdisjoint(ids[evaluate])
        fold = (seed-config["seed"])//1000 if seed>=1000 else seed-config["seed"]
        assert all(lookup[str(patient)]!=fold for patient in ids[train])
        if seed>=1000:
            assert all(lookup[str(patient)]!=fold for patient in ids[evaluate])
        else:
            assert all(lookup[str(patient)]==fold for patient in ids[evaluate])
        calls.append((learner,len(train)))
        return real(X,y,ids,train,evaluate,candidate,config,source,learner,seed)
    monkeypatch.setattr(study,"fit_classifier",tracked)
    root = study.run_study(cfg)
    summary = report.summarize_study(cfg)
    assert summary["total_fit_slots"]==294 and summary["reused_legacy_inner_fits"]==66
    assert summary["new_inner_fits"]==210 and summary["outer_refits"]==18
    assert len(calls)==228 and len(summary["metrics"])==6 and len(summary["contrasts"])==6
    monkeypatch.setattr(study,"fit_classifier",lambda *a,**k:pytest.fail("Completed fits must resume"))
    assert study.run_study(cfg)==root
    assert report.summarize_study(cfg)==summary
    with pytest.raises(ValueError,match="changed"):
        study.run_study(dict(cfg,tree_estimators=11))
    path = root/"fold_01_linear_refined.json"
    audit = json.loads(path.read_text())
    audit["selected_indices"][0]+=1
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="outer selector/selection changed"):
        report.summarize_study(cfg)
