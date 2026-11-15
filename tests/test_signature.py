import copy
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.naive_bayes import GaussianNB

from mi_localization import signature_tensor as tensor, signature_bank as bank
from mi_localization import signature_study as study, signature_report as report
from test_deviation import fixture as model_fixture


def test_exact_straight_line_signature_and_interpolation_invariance():
    rng = np.random.default_rng(42)
    endpoints = rng.normal(size=(5, 2, 4))
    actual = tensor.signature(endpoints)
    expected = tensor.endpoint_tensors(endpoints[:, 1]-endpoints[:, 0])
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-13, rtol=1e-13)
    fraction = np.linspace(0, 1, 23)[None, :, None]
    interpolated = endpoints[:, :1]+fraction*(endpoints[:, 1:]-endpoints[:, :1])
    for a, b in zip(actual, tensor.signature(interpolated), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_chen_shuffle_translation_and_axis_amplitude_laws():
    rng = np.random.default_rng(101)
    path = rng.normal(size=(3, 21, 4)).cumsum(1)
    whole = tensor.signature(path)
    combined = tensor.concatenate(tensor.signature(path[:, :9]), tensor.signature(path[:, 8:]))
    translated = tensor.signature(path+np.array([10., 3., -7., 20.]))
    scaled = tensor.signature(path*np.array([1., -2., 1., 1.]))
    for degree, (a, b, c, changed) in enumerate(zip(whole, combined, translated, scaled, strict=True), 1):
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-11)
        np.testing.assert_allclose(a, c, atol=1e-10, rtol=1e-11)
        for word in product(range(4), repeat=degree):
            np.testing.assert_allclose(changed[(slice(None), *word)], a[(slice(None), *word)]*(-2)**word.count(1), atol=1e-10, rtol=1e-11)
    assert max(tensor.mathematical_audit(path).values()) < 1e-10
    with pytest.raises(ValueError, match="finite"):
        tensor.signature(np.full((1, 4, 4), np.nan))


def test_signed_area_and_endpoint_control_distinguish_loop_order():
    # Closed square: X then Y has positive unit area; reversal flips it.
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 0.]])
    waves = np.stack([xyz.T, xyz[::-1].T])
    config = dict(sampling_rate=1000, r_index=0, windows={"whole": [0, 5]})
    value = tensor.extract(waves, config)
    schema = tensor.names(config)
    index = schema["signed_area"].index("whole/area/X_Y")
    np.testing.assert_allclose(value["signed_area"][:, index], [1., -1.], atol=1e-13)
    np.testing.assert_allclose(value["endpoint_polynomial"][0], value["endpoint_polynomial"][1], atol=1e-13)
    assert not np.allclose(value["signature3"][0], value["signature3"][1])
    shifted = tensor.extract(waves+2., config)
    for family in tensor.FAMILIES:
        # All signatures/area increments stay the same; physical mean changes.
        np.testing.assert_allclose(value[family][:, :-3], shifted[family][:, :-3], atol=1e-12)
        np.testing.assert_allclose(shifted[family][:, -3:]-value[family][:, -3:], 2.)


def test_clock_only_exclusion_is_structural_and_keeps_mixed_interactions():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/signature_vcg.yaml").read_text())
    raw_names = tensor.names(cfg)
    expected = dict(signed_area=120, endpoint_polynomial=840, signature2=210, signature3=840)
    for family, names in raw_names.items():
        schema = [f"record_{summary}/{name}" for summary in ("median", "iqr") for name in names]
        mask = study.structural_mask(schema)
        assert mask.sum() == expected[family]
        selected = [name for name, keep in zip(schema, mask, strict=True) if keep]
        assert "record_median/whole/physical_mean/X" in selected
        if family != "signed_area":
            assert "record_median/whole/level2/time_X" in selected
        assert "record_median/whole/level1/time" not in selected


def test_label_free_bank_dimensions_aggregation_resume_and_corruption(tmp_path, monkeypatch):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/signature_vcg.yaml").read_text())
    cfg.update(feature_bank_dir=str(tmp_path/"bank"), output_dir=str(tmp_path/"results"))
    ids = np.array([f"patient{i:03d}" for i in range(12)])
    y = np.repeat(np.arange(6), 2)
    groups = np.repeat(ids, 4)
    labels = np.repeat(y, 4)
    t = np.linspace(-.25, .4, 651)
    rng = np.random.default_rng(8)
    waves = (rng.normal(0, .02, (48, 3, 651))+np.sin(t[None, None]*12)).astype(np.float32)
    raw = waves+rng.normal(0, .002, waves.shape).astype(np.float32)
    records = np.repeat([f"record{i:03d}" for i in range(24)], 2)
    index = np.repeat(np.arange(24), 2)
    templates = np.stack([np.median(waves[index == i], axis=0) for i in range(24)])
    beats = tmp_path/"beats.csv"
    classes = ["AMI", "ALMI", "ASMI", "IMI", "ILMI", "HC"]
    pd.DataFrame(dict(record=records, patient_id=groups, label=np.asarray(classes)[labels])).to_csv(beats, index=False)
    source = dict(classes=classes, beats_path=str(beats))
    source_path = tmp_path/"source.yaml";source_path.write_text(yaml.safe_dump(source))
    cfg["source_config"] = str(source_path)
    old = tmp_path/"original";old.mkdir()
    original = rng.normal(size=(len(ids), 480))
    np.savez_compressed(old/"patient_features.npz", patient_id=ids, y=y, morphology=original)
    (old/"feature_names.json").write_text(json.dumps({"morphology": [f"f{i}" for i in range(480)]}))
    (old/"manifest.json").write_text(json.dumps({"fixture": True}))
    monkeypatch.setattr(bank.morphology, "prepare_bank", lambda source: old)
    monkeypatch.setattr(bank.multiview_bank, "prepare_bank", lambda source: (waves, raw, templates, index, labels, groups, {"fixture": True}))
    root = bank.prepare_bank(cfg)
    with np.load(root/"patient_features.npz") as result:
        np.testing.assert_array_equal(result["morphology"], original)
        assert result["signed_area"].shape == (12, 130)
        assert result["signature2"].shape == (12, 230)
        assert result["signature3"].shape == result["endpoint_polynomial"].shape == (12, 870)
        features = tensor.extract(waves[:4], cfg)
        summary = np.median(np.stack([bank.morphology.robust_record_summary(features["signature3"][:2]),
                                     bank.morphology.robust_record_summary(features["signature3"][2:])]), axis=0)
        np.testing.assert_allclose(result["signature3"][0], summary, atol=1e-14)
    monkeypatch.setattr(tensor, "extract", lambda *a, **k: pytest.fail("Completed bank must resume without extraction"))
    assert bank.prepare_bank(cfg) == root
    with pytest.raises(ValueError, match="construction changed"):
        bank.prepare_bank(dict(cfg, windows=dict(cfg["windows"], whole=[0, 650])))
    path = root/"patient_features.npz";path.write_bytes(path.read_bytes()+b"corruption")
    with pytest.raises(ValueError, match="content changed"):
        bank.prepare_bank(cfg)


def test_training_selector_ignores_heldout_features(tmp_path):
    old, cfg = model_fixture(tmp_path)
    cfg.update(sampling_seeds=[42, 123, 2026], arms=study.ARMS)
    data = dict(old, features={arm: old["X"].copy() for arm in study.ARMS}, names={arm: old["names"] for arm in study.ARMS})
    train, query = np.flatnonzero(data["assigned"] != 1), np.flatnonzero(data["assigned"] == 1)
    a, _ = study.prepare(data, cfg, tmp_path, "signature3", train, query, 6, 42, 1, 0, {})
    altered = copy.deepcopy(data);altered["features"]["signature3"][query] += 10000;altered["y"][query] = 0
    b, _ = study.prepare(altered, cfg, tmp_path, "signature3", train, query, 6, 42, 1, 0, {})
    assert a.audit() == b.audit()
    with pytest.raises(ValueError, match="overlap"):
        study.prepare(data, cfg, tmp_path, "signature3", train, train[:2], 6, 42, 1, 0, {})


def test_full_nested_signature_study_report_resume_and_tamper(tmp_path, monkeypatch):
    old, cfg = model_fixture(tmp_path)
    cfg.update(sampling_seeds=[42, 123, 2026], arms=study.ARMS)
    rng = np.random.default_rng(101)
    features = {"morphology": old["X"]}
    for arm in tensor.FAMILIES:
        features[arm] = np.concatenate([old["X"], rng.normal(size=(len(old["y"]), 8))], axis=1)
    data = dict(old, features=features, names={arm: [f"f{i}" for i in range(X.shape[1])] for arm, X in features.items()})
    monkeypatch.setattr(study, "load_inputs", lambda config: data)
    contexts = []
    real_prepare = study.prepare
    def checked_prepare(data, config, root, arm, train, query, k, sampling_seed, fold, inner, memo, verify=False):
        assert np.all(data["assigned"][train] != fold)
        assert np.all(data["assigned"][query] != fold) if inner else np.all(data["assigned"][query] == fold)
        return real_prepare(data, config, root, arm, train, query, k, sampling_seed, fold, inner, memo, verify)
    monkeypatch.setattr(study, "prepare", checked_prepare)
    def actual(model, seed, config):
        return {"architecture": "TabPFN-v2", "package_version": config["package_version"], "n_estimators": 8,
                "fitted_ensemble_members": 8, "classes": model.classes_.tolist(), "n_features_in": model.n_features_in_,
                "backend_device": "cpu", "random_state": seed, "balance_probabilities": False,
                "softmax_temperature": .9, "fit_mode": "fit_preprocessors", "n_preprocessing_jobs": 1}
    def fake_fit(config, seed, X, y, query, state=None):
        contexts.append(seed)
        model = GaussianNB().fit(X, y)
        if state is not None:
            joblib.dump((model, seed), state)
        return model.predict_proba(query), actual(model, seed, config)
    def fake_restore(config, state, query):
        model, seed = joblib.load(state)
        return model.predict_proba(query), actual(model, seed, config)
    monkeypatch.setattr(study.backend, "run_fit", fake_fit)
    monkeypatch.setattr(study.backend, "run_predict", fake_restore)
    root = study.run_study(cfg)
    summary = report.summarize_study(cfg)
    assert len(contexts) == summary["total_contexts"] == 150 and summary["outer_contexts"] == 30
    assert summary["all_outer_contexts_restored"] and len(summary["contrasts"]) == 7
    monkeypatch.setattr(study.backend, "run_fit", lambda *a, **k: pytest.fail("A complete study must resume"))
    assert study.run_study(cfg) == root and report.summarize_study(cfg) == summary
    with pytest.raises(ValueError, match="changed"):
        study.run_study(dict(cfg, seed=123))
    path = next(root.glob("outer_*signature3*.json"))
    audit = json.loads(path.read_text());audit["selector_seed"] += 1;path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="changed"):
        report.summarize_study(cfg)
