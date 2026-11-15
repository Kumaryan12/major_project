import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

from mi_localization import deviation_tensor as tensor, deviation_study as study, deviation_report as report


def fixture(tmp_path):
    rng = np.random.default_rng(73)
    classes = ["AMI", "ALMI", "ASMI", "IMI", "ILMI", "HC"]
    y = np.repeat(np.arange(6), 24)
    ids = np.array([f"patient{i:03d}" for i in range(len(y))])
    assigned = np.tile(np.tile([1, 2], 12), 6)
    t = np.linspace(0, 1, 651)
    waves = rng.normal(0, .025, (len(y), 3, len(t)))
    waves += np.sin(t[None, None]*np.pi*(2+y[:, None, None]/3))
    X = rng.normal(size=(len(y), 12))+y[:, None]*.1
    source = dict(classes=classes, outer_splits=2, inner_splits=2, wavelet="sym4", level=1,
                  windows={"whole": [0, 651]}, feature_counts=[3, 6], correlation_threshold=.95)
    pfn = dict(package_version="test", n_estimators=8, device="cpu", softmax_temperature=.9)
    data = dict(source=source, pfn=pfn, X=X, y=y, ids=ids, names=[f"f{i}" for i in range(12)],
                assigned=assigned, waves=waves, record_patients=ids.copy(), patient_waves=waves.copy(),
                provenance={"fixture": "independent synthetic patients"})
    config = dict(seed=42, split_seed=42, reference_seeds=[42, 123, 2026], output_dir=str(tmp_path/"study"),
                  arms=study.ARMS.copy(), reference_parts=3, medoids=2, reference_dct=32,
                  feature_counts=[3, 6], bootstrap_replicates=2, audit_replicates=2, bootstrap_draws=100)
    return data, config


def test_exact_medoids_and_reference_patient_participation(tmp_path):
    np.testing.assert_array_equal(tensor.exact_two_medoids(np.array([[0.], [1.], [10.], [11.]])), [0, 2])
    data, cfg = fixture(tmp_path)
    train, test, _ = study.splits(data, cfg, 1)
    members = tensor.reference_members(data["y"], data["ids"], train, test, 42)
    for fit, ref in members:
        assert set(fit).isdisjoint(ref) and set(ref).isdisjoint(test)
        assert set(fit) | set(ref) == set(train)
    assert all(sum(i in fit for fit, _ in members) == 2 for i in train)
    with pytest.raises(ValueError, match="overlap"):
        tensor.check_partition(data["ids"], train, train[:2], test)
    with pytest.raises(ValueError, match="one row"):
        tensor.check_partition(np.array(["duplicate", "duplicate"]), [0], [], [1])


def test_atlas_never_uses_query_or_classifier_waveforms_or_labels(tmp_path):
    data, cfg = fixture(tmp_path)
    train, test, _ = study.splits(data, cfg, 1)
    fit, ref = study.members(data, cfg, "healthy", train, test, 42, 1, 1)[0]
    a, audit = tensor.fit_atlas(data["patient_waves"], data["y"], data["ids"], fit, ref, test, 5, "healthy", 42)
    changed = data["patient_waves"].copy();changed[np.r_[fit, test]] += 1e6
    labels = data["y"].copy();labels[np.r_[fit, test]] = 0
    b, altered = tensor.fit_atlas(changed, labels, data["ids"], fit, ref, test, 5, "healthy", 42)
    np.testing.assert_array_equal(a, b)
    assert audit == altered
    mixed, mixed_audit = tensor.fit_atlas(data["patient_waves"], data["y"], data["ids"], fit, ref, test, 5, "mixed", 42)
    assert mixed_audit["eligible_reference_patients"] == audit["eligible_reference_patients"]
    assert mixed.shape == a.shape


def test_signed_deviation_axes_distances_and_equal_record_aggregation():
    source = dict(level=1, wavelet="sym4", windows={"whole": [0, 651]})
    # Constant query offset: detail means vanish; approximation keeps its sign.
    waves = np.stack([np.full((3, 651), value) for value in (1., 3., -2.)])
    ids = np.array(["a", "a", "b"])
    atlas = np.stack([np.zeros((3, 651)), np.ones((3, 651))])
    result = tensor.deviation_features(waves, ids, ["a", "b"], atlas, source)
    assert result.shape == (2, 27)
    names = tensor.feature_names(source)
    assert names[:4] == ["reference_0/X/A1/whole/mean", "reference_0/X/A1/whole/rms",
                         "reference_0/X/D1/whole/mean", "reference_0/X/D1/whole/rms"]
    assert result[0, 0] == pytest.approx(2.) and result[1, 0] == pytest.approx(-2.)
    assert result[0, 1] == pytest.approx(2.) and result[1, 1] == pytest.approx(2.)
    assert result[0, -3] == pytest.approx(2*np.sqrt(3*651))
    # An unrelated query cannot change any previously computed patient features.
    alone = tensor.deviation_features(waves, ids, ["a"], atlas, source)
    np.testing.assert_array_equal(alone[0], result[0])
    with pytest.raises(ValueError, match="Missing"):
        tensor.deviation_features(waves, ids, ["missing"], atlas, source)


def test_heldout_features_do_not_change_classifier_features(tmp_path):
    data, cfg = fixture(tmp_path)
    train, query, _ = study.splits(data, cfg, 1)
    fit, ref = study.members(data, cfg, "healthy", train, query, 42, 1, 1)[0]
    X, names, atlas, _ = study.feature_context(data, cfg, "healthy", fit, ref, query, 42)
    changed = copy.deepcopy(data)
    changed["waves"][query] += 100
    changed["y"][query] = 0
    X2, _, atlas2, _ = study.feature_context(changed, cfg, "healthy", fit, ref, query, 42)
    np.testing.assert_array_equal(atlas, atlas2)
    np.testing.assert_array_equal(X[fit], X2[fit])
    assert not np.array_equal(X[query], X2[query])


def test_mi_only_scoring_keeps_hc_predictions_as_errors():
    classes = ["AMI", "ALMI", "ASMI", "IMI", "ILMI", "HC"]
    y = np.arange(6)
    p = np.eye(6)*.9+.1/6
    p[0] = p[5]
    assert report.scores(y, p, classes)["mi_only_macro_f1_six_way_predictions"] == pytest.approx(.8)
    result = report.paired_differences(y, [p]*3, [p]*3, 100, 42)
    assert result["mean_seed_delta_macro_f1"] == 0 and result["descriptive_patient_interval"] == [0., 0.]


def test_nested_run_resume_report_and_tamper_guards(tmp_path, monkeypatch):
    data, cfg = fixture(tmp_path)
    monkeypatch.setattr(study, "load_inputs", lambda config: data)
    calls = []
    # This backend substitutes only numerical PFN inference; all nested splits,
    # atlas transforms, selection, caching and report replay remain real.
    def actual(model, seed, config):
        return {"architecture": "TabPFN-v2", "package_version": config["package_version"], "n_estimators": 8,
                "fitted_ensemble_members": 8, "classes": model.classes_.tolist(), "n_features_in": model.n_features_in_,
                "backend_device": "cpu", "random_state": seed, "balance_probabilities": False,
                "softmax_temperature": .9, "fit_mode": "fit_preprocessors", "n_preprocessing_jobs": 1}
    def fake_fit(config, seed, X, y, query, state=None):
        calls.append(seed)
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
    assert len(calls) == 280
    summary = report.summarize_study(cfg)
    assert summary["total_member_contexts"] == 280 and summary["outer_member_contexts"] == 56
    assert summary["n_patients"] == len(data["ids"]) and len(summary["contrasts"]) == 4
    monkeypatch.setattr(study.backend, "run_fit", lambda *a, **k: pytest.fail("A completed study must resume without fitting"))
    assert study.run_study(cfg) == root
    assert report.summarize_study(cfg) == summary
    with pytest.raises(ValueError, match="changed"):
        study.run_study(dict(cfg, seed=123))
    path = next(root.glob("outer_*healthy*.json"))
    audit = json.loads(path.read_text());audit["atlas"]["self_reference_overlap"] = 1
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="scope changed"):
        report.summarize_study(cfg)
