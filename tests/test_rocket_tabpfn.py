"""Actual nested selection/source restoration with an explicit fake PFN backend."""
import copy
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.naive_bayes import GaussianNB
from threadpoolctl import threadpool_limits
import yaml

from mi_localization import rocket_study as source, rocket_report as source_report
from mi_localization.rocket_bank import TemplateBank
from mi_localization import rocket_tabpfn_study as study, rocket_tabpfn_report as report

pytestmark = pytest.mark.filterwarnings("ignore:Features .*:UserWarning")

@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    with threadpool_limits(limits=1): yield
    torch.set_num_threads(old)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    original = yaml.safe_load(Path("configs/rocket_vcg_v1.yaml").read_text())
    original.update(output_dir=str(tmp_path/"source"), folds_path=str(tmp_path/"folds.csv"), outer_splits=2, inner_splits=2,
        transform_seeds=[42], dct_coefficients=8, multirocket_dilations=2, hydra_groups=2, hydra_kernels=2, chunk_size=16,
        bootstrap_draws=100)
    ids = np.array([f"patient{i:02d}" for i in range(48)])
    y = np.repeat(np.arange(6), 8)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(96, 3, 33)).astype(np.float32)
    x[:, 0] += np.repeat(y, 2)[:, None]/3
    bank = TemplateBank(x, np.repeat(ids, 2), ids, y, {"synthetic": True})
    assigned = np.tile([1, 2], 24)
    pd.DataFrame(dict(patient_id=ids, label=np.asarray(original["classes"])[y], fold=assigned)).to_csv(original["folds_path"], index=False)
    monkeypatch.setattr(source, "load_inputs", lambda _: (bank, assigned))
    source.run(original)
    source_report.summarize(original)
    source_path = tmp_path/"source.yaml"
    source_path.write_text(yaml.safe_dump(original))
    cfg = yaml.safe_load(Path("configs/rocket_tabpfn_v1.yaml").read_text())
    cfg.update(source_config=str(source_path), output_dir=str(tmp_path/"new"), feature_counts=[4, 8],
               bootstrap_replicates=2, audit_replicates=2, bootstrap_draws=100)
    monkeypatch.setattr(study.backend, "prepare_checkpoint", lambda _: {"synthetic": True})
    monkeypatch.setattr(study.backend, "runtime_info", lambda: {"synthetic": True})
    calls = []
    def actual(model, seed, config):
        return dict(architecture="TabPFN-v2", package_version=config["package_version"], n_estimators=config["n_estimators"],
            fitted_ensemble_members=config["n_estimators"], classes=model.classes_.tolist(), n_features_in=model.n_features_in_,
            backend_device="cpu", random_state=seed, balance_probabilities=False, softmax_temperature=config["softmax_temperature"],
            fit_mode="fit_preprocessors", n_preprocessing_jobs=1)
    def fit(config, seed, X, y, query, state=None):
        calls.append(dict(X=X.copy(), y=y.copy(), query=query.copy(), state=state))
        model = GaussianNB().fit(X, y)
        if state is not None: joblib.dump((model, seed), state)
        return model.predict_proba(query), actual(model, seed, config)
    def restore(config, state, query):
        model, seed = joblib.load(state)
        return model.predict_proba(query), actual(model, seed, config)
    monkeypatch.setattr(study.backend, "run_fit", fit)
    monkeypatch.setattr(study.backend, "run_predict", restore)
    return cfg, study.load_inputs(cfg), calls


def test_default_budget_and_candidate_order():
    config = yaml.safe_load(Path("configs/rocket_tabpfn_v1.yaml").read_text())
    original = yaml.safe_load(Path(config["source_config"]).read_text())
    assert study.budget(config, original) == dict(fits=660, ridge_fits=420, tabpfn_fits=240, outer_models=120,
        source_feature_contexts=120, source_ridge_contexts=210, selector_contexts=240, reference_outer_models=30)
    assert study.decide([dict(mean_inner_macro_f1=.5), dict(mean_inner_macro_f1=.5)]) == 0
    with pytest.raises(ValueError): study.validate(dict(config, ridge_alphas=[3., 4.]), original)
    assert len(report.procedures(config)) == 7


def test_train_only_bootstrap_and_masks_shared_between_classifiers(fixture, tmp_path):
    cfg, data, _ = fixture
    root = tmp_path/"prepare";root.mkdir()
    scope = study.make_scope(cfg, data)
    train, query = np.flatnonzero(data["assigned"] != 1), np.flatnonzero(data["assigned"] == 1)
    x, path = study.get_features(data, 42, 1, 0, train, query)
    a = study.prepare(data, cfg, scope, root, 42, 1, 0, train, query, x, path, {})
    changed = copy.deepcopy(data)
    changed["bank"].labels[query] = (changed["bank"].labels[query]+1) % 6
    altered = x.copy();altered[len(train):] += 10000
    other = tmp_path/"other";other.mkdir()
    b = study.prepare(changed, cfg, scope, other, 42, 1, 0, train, query, altered, path, {})
    for k in cfg["feature_counts"]:
        assert a[k]["selector"].audit() == b[k]["selector"].audit()
        assert a[k]["audit"]["selected_training_features_sha256"] == b[k]["audit"]["selected_training_features_sha256"]
        np.testing.assert_array_equal(a[k]["train"], b[k]["train"])
        np.testing.assert_array_equal(a[k]["query"], x[len(train):, a[k]["selector"].selected])
    with pytest.raises(ValueError, match="overlapping"):
        study.prepare(data, cfg, scope, root, 42, 1, 0, train, train, x, path, {})


def test_actual_nested_fit_report_resume_and_source_immutability(fixture, monkeypatch):
    cfg, data, calls = fixture
    before = study.source_artifacts(data["root"])
    study.run(cfg)
    result = report.summarize(cfg)
    assert result["fits"] == result["all_new_models_restored"] == 32
    assert result["ridge_models_restored"] == 20 and result["tabpfn_models_restored"] == len(calls) == 12
    assert result["outer_models_restored"] == 8 and result["source_ridge_models_restored"] == 10
    assert result["source_features_reconstructed"] == result["bootstrap_caches_replayed"] == 6
    assert result["selector_masks_verified"] == 12 and result["frozen_reference_reproduced"]
    assert result["source_artifacts_unchanged"] and study.source_artifacts(data["root"]) == before
    assert all(c["state"] is not None for c in calls)
    root = Path(cfg["output_dir"])/"complete"
    for k in cfg["feature_counts"]:
        paths = list(root.glob(f"outer_*_i0_k{k}_*.json"))
        by_partition = {}
        for p in paths:
            a = json.loads(p.read_text())
            by_partition.setdefault((a["outer_fold"], a["transform_seed"]), []).append(a)
        for audits in by_partition.values():
            assert len(audits) == 2
            for key in ["selector", "selected_training_features_sha256", "selected_query_features_sha256"]:
                assert audits[0][key] == audits[1][key]
    monkeypatch.setattr(study.backend, "run_fit", lambda *a, **k: pytest.fail("Resume fitted a new PFN"))
    monkeypatch.setattr(study.RidgeClassifier, "fit", lambda *a, **k: pytest.fail("Resume fitted a new ridge"))
    study.run(cfg)
    assert report.summarize(cfg) == result
    with pytest.raises(ValueError, match="changed"):
        study.run(dict(cfg, correlation_threshold=.94))
    path = next(root.glob("outer_*_tabpfn_*.tabpfn_fit"))
    path.write_bytes(path.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="changed"):
        study.run(cfg)


def test_outer_label_counterfactual_preserves_choices_and_predictions(fixture, tmp_path):
    cfg, data, _ = fixture
    root = tmp_path/"counterfactual";root.mkdir()
    scope = study.make_scope(cfg, data)
    rows, before = study.evaluate_fold(data, cfg, scope, root, 42, 1)
    decisions = {p.name:p.read_bytes() for p in root.glob("selection_*.json")}
    changed = copy.deepcopy(data)
    changed["bank"].labels[rows] = (changed["bank"].labels[rows]+1) % 6
    _, after = study.evaluate_fold(changed, cfg, scope, root, 42, 1, verify=True)
    assert decisions == {p.name:p.read_bytes() for p in root.glob("selection_*.json")}
    for arm in before: np.testing.assert_array_equal(before[arm], after[arm])


def test_missing_partition_never_falls_back_to_fitting_source(fixture, monkeypatch):
    cfg, data, _ = fixture
    train, query = np.flatnonzero(data["assigned"] != 1), np.flatnonzero(data["assigned"] == 1)
    path = data["root"]/"context_01_s42_i0.npz"
    path.unlink()
    monkeypatch.setattr(source.features.FeatureTransform, "fit", lambda *a, **k: pytest.fail("Source was refitted"))
    with pytest.raises(ValueError, match="Missing source"):
        study.get_features(data, 42, 1, 0, train, query)


def test_selector_and_frozen_source_tampering_rejected(fixture, tmp_path):
    cfg, data, _ = fixture
    root = tmp_path/"guard";root.mkdir()
    scope = study.make_scope(cfg, data)
    train, query = np.flatnonzero(data["assigned"] != 1), np.flatnonzero(data["assigned"] == 1)
    x, path = study.get_features(data, 42, 1, 0, train, query)
    prepared = study.prepare(data, cfg, scope, root, 42, 1, 0, train, query, x, path, {})
    p = prepared[cfg["feature_counts"][0]]["path"]
    audit = json.loads(p.read_text());audit["selector_seed"] += 1;p.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="selector changed"):
        study.prepare(data, cfg, scope, root, 42, 1, 0, train, query, x, path, {}, verify=True)
    original = data["root"]/"outer_01_combined_s42.npz"
    original.write_bytes(original.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="source artifacts changed"): study.unchanged_source(data)


def test_real_pipeline_smoke_contract_with_fake_backend(fixture):
    cfg, _, calls = fixture
    result = study.smoke(cfg)
    assert result["classifier_restorations"] == 4 and len(calls) == 2
    assert result["bootstrap_cache_replayed"] and result["source_unchanged"] and result["query_batch_invariance"]


def test_live_monitor_and_complete_publication_guard(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("pfn_live", "scripts/live_status.py")
    monitor = importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    monkeypatch.setattr(monitor, "process_status", lambda *a: "running")
    root = tmp_path/"results/rocket_tabpfn_v1/complete"
    root.mkdir(parents=True)
    (root/"progress.json").write_text(json.dumps(dict(pid=123, state="fitting_classifier", learner="tabpfn", feature_budget=64, outer_fold=1)))
    value = monitor.snapshot(tmp_path)
    assert value["rocket_tabpfn"]["planned_fits"] == 660
    monitor.display(value)
    assert "feature_budget=64" in capsys.readouterr().out
    spec = importlib.util.spec_from_file_location("pfn_runner", "scripts/run_rocket_tabpfn.py")
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    (root/"REPORT.md").write_text("verified")
    (root/"study_manifest.json").write_text(json.dumps(dict(study_fingerprint="test")))
    summary = dict(complete=True, study_fingerprint="test", fits=660, all_new_models_restored=660, ridge_models_restored=420,
        tabpfn_models_restored=240, outer_models_restored=120, source_features_reconstructed=120, source_ridge_models_restored=210,
        bootstrap_caches_replayed=120, selector_masks_verified=240, source_artifacts_unchanged=True,
        frozen_reference_reproduced=True, all_outer_patients_covered=True)
    (root/"summary.json").write_text(json.dumps(summary))
    assert runner.verified_ready(tmp_path)
    summary["tabpfn_models_restored"] = 239
    (root/"summary.json").write_text(json.dumps(summary))
    assert not runner.verified_ready(tmp_path)
