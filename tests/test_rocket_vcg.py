"""Numerical oracles, partition isolation and actual nested fit/replay checks."""
import copy
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits
import yaml

from mi_localization import rocket_features as features, rocket_study as study, rocket_report as report
from mi_localization.rocket_bank import TemplateBank


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    with threadpool_limits(limits=1): yield
    torch.set_num_threads(old)


@pytest.fixture
def small(tmp_path):
    config = yaml.safe_load(Path("configs/rocket_vcg_v1.yaml").read_text())
    config.update(output_dir=str(tmp_path/"results"), folds_path=str(tmp_path/"folds.csv"), outer_splits=2, inner_splits=2,
                  transform_seeds=[42], dct_coefficients=8, multirocket_dilations=2, hydra_groups=2, hydra_kernels=2,
                  chunk_size=8, bootstrap_draws=100)
    rng = np.random.default_rng(17)
    ids = np.array([f"p{i:02d}" for i in range(24)])
    labels = np.repeat(np.arange(6), 4)
    records = np.repeat(ids, 2)
    x = rng.normal(size=(48, 3, 33)).astype(np.float32)
    x[:, 0] += np.repeat(labels, 2)[:, None]/3
    bank = TemplateBank(x, records, ids, labels, {"synthetic": True})
    assigned = np.tile([1, 2, 1, 2], 6)
    Path(config["folds_path"]).write_text("synthetic frozen folds\n")
    return config, bank, assigned


def slow_pool(x):
    positive = np.flatnonzero(x > 0)
    longest = current = 0
    for value in x:
        current = current+1 if value > 0 else 0
        longest = max(longest, current)
    return [len(positive)/len(x), x[positive].mean() if len(positive) else 0,
            positive.mean() if len(positive) else -1, longest]


def test_pooling_equality_boundaries_runs_and_no_positive():
    x = np.array([[-1, 2, 3, 0, 4, 5, 6, -1], [0]*8, [1]*8, [-1]*8], np.float32)
    actual = features.pool_positive(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(actual, [slow_pool(row) for row in x])
    assert actual[0, -1] == 3 and actual[2, -1] == 8
    rng = np.random.default_rng(3)
    values = rng.integers(-2, 3, (13, 41)).astype(np.float32)
    np.testing.assert_allclose(features.pool_positive(torch.from_numpy(values)).numpy(), [slow_pool(v) for v in values], atol=1e-6)


def test_multirocket_matches_numpy_convolution_oracle_and_batching():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 3, 33)).astype(np.float32)
    transform = features.MultiRocketFeatures(33, 2, 12).fit(x, np.array(["a", "b", "c", "d"]))
    query = rng.normal(size=(2, 3, 33)).astype(np.float32)
    result = transform.transform(query, 1)
    np.testing.assert_allclose(result, transform.transform(query, 2), rtol=2e-5, atol=2e-5)
    oracle, wi = [], 0
    for view, ds in enumerate(transform.dilations):
        values = query if view == 0 else np.diff(query, axis=-1)
        for di, dilation in enumerate(ds):
            p = 4*dilation
            padded = np.pad(values, ((0, 0), (0, 0), (p, p)))
            rows = []
            for b in range(len(query)):
                cells = []
                for k, weight in enumerate(transform.weights[wi]):
                    c = sum(padded[b, ch, j*dilation:j*dilation+values.shape[-1]]*weight[ch, j]
                            for ch in range(3) for j in range(9))
                    c = c-transform.biases[wi][k]
                    if (di+k) % 2: c = c[p:-p]
                    cells.extend(slow_pool(c))
                rows.append(cells)
            oracle.append(np.asarray(rows))
            wi += 1
    np.testing.assert_allclose(result, np.concatenate(oracle, axis=1), rtol=2e-5, atol=2e-5)


def test_transform_train_only_rng_and_hydra_restoration(small, tmp_path):
    config, bank, _ = small
    before = torch.get_rng_state().clone()
    train = np.arange(24)
    model = features.FeatureTransform(config, 42).fit(bank.templates[train], bank.record_patients[train])
    assert torch.equal(before, torch.get_rng_state())
    other = copy.deepcopy(bank.templates)
    other[24:] *= 10000
    same = features.FeatureTransform(config, 42).fit(other[train], bank.record_patients[train])
    assert model.audit() == same.audit()
    expected_scale = np.sqrt(np.mean(bank.templates[train].astype(float)**2))
    assert model.scale == pytest.approx(expected_scale)
    p = tmp_path/"transform.joblib"
    joblib.dump(model, p)
    restored = joblib.load(p)
    assert restored.audit() == model.audit()
    values = model.transform(bank.templates[24:28])
    for arm, x in restored.transform(bank.templates[24:28]).items():
        np.testing.assert_allclose(x, values[arm], atol=1e-6)
    with torch.inference_mode():
        upstream = model.hydra(torch.from_numpy((bank.templates[24:28]/model.scale).copy())).numpy()
    np.testing.assert_allclose(values["hydra"], upstream, atol=1e-6)
    different = features.FeatureTransform(config, 123).fit(bank.templates[train], bank.record_patients[train])
    assert different.audit()["hydra_weights"] != model.audit()["hydra_weights"]
    np.testing.assert_array_equal(different.transform(bank.templates[24:28])["dct"], values["dct"])


def test_equal_record_aggregation_not_beat_or_record_count_weighting():
    value = features.aggregate({"x": np.array([[1.], [9.], [3.]])}, np.array(["a", "a", "b"]), np.array(["b", "a"]))
    np.testing.assert_array_equal(value["x"], [[3.], [5.]])
    with pytest.raises(ValueError, match="membership"):
        features.aggregate({"x": np.ones((1, 1))}, np.array(["a"]), np.array(["b"]))


def test_template_bank_uses_all_beats_own_records_and_rejects_changes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mi_localization import rocket_bank
    waves = np.stack([np.full((3, 33), value, np.float32) for value in [1, 2, 10, 20, 30]])
    original = SimpleNamespace(waves=waves, record_patients=np.array(["a", "b"]), ids=np.array(["a", "b"]),
                               labels=np.array([0, 1]), provenance={"synthetic": True},
                               rows=lambda i: np.array([0, 1]) if i == 0 else np.array([2, 3, 4]))
    monkeypatch.setattr(rocket_bank.chronology_bank, "prepare_bank", lambda _: original)
    config = dict(template_bank_dir=str(tmp_path/"bank"), output_dir=str(tmp_path/"results"))
    bank = rocket_bank.prepare(config)
    np.testing.assert_array_equal(bank.templates[:, 0, 0], [1.5, 20])
    assert bank.provenance["bank_manifest"]["n_beats"] == 5
    np.testing.assert_array_equal(rocket_bank.prepare(config).templates, bank.templates)
    path = Path(config["template_bank_dir"])/"templates.npz"
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="source/content changed"): rocket_bank.prepare(config)


def test_declared_budget_and_default_dimensions():
    config = yaml.safe_load(Path("configs/rocket_vcg_v1.yaml").read_text())
    assert study.budget(config) == {"feature_contexts": 120, "fits": 840, "outer_models": 120}
    assert features.MultiRocketFeatures(651, 4, 42).dilations == [[1, 4, 18, 81], [1, 4, 18, 81]]
    invalid = dict(config, hydra_groups=3)
    with pytest.raises(ValueError): study.validate(invalid)


def test_actual_nested_study_restoration_resume_and_guards(small, monkeypatch):
    config, bank, assigned = small
    monkeypatch.setattr(study, "load_inputs", lambda _: (bank, assigned))
    study.run(config)
    result = report.summarize(config)
    assert result["fits"] == result["all_models_restored"] == 40
    assert result["outer_models_restored"] == 8 and result["feature_contexts_reconstructed"] == 6
    assert result["deterministic_dct_control_verified"] and result["all_original_templates_covered"]
    root = Path(config["output_dir"])/"complete"
    reports = (root/"REPORT.md").read_text()
    assert "not calibrated" in reports and "experimental multivariate" in reports
    monkeypatch.setattr(features.FeatureTransform, "fit", lambda *a, **k: pytest.fail("Resume unexpectedly fitted a transform"))
    monkeypatch.setattr(study.RidgeClassifier, "fit", lambda *a, **k: pytest.fail("Resume unexpectedly fitted a classifier"))
    study.run(config)
    with pytest.raises(ValueError, match="scope changed"):
        study.run(dict(config, ridge_alphas=[1., 101.]))
    path = next(root.glob("outer_*.npz"))
    path.write_bytes(path.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="context changed"):
        study.run(config)


def test_outer_labels_cannot_choose_features_alpha_or_predictions(small, monkeypatch):
    config, bank, assigned = small
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True)
    scope = study.make_scope(config, bank)
    rows, before = study.evaluate_fold(bank, assigned, config, scope, root, 42, 1)
    decisions = {p.name: p.read_bytes() for p in root.glob("selection_*.json")}
    changed = copy.deepcopy(bank)
    changed.labels[rows] = (changed.labels[rows]+1) % 6
    _, after = study.evaluate_fold(changed, assigned, config, scope, root, 42, 1, verify=True)
    for arm in config["arms"]: np.testing.assert_array_equal(before[arm], after[arm])
    assert decisions == {p.name: p.read_bytes() for p in root.glob("selection_*.json")}


def test_context_transform_tampering_and_wrong_partition_rejected(small):
    config, bank, _ = small
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True)
    scope = study.make_scope(config, bank)
    train, query = np.arange(12), np.arange(12, 24)
    _, path = study.feature_context(bank, config, scope, root, 42, 1, 0, train, query)
    with pytest.raises(ValueError, match="context changed"):
        study.feature_context(bank, config, scope, root, 42, 1, 0, query, train)
    with pytest.raises(ValueError, match="overlapping"):
        study.feature_context(bank, config, scope, root, 42, 1, 0, train, train)
    model = joblib.load(path.with_suffix(".joblib"))
    model.multirocket.biases[0][0] += 1
    joblib.dump(model, path.with_suffix(".joblib"))
    with pytest.raises(ValueError, match="context changed"):
        study.feature_context(bank, config, scope, root, 42, 1, 0, train, query, verify=True)


def test_runner_complete_guard_requires_all_restorations(tmp_path):
    spec = importlib.util.spec_from_file_location("rocket_runner", "scripts/run_rocket_vcg.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = tmp_path/runner.RESULTS/"complete"
    root.mkdir(parents=True)
    (root/"REPORT.md").write_text("verified")
    (root/"study_manifest.json").write_text(json.dumps(dict(study_fingerprint="abc")))
    value = dict(complete=True, study_fingerprint="abc", fits=840, all_models_restored=840, outer_models_restored=120,
                 feature_contexts_reconstructed=120, deterministic_dct_control_verified=True, all_original_templates_covered=True)
    (root/"summary.json").write_text(json.dumps(value))
    assert runner.verified_ready(tmp_path)
    value["all_models_restored"] = 839
    (root/"summary.json").write_text(json.dumps(value))
    assert not runner.verified_ready(tmp_path)


def test_live_monitor_exposes_rocket_context_and_fit_budget(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("rocket_live", "scripts/live_status.py")
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    monkeypatch.setattr(monitor, "process_status", lambda *args: "running")
    root = tmp_path/"results/rocket_vcg_v1/complete"
    root.mkdir(parents=True)
    (root/"progress.json").write_text(json.dumps(dict(pid=123, state="extracting_features", transform_seed=42, outer_fold=1, inner_fold=2)))
    value = monitor.snapshot(tmp_path)
    assert value["rocket"]["planned_fits"] == 840 and value["rocket"]["planned_outer"] == 120
    monitor.display(value)
    output = capsys.readouterr().out
    assert "BOUNDED RANDOM-CONVOLUTION FEATURES" in output and "transform_seed=42" in output
