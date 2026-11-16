"""Numerical distribution oracles, patient isolation and real nested restoration."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri
from threadpoolctl import threadpool_limits
import torch
import yaml

from mi_localization import distribution_features as features, distribution_study as study, distribution_report as report
from mi_localization.distribution_bank import DistributionBank


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    with threadpool_limits(limits=1): yield
    torch.set_num_threads(old)


@pytest.fixture
def small(tmp_path):
    config = yaml.safe_load(Path("configs/distribution_pooling_v1.yaml").read_text())
    config.update(output_dir=str(tmp_path/"results"), folds_path=str(tmp_path/"folds.csv"), outer_splits=2, inner_splits=2,
                  projection_seeds=[42], windows=[[0, 17], [16, 33], [0, 33]], dct_coefficients=2,
                  projections=3, quantiles=[.1, .5, .9], bootstrap_draws=50)
    rng = np.random.default_rng(19)
    ids = np.array([f"p{i:02}" for i in range(24)]); labels = np.repeat(np.arange(6), 4)
    patients = np.repeat(ids, 2); lengths = rng.integers(3, 7, len(patients)); offsets = np.r_[0, lengths.cumsum()]
    waves = rng.normal(size=(offsets[-1], 3, 33))
    waves[:, 0] += np.repeat(np.repeat(labels, 2), lengths)[:, None]/2
    values = features.extract(waves, config)
    templates = np.stack([features.extract(np.median(waves[offsets[i]:offsets[i+1]], axis=0)[None], config)[0] for i in range(len(patients))])
    bank = DistributionBank(values, templates, np.arange(len(waves)), offsets, patients, ids, labels, {"synthetic": True})
    assigned = np.tile([1, 2, 1, 2], 6)
    Path(config["folds_path"]).write_text("synthetic frozen folds\n")
    return config, bank, assigned


def test_signed_dct_physical_oracle_and_positive_gain(small):
    config, _, _ = small
    x = np.arange(99, dtype=float).reshape(1, 3, 33)/100-0.3
    y = features.extract(x, config)
    oracle = []
    for lo, hi in config["windows"]:
        record = []
        for z in x[0, :, lo:hi]:
            n = len(z)
            coeff = [np.sum(z*np.cos(np.pi*k*(np.arange(n)+.5)/n))*np.sqrt((1 if k == 0 else 2)/n)/np.sqrt(n)
                     for k in range(config["dct_coefficients"])]
            record.extend(coeff+[np.sqrt(np.mean(z*z)), np.abs(z).mean(), z.min(), z.max()])
        oracle.append(record)
    np.testing.assert_allclose(y[0], oracle, atol=1e-14)
    np.testing.assert_allclose(features.extract(2*x, config), 2*y, atol=1e-14)
    assert not np.allclose(features.extract(-x, config), y)


def test_shape_beyond_identical_mean_and_variance():
    q = [.1, .5, .9]
    a = np.array([-np.sqrt(2), 0, 0, np.sqrt(2)])[:, None, None]
    b = np.array([-1., -1., 1., 1.])[:, None, None]
    x, y = [features.summaries(z, np.zeros((1, 1)), q) for z in (a, b)]
    np.testing.assert_allclose(x["mean_embedding"], y["mean_embedding"])
    np.testing.assert_allclose(x["mean_spread"], y["mean_spread"])
    assert not np.allclose(x["quantiles"], y["quantiles"])
    for arm, values in features.summaries(a[::-1], np.zeros((1, 1)), q).items():
        np.testing.assert_allclose(x[arm], values)


def test_quantile_moment_oracle_and_kernel_scaling():
    z = np.arange(48, dtype=float).reshape(8, 2, 3)/4
    q = [.1, .5, .9]
    mean, sd = z.mean(0), z.std(0)
    result = features.summaries(z, mean, q)
    np.testing.assert_allclose(result["quantiles"].reshape(2, 3, 3), np.moveaxis(np.quantile(z, q, axis=0), 0, -1)/np.sqrt(3))
    np.testing.assert_allclose(result["mean_spread"].reshape(2, 3, 3), (mean[..., None]+sd[..., None]*ndtri(q))/np.sqrt(3))
    np.testing.assert_allclose(np.dot(result["mean_embedding"], result["mean_embedding"]), np.sum(mean*mean))
    only = features.summaries(z[:1], z[0], q)
    for arm in features.ARMS:np.testing.assert_allclose(only[arm], only["mean_embedding"])


def test_equal_patient_record_weights_and_query_isolation(small):
    config, bank, assigned = small
    train, query = bank.ids[assigned != 1], bank.ids[assigned == 1]
    t = features.DistributionTransform(config, 42).fit(bank, train)
    expected = np.mean([np.mean([bank.features[bank.rows(r)].mean(0) for r in bank.records_for([p])], axis=0) for p in train], axis=0)
    np.testing.assert_allclose(t.mean, expected, atol=1e-14)
    baseline = t.transform(bank, query)
    one = t.transform(bank, query[:1])
    for arm in features.ARMS:np.testing.assert_allclose(one[arm], baseline[arm][:1])
    changed = copy.deepcopy(bank)
    for r in changed.records_for(query):changed.features[changed.rows(r)] += 10000
    changed.labels[assigned == 1] = (changed.labels[assigned == 1]+1)%6
    fresh = features.DistributionTransform(config, 42).fit(changed, train)
    assert fresh.audit() == t.audit()
    for arm in features.ARMS:np.testing.assert_allclose(fresh.transform(changed, train)[arm], t.transform(bank, train)[arm])


def test_constant_feature_and_duplicate_patient_rejection(small):
    config, bank, _ = small
    bank.features[:] = 0; bank.template_features[:] = 0
    t = features.DistributionTransform(config, 42).fit(bank, bank.ids[:12])
    assert t.constant.all() and np.all(t.scale == 1)
    assert all(np.isfinite(v).all() and not v.any() for v in t.transform(bank, bank.ids[12:]).values())
    with pytest.raises(ValueError):t.fit(bank, np.array([bank.ids[0], bank.ids[0]]))
    with pytest.raises(ValueError):bank.records_for(["missing"])


def test_cache_replay_tamper_and_membership_rejection(small):
    config, bank, assigned = small; scope = study.make_scope(config, bank);root = Path(config["output_dir"])
    train, query = np.flatnonzero(assigned != 1), np.flatnonzero(assigned == 1)
    values, context = study.feature_context(bank, config, scope, root, 42, 1, 0, train, query)
    study.feature_context(bank, config, scope, root, 42, 1, 0, train, query, True)
    with pytest.raises(ValueError):study.feature_context(bank, config, scope, root, 42, 1, 0, query, train)
    with pytest.raises(ValueError):study.membership(bank, scope, train, train[:1])
    changed = copy.deepcopy(bank); changed.features[changed.rows(0)] += .1
    with pytest.raises(ValueError):study.feature_context(changed, config, scope, root, 42, 1, 0, train, query, True)
    with context.with_suffix(".npz").open("ab") as f:f.write(b"tamper")
    with pytest.raises(ValueError):study.feature_context(bank, config, scope, root, 42, 1, 0, train, query, True)


def test_outer_labels_cannot_change_selection_or_predictions(small):
    config, bank, assigned = small; scope = study.make_scope(config, bank);root = Path(config["output_dir"])
    rows, first = study.evaluate_fold(bank, assigned, config, scope, root, 42, 1)
    decisions = {p.name: p.read_bytes() for p in (root/"review").glob("selection_*.json")}
    bank.labels[rows] = (bank.labels[rows]+1)%6
    _, second = study.evaluate_fold(bank, assigned, config, scope, root, 42, 1)
    for a in features.ARMS:np.testing.assert_array_equal(first[a], second[a])
    assert decisions == {p.name: p.read_bytes() for p in (root/"review").glob("selection_*.json")}
    assert study.select([[.5, .5], [.5, .5]])[0] == 0


def test_full_nested_report_restores_every_classifier_without_refit(small, monkeypatch):
    config, bank, assigned = small
    monkeypatch.setattr(study, "load_inputs", lambda config, verify=False:(bank, assigned))
    study.run(config)
    def forbid(*args, **kwargs):raise AssertionError("Independent report may not refit classifiers")
    monkeypatch.setattr(study.RidgeClassifier, "fit", forbid)
    result = report.summarize(config)
    assert result["fits"] == result["all_models_restored"] == 40
    assert result["feature_contexts_reconstructed"] == result["matched_context_checks"] == 6
    assert result["outer_models_restored"] == 8
    assert set(result["ensemble_metrics"]) == set(features.ARMS)
    assert result["contrasts"]["quantiles_minus_mean_spread"]["primary"]
    assert all(m["classification_report"]["AMI"]["support"] == 4 for m in result["ensemble_metrics"].values())
    spec = importlib.util.spec_from_file_location("distribution_runner", "scripts/run_distribution_pooling.py")
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    assert runner.verified_ready(config)
    p = Path(config["output_dir"])/"complete/REPORT.md";p.write_text("tampered")
    assert not runner.verified_ready(config)


def test_scope_and_bad_config_rejected(small, monkeypatch):
    config, bank, assigned = small
    monkeypatch.setattr(study, "load_inputs", lambda config, verify=False:(bank, assigned))
    root = Path(config["output_dir"])/"complete";root.mkdir(parents=True)
    (root/"study_manifest.json").write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError):study.run(config)
    for key, value in [("arms", ["quantiles"]), ("ridge_alphas", [0]), ("quantiles", [.1, float("nan"), .9]),
                       ("projection_seeds", [42, 42]), ("windows", [[0, 1]])]:
        bad = copy.deepcopy(config);bad[key] = value
        with pytest.raises(ValueError):study.validate(bad)


def test_declared_real_budget_and_private_outputs():
    c = yaml.safe_load(Path("configs/distribution_pooling_v1.yaml").read_text());study.validate(c)
    assert study.budget(c) == dict(feature_contexts=120, fits=840, outer_models=120)
    assert len(c["windows"])*c["projections"]*len(c["quantiles"]) == 720
    ignored = Path(".gitignore").read_text()
    assert "results/distribution_pooling_v1/**/review/" in ignored
