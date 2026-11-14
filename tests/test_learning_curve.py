import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from mi_localization import learning_curve as study
from mi_localization.benchmark import inner_patient_splits
from mi_localization.learning_curve_report import summarize_study
from test_regularization import tiny_bank as regularization_bank
from test_preprocessing import fake_fit


def tiny_bank(tmp_path):
    config, waves, y, groups = regularization_bank(tmp_path)
    config.pop("candidates")
    config.update(training_fractions=[0.25, 0.5, 0.75, 1.0], subset_seed=42, dropout=0.2, weight_decay=0.01)
    return config, waves, y, groups


def test_subsets_are_complete_nested_stratified_and_model_seed_independent(tmp_path):
    config, waves, y, groups = tiny_bank(tmp_path)
    eligible = np.arange(81)
    previous = set()
    for fraction in config["training_fractions"]:
        rows, description = study.select_subset(y, groups, eligible, fraction, config["classes"], 2, 42)
        ids = set(groups[rows])
        assert previous <= ids
        assert len(rows) == len(ids) * 3
        assert set(description["class_patient_counts"].values()) == {int(np.ceil(9 * fraction))}
        same, _ = study.select_subset(y, groups, eligible[::-1], fraction, config["classes"], 2, 42)
        np.testing.assert_array_equal(rows, same)
        previous = ids
    np.testing.assert_array_equal(rows, eligible)
    with pytest.raises(ValueError, match="complete patients"):
        study.select_subset(y, groups, eligible[1:], 0.5, config["classes"], 2, 42)
    with pytest.raises(ValueError, match="Too few patients"):
        study.select_subset(y, groups, eligible, 0.1, config["classes"], 2, 42)


def test_plan_holds_outer_test_constant_and_full_size_preserves_original_inner_folds(tmp_path):
    config, _, y, groups = tiny_bank(tmp_path)
    *_, plan, descriptions = study.inputs_and_plan(config, [1, 2, 3])
    for fold in [1, 2, 3]:
        full_train, full_test, full_inner = plan[study.fraction_key(1)][fold]
        original = list(inner_patient_splits(y[full_train], groups[full_train], config["inner_splits"], config["split_seed"] + fold * 1000))
        for (ti, vi), (expected_ti, expected_vi) in zip(full_inner, original):
            np.testing.assert_array_equal(ti, full_train[expected_ti])
            np.testing.assert_array_equal(vi, full_train[expected_vi])
        for fraction in config["training_fractions"]:
            train, test, inner = plan[study.fraction_key(fraction)][fold]
            np.testing.assert_array_equal(test, full_test)
            selected = set(groups[train])
            assert not selected & set(groups[test])
            for ti, vi in inner:
                assert set(groups[ti]) | set(groups[vi]) == selected
                assert not set(groups[ti]) & set(groups[vi])


def test_partial_curve_keeps_all_test_patients_resumes_and_refuses_drift(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    calls = []
    monkeypatch.setattr(study, "fit_waves", fake_fit(calls))
    result = study.run_study(config, [1])
    assert len(result) == 12 and not result.complete.any() and len(calls) == 36
    frozen = pd.read_csv(config["folds_path"])
    excluded = set(frozen.loc[frozen.fold == 1, "patient_id"])
    for train, valid, _ in calls:
        assert not train & excluded
        if valid is not None:
            assert not valid & excluded and not train & valid
    calls.clear()
    study.run_study(config, [1])
    assert not calls and "partial_complete" in study.status(config, [1])
    assert not (Path(config["output_dir"]) / "complete").exists()
    with pytest.raises(ValueError, match="settings/cache/runtime/subsets changed"):
        study.run_study(dict(config, subset_seed=123), [1])
    first = Path(config["output_dir"]) / "partial_folds_1/seed_42/fraction_025"
    (first / "fold_01.json").unlink()
    (first / "fold_01.npz").unlink()
    inner_path = first / "inner_01_1.json"
    inner = json.loads(inner_path.read_text())
    inner["training_fraction"] = 0.5
    inner_path.write_text(json.dumps(inner))
    with pytest.raises(ValueError, match="identity/settings changed"):
        study.run_study(config, [1])
    assert not calls


def test_complete_curve_checks_resubstitution_and_pooled_predictions(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    monkeypatch.setattr(study, "fit_waves", fake_fit([]))
    assert study.run_study(config).complete.all()
    report = summarize_study(config, draws=100)
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["complete"] and summary["n_patients"] == 27 and len(summary["curve"]) == 4
    assert summary["fit_count"] == 108 and len(summary["per_seed"]) == 12
    assert summary["curve"][0]["mean_training_patients"] < summary["curve"][-1]["mean_training_patients"]
    assert (report.parent / "learning_curve.png").exists() and (report.parent / "learning_curve.svg").exists()
    intervals = json.loads((report.parent / "paired_bootstrap.json").read_text())["comparisons"]
    assert len(intervals) == 3 and all(value["unique_patients"] == 27 for value in intervals.values())
    path = report.parent / "seed_42/fraction_025/fold_01.json"
    saved = json.loads(path.read_text())
    saved["final_fit"]["normalizer_fit_beats"] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="scope/seed/augmentation mismatch"):
        summarize_study(config, draws=100)


def test_curve_production_configuration_retains_original_baseline_budget():
    root = Path(__file__).resolve().parents[1] / "configs"
    config = yaml.safe_load((root / "learning_curve_vcg.yaml").read_text())
    original = yaml.safe_load((root / "preprocessing_vcg.yaml").read_text())
    study.validate_config(config)
    for key in ["seed", "split_seed", "training_seeds", "classes", "folds_path", "outer_splits", "inner_splits", "widths", "pool_bins", "dropout", "learning_rate", "weight_decay", "batch_size", "samples_per_epoch", "max_epochs", "patience"]:
        assert config[key] == original[key]
    assert config["training_fractions"] == [0.25, 0.5, 0.75, 1.0]
