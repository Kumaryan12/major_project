import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from mi_localization import context_bank as bank, context_study as study
from mi_localization.neural import _hash
from mi_localization.preprocessing_report import check_pair
from mi_localization.context_report import summarize_study
from mi_localization.signal_audit import file_hash
from test_preprocessing import tiny_bank as preprocessing_bank, fake_fit, audit_fixture


def tiny_bank(tmp_path):
    config, raw, y, groups = preprocessing_bank(tmp_path)
    short = Path(config["denoised_bank_dir"])
    long = tmp_path / "long"
    long.mkdir()
    config.update(short_bank_dir=str(short), long_bank_dir=str(long), audit_dir=str(tmp_path / "audit"),
                  arms=list(study.ARMS), before_r=35, after_r=61, boundary_mode="reflect")
    waves = np.random.default_rng(7).normal(size=(81, 3, 97)).astype(np.float32)
    waves[:, :, 10:75] = np.load(short / "waveforms.npy")
    for name, values in [("waveforms", waves), ("y", np.load(short / "y.npy")), ("groups", groups), ("padding", np.zeros((81, 2), dtype=np.int32))]:
        np.save(long / f"{name}.npy", values)
    provenance = {"classes": config["classes"], "before_r": 35, "after_r": 61, "boundary_mode": "reflect",
        "control_fingerprint": "control", "control_files": {name: file_hash(short / f"{name}.npy") for name in ["waveforms", "y", "groups"]},
        "paper": {"data": {"before_r": 25, "after_r": 39}}}
    manifest = {"classes": config["classes"], "shape": [81, 3, 97], "n_beats": 81, "n_patients": 27,
        "fingerprint": _hash(provenance), "provenance": provenance,
        "files": {name: file_hash(long / f"{name}.npy") for name in ["waveforms", "y", "groups", "padding"]}}
    (long / "manifest.json").write_text(json.dumps(manifest))
    audit = Path(config["audit_dir"])
    audit.mkdir()
    (audit / "summary.json").write_text(json.dumps({"bank_fingerprint": manifest["fingerprint"], "central_short_crop_verified_exactly": True,
        "padded_beats": 0, "padded_patients": 0, "n_beats": 81}))
    return config, y, groups


def test_reflection_crop_preserves_observed_samples_and_reports_missing_edges():
    raw = np.arange(90).reshape(30, 3).astype(float)
    crop, padding = bank.context_crop(raw, 4, 7, 6)
    assert padding == (3, 0) and crop.shape == (3, 14)
    np.testing.assert_array_equal(crop[:, 3:], raw[:11].T)
    np.testing.assert_array_equal(crop[:, :3], raw[[3, 2, 1]].T)
    crop, padding = bank.context_crop(raw, 27, 3, 7)
    assert padding == (0, 5)
    np.testing.assert_array_equal(crop[:, :6], raw[24:].T)
    np.testing.assert_array_equal(crop[:, 6:], raw[[28, 27, 26, 25, 24]].T)
    crop, padding = bank.context_crop(raw, 10, 3, 5)
    assert padding == (0, 0)
    np.testing.assert_array_equal(crop, raw[7:16].T)
    with pytest.raises(ValueError, match="bounds"):
        bank.context_crop(raw, -1, 3, 5)


def test_bank_preserves_all_central_crops_counts_padding_and_guards_cache(tmp_path, monkeypatch):
    config, paper, beats, records = audit_fixture(tmp_path, monkeypatch)
    config.update(short_bank_dir=config["denoised_bank_dir"], long_bank_dir=str(tmp_path / "long"),
                  before_r=75, after_r=100, boundary_mode="reflect")
    monkeypatch.setattr(bank, "read_vcg", lambda root, record, leads: (records[record], 1000))
    monkeypatch.setattr(bank.signal, "denoise_vcg", lambda raw, *args: raw * 0.9)
    result = bank.build_context_bank(config)
    waves = np.load(result / "waveforms.npy")
    short = np.load(Path(config["short_bank_dir"]) / "waveforms.npy")
    np.testing.assert_array_equal(waves[:, :, 50:115], short)
    assert waves.shape == (18, 3, 176)
    assert bank.build_context_bank(config) == result
    audit = json.loads((Path(config["audit_dir"]) / "summary.json").read_text())
    assert audit["n_patients"] == 6 and audit["n_beats"] == 18
    assert audit["padded_beats"] == 12 and audit["left_padded_beats"] == audit["right_padded_beats"] == 6
    assert audit["central_short_crop_verified_exactly"]
    waves[0, 0, 0] += 1
    np.save(result / "waveforms.npy", waves)
    with pytest.raises(ValueError, match="cache/source settings changed"):
        bank.build_context_bank(config)


def test_context_rejects_modified_denoising_and_noncontaining_window(tmp_path, monkeypatch):
    config, paper, _, records = audit_fixture(tmp_path, monkeypatch)
    config.update(short_bank_dir=config["denoised_bank_dir"], long_bank_dir=str(tmp_path / "long"),
                  before_r=35, after_r=61, boundary_mode="reflect")
    monkeypatch.setattr(bank, "read_vcg", lambda root, record, leads: (records[record], 1000))
    monkeypatch.setattr(bank.signal, "denoise_vcg", lambda raw, *args: raw * 0.8)
    with pytest.raises(ValueError, match="exactly match"):
        bank.build_context_bank(config)
    assert not (Path(config["long_bank_dir"]) / "manifest.json").exists()
    with pytest.raises(ValueError, match="entire original"):
        bank.validate_windows(dict(config, before_r=20), paper)
    with pytest.raises(ValueError, match="boundary policy"):
        bank.validate_windows(dict(config, boundary_mode="zero"), paper)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_real_variable_length_fits_match_initialization_rows_and_parameter_count(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    config, y, groups = tiny_bank(tmp_path)
    config["device"] = device
    waves, *_ = study.load_banks(config)
    fits = []
    for arm in study.ARMS:
        model, normalizer, fit = study.fit_waves(waves[arm], y, groups, np.arange(54), np.arange(54, 81), 2, config, 42)
        assert np.isfinite(study.predict_waves(model, normalizer, waves[arm], y, np.arange(9), config)).all()
        fits.append(fit)
    check_pair(*fits, config)
    assert fits[0]["normalization"] != fits[1]["normalization"]


def test_fixed_nested_scope_resume_partial_and_changed_settings(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    calls = []
    monkeypatch.setattr(study, "fit_waves", fake_fit(calls))
    result = study.run_study(config, [1])
    assert result.complete.tolist() == [False] * 6 and len(calls) == 18
    frozen = pd.read_csv(config["folds_path"])
    test_ids = set(frozen.loc[frozen.fold == 1, "patient_id"])
    for train, valid, _ in calls:
        assert not train & test_ids
        if valid is not None:
            assert not valid & test_ids and not train & valid
    memberships = [(train, valid) for train, valid, _ in calls]
    assert all(memberships[:3] == memberships[offset:offset + 3] for offset in range(0, 18, 3))
    calls.clear()
    study.run_study(config, [1])
    assert not calls and "partial_complete" in study.status(config, [1])
    assert not (Path(config["output_dir"]) / "complete").exists()
    with pytest.raises(ValueError, match="settings/runtime changed"):
        study.run_study(dict(config, max_epochs=3), [1])
    with pytest.raises(ValueError, match="window settings changed"):
        study.run_study(dict(config, before_r=40), [1])


def test_complete_report_validates_all_seeds_sampling_and_no_test_stopping(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    monkeypatch.setattr(study, "fit_waves", fake_fit([]))
    assert study.run_study(config).complete.all()
    report = summarize_study(config, draws=100)
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["complete"] and summary["n_patients"] == 27 and len(summary["per_seed"]) == 3
    bootstrap = json.loads((report.parent / "paired_bootstrap.json").read_text())
    assert bootstrap["unique_patients"] == 27 and set(bootstrap["mean_macro_f1_intervals"]) == {"short", "long"}
    assert "require reflection at recording boundaries" in report.read_text()
    path = report.parent / "seed_42/metrics_long.json"
    saved = json.loads(path.read_text())
    broken = copy.deepcopy(saved)
    broken["folds"][0]["inner_selection"][0]["history"][0]["sampled_rows_sha256"] = "tampered"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="identical sampled rows"):
        summarize_study(config, draws=100)
    broken = copy.deepcopy(saved)
    broken["folds"][0]["final_fit"]["validation_patients"] = 1
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="invalid stopping"):
        summarize_study(config, draws=100)


def test_context_loader_rejects_changed_control_and_long_metadata(tmp_path):
    config, *_ = tiny_bank(tmp_path)
    path = Path(config["long_bank_dir"]) / "padding.npy"
    np.save(path, np.ones((81, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="cache content"):
        study.load_banks(config)


def test_production_config_changes_window_not_model_or_training_budget():
    root = Path(__file__).resolve().parents[1] / "configs"
    original = yaml.safe_load((root / "preprocessing_vcg.yaml").read_text())
    config = yaml.safe_load((root / "context_vcg.yaml").read_text())
    for key in ["seed", "split_seed", "training_seeds", "classes", "folds_path", "outer_splits", "inner_splits", "widths", "pool_bins", "dropout", "learning_rate", "weight_decay", "batch_size", "samples_per_epoch", "max_epochs", "patience"]:
        assert config[key] == original[key]
    assert config["before_r"] == 350 and config["after_r"] == 650


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_production_long_shape_six_class_gpu_fit_is_finite():
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/context_vcg.yaml").read_text())
    config.update(device="mps", torch_threads=1, samples_per_epoch=128)
    waves = np.random.default_rng(27).normal(size=(48, 3, 1001)).astype(np.float32)
    y, groups = np.repeat(np.arange(6), 8), np.repeat(np.arange(24), 2)
    model, normalizer, fit = study.fit_waves(waves, y, groups, np.arange(48), None, 1, config, 42)
    assert fit["parameter_count"] == 78198 and np.isfinite(fit["history"][0]["training_loss"])
    assert np.isfinite(study.predict_waves(model, normalizer, waves, y, np.arange(12), config)).all()
