import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from mi_localization import preprocessing_study, regularization as study
from mi_localization.neural import FoldNormalizer
from mi_localization.regularization_report import matching, summarize_study, validate_fit
from test_preprocessing import tiny_bank as preprocessing_bank, fake_fit as preprocessing_fake


def tiny_bank(tmp_path):
    original, waves, y, groups = preprocessing_bank(tmp_path)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/regularization_vcg.yaml").read_text())
    config.update({key: original[key] for key in ["seed", "split_seed", "training_seeds", "classes", "folds_path", "output_dir", "outer_splits", "inner_splits", "device", "torch_threads", "widths", "pool_bins", "batch_size", "samples_per_epoch", "max_epochs", "patience"]})
    config.update(bank_dir=original["denoised_bank_dir"], paper_config=str(tmp_path / "paper.yaml"), beats_path=str(tmp_path / "beats.csv"))
    paper = {"data": {"leads": ["vx", "vy", "vz"], "sampling_rate": 1000, "before_r": 25, "after_r": 39}}
    Path(config["paper_config"]).write_text(yaml.safe_dump(paper))
    pd.DataFrame({"patient_id": groups, "label": np.array(config["classes"])[y], "record": [f"{patient}/r" for patient in groups], "r_peak": np.tile([50, 100, 150], 27)}).to_csv(config["beats_path"], index=False)
    return config, waves * 0.9, y, groups


def normalizer():
    norm = FoldNormalizer()
    norm.wave_mean_ = np.array([0.3, -0.2, 0.7], dtype=np.float32)
    norm.wave_scale_ = np.array([0.4, 0.2, 0.3], dtype=np.float32)
    return norm


def test_gain_is_shared_in_physical_xyz_and_uses_no_global_rng_or_mutation():
    norm = normalizer()
    x = torch.randn(10, 3, 65)
    untouched = x.clone()
    candidate = {"gain_min": 0.95, "gain_max": 1.05, "noise_std": 0}
    global_state = torch.get_rng_state().clone()
    result = study.augment_training_batch(x, norm, candidate, torch.Generator().manual_seed(17))
    assert torch.equal(global_state, torch.get_rng_state()) and torch.equal(x, untouched)
    expected_gain = torch.empty(10, 1, 1).uniform_(0.95, 1.05, generator=torch.Generator().manual_seed(17))
    mean, scale = torch.tensor(norm.wave_mean_)[None, :, None], torch.tensor(norm.wave_scale_)[None, :, None]
    torch.testing.assert_close(result * scale + mean, expected_gain * (x * scale + mean))
    repeated = study.augment_training_batch(x, norm, candidate, torch.Generator().manual_seed(17))
    assert torch.equal(result, repeated)
    identity = {"gain_min": 1, "gain_max": 1, "noise_std": 0}
    generator = torch.Generator().manual_seed(17)
    state = generator.get_state().clone()
    assert study.augment_training_batch(x, norm, identity, generator) is x
    assert torch.equal(state, generator.get_state())


def test_noise_strength_is_training_standardized_and_draws_are_fresh():
    x = torch.zeros(256, 3, 65)
    candidate = {"gain_min": 1, "gain_max": 1, "noise_std": 0.005}
    generator = torch.Generator().manual_seed(3)
    first = study.augment_training_batch(x, normalizer(), candidate, generator)
    second = study.augment_training_batch(x, normalizer(), candidate, generator)
    assert not torch.equal(first, second)
    assert float(first.std()) == pytest.approx(0.005, rel=0.03)
    assert abs(float(first.mean())) < 0.0001


def test_frozen_control_and_zero_strength_custom_trainer_match_exactly(tmp_path):
    config, waves, y, groups = tiny_bank(tmp_path)
    candidate = config["candidates"][0]
    settings = dict(config, dropout=candidate["dropout"], weight_decay=candidate["weight_decay"])
    first, norm, control = study.fit_candidate(waves, y, groups, np.arange(81), None, 2, config, 42, candidate)
    original, _, _ = preprocessing_study.fit_waves(waves, y, groups, np.arange(81), None, 2, settings, 42)
    neutral = dict(candidate, id="regularized")
    second, _, custom = study.fit_candidate(waves, y, groups, np.arange(81), None, 2, config, 42, neutral)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    matching(control, custom, config)


def test_real_candidates_share_rows_normalization_weights_and_training_only_augmentation(tmp_path, monkeypatch):
    config, waves, y, groups = tiny_bank(tmp_path)
    untouched = waves.copy()
    fits, calls = [], []
    augment = study.augment_training_batch
    monkeypatch.setattr(study, "augment_training_batch", lambda *args: (calls.append(len(args[0])), augment(*args))[1])
    for candidate in config["candidates"]:
        model, norm, fit = study.fit_candidate(waves, y, groups, np.arange(54), np.arange(54, 81), 2, config, 42, candidate)
        fits.append(fit)
        count = len(calls)
        scores = study.predict_waves(model, norm, waves, y, np.arange(54, 81), config)
        assert len(calls) == count and np.isfinite(scores).all()
        validate_fit(fit, candidate, 42, config, 54, 18, 9)
        matching(fits[0], fit, config)
    matching(fits[2], fits[3], config, augmentation_stream=True)
    assert sum(calls) == sum(sum(row["augmented_presentations"] for row in fit["history"]) for fit in fits)
    np.testing.assert_array_equal(waves, untouched)
    assert all(fit["normalization"]["stat_mean_"] == [] for fit in fits)
    with pytest.raises(ValueError, match="overlap"):
        study.fit_candidate(waves, y, groups, np.arange(54), np.arange(54), 1, config, 42, config["candidates"][3])


def test_selection_is_inner_only_with_baseline_tie_break_and_no_outer_score():
    config = {"inner_splits": 2, "max_epochs": 25, "candidates": [{"id": name} for name in study.CANDIDATES]}
    fits = {name: [{"best_validation_macro_f1": 0.5, "selected_epoch": epoch, "outer_test_macro_f1": 1 if name == "combined" else 0} for epoch in [3, 4]] for name in study.CANDIDATES}
    selected = study.choose_candidate(fits, config)
    assert selected["selected_candidate"] == "baseline"
    assert selected["candidate_scores"][0]["final_epochs"] == 4
    fits["augmented"][0]["best_validation_macro_f1"] = 0.7
    assert study.choose_candidate(fits, config)["selected_candidate"] == "augmented"
    fits["augmented"].pop()
    with pytest.raises(ValueError, match="Missing candidate"):
        study.choose_candidate(fits, config)


def fake_fit(calls, winner="combined"):
    original = preprocessing_fake([])
    def fit(waves, y, groups, train, valid, epochs, config, seed, candidate, progress=None):
        calls.append((candidate["id"], set(groups[train]), None if valid is None else set(groups[valid]), seed))
        settings = dict(config, dropout=candidate["dropout"], weight_decay=candidate["weight_decay"])
        model, norm, audit = original(waves, y, groups, train, valid, epochs, settings, seed)
        augmented = study.has_augmentation(candidate)
        score = 0.6 if candidate["id"] == winner else 0.3
        audit.update(candidate=candidate, augmentation_seed=seed + study.AUGMENTATION_SEED_OFFSET if augmented else None, validation_augmentation=False)
        if valid is not None:
            audit["best_validation_macro_f1"] = score
        for row in audit["history"]:
            row.update(augmented_presentations=config["samples_per_epoch"] if augmented else 0,
                       augmentation_rng_state_sha256=study._hash([seed, row["epoch"], "augmentation"]) if augmented else None)
            if valid is not None:
                row["validation_patient_macro_f1"] = score
        if progress:
            progress(audit["history"][-1])
        return model, norm, audit
    return fit


def test_nested_membership_budget_resume_and_partial_scope(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    calls = []
    monkeypatch.setattr(study, "fit_candidate", fake_fit(calls))
    result = study.run_study(config, [1])
    assert result.complete.tolist() == [False] * 6 and len(calls) == 30
    frozen = pd.read_csv(config["folds_path"])
    test_ids = set(frozen.loc[frozen.fold == 1, "patient_id"])
    for _, train, valid, _ in calls:
        assert not train & test_ids
        if valid is not None:
            assert not valid & test_ids and not train & valid
    memberships = [(train, valid) for _, train, valid, _ in calls]
    for seed_offset in [0, 10, 20]:
        for candidate_offset in [0, 2, 4, 6]:
            assert memberships[:2] == memberships[seed_offset + candidate_offset:seed_offset + candidate_offset + 2]
    calls.clear()
    study.run_study(config, [1])
    assert not calls and "partial_complete" in study.status(config, [1])
    assert not (Path(config["output_dir"]) / "complete").exists()
    with pytest.raises(ValueError, match="settings/cache/runtime changed"):
        study.run_study(dict(config, max_epochs=3), [1])
    with pytest.raises(FileNotFoundError):
        summarize_study(config, draws=100)


@pytest.mark.parametrize("winner", ["baseline", "combined"])
def test_complete_report_all_seeds_selection_and_exact_baseline_fallback(tmp_path, monkeypatch, winner):
    config, *_ = tiny_bank(tmp_path)
    calls = []
    monkeypatch.setattr(study, "fit_candidate", fake_fit(calls, winner))
    assert study.run_study(config).complete.all()
    report = summarize_study(config, draws=100)
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["complete"] and summary["n_patients"] == 27 and summary["n_records"] == 27 and summary["n_beats"] == 81
    assert summary["training_seeds"] == [42, 123, 2026] and summary["selection_counts"][winner] == 9
    assert summary["unique_fit_count"] == len(calls) == (81 if winner == "baseline" else 90)
    bootstrap = json.loads((report.parent / "paired_bootstrap.json").read_text())
    assert bootstrap["unique_patients"] == 27
    path = report.parent / "seed_42/fold_01.json"
    saved = json.loads(path.read_text())
    broken = copy.deepcopy(saved)
    broken["selection"]["selected_candidate"] = "augmented"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="inner-only rule"):
        summarize_study(config, draws=100)
    broken = copy.deepcopy(saved)
    broken["inner_candidates"]["combined"][0]["history"][0]["sampled_rows_sha256"] = "tampered"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="identical sampled rows"):
        summarize_study(config, draws=100)
    broken = copy.deepcopy(saved)
    broken["final_fits"]["selected"]["history"][0]["validation_patient_macro_f1"] = 0.9
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="invalid stopping"):
        summarize_study(config, draws=100)
    if winner == "baseline":
        path.write_text(json.dumps(saved))
        csv = report.parent / "seed_42/patient_predictions_selected.csv"
        frame = pd.read_csv(csv)
        first, second = [f"probability_{label}" for label in config["classes"][:2]]
        frame.loc[0, first] += 1e-7
        frame.loc[0, second] -= 1e-7
        frame.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="fallback was not reused exactly"):
            summarize_study(config, draws=100)


def test_tampered_cache_refuses_resume_before_any_new_fit(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    calls = []
    monkeypatch.setattr(study, "fit_candidate", fake_fit(calls))
    study.run_study(config, [1])
    calls.clear()
    path = Path(config["bank_dir"]) / "waveforms.npy"
    waves = np.load(path)
    waves[0, 0, 0] += 1
    np.save(path, waves)
    with pytest.raises(ValueError, match="settings/cache/runtime changed"):
        study.run_study(config, [1])
    assert not calls


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_six_class_production_shape_augmented_smoothed_gpu_training():
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/regularization_vcg.yaml").read_text())
    config.update(device="mps", samples_per_epoch=128, torch_threads=1)
    waves = np.random.default_rng(23).normal(size=(48, 3, 651)).astype(np.float32)
    y, groups = np.repeat(np.arange(6), 8), np.repeat(np.arange(24), 2)
    model, norm, fit = study.fit_candidate(waves, y, groups, np.arange(48), None, 1, config, 42, config["candidates"][3])
    assert fit["parameter_count"] == 78198 and np.isfinite(fit["history"][0]["training_loss"])
    assert fit["history"][0]["augmented_presentations"] == 128
    assert np.isfinite(study.predict_waves(model, norm, waves, y, np.arange(12), config)).all()


def test_production_budget_and_invalid_preset_design():
    root = Path(__file__).resolve().parents[1] / "configs"
    config = yaml.safe_load((root / "regularization_vcg.yaml").read_text())
    original = yaml.safe_load((root / "preprocessing_vcg.yaml").read_text())
    study.validate_config(config)
    for key in ["seed", "split_seed", "training_seeds", "classes", "folds_path", "outer_splits", "inner_splits", "widths", "pool_bins", "learning_rate", "batch_size", "samples_per_epoch", "max_epochs", "patience"]:
        assert config[key] == original[key]
    assert config["candidates"][0]["dropout"] == original["dropout"] and config["candidates"][0]["weight_decay"] == original["weight_decay"]
    broken = copy.deepcopy(config)
    broken["candidates"][2]["dropout"] = 0.5
    with pytest.raises(ValueError, match="two-by-two"):
        study.validate_config(broken)
