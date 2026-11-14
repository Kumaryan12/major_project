import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from mi_localization import preprocessing_study as study, signal_audit as audit
from mi_localization.benchmark import create_fixed_folds, patient_balanced_weights
from mi_localization.contrastive import initial_state_hash
from mi_localization.neural import FoldNormalizer, VCGResNet, _hash, _seed
from mi_localization.preprocessing_report import check_pair, summarize_study


def tiny_bank(tmp_path):
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(81, 3, 65)).astype(np.float32)
    groups = np.repeat([f"p{i:02d}" for i in range(27)], 3)
    y = np.repeat(np.repeat(np.arange(3), 9), 3)
    classes = ["AMI", "IMI", "HC"]
    control, bank = tmp_path / "control", tmp_path / "raw"
    for root, waves in [(control, raw * 0.9), (bank, raw)]:
        root.mkdir()
        for name, values in [("waveforms", waves), ("groups", groups), ("y", np.array(classes)[y])]:
            np.save(root / f"{name}.npy", values)
        manifest = {"classes": classes, "shape": [81, 3, 65], "n_patients": 27, "n_beats": 81,
                    "fingerprint": root.name}
        if root == bank:
            manifest.update(control_fingerprint="control", control_waveforms_sha256=audit.file_hash(control / "waveforms.npy"),
                            waveforms_sha256=audit.file_hash(bank / "waveforms.npy"))
        (root / "manifest.json").write_text(json.dumps(manifest))
    config = {"seed": 42, "split_seed": 42, "training_seeds": [42, 123, 2026],
        "denoised_bank_dir": str(control), "raw_bank_dir": str(bank), "folds_path": str(tmp_path / "folds.csv"),
        "output_dir": str(tmp_path / "results"), "classes": classes, "outer_splits": 3, "inner_splits": 2,
        "arms": list(study.ARMS), "device": "cpu", "torch_threads": 1, "widths": [4, 8, 12], "pool_bins": 2,
        "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.01, "batch_size": 8,
        "samples_per_epoch": 16, "max_epochs": 2, "patience": 1}
    create_fixed_folds(y, groups, classes, Path(config["folds_path"]), 3, 42)
    return config, raw, y, groups


def test_signal_metrics_identity_scaling_flat_and_invalid():
    raw = np.random.default_rng(3).normal(size=(5, 3, 65))
    identity = audit.signal_metrics(raw, raw)
    np.testing.assert_allclose(identity["centered_relative_l2_change"], 0)
    np.testing.assert_allclose(identity["centered_correlation"], 1)
    scaled = audit.signal_metrics(raw, raw * 0.5)
    for key in ["centered_relative_l2_change", "centered_rms_ratio_denoised_over_raw", "derivative_rms_ratio_denoised_over_raw"]:
        np.testing.assert_allclose(scaled[key], 0.5)
    flat = audit.signal_metrics(np.zeros_like(raw), np.zeros_like(raw))
    assert all(np.isfinite(value).all() for value in flat.values())
    assert flat["raw_flat_beat"].tolist() == [1] * 5
    with pytest.raises(ValueError, match="finite aligned"):
        audit.signal_metrics(raw, raw[:, :, :-1])


def audit_fixture(tmp_path, monkeypatch):
    root, control = tmp_path / "source", tmp_path / "control"
    root.mkdir()
    control.mkdir()
    records, rows, waves = {}, [], []
    rng = np.random.default_rng(7)
    classes = ["AMI", "IMI", "HC"]
    for i in range(6):
        label, record = classes[i // 2], f"p{i}/r"
        records[record] = rng.normal(size=(250, 3))
        (root / f"p{i}").mkdir()
        header = "# Reason for admission: Healthy control" if label == "HC" else (
            "# Reason for admission: Myocardial infarction\n# Acute infarction (localization): " +
            ("anterior" if label == "AMI" else "inferior"))
        (root / f"{record}.hea").write_text(header)
        (root / f"{record}.xyz").write_bytes(records[record].tobytes())
        for peak in [50, 110, 170]:
            rows.append({"patient_id": f"p{i}", "record": record, "label": label, "r_peak": peak})
            waves.append((records[record] * 0.9)[peak - 25:peak + 40].T.astype(np.float32))
    beats = pd.DataFrame(rows)
    beats.to_csv(tmp_path / "beats.csv", index=False)
    np.save(control / "waveforms.npy", waves)
    np.save(control / "y.npy", beats.label.to_numpy(dtype=str))
    np.save(control / "groups.npy", beats.patient_id.to_numpy(dtype=str))
    (control / "manifest.json").write_text(json.dumps({"classes": classes, "shape": [18, 3, 65], "fingerprint": "test"}))
    paper = {"data": {"raw_dir": str(root), "leads": ["vx", "vy", "vz"], "sampling_rate": 1000,
                      "before_r": 25, "after_r": 39}, "signal": {"denoise_wavelet": "bior6.8", "denoise_level": 9}}
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper))
    config = {"paper_config": str(tmp_path / "paper.yaml"), "beats_path": str(tmp_path / "beats.csv"),
        "denoised_bank_dir": str(control), "raw_bank_dir": str(tmp_path / "bank"), "audit_dir": str(tmp_path / "audit"),
        "classes": classes, "seed": 42, "review_patients_per_class": 1, "feature_n_jobs": 1}
    monkeypatch.setattr(audit, "read_vcg", lambda root, record, leads: (records[record], 1000))
    monkeypatch.setattr(audit, "denoise_vcg", lambda raw, *args: raw * 0.9)
    monkeypatch.setattr(audit, "_review_plots", lambda examples, classes, directory: directory.mkdir(parents=True, exist_ok=True))
    return config, paper, beats, records


def test_raw_bank_retains_exact_rows_reuses_and_refuses_source_changes(tmp_path, monkeypatch):
    config, _, beats, records = audit_fixture(tmp_path, monkeypatch)
    selected = audit.review_selection(beats, config["classes"], 1, 42)
    assert selected == audit.review_selection(beats.assign(model_error=np.arange(len(beats))), config["classes"], 1, 42)
    assert sorted(item["label"] for item in selected.values()) == sorted(config["classes"])
    bank = audit.build_raw_bank_and_audit(config)
    expected = np.stack([records[row.record][row.r_peak - 25:row.r_peak + 40].T for row in beats.itertuples()]).astype(np.float32)
    np.testing.assert_array_equal(np.load(bank / "waveforms.npy"), expected)
    assert audit.build_raw_bank_and_audit(config) == bank
    summary = json.loads((Path(config["audit_dir"]) / "summary.json").read_text())
    assert summary["n_patients"] == 6 and summary["n_beats"] == 18 and summary["denoised_cache_verified_exactly"]
    assert summary["independent_peak_or_delineation_annotations_used"] is False
    source = Path(config["paper_config"]).parent / "source/p0/r.xyz"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="cache/source settings changed"):
        audit.build_raw_bank_and_audit(config)


def test_audit_refuses_denoised_mismatch_before_publishing_manifest(tmp_path, monkeypatch):
    config, *_ = audit_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "denoise_vcg", lambda raw, *args: raw * 0.8)
    with pytest.raises(ValueError, match="exactly match"):
        audit.build_raw_bank_and_audit(config)
    assert not (Path(config["raw_bank_dir"]) / "manifest.json").exists()


def test_audit_refuses_header_duplicates_and_incomplete_crops(tmp_path, monkeypatch):
    config, paper, beats, _ = audit_fixture(tmp_path, monkeypatch)
    rows = beats.iloc[:3].copy()
    reference = Path(config["denoised_bank_dir"]) / "waveforms.npy"
    wrong = rows.assign(label="HC")
    with pytest.raises(ValueError, match="Header/cached label"):
        audit._record("p0/r", wrong, paper, reference, None)
    wrong = rows.copy()
    wrong.loc[1, "r_peak"] = 50
    with pytest.raises(ValueError, match="Duplicate"):
        audit._record("p0/r", wrong, paper, reference, None)
    wrong.loc[1, "r_peak"] = 5
    with pytest.raises(ValueError, match="incomplete crop"):
        audit._record("p0/r", wrong, paper, reference, None)


def test_real_training_matches_init_sampling_and_uses_separate_training_scaling(tmp_path):
    config, raw, y, groups = tiny_bank(tmp_path)
    train, valid = np.arange(54), np.arange(54, 81)
    fits = []
    for waves in [raw * 0.9, raw]:
        model, norm, fit = study.fit_waves(waves, y, groups, train, valid, 2, config, 42)
        fits.append(fit)
        expected = FoldNormalizer().fit(waves, np.empty((81, 0)), train, patient_balanced_weights(y[train], groups[train]))
        np.testing.assert_array_equal(norm.wave_mean_, expected.wave_mean_)
        scores = study.predict_waves(model, norm, waves, y, valid, config)
        assert scores.shape == (27, 3) and np.isfinite(scores).all()
    check_pair(*fits, config)
    assert fits[0]["normalization"] != fits[1]["normalization"]
    _, _, final = study.fit_waves(raw, y, groups, train, None, 2, config, 42)
    assert final["validation_patients"] == 0 and len(final["history"]) == 2
    assert all("validation_patient_macro_f1" not in row for row in final["history"])
    with pytest.raises(ValueError, match="overlap"):
        study.fit_waves(raw, y, groups, train, train, 1, config, 42)


def fake_fit(calls):
    def fit(waves, y, groups, train, valid, epochs, config, seed, progress=None):
        calls.append((set(groups[train]), None if valid is None else set(groups[valid]), seed))
        _seed(seed, config)
        norm = FoldNormalizer().fit(waves, np.empty((len(waves), 0)), train, patient_balanced_weights(y[train], groups[train]))
        model = VCGResNet(n_classes=3, widths=(4, 8, 12), pool_bins=2, dropout=config["dropout"])
        history = [{"epoch": epoch, "training_loss": 0.7, "sampled_rows_sha256": _hash([seed, train.tolist(), epoch]),
                    "beat_presentations": config["samples_per_epoch"]} for epoch in range(1, epochs + 1)]
        if valid is not None:
            for row in history:
                row["validation_patient_macro_f1"] = 0.3
        if progress:
            progress(history[-1])
        return model, norm, {"seed": seed, "initial_state_sha256": initial_state_hash(model), "selected_epoch": 1 if valid is not None else epochs,
            "best_validation_macro_f1": 0.3 if valid is not None else None, "history": history, "patient_overlap": 0,
            "parameter_count": sum(p.numel() for p in model.parameters()), "train_patients": len(np.unique(groups[train])),
            "validation_patients": 0 if valid is None else len(np.unique(groups[valid])), "normalizer_fit_beats": len(train),
            "normalization": norm.audit()}
    return fit


def test_nested_patient_scope_matching_resume_and_partial_separation(tmp_path, monkeypatch):
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
    assert calls == [] and "partial_complete" in study.status(config, [1])
    assert not (Path(config["output_dir"]) / "complete").exists()
    with pytest.raises(ValueError, match="settings/runtime changed"):
        study.run_study(dict(config, max_epochs=3), [1])
    with pytest.raises(FileNotFoundError):
        summarize_study(config, draws=100)


def test_study_rejects_tampered_waveform_bank_before_training(tmp_path):
    config, *_ = tiny_bank(tmp_path)
    path = Path(config["raw_bank_dir"]) / "waveforms.npy"
    values = np.load(path)
    values[0, 0, 0] += 1
    np.save(path, values)
    with pytest.raises(ValueError, match="cache content changed"):
        study.run_study(config)


def test_complete_report_uses_all_seeds_and_rejects_unmatched_draws(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    monkeypatch.setattr(study, "fit_waves", fake_fit([]))
    result = study.run_study(config)
    assert result.complete.all() and len(result) == 6
    report = summarize_study(config, draws=100)
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["complete"] and summary["training_seeds"] == [42, 123, 2026]
    assert summary["n_patients"] == 27 and summary["n_beats"] == 81 and len(summary["per_seed"]) == 3
    bootstrap = json.loads((report.parent / "paired_bootstrap.json").read_text())
    assert bootstrap["unique_patients"] == 27
    assert set(bootstrap["mean_macro_f1_intervals"]) == set(study.ARMS)
    path = report.parent / "seed_42/metrics_raw.json"
    saved = json.loads(path.read_text())
    broken = copy.deepcopy(saved)
    broken["folds"][0]["final_fit"]["validation_patients"] = 1
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="invalid stopping"):
        summarize_study(config, draws=100)
    broken = copy.deepcopy(saved)
    broken["folds"][0]["inner_selection"][0]["history"][0]["sampled_rows_sha256"] = "tampered"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="identical sampled rows"):
        summarize_study(config, draws=100)
    broken = copy.deepcopy(saved)
    broken["folds"][0]["inner_selection"][0]["selected_epoch"] = 2
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="stopping epoch"):
        summarize_study(config, draws=100)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_waveform_fit_mps_finite(tmp_path):
    config, raw, y, groups = tiny_bank(tmp_path)
    config["device"] = "mps"
    model, norm, fit = study.fit_waves(raw, y, groups, np.arange(81), None, 1, config, 42)
    assert np.isfinite(fit["history"][0]["training_loss"])
    assert np.isfinite(study.predict_waves(model, norm, raw, y, np.arange(9), config)).all()
