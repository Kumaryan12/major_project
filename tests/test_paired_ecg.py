import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb
import yaml

from mi_localization import paired_bank as bank, paired_study as study
from mi_localization.benchmark import create_fixed_folds, patient_balanced_weights
from mi_localization.contrastive import initial_state_hash
from mi_localization.neural import FoldNormalizer
from mi_localization.preprocessing_study import fit_waves


def fixture(tmp_path):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/paired_ecg_vcg.yaml").read_text())
    cfg.update(classes=["AMI", "IMI", "HC"], widths=[4, 8, 12], pool_bins=2, fusion_rank=4,
        paper_config=str(tmp_path / "paper.yaml"), beats_path=str(tmp_path / "beats.csv"), bank_dir=str(tmp_path / "vcg"),
        ecg_bank_dir=str(tmp_path / "ecg"), output_dir=str(tmp_path / "results"), folds_path=str(tmp_path / "folds.csv"),
        outer_splits=3, inner_splits=2, device="cpu", torch_threads=1, conversion_workers=1,
        batch_size=8, samples_per_epoch=16, max_epochs=2, patience=1)
    raw, target = tmp_path / "raw", Path(cfg["bank_dir"])
    raw.mkdir()
    target.mkdir()
    rows, crops = [], []
    names = list(bank.LEADS) + ["vx", "vy", "vz"]
    for i in range(27):
        patient, record, label = f"p{i:02}", f"r{i:02}", cfg["classes"][i // 9]
        comments = ["Reason for admission: Healthy control"] if label == "HC" else ["Reason for admission: Myocardial infarction", "Acute infarction (localization): " + ("anterior" if label == "AMI" else "inferior")]
        wave = np.random.default_rng(i).normal(0, .2, (500, 15))
        wfdb.wrsamp(record, 1000, ["mV"]*15, names, p_signal=wave, fmt=["16"]*15, comments=comments, write_dir=str(raw))
        recorded = wfdb.rdrecord(str(raw / record)).p_signal
        cleaned = bank.signal.denoise_vcg(recorded[:, 12:], "bior6.8", 9)
        for peak in [100, 200, 300]:
            rows.append({"patient_id": patient, "record": record, "label": label, "r_peak": peak})
            crops.append(cleaned[peak-25:peak+40].T.astype(np.float32))
    frame = pd.DataFrame(rows)
    frame.to_csv(cfg["beats_path"], index=False)
    # Bank provenance checks expect separate encoded XYZ files, as in real PTB.
    for record in frame.record.unique():
        (raw / f"{record}.xyz").write_bytes(b"fixture source XYZ provenance")
    y = np.array([cfg["classes"].index(label) for label in frame.label])
    groups = frame.patient_id.to_numpy(dtype=str)
    for name, value in [("waveforms", np.stack(crops)), ("y", frame.label.to_numpy(dtype=str)), ("groups", groups)]:
        np.save(target / f"{name}.npy", value)
    (target / "manifest.json").write_text(json.dumps({"fingerprint": "toy", "classes": cfg["classes"], "shape": [81, 3, 65]}))
    paper = {"data": {"raw_dir": str(raw), "leads": ["vx", "vy", "vz"], "sampling_rate": 1000, "before_r": 25, "after_r": 39},
        "signal": {"denoise_wavelet": "bior6.8", "denoise_level": 9}}
    Path(cfg["paper_config"]).write_text(yaml.safe_dump(paper))
    create_fixed_folds(y, groups, cfg["classes"], Path(cfg["folds_path"]), 3, 42)
    return cfg, np.stack(crops), y, groups


def test_paired_bank_preserves_every_original_crop_and_rejects_changed_inputs(tmp_path):
    cfg, vcg, y, groups = fixture(tmp_path)
    path = bank.prepare_bank(cfg)
    ecg = np.load(path / "waveforms.npy")
    assert ecg.shape == (81, 12, 65)
    source = wfdb.rdrecord(str(tmp_path / "raw/r00"), channels=list(range(12)))
    expected = bank.signal.denoise_vcg(source.p_signal, "bior6.8", 9)[75:140].T.astype(np.float32)
    np.testing.assert_array_equal(ecg[0], expected)
    assert bank.prepare_bank(cfg) == path
    summary = json.loads((Path(cfg["output_dir"]) / "bank_audit/summary.json").read_text())
    assert summary["all_frank_crops_exactly_reproduced"] and summary["n_beats"] == 81
    ecg[0, 0, 0] += 1
    np.save(path / "waveforms.npy", ecg)
    with pytest.raises(ValueError, match="changed"):
        bank.prepare_bank(cfg)


def test_zero_threshold_guard_preserves_flat_lead_and_matches_normal_denoising():
    values = np.random.default_rng(55).normal(0, .2, (10000, 12))
    safe, zero = bank.denoise_ecg(values, "bior6.8", 9)
    np.testing.assert_array_equal(safe, bank.signal.denoise_vcg(values, "bior6.8", 9))
    assert zero == []
    values[:, 6] = 0
    safe, zero = bank.denoise_ecg(values, "bior6.8", 9)
    assert np.isfinite(safe).all() and np.all(safe[:, 6] == 0) and "v1" in zero


def test_bank_checks_header_labels_units_and_original_frank_alignment(tmp_path, monkeypatch):
    cfg, _, _, _ = fixture(tmp_path)
    paper = yaml.safe_load(Path(cfg["paper_config"]).read_text())
    rows = pd.read_csv(cfg["beats_path"]).iloc[:3]
    reference = Path(cfg["bank_dir"]) / "waveforms.npy"
    with pytest.raises(ValueError, match="label disagreement"):
        bank._record("r00", rows.assign(label="HC"), paper, reference)
    actual = wfdb.rdrecord(str(tmp_path / "raw/r00"))
    actual.units[0] = "uV"
    monkeypatch.setattr(bank.wfdb, "rdrecord", lambda *a, **k: actual)
    with pytest.raises(ValueError, match="in mV"):
        bank._record("r00", rows, paper, reference)
    actual.units[0] = "mV"
    actual.p_signal[:, 12] += 1
    with pytest.raises(ValueError, match="exactly reproduce"):
        bank._record("r00", rows, paper, reference)


def test_models_separate_information_and_match_dual_capacity(tmp_path):
    cfg, _, _, _ = fixture(tmp_path)
    x, e = torch.randn(2, 3, 65), torch.randn(2, 12, 65)
    models = {arm: study.initial_model(cfg, 42, arm).eval() for arm in study.ARMS}
    dual = [models[arm] for arm in study.DUAL]
    assert len({initial_state_hash(model) for model in dual}) == 1
    assert len({sum(p.numel() for p in model.parameters()) for model in dual}) == 1
    for arm in ["vcg", "vcg_twin"]:
        torch.testing.assert_close(models[arm](x, e), models[arm](x, e * 100 + 20), atol=0, rtol=0)
    torch.testing.assert_close(models["ecg"](x, e), models["ecg"](x * 100, e), atol=0, rtol=0)
    assert not torch.equal(models["joint_bilinear"](x, e), models["joint_bilinear"](x, e * 10))
    a, b = torch.randn(2, 64), torch.randn(2, 64)
    model = models["joint_bilinear"]
    torch.testing.assert_close(model.interaction(a, b), torch.tanh(model.factor_vcg(a) * model.factor_ecg(b)))
    assert not torch.equal(models["joint_additive"](x, e), models["joint_bilinear"](x, e))
    for model in models.values():
        model(x, e).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_vcg_matches_frozen_trainer_and_all_arms_share_rows_and_training_moments(tmp_path):
    cfg, vcg, y, groups = fixture(tmp_path)
    ecg = np.random.default_rng(55).normal(size=(81, 12, 65)).astype(np.float32)
    train, valid = np.arange(54), np.arange(54, 81)
    models, fits = {}, {}
    for arm in study.ARMS:
        model, norms, fit = study.fit(vcg, ecg, y, groups, train, valid, 2, cfg, 42, arm)
        study.validate_fit(fit, cfg, 42, arm, train, valid, groups)
        models[arm], fits[arm] = model, fit
        study.check_pair(fits["vcg"], fit)
    for arm in study.DUAL:
        study.check_pair(fits["vcg_twin"], fits[arm])
    original, norm, audit = fit_waves(vcg, y, groups, train, valid, 2, cfg, 42)
    assert audit["history"] == fits["vcg"]["history"]
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, models["vcg"].state_dict()[key], atol=0, rtol=0)
    assert norm.wave_mean_.tolist() == fits["vcg"]["normalization"]["vcg"]["wave_mean_"]
    weights = patient_balanced_weights(y[train], groups[train])
    expected = study.LeadNormalizer().fit(ecg, train, weights).audit()
    contaminated = ecg.copy()
    contaminated[valid] += 999
    assert study.LeadNormalizer().fit(contaminated, train, weights).audit() == expected
    class Forbidden:
        def __getitem__(self, key):
            pytest.fail("VCG-only arms must never index ECG")
    norms = {"vcg": study.LeadNormalizer().fit(vcg, train, weights), "ecg": None}
    study.PairedRows(vcg, Forbidden(), y, train, norms)[0]
    with pytest.raises(ValueError, match="overlap"):
        study.fit(vcg, ecg, y, groups, train, train, 1, cfg, 42, "joint_bilinear")


def test_all_folds_seeds_and_arms_report_resume_and_tamper_rejection(tmp_path, monkeypatch):
    from mi_localization import paired_report as report
    cfg, _, _, _ = fixture(tmp_path)
    calls = []
    real_fit = study.fit
    def tracked(vcg, ecg, y, groups, train, valid, *args, **kwargs):
        calls.append((set(groups[train]), None if valid is None else set(groups[valid])))
        return real_fit(vcg, ecg, y, groups, train, valid, *args, **kwargs)
    monkeypatch.setattr(study, "fit", tracked)
    table = study.run_study(cfg)
    assert len(calls) == 135 and table.complete.all()
    frozen = pd.read_csv(cfg["folds_path"])
    for i, (train, valid) in enumerate(calls):
        fold = (i // 15) % 3 + 1
        test = set(frozen.loc[frozen.fold == fold, "patient_id"])
        assert not train & test
        if valid is not None:
            assert not train & valid and not valid & test
    root = Path(cfg["output_dir"]) / "complete"
    result = report.summarize_study(cfg, draws=100)
    assert result.exists()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["unique_fit_count"] == 135 and len(summary["prespecified_contrasts"]) == 4
    monkeypatch.setattr(study, "fit", lambda *a, **k: pytest.fail("Completed fits must resume without retraining"))
    pd.testing.assert_frame_equal(table, study.run_study(cfg))
    with pytest.raises(ValueError, match="changed"):
        study.run_study(dict(cfg, fusion_rank=5))
    path = root / "seed_42/fold_01_joint_bilinear.json"
    audit = json.loads(path.read_text())
    audit["final_fit"]["initial_state_sha256"] = "0" * 64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="initialization mismatch"):
        report.summarize_study(cfg, draws=100)


def test_partial_runs_stay_separate_and_invalid_protocols_fail(tmp_path, monkeypatch):
    cfg, _, _, _ = fixture(tmp_path)
    cfg["training_seeds"] = [42]
    table = study.run_study(cfg, outer_folds=[1])
    assert not table.complete.any()
    assert not (Path(cfg["output_dir"]) / "complete").exists()
    assert (Path(cfg["output_dir"]) / "partial_folds_1/study_manifest.json").exists()
    with pytest.raises(ValueError, match="all five"):
        study.validate_config(dict(cfg, arms=["vcg", "joint_bilinear"]))
    with pytest.raises(ValueError, match="Invalid positive"):
        study.validate_config(dict(cfg, fusion_rank=0))
