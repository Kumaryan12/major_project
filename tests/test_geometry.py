import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mi_localization import geometry_bank as bank


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/geometry_vcg.yaml").read_text())


def test_line_circle_flat_geometry_and_physical_time_velocity():
    fs = 1000
    line = np.column_stack([np.arange(101)/fs, np.zeros((101, 2))])
    values, tensor, peak = bank.phase_features(line, fs, 32)
    data = dict(zip(bank.BASE_NAMES, values))
    assert data["duration_s"] == pytest.approx(.1)
    assert data["path_length"] == pytest.approx(.1)
    assert data["mean_speed"] == pytest.approx(1.)
    assert data["linearity_fraction"] == pytest.approx(1.)
    np.testing.assert_allclose(tensor[3], 1, atol=1e-12)
    assert tensor.shape == (7, 32)
    theta = np.linspace(0, 2*np.pi, 1001)
    circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    circular, _, _ = bank.phase_features(circle, fs, 32)
    assert circular[bank.BASE_NAMES.index("area_z")] == pytest.approx(np.pi, rel=1e-4)
    assert circular[bank.BASE_NAMES.index("planarity_fraction")] == 0
    flat, tensor, _ = bank.phase_features(np.zeros((50, 3)), fs, 32)
    assert np.isfinite(flat).all() and np.isfinite(tensor).all()
    assert flat[bank.BASE_NAMES.index("mean_turn_rad")] == 0
    with pytest.raises(ValueError, match="finite"):
        bank.phase_features(np.full((20, 3), np.nan), fs, 32)


def test_fixed_boundaries_cap_next_r_and_durations_are_preserved():
    cfg = config()
    bounds = bank.fixed_boundaries(1000, 1500, 3000, 1000, cfg)
    np.testing.assert_array_equal(bounds, [940, 1080, 1140, 1420])
    assert not bank.valid_boundaries([940, 1080, 1140, 1600], 1000, 1420, 1000)
    assert not bank.valid_boundaries([940, 990, 1140, 1400], 1000, 1420, 1000)
    x = np.column_stack([np.linspace(0, 1, 3000), np.zeros((3000, 2))])
    values, tensor = bank.beat_features(x, 1000, bounds, 1000, cfg)
    assert values.shape == (78,) and tensor.shape == (3, 7, 32)
    assert values[-1] == pytest.approx(.48)


def test_consensus_uses_frozen_r_positions_and_falls_back_without_exclusion(monkeypatch):
    cfg = config()
    peaks = np.array([500, 1500, 2500])
    x = np.zeros((3500, 3)); x[peaks] = [1, -2, 3]
    calls = []
    def delineate(oriented, rpeaks, sampling_rate, method, check):
        calls.append(rpeaks.copy())
        assert np.median(oriented[peaks]) > 0
        offsets = [-50, 70, 150, 400]
        waves = {key: (peaks+offset).astype(float) for key, offset in zip(bank.MARKERS, offsets)}
        # A missing T-end on beat two forces a complete, visible fallback.
        waves["ECG_T_Offsets"][1] = np.nan
        return None, waves
    monkeypatch.setattr(bank.nk, "ecg_delineate", delineate)
    fixed, adaptive, accepted, valid, audit = bank.estimate_boundaries(x, peaks, 1000, cfg)
    assert accepted.tolist() == [True, False, True]
    assert valid.tolist() == [3, 0, 3] and len(calls) == 3
    for used in calls:
        np.testing.assert_array_equal(used, peaks)
    np.testing.assert_array_equal(adaptive[1], fixed[1])
    assert len(adaptive) == len(peaks) and audit["lead_errors"] == []
    monkeypatch.setattr(bank.nk, "ecg_delineate", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad waveform")))
    fixed, adaptive, accepted, valid, audit = bank.estimate_boundaries(x, peaks, 1000, cfg)
    assert not accepted.any() and len(audit["lead_errors"]) == 3
    np.testing.assert_array_equal(fixed, adaptive)


def test_scale_rotation_translation_and_aggregation_do_not_discard_geometry():
    cfg = config()
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.normal(0, .001, (2000, 3)), axis=0)
    bounds = np.array([440, 580, 640, 1000])
    original, tensor = bank.beat_features(x, 500, bounds, 1000, cfg)
    shifted, _ = bank.beat_features(x + np.array([1, 2, 3]), 500, bounds, 1000, cfg)
    np.testing.assert_allclose(original, shifted, atol=1e-7, rtol=1e-5)
    scaled, _ = bank.beat_features(2*x, 500, bounds, 1000, cfg)
    assert scaled[bank.FEATURE_NAMES.index("qrs_max_magnitude")] == pytest.approx(2*original[bank.FEATURE_NAMES.index("qrs_max_magnitude")])
    assert scaled[bank.FEATURE_NAMES.index("qrs_duration_s")] == original[bank.FEATURE_NAMES.index("qrs_duration_s")]
    rotation = np.diag([-1, 1, -1])
    rotated, _ = bank.beat_features(x@rotation, 500, bounds, 1000, cfg)
    assert rotated[bank.FEATURE_NAMES.index("qrs_peak_x")] == pytest.approx(-original[bank.FEATURE_NAMES.index("qrs_peak_x")])
    ids, values = bank.aggregate_features(np.array([[0, 2], [2, 4], [100, 200]]), np.array(["p1", "p1", "p2"]))
    np.testing.assert_array_equal(ids, ["p1", "p2"])
    np.testing.assert_allclose(values, [[1, 3, 1, 1], [100, 200, 0, 0]])


def test_pinned_neurokit_delineation_runs_on_synthetic_xyz_without_redetecting():
    nk = bank.nk
    x = nk.ecg_simulate(duration=12, sampling_rate=1000, random_state=42)
    _, known = nk.ecg_peaks(x, sampling_rate=1000)
    peaks = np.asarray(known["ECG_R_Peaks"])
    peaks = peaks[(peaks >= 250) & (peaks < len(x)-700)]
    xyz = np.column_stack([x, -.5*x, .7*x])
    fixed, adaptive, accepted, valid, audit = bank.estimate_boundaries(xyz, peaks, 1000, config())
    assert fixed.shape == adaptive.shape == (len(peaks), 4)
    assert np.isfinite(adaptive).all() and len(accepted) == len(peaks)


def geometry_fixture(tmp_path, monkeypatch):
    import wfdb
    from mi_localization.benchmark import create_fixed_folds
    cfg = config()
    cfg.update(classes=["AMI", "IMI", "HC"], outer_splits=3, inner_splits=2, workers=1, review_patients_per_class=1,
        bank_dir=str(tmp_path / "reference"), geometry_bank_dir=str(tmp_path / "geometry"), output_dir=str(tmp_path / "results"),
        paper_config=str(tmp_path / "paper.yaml"), beats_path=str(tmp_path / "beats.csv"), folds_path=str(tmp_path / "folds.csv"))
    raw, ref = tmp_path / "raw", Path(cfg["bank_dir"])
    raw.mkdir(); ref.mkdir()
    rows, waveforms = [], []
    for i in range(18):
        patient, record, label = f"p{i:02}", f"r{i:02}", cfg["classes"][i//6]
        comments = ["Reason for admission: Healthy control"] if label == "HC" else ["Reason for admission: Myocardial infarction", "Acute infarction (localization): "+("anterior" if label == "AMI" else "inferior")]
        x = np.random.default_rng(i).normal(0, .1, (1600, 3))
        wfdb.wrsamp(record, 1000, ["mV"]*3, ["vx", "vy", "vz"], p_signal=x, fmt=["16"]*3, comments=comments, write_dir=str(raw))
        (raw/f"{record}.dat").rename(raw/f"{record}.xyz")
        header = raw/f"{record}.hea"
        header.write_text(header.read_text().replace(".dat", ".xyz"))
        physical = wfdb.rdrecord(str(raw/record)).p_signal
        clean = bank.signal.denoise_vcg(physical, "bior6.8", 9)
        for peak in [300, 700, 1100]:
            rows.append({"patient_id": patient, "record": record, "label": label, "r_peak": peak})
            waveforms.append(clean[peak-250:peak+401].T.astype(np.float32))
    frame = pd.DataFrame(rows)
    frame.to_csv(cfg["beats_path"], index=False)
    groups = frame.patient_id.to_numpy(dtype=str)
    y = np.array([cfg["classes"].index(label) for label in frame.label])
    for name, values in [("waveforms", waveforms), ("groups", groups), ("y", frame.label.to_numpy(dtype=str)),
                         ("statistics", np.random.default_rng(11).normal(size=(54, 192)).astype(np.float32))]:
        np.save(ref/f"{name}.npy", values)
    (ref/"manifest.json").write_text(json.dumps({"fingerprint": "toy", "classes": cfg["classes"], "shape": [54, 3, 651]}))
    paper = {"data": {"raw_dir": str(raw), "leads": ["vx", "vy", "vz"], "sampling_rate": 1000, "before_r": 250, "after_r": 400},
        "signal": {"denoise_wavelet": "bior6.8", "denoise_level": 9}}
    Path(cfg["paper_config"]).write_text(yaml.safe_dump(paper))
    create_fixed_folds(y, groups, cfg["classes"], Path(cfg["folds_path"]), 3, 42)
    def delineate(x, rpeaks, **kwargs):
        return None, {key: rpeaks+offset for key, offset in zip(bank.MARKERS, [-50, 70, 130, 280])}
    monkeypatch.setattr(bank.nk, "ecg_delineate", delineate)
    return cfg


def test_complete_bank_reuses_and_detects_tampering(tmp_path, monkeypatch):
    cfg = geometry_fixture(tmp_path, monkeypatch)
    path = bank.prepare_bank(cfg)
    assert bank.prepare_bank(cfg) == path
    with np.load(path/"patient_features.npz") as saved:
        assert saved["patient_id"].shape == (18,)
        assert saved["adaptive_geometry"].shape == (18, 156)
        assert saved["statistics_geometry"].shape == (18, 540)
    manifest = json.loads((path/"manifest.json").read_text())
    assert manifest["summary"]["n_beats"] == 54 and manifest["summary"]["patient_exclusions"] == 0
    values = np.load(path/"adaptive_tensor.npy")
    assert values.shape == (54, 3, 7, 32)
    values[0, 0, 0, 0] += 1
    np.save(path/"adaptive_tensor.npy", values)
    with pytest.raises(ValueError, match="content changed"):
        bank.prepare_bank(cfg)


def test_record_checkpoint_verifies_payload_and_cached_r_alignment(tmp_path, monkeypatch):
    cfg = geometry_fixture(tmp_path, monkeypatch)
    Path(cfg["geometry_bank_dir"], "records").mkdir(parents=True)
    paper = yaml.safe_load(Path(cfg["paper_config"]).read_text())
    rows = pd.read_csv(cfg["beats_path"]).iloc[:3]
    path = bank._record("r00", rows, paper, cfg, Path(cfg["bank_dir"])/"waveforms.npy", "scope")
    with np.load(path) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload["adaptive_features"][0, 0] += 1
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checkpoint scope changed"):
        bank._record("r00", rows, paper, cfg, Path(cfg["bank_dir"])/"waveforms.npy", "scope")


def test_small_classifier_scaling_is_training_only_and_selection_ignores_outer_scores(tmp_path, monkeypatch):
    from mi_localization import geometry_study as study
    cfg = geometry_fixture(tmp_path, monkeypatch)
    rng = np.random.default_rng(55)
    X = rng.normal(size=(18, 10)).astype(np.float32)
    y = np.tile(np.arange(3), 6)
    ids = np.array([f"p{i:02}" for i in range(18)])
    train, valid = np.arange(12), np.arange(12, 18)
    model, p, audit = study.fit_classifier(X, y, ids, train, valid, "linear", {"C": .01}, cfg, 42, "inner_validation")
    changed = X.copy(); changed[valid] += 999
    other, q, second = study.fit_classifier(changed, y, ids, train, valid, "linear", {"C": .01}, cfg, 42, "inner_validation")
    assert audit == second
    np.testing.assert_array_equal(model.named_steps["model"].coef_, other.named_steps["model"].coef_)
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, other.named_steps["scale"].mean_)
    fits = [{"params": {"C": c}, "fits": [{"validation_macro_f1": .5, "outer_score": i} for _ in range(2)]} for i, c in enumerate(cfg["linear_c"])]
    assert study.choose_candidate(fits, cfg, "linear")["candidate_index"] == 0
    fits[1]["fits"][0]["validation_macro_f1"] = .7
    assert study.choose_candidate(fits, cfg, "linear")["candidate_index"] == 1
    with pytest.raises(ValueError, match="overlap"):
        study.fit_classifier(X, y, ids, train, train, "linear", {"C": .01}, cfg, 42, "inner_validation")


def test_complete_patient_screen_validates_every_fit_and_resumes(tmp_path, monkeypatch):
    from mi_localization import geometry_study as study
    cfg = geometry_fixture(tmp_path, monkeypatch)
    cfg.update(tree_estimators=5)
    calls = []
    original = study.fit_classifier
    def tracked(X, y, ids, train, evaluate, *args, **kwargs):
        calls.append((set(ids[train]), set(ids[evaluate]), args[-1] if args else kwargs["role"]))
        return original(X, y, ids, train, evaluate, *args, **kwargs)
    monkeypatch.setattr(study, "fit_classifier", tracked)
    table = study.run_study(cfg)
    assert len(calls) == 120 and table.complete.all() and len(table) == 8
    frozen = pd.read_csv(cfg["folds_path"])
    for i, (train, evaluate, role) in enumerate(calls):
        fold = (i // 5) % 3 + 1
        test = set(frozen.loc[frozen.fold == fold, "patient_id"])
        assert not train & test and not train & evaluate
        if role == "inner_validation":
            assert not evaluate & test
        else:
            assert evaluate == test
    report = study.summarize_study(cfg, draws=100)
    assert report.exists()
    summary = json.loads((report.parent/"summary.json").read_text())
    assert summary["unique_fit_count"] == 120 and summary["n_patients"] == 18
    monkeypatch.setattr(study, "fit_classifier", lambda *a, **k: pytest.fail("Completed fits must resume"))
    pd.testing.assert_frame_equal(table, study.run_study(cfg))
    path = report.parent/"fold_01_linear_statistics.json"
    audit = json.loads(path.read_text())
    audit["final_fit"]["scaler_mean_sha256"] = "0"*64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="scaling mismatch"):
        study.summarize_study(cfg, draws=100)
