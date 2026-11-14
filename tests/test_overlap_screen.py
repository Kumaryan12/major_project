import json
from pathlib import Path

import numpy as np
import pytest
import wfdb
import yaml

from mi_localization import overlap_screen as screen


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/source_target_overlap.yaml").read_text())


def test_continuous_copy_is_found_at_correct_lag_despite_gain_offset_noise():
    cfg = config()
    rng = np.random.default_rng(17)
    target = rng.normal(0, .2, (8000, 8))
    source = target[1234:3734] * 1.03 + np.arange(8)[None] * .02
    source += rng.normal(0, .0001, source.shape)
    found = screen.CorrelationTarget(target, 2000).compare(source, cfg)
    assert found["candidate"] and found["target_query_start"] == 1484
    assert found["full_source_available"] and found["full_minimum_correlation"] > .999
    assert not found["exact_full_segment_equal"]
    exact = screen.CorrelationTarget(target, 2000).compare(target[1234:3734], cfg)
    assert exact["exact_full_segment_equal"]


def test_independent_signals_and_flat_leads_do_not_pass():
    cfg = config()
    rng = np.random.default_rng(18)
    target = rng.normal(size=(4000, 8))
    assert not screen.CorrelationTarget(target, 2000).compare(rng.normal(size=(2500, 8)), cfg)["candidate"]
    source = np.ones((2500, 8))
    source[:, :5] = target[:2500, :5]
    found = screen.CorrelationTarget(target, 2000).compare(source, cfg)
    assert not found["candidate"] and found["active_leads"] == 5
    with pytest.raises(ValueError, match="longer than"):
        screen.CorrelationTarget(target[:1000], 2000)


def test_retrieval_includes_missing_template_fallbacks_and_known_record():
    rng = np.random.default_rng(42)
    target = rng.normal(size=(12, 4, 528)).astype(np.float32)
    target /= np.linalg.norm(target, axis=2, keepdims=True)
    selected = screen.rank_candidates(target[[7, 5]], target, [True, False], [True] * 11 + [False], 2)
    assert 7 in selected[0] and 11 in selected[0]
    np.testing.assert_array_equal(selected[1], np.arange(12))


def test_common_lead_cache_orders_channels_and_detects_tampering(tmp_path, monkeypatch):
    cfg = dict(config(), cache_dir=str(tmp_path))
    names = list(reversed([lead.upper() for lead in screen.ECG_LEADS]))
    wave = np.random.default_rng(20).normal(0, .1, (5000, 8))
    wfdb.wrsamp("record", 500, ["mV"] * 8, names, p_signal=wave, fmt=["16"] * 8, write_dir=str(tmp_path))
    monkeypatch.setattr(screen, "templates", lambda ecg, cfg: (np.zeros((4, 528), np.float32), False))
    item = {"id": "source", "path": str(tmp_path / "record"), "sampling_rate": 500}
    path, _, available = screen._cache_record(item, cfg, "scope")
    assert not available
    with np.load(path) as saved:
        ecg, features, metadata = saved["ecg"], saved["features"], saved["metadata"]
    from scipy.signal import resample_poly
    expected = resample_poly(wfdb.rdrecord(item["path"]).p_signal[:, ::-1], 1, 2, axis=0).astype(np.float32)
    np.testing.assert_array_equal(ecg, expected)
    assert screen._cache_record(item, cfg, "scope")[0] == path
    with pytest.raises(ValueError, match="changed"):
        screen._cache_record(item, cfg, "other scope")
    ecg[0, 0] += 1
    np.savez_compressed(path, ecg=ecg, features=features, metadata=metadata)
    with pytest.raises(ValueError, match="changed"):
        screen._cache_record(item, cfg, "scope")


def test_controls_exercise_retrieval_and_continuous_confirmation(tmp_path, monkeypatch):
    cfg = dict(config(), positive_controls=3)
    rng = np.random.default_rng(44)
    paths, features = [], []
    # A stable feature isolates the retrieval/continuous-confirmation plumbing.
    def feature(ecg, cfg):
        return np.tile(ecg[:66].T.ravel(), (4, 1)).astype(np.float32), True
    monkeypatch.setattr(screen, "templates", feature)
    for i in range(3):
        ecg = rng.normal(size=(2500, 8))
        path = tmp_path / f"{i}.npz"
        np.savez(path, ecg=ecg)
        paths.append(str(path))
        features.append(feature(ecg, cfg)[0])
    count, rows = screen._controls(paths, np.stack(features), np.ones(3, bool), cfg)
    assert count == 8 and all(row["retrieved_known_record"] and row["confirmed_copied_segment"] for row in rows)


def test_thresholds_are_validated():
    with pytest.raises(ValueError, match="thresholds"):
        screen.validate_config(dict(config(), minimum_active_leads=2))
    with pytest.raises(ValueError, match="Freeze"):
        screen.validate_config(dict(config(), query_seconds=1))


def test_end_to_end_screen_flags_resampled_copy_and_keeps_identity_unresolved(tmp_path, monkeypatch):
    import pandas as pd
    from scipy.signal import resample_poly
    from test_pretraining_source import tiny_config, create_records
    cfg = dict(config(), cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "screen"), workers=1)
    source, metadata = tiny_config(tmp_path)
    selected = screen.pretraining_source.select_cohort(metadata, source)
    create_records(source, selected)
    review = Path(source["audit_dir"]) / "review"
    review.mkdir(parents=True)
    selected.to_csv(review / "selected_cohort.csv", index=False)
    selected.to_csv(review / "retained_records.csv", index=False)
    bank = tmp_path / "source_bank"
    bank.mkdir()
    (bank / "manifest.json").write_text('{"fixture":true}')
    monkeypatch.setattr(screen.pretraining_source, "prepare_bank", lambda config: bank)
    raw = tmp_path / "target"
    raw.mkdir()
    source_path = str(Path(source["waveform_dir"]) / selected.filename_hr.iloc[0])
    record = wfdb.rdrecord(source_path)
    copied = resample_poly(record.p_signal, 2, 1, axis=0)
    wfdb.wrsamp("r1", 1000, ["mV"] * 12, record.sig_name, p_signal=copied, fmt=["16"]*12, write_dir=str(raw))
    wfdb.wrsamp("r2", 1000, ["mV"] * 12, record.sig_name, p_signal=np.random.default_rng(92).normal(0, .1, copied.shape), fmt=["16"]*12, write_dir=str(raw))
    paper_path = tmp_path / "target_paper.yaml"
    paper_path.write_text(yaml.safe_dump({"data": {"raw_dir": str(raw)}}))
    beats_path = tmp_path / "target_beats.csv"
    pd.DataFrame({"record": ["r1", "r2"], "patient_id": ["p1", "p2"], "label": ["AMI", "HC"]}).to_csv(beats_path, index=False)
    target = {"classes": ["AMI", "HC"], "paper_config": str(paper_path), "beats_path": str(beats_path)}
    for key, value in [("source_config", source), ("target_config", target)]:
        cfg[key] = str(tmp_path / f"{key}.yaml")
        Path(cfg[key]).write_text(yaml.safe_dump(value))
    monkeypatch.setattr(screen, "load_bank", lambda config: (None, np.arange(2), np.array(["p1", "p2"]), {"fixture": True}))
    destination = screen.run_screen(cfg)
    result = json.loads((destination.parent / "summary.json").read_text())
    assert result["near_recording_candidate_pairs"] >= 1 and result["recording_screen_passed"] is False
    assert result["patient_identity_independence_cleared"] is False
    assert result["positive_controls_retrieved_and_confirmed"] == 2
    assert screen.run_screen(cfg) == destination
    with pytest.raises(ValueError, match="changed"):
        screen.run_screen(dict(cfg, mean_correlation_threshold=.996))
