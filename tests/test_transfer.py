import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mi_localization import transfer as study, transfer_report as report
from mi_localization.contrastive import initial_state_hash
from mi_localization.neural import _hash
from mi_localization.preprocessing_study import fit_waves
from mi_localization.signal_audit import file_hash
from test_regularization import tiny_bank


def fixture(tmp_path, monkeypatch):
    cfg, waves, y, groups = tiny_bank(tmp_path)
    cfg.update(arms=list(study.ARMS), dropout=.2, weight_decay=.01, learning_rates=[.001, .0001],
               source_config=str(tmp_path / "source.yaml"), overlap_config=str(tmp_path / "overlap.yaml"),
               encoder_path=str(tmp_path / "encoder.pt"))
    source_dir, bank_dir, screen_dir = [tmp_path / name for name in ["pilot", "source_bank", "screen"]]
    for path in [source_dir, bank_dir, screen_dir / "review"]:
        path.mkdir(parents=True)
    source = {"output_dir": str(source_dir), "seed": 42}
    Path(cfg["source_config"]).write_text(yaml.safe_dump(source))
    (bank_dir / "manifest.json").write_text('{"source": "frozen"}')
    pilot = {"config": source, "bank_manifest_sha256": file_hash(bank_dir / "manifest.json")}
    pilot["pilot_fingerprint"] = _hash(pilot)
    (source_dir / "pilot_manifest.json").write_text(json.dumps(pilot))
    encoder = study.initial_model(cfg, 17, "scratch", None).encoder.state_dict()
    torch.save({"encoder_state_dict": encoder, "pilot_fingerprint": pilot["pilot_fingerprint"], "widths": cfg["widths"],
                "pool_bins": cfg["pool_bins"], "normalization": {"wave_mean_": [1000]*3}, "ptb_patient_independence_cleared": False}, cfg["encoder_path"])
    (source_dir / "summary.json").write_text(json.dumps({"complete": True, "pilot_fingerprint": pilot["pilot_fingerprint"],
        "encoder_export_sha256": file_hash(cfg["encoder_path"]), "source_training_patients": 4, "source_validation_patients": 2}))
    screen_cfg = {"output_dir": str(screen_dir), "source_config": cfg["source_config"]}
    Path(cfg["overlap_config"]).write_text(yaml.safe_dump(screen_cfg))
    _, _, _, target_bank = study.load_bank(cfg)
    inventories = {}
    for role in ["source", "target"]:
        path = tmp_path / f"{role}_record"
        Path(str(path) + ".dat").write_bytes(role.encode())
        Path(str(path) + ".hea").write_text(role)
        inventories[role] = [{"path": str(path), "data_sha256": file_hash(str(path) + ".dat"), "header_sha256": file_hash(str(path) + ".hea")}]
    (screen_dir / "review/input_provenance.json").write_text(json.dumps(inventories))
    screen = {"config": screen_cfg, "code_sha256": file_hash(study.overlap_screen.__file__),
        "source_bank_manifest_sha256": file_hash(bank_dir / "manifest.json"), "target_bank": target_bank,
        **{f"{role}_inventory_sha256": _hash(inventories[role]) for role in inventories}}
    screen["screen_fingerprint"] = _hash(screen)
    (screen_dir / "screen_manifest.json").write_text(json.dumps(screen))
    screened = {"screen_fingerprint": screen["screen_fingerprint"], "recording_screen_passed": True,
        "patient_identity_independence_cleared": False, "near_recording_candidate_pairs": 0, "exact_encoded_data_matches": 0,
        "incomplete_active_lead_pairs": 0, "positive_controls": 2, "positive_controls_retrieved_and_confirmed": 2}
    (screen_dir / "summary.json").write_text(json.dumps(screened))
    monkeypatch.setattr(study.pretraining_source, "prepare_bank", lambda config: bank_dir)
    return cfg, waves, y, groups, encoder, screen_dir


def test_encoder_only_loading_and_scratch_trainer_equivalence(tmp_path, monkeypatch):
    cfg, waves, y, groups, encoder, _ = fixture(tmp_path, monkeypatch)
    state, provenance = study.load_source(cfg, study.load_bank(cfg)[3])
    assert provenance["patient_identity_independence_cleared"] is False
    settings = dict(cfg, learning_rate=.001)
    scratch, norm, first = study.fit_transfer(waves, y, groups, np.arange(81), None, 2, settings, 42, "scratch", state)
    original, _, original_audit = fit_waves(waves, y, groups, np.arange(81), None, 2, settings, 42)
    for key, value in scratch.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], atol=0, rtol=0)
    assert first["history"] == original_audit["history"]
    pretrained, second_norm, second = study.fit_transfer(waves, y, groups, np.arange(81), None, 2, settings, 42, "pretrained", state)
    study.check_pair(first, second)
    assert first["initial_encoder_sha256"] != second["initial_encoder_sha256"]
    assert first["initial_state_sha256"] != second["initial_state_sha256"]
    assert first["normalization"] == second["normalization"] and max(abs(v) for v in second["normalization"]["wave_mean_"]) < 1
    expected = initial_state_hash(study.initial_model(cfg, 42, "pretrained", state).encoder)
    study.validate_fit(second, cfg, 42, "pretrained", .001, 81, 27, 0, expected, 2)
    assert any(not torch.equal(value, state[key]) for key, value in pretrained.encoder.state_dict().items())
    # Neutral initialization verifies the custom fine-tuning loop exactly.
    neutral = study.initial_model(cfg, 42, "scratch", None).encoder.state_dict()
    same, _, audit = study.fit_transfer(waves, y, groups, np.arange(81), None, 2, settings, 42, "pretrained", neutral)
    for key, value in scratch.state_dict().items():
        torch.testing.assert_close(value, same.state_dict()[key], atol=0, rtol=0)
    assert first["history"] == audit["history"]


def test_screen_gate_blocks_known_candidates_identity_claims_and_raw_drift(tmp_path, monkeypatch):
    cfg, _, _, _, _, screen_dir = fixture(tmp_path, monkeypatch)
    target = study.load_bank(cfg)[3]
    path = screen_dir / "summary.json"
    original = json.loads(path.read_text())
    for changes in [{"near_recording_candidate_pairs": 1}, {"patient_identity_independence_cleared": True},
                    {"positive_controls_retrieved_and_confirmed": 1}, {"recording_screen_passed": False}]:
        path.write_text(json.dumps(dict(original, **changes)))
        with pytest.raises(ValueError, match="did not pass"):
            study.load_source(cfg, target)
    path.write_text(json.dumps(original))
    (tmp_path / "target_record.dat").write_bytes(b"changed")
    with pytest.raises(ValueError, match="waveform/header changed"):
        study.load_source(cfg, target)


def test_selection_uses_inner_scores_only_and_keeps_original_rate_on_ties():
    cfg = {"learning_rates": [.001, .0001], "inner_splits": 2, "max_epochs": 25}
    fits = {f"rate_{i}": [{"best_validation_macro_f1": .5, "selected_epoch": e, "outer_f1": i} for e in [3, 4]] for i in range(2)}
    selected = study.choose_rate(fits, cfg)
    assert selected["selected"]["learning_rate"] == .001 and selected["selected"]["final_epochs"] == 4
    fits["rate_1"][0]["best_validation_macro_f1"] = .7
    assert study.choose_rate(fits, cfg)["selected"]["learning_rate"] == .0001


def test_full_nested_run_validated_report_resume_and_tamper_guards(tmp_path, monkeypatch):
    cfg, _, _, _, _, _ = fixture(tmp_path, monkeypatch)
    calls = []
    real_fit = study.fit_transfer
    def tracked(waves, y, groups, train, valid, *args, **kwargs):
        calls.append((set(groups[train]), None if valid is None else set(groups[valid])))
        return real_fit(waves, y, groups, train, valid, *args, **kwargs)
    monkeypatch.setattr(study, "fit_transfer", tracked)
    table = study.run_study(cfg)
    assert len(calls) == 90 and table.complete.all()
    frozen = pd.read_csv(cfg["folds_path"])
    for i, (train, valid) in enumerate(calls):
        fold = (i // 10) % 3 + 1
        test = set(frozen.loc[frozen.fold == fold, "patient_id"])
        assert not train & test
        if valid is not None:
            assert not train & valid and not valid & test
    root = Path(cfg["output_dir"]) / "complete"
    result = report.summarize_study(cfg, draws=100)
    assert result.exists()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["unique_target_fit_count"] == 90 and summary["patient_identity_independence_cleared"] is False
    monkeypatch.setattr(study, "fit_transfer", lambda *a, **k: pytest.fail("Completed run must resume without training"))
    pd.testing.assert_frame_equal(table, study.run_study(cfg))
    with pytest.raises(ValueError, match="changed"):
        study.run_study(dict(cfg, patience=2))
    path = root / "seed_42/fold_01_pretrained.json"
    saved = json.loads(path.read_text())
    saved["final_fit"]["initial_classifier_sha256"] = "0" * 64
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="classifier initialization"):
        report.summarize_study(cfg, draws=100)


def test_patient_overlap_and_invalid_rates_fail(tmp_path, monkeypatch):
    cfg, waves, y, groups, encoder, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="overlap"):
        study.fit_transfer(waves, y, groups, np.arange(10), np.arange(10), 1, dict(cfg, learning_rate=.001), 42, "pretrained", encoder)
    with pytest.raises(ValueError, match="two positive"):
        study.validate_config(dict(cfg, learning_rates=[.001, .001]))
