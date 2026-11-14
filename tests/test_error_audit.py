import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from mi_localization import error_audit as audit, regularization
from mi_localization.regularization_report import summarize_study
from test_regularization import tiny_bank, fake_fit


def test_error_counts_are_unique_patients_not_seed_repetitions():
    classes, seeds = ["AMI", "IMI", "HC"], [42, 123, 2026]
    frozen = pd.DataFrame({"patient_id": ["p0", "p1", "p2"], "label": classes, "fold": [1, 2, 3]}).set_index("patient_id")
    frames = {}
    for index, seed in enumerate(seeds):
        pred = ["IMI", "HC" if index == 0 else "IMI", "HC"]
        frame = pd.DataFrame({"patient_id": frozen.index, "truth": classes, "prediction": pred, "fold": [1, 2, 3]}).set_index("patient_id")
        for label in classes:
            frame[f"probability_{label}"] = [0.8 if value == label else 0.1 for value in pred]
        frames[seed] = {arm: frame.copy() for arm in ["baseline", "selected"]}
    result = audit.error_table(frames, frozen, classes, seeds)
    assert result.selected_error_count.tolist() == [3, 1, 0]
    assert result.selected_persistent_error.tolist() == [True, False, False]
    assert len(result) == 3
    frames[42]["selected"].loc["p0", "fold"] = 2
    with pytest.raises(ValueError, match="do not align"):
        audit.error_table(frames, frozen, classes, seeds)


def test_source_and_signal_audit_does_not_relabel_a_header_disagreement(tmp_path):
    raw = tmp_path / "raw/p0"
    raw.mkdir(parents=True)
    (raw / "r.hea").write_text("# Reason for admission: Myocardial infarction\n# Acute infarction (localization): inferior\n# Former infarction (localization): anterior")
    (raw / "r.xyz").write_bytes(b"fake signal")
    beats = pd.DataFrame({"patient_id": ["p0"] * 3, "record": ["p0/r"] * 3, "label": ["AMI"] * 3, "r_peak": [50, 100, 150]})
    prior = pd.DataFrame([{ "patient_id": "p0", "record": "p0/r", "label": "AMI", "n_beats": 3, "duration_seconds": 0.2,
        "n_intervals": 2, "cached_intervals_below_300ms": 2, "cached_intervals_above_2000ms": 0, "next_cached_peak_inside_crop": 0}])
    records, _ = audit.source_records(beats, {"data": {"raw_dir": str(tmp_path / "raw")}}, prior)
    assert records.label.tolist() == ["AMI"] and records.header_label.tolist() == ["IMI"]
    assert records.header_label_disagreement.tolist() == [True] and records.former_location_reported.tolist() == [True]
    waves = np.random.default_rng(8).normal(size=(3, 3, 65))
    diagnostics = audit.patient_diagnostics(beats, records, waves, waves)
    assert diagnostics.loc["p0", "header_label_disagreements"] == 1
    assert diagnostics.loc["p0", "median_relative_denoising_change"] == 0
    assert beats.label.tolist() == ["AMI"] * 3


def test_complete_audit_packet_and_provenance_guard(tmp_path, monkeypatch):
    source, waves, y, groups = tiny_bank(tmp_path)
    paper = yaml.safe_load(Path(source["paper_config"]).read_text())
    paper["data"]["raw_dir"] = str(tmp_path / "records")
    Path(source["paper_config"]).write_text(yaml.safe_dump(paper))
    beats = pd.read_csv(source["beats_path"])
    rows = []
    for patient, part in beats.groupby("patient_id"):
        root = Path(paper["data"]["raw_dir"]) / patient
        root.mkdir(parents=True)
        label = part.label.iloc[0]
        text = "# Reason for admission: Healthy control" if label == "HC" else (
            "# Reason for admission: Myocardial infarction\n# Acute infarction (localization): " + ("anterior" if label == "AMI" else "inferior"))
        (root / "r.hea").write_text(text + "\n# Former infarction (localization): no")
        (root / "r.xyz").write_bytes(patient.encode())
        rows.append({"patient_id": patient, "record": f"{patient}/r", "label": label, "n_beats": 3, "duration_seconds": 0.2,
            "n_intervals": 2, "cached_intervals_below_300ms": 2, "cached_intervals_above_2000ms": 0, "next_cached_peak_inside_crop": 0})
    monkeypatch.setattr(regularization, "fit_candidate", fake_fit([]))
    regularization.run_study(source)
    summarize_study(source, draws=100)
    source_path = tmp_path / "regularization.yaml"
    source_path.write_text(yaml.safe_dump(source))
    signal = {"classes": source["classes"], "beats_path": source["beats_path"], "denoised_bank_dir": source["bank_dir"], "audit_dir": str(tmp_path / "signals")}
    signal_path = tmp_path / "signal.yaml"
    signal_path.write_text(yaml.safe_dump(signal))
    record_path = Path(signal["audit_dir"]) / "review"
    record_path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(record_path / "record_audit.csv", index=False)
    raw = tmp_path / "raw_bank"
    raw.mkdir()
    np.save(raw / "waveforms.npy", waves / 0.9)
    (raw / "manifest.json").write_text(json.dumps({"fingerprint": "tiny-test"}))
    monkeypatch.setattr(audit, "build_raw_bank_and_audit", lambda config: raw)
    # The plotting template is exercised once below; avoid rendering every case.
    plot = audit.review_plots
    def one_case(table, *args):
        copy = table.copy()
        keep = copy.index[copy.selected_persistent_error][:1]
        copy["selected_persistent_error"] = copy.index.isin(keep)
        return plot(copy, *args)
    monkeypatch.setattr(audit, "review_plots", one_case)
    config = {"source_config": str(source_path), "signal_audit_config": str(signal_path), "output_dir": str(tmp_path / "error_audit"), "review_records_per_patient": 3}
    report = audit.run_audit(config)
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["n_patients"] == 27 and summary["n_beats"] == 81 and summary["n_records"] == 27
    assert sum(summary["selected_error_count_distribution"].values()) == 27
    assert summary["independent_expert_adjudication"] is False
    assert summary["cross_patient_exact_waveform_duplicate_groups"] == 0
    assert (report.parent / "review/patient_error_audit.csv").exists()
    assert (report.parent / "review/INDEX.md").exists()
    form = pd.read_csv(report.parent / "review/review_form.csv")
    assert form.adjudication.isna().all()
    assert (report.parent / "confusion_selected.png").exists()
    assert audit.run_audit(config) == report
    path = Path(source["output_dir"]) / "complete/seed_42/patient_predictions_selected.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "probability_AMI"] += 1e-7
    frame.loc[0, "probability_IMI"] -= 1e-7
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inputs/settings changed"):
        audit.run_audit(config)
