import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import wfdb
import yaml
from scipy.signal import resample_poly

pytest.importorskip("torch")

from mi_localization import pretraining_source as source
from mi_localization.external import kors_transform


def tiny_config(tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/source_pretraining_vcg.yaml").read_text())
    config.update(metadata_path=str(tmp_path / "metadata.csv"), waveform_dir=str(tmp_path / "waveforms"),
                  bank_dir=str(tmp_path / "bank"), audit_dir=str(tmp_path / "audit"), output_dir=str(tmp_path / "pilot"),
                  train_patients=4, validation_patients=2, download_workers=1, conversion_workers=1, review_records_per_role=1)
    rows = []
    for fold in range(1, 11):
        for offset in [1, 2]:
            ecg_id = (fold - 1) * 2 + offset
            rows.append({"ecg_id": ecg_id, "patient_id": float(ecg_id), "strat_fold": fold,
                         "filename_hr": f"records500/00000/{ecg_id:05d}_hr", "scp_codes": "{'NORM': 100}"})
    frame = pd.DataFrame(rows)
    frame.to_csv(config["metadata_path"], index=False)
    return config, frame


def create_records(config, cohort):
    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for row in cohort.itertuples():
        path = Path(config["waveform_dir"]) / row.filename_hr
        path.parent.mkdir(parents=True, exist_ok=True)
        wave = np.random.default_rng(int(row.ecg_id)).normal(scale=0.01, size=(5000, 12))
        wave += np.sin(np.arange(5000)[:, None] * 0.03 + np.arange(12)[None, :]) * 0.2
        wfdb.wrsamp(path.name, 500, units=["mV"] * 12, sig_name=names, p_signal=wave,
                    fmt=["16"] * 12, write_dir=str(path.parent))


def test_source_selection_is_patient_disjoint_deterministic_and_label_free(tmp_path):
    config, metadata = tiny_config(tmp_path)
    selected = source.select_cohort(metadata, config)
    repeated = source.select_cohort(metadata.sample(frac=1, random_state=17).assign(scp_codes="arbitrary altered labels"), config)
    pd.testing.assert_frame_equal(selected, repeated)
    assert selected.patient_id.is_unique and len(selected) == 6
    assert set(selected.loc[selected.role == "train", "strat_fold"]) <= set(range(1, 9))
    assert set(selected.loc[selected.role == "validation", "strat_fold"]) == {9}
    assert not (selected.strat_fold == 10).any()
    bad = metadata.copy()
    bad.loc[2, "patient_id"] = bad.loc[0, "patient_id"]
    with pytest.raises(ValueError, match="cross fold"):
        source.select_cohort(bad, config)
    with pytest.raises(ValueError, match="fold 10"):
        source.validate_config(dict(config, validation_fold=10))
    bad = metadata.copy()
    bad.loc[0, "filename_hr"] = "../../secrets"
    with pytest.raises(ValueError, match="unsafe"):
        source.select_cohort(bad, config)


def test_excluded_fold_is_rejected_before_network_or_waveform_io(tmp_path, monkeypatch):
    config, _ = tiny_config(tmp_path)
    calls = []
    monkeypatch.setattr(source, "download_file", lambda *args: calls.append(args))
    monkeypatch.setattr(source.wfdb, "rdrecord", lambda *args: calls.append(args))
    row = SimpleNamespace(ecg_id=19, patient_id="ptbxl:19", strat_fold=10, role="validation", filename_hr="records500/00000/00019_hr")
    with pytest.raises(ValueError, match="Refusing waveform access"):
        source._download(row, config)
    with pytest.raises(ValueError, match="Refusing waveform access"):
        source.convert_record(row, config)
    assert calls == []


def test_freeze_has_explicit_unresolved_identity_and_refuses_changed_metadata(tmp_path):
    config, metadata = tiny_config(tmp_path)
    selected, scope = source.freeze_cohort(config)
    assert source.freeze_cohort(config)[1] == scope
    summary = json.loads((Path(config["audit_dir"]) / "cohort_summary.json").read_text())
    assert summary["ptb_patient_independence_cleared"] is False
    assert summary["diagnostic_labels_used_for_selection"] is False
    assert summary["excluded_fold_waveforms_selected"] == 0
    metadata.loc[0, "scp_codes"] = "changed unused metadata"
    metadata.to_csv(config["metadata_path"], index=False)
    with pytest.raises(ValueError, match="settings/metadata/code changed"):
        source.freeze_cohort(config)


def test_conversion_matches_existing_kors_and_resampling_without_beat_resizing(tmp_path, monkeypatch):
    config, metadata = tiny_config(tmp_path)
    selected = source.select_cohort(metadata, config)
    create_records(config, selected)
    row = next(selected.itertuples(index=False))
    monkeypatch.setattr(source.signal, "denoise_vcg", lambda values, *args: values)
    monkeypatch.setattr(source.signal, "detect_r_peaks", lambda *args: np.array([1000, 5000, 9000]))
    values, audit = source.convert_record(row, config)
    record = wfdb.rdrecord(str(Path(config["waveform_dir"]) / row.filename_hr))
    expected = resample_poly(kors_transform(record.p_signal, record.sig_name), 2, 1, axis=0)[4750:5401].T.astype(np.float32)
    np.testing.assert_array_equal(values, expected)
    assert audit["r_peak"] == 5000 and audit["complete_detected_crops"] == 3
    assert audit["accepted"] and values.shape == (3, 651)


def test_source_units_fail_loudly_and_nonfinite_signals_are_counted(tmp_path, monkeypatch):
    config, metadata = tiny_config(tmp_path)
    cohort = source.select_cohort(metadata, config)
    create_records(config, cohort)
    row = next(cohort.itertuples(index=False))
    record = wfdb.rdrecord(str(Path(config["waveform_dir"]) / row.filename_hr))
    monkeypatch.setattr(source.wfdb, "rdrecord", lambda *args: record)
    record.units = ["uV"] * 12
    with pytest.raises(ValueError, match="non-mV units"):
        source.convert_record(row, config)
    record.units = ["mV"] * 12
    record.p_signal[0, 0] = np.nan
    values, audit = source.convert_record(row, config)
    assert values is None and audit["reason"] == "nonfinite_ecg" and not audit["accepted"]


def test_bank_reuses_checkpoints_quarantines_duplicates_and_never_reads_fold10(tmp_path, monkeypatch):
    config, metadata = tiny_config(tmp_path)
    cohort = source.select_cohort(metadata, config)
    create_records(config, cohort)
    # Duplicate two training patients' exact source signal bytes.
    train = list(cohort.loc[cohort.role == "train"].itertuples())
    first = Path(config["waveform_dir"]) / f"{train[0].filename_hr}.dat"
    second = Path(config["waveform_dir"]) / f"{train[1].filename_hr}.dat"
    second.write_bytes(first.read_bytes())
    monkeypatch.setattr(source.signal, "denoise_vcg", lambda values, *args: values)
    monkeypatch.setattr(source.signal, "detect_r_peaks", lambda *args: np.array([1000, 5000, 9000]))
    bank = source.prepare_bank(config)
    assert np.load(bank / "waveforms.npy").shape == (4, 3, 651)
    summary = json.loads((Path(config["audit_dir"]) / "bank_summary.json").read_text())
    assert summary["cross_patient_exact_source_duplicate_groups"] == 1
    assert summary["exclusion_counts"]["cross_patient_exact_source_duplicate"] == 2
    assert summary["fold_10_waveforms_accessed"] == 0 and summary["ptb_patient_independence_cleared"] is False
    assert source.prepare_bank(config) == bank
    path = bank / "waveforms.npy"
    waves = np.load(path)
    waves[0, 0, 0] += 1
    np.save(path, waves)
    with pytest.raises(ValueError, match="bank content changed"):
        source.prepare_bank(config)
