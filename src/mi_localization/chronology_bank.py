"""Original within-record chronology and preceding-RR metadata; no R redetection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .neural import _hash, _save_json
from .regularization import load_bank
from .signal_audit import file_hash


class RecordBank:
    def __init__(self, waves, y, groups, record_rows, offsets, record_patients, rr, rr_valid, provenance):
        self.waves, self.y, self.groups = waves, np.asarray(y), np.asarray(groups)
        self.record_rows, self.offsets = np.asarray(record_rows), np.asarray(offsets)
        self.record_patients = np.asarray(record_patients)
        self.rr, self.rr_valid = np.asarray(rr), np.asarray(rr_valid)
        self.provenance = provenance
        self.ids = np.unique(self.groups)
        self.labels = np.array([self.y[np.flatnonzero(self.groups == p)[0]] for p in self.ids])
        if len(self.y) != len(waves) or len(self.groups) != len(waves) or self.rr.shape != (len(waves),) or self.rr_valid.shape != (len(waves),):
            raise ValueError("Chronology beat dimensions differ")
        if not np.array_equal(np.sort(self.record_rows), np.arange(len(waves))) or len(self.offsets) != len(self.record_patients)+1 or self.offsets[0] != 0 or self.offsets[-1] != len(waves) or np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Chronology records do not partition original beats")
        if not np.isfinite(self.rr).all() or np.any(self.rr[self.rr_valid] <= 0) or np.any(self.rr[~self.rr_valid] != 0):
            raise ValueError("Invalid original preceding-RR intervals")
        for record in range(len(self.record_patients)):
            rows = self.rows(record)
            if not np.all(self.groups[rows] == self.record_patients[record]) or len(np.unique(self.y[rows])) != 1:
                raise ValueError("Record crosses patient/label boundaries")
            if self.rr_valid[rows[0]] or not self.rr_valid[rows[1:]].all():
                raise ValueError("Preceding RR must be missing only at the record start")
        if any(len(np.unique(self.y[self.groups == p])) != 1 for p in self.ids):
            raise ValueError("Patient has inconsistent source labels")

    def rows(self, record):
        return self.record_rows[self.offsets[record]:self.offsets[record+1]]

    def records_for(self, patients):
        patients = np.asarray(patients)
        if not len(patients) or len(np.unique(patients)) != len(patients) or not set(patients).issubset(set(self.ids)):
            raise ValueError("Invalid chronology patient set")
        return np.flatnonzero(np.isin(self.record_patients, patients))

    def beats_for(self, patients):
        records = self.records_for(patients)
        return np.sort(np.concatenate([self.rows(i) for i in records]))


def prepare_bank(config):
    waves, y, groups, source = load_bank(config)
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    if len(beats) != len(waves) or not np.array_equal(beats.patient_id.astype(str).to_numpy(), groups):
        raise ValueError("Chronology metadata differs from original waveform rows")
    scope = {"source": source, "code_sha256": file_hash(__file__), "numpy": np.__version__,
             "timing": "Original cached R positions at 1000 Hz; preceding retained-beat RR only; no absolute time model input",
             "metadata_sha256": _hash(beats[["record", "patient_id", "beat_index", "r_peak"]].to_dict("list"))}
    root = Path(config["sequence_bank_dir"])
    root.mkdir(parents=True, exist_ok=True)
    guard = root/"build_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Chronology source/code changed; use a new sequence_bank_dir")
    _save_json(guard, scope)
    path = root/"metadata.npz"
    if path.exists():
        manifest = json.loads((root/"manifest.json").read_text())
        if manifest["bank_fingerprint"] != _hash(scope) or manifest["metadata_file_sha256"] != file_hash(path):
            raise ValueError("Chronology metadata cache changed")
        with np.load(path) as saved:
            arrays = {key: saved[key] for key in saved.files}
    else:
        rows_by_record, patients = [], []
        rr = np.zeros(len(beats), dtype=np.float32)
        valid = np.zeros(len(beats), dtype=bool)
        lengths, intervals = [], []
        for _, part in beats.groupby("record", sort=True):
            part = part.sort_values("r_peak", kind="stable")
            rows = part.index.to_numpy()
            peaks = part.r_peak.to_numpy(dtype=np.float64)
            if part.patient_id.nunique() != 1 or not np.isfinite(peaks).all() or not np.equal(peaks, np.floor(peaks)).all() or np.any(np.diff(peaks) <= 0) or np.any(np.diff(part.beat_index.to_numpy()) != 1):
                raise ValueError("Original retained-beat chronology is ambiguous/incomplete")
            interval = np.diff(peaks)/1000.
            rr[rows[1:]], valid[rows[1:]] = interval, True
            rows_by_record.append(rows);patients.append(str(part.patient_id.iloc[0]));lengths.append(len(rows));intervals.extend(interval.tolist())
        arrays = {"record_rows": np.concatenate(rows_by_record).astype(np.int64),
                  "offsets": np.r_[0, np.cumsum(lengths)].astype(np.int64),
                  "record_patients": np.asarray(patients), "rr": rr, "rr_valid": valid}
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **arrays);temporary.replace(path)
        manifest = {"bank_fingerprint": _hash(scope), "metadata_file_sha256": file_hash(path),
                    "n_patients": len(np.unique(groups)), "n_records": len(lengths), "n_beats": len(waves),
                    "record_beats_min_median_max": [int(min(lengths)), float(np.median(lengths)), int(max(lengths))],
                    "preceding_rr_seconds_min_median_max": [float(min(intervals)), float(np.median(intervals)), float(max(intervals))] if intervals else [None, None, None],
                    "missing_preceding_rr": len(lengths), "sampling_rate": 1000,
                    "scope": "Timing from cached detector positions, not independent clinical R annotations; no interval/record/patient filtering"}
        _save_json(root/"manifest.json", manifest)
    provenance = dict(source=source, metadata_manifest_sha256=file_hash(root/"manifest.json"),
                      metadata_sha256=file_hash(path), bank_fingerprint=_hash(scope))
    result = RecordBank(waves, y, groups, provenance=provenance, **arrays)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chronological_vcg.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    bank = prepare_bank(cfg)
    print(json.dumps({"patients": len(bank.ids), "records": len(bank.record_patients), "beats": len(bank.waves)}))
