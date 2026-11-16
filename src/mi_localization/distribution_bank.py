"""Cache only deterministic per-beat morphology and all-beat record medians."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import scipy

from . import chronology_bank, distribution_features as features
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash


@dataclass
class DistributionBank:
    features: np.ndarray
    template_features: np.ndarray
    record_rows: np.ndarray
    offsets: np.ndarray
    record_patients: np.ndarray
    ids: np.ndarray
    labels: np.ndarray
    provenance: dict

    def rows(self, record):
        return self.record_rows[self.offsets[record]:self.offsets[record+1]]

    def records_for(self, patients):
        patients = np.asarray(patients)
        if not len(patients) or len(np.unique(patients)) != len(patients) or not set(patients).issubset(set(self.ids)):
            raise ValueError("Invalid patient membership")
        return np.flatnonzero(np.isin(self.record_patients, patients))


def prepare(config, verify=False):
    source = chronology_bank.prepare_bank(config)
    root = Path(config["distribution_bank_dir"])
    root.mkdir(parents=True, exist_ok=True)
    scope = dict(source=source.provenance, windows=config["windows"], dct_coefficients=config["dct_coefficients"],
                 features_sha256=file_hash(features.__file__), bank_code_sha256=file_hash(__file__),
                 numpy=np.__version__, scipy=scipy.__version__,
                 rule="All original beats; fixed DCT/sqrt(window length), RMS, mean absolute, signed minimum/maximum; no learned bank state")
    guard, path, median_path = root/"manifest.json", root/"beat_features.npy", root/"median_features.npy"
    if verify and not guard.exists(): raise ValueError("No completed distribution bank to verify")
    if guard.exists():
        manifest = json.loads(guard.read_text())
        if manifest["scope"] != scope or file_hash(path) != manifest["beat_features_sha256"] or file_hash(median_path) != manifest["median_features_sha256"]:
            raise ValueError("Distribution bank source/content changed; use a new namespace")
        values = np.load(path, mmap_mode="r")
        templates = np.load(median_path)
    else:
        shape = (len(source.waves), len(config["windows"]), 3*(config["dct_coefficients"]+4))
        temporary = path.with_suffix(".partial.npy")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=shape)
        for start in range(0, len(source.waves), config["chunk_size"]):
            end = min(start+config["chunk_size"], len(source.waves))
            values[start:end] = features.extract(source.waves[start:end], config)
        values.flush(); del values
        temporary.replace(path); values = np.load(path, mmap_mode="r")
        templates = np.stack([features.extract(np.median(np.asarray(source.waves[source.rows(i)]), axis=0)[None], config)[0]
                              for i in range(len(source.record_patients))])
        temporary = median_path.with_suffix(".partial.npy")
        np.save(temporary, templates); temporary.replace(median_path)
        manifest = dict(scope=scope, beat_features_sha256=file_hash(path), median_features_sha256=file_hash(median_path),
                        n_beats=len(values), n_records=len(templates), n_patients=len(source.ids), shape=list(values.shape),
                        record_membership_sha256=array_hash(source.record_patients))
        _save_json(guard, manifest)
    if (values.shape != (len(source.waves), len(config["windows"]), 3*(config["dct_coefficients"]+4))
            or templates.shape != (len(source.record_patients), *values.shape[1:])
            or not np.isfinite(values).all() or not np.isfinite(templates).all()
            or manifest["record_membership_sha256"] != array_hash(source.record_patients)):
        raise ValueError("Invalid distribution bank dimensions/membership")
    if verify:
        for start in range(0, len(source.waves), config["chunk_size"]):
            end = min(start+config["chunk_size"], len(source.waves))
            if not np.array_equal(features.extract(source.waves[start:end], config), values[start:end]):
                raise ValueError("Original beat feature reconstruction changed")
        for i in range(len(source.record_patients)):
            fresh = features.extract(np.median(np.asarray(source.waves[source.rows(i)]), axis=0)[None], config)[0]
            if not np.array_equal(fresh, templates[i]): raise ValueError("Original median feature reconstruction changed")
    audit = Path(config["output_dir"])/"bank_audit"
    audit.mkdir(parents=True, exist_ok=True)
    _save_json(audit/"summary.json", dict(bank_fingerprint=_hash(manifest), n_patients=len(source.ids),
                                        n_records=len(templates), n_beats=len(values), shape=list(values.shape),
                                        all_beats_reconstructed=verify, source_cohort_unchanged=True))
    return DistributionBank(values, templates, source.record_rows, source.offsets, source.record_patients,
                            source.ids, source.labels, dict(manifest=manifest, manifest_sha256=file_hash(guard)))
