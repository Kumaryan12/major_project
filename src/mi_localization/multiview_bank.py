"""Lazy measured-VCG views preserving raw data and within-record variation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .neural import FoldNormalizer, _hash, _save_json
from .regularization import load_bank
from .signal_audit import file_hash

ARMS = ("vcg", "capacity_control", "raw_denoised", "template_only", "template_residual", "full_tensor")
VIEW_NAMES = ("raw", "denoised", "record_median", "beat_residual", "filter_residual")


def array_hash(value):
    value = np.asarray(value)
    return _hash([str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()])


def load_sources(config):
    denoised, y, groups, target = load_bank(config)
    root = Path(config["raw_bank_dir"])
    raw_manifest = json.loads((root/"manifest.json").read_text())
    if raw_manifest["classes"] != config["classes"] or raw_manifest["shape"] != list(denoised.shape) or raw_manifest["control_fingerprint"] != target["fingerprint"] or raw_manifest["control_waveforms_sha256"] != target["files"]["waveforms.npy"] or raw_manifest["waveforms_sha256"] != file_hash(root/"waveforms.npy"):
        raise ValueError("Raw/denoised source provenance or content changed")
    labels = np.load(Path(config["bank_dir"])/"y.npy")
    if not np.array_equal(np.load(root/"y.npy"), labels) or not np.array_equal(np.load(root/"groups.npy"), groups):
        raise ValueError("Raw/denoised source rows do not align")
    raw = np.load(root/"waveforms.npy", mmap_mode="r")
    if raw.shape != denoised.shape or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("Raw source waveform bank is incomplete/nonfinite")
    return denoised, raw, y, groups, {"denoised": target, "raw_manifest_sha256": file_hash(root/"manifest.json"),
        "raw_waveforms_sha256": raw_manifest["waveforms_sha256"], "raw_fingerprint": raw_manifest["fingerprint"]}


def prepare_bank(config):
    d, raw, y, groups, source = load_sources(config)
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    # No diagnostic label enters median estimation. Membership checks guard
    # against an accidental record identifier shared by different patients.
    assignments = beats[["record", "patient_id", "r_peak"]]
    scope = {"source": source, "record_assignments_sha256": _hash(assignments.to_dict("list")),
        "code_sha256": file_hash(__file__), "numpy": np.__version__, "template": "per-record median of all original denoised beats; no fitted cohort dictionary"}
    fingerprint = _hash(scope)
    root = Path(config["template_bank_dir"])
    root.mkdir(parents=True, exist_ok=True)
    guard = root/"build_manifest.json"
    build = dict(scope, bank_fingerprint=fingerprint)
    if guard.exists() and json.loads(guard.read_text()) != build:
        raise ValueError("Multiview template source/code changed; use a new bank directory")
    _save_json(guard, build)
    manifest_path = root/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["bank_fingerprint"] != fingerprint or any(file_hash(root/name) != digest for name, digest in manifest["files"].items()):
            raise ValueError("Multiview template cache content changed")
    else:
        templates, indices, audit_rows = [], np.full(len(beats), -1, np.int32), []
        errors = []
        for record, part in beats.groupby("record", sort=True):
            if part.patient_id.nunique() != 1 or part.label.nunique() != 1 or not np.all(np.diff(part.sort_values("r_peak").r_peak) > 0):
                raise ValueError("Record membership or original R positions are inconsistent")
            rows = part.index.to_numpy()
            values = np.asarray(d[rows])
            template = np.median(values, axis=0).astype(np.float32)
            indices[rows] = len(templates)
            templates.append(template)
            v, f = values-template, np.asarray(raw[rows])-values
            if not np.allclose(template+v, values, atol=1e-6, rtol=1e-6) or not np.allclose(values+f, raw[rows], atol=1e-6, rtol=1e-6):
                raise ValueError("Residual views do not preserve the source signals")
            errors.append([float(np.max(np.abs(template+v-values))), float(np.max(np.abs(values+f-raw[rows])))])
            audit_rows.append({"record": record, "patient_id": str(part.patient_id.iloc[0]), "beats": len(rows), "template_index": len(templates)-1})
        if np.any(indices < 0):
            raise ValueError("Missing record template assignments")
        np.save(root/"templates.npy", np.stack(templates))
        np.save(root/"record_indices.npy", indices)
        _save_json(manifest_path, {"bank_fingerprint": fingerprint, "source": source,
            "files": {name: file_hash(root/name) for name in ["templates.npy", "record_indices.npy"]},
            "template_shape": list(np.stack(templates).shape), "tensor_shape": [len(d), 5, 3, d.shape[-1]],
            "view_names": list(VIEW_NAMES), "n_patients": len(np.unique(groups)), "n_records": len(templates), "n_beats": len(d),
            "max_denoised_reconstruction_error_mv": float(np.max(errors, axis=0)[0]), "max_raw_reconstruction_error_mv": float(np.max(errors, axis=0)[1])})
        audit = Path(config["output_dir"])/"bank_audit"
        (audit/"review").mkdir(parents=True, exist_ok=True)
        _save_json(audit/"review/record_templates.json", audit_rows)
        _save_json(audit/"summary.json", json.loads(manifest_path.read_text()))
        (audit/"REPORT.md").write_text("# Information-preserving VCG views\n\n"
            f"All {len(np.unique(groups))} patients, {len(templates)} recordings and {len(d):,} original beats retained. "
            "Raw and denoised sources have matching hashes, labels, row order, XYZ leads and original crop dimensions. "
            "Every template is the median of one recording's own denoised beats, never another patient's beats or a label-conditioned dictionary. "
            "All beats of a held-out recording are available for its own unlabeled median: this is offline within-exam context, not streaming inference.\n\n"
            "Five views are raw, denoised, record median, denoised-minus-median and raw-minus-denoised. Both residual reconstructions pass 1e-6 absolute/relative tolerance. "
            "These deterministic views create no new independent patients or guaranteed diagnostic information. Residuals are not labeled noise. "
            "No R redetection, temporal shift, spatial rotation, time warping, quality filtering, per-beat amplitude normalization, delineation or relabeling is used. "
            "All view/lead scaling is deferred to training partitions. Original waveforms are unchanged; views are assembled lazily to avoid duplicating gigabytes. "
            "This stage is per-beat classification with record context; it does not train a chronological sequence encoder or Tucker factor along beat order.\n")
    templates = np.load(root/"templates.npy", mmap_mode="r")
    indices = np.load(root/"record_indices.npy")
    if indices.shape != (len(d),) or np.any(indices < 0) or np.any(indices >= len(templates)) or templates.shape[1:] != d.shape[1:] or not np.isfinite(templates).all():
        raise ValueError("Invalid cached template mapping")
    provenance = {"bank_fingerprint": fingerprint, "manifest_sha256": file_hash(manifest_path), "source": source,
        "template_files": json.loads(manifest_path.read_text())["files"]}
    return d, raw, templates, indices, y, groups, provenance


class ViewBank:
    """Array-like beat × (view*XYZ) × time; view axes remain explicit until fetch."""
    def __init__(self, denoised, raw, templates, record_indices, arm):
        if arm not in ARMS:
            raise ValueError("Unknown VCG view arm")
        self.d, self.raw, self.templates, self.record_indices, self.arm = denoised, raw, templates, record_indices, arm
        self.shape = (len(denoised), 3 if arm == "vcg" else 15, denoised.shape[-1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        d = np.asarray(self.d[rows], dtype=np.float32)
        if self.arm == "vcg":
            return d
        if self.arm == "capacity_control":
            values = [d]*5
        elif self.arm == "raw_denoised":
            values = [np.asarray(self.raw[rows]), d, d, d, d]
        else:
            median = np.asarray(self.templates[self.record_indices[rows]], dtype=np.float32)
            if self.arm == "template_only":
                values = [median]*5
            elif self.arm == "template_residual":
                values = [d, d, median, d-median, d]
            else:
                raw = np.asarray(self.raw[rows], dtype=np.float32)
                values = [raw, d, median, d-median, raw-d]
        tensor = np.stack(values, axis=-3)
        return tensor.reshape(*tensor.shape[:-3], 15, tensor.shape[-1]).astype(np.float32, copy=False)


class ViewNormalizer:
    def __init__(self, mean, scale):
        self.wave_mean_, self.wave_scale_ = np.asarray(mean, np.float32), np.asarray(scale, np.float32)

    def audit(self):
        return {"wave_mean_": self.wave_mean_.tolist(), "wave_scale_": self.wave_scale_.tolist(), "stat_mean_": [], "stat_scale_": []}


def fit_normalization(full, y, groups, train, weights):
    weights = np.asarray(weights, dtype=float)
    normalized = weights/weights.sum()
    mean, second = np.zeros(15), np.zeros(15)
    for start in range(0, len(train), 256):
        x = np.asarray(full[train[start:start+256]], dtype=np.float64)
        w = normalized[start:start+256]
        mean += w@x.mean(2)
        second += w@np.square(x).mean(2)
    scale = np.sqrt(np.maximum(second-mean**2, 1e-12)).astype(np.float32)
    mean = mean.astype(np.float32)
    # Preserve exact frozen baseline moments even when BLAS reduction layouts
    # differ between 3- and 15-channel arrays.
    original = FoldNormalizer().fit(full.d, np.empty((len(full),0),np.float32), train, weights)
    mean[3:6], scale[3:6] = original.wave_mean_, original.wave_scale_
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid multiview training-only moments")
    return mean.reshape(5,3), scale.reshape(5,3)


def select_normalization(mean, scale, arm):
    slots = {"vcg":[1], "capacity_control":[1]*5, "raw_denoised":[0,1,1,1,1], "template_only":[2]*5,
        "template_residual":[1,1,2,3,1], "full_tensor":[0,1,2,3,4]}[arm]
    return ViewNormalizer(np.asarray(mean)[slots].ravel(), np.asarray(scale)[slots].ravel())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiview_vcg.yaml")
    args = parser.parse_args()
    print(prepare_bank(yaml.safe_load(Path(args.config).read_text()))[-1])
