"""Label-free ordered trajectory features with matched beat/record aggregation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import multiview_bank, signature_tensor as tensor, tensor_feature_bank as morphology
from .contrastive import _progress
from .neural import _hash, _save_json
from .signal_audit import file_hash


def sensitivity(a, b, config):
    """Relative errors grouped by homogeneous physical units, not across levels."""
    schema = tensor.names(config)["signature3"]
    result = {}
    for degree in (1, 2, 3):
        for spatial in range(1, degree+1):
            selected = [i for i, name in enumerate(schema) if f"/level{degree}/" in name
                        and sum(word != "time" for word in name.split("/")[-1].split("_")) == spatial]
            if not selected:
                continue
            reference, delta = a[selected], b[selected]-a[selected]
            norm = float(np.linalg.norm(reference))
            result[f"level{degree}_mV{spatial}_seconds{degree-spatial}"] = {
                "relative_l2": float(np.linalg.norm(delta)/max(norm, 1e-12)),
                "reference_near_zero": norm < 1e-12,
                "absolute_rmse": float(np.sqrt(np.mean(delta**2)))}
    return result


def prepare_bank(config):
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    # These are raw/label-free bank prerequisites, not completed classifier studies.
    old_root = morphology.prepare_bank(source)
    old_manifest = json.loads((old_root/"manifest.json").read_text())
    with np.load(old_root/"patient_features.npz") as saved:
        ids, labels, baseline = saved["patient_id"], saved["y"], saved["morphology"]
    old_names = json.loads((old_root/"feature_names.json").read_text())["morphology"]
    d, raw, templates, record_index, y, groups, provenance = multiview_bank.prepare_bank(source)
    tensor.validate(config, d.shape[-1])
    if config["sampling_rate"] != 1000 or config["r_index"] != 250 or type(config["chunk_size"]) != int or config["chunk_size"] < 1 or config["jitter_samples"] != 4:
        raise ValueError("Require declared physical clock, chunk size and audit jitter")
    scope = {"source": provenance, "original_feature_manifest_sha256": file_hash(old_root/"manifest.json"),
             "original_feature_manifest": old_manifest, "source_config": source,
             "construction": {key: config[key] for key in ("sampling_rate", "r_index", "windows", "chunk_size", "jitter_samples")},
             "code_sha256": file_hash(__file__), "tensor_code_sha256": file_hash(tensor.__file__),
             "numpy": np.__version__, "aggregation": "beat median/IQR within recording; equal-record coordinate median per patient",
             "coordinates": "physical seconds, signed XYZ mV; piecewise-linear samples; no per-beat normalization"}
    fingerprint = _hash(scope)
    root = Path(config["feature_bank_dir"])
    root.mkdir(parents=True, exist_ok=True)
    guard = root/"build_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Signature source/construction changed; use a new feature_bank_dir")
    _save_json(guard, scope)
    manifest_path = root/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["bank_fingerprint"] != fingerprint or any(file_hash(root/name) != digest for name, digest in manifest["files"].items()):
            raise ValueError("Signature feature bank content changed")
        return root
    beats = pd.read_csv(source["beats_path"])
    beats = beats.loc[beats.label.isin(source["classes"])].reset_index(drop=True)
    if len(beats) != len(d) or not np.array_equal(beats.patient_id.astype(str).to_numpy(), groups):
        raise ValueError("Signature source beat membership changed")
    cache_root = root/"records"
    cache_root.mkdir(exist_ok=True)
    patient_rows, diagnostics = {}, []
    maximum_math = {"endpoint_max_abs_error": 0., "degree2_shuffle_max_abs_error": 0., "degree3_shuffle_max_abs_error": 0.}
    audit_root = Path(config["output_dir"])/"bank_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    for number, (record, part) in enumerate(beats.groupby("record", sort=True), 1):
        rows = part.index.to_numpy()
        patient = str(part.patient_id.iloc[0])
        if part.patient_id.nunique() != 1 or len(np.unique(record_index[rows])) != 1:
            raise ValueError("Mixed recording identities")
        path = cache_root/f"{_hash(record)}.npz"
        if path.exists():
            with np.load(path) as saved:
                if str(saved["fingerprint"].item()) != fingerprint or not np.array_equal(saved["rows"], rows):
                    raise ValueError("Signature record cache membership changed")
                summaries = {family: saved[family] for family in tensor.FAMILIES}
                audit = json.loads(str(saved["audit"].item()))
                digest = str(saved["digest"].item())
            if digest != _hash([multiview_bank.array_hash(summaries[key]) for key in tensor.FAMILIES]+[audit]):
                raise ValueError("Signature record cache content changed")
        else:
            chunks = {key: [] for key in tensor.FAMILIES}
            for start in range(0, len(rows), config["chunk_size"]):
                x = np.asarray(d[rows[start:start+config["chunk_size"]]], dtype=np.float64)
                values = tensor.extract(x, config)
                for key in chunks:
                    chunks[key].append(values[key])
            summaries = {key: morphology.robust_record_summary(np.concatenate(value)) for key, value in chunks.items()}
            median = np.asarray(templates[record_index[rows[0]]], dtype=np.float64)
            raw_median = np.median(np.asarray(raw[rows], dtype=np.float64), axis=0)
            base = tensor.extract(median[None], config)["signature3"][0]
            raw_values = tensor.extract(raw_median[None], config)["signature3"][0]
            shifts = {}
            for shift in (-config["jitter_samples"], config["jitter_samples"]):
                # Edge extension is declared, never wrap the beat or change cached R.
                indices = np.clip(np.arange(median.shape[-1])+shift, 0, median.shape[-1]-1)
                shifted = tensor.extract(median[:, indices][None], config)["signature3"][0]
                shifts[str(shift)] = sensitivity(base, shifted, config)
            clock = (np.arange(median.shape[-1])-config["r_index"])/config["sampling_rate"]
            mathematical = tensor.mathematical_audit(np.concatenate([clock[:, None], median.T], axis=1)[None])
            if max(mathematical.values()) > 1e-8:
                raise ValueError("Real trajectory signature algebra failed")
            audit = {"beats": len(rows), "raw_vs_denoised": sensitivity(base, raw_values, config), "jitter": shifts, "mathematical": mathematical}
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, fingerprint=fingerprint, rows=rows, audit=json.dumps(audit),
                                digest=_hash([multiview_bank.array_hash(summaries[key]) for key in tensor.FAMILIES]+[audit]), **summaries)
            temporary.replace(path)
        for key in maximum_math:
            maximum_math[key] = max(maximum_math[key], audit["mathematical"][key])
        patient_rows.setdefault(patient, []).append(summaries)
        diagnostics.append(audit)
        _progress(audit_root, state="constructing_features", records_complete=number, total_records=len(templates))
        if number % 25 == 0:
            print(f"signature bank: {number}/{len(templates)} recordings", flush=True)
    if set(patient_rows) != set(ids) or any(len(np.unique(y[groups == patient])) != 1 for patient in ids):
        raise ValueError("Signature patient cohort changed")
    if not np.array_equal(labels, np.array([y[np.flatnonzero(groups == patient)[0]] for patient in ids])):
        raise ValueError("Signature patient labels changed")
    feature_names = {"morphology": old_names}
    values = {"morphology": baseline}
    schema = tensor.names(config)
    for key in tensor.FAMILIES:
        feature_names[key] = [f"record_{summary}/{name}" for summary in ("median", "iqr") for name in schema[key]]
        values[key] = np.stack([np.median(np.stack([r[key] for r in patient_rows[patient]]), axis=0) for patient in ids])
        if values[key].shape != (len(ids), len(feature_names[key])) or not np.isfinite(values[key]).all():
            raise ValueError("Invalid patient signature feature bank")
    np.savez_compressed(root/"patient_features.npz", patient_id=ids, y=labels, **values)
    _save_json(root/"feature_names.json", feature_names)
    sensitivity_summary = {}
    variants = {"raw_vs_denoised": lambda row: row["raw_vs_denoised"],
                "minus4_samples": lambda row: row["jitter"]["-4"], "plus4_samples": lambda row: row["jitter"]["4"]}
    for variant, select in variants.items():
        sensitivity_summary[variant] = {}
        for group in select(diagnostics[0]):
            items = [select(row)[group] for row in diagnostics]
            sensitivity_summary[variant][group] = {"median_relative_l2": float(np.median([x["relative_l2"] for x in items])),
                "p95_relative_l2": float(np.quantile([x["relative_l2"] for x in items], .95)),
                "median_absolute_rmse": float(np.median([x["absolute_rmse"] for x in items])),
                "near_zero_reference_records": sum(x["reference_near_zero"] for x in items)}
    manifest = {"bank_fingerprint": fingerprint, "n_patients": len(ids), "n_records": len(diagnostics), "n_beats": len(d),
                "feature_dimensions": {key: x.shape[1] for key, x in values.items()}, "max_mathematical_errors": maximum_math,
                "sensitivity": sensitivity_summary, "files": {name: file_hash(root/name) for name in ("patient_features.npz", "feature_names.json")}}
    _save_json(manifest_path, manifest)
    _save_json(audit_root/"summary.json", manifest)
    lines = ["# Ordered VCG trajectory feature audit", "", "Label-free construction and mathematical/sensitivity diagnostics only; no classifier score.", "",
             f"Retained {len(ids)} patients, {len(diagnostics)} recordings and {len(d):,} original beats. Every denoised beat contributes. Record summaries are beat median/IQR; patients are equal-record coordinate medians.", "",
             "| Family | Patient columns before appending morphology |", "|---|---:|"]
    lines += [f"| {key} | {x.shape[1]} |" for key, x in values.items()]
    lines += ["", "Paths use physical seconds and signed XYZ mV, exact piecewise-linear signatures through degree three, computed using balanced Chen products. Each fixed window includes physical XYZ means to retain offsets lost by increment signatures. All original morphology columns are copied unchanged. No rotation, resampling, quality deletion, new R positions or learned full-cohort basis is introduced.", "",
              "The endpoint-polynomial control has the same column count as degree-three signatures, but replaces the path by its straight endpoint increment: it preserves endpoint polynomial terms and window means while removing interior ordered interactions. The signed-area control retains level-one increments, six time/XYZ pair areas and means. Signature2/3 keep all tensor words, including algebraic redundancies; train-only selection handles variable/redundant columns. Fixed windows are not clinical phases.", "",
              "Sensitivity diagnostics use every recording median, not the classification feature matrix: raw versus denoised medians, and ±4-sample edge-extended shifts of denoised medians. Groups share physical units mV^q seconds^(degree−q). Relative errors use a 1e−12 denominator floor with near-zero counts reported. These are descriptive checks, not clean-signal accuracy, physiological importance or a reason to remove patients. No audit chooses a feature mask or classifier.", ""]
    (audit_root/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _progress(audit_root, state="complete", records_complete=len(diagnostics))
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/signature_vcg.yaml")
    args = parser.parse_args()
    print(prepare_bank(yaml.safe_load(Path(args.config).read_text())))
