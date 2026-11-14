"""Non-adjudicated signal audit and immutable-row raw XYZ cache construction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from .data import read_vcg
from .labels import label_from_comments
from .neural import _hash, _save_json
from .signal import denoise_vcg


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signal_metrics(raw, denoised, edge_samples=50):
    """Descriptive changes; no labels of artifact, damage or true wave boundaries."""
    raw, denoised = np.asarray(raw, dtype=float), np.asarray(denoised, dtype=float)
    if raw.ndim != 3 or raw.shape != denoised.shape or raw.shape[1] != 3 or not np.isfinite(raw).all() or not np.isfinite(denoised).all():
        raise ValueError("Expected finite aligned beat x XYZ x time arrays")
    rc, dc = raw - raw.mean(2, keepdims=True), denoised - denoised.mean(2, keepdims=True)
    rn = np.linalg.norm(rc.reshape(len(raw), -1), axis=1)
    dn = np.linalg.norm(dc.reshape(len(raw), -1), axis=1)
    energy = np.maximum(rn ** 2, 1e-24)
    derivative_raw = np.linalg.norm(np.diff(raw, axis=2).reshape(len(raw), -1), axis=1)
    derivative_denoised = np.linalg.norm(np.diff(denoised, axis=2).reshape(len(raw), -1), axis=1)
    edge = min(edge_samples, raw.shape[2])
    return {
        "centered_relative_l2_change": np.linalg.norm((dc - rc).reshape(len(raw), -1), axis=1) / np.maximum(rn, 1e-12),
        "centered_correlation": (rc * dc).sum((1, 2)) / np.maximum(rn * dn, 1e-24),
        "centered_rms_ratio_denoised_over_raw": dn / np.maximum(rn, 1e-12),
        "derivative_rms_ratio_denoised_over_raw": derivative_denoised / np.maximum(derivative_raw, 1e-12),
        "mean_vector_change": np.linalg.norm(denoised.mean(2) - raw.mean(2), axis=1),
        "raw_left_edge_energy_fraction": np.square(rc[:, :, :edge]).sum((1, 2)) / energy,
        "raw_right_edge_energy_fraction": np.square(rc[:, :, -edge:]).sum((1, 2)) / energy,
        "raw_flat_beat": (rn < 1e-12).astype(float),
        "denoised_flat_beat": (dn < 1e-12).astype(float),
    }


def review_selection(beats, classes, per_class, seed):
    """Class-balanced, label-only sampling; never use model errors or scores."""
    rng, selected = np.random.default_rng(seed), {}
    for label in classes:
        patients = sorted(beats.loc[beats.label == label, "patient_id"].unique())
        for patient in rng.choice(patients, min(per_class, len(patients)), replace=False):
            rows = beats.loc[beats.patient_id == patient]
            record = sorted(rows.record.unique())[0]
            peaks = np.sort(rows.loc[rows.record == record, "r_peak"].astype(int).to_numpy())
            selected[record] = {"patient_id": str(patient), "label": label, "peak": int(peaks[len(peaks) // 2])}
    return selected


def _record(record, rows, paper, control_path, selected):
    data, signal = paper["data"], paper["signal"]
    if rows.patient_id.nunique() != 1 or rows.label.nunique() != 1:
        raise ValueError("A recording has inconsistent cached patient/label values")
    header = (Path(data["raw_dir"]) / f"{record}.hea").read_text(encoding="latin1")
    if label_from_comments(header) != str(rows.label.iloc[0]):
        raise ValueError(f"Header/cached label disagreement in {record}; do not silently relabel")
    raw, fs = read_vcg(Path(data["raw_dir"]), record, data["leads"])
    raw = np.asarray(raw)
    if fs != data["sampling_rate"] or raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError("Invalid raw measured XYZ recording")
    peaks = rows.r_peak.astype(int).to_numpy()
    if len(np.unique(peaks)) != len(peaks):
        raise ValueError("Duplicate cached R position within a recording")
    before, after = int(data["before_r"]), int(data["after_r"])
    if np.any(peaks < before) or np.any(peaks + after >= len(raw)):
        raise ValueError("Cached beat has an incomplete crop")
    clean = denoise_vcg(raw, signal["denoise_wavelet"], signal["denoise_level"])
    raw_beats = np.stack([raw[peak - before:peak + after + 1].T for peak in peaks]).astype(np.float32)
    clean_beats = np.stack([clean[peak - before:peak + after + 1].T for peak in peaks]).astype(np.float32)
    reference = np.load(control_path, mmap_mode="r")
    if not np.array_equal(clean_beats, reference[rows.index.to_numpy()]):
        raise ValueError("Recomputed denoised beats do not exactly match the retained cache")
    ordered = np.sort(peaks)
    rr_ms = np.diff(ordered) * 1000 / fs
    next_inside = int((np.diff(ordered) <= after).sum())
    metrics = pd.DataFrame(signal_metrics(raw_beats, clean_beats))
    metrics["label"], metrics["patient_id"], metrics["record"] = rows.label.iloc[0], rows.patient_id.iloc[0], record
    record_metrics = {"record": record, "patient_id": str(rows.patient_id.iloc[0]), "label": str(rows.label.iloc[0]),
        "n_beats": len(peaks), "duration_seconds": len(raw) / fs, "n_intervals": len(rr_ms),
        "cached_intervals_below_300ms": int((rr_ms < 300).sum()), "cached_intervals_above_2000ms": int((rr_ms > 2000).sum()),
        "next_cached_peak_inside_crop": next_inside,
        "median_cached_interval_ms": float(np.median(rr_ms)) if len(rr_ms) else None}
    review = None
    if selected is not None:
        peak = selected["peak"]
        left, right = max(0, peak - int(0.45 * fs)), min(len(raw), peak + int(0.65 * fs) + 1)
        review = {**selected, "record": record, "time_ms": (np.arange(left, right) - peak) * 1000 / fs,
                  "raw": raw[left:right], "denoised": clean[left:right],
                  "before_ms": before * 1000 / fs, "after_ms": after * 1000 / fs}
    return rows.index.to_numpy(), raw_beats, metrics, record_metrics, review


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite signal diagnostic")
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95))}


def _review_plots(examples, classes, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for label in classes:
        cases = sorted([item for item in examples if item["label"] == label], key=lambda item: item["patient_id"])
        if not cases:
            continue
        figure, axes = plt.subplots(3, len(cases), figsize=(6 * len(cases), 8), squeeze=False, sharex=True)
        for column, case in enumerate(cases):
            for lead in range(3):
                axis = axes[lead, column]
                axis.plot(case["time_ms"], case["raw"][:, lead], color="0.65", linewidth=1, label="Raw")
                axis.plot(case["time_ms"], case["denoised"][:, lead], color="#24578c", linewidth=1, label="Denoised")
                axis.axvspan(-case["before_ms"], case["after_ms"], color="0.85", alpha=0.35, label="Training crop")
                axis.axvline(0, color="#a53232", linestyle="--", linewidth=0.8, label="Cached R")
                axis.set_ylabel(f"{'XYZ'[lead]} (physical units)")
                axis.grid(alpha=0.2)
                if lead == 0:
                    axis.set_title(f"{label}: {case['patient_id']}\n{case['record']}; R={case['peak']}", fontsize=9)
                if lead == 2:
                    axis.set_xlabel("Time from cached R position (ms)")
        axes[0, 0].legend(fontsize=7, loc="best")
        figure.suptitle("Non-adjudicated signal review: no confirmed R or wave-boundary annotations", fontsize=11)
        figure.tight_layout()
        figure.savefig(directory / f"review_{label}.png", dpi=150)
        plt.close(figure)


def build_raw_bank_and_audit(config):
    with Path(config["paper_config"]).open() as stream:
        paper = yaml.safe_load(stream)
    if paper["data"]["leads"] != ["vx", "vy", "vz"]:
        raise ValueError("Audit is restricted to measured Frank XYZ")
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    control = Path(config["denoised_bank_dir"])
    control_manifest = json.loads((control / "manifest.json").read_text())
    if beats.empty or control_manifest["classes"] != config["classes"]:
        raise ValueError("Empty or changed signal audit cohort")
    for name, expected in [("y", beats.label.astype(str)), ("groups", beats.patient_id.astype(str))]:
        if not np.array_equal(np.load(control / f"{name}.npy"), expected.to_numpy()):
            raise ValueError("Audit row order does not match the denoised cache")
    shape = (len(beats), 3, paper["data"]["before_r"] + paper["data"]["after_r"] + 1)
    if control_manifest["shape"] != list(shape):
        raise ValueError("Changed waveform window or cohort shape")
    sources = []
    for record in beats.record.unique():
        for extension in ["hea", "xyz"]:
            path = Path(paper["data"]["raw_dir"]) / f"{record}.{extension}"
            sources.append((str(path), file_hash(path)))
    control_hash = file_hash(control / "waveforms.npy")
    fingerprint = _hash({"paper": paper, "classes": config["classes"], "sources": sources,
        "beats_sha256": file_hash(config["beats_path"]), "control_waveforms_sha256": control_hash,
        "control_fingerprint": control_manifest["fingerprint"], "code_sha256": file_hash(__file__),
        "review_patients_per_class": config["review_patients_per_class"], "seed": config["seed"]})
    bank, audit = Path(config["raw_bank_dir"]), Path(config["audit_dir"])
    manifest_path = bank / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["fingerprint"] != fingerprint or saved["waveforms_sha256"] != file_hash(bank / "waveforms.npy"):
            raise ValueError("Raw cache/source settings changed; choose a new bank/audit directory")
        if not (audit / "summary.json").exists() or json.loads((audit / "summary.json").read_text())["bank_fingerprint"] != fingerprint:
            raise ValueError("Completed raw cache is missing its matched audit")
        return bank
    bank.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    scope = {"bank_fingerprint": fingerprint}
    guard = audit / "audit_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Audit settings changed; choose a new directory")
    _save_json(guard, scope)
    temporary = bank / "waveforms.partial.npy"
    raw_waves = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    selected = review_selection(beats, config["classes"], config["review_patients_per_class"], config["seed"])
    jobs = Parallel(n_jobs=config.get("feature_n_jobs", 2), return_as="generator_unordered")(
        delayed(_record)(record, rows, paper, control / "waveforms.npy", selected.get(record))
        for record, rows in beats.groupby("record", sort=False))
    diagnostics, records, examples = [], [], []
    for completed, (indices, values, metrics, record_metrics, example) in enumerate(jobs, 1):
        raw_waves[indices] = values
        diagnostics.append(metrics)
        records.append(record_metrics)
        if example is not None:
            examples.append(example)
        if completed % 25 == 0 or completed == beats.record.nunique():
            print(f"Verified signal audit/raw cache: {completed}/{beats.record.nunique()} records", flush=True)
    raw_waves.flush()
    if not np.isfinite(raw_waves).all():
        raise ValueError("Nonfinite raw cache")
    del raw_waves
    temporary.replace(bank / "waveforms.npy")
    for name in ["y", "groups"]:
        np.save(bank / f"{name}.npy", np.load(control / f"{name}.npy"))
    diagnostics, records = pd.concat(diagnostics, ignore_index=True), pd.DataFrame(records).sort_values("record")
    numeric = [column for column in diagnostics if column not in {"label", "patient_id", "record"}]
    by_class = {}
    for label in config["classes"]:
        rows = diagnostics.loc[diagnostics.label == label]
        by_class[label] = {"n_beats": len(rows), "n_patients": int(rows.patient_id.nunique()),
                          "metrics": {name: _distribution(rows[name]) for name in numeric}}
    summary = {"bank_fingerprint": fingerprint, "n_patients": int(beats.patient_id.nunique()),
        "n_records": int(beats.record.nunique()), "n_beats": len(beats), "shape": shape,
        "header_label_disagreements": 0, "duplicate_cached_peaks": 0, "incomplete_crops": 0,
        "denoised_cache_verified_exactly": True, "classes": by_class,
        "all_beats": {name: _distribution(diagnostics[name]) for name in numeric},
        "cached_intervals": {name: int(records[name].sum()) for name in ["n_intervals", "cached_intervals_below_300ms", "cached_intervals_above_2000ms", "next_cached_peak_inside_crop"]},
        "review_examples": len(examples), "independent_peak_or_delineation_annotations_used": False,
        "disclosure": "Descriptive diagnostics, not validated detector accuracy, artifact labels, proof of denoising harm, or confirmed P/QRS/ST/T coverage. No beats, patients or labels were removed or changed."}
    _save_json(audit / "summary.json", summary)
    review = audit / "review"
    _review_plots(examples, config["classes"], review)
    records.to_csv(review / "record_audit.csv", index=False)
    pd.DataFrame([{key: item[key] for key in ["patient_id", "label", "record", "peak"]} for item in examples]).assign(
        adjudicated_peak_correct="", adjudicated_window_coverage="", notes="").to_csv(review / "review_form.csv", index=False)
    lines = ["# PTB waveform preprocessing audit", "",
        f"{summary['n_patients']} patients, {summary['n_records']} records, {summary['n_beats']:,} unchanged cached beats.", "",
        "All header-derived labels agree with cached labels. Crop boundaries are complete, cached R positions are unique per record, and recomputed denoised crops reproduce the retained cache exactly. This is consistency checking, not independent clinical or detector validation.", "",
        "| Class | Patients | Beats | Median centered relative L2 change | Median centered RMS ratio (denoised/raw) | Median derivative RMS ratio |",
        "|---|---:|---:|---:|---:|---:|"]
    for label, values in by_class.items():
        m = values["metrics"]
        lines.append(f"| {label} | {values['n_patients']} | {values['n_beats']} | {m['centered_relative_l2_change']['median']:.3f} | {m['centered_rms_ratio_denoised_over_raw']['median']:.3f} | {m['derivative_rms_ratio_denoised_over_raw']['median']:.3f} |")
    intervals = summary["cached_intervals"]
    lines += ["", "## Screening counts—not error diagnoses", "",
        f"Among {intervals['n_intervals']:,} consecutive cached-peak intervals, {intervals['cached_intervals_below_300ms']:,} are shorter than 300 ms and {intervals['cached_intervals_above_2000ms']:,} exceed 2,000 ms. These may reflect rhythm variation or detector behavior; there are no independent annotations to decide which.", "",
        f"{intervals['next_cached_peak_inside_crop']:,} intervals place the next cached peak at or before the +400 ms crop endpoint. This describes neighboring-peak overlap, not a confirmed false R peak or a measured T-wave truncation rate.", "",
        f"Raw/denoised review plots were selected for {len(examples)} patients using a fixed class-balanced seed, not model errors. Local-only plots and an UNFILLED adjudication form are in `review/`. No expert adjudication or independent peak/boundary annotations were supplied. The ±wider context and shaded original crop enable review; no sensitivity/PPV or phase-coverage accuracy is claimed.", "",
        "## Interpretation", "",
        "Changes in derivative energy and amplitude quantify what denoising does, not whether it helps localization. Centering is used only for these diagnostics, not to impose per-beat normalization in the model. Edge-energy fractions are not phase delineations. Raw versus denoised classification needs a matched patient-separated experiment; longer windows should not be added or declared better from these screening counts alone.", "",
        "No data were excluded, re-detected or relabeled. Raw XYZ crops retain the original ordering and beat positions. The raw cache is newly created; the denoised cache and all completed experiments are unchanged. Raw signals and review plots remain local; public outputs contain aggregate diagnostics only.", "",
        "Sources: [PTB database](https://physionet.org/content/ptbdb/1.0.0/) and [PyWavelets thresholding documentation](https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html). The existing pipeline soft-thresholds detail coefficients across all used decomposition levels; these sources do not establish whether that choice is optimal for MI localization.", ""]
    (audit / "REPORT.md").write_text("\n".join(lines))
    _save_json(manifest_path, {"fingerprint": fingerprint, "waveforms_sha256": file_hash(bank / "waveforms.npy"),
        "shape": shape, "n_beats": len(beats), "n_patients": summary["n_patients"], "classes": config["classes"],
        "control_fingerprint": control_manifest["fingerprint"], "control_waveforms_sha256": control_hash,
        "input": "Unfiltered measured Frank XYZ, same original cached R crops; no fitted normalization"})
    return bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preprocessing_vcg.yaml")
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(build_raw_bank_and_audit(config))
