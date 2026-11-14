"""Longer denoised XYZ crops at immutable cached R positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from . import signal
from .data import read_vcg
from .labels import label_from_comments
from .neural import _hash, _save_json
from .signal_audit import file_hash


def context_crop(recording, peak, before, after):
    """Reflect only samples beyond the recording; no time resizing or recentering."""
    recording = np.asarray(recording)
    if recording.ndim != 2 or recording.shape[1] != 3 or len(recording) < 2 or not np.isfinite(recording).all():
        raise ValueError("Expected finite full-record XYZ")
    if not 0 <= peak < len(recording) or before < 0 or after < 0:
        raise ValueError("Invalid context crop bounds")
    left, right = peak - before, peak + after + 1
    padding = (max(0, -left), max(0, right - len(recording)))
    # Pad the complete recording, not an already shortened beat. Real samples
    # remain untouched, including the original central 651-sample crop.
    if any(padding):
        padded = np.pad(recording, (padding, (0, 0)), mode="reflect")
        result = padded[left + padding[0]:right + padding[0]]
    else:
        result = recording[left:right]
    return result.T.astype(np.float32), padding


def _record(record, rows, paper, config):
    data = paper["data"]
    root = Path(data["raw_dir"])
    header = (root / f"{record}.hea").read_text(encoding="latin1")
    if rows.patient_id.nunique() != 1 or rows.label.nunique() != 1 or label_from_comments(header) != rows.label.iloc[0]:
        raise ValueError("Context header/cached patient or label disagreement")
    raw, fs = read_vcg(root, record, data["leads"])
    if fs != data["sampling_rate"] or not np.isfinite(raw).all():
        raise ValueError("Invalid context recording/sample rate")
    peaks = rows.r_peak.to_numpy(dtype=int)
    if len(np.unique(peaks)) != len(peaks) or np.any(peaks < data["before_r"]) or np.any(peaks + data["after_r"] >= len(raw)):
        raise ValueError("Invalid original cached crop positions")
    clean = signal.denoise_vcg(raw, paper["signal"]["denoise_wavelet"], paper["signal"]["denoise_level"])
    original = np.stack([clean[peak - data["before_r"]:peak + data["after_r"] + 1].T for peak in peaks]).astype(np.float32)
    reference = np.load(Path(config["short_bank_dir"]) / "waveforms.npy", mmap_mode="r")
    if not np.array_equal(original, reference[rows.index.to_numpy()]):
        raise ValueError("Context regenerated short crops do not exactly match retained cache")
    cropped = [context_crop(clean, peak, config["before_r"], config["after_r"]) for peak in peaks]
    waves, padding = np.stack([item[0] for item in cropped]), np.array([item[1] for item in cropped], dtype=np.int32)
    start = config["before_r"] - data["before_r"]
    if not np.array_equal(waves[:, :, start:start + original.shape[2]], original):
        raise ValueError("Longer context modified the original central samples")
    return rows.index.to_numpy(), waves, padding


def validate_windows(config, paper):
    if paper["data"]["leads"] != ["vx", "vy", "vz"] or paper["data"]["sampling_rate"] != 1000:
        raise ValueError("Require measured Frank XYZ at 1000 Hz")
    for key in ["before_r", "after_r"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < paper["data"][key]:
            raise ValueError("Context must contain the entire original crop")
    if config["before_r"] + config["after_r"] <= paper["data"]["before_r"] + paper["data"]["after_r"] or config["boundary_mode"] != "reflect":
        raise ValueError("Require a longer window and explicit reflect boundary policy")


def build_context_bank(config):
    paper = yaml.safe_load(Path(config["paper_config"]).read_text())
    validate_windows(config, paper)
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    short, bank, audit = (Path(config[key]) for key in ["short_bank_dir", "long_bank_dir", "audit_dir"])
    control = json.loads((short / "manifest.json").read_text())
    short_shape = [len(beats), 3, paper["data"]["before_r"] + paper["data"]["after_r"] + 1]
    if beats.empty or control["classes"] != config["classes"] or control["shape"] != short_shape or np.load(short / "waveforms.npy", mmap_mode="r").shape != tuple(short_shape):
        raise ValueError("Context control cohort/window/classes changed")
    for name, expected in [("y", beats.label.astype(str).to_numpy()), ("groups", beats.patient_id.astype(str).to_numpy())]:
        if not np.array_equal(np.load(short / f"{name}.npy"), expected):
            raise ValueError("Context control rows do not match the cached beats")
    sources = [(str(path), file_hash(path)) for record in beats.record.unique() for extension in ["hea", "xyz"]
               for path in [Path(paper["data"]["raw_dir"]) / f"{record}.{extension}"]]
    provenance = {"paper": paper, "classes": config["classes"], "before_r": config["before_r"], "after_r": config["after_r"],
        "boundary_mode": config["boundary_mode"], "sources": sources, "beats_sha256": file_hash(config["beats_path"]),
        "control_fingerprint": control["fingerprint"], "control_files": {name: file_hash(short / f"{name}.npy") for name in ["waveforms", "y", "groups"]},
        "code_sha256": file_hash(__file__), "signal_code_sha256": file_hash(signal.__file__)}
    fingerprint = _hash(provenance)
    if (bank / "manifest.json").exists():
        manifest = json.loads((bank / "manifest.json").read_text())
        if manifest["fingerprint"] != fingerprint or any(file_hash(bank / f"{name}.npy") != digest for name, digest in manifest["files"].items()):
            raise ValueError("Context cache/source settings changed; use a new directory")
        if not (audit / "summary.json").exists() or json.loads((audit / "summary.json").read_text())["bank_fingerprint"] != fingerprint:
            raise ValueError("Context bank is missing its matched audit")
        return bank
    bank.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    guard = audit / "audit_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != {"bank_fingerprint": fingerprint}:
        raise ValueError("Context audit settings changed; use a new directory")
    _save_json(guard, {"bank_fingerprint": fingerprint})
    shape = [len(beats), 3, config["before_r"] + config["after_r"] + 1]
    temporary = bank / "waveforms.partial.npy"
    waves = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=tuple(shape))
    padding = np.zeros((len(beats), 2), dtype=np.int32)
    jobs = Parallel(n_jobs=config.get("feature_n_jobs", 2), return_as="generator_unordered")(
        delayed(_record)(record, rows, paper, config) for record, rows in beats.groupby("record", sort=False))
    for count, (indices, values, edges) in enumerate(jobs, 1):
        waves[indices], padding[indices] = values, edges
        if count % 25 == 0 or count == beats.record.nunique():
            print(f"Verified longer context: {count}/{beats.record.nunique()} records", flush=True)
    waves.flush()
    if not np.isfinite(waves).all():
        raise ValueError("Nonfinite context cache")
    del waves
    temporary.replace(bank / "waveforms.npy")
    np.save(bank / "padding.npy", padding)
    for name in ["y", "groups"]:
        np.save(bank / f"{name}.npy", np.load(short / f"{name}.npy"))
    padded = padding.any(1)
    summary = {"bank_fingerprint": fingerprint, "n_patients": int(beats.patient_id.nunique()), "n_records": int(beats.record.nunique()),
        "n_beats": len(beats), "short_window_samples": short_shape[2], "long_window_samples": shape[2],
        "before_r": config["before_r"], "after_r": config["after_r"], "boundary_mode": "reflect",
        "padded_beats": int(padded.sum()), "padded_patients": int(beats.loc[padded, "patient_id"].nunique()),
        "left_padded_beats": int((padding[:, 0] > 0).sum()), "right_padded_beats": int((padding[:, 1] > 0).sum()),
        "left_padded_samples": int(padding[:, 0].sum()), "right_padded_samples": int(padding[:, 1].sum()),
        "max_left_padding": int(padding[:, 0].max()), "max_right_padding": int(padding[:, 1].max()),
        "class_padding": {label: {"beats": int((beats.label == label).sum()), "padded_beats": int((padded & (beats.label == label)).sum())} for label in config["classes"]},
        "central_short_crop_verified_exactly": True,
        "disclosure": "No redetection, label change, beat exclusion, time resizing or independent physiological coverage validation. Reflection padding is synthetic context only at recording boundaries."}
    _save_json(audit / "summary.json", summary)
    lines = ["# Longer denoised VCG context: cache audit", "",
        f"{summary['n_patients']} patients, {summary['n_records']} records, {len(beats):,} original beats retained.", "",
        f"Original: {short_shape[2]} samples (−{paper['data']['before_r']} to +{paper['data']['after_r']} ms). Longer: {shape[2]} samples (−{config['before_r']} to +{config['after_r']} ms), both at 1000 Hz.", "",
        "Every original short crop was regenerated exactly, and every embedded central crop is bitwise equal to the retained float32 cache. Both use the original full-record wavelet denoising. No new R detection or beat-wise denoising is used.", "",
        f"Reflection padding affects {summary['padded_beats']:,} beats ({summary['padded_beats'] / len(beats):.3%}) across {summary['padded_patients']} patients: {summary['left_padded_beats']} left boundaries and {summary['right_padded_beats']} right boundaries. Maximum missing samples: left {summary['max_left_padding']}, right {summary['max_right_padding']}. Padding is applied after full-record denoising; no existing samples are replaced.", "",
        "| Class | Beats | Padded beats |", "|---|---:|---:|"]
    lines += [f"| {label} | {values['beats']} | {values['padded_beats']} |" for label, values in summary["class_padding"].items()]
    lines += ["", summary["disclosure"], "", "Longer windows may contain neighboring beats; they are not adjudicated P/QRS/ST/T segments or guaranteed full cardiac cycles. Existing peak screening flags do not establish physiological coverage. Classification benefit requires the matched patient-separated study.", "",
        "Waveforms, per-beat padding and patient identities remain local. Only aggregate cache diagnostics are published.", ""]
    (audit / "REPORT.md").write_text("\n".join(lines))
    _save_json(bank / "manifest.json", {"fingerprint": fingerprint, "classes": config["classes"], "shape": shape,
        "n_beats": len(beats), "n_patients": summary["n_patients"], "provenance": provenance,
        "files": {name: file_hash(bank / f"{name}.npy") for name in ["waveforms", "y", "groups", "padding"]}})
    return bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/context_vcg.yaml")
    args = parser.parse_args()
    print(build_context_bank(yaml.safe_load(Path(args.config).read_text())))
