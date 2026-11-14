"""Measured simultaneous 12-lead ECG at immutable Frank-VCG beat positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb
import yaml
from joblib import Parallel, delayed

from . import signal
from .labels import label_from_comments
from .neural import _hash, _save_json
from .regularization import load_bank
from .signal_audit import file_hash

LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")


def denoise_ecg(values, wavelet, level):
    """Same shrinkage as the frozen denoiser, with soft(x, 0) = x explicitly.

    PyWavelets' soft-threshold implementation can yield 0/0 for zero-valued
    coefficients with threshold zero. Keeping those coefficients implements
    the mathematical identity, without filling NaNs or excluding a lead.
    """
    output, zero_threshold = [], []
    for lead in range(values.shape[1]):
        x = values[:, lead]
        used = min(level, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
        coeffs = pywt.wavedec(x, wavelet, level=used, mode="symmetric")
        sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / .6745
        threshold = sigma * np.sqrt(2 * np.log(len(x)))
        if threshold == 0:
            zero_threshold.append(LEADS[lead])
            cleaned = coeffs
        else:
            cleaned = [coeffs[0], *[pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]]
        output.append(pywt.waverec(cleaned, wavelet, mode="symmetric")[:len(x)])
    return np.column_stack(output), zero_threshold


def _inventory(beats, paper):
    output = []
    for record in sorted(beats.record.unique()):
        stem = Path(paper["data"]["raw_dir"]) / record
        output.append({"record": record, "files": {ext: file_hash(str(stem) + "." + ext) for ext in ["hea", "dat", "xyz"]}})
    return output


def _record(record, rows, paper, reference_path):
    stem = str(Path(paper["data"]["raw_dir"]) / record)
    header = wfdb.rdheader(stem)
    names = [name.lower() for name in header.sig_name]
    required = LEADS + ("vx", "vy", "vz")
    if any(names.count(lead) != 1 for lead in required):
        raise ValueError("Paired bank requires twelve distinct ECG and measured Frank XYZ channels")
    if rows.patient_id.nunique() != 1 or rows.label.nunique() != 1 or label_from_comments(header.comments) != str(rows.label.iloc[0]):
        raise ValueError("Paired header/cached label disagreement; do not relabel")
    indices = [names.index(lead) for lead in required]
    data = wfdb.rdrecord(stem, channels=indices)
    units = [str(unit).lower() for unit in data.units]
    if header.fs != paper["data"]["sampling_rate"] or data.fs != header.fs or len(data.p_signal) != header.sig_len or data.p_signal.shape[1] != 15 or not np.isfinite(data.p_signal).all() or any(unit != "mv" for unit in units):
        raise ValueError("Paired ECG/VCG must have finite simultaneous 1000 Hz samples in mV")
    # Both streams come from the same WFDB time axis. No ECG redetection,
    # resampling, alignment search or data-dependent patient exclusion.
    before, after = paper["data"]["before_r"], paper["data"]["after_r"]
    peaks = rows.r_peak.to_numpy(dtype=int)
    if len(np.unique(peaks)) != len(peaks) or np.any(peaks < before) or np.any(peaks + after >= len(data.p_signal)):
        raise ValueError("Invalid/duplicate frozen paired R positions")
    settings = paper["signal"]
    ecg, zero_threshold = denoise_ecg(data.p_signal[:, :12], settings["denoise_wavelet"], settings["denoise_level"])
    frank = signal.denoise_vcg(data.p_signal[:, 12:], settings["denoise_wavelet"], settings["denoise_level"])
    cleaned = np.column_stack([ecg, frank])
    if not np.isfinite(cleaned).all():
        raise ValueError("Paired denoising produced nonfinite values")
    crops = np.stack([cleaned[peak-before:peak+after+1].T for peak in peaks]).astype(np.float32)
    reference = np.load(reference_path, mmap_mode="r")
    if not np.array_equal(crops[:, 12:], reference[rows.index.to_numpy()]):
        raise ValueError("Paired Frank crops do not exactly reproduce the retained VCG bank")
    return rows.index.to_numpy(), crops[:, :12], {"record": record, "zero_threshold_ecg_leads": zero_threshold,
        "constant_ecg_leads": [LEADS[lead] for lead in range(12) if np.ptp(data.p_signal[:, lead]) <= 1e-12]}


def prepare_bank(config):
    _, _, _, target = load_bank(config)
    paper = yaml.safe_load(Path(config["paper_config"]).read_text())
    if paper["data"]["sampling_rate"] != 1000:
        raise ValueError("Require simultaneous 1000 Hz target recordings")
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    inventory = _inventory(beats, paper)
    scope = {"target_bank": target, "ecg_leads": list(LEADS), "raw_inventory_sha256": _hash(inventory),
        "code_sha256": file_hash(__file__), "signal_code_sha256": file_hash(signal.__file__),
        "wfdb": wfdb.__version__, "pywavelets": pywt.__version__, "numpy": np.__version__}
    fingerprint = _hash(scope)
    root = Path(config["ecg_bank_dir"])
    shape = [len(beats), 12, target["shape"][2]]
    guard = root / "manifest.json"
    if guard.exists():
        manifest = json.loads(guard.read_text())
        if manifest["fingerprint"] != fingerprint or manifest["scope"] != scope or manifest["shape"] != shape or manifest["waveforms_sha256"] != file_hash(root / "waveforms.npy"):
            raise ValueError("Paired bank input/settings/content changed; use a new cache directory")
        values = np.load(root / "waveforms.npy", mmap_mode="r")
        if values.shape != tuple(shape) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Paired bank is incomplete/nonfinite")
        return root
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "waveforms.partial.npy"
    waves = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=tuple(shape))
    reference = Path(config["bank_dir"]) / "waveforms.npy"
    jobs = Parallel(n_jobs=config["conversion_workers"], return_as="generator_unordered")(
        delayed(_record)(record, rows, paper, reference) for record, rows in beats.groupby("record", sort=True))
    record_qc = []
    for number, (indices, values, qc) in enumerate(jobs, 1):
        waves[indices] = values
        record_qc.append(qc)
        if number % 25 == 0 or number == beats.record.nunique():
            print(f"Measured ECG bank: {number}/{beats.record.nunique()} records", flush=True)
    waves.flush()
    if not np.isfinite(waves).all():
        raise ValueError("Nonfinite ECG crops")
    del waves
    temporary.replace(root / "waveforms.npy")
    manifest = {"fingerprint": fingerprint, "scope": scope, "shape": shape,
        "waveforms_sha256": file_hash(root / "waveforms.npy"), "input": "actual simultaneous measured 12-lead ECG; not reconstructed from VCG",
        "unchanged_rows_labels_peaks": True, "normalization_fitted": False, "processing": "offline full-record wavelet denoising; noncausal"}
    _save_json(guard, manifest)
    review = Path(config["output_dir"]) / "bank_audit/review"
    review.mkdir(parents=True, exist_ok=True)
    _save_json(review / "raw_inventory.json", inventory)
    _save_json(review / "record_qc.json", sorted(record_qc, key=lambda row: row["record"]))
    audit = {"bank_fingerprint": fingerprint, "n_patients": target["n_patients"], "n_records": target["n_records"], "n_beats": len(beats),
        "ecg_shape": shape, "ecg_leads": list(LEADS), "all_frank_crops_exactly_reproduced": True,
        "new_r_peak_detection": False, "patient_exclusions": 0, "new_label_changes": 0, "new_dataset_normalization": False,
        "records_with_constant_ecg_leads": sum(bool(row["constant_ecg_leads"]) for row in record_qc),
        "records_with_zero_threshold_ecg_leads": sum(bool(row["zero_threshold_ecg_leads"]) for row in record_qc)}
    _save_json(review.parent / "summary.json", audit)
    (review.parent / "REPORT.md").write_text("# Simultaneous measured ECG bank\n\n"
        f"Retained all {target['n_patients']} patients, {target['n_records']} recordings and {len(beats):,} original beats. "
        "Twelve measured ECG channels share the WFDB time axis with Frank XYZ. Every recomputed denoised Frank crop exactly matched the existing VCG cache. "
        "ECG uses the same full-record wavelet denoising and frozen R positions/crops, with no redetection, time resizing, exclusions or label changes. "
        "Physical units, finite samples, complete lead inventory and rate were checked. Scaling is deferred to each training partition. "
        f"Records with constant ECG leads: {audit['records_with_constant_ecg_leads']}; with zero-threshold ECG leads: {audit['records_with_zero_threshold_ecg_leads']}. "
        "At zero shrinkage threshold, wavelet coefficients are kept unchanged (soft(x, 0) = x), avoiding 0/0 in zero-valued coefficients. "
        "No missing signal is synthesized; constant leads and their recordings are retained. "
        "Full-record denoising is offline/noncausal. Record identities and raw inventories remain local.\n\n"
        "This bank enables a separate ECG extension; VCG-only results remain the primary benchmark. No localization gain is established by cache construction.\n")
    return root


def load_paired(config):
    vcg, y, groups, target = load_bank(config)
    root = prepare_bank(config)
    manifest = json.loads((root / "manifest.json").read_text())
    ecg = np.load(root / "waveforms.npy", mmap_mode="r")
    return vcg, ecg, y, groups, {"target": target, "ecg_manifest_sha256": file_hash(root / "manifest.json"),
        "ecg_fingerprint": manifest["fingerprint"], "ecg_waveforms_sha256": manifest["waveforms_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paired_ecg_vcg.yaml")
    args = parser.parse_args()
    print(prepare_bank(yaml.safe_load(Path(args.config).read_text())))
