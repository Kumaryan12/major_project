"""Measured-XYZ phase geometry, explicit boundary fallbacks and immutable rows."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy
import yaml
import wfdb
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

from . import signal
from .data import read_vcg
from .labels import label_from_comments
from .neural import _hash, _save_json
from .regularization import load_bank
from .signal_audit import file_hash, review_selection

PHASES = ("qrs", "st", "t")
MARKERS = ("ECG_R_Onsets", "ECG_R_Offsets", "ECG_T_Onsets", "ECG_T_Offsets")
BASE_NAMES = ["duration_s", *[f"mean_{a}" for a in "xyz"], *[f"integral_{a}" for a in "xyz"],
    *[f"peak_{a}" for a in "xyz"], "max_magnitude", "rms_magnitude", "path_length", "mean_speed", "p95_speed",
    *[f"area_{a}" for a in "xyz"], "planarity_fraction", "linearity_fraction", "mean_turn_rad", *[f"displacement_{a}" for a in "xyz"]]
FEATURE_NAMES = [f"{phase}_{name}" for phase in PHASES for name in BASE_NAMES] + [
    "qrs_t_peak_cosine", "qrs_st_peak_cosine", "st_t_peak_cosine", "t_qrs_path_ratio", "st_qrs_path_ratio", "qt_span_s"]


def payload_hash(payload):
    digest = hashlib.sha256()
    for key in sorted(set(payload)-{"content_sha256"}):
        value = np.asarray(payload[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_config(config):
    if config["phase_points"] < 4 or config["savgol_samples"] < 5 or config["savgol_samples"] % 2 != 1 or config["workers"] < 1:
        raise ValueError("Invalid geometry resolution/smoothing/workers")
    if config["fixed_offsets_ms"] != [-60, 80, 140] or config["maximum_post_r_ms"] != 650 or config["next_r_guard_ms"] != 80:
        raise ValueError("Freeze the geometry context and fixed phase design")
    if config["minimum_consensus_leads"] not in [2, 3] or config["maximum_qrs_spread_ms"] <= 0 or config["maximum_t_spread_ms"] <= 0:
        raise ValueError("Invalid consensus requirements")


def fixed_boundaries(peak, next_peak, length, fs, config):
    values = [peak + int(round(ms*fs/1000)) for ms in config["fixed_offsets_ms"]]
    end = min(length-1, peak + round(config["maximum_post_r_ms"]*fs/1000))
    if next_peak is not None:
        end = min(end, next_peak - round(config["next_r_guard_ms"]*fs/1000))
    values.append(end)
    if values[0] < 0 or min(np.diff(values)) < 2 or values[-1] >= length:
        raise ValueError("Frozen R positions do not support the declared fallback phases")
    return np.asarray(values, dtype=int)


def valid_boundaries(bounds, peak, ceiling, fs):
    if len(bounds) != 4 or not np.isfinite(bounds).all():
        return False
    on, off, ton, end = np.asarray(bounds, dtype=float)
    return bool(peak-.16*fs <= on < peak < off <= peak+.18*fs and .03*fs <= off-on <= .24*fs
        and ton-off >= .01*fs and ton >= peak+.06*fs and end-ton >= .04*fs and end <= ceiling and end > peak+.12*fs)


def estimate_boundaries(clean, peaks, fs, config):
    """Apply the pinned single-lead DWT tool to each oriented Frank axis.

    This adaptation and its median consensus are NOT clinically validated.
    R positions are supplied unchanged; the tool never replaces them.
    """
    candidates = np.full((len(peaks), 3, 4), np.nan)
    errors, warning_counts = [], []
    for lead in range(3):
        x = clean[:, lead]
        baseline = np.median(x)
        r_values = x[peaks] - baseline
        scale = float(np.median(np.abs(r_values)))
        if scale < 1e-8:
            errors.append({"lead": lead, "reason": "near_zero_r_scale"})
            warning_counts.append(0)
            continue
        sign = 1 if np.median(r_values) >= 0 else -1
        oriented = sign * (x-baseline) / scale
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                _, waves = nk.ecg_delineate(oriented, rpeaks=np.asarray(peaks, dtype=int), sampling_rate=fs, method="dwt", check=False)
            except (ValueError, IndexError, ZeroDivisionError) as error:
                errors.append({"lead": lead, "reason": type(error).__name__})
                warning_counts.append(len(caught))
                continue
        warning_counts.append(len(caught))
        for column, marker in enumerate(MARKERS):
            values = np.asarray(waves.get(marker, []), dtype=float)
            if values.shape != (len(peaks),):
                raise ValueError("Delineator did not return one marker slot per frozen R")
            candidates[:, lead, column] = values
    fixed, adaptive, accepted, valid_leads = [], [], [], []
    for i, peak in enumerate(peaks):
        next_peak = None if i+1 == len(peaks) else int(peaks[i+1])
        fallback = fixed_boundaries(int(peak), next_peak, len(clean), fs, config)
        possible = [row for row in candidates[i] if valid_boundaries(row, peak, fallback[-1], fs)]
        valid_leads.append(len(possible))
        accept, proposed = False, fallback
        if len(possible) >= config["minimum_consensus_leads"]:
            spread = np.ptp(possible, axis=0)*1000/fs
            proposed = np.rint(np.median(possible, axis=0)).astype(int)
            accept = bool(max(spread[:2]) <= config["maximum_qrs_spread_ms"] and max(spread[2:]) <= config["maximum_t_spread_ms"]
                and valid_boundaries(proposed, peak, fallback[-1], fs))
        fixed.append(fallback)
        adaptive.append(proposed if accept else fallback)
        accepted.append(accept)
    return np.stack(fixed), np.stack(adaptive), np.asarray(accepted), np.asarray(valid_leads), {"lead_errors": errors, "warning_counts": warning_counts}


def phase_features(curve, fs, points):
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 3 or not np.isfinite(curve).all():
        raise ValueError("Expected a finite three-dimensional phase curve")
    velocity = np.gradient(curve, 1/fs, axis=0)
    magnitude = np.linalg.norm(curve, axis=1)
    delta = np.diff(curve, axis=0)
    step = np.linalg.norm(delta, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    peak = curve[np.argmax(magnitude)]
    area = .5*np.cross(curve, np.roll(curve, -1, axis=0)).sum(0)
    singular = np.linalg.svd(curve-curve.mean(0), compute_uv=False)**2
    total = max(float(singular.sum()), 1e-18)
    denominator = step[:-1]*step[1:]
    active = denominator > 1e-16
    cosine = np.sum(delta[:-1]*delta[1:], axis=1)[active] / denominator[active]
    turn = float(np.arccos(np.clip(cosine, -1, 1)).mean()) if len(cosine) else 0.
    values = np.r_[(len(curve)-1)/fs, curve.mean(0), np.trapezoid(curve, dx=1/fs, axis=0), peak,
        magnitude.max(), np.sqrt(np.mean(magnitude**2)), step.sum(), speed.mean(), np.quantile(speed, .95), area,
        singular[-1]/total, singular[0]/total, turn, curve[-1]-curve[0]]
    grid = np.linspace(0, len(curve)-1, points)
    signals = np.column_stack([curve, velocity, magnitude])
    tensor = np.stack([np.interp(grid, np.arange(len(curve)), signals[:, i]) for i in range(7)])
    return values, tensor, peak


def beat_features(clean, peak, bounds, fs, config):
    # One beat's physical pre-R reference only; no cohort-fitted baseline.
    baseline = np.median(clean[peak-round(.20*fs):peak-round(.12*fs)], axis=0)
    pieces, tensors, peaks = [], [], []
    for left, right in zip(bounds[:-1], bounds[1:]):
        values, tensor, vector = phase_features(clean[left:right+1]-baseline, fs, config["phase_points"])
        pieces.append(values)
        tensors.append(tensor)
        peaks.append(vector)
    def cosine(a, b):
        return float(np.clip(np.dot(a, b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-18), -1, 1))
    paths = [values[BASE_NAMES.index("path_length")] for values in pieces]
    values = np.r_[np.concatenate(pieces), cosine(peaks[0], peaks[2]), cosine(peaks[0], peaks[1]), cosine(peaks[1], peaks[2]),
        paths[2]/max(paths[0], 1e-12), paths[1]/max(paths[0], 1e-12), (bounds[-1]-bounds[0])/fs]
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Nonfinite/misaligned phase descriptors")
    return values.astype(np.float32), np.asarray(tensors, dtype=np.float32)


def _record(record, rows, paper, config, reference_path, fingerprint):
    cache = Path(config["geometry_bank_dir"]) / "records"
    path = cache / (hashlib.sha256(record.encode()).hexdigest()[:24] + ".npz")
    if path.exists():
        with np.load(path) as saved:
            payload = {key: saved[key] for key in saved.files}
            if str(saved["fingerprint"].item()) != fingerprint or not np.array_equal(saved["rows"], rows.index.to_numpy()) or str(saved["content_sha256"].item()) != payload_hash(payload):
                raise ValueError("Geometry record checkpoint scope changed")
        return str(path)
    header = (Path(paper["data"]["raw_dir"]) / f"{record}.hea").read_text(encoding="latin1")
    if rows.patient_id.nunique() != 1 or rows.label.nunique() != 1 or label_from_comments(header) != str(rows.label.iloc[0]):
        raise ValueError("Geometry source labels disagree with the frozen cache")
    metadata = wfdb.rdheader(str(Path(paper["data"]["raw_dir"])/record))
    channels = [metadata.sig_name.index(lead) for lead in ["vx", "vy", "vz"]]
    if any(str(metadata.units[index]).lower() != "mv" for index in channels):
        raise ValueError("Geometry requires measured Frank channels in mV")
    raw, fs = read_vcg(Path(paper["data"]["raw_dir"]), record, ["vx", "vy", "vz"])
    if fs != 1000 or not np.isfinite(raw).all():
        raise ValueError("Require finite measured Frank XYZ at 1000 Hz")
    clean = signal.denoise_vcg(raw, paper["signal"]["denoise_wavelet"], paper["signal"]["denoise_level"])
    peaks = rows.r_peak.to_numpy(dtype=int)
    before, after = paper["data"]["before_r"], paper["data"]["after_r"]
    if not np.all(np.diff(peaks) > 0) or min(peaks) < max(before, 200) or max(peaks)+after >= len(clean):
        raise ValueError("Invalid frozen geometry R positions")
    reference = np.load(reference_path, mmap_mode="r")
    check = np.stack([clean[p-before:p+after+1].T for p in peaks]).astype(np.float32)
    if not np.array_equal(check, reference[rows.index.to_numpy()]):
        raise ValueError("Geometry Frank crops do not exactly reproduce the retained cache")
    # Derivatives use a declared 11 ms cubic smoother. Never rotate axes or
    # divide away beat amplitude; physical amplitudes/durations remain features.
    smooth = savgol_filter(clean, config["savgol_samples"], 3, axis=0)
    fixed, adaptive, accepted, valid_leads, audit = estimate_boundaries(clean, peaks, fs, config)
    result = {"rows": rows.index.to_numpy(), "r_peaks": peaks, "fixed_bounds": fixed, "adaptive_bounds": adaptive,
        "accepted": accepted, "valid_leads": valid_leads, "fingerprint": fingerprint, "record_audit": json.dumps(audit)}
    for name, bounds in [("fixed", fixed), ("adaptive", adaptive)]:
        output = [beat_features(smooth, int(p), b, fs, config) for p, b in zip(peaks, bounds)]
        result[name + "_features"] = np.stack([item[0] for item in output])
        result[name + "_tensor"] = np.stack([item[1] for item in output])
    result["content_sha256"] = payload_hash(result)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    return str(path)


def aggregate_features(values, groups):
    patients = np.unique(groups)
    output = []
    for patient in patients:
        part = values[groups == patient]
        output.append(np.r_[np.median(part, axis=0), np.quantile(part, .75, axis=0)-np.quantile(part, .25, axis=0)])
    return patients, np.asarray(output, dtype=np.float32)


def prepare_bank(config):
    validate_config(config)
    _, _, groups, source = load_bank(config)
    paper = yaml.safe_load(Path(config["paper_config"]).read_text())
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    inventory = [(record, {ext: file_hash(Path(paper["data"]["raw_dir"]) / f"{record}.{ext}") for ext in ["hea", "xyz"]}) for record in sorted(beats.record.unique())]
    scope = {"config": config, "source": source, "raw_inventory_sha256": _hash(inventory), "code_sha256": file_hash(__file__),
        "signal_code_sha256": file_hash(signal.__file__), "delineator_code_sha256": file_hash(inspect.getfile(nk.ecg_delineate)),
        "neurokit2": nk.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "statistics_sha256": file_hash(Path(config["bank_dir"]) / "statistics.npy")}
    fingerprint = _hash(scope)
    root = Path(config["geometry_bank_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "records").mkdir(exist_ok=True)
    guard = root / "build_manifest.json"
    manifest = dict(scope, bank_fingerprint=fingerprint)
    if guard.exists() and json.loads(guard.read_text()) != manifest:
        raise ValueError("Geometry bank input/code/settings changed; use a new directory")
    _save_json(guard, manifest)
    if (root / "manifest.json").exists():
        saved = json.loads((root / "manifest.json").read_text())
        if saved["bank_fingerprint"] != fingerprint or any(file_hash(root / name) != digest for name, digest in saved["files"].items()):
            raise ValueError("Geometry completed cache content changed")
        return root
    jobs = Parallel(n_jobs=config["workers"], return_as="generator_unordered")(delayed(_record)(record, rows, paper, config,
        Path(config["bank_dir"]) / "waveforms.npy", fingerprint) for record, rows in beats.groupby("record", sort=True))
    paths = []
    for number, path in enumerate(jobs, 1):
        paths.append(path)
        if number % 25 == 0 or number == beats.record.nunique():
            print(f"Geometry extraction: {number}/{beats.record.nunique()} records", flush=True)
    matrices = {name: np.empty((len(beats), len(FEATURE_NAMES)), np.float32) for name in ["fixed", "adaptive"]}
    tensors = {name: np.lib.format.open_memmap(root / f"{name}_tensor.partial.npy", mode="w+", dtype=np.float32,
        shape=(len(beats), 3, 7, config["phase_points"])) for name in matrices}
    boundaries, record_rows = [], []
    for number, path in enumerate(paths, 1):
        with np.load(path) as saved:
            indices = saved["rows"]
            for name in matrices:
                matrices[name][indices] = saved[name+"_features"]
                tensors[name][indices] = saved[name+"_tensor"]
            frame = beats.iloc[indices][["record", "patient_id", "label", "r_peak"]].copy()
            frame["accepted_consensus"] = saved["accepted"]
            frame["valid_leads"] = saved["valid_leads"]
            for col, name in enumerate(["qrs_on", "qrs_off", "t_on", "t_off"]):
                frame[name] = saved["adaptive_bounds"][:, col]
                frame["fixed_"+name] = saved["fixed_bounds"][:, col]
            boundaries.append(frame)
            audit = json.loads(str(saved["record_audit"].item()))
            record_rows.append({"record": frame.record.iloc[0], "patient_id": frame.patient_id.iloc[0], "label": frame.label.iloc[0],
                "beats": len(indices), "accepted_beats": int(saved["accepted"].sum()), "lead_errors": audit["lead_errors"], "delineator_warnings": sum(audit["warning_counts"])})
        if number % 50 == 0:
            print(f"Geometry assembly: {number}/{len(paths)} records", flush=True)
    for name in matrices:
        if not np.isfinite(matrices[name]).all() or not np.isfinite(tensors[name]).all():
            raise ValueError("Nonfinite geometry features/tensor")
        tensors[name].flush()
    del tensors
    for name in matrices:
        (root / f"{name}_tensor.partial.npy").replace(root / f"{name}_tensor.npy")
    statistics = np.load(Path(config["bank_dir"]) / "statistics.npy", mmap_mode="r")
    if statistics.shape != (len(beats), 192) or not np.isfinite(statistics).all():
        raise ValueError("Expected the aligned finite 192-feature wavelet-statistics control")
    patients, stats_patient = aggregate_features(statistics, groups)
    views = {"statistics": stats_patient}
    for name, values in matrices.items():
        _, views[name+"_geometry"] = aggregate_features(values, groups)
    views["statistics_geometry"] = np.column_stack([stats_patient, views["adaptive_geometry"]])
    labels = np.array([beats.loc[beats.patient_id == patient, "label"].iloc[0] for patient in patients], dtype=str)
    np.savez_compressed(root / "patient_features.npz", patient_id=patients, labels=labels, **views)
    all_boundaries = pd.concat(boundaries).sort_index()
    audit_dir = Path(config["output_dir"]) / "boundary_audit"
    review = audit_dir / "review"
    review.mkdir(parents=True, exist_ok=True)
    all_boundaries.to_csv(review / "boundaries.csv", index=False)
    _save_json(review / "raw_inventory.json", inventory)
    _save_json(review / "record_audit.json", record_rows)
    summary = {"bank_fingerprint": fingerprint, "n_patients": len(patients), "n_records": len(paths), "n_beats": len(beats),
        "accepted_consensus_beats": int(all_boundaries.accepted_consensus.sum()), "accepted_fraction": float(all_boundaries.accepted_consensus.mean()),
        "fallback_beats": int((~all_boundaries.accepted_consensus).sum()), "view_dimensions": {name: value.shape[1] for name, value in views.items()},
        "beat_geometry_dimensions": len(FEATURE_NAMES), "tensor_shape": [len(beats), 3, 7, config["phase_points"]],
        "independent_boundary_annotations_available": False, "patient_exclusions": 0, "label_changes": 0,
        "limits": "Single-lead ECG DWT adapted to XYZ with median consensus; plausible limits are not annotation accuracy. Fallbacks retained. Review plots are illustrative, not expert adjudication."}
    _save_json(audit_dir / "summary.json", summary)
    byclass = all_boundaries.groupby("label").accepted_consensus.agg(["count", "sum", "mean"]).reindex(config["classes"])
    byclass.to_csv(audit_dir / "boundary_coverage.csv")
    _review_plots(beats, all_boundaries, paper, config, review)
    files = ["patient_features.npz", "fixed_tensor.npy", "adaptive_tensor.npy"]
    _save_json(root / "manifest.json", {"bank_fingerprint": fingerprint, "files": {name: file_hash(root/name) for name in files},
        "feature_names": FEATURE_NAMES, "patient_aggregation": "median and IQR across every eligible beat; one row per patient; no cohort-fitted normalization",
        "tensor_axes": ["beat", "phase_QRS_ST_T", "XYZ_velocityXYZ_magnitude", "phase_normalized_time"], "summary": summary})
    lines = ["# VCG-only automatic boundary and geometry audit", "",
        f"{len(patients)} unchanged patients, {len(paths)} records and {len(beats):,} original R positions. Every recomputed Frank crop exactly reproduced the original VCG cache.", "",
        f"Accepted multi-axis boundary consensus: **{summary['accepted_consensus_beats']:,}/{len(beats):,} ({summary['accepted_fraction']:.1%})**. Remaining beats use the declared fixed fallback; no patient or beat is excluded.", "",
        "NeuroKit2 0.3.0 DWT is run on each record's three oriented/scaled Frank axes using supplied original R positions. At least two complete plausible axis estimates must agree within 40 ms for QRS and 80 ms for T. These are engineering checks, not accuracy against manual annotations. No clinical boundary ground truth is available here.", "",
        "Fallback windows start at R−60/R+80/R+140 ms, ending at min(R+650 ms, next cached R−80 ms, recording end). Adaptive estimates stay within that same end ceiling. Phase durations, coordinates, integrals, speed, projected polygon area and orientation are retained. Cubic 11 ms smoothing precedes derivatives. Closed polygon area is a geometric summary, including for open physiological trajectories; it is not an anatomical area.", "",
        "A beat's baseline is the XYZ median from R−200 to R−120 ms. The actual amplitude and axes are retained. Per-phase interpolation creates a 3 × 7 × 32 tensor of XYZ, physical-time velocity XYZ and magnitude; original durations remain separate descriptors. Fixed and adaptive paths have identical feature counts. All beats contribute to patient medians/IQR; longer recordings contribute more beats within a patient. Scaling/model fitting are deferred to training folds.", "",
        "## Boundary coverage by source class", "", "| Class | Beats | Accepted | Fraction |", "|---|---:|---:|---:|"]
    lines += [f"| {label} | {int(row['count'])} | {int(row['sum'])} | {row['mean']:.1%} |" for label, row in byclass.iterrows()]
    lines += ["", "Consensus success/fallback flags are audit outputs, not classifier features. Class-dependent success may still affect adaptive descriptors; compare the fixed-window control. Source labels never enter delineation. The adaptive and fixed methods share an outer context ceiling but select different intervals within it; this is not a pure tensor-compression comparison.", "",
        "The private `review/` packet has label-stratified examples chosen before classification using only IDs/labels, plus per-record/beat audits. Plots are not clinical adjudication. No hard cases are removed. Full-record denoising remains offline/noncausal; VCG is the only signal input.", "",
        "References: [NeuroKit delineation](https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-delineate), [Martínez et al.](https://pubmed.ncbi.nlm.nih.gov/15072211/), [VCG loop variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC3498118/). Tool performance on ECG does not establish accuracy of this VCG adaptation.", ""]
    (audit_dir / "REPORT.md").write_text("\n".join(lines))
    return root


def _review_plots(beats, boundaries, paper, config, review):
    chosen = review_selection(beats, config["classes"], config["review_patients_per_class"], config["seed"])
    entries = []
    for record, choice in chosen.items():
        rows = boundaries.loc[(boundaries.record == record) & (boundaries.r_peak == choice["peak"])]
        row = rows.iloc[0]
        raw, fs = read_vcg(Path(paper["data"]["raw_dir"]), record, ["vx", "vy", "vz"])
        clean = signal.denoise_vcg(raw, paper["signal"]["denoise_wavelet"], paper["signal"]["denoise_level"])
        peak = int(row.r_peak)
        left, right = max(0, peak-250), min(len(raw), int(row.t_off)+100)
        figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for lead, axis in enumerate(axes):
            axis.plot((np.arange(left,right)-peak)*1000/fs, clean[left:right, lead], linewidth=1)
            for start, end, color, phase in [(row.qrs_on,row.qrs_off,"#dce7ef","QRS estimate"), (row.qrs_off,row.t_on,"#eee8dc","ST estimate"), (row.t_on,row.t_off,"#dfebdf","T estimate")]:
                axis.axvspan((start-peak)*1000/fs, (end-peak)*1000/fs, alpha=.65, color=color, label=phase)
            axis.axvline(0, color="black", linestyle=":", linewidth=.7)
            axis.set_ylabel("XYZ"[lead]+" (mV)")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Time from unchanged cached R (ms)")
        figure.suptitle(f"Non-adjudicated example: {choice['label']} {choice['patient_id']}\naccepted consensus={bool(row.accepted_consensus)}; valid axes={row.valid_leads}")
        figure.tight_layout()
        filename = f"{choice['label']}_{choice['patient_id']}.png"
        figure.savefig(review / filename, dpi=130)
        plt.close(figure)
        entries.append(f"- [{choice['label']} {choice['patient_id']}]({filename}): accepted={bool(row.accepted_consensus)}")
    (review / "INDEX.md").write_text("# Illustrative boundary review; no independent clinical adjudication\n\n"+"\n".join(entries)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geometry_vcg.yaml")
    args = parser.parse_args()
    print(prepare_bank(yaml.safe_load(Path(args.config).read_text())))
