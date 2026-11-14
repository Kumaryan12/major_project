"""Freeze label-free PTB-XL source splits and build a bounded XYZ pilot bank."""
from __future__ import annotations

import argparse
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import wfdb
import yaml
from joblib import Parallel, delayed
from scipy.signal import resample_poly

from . import external, signal
from .data import download_file
from .external import ECG_LEADS, KORS_MATRIX, kors_transform
from .neural import _hash, _save_json
from .signal_audit import file_hash

IDENTITY_STATUS = "unresolved: unrelated de-identified namespaces; no cross-dataset identity linkage or comprehensive waveform-overlap screen"


def validate_config(config):
    if config["version"] != "1.0.3" or config["train_folds"] != list(range(1, 9)) or config["validation_fold"] != 9 or config["excluded_fold"] != 10:
        raise ValueError("Freeze PTB-XL 1.0.3 folds 1–8/9; fold 10 is excluded")
    for key in ["train_patients", "validation_patients", "download_workers", "conversion_workers", "review_records_per_role"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    settings = config["preprocessing"]
    if settings["sampling_rate"] != 1000 or settings["before_r"] != 250 or settings["after_r"] != 400:
        raise ValueError("Pilot must retain the 1000 Hz, 651-sample target window")


def select_cohort(database, config):
    """Patient-uniform selection; diagnostic labels and quality fields are unused."""
    validate_config(config)
    required = ["ecg_id", "patient_id", "strat_fold", "filename_hr"]
    if not set(required) <= set(database):
        raise ValueError("Missing source metadata columns")
    frame = database[required].copy()
    for name in ["ecg_id", "patient_id", "strat_fold"]:
        values = pd.to_numeric(frame[name], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values <= 0):
            raise ValueError("Source IDs/folds must be finite positive integers")
        frame[name] = values.astype(np.int64)
    if not frame.ecg_id.is_unique or not frame.filename_hr.is_unique or set(frame.strat_fold) != set(range(1, 11)):
        raise ValueError("Duplicate source recordings or missing/unexpected folds")
    if (frame.groupby("patient_id").strat_fold.nunique() != 1).any():
        raise ValueError("Source patients cross fold boundaries")
    for row in frame.itertuples():
        if not isinstance(row.filename_hr, str) or not re.fullmatch(r"records500/\d{5}/\d{5}_hr", row.filename_hr) or int(row.filename_hr.rsplit("/", 1)[1].split("_")[0]) != row.ecg_id:
            raise ValueError("Unexpected or unsafe source waveform path")
    rng, selected = np.random.default_rng(config["seed"]), []
    for role, folds, limit in [("train", config["train_folds"], config["train_patients"]), ("validation", [9], config["validation_patients"])]:
        eligible = frame.loc[frame.strat_fold.isin(folds)]
        ids = np.sort(eligible.patient_id.unique())
        if len(ids) < limit:
            raise ValueError("Requested source pilot exceeds available unique patients")
        chosen = rng.permutation(ids)[:limit]
        # One lowest-ID recording per chosen patient, with no label filter.
        part = eligible.loc[eligible.patient_id.isin(chosen)].sort_values(["patient_id", "ecg_id"]).drop_duplicates("patient_id").copy()
        part["role"] = role
        part["patient_id"] = part.patient_id.map(lambda value: f"ptbxl:{value}")
        selected.append(part)
    cohort = pd.concat(selected, ignore_index=True)
    if not cohort.patient_id.is_unique or (cohort.strat_fold == 10).any():
        raise ValueError("Pilot cohort leaks source patients or excluded fold")
    return cohort


def assert_source_row(row, config):
    role, fold, name = str(row.role), int(row.strat_fold), str(row.filename_hr)
    if (role == "train" and fold not in config["train_folds"]) or (role == "validation" and fold != config["validation_fold"]) or role not in {"train", "validation"} or fold == 10:
        raise ValueError("Refusing waveform access outside frozen source train/validation folds")
    if not re.fullmatch(r"records500/\d{5}/\d{5}_hr", name) or int(name.rsplit("/", 1)[1].split("_")[0]) != int(row.ecg_id):
        raise ValueError("Unsafe/mismatched source record path")


def freeze_cohort(config):
    validate_config(config)
    metadata = Path(config["metadata_path"])
    if not metadata.exists():
        download_file(f"{config['base_url'].rstrip('/')}/ptbxl_database.csv", metadata)
    database = pd.read_csv(metadata)
    cohort = select_cohort(database, config)
    cohort_bytes = cohort.to_csv(index=False).encode()
    import hashlib
    scope = {"dataset_version": config["version"], "metadata_sha256": file_hash(metadata),
        "seed": config["seed"], "train_folds": config["train_folds"], "validation_fold": 9, "excluded_fold": 10,
        "train_patients": config["train_patients"], "validation_patients": config["validation_patients"],
        "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest(), "code_sha256": file_hash(__file__)}
    fingerprint = _hash(scope)
    scope["cohort_fingerprint"] = fingerprint
    audit = Path(config["audit_dir"])
    audit.mkdir(parents=True, exist_ok=True)
    guard = audit / "cohort_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Source cohort settings/metadata/code changed; use a new audit/bank directory")
    _save_json(guard, scope)
    review = audit / "review"
    review.mkdir(exist_ok=True)
    private = review / "selected_cohort.csv"
    if private.exists() and private.read_bytes() != cohort_bytes:
        raise ValueError("Frozen source cohort was modified")
    private.write_bytes(cohort_bytes)
    summary = {"cohort_fingerprint": fingerprint, "total_records": len(database), "total_patients": int(database.patient_id.nunique()),
        "fold_counts": {str(fold): {"records": len(part), "patients": int(part.patient_id.nunique())} for fold, part in database.groupby("strat_fold")},
        "selected": {role: {"records": len(part), "patients": int(part.patient_id.nunique())} for role, part in cohort.groupby("role")},
        "within_source_patient_overlap": 0, "excluded_fold_waveforms_selected": 0,
        "diagnostic_labels_used_for_selection": False, "cross_dataset_identity_overlap": IDENTITY_STATUS,
        "ptb_patient_independence_cleared": False,
        "permitted_current_endpoint": "source-only masked reconstruction on held-out PTB-XL patients; no downstream PTB score or SOTA claim"}
    _save_json(audit / "cohort_summary.json", summary)
    lines = ["# Frozen source pretraining cohort", "", f"PTB-XL {config['version']}: {len(database):,} records, {summary['total_patients']:,} unique source patients.", "",
        f"Pilot: {config['train_patients']:,} training patients from folds 1–8 and {config['validation_patients']:,} validation patients from fold 9. One recording per patient, fixed seeded patient selection, lowest ECG ID per selected patient. No diagnostic labels, six-class mapping, model errors or metadata quality flags enter selection.", "",
        "All source patient IDs stay within one metadata fold. Fold 10 is excluded from the new waveform pipeline, even though old external-test files already exist locally. Source access functions independently reject excluded/mismatched roles before downloads or waveform reads.", "",
        "## Provenance and identity limits", "",
        "PTB-XL uses Schiller ECG data acquired in 1989–1996; PTB Diagnostic uses a PTB prototype recorder and includes measured Frank XYZ. These are different acquisition domains. A source-derived XYZ signal is not a measured Frank signal.", "",
        "PTB and PTB-XL patient identifiers occupy unrelated de-identified namespaces. Different identifier strings, institutions/devices or a lack of exact file duplicates do not establish that the underlying people are disjoint. Cross-dataset identity linkage and comprehensive aligned waveform-overlap screening are unresolved. This audit does not clear downstream PTB patient independence.", "",
        "Current permitted experiment: source-only masked reconstruction with source patient-disjoint validation. No PTB fine-tuning/evaluation is launched here. This is a pretraining feasibility pilot, not evidence of MI localization improvement. Subsequent transfer must explicitly address the unresolved provenance limitation before a clean patient-independent claim.", "",
        "Next stage checks selected waveform shapes, units, finite values and exact duplicates, converts 12-lead ECG to Kors XYZ, resamples to 1000 Hz, applies the existing denoising/detector, and keeps one median eligible cached crop per retained source patient. Any source QC exclusions are counted; target PTB patients/beats/labels remain unchanged.", "",
        "Sources: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), [PTB Diagnostic](https://physionet.org/content/ptbdb/1.0.0/), [Kors coefficient implementation](https://github.com/Tereshchenkolab/Pacing_spike_removal_v2/blob/main/kors.m). Selected patient/record identities remain local in `review/selected_cohort.csv`.", ""]
    (audit / "COHORT_REPORT.md").write_text("\n".join(lines))
    return cohort, scope


def _download(row, config):
    assert_source_row(row, config)
    for extension in ["hea", "dat"]:
        relative = f"{row.filename_hr}.{extension}"
        download_file(f"{config['base_url'].rstrip('/')}/{relative}", Path(config["waveform_dir"]) / relative)


def convert_record(row, config):
    assert_source_row(row, config)
    root = Path(config["waveform_dir"]) / str(row.filename_hr)
    hashes = {extension: file_hash(Path(f"{root}.{extension}")) for extension in ["hea", "dat"]}
    record = wfdb.rdrecord(str(root))
    if record.fs != 500 or record.p_signal.shape != (5000, 12) or len(record.sig_name) != len(set(name.lower() for name in record.sig_name)) or any(str(unit).lower() != "mv" for unit in record.units):
        raise ValueError("Unexpected source shape/rate, duplicate leads or non-mV units")
    item = {"ecg_id": int(row.ecg_id), "patient_id": str(row.patient_id), "strat_fold": int(row.strat_fold), "role": str(row.role),
            "header_sha256": hashes["hea"], "data_sha256": hashes["dat"], "accepted": False, "reason": None}
    if not np.isfinite(record.p_signal).all():
        return None, dict(item, reason="nonfinite_ecg")
    settings = config["preprocessing"]
    vcg = resample_poly(kors_transform(record.p_signal, record.sig_name), 2, 1, axis=0)
    if not np.isfinite(vcg).all() or np.linalg.norm(vcg - vcg.mean(0)) < 1e-8:
        return None, dict(item, reason="nonfinite_or_flat_xyz")
    clean = signal.denoise_vcg(vcg, settings["wavelet"], settings["denoise_level"])
    peaks = signal.detect_r_peaks(clean, 1000, settings["rpeak_refractory_ms"], settings["rpeak_integration_ms"], settings["rpeak_threshold_scale"])
    valid = peaks[(peaks >= settings["before_r"]) & (peaks + settings["after_r"] < len(clean))]
    if len(valid) == 0:
        return None, dict(item, reason="no_complete_detected_crop")
    peak = int(valid[len(valid) // 2])
    beat = clean[peak - settings["before_r"]:peak + settings["after_r"] + 1].T.astype(np.float32)
    if beat.shape != (3, 651) or not np.isfinite(beat).all() or np.linalg.norm(beat - beat.mean(1, keepdims=True)) < 1e-8:
        return None, dict(item, reason="invalid_or_flat_crop")
    return beat, dict(item, accepted=True, reason="accepted", r_peak=peak, complete_detected_crops=len(valid),
                      lead_rms=np.sqrt(np.mean(beat.astype(float) ** 2, axis=1)).tolist())


def _conversion_job(row, config, fingerprint):
    assert_source_row(row, config)
    part = Path(config["bank_dir"]) / "parts" / f"record_{int(row.ecg_id):05d}.npz"
    root = Path(config["waveform_dir"]) / str(row.filename_hr)
    current = {suffix: file_hash(Path(f"{root}.{suffix}")) for suffix in ["hea", "dat"]}
    if part.exists():
        with np.load(part) as saved:
            item = json.loads(str(saved["audit_json"].item()))
            if item["bank_scope"] != fingerprint or item["header_sha256"] != current["hea"] or item["data_sha256"] != current["dat"] or item["role"] != row.role or item["patient_id"] != row.patient_id:
                raise ValueError("Source conversion checkpoint inputs changed")
            values = saved["waveform"]
            return (values if item["accepted"] else None), item
    values, item = convert_record(row, config)
    item["bank_scope"] = fingerprint
    temporary = part.with_suffix(".partial.npz")
    np.savez_compressed(temporary, waveform=values if values is not None else np.empty((0,), dtype=np.float32), audit_json=json.dumps(item))
    temporary.replace(part)
    return values, item


def _review_plots(waves, records, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    time = np.arange(-250, 401)
    for role in ["train", "validation"]:
        rows = records.loc[records.role == role].sort_values("ecg_id").head(config["review_records_per_role"])
        fig, axes = plt.subplots(3, len(rows), figsize=(4.5 * len(rows), 7), squeeze=False, sharex=True)
        for column, (index, row) in enumerate(rows.iterrows()):
            for lead in range(3):
                ax = axes[lead, column]
                ax.plot(time, waves[index, lead], color="0.3", linewidth=0.8)
                ax.axvline(0, linestyle="--", color="0.6", linewidth=0.8)
                ax.set_ylabel(f"{'XYZ'[lead]} (mV)")
                ax.grid(alpha=0.2)
                if lead == 0:
                    ax.set_title(f"ECG {int(row.ecg_id)}; fold {int(row.strat_fold)}", fontsize=9)
                if lead == 2:
                    ax.set_xlabel("Time from detector position (ms)")
        fig.suptitle(f"Source {role}: Kors-derived XYZ; detector positions are not adjudicated", fontsize=10)
        fig.tight_layout()
        fig.savefig(directory / f"source_{role}.png", dpi=140)
        plt.close(fig)


def prepare_bank(config):
    cohort, cohort_scope = freeze_cohort(config)
    bank, audit = Path(config["bank_dir"]), Path(config["audit_dir"])
    scope = {"cohort_fingerprint": cohort_scope["cohort_fingerprint"], "preprocessing": config["preprocessing"],
        "code_sha256": file_hash(__file__), "conversion_code_sha256": file_hash(external.__file__), "signal_code_sha256": file_hash(signal.__file__),
        "kors_coefficients": KORS_MATRIX.tolist(), "wfdb": wfdb.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    fingerprint = _hash(scope)
    bank.mkdir(parents=True, exist_ok=True)
    guard = bank / "build_scope.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Source bank build settings/code changed; choose new directories")
    _save_json(guard, scope)
    manifest = bank / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["fingerprint"] != fingerprint or any(file_hash(bank / name) != digest for name, digest in saved["files"].items()):
            raise ValueError("Source bank content changed")
        # Selected source files only; never inspect the excluded fold's cache.
        expected = pd.read_csv(audit / "review/record_audit.csv")
        for row in cohort.itertuples(index=False):
            assert_source_row(row, config)
            prior = expected.loc[expected.ecg_id == row.ecg_id].iloc[0]
            for extension, field in [("hea", "header_sha256"), ("dat", "data_sha256")]:
                if file_hash(Path(config["waveform_dir"]) / f"{row.filename_hr}.{extension}") != prior[field]:
                    raise ValueError("Selected source waveform changed since bank creation")
        return bank
    with ThreadPoolExecutor(max_workers=config["download_workers"]) as pool:
        jobs = [pool.submit(_download, row, config) for row in cohort.itertuples(index=False)]
        for count, job in enumerate(as_completed(jobs), 1):
            job.result()
            if count % 100 == 0 or count == len(jobs):
                print(f"Source recordings downloaded: {count}/{len(jobs)}", flush=True)
    (bank / "parts").mkdir(exist_ok=True)
    converted = Parallel(n_jobs=config["conversion_workers"])(delayed(_conversion_job)(row, config, fingerprint) for row in cohort.itertuples(index=False))
    records = pd.DataFrame([item for _, item in converted])
    duplicate_patients = records.groupby("data_sha256").patient_id.nunique()
    duplicate_hashes = set(duplicate_patients.index[duplicate_patients > 1])
    records.loc[records.data_sha256.isin(duplicate_hashes), ["accepted", "reason"]] = [False, "cross_patient_exact_source_duplicate"]
    accepted = records.loc[records.accepted].copy().reset_index(drop=True)
    if any((accepted.role == role).sum() < 2 for role in ["train", "validation"]):
        raise ValueError("Too few retained source patients after QC/duplicate quarantine")
    by_id = {item["ecg_id"]: values for values, item in converted}
    waves = np.stack([by_id[int(row.ecg_id)] for row in accepted.itertuples()])
    for name, values in [("waveforms", waves), ("groups", accepted.patient_id.to_numpy(dtype=str)), ("roles", accepted.role.to_numpy(dtype=str)), ("ecg_ids", accepted.ecg_id.to_numpy(dtype=np.int64))]:
        np.save(bank / f"{name}.npy", values)
    records.to_csv(audit / "review/record_audit.csv", index=False)
    accepted.to_csv(audit / "review/retained_records.csv", index=False)
    _review_plots(waves, accepted, config, audit / "review")
    summary = {"bank_fingerprint": fingerprint, "cohort_fingerprint": cohort_scope["cohort_fingerprint"], "selected_records": len(records),
        "retained_records": len(accepted), "retained_patients": int(accepted.patient_id.nunique()), "shape": list(waves.shape),
        "role_counts": {role: {"records": len(part), "patients": int(part.patient_id.nunique())} for role, part in accepted.groupby("role")},
        "exclusion_counts": {str(reason): int(count) for reason, count in records.loc[~records.accepted].reason.value_counts().items()},
        "cross_patient_exact_source_duplicate_groups": len(duplicate_hashes), "fold_10_waveforms_accessed": 0,
        "source_patient_train_validation_overlap": 0, "cross_dataset_identity_overlap": IDENTITY_STATUS,
        "ptb_patient_independence_cleared": False, "independent_detector_annotations": False,
        "disclosure": "Kors-derived XYZ source-only bank; fixed median complete detector crop per retained patient. No PTB target changes or downstream scoring. Exact source-file duplicate screening is not cross-dataset identity proof."}
    _save_json(audit / "bank_summary.json", summary)
    _save_json(manifest, {"fingerprint": fingerprint, "scope": scope, "shape": list(waves.shape),
        "files": {f"{name}.npy": file_hash(bank / f"{name}.npy") for name in ["waveforms", "groups", "roles", "ecg_ids"]},
        "ptb_patient_independence_cleared": False, "cross_dataset_identity_overlap": IDENTITY_STATUS})
    lines = ["# Derived-XYZ source pilot bank", "", f"{len(accepted):,}/{len(records):,} selected recordings retained; one 651-sample crop per unique retained source patient.", "",
        "| Role | Retained patients |", "|---|---:|"]
    lines += [f"| {role} | {values['patients']} |" for role, values in summary["role_counts"].items()]
    lines += ["", f"Exclusions: `{json.dumps(summary['exclusion_counts'])}`. Exact cross-patient source-file duplicate groups: {len(duplicate_hashes)}; every selected member of a detected group is quarantined.", "",
        "Waveforms are checked for 500 Hz, 5000 samples, 12 unique leads, mV units and finite values. Eight independent leads feed the existing Kors matrix; polyphase resampling produces 1000 Hz XYZ. Existing bior6.8 full-record denoising and detector settings are reused. One median eligible detector crop is retained. Flat/nonfinite signals and missing complete crops are recorded as exclusions. Malformed files fail the build rather than being silently skipped.", "",
        "No diagnostic labels enter selection or pretraining. Source training and validation patients are disjoint; fold 10 waveform access is rejected before I/O. Target PTB data, folds, labels and scores are untouched.", "",
        "Detector positions and derived-vector fidelity are not independently adjudicated on PTB-XL. Local train/validation example plots and per-record audits are in `review/`. No sensitivity/PPV, physiological coverage, cross-dataset patient independence, localization improvement or SOTA is claimed. Comprehensive PTB-versus-PTB-XL waveform/identity overlap remains unresolved; downstream PTB benchmarking has not been cleared by this audit.", ""]
    (audit / "BANK_REPORT.md").write_text("\n".join(lines))
    return bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/source_pretraining_vcg.yaml")
    parser.add_argument("--prepare", action="store_true", help="Download/verify the selected source-only pilot and build XYZ crops")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    print(prepare_bank(config) if args.prepare else freeze_cohort(config)[1]["cohort_fingerprint"])
