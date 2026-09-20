"""End-to-end orchestration."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from .data import load_manifest, prepare_database, read_vcg
from .evaluation import evaluate_cv
from .features import extract_features
from .signal import denoise_vcg, detect_r_peaks, extract_beats


def prepare(config: dict) -> list[dict[str, str]]:
    data = config["data"]
    return prepare_database(
        data["database_url"],
        Path(data["raw_dir"]),
        set(data.get("excluded_patients", [])),
    )


def _featurize_record(item: dict[str, str], data: dict, signal_cfg: dict, feature_cfg: dict) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    raw_dir = Path(data["raw_dir"])
    vcg, fs = read_vcg(raw_dir, item["record"], list(data["leads"]))
    if fs != int(data["sampling_rate"]):
        raise ValueError(f"{item['record']}: expected {data['sampling_rate']} Hz, got {fs}")
    clean = denoise_vcg(vcg, signal_cfg["denoise_wavelet"], int(signal_cfg["denoise_level"]))
    peaks = detect_r_peaks(
        clean, fs,
        int(signal_cfg["rpeak_refractory_ms"]),
        int(signal_cfg["rpeak_integration_ms"]),
        float(signal_cfg["rpeak_threshold_scale"]),
    )
    beats, valid_peaks = extract_beats(clean, peaks, int(data["before_r"]), int(data["after_r"]))
    maximum = data.get("max_beats_per_record")
    if maximum is not None:
        beats, valid_peaks = beats[: int(maximum)], valid_peaks[: int(maximum)]
    vectors = [
        extract_features(
            beat,
            wavelet=feature_cfg["wavelet"],
            level=int(feature_cfg["level"]),
            detail_levels=tuple(feature_cfg["detail_levels"]),
            time_rank=int(feature_cfg["time_rank"]),
            canonicalize_svd_sign=bool(feature_cfg["canonicalize_svd_sign"]),
        )
        for beat in beats
    ]
    rows = [
        {**item, "beat_index": beat_index, "r_peak": int(peak)}
        for beat_index, peak in enumerate(valid_peaks)
    ]
    return vectors, rows


def featurize(config: dict) -> Path:
    data, signal_cfg, feature_cfg = config["data"], config["signal"], config["features"]
    raw_dir = Path(data["raw_dir"])
    manifest = load_manifest(raw_dir / "manifest.csv")
    vectors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    jobs = Parallel(n_jobs=int(feature_cfg.get("n_jobs", -1)), return_as="generator_unordered")(
        delayed(_featurize_record)(item, data, signal_cfg, feature_cfg) for item in manifest
    )
    for record_index, (record_vectors, record_rows) in enumerate(jobs, 1):
        vectors.extend(record_vectors)
        rows.extend(record_rows)
        print(f"Featurized {record_index}/{len(manifest)} records; total beats={len(vectors)}", flush=True)
    processed = Path(data["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    feature_path = processed / "paper_features.npz"
    np.savez_compressed(
        feature_path,
        X=np.vstack(vectors),
        y=np.asarray([row["label"] for row in rows]),
        groups=np.asarray([row["patient_id"] for row in rows]),
    )
    with (processed / "beats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return feature_path


def evaluate(config: dict, modes: list[str] | None = None) -> dict[str, dict]:
    path = Path(config["data"]["processed_dir"]) / "paper_features.npz"
    arrays = np.load(path)
    chosen = modes or list(config["evaluation"]["modes"])
    output = Path("results")
    return {
        mode: evaluate_cv(arrays["X"], arrays["y"], arrays["groups"], mode, {**config["model"], **config["evaluation"]}, int(config["seed"]), output)
        for mode in chosen
    }
