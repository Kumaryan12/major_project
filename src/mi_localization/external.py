"""Locked PTB-trained evaluation on PTB-XL's recommended patient test fold."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed
from scipy.signal import resample_poly
from sklearn.metrics import confusion_matrix

from .benchmark import (
    aggregate_patients,
    fit_estimator,
    make_estimator,
    metric_bundle,
    patient_balanced_weights,
    predict_probabilities,
)
from .data import download_file
from .features import tensorize_beat, wavelet_statistical_features
from .signal import denoise_vcg, detect_r_peaks, extract_beats


ECG_LEADS = ("i", "ii", "v1", "v2", "v3", "v4", "v5", "v6")
# Kors regression ECG -> XYZ coefficients, in ECG_LEADS order.
# Source: Tereshchenko Lab's published kors.m implementation.
KORS_MATRIX = np.array([
    [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
    [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
    [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31],
], dtype=np.float64)


def kors_transform(ecg: np.ndarray, signal_names: list[str]) -> np.ndarray:
    names = [name.lower() for name in signal_names]
    missing = [lead for lead in ECG_LEADS if lead not in names]
    if missing:
        raise ValueError(f"ECG leads missing for Kors transform: {missing}")
    selected = ecg[:, [names.index(lead) for lead in ECG_LEADS]]
    return selected @ KORS_MATRIX.T


def _download_record(row: object, config: dict) -> None:
    relative = str(row.filename_hr)
    base = config["base_url"].rstrip("/")
    target = Path(config["waveform_dir"]) / relative
    for suffix in (".hea", ".dat"):
        download_file(f"{base}/{relative}{suffix}", Path(f"{target}{suffix}"))


def download_locked_test(config: dict) -> pd.DataFrame:
    manifest = pd.read_csv(Path(config["audit_dir"]) / "locked_test_manifest.csv", dtype={"patient_id": str})
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(_download_record, row, config) for row in manifest.itertuples()]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 50 == 0 or completed == len(futures):
                print(f"Downloaded {completed}/{len(futures)} locked PTB-XL records", flush=True)
    return manifest


def _record_features(row: object, config: dict) -> tuple[int, str, str, np.ndarray]:
    path = Path(config["waveform_dir"]) / str(row.filename_hr)
    record = wfdb.rdrecord(str(path))
    if record.fs != 500 or record.p_signal.shape != (5000, 12):
        raise ValueError(f"Unexpected PTB-XL waveform shape/rate for {path}: {record.p_signal.shape}, {record.fs}")
    vcg = kors_transform(record.p_signal, record.sig_name)
    settings = config["preprocessing"]
    target_fs = int(settings["target_sampling_rate"])
    if target_fs != 1000:
        raise ValueError("Current PTB-trained feature extractor requires 1000 Hz")
    vcg = resample_poly(vcg, 2, 1, axis=0)
    clean = denoise_vcg(vcg, settings["wavelet"], int(settings["level"]))
    peaks = detect_r_peaks(
        clean, target_fs,
        int(settings["rpeak_refractory_ms"]),
        int(settings["rpeak_integration_ms"]),
        float(settings["rpeak_threshold_scale"]),
    )
    beats, _ = extract_beats(clean, peaks, int(settings["before_r"]), int(settings["after_r"]))
    vectors = [
        wavelet_statistical_features(tensorize_beat(
            beat,
            settings["wavelet"],
            int(settings["level"]),
            tuple(settings["detail_levels"]),
        ))
        for beat in beats
    ]
    if not vectors:
        raise ValueError(f"No complete beats detected in {path}")
    return int(row.ecg_id), str(row.patient_id), str(row.label), np.vstack(vectors)


def make_external_features(config: dict) -> Path:
    manifest = download_locked_test(config)
    results = Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(_record_features)(row, config) for row in manifest.itertuples()
    )
    chunks = []
    rows = []
    for completed, (ecg_id, patient_id, label, matrix) in enumerate(results, 1):
        chunks.append(matrix)
        rows.extend([{"ecg_id": ecg_id, "patient_id": patient_id, "label": label}] * len(matrix))
        if completed % 50 == 0 or completed == len(manifest):
            print(f"Featurized {completed}/{len(manifest)} PTB-XL records; beats={len(rows)}", flush=True)
    destination = Path(config["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "external_features.npz",
        X=np.vstack(chunks).astype(np.float32),
        y=np.asarray([row["label"] for row in rows], dtype="U5"),
        groups=np.asarray([row["patient_id"] for row in rows], dtype="U20"),
        ecg_ids=np.asarray([row["ecg_id"] for row in rows], dtype=np.int32),
    )
    return destination / "external_features.npz"


def evaluate_external(config: dict) -> dict:
    classes = list(config["classes"])
    train = np.load(config["ptb_features_path"])
    external = np.load(Path(config["output_dir"]) / "external_features.npz")
    selected = np.isin(train["y"], classes)
    mapping = {name: index for index, name in enumerate(classes)}
    X_train = train[config["ptb_feature_key"]][selected]
    y_train = np.asarray([mapping[str(label)] for label in train["y"][selected]], dtype=int)
    groups_train = train["groups"][selected]
    X_test = external["X"]
    y_test = np.asarray([mapping[str(label)] for label in external["y"]], dtype=int)
    groups_test = external["groups"]
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and external feature dimensions differ")
    model = make_estimator("xgboost", dict(config["model"]), int(config["seed"]), len(classes))
    fit_estimator(model, "xgboost", X_train, y_train, patient_balanced_weights(y_train, groups_train))
    probabilities = predict_probabilities(model, X_test, len(classes))
    patient_ids, patient_truth, patient_probabilities = aggregate_patients(probabilities, y_test, groups_test)
    result = {
        "protocol": "PTB measured Frank XYZ training; PTB-XL fold-10 ECG transformed to Kors XYZ; zero PTB-XL tuning",
        "representation": str(config["ptb_feature_key"]),
        "model_config": dict(config["model"]),
        "test_fold": int(config["test_fold"]),
        "classes": classes,
        "train_patients": int(len(np.unique(groups_train))),
        "train_beats": int(len(y_train)),
        "test_patients": int(len(patient_ids)),
        "test_records": int(len(np.unique(external["ecg_ids"]))),
        "test_beats": int(len(y_test)),
        "patient_metrics": metric_bundle(patient_truth, patient_probabilities, classes),
        "beat_metrics": metric_bundle(y_test, probabilities, classes),
    }
    output = Path(config["output_dir"])
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions = patient_probabilities.argmax(axis=1)
    frame = pd.DataFrame({
        "patient_id": patient_ids,
        "truth": [classes[value] for value in patient_truth],
        "prediction": [classes[value] for value in predictions],
        **{f"probability_{name}": patient_probabilities[:, index] for index, name in enumerate(classes)},
    })
    frame.to_csv(output / "patient_predictions.csv", index=False)
    matrix = confusion_matrix(patient_truth, predictions, labels=np.arange(len(classes)))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output / "confusion_patient.csv")
    joblib.dump(model, output / "ptb_trained_xgboost.joblib")
    return result
