"""Paired PTB measured-Frank versus ECG-derived Kors VCG experiment."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score

from .benchmark import (
    aggregate_patients, fit_estimator, make_estimator, metric_bundle,
    patient_balanced_weights, predict_probabilities,
)
from .data import download_file, read_vcg
from .external import ECG_LEADS, kors_transform
from .features import tensorize_beat, wavelet_statistical_features
from .signal import denoise_vcg, detect_r_peaks


def _load_yaml(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _download_ecg(record: str, raw_dir: Path, base_url: str) -> None:
    download_file(f"{base_url.rstrip('/')}/{record}.dat", raw_dir / f"{record}.dat")


def download_paired_ecg(config: dict, records: list[str]) -> None:
    paper = _load_yaml(config["paper_config"])
    data = paper["data"]
    raw_dir = Path(data["raw_dir"])
    with ThreadPoolExecutor(max_workers=int(config["download_workers"])) as pool:
        futures = [pool.submit(_download_ecg, record, raw_dir, data["database_url"]) for record in records]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 25 == 0 or completed == len(futures):
                print(f"Downloaded paired ECG {completed}/{len(futures)}", flush=True)


def _peak_recall(reference: np.ndarray, candidate: np.ndarray, tolerance: int) -> float:
    if len(reference) == 0:
        return float("nan")
    if len(candidate) == 0:
        return 0.0
    positions = np.searchsorted(candidate, reference)
    left = candidate[np.maximum(positions - 1, 0)]
    right = candidate[np.minimum(positions, len(candidate) - 1)]
    nearest = np.minimum(np.abs(reference - left), np.abs(reference - right))
    return float(np.mean(nearest <= tolerance))


def _paired_macro_f1_intervals(
    truth: np.ndarray, scores: dict[str, np.ndarray], n_classes: int, seed: int, draws: int = 5000,
) -> dict[str, dict[str, float | list[float]]]:
    """Class-stratified patient bootstrap, paired across all model conditions."""
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(truth == label) for label in range(n_classes)]
    predictions = {name: values.argmax(axis=1) for name, values in scores.items()}
    comparisons = {
        "measured_to_derived_minus_measured_to_measured": ("measured_to_derived", "measured_to_measured"),
        "derived_to_derived_minus_measured_to_derived": ("derived_to_derived", "measured_to_derived"),
    }
    distributions = {name: np.empty(draws, dtype=float) for name in comparisons}
    labels = np.arange(n_classes)
    for draw in range(draws):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in class_indices])
        f1 = {
            name: f1_score(truth[sample], predicted[sample], labels=labels, average="macro", zero_division=0)
            for name, predicted in predictions.items()
        }
        for name, (left, right) in comparisons.items():
            distributions[name][draw] = f1[left] - f1[right]
    observed = {
        name: f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        for name, predicted in predictions.items()
    }
    return {
        name: {
            "observed_delta": float(observed[left] - observed[right]),
            "bootstrap_95_percentile_interval": [float(value) for value in np.quantile(distributions[name], [0.025, 0.975])],
        }
        for name, (left, right) in comparisons.items()
    }


def _paired_record(record: str, rows: pd.DataFrame, paper: dict, tolerance_ms: int) -> tuple[np.ndarray, np.ndarray, dict]:
    data, signal, features = paper["data"], paper["signal"], paper["features"]
    path = Path(data["raw_dir"]) / record
    ecg = wfdb.rdrecord(str(path), channel_names=list(ECG_LEADS))
    measured, fs = read_vcg(Path(data["raw_dir"]), record, list(data["leads"]))
    if int(ecg.fs) != fs or ecg.p_signal.shape[0] != measured.shape[0] or fs != 1000:
        raise ValueError(f"Paired signals have incompatible length/rate: {record}")
    derived = kors_transform(ecg.p_signal, ecg.sig_name)
    centered_measured = measured - measured.mean(axis=0)
    centered_derived = derived - derived.mean(axis=0)
    rms_measured = np.sqrt(np.mean(centered_measured ** 2, axis=0))
    rms_derived = np.sqrt(np.mean(centered_derived ** 2, axis=0))
    correlation = np.sum(centered_measured * centered_derived, axis=0) / np.sqrt(
        np.sum(centered_measured ** 2, axis=0) * np.sum(centered_derived ** 2, axis=0)
    )
    relative_rmse = np.sqrt(np.mean((centered_measured - centered_derived) ** 2, axis=0)) / rms_measured
    cleaned = denoise_vcg(derived, signal["denoise_wavelet"], int(signal["denoise_level"]))
    detected = detect_r_peaks(
        cleaned, fs, int(signal["rpeak_refractory_ms"]),
        int(signal["rpeak_integration_ms"]), float(signal["rpeak_threshold_scale"]),
    )
    before, after = int(data["before_r"]), int(data["after_r"])
    detected = detected[(detected >= before) & (detected + after < len(derived))]
    shared = rows.r_peak.to_numpy(dtype=int)
    vectors = [
        wavelet_statistical_features(tensorize_beat(
            cleaned[peak - before : peak + after + 1].T,
            features["wavelet"], int(features["level"]), tuple(features["detail_levels"]),
        ))
        for peak in shared
    ]
    result = {
        "record": record, "patient_id": str(rows.patient_id.iloc[0]), "label": str(rows.label.iloc[0]),
        "samples": len(measured), "shared_beats": len(shared), "derived_detected_beats": len(detected),
        "derived_peak_recall": _peak_recall(shared, detected, int(tolerance_ms * fs / 1000)),
    }
    for lead, index in zip("xyz", range(3), strict=True):
        result[f"correlation_{lead}"] = float(correlation[index])
        result[f"relative_rmse_{lead}"] = float(relative_rmse[index])
        result[f"rms_ratio_{lead}"] = float(rms_derived[index] / rms_measured[index])
    return rows.index.to_numpy(), np.vstack(vectors), result


def make_paired_features(config: dict) -> Path:
    paper = _load_yaml(config["paper_config"])
    beats = pd.read_csv(config["beats_path"])
    beats = beats[beats.label.isin(config["classes"])].reset_index(drop=True)
    groups = list(beats.groupby("record", sort=False))
    download_paired_ecg(config, [record for record, _ in groups])
    outputs = Parallel(n_jobs=int(config["feature_workers"]), return_as="generator_unordered")(
        delayed(_paired_record)(record, rows, paper, int(config["rpeak_tolerance_ms"]))
        for record, rows in groups
    )
    matrix: np.ndarray | None = None
    signal_rows = []
    for completed, (indices, vectors, diagnostics) in enumerate(outputs, 1):
        if matrix is None:
            matrix = np.empty((len(beats), vectors.shape[1]), dtype=np.float32)
        matrix[indices] = vectors
        signal_rows.append(diagnostics)
        if completed % 25 == 0 or completed == len(groups):
            print(f"Paired features {completed}/{len(groups)}", flush=True)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Paired feature matrix is empty or non-finite")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(signal_rows).sort_values("record").to_csv(output / "paired_signal_diagnostics.csv", index=False)
    path = output / "paired_features.npz"
    np.savez_compressed(path, X=matrix, y=beats.label.to_numpy(dtype="U5"), groups=beats.patient_id.to_numpy(dtype="U20"))
    return path


def evaluate_paired(config: dict) -> dict:
    external = _load_yaml(config["external_config"])
    measured = np.load(config["measured_features_path"])
    derived = np.load(Path(config["output_dir"]) / "paired_features.npz")
    classes = list(config["classes"])
    selected = np.isin(measured["y"], classes)
    X_measured = measured["wavelet_statistics"][selected]
    X_derived = derived["X"]
    y_text, groups = measured["y"][selected], measured["groups"][selected]
    if not (np.array_equal(y_text, derived["y"]) and np.array_equal(groups, derived["groups"])):
        raise ValueError("Measured and derived features are not aligned beat-for-beat")
    mapping = {name: index for index, name in enumerate(classes)}
    y = np.array([mapping[label] for label in y_text], dtype=np.int16)
    folds = pd.read_csv(config["folds_path"])
    lookup = dict(zip(folds.patient_id, folds.fold, strict=True))
    beat_folds = np.array([lookup[group] for group in groups], dtype=int)
    conditions = ("measured_to_measured", "measured_to_derived", "derived_to_derived")
    oof = {condition: np.zeros((len(y), len(classes)), dtype=np.float32) for condition in conditions}
    for fold in sorted(np.unique(beat_folds)):
        train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
        for domain, X_train in (("measured", X_measured), ("derived", X_derived)):
            model = make_estimator("xgboost", dict(external["model"]), int(config["seed"]) + int(fold), len(classes))
            fit_estimator(model, "xgboost", X_train[train], y[train], patient_balanced_weights(y[train], groups[train]))
            if domain == "measured":
                oof["measured_to_measured"][test] = predict_probabilities(model, X_measured[test], len(classes))
                oof["measured_to_derived"][test] = predict_probabilities(model, X_derived[test], len(classes))
            else:
                oof["derived_to_derived"][test] = predict_probabilities(model, X_derived[test], len(classes))
        print(f"Evaluated paired fold {fold}/{len(np.unique(beat_folds))}", flush=True)
    result = {
        "protocol": "same PTB patients and beat positions; fixed patient folds; no PTB-XL data used",
        "classes": classes, "patients": int(len(np.unique(groups))), "beats": int(len(y)),
        "model_config": dict(external["model"]),
        "rpeak_tolerance_ms": int(config["rpeak_tolerance_ms"]), "conditions": {},
    }
    measured_centered = X_measured.astype(np.float64) - X_measured.mean(axis=0)
    derived_centered = X_derived.astype(np.float64) - X_derived.mean(axis=0)
    measured_scale = np.sqrt(np.mean(measured_centered ** 2, axis=0))
    derived_scale = np.sqrt(np.mean(derived_centered ** 2, axis=0))
    valid = (measured_scale > 1e-8) & (derived_scale > 1e-8)
    feature_correlations = np.mean(measured_centered[:, valid] * derived_centered[:, valid], axis=0) / (
        measured_scale[valid] * derived_scale[valid]
    )
    result["paired_feature_shift"] = {
        "median_feature_correlation": float(np.median(feature_correlations)),
        "median_absolute_gap_in_measured_sd": float(np.median(
            np.abs(X_measured[:, valid] - X_derived[:, valid]) / measured_scale[valid]
        )),
        "features_with_nonzero_variance": int(valid.sum()),
    }
    output = Path(config["output_dir"])
    patient_scores: dict[str, np.ndarray] = {}
    patient_truth: np.ndarray | None = None
    for condition in conditions:
        ids, truth, scores = aggregate_patients(oof[condition], y, groups)
        patient_scores[condition] = scores
        patient_truth = truth
        predictions = scores.argmax(axis=1)
        result["conditions"][condition] = {"patient_metrics": metric_bundle(truth, scores, classes)}
        pd.DataFrame(confusion_matrix(truth, predictions, labels=np.arange(len(classes))), index=classes, columns=classes).to_csv(
            output / f"confusion_{condition}.csv"
        )
        pd.DataFrame({
            "patient_id": ids, "truth": [classes[value] for value in truth],
            "prediction": [classes[value] for value in predictions],
            **{f"probability_{name}": scores[:, index] for index, name in enumerate(classes)},
        }).to_csv(output / f"patient_predictions_{condition}.csv", index=False)
    assert patient_truth is not None
    result["paired_macro_f1_differences"] = _paired_macro_f1_intervals(
        patient_truth, patient_scores, len(classes), int(config["seed"])
    )
    diagnostics = pd.read_csv(output / "paired_signal_diagnostics.csv")
    result["signal_medians"] = {
        column: float(diagnostics[column].median())
        for column in diagnostics.columns if column.startswith(("correlation_", "relative_rmse_", "rms_ratio_", "derived_peak_recall_"))
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
