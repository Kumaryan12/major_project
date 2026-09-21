"""Representation ablations on fixed, leakage-free patient folds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .benchmark import (
    aggregate_patients,
    fit_estimator,
    make_estimator,
    metric_bundle,
    patient_balanced_weights,
    predict_probabilities,
)
from .data import read_vcg
from .features import (
    normalize_vector_rms,
    tensorize_beat,
    tucker_time_features,
    tucker_time_features_multi,
    wavelet_statistical_features,
)
from .signal import denoise_vcg


REPRESENTATION_ORDER = [
    "tucker_clean_r1", "tucker_clean_r2", "tucker_clean_r3",
    "tucker_clean_r5", "tucker_clean_r8", "tucker_raw_r3",
    "tucker_vector_norm_r3", "tucker_x_r3", "tucker_y_r3",
    "tucker_z_r3", "tucker_no_wavelet_r3", "wavelet_statistics",
]


def _paper_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def beat_representations(clean_beat: np.ndarray, raw_beat: np.ndarray, feature_cfg: dict) -> dict[str, np.ndarray]:
    kwargs = {
        "wavelet": feature_cfg["wavelet"],
        "level": int(feature_cfg["level"]),
        "detail_levels": tuple(feature_cfg["detail_levels"]),
    }
    canonical = bool(feature_cfg["canonicalize_svd_sign"])
    clean_tensor = tensorize_beat(clean_beat, **kwargs)
    clean_ranks = tucker_time_features_multi(clean_tensor, (1, 2, 3, 5, 8), canonical)
    raw_tensor = tensorize_beat(raw_beat, **kwargs)
    normalized_tensor = tensorize_beat(normalize_vector_rms(clean_beat), **kwargs)
    result = {f"tucker_clean_r{rank}": values for rank, values in clean_ranks.items()}
    result.update({
        "tucker_raw_r3": tucker_time_features(raw_tensor, 3, canonical),
        "tucker_vector_norm_r3": tucker_time_features(normalized_tensor, 3, canonical),
        "tucker_x_r3": tucker_time_features(clean_tensor[0:1], 3, canonical),
        "tucker_y_r3": tucker_time_features(clean_tensor[1:2], 3, canonical),
        "tucker_z_r3": tucker_time_features(clean_tensor[2:3], 3, canonical),
        "tucker_no_wavelet_r3": tucker_time_features(clean_beat[:, :, None], 3, canonical),
        "wavelet_statistics": wavelet_statistical_features(clean_tensor),
    })
    return result


def _record_feature_rows(record: str, rows: pd.DataFrame, paper: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data, signal_cfg, feature_cfg = paper["data"], paper["signal"], paper["features"]
    raw, fs = read_vcg(Path(data["raw_dir"]), record, list(data["leads"]))
    clean = denoise_vcg(raw, signal_cfg["denoise_wavelet"], int(signal_cfg["denoise_level"]))
    before, after = int(data["before_r"]), int(data["after_r"])
    indices = rows.index.to_numpy()
    accumulated = {name: [] for name in REPRESENTATION_ORDER}
    for peak in rows.r_peak.astype(int):
        raw_beat = raw[peak - before : peak + after + 1].T
        clean_beat = clean[peak - before : peak + after + 1].T
        values = beat_representations(clean_beat, raw_beat, feature_cfg)
        for name in REPRESENTATION_ORDER:
            accumulated[name].append(values[name])
    return indices, {name: np.vstack(values) for name, values in accumulated.items()}


def build_feature_bank(config: dict) -> Path:
    paper = _paper_config(config["paper_config"])
    beats = pd.read_csv(config["beats_path"])
    outputs: dict[str, np.ndarray | None] = {name: None for name in REPRESENTATION_ORDER}
    tasks = Parallel(n_jobs=int(paper["features"].get("n_jobs", -1)), return_as="generator_unordered")(
        delayed(_record_feature_rows)(record, rows, paper) for record, rows in beats.groupby("record", sort=False)
    )
    for completed, (indices, values) in enumerate(tasks, 1):
        for name, matrix in values.items():
            if outputs[name] is None:
                outputs[name] = np.empty((len(beats), matrix.shape[1]), dtype=np.float32)
            outputs[name][indices] = matrix
        if completed % 25 == 0 or completed == beats.record.nunique():
            print(f"Ablation features: {completed}/{beats.record.nunique()} records", flush=True)
    destination = Path(config["feature_bank_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **{name: matrix for name, matrix in outputs.items() if matrix is not None},
        y=beats.label.astype(str).to_numpy(dtype="U"),
        groups=beats.patient_id.astype(str).to_numpy(dtype="U"),
    )
    return destination


def _save_ablation_confusion(truth: np.ndarray, prediction: np.ndarray, classes: list[str], name: str, output_dir: Path) -> None:
    matrix = confusion_matrix(truth, prediction, labels=np.arange(len(classes)))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output_dir / f"confusion_{name}.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=name.replace("_", " ").title())
    fig.tight_layout(); fig.savefig(output_dir / f"confusion_{name}.png", dpi=180); plt.close(fig)


def evaluate_representation(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: pd.DataFrame, config: dict) -> dict:
    classes = list(config["classes"])
    n_classes = len(classes)
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.asarray([lookup[str(group)] for group in groups])
    probabilities = np.zeros((len(y), n_classes), dtype=np.float64)
    fold_rows = []
    for fold in range(1, int(config["outer_splits"]) + 1):
        train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
        estimator = make_estimator("xgboost", dict(config["model"]), int(config["seed"]) + fold, n_classes)
        fit_estimator(estimator, "xgboost", X[train], y[train], patient_balanced_weights(y[train], groups[train]))
        probabilities[test] = predict_probabilities(estimator, X[test], n_classes)
        _, truth, patient_scores = aggregate_patients(probabilities[test], y[test], groups[test])
        prediction = patient_scores.argmax(axis=1)
        fold_rows.append({
            "fold": fold,
            "patient_accuracy": float(accuracy_score(truth, prediction)),
            "patient_macro_f1": float(f1_score(truth, prediction, labels=np.arange(n_classes), average="macro", zero_division=0)),
        })
    patient_ids, truth, patient_scores = aggregate_patients(probabilities, y, groups)
    prediction = patient_scores.argmax(axis=1)
    result = {
        "representation": name,
        "feature_count": int(X.shape[1]),
        "n_patients": int(len(patient_ids)),
        "patient_metrics": metric_bundle(truth, patient_scores, classes),
        "beat_metrics": metric_bundle(y, probabilities, classes),
        "folds": fold_rows,
        "protocol": "exploratory fixed XGBoost on frozen outer folds; no ablation-specific tuning",
    }
    output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"metrics_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _save_ablation_confusion(truth, prediction, classes, name, output_dir)
    print(f"{name}: accuracy={result['patient_metrics']['accuracy']:.4f}, macro F1={result['patient_metrics']['macro_f1']:.4f}", flush=True)
    return result


def run_ablation(config: dict, representations: list[str] | None = None) -> pd.DataFrame:
    arrays = np.load(config["feature_bank_path"])
    classes = list(config["classes"]); mapping = {name: index for index, name in enumerate(classes)}
    mask = np.isin(arrays["y"], classes)
    y = np.asarray([mapping[str(label)] for label in arrays["y"][mask]], dtype=np.int16)
    groups = arrays["groups"][mask]
    folds = pd.read_csv(config["folds_path"])
    selected = representations or list(config["representations"])
    results = [evaluate_representation(name, arrays[name][mask], y, groups, folds, config) for name in selected]
    rows = []
    for result in results:
        rows.append({
            "representation": result["representation"],
            "feature_count": result["feature_count"],
            **{f"patient_{key}": value for key, value in result["patient_metrics"].items() if isinstance(value, float)},
            **{f"beat_{key}": value for key, value in result["beat_metrics"].items() if isinstance(value, float)},
        })
    comparison = pd.DataFrame(rows).sort_values("patient_macro_f1", ascending=False)
    comparison.to_csv(Path(config["output_dir"]) / "comparison.csv", index=False)
    return comparison
