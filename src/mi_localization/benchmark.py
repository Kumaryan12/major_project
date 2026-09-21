"""Leakage-free, nested patient-level benchmark for the viable six-class task."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def load_primary_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.load(config["features_path"])
    feature_key = config.get("feature_key", "X")
    if feature_key not in arrays.files:
        raise KeyError(f"Feature key {feature_key!r} not found in {config['features_path']}")
    classes = list(config["classes"])
    mask = np.isin(arrays["y"], classes)
    X = arrays[feature_key][mask]
    y_text = arrays["y"][mask]
    groups = arrays["groups"][mask]
    mapping = {name: index for index, name in enumerate(classes)}
    y = np.asarray([mapping[str(label)] for label in y_text], dtype=np.int16)
    return X, y, groups


def patient_table(y: np.ndarray, groups: np.ndarray, classes: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"patient_id": groups, "label_index": y})
    uniqueness = frame.groupby("patient_id").label_index.nunique()
    if (uniqueness != 1).any():
        raise ValueError("A patient has more than one class label")
    patients = frame.drop_duplicates("patient_id").sort_values("patient_id").reset_index(drop=True)
    patients["label"] = patients.label_index.map(dict(enumerate(classes)))
    return patients


def create_fixed_folds(y: np.ndarray, groups: np.ndarray, classes: list[str], path: Path, n_splits: int, seed: int) -> pd.DataFrame:
    patients = patient_table(y, groups, classes)
    folds = np.full(len(patients), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(patients.patient_id, patients.label_index), 1):
        folds[test] = fold
    patients["fold"] = folds
    path.parent.mkdir(parents=True, exist_ok=True)
    patients[["patient_id", "label", "fold"]].to_csv(path, index=False)
    return patients[["patient_id", "label", "fold"]]


def load_or_create_folds(y: np.ndarray, groups: np.ndarray, config: dict) -> pd.DataFrame:
    path = Path(config["folds_path"])
    expected = patient_table(y, groups, list(config["classes"]))[["patient_id", "label"]]
    if not path.exists():
        return create_fixed_folds(
            y, groups, list(config["classes"]), path,
            int(config["outer_splits"]), int(config["seed"]),
        )
    folds = pd.read_csv(path, dtype={"patient_id": str, "label": str, "fold": int})
    merged = expected.merge(folds, on=["patient_id", "label"], how="left", validate="one_to_one")
    if merged.fold.isna().any() or len(folds) != len(expected):
        raise ValueError(f"Existing fold file {path} does not match the selected cohort")
    return merged


def patient_balanced_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Give every class equal total weight and every patient equal weight within class."""
    weights = np.zeros(len(y), dtype=np.float64)
    classes = np.unique(y)
    for label in classes:
        class_mask = y == label
        class_patients = np.unique(groups[class_mask])
        for patient in class_patients:
            mask = class_mask & (groups == patient)
            weights[mask] = 1.0 / (len(classes) * len(class_patients) * int(mask.sum()))
    return weights * len(weights)


def make_estimator(model_name: str, params: dict, seed: int, n_classes: int) -> BaseEstimator:
    if model_name == "rf":
        return RandomForestClassifier(
            **params,
            bootstrap=True,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(**params, class_weight=None, dual="auto", random_state=seed, max_iter=10000),
        )
    if model_name == "xgboost":
        return XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def fit_estimator(estimator: BaseEstimator, model_name: str, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> BaseEstimator:
    if model_name == "svm":
        estimator.fit(X, y, linearsvc__sample_weight=weights)
    else:
        estimator.fit(X, y, sample_weight=weights)
    return estimator


def predict_probabilities(estimator: BaseEstimator, X: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(X)
    else:
        raw = softmax(estimator.decision_function(X), axis=1)
    aligned = np.zeros((len(X), n_classes), dtype=np.float64)
    for source, label in enumerate(estimator.classes_):
        aligned[:, int(label)] = raw[:, source]
    return aligned


def aggregate_patients(probabilities: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.unique(groups)
    aggregated = np.zeros((len(ids), probabilities.shape[1]), dtype=np.float64)
    truth = np.zeros(len(ids), dtype=int)
    for index, patient in enumerate(ids):
        mask = groups == patient
        labels = np.unique(y[mask])
        if len(labels) != 1:
            raise ValueError(f"Patient {patient} has inconsistent labels")
        truth[index] = int(labels[0])
        aggregated[index] = probabilities[mask].mean(axis=0)
    return ids, truth, aggregated


def inner_patient_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    patients = patient_table(y, groups, [str(i) for i in range(int(y.max()) + 1)])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_patients, valid_patients in splitter.split(patients.patient_id, patients.label_index):
        train_ids = set(patients.iloc[train_patients].patient_id)
        valid_ids = set(patients.iloc[valid_patients].patient_id)
        yield np.flatnonzero(np.isin(groups, list(train_ids))), np.flatnonzero(np.isin(groups, list(valid_ids)))


def select_parameters(
    model_name: str,
    candidates: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    inner_splits: int,
    seed: int,
    n_classes: int,
) -> tuple[dict, float, list[dict]]:
    splits = list(inner_patient_splits(y, groups, inner_splits, seed))
    candidate_results: list[dict] = []
    for candidate_index, params in enumerate(candidates):
        scores = []
        for split_index, (train, valid) in enumerate(splits):
            estimator = make_estimator(model_name, params, seed + candidate_index * 100 + split_index, n_classes)
            weights = patient_balanced_weights(y[train], groups[train])
            fit_estimator(estimator, model_name, X[train], y[train], weights)
            probabilities = predict_probabilities(estimator, X[valid], n_classes)
            _, truth, patient_probabilities = aggregate_patients(probabilities, y[valid], groups[valid])
            prediction = patient_probabilities.argmax(axis=1)
            scores.append(float(f1_score(truth, prediction, labels=np.arange(n_classes), average="macro", zero_division=0)))
        candidate_results.append({"params": params, "fold_scores": scores, "mean_patient_macro_f1": float(np.mean(scores))})
    best = max(candidate_results, key=lambda row: row["mean_patient_macro_f1"])
    return dict(best["params"]), float(best["mean_patient_macro_f1"]), candidate_results


def metric_bundle(truth: np.ndarray, probabilities: np.ndarray, classes: list[str]) -> dict:
    prediction = probabilities.argmax(axis=1)
    binary = label_binarize(truth, classes=np.arange(len(classes)))
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=np.arange(len(classes)), average="macro", zero_division=0)),
        "macro_roc_auc_ovr": float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")),
        "macro_average_precision": float(average_precision_score(binary, probabilities, average="macro")),
        "classification_report": classification_report(
            truth, prediction, labels=np.arange(len(classes)), target_names=classes,
            output_dict=True, zero_division=0,
        ),
    }


def save_confusion(truth: np.ndarray, prediction: np.ndarray, classes: list[str], model_name: str, output_dir: Path) -> None:
    matrix = confusion_matrix(truth, prediction, labels=np.arange(len(classes)))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output_dir / f"confusion_patient_{model_name}.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=f"Patient-independent nested CV: {model_name.upper()}")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_patient_{model_name}.png", dpi=180)
    plt.close(fig)


def run_model_benchmark(model_name: str, config: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: pd.DataFrame) -> dict:
    classes = list(config["classes"])
    n_classes = len(classes)
    fold_lookup = dict(zip(folds.patient_id, folds.fold, strict=True))
    beat_folds = np.asarray([fold_lookup[str(patient)] for patient in groups], dtype=int)
    oof_probabilities = np.zeros((len(y), n_classes), dtype=np.float64)
    fold_results = []
    tuning_results: dict[str, list[dict]] = {}
    for outer_fold in range(1, int(config["outer_splits"]) + 1):
        train = np.flatnonzero(beat_folds != outer_fold)
        test = np.flatnonzero(beat_folds == outer_fold)
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError(f"Patient leakage in outer fold {outer_fold}")
        params, inner_score, candidates = select_parameters(
            model_name, list(config["models"][model_name]),
            X[train], y[train], groups[train], int(config["inner_splits"]),
            int(config["seed"]) + outer_fold * 1000, n_classes,
        )
        estimator = make_estimator(model_name, params, int(config["seed"]) + outer_fold, n_classes)
        weights = patient_balanced_weights(y[train], groups[train])
        fit_estimator(estimator, model_name, X[train], y[train], weights)
        probabilities = predict_probabilities(estimator, X[test], n_classes)
        oof_probabilities[test] = probabilities
        patient_ids, patient_truth, patient_probabilities = aggregate_patients(probabilities, y[test], groups[test])
        patient_prediction = patient_probabilities.argmax(axis=1)
        fold_results.append({
            "fold": outer_fold,
            "best_params": params,
            "inner_patient_macro_f1": inner_score,
            "test_patient_accuracy": float(accuracy_score(patient_truth, patient_prediction)),
            "test_patient_macro_f1": float(f1_score(patient_truth, patient_prediction, labels=np.arange(n_classes), average="macro", zero_division=0)),
            "train_patients": int(len(np.unique(groups[train]))),
            "test_patients": int(len(patient_ids)),
            "patient_overlap": 0,
        })
        tuning_results[str(outer_fold)] = candidates
        print(
            f"{model_name} outer fold {outer_fold}/{config['outer_splits']}: "
            f"patient accuracy={fold_results[-1]['test_patient_accuracy']:.4f}, "
            f"macro F1={fold_results[-1]['test_patient_macro_f1']:.4f}",
            flush=True,
        )

    patient_ids, patient_truth, patient_probabilities = aggregate_patients(oof_probabilities, y, groups)
    patient_prediction = patient_probabilities.argmax(axis=1)
    result = {
        "model": model_name,
        "classes": classes,
        "n_beats": int(len(y)),
        "n_patients": int(len(patient_ids)),
        "weighting": "equal classes, equal patients within class",
        "selection_metric": config["selection_metric"],
        "patient_metrics": metric_bundle(patient_truth, patient_probabilities, classes),
        "beat_metrics": metric_bundle(y, oof_probabilities, classes),
        "folds": fold_results,
        "inner_tuning": tuning_results,
    }
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"metrics_{model_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({
        "patient_id": patient_ids,
        "truth": [classes[index] for index in patient_truth],
        "prediction": [classes[index] for index in patient_prediction],
        "fold": [fold_lookup[str(patient)] for patient in patient_ids],
    }).to_csv(output_dir / f"patient_predictions_{model_name}.csv", index=False)
    save_confusion(patient_truth, patient_prediction, classes, model_name, output_dir)
    return result


def run_benchmark(config: dict, model_names: list[str]) -> dict[str, dict]:
    X, y, groups = load_primary_data(config)
    folds = load_or_create_folds(y, groups, config)
    results = {name: run_model_benchmark(name, config, X, y, groups, folds) for name in model_names}
    write_comparison(Path(config["output_dir"]))
    return results


def write_comparison(output_dir: Path) -> Path:
    """Consolidate every completed model, including models run separately."""
    rows = []
    for metrics_path in sorted(output_dir.glob("metrics_*.json")):
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        name = result["model"]
        row = {"model": name}
        row.update({f"patient_{key}": value for key, value in result["patient_metrics"].items() if isinstance(value, float)})
        row.update({f"beat_{key}": value for key, value in result["beat_metrics"].items() if isinstance(value, float)})
        rows.append(row)
    destination = output_dir / "comparison.csv"
    pd.DataFrame(rows).sort_values("patient_macro_f1", ascending=False).to_csv(destination, index=False)
    return destination
