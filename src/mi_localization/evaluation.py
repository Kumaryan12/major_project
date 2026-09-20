"""Leak-prone paper protocol and patient-independent comparison protocol."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import label_binarize

from .labels import CLASS_ORDER


def make_classifier(config: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(config.get("n_estimators", 500)),
        max_features=config.get("max_features", "sqrt"),
        bootstrap=True,
        oob_score=False,
        n_jobs=int(config.get("n_jobs", -1)),
        random_state=seed,
        class_weight=None,
    )


def evaluate_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, mode: str, config: dict, seed: int, output_dir: Path) -> dict:
    n_splits = int(config.get("n_splits", 10))
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if mode == "beat"
        else StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    probabilities = np.zeros((len(y), len(CLASS_ORDER)), dtype=np.float32)
    predictions = np.empty(len(y), dtype="U8")
    fold_ids = np.full(len(y), -1, dtype=int)
    split_iter = splitter.split(X, y) if mode == "beat" else splitter.split(X, y, groups)
    fold_rows = []
    for fold, (train, test) in enumerate(split_iter, 1):
        model = make_classifier(config, seed + fold)
        model.fit(X[train], y[train])
        raw = model.predict_proba(X[test])
        aligned = np.zeros((len(test), len(CLASS_ORDER)), dtype=np.float32)
        for source, name in enumerate(model.classes_):
            aligned[:, CLASS_ORDER.index(str(name))] = raw[:, source]
        probabilities[test] = aligned
        predictions[test] = model.predict(X[test])
        fold_ids[test] = fold
        fold_rows.append({
            "fold": fold,
            "accuracy": float(accuracy_score(y[test], predictions[test])),
            "train_beats": int(len(train)),
            "test_beats": int(len(test)),
            "train_patients": int(len(np.unique(groups[train]))),
            "test_patients": int(len(np.unique(groups[test]))),
            "patient_overlap": int(len(set(groups[train]) & set(groups[test]))),
        })
        print(f"{mode} fold {fold}/{n_splits}: accuracy={fold_rows[-1]['accuracy']:.4f}", flush=True)

    present = [name for name in CLASS_ORDER if name in set(y)]
    indices = [CLASS_ORDER.index(name) for name in present]
    binary = label_binarize(y, classes=CLASS_ORDER)
    report = classification_report(y, predictions, labels=present, output_dict=True, zero_division=0)
    metrics = {
        "mode": mode,
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_roc_auc_ovr": float(roc_auc_score(binary[:, indices], probabilities[:, indices], average="macro", multi_class="ovr")),
        "macro_average_precision": float(average_precision_score(binary[:, indices], probabilities[:, indices], average="macro")),
        "n_beats": int(len(y)),
        "n_patients": int(len(np.unique(groups))),
        "classes": present,
        "classification_report": report,
        "folds": fold_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"metrics_{mode}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"truth": y, "prediction": predictions, "patient_id": groups, "fold": fold_ids}).to_csv(output_dir / f"predictions_{mode}.csv", index=False)
    matrix = confusion_matrix(y, predictions, labels=present)
    pd.DataFrame(matrix, index=present, columns=present).to_csv(output_dir / f"confusion_matrix_{mode}.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=present, yticklabels=present, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{mode.title()}-level 10-fold CV")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{mode}.png", dpi=180)
    plt.close(fig)
    final_model = make_classifier(config, seed)
    final_model.fit(X, y)
    joblib.dump(final_model, output_dir / f"model_{mode}.joblib")
    return metrics

