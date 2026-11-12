"""Repeat the plain ResNet across seeds without changing patient partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .neural import _hash, _save_json, run_neural_study
from .neural_report import _read_predictions


def load_reference(config, base):
    reference = Path(config["reference_dir"])
    scope = json.loads((reference / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope) != fingerprint:
        raise ValueError("Invalid reference manifest")
    old = scope["config"]
    # All differences are explicitly non-training paths or repetition controls.
    omitted = {"models", "output_dir", "training_seed", "split_seed"}
    if {k: v for k, v in old.items() if k not in omitted} != {k: v for k, v in base.items() if k not in omitted}:
        raise ValueError("Reference training protocol changed")
    if int(old["seed"]) != int(config["reference_seed"]) or int(old.get("split_seed", old["seed"])) != int(config["split_seed"]):
        raise ValueError("Reference split/training seed changed")
    bank = json.loads((Path(base["bank_dir"]) / "manifest.json").read_text())
    fold_hash = hashlib.sha256(Path(base["folds_path"]).read_bytes()).hexdigest()
    if bank["fingerprint"] != scope["bank_fingerprint"] or fold_hash != scope["folds_sha256"]:
        raise ValueError("Reference bank or frozen folds changed")
    saved = json.loads((reference / "metrics_resnet.json").read_text())
    if saved["study_fingerprint"] != fingerprint or saved["evaluated_folds"] != list(range(1, base["outer_splits"] + 1)):
        raise ValueError("Incomplete reference ResNet")
    frozen = pd.read_csv(base["folds_path"]).set_index("patient_id").sort_index()
    _read_predictions(reference / "patient_predictions_resnet.csv", saved, frozen, base["classes"])
    for audit in saved["folds"]:
        if audit["patient_overlap"] or audit["final_fit"]["seed"] != config["reference_seed"] + audit["fold"]:
            raise ValueError("Invalid reference training audit")
        for inner in audit["inner_selection"]:
            if inner["seed"] != config["reference_seed"] + audit["fold"] * 1000 + inner["inner_fold"]:
                raise ValueError("Invalid reference inner seeds")
    return saved


def run_repeats(config):
    with Path(config["base_config"]).open() as stream:
        base = yaml.safe_load(stream)
    seeds = config["training_seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or config["reference_seed"] not in seeds:
        raise ValueError("Need at least three unique seeds including the reference")
    reference = load_reference(config, base)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    scope = {"config": config, "base": base,
             "reference_fingerprint": reference["study_fingerprint"],
             "reference_predictions_sha256": hashlib.sha256((Path(config["reference_dir"]) / "patient_predictions_resnet.csv").read_bytes()).hexdigest(),
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    guard = root / "repeat_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Repetition settings changed; use a new output_dir")
    _save_json(guard, scope)
    rows, reports = [], []
    frozen = pd.read_csv(base["folds_path"]).set_index("patient_id").sort_index()
    for seed in seeds:
        if seed == config["reference_seed"]:
            saved, source = reference, Path(config["reference_dir"])
        else:
            child = dict(base, models=["resnet"], split_seed=int(config["split_seed"]),
                         training_seed=int(seed), output_dir=str(root / f"seed_{seed}"))
            run_neural_study(child)
            source = Path(child["output_dir"]) / "complete"
            saved = json.loads((source / "metrics_resnet.json").read_text())
            _read_predictions(source / "patient_predictions_resnet.csv", saved, frozen, base["classes"])
        rows.append({"training_seed": seed, "split_seed": config["split_seed"],
                     "source": str(source), "patient_macro_f1": saved["patient_metrics"]["macro_f1"],
                     "patient_accuracy": saved["patient_metrics"]["accuracy"]})
        reports.append(saved)
        pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    scores = np.array([row["patient_macro_f1"] for row in rows])
    summary = {"complete": True, "training_seeds": seeds, "split_seed": config["split_seed"],
               "macro_f1_mean": float(scores.mean()), "macro_f1_sample_std": float(scores.std(ddof=1)),
               "macro_f1_min": float(scores.min()), "macro_f1_max": float(scores.max()), "runs": rows,
               "disclosure": "Reference seed reused after protocol/cohort/audit verification; other seeds refit on fixed outer and inner patient partitions. No best-seed selection or ensemble. Development data, not independent validation."}
    _save_json(root / "summary.json", summary)
    lines = ["# VCG ResNet training-seed repeatability", "",
             "Fixed outer AND inner patient partitions. All requested seeds are reported; no best-seed selection or ensemble.", "",
             "| Training seed | Patient macro F1 | Accuracy |", "|---|---:|---:|"]
    lines += [f"| {row['training_seed']} | {row['patient_macro_f1']:.3f} | {row['patient_accuracy']:.1%} |" for row in rows]
    lines += ["", f"Macro F1 mean ± sample standard deviation: **{scores.mean():.3f} ± {scores.std(ddof=1):.3f}**.", "",
              "The original seed is reused only after checking its model protocol, frozen cohort, predictions and training seed audits. Runtime/backend/source hashes are retained per run; this is a training-seed check, not a guarantee of bitwise reproduction across MPS/PyTorch versions.", "",
              "These are repeatedly studied PTB development patients. The spread across a few training seeds is not a confidence interval for generalization and does not remove model-selection bias.", "",
              "## Class-level F1 across seeds", "", "| Class | " + " | ".join(map(str, seeds)) + " |", "|---|" + "---:|" * len(seeds)]
    for label in base["classes"]:
        lines.append(f"| {label} | " + " | ".join(f"{saved['patient_metrics']['classification_report'][label]['f1-score']:.3f}" for saved in reports) + " |")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neural_repeat.yaml")
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(run_repeats(config).to_string(index=False))
