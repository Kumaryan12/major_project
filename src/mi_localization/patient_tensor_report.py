"""Validate full patient-tensor ablations; report paired development evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .relational_report import paired_bootstrap


def summarize_patient_study(config, draws=5000):
    output = Path(config["output_dir"]) / "complete"
    scope = json.loads((output / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope) != fingerprint or scope["config"] != config or scope["folds_sha256"] != hashlib.sha256(Path(config["folds_path"]).read_bytes()).hexdigest():
        raise ValueError("Patient report settings or folds changed")
    manifest = json.loads((Path(config["bank_dir"]) / "manifest.json").read_text())
    if scope["bank_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Patient feature bank changed")
    folds = list(range(1, config["outer_splits"] + 1))
    if scope["evaluated_folds"] != folds:
        raise ValueError("Patient report requires complete cross-validation")
    frozen = pd.read_csv(config["folds_path"]).set_index("patient_id").sort_index()
    if not frozen.index.is_unique:
        raise ValueError("Duplicate frozen patient IDs")
    reference = Path(config["control_dir"])
    control_scope = json.loads((reference / "study_manifest.json").read_text())
    control_hash = control_scope.pop("study_fingerprint")
    if _hash(control_scope) != control_hash or control_scope["folds_sha256"] != scope["folds_sha256"] or control_scope["bank_fingerprint"] != scope["bank_fingerprint"]:
        raise ValueError("Reference ResNet bank/folds changed")
    baseline = json.loads((reference / "metrics_resnet.json").read_text())
    if baseline["study_fingerprint"] != control_hash or baseline["model"] != "resnet" or baseline["evaluated_folds"] != folds or baseline["n_beats"] != manifest["n_beats"]:
        raise ValueError("Reference is not the declared ResNet study")
    truth, pred = _read_predictions(reference / "patient_predictions_resnet.csv", baseline, frozen, config["classes"])
    metrics, predictions = {"resnet_reference": baseline}, {"resnet_reference": pred}
    for name in config["models"]:
        saved = json.loads((output / f"metrics_{name}.json").read_text())
        if saved["study_fingerprint"] != fingerprint or saved["evaluated_folds"] != folds or saved["n_beats"] != manifest["n_beats"]:
            raise ValueError("Patient model cohort/settings changed")
        if sorted(row["fold"] for row in saved["folds"]) != folds:
            raise ValueError("Missing patient fold audits")
        for audit in saved["folds"]:
            selections = audit["inner_selection"]
            if audit["patient_overlap"] or audit["study_fingerprint"] != fingerprint or len(selections) != config["inner_splits"]:
                raise ValueError("Invalid patient nested audit")
            if any(row["patient_overlap"] or not 1 <= row["selected_epoch"] <= config["max_epochs"] for row in selections):
                raise ValueError("Invalid inner epoch/overlap audit")
            epochs = int(np.floor(np.median([row["selected_epoch"] for row in selections]) + 0.5))
            if audit["final_epochs"] != epochs or audit["final_fit"]["validation_patients"] != 0:
                raise ValueError("Final patient fit must use inner-selected epochs without test stopping")
        _, prediction = _read_predictions(output / f"patient_predictions_{name}.csv", saved, frozen, config["classes"])
        metrics[name], predictions[name] = saved, prediction
    references = ("resnet_reference", "bag_mean", "bag_attention", "multiscale_attention")
    uncertainty = paired_bootstrap(truth, predictions, len(config["classes"]), draws, config["seed"], references=references)
    uncertainty.update(study_fingerprint=fingerprint, n_patients=len(frozen), classes=config["classes"])
    errors = {}
    for name, prediction in predictions.items():
        value = metrics[name]["patient_metrics"]
        uncertainty["arms"][name]["macro_f1"] = value["macro_f1"]
        for control_name, comparison in uncertainty["arms"][name]["paired_comparisons"].items():
            comparison["macro_f1_difference"] = value["macro_f1"] - metrics[control_name]["patient_metrics"]["macro_f1"]
        counts = confusion_matrix(truth, prediction, labels=np.arange(len(config["classes"])))
        pairs = [{"truth": config["classes"][i], "prediction": config["classes"][j], "patients": int(counts[i, j])}
                 for i in range(len(config["classes"])) for j in range(len(config["classes"])) if i != j and counts[i, j]]
        errors[name] = sorted(pairs, key=lambda row: (-row["patients"], row["truth"], row["prediction"]))
    _save_json(output / "paired_bootstrap.json", uncertainty)
    _save_json(output / "aggregate_error_audit.json", {"disclosure": "Descriptive prediction errors, not grounds for relabeling cases or proof that labels are incorrect.", "models": errors})
    names = {"resnet_reference": "Single-beat ResNet reference", "bag_mean": "Patient mean embedding",
             "bag_attention": "Patient gated attention", "multiscale_attention": "Learned scales + fixed compression + attention",
             "tucker_attention": "Learned scales + supervised Tucker + attention"}
    lines = ["# Patient-level multi-scale tensor results", "",
             f"Complete patient-disjoint development evaluation: {len(frozen)} patients, {manifest['n_beats']:,} beats, {len(config['classes'])} classes, {config['outer_splits']} outer folds.", "",
             "These PTB patients have been repeatedly inspected. This is not independent confirmation, clinical validation or an established SOTA claim. No 0.80–0.90 score is promised.", "",
             "| Model | Patient macro F1 | Accuracy | Δ vs ResNet reference | Paired 95% interval for Δ |", "|---|---:|---:|---:|---:|"]
    for name, saved in metrics.items():
        value = saved["patient_metrics"]
        delta, interval = value["macro_f1"] - baseline["patient_metrics"]["macro_f1"], "—"
        if name != "resnet_reference":
            lo, hi = uncertainty["arms"][name]["paired_comparisons"]["resnet_reference"]["macro_f1_difference_95_interval"]
            interval = f"[{lo:+.3f}, {hi:+.3f}]"
        lines.append(f"| {names[name]} | {value['macro_f1']:.3f} | {value['accuracy']:.1%} | {delta:+.3f} | {interval} |")
    lines.extend(["", f"Intervals use {draws:,} class-stratified paired patient draws, conditional on fitted predictions. No model refits or correction for repeated development, correlated training folds or multiple comparisons. They do not include training-seed uncertainty.", "",
                  "## Tensor and patient learning", "",
                  f"A training bag contains up to {config['bag_size']} beats from one patient. It is an unordered set, potentially spanning that patient's recordings—not a claimed continuous time series. All eligible beats are used exactly once at inference, with chunked encoding followed by full-patient embedding pooling.", "",
                  "The primary tensor is beats × XYZ × time × scales. The raw waveform is retained; three shared trainable filters initialize as local averages. They can learn signed responses and are not validated physiological frequency bands. The XYZ axis is never treated as interchangeable leads.", "",
                  f"The auxiliary tensor branch pools time to {config['tensor_time_bins']} bins. The fixed control uses further average pooling; the learned branch uses supervised mode factors with ranks {config['tensor_ranks']}. The beat axis is never contracted by an order-dependent Tucker factor. The CNN waveform path remains present in all models.", "",
                  "Factors and filters are optimized using patient-label loss, rather than a reconstruction-energy objective. An orthogonality penalty regularizes the Tucker factors. Mean and gated-attention controls distinguish patient pooling from the incremental scale/compression contribution. Parameter counts are recorded; these are not perfectly capacity-matched architectures.", "",
                  f"Each epoch samples {config['bags_per_epoch']} class/patient-balanced bags, up to {config['bags_per_epoch'] * config['bag_size']:,} beat presentations. Normalization uses only training patients. Fixed inner patient splits select early-stopping epochs; the rounded median determines a fresh outer-training refit without test monitoring. No encoder is initialized from a model trained on its validation patients.", "",
                  "The original reference averages beat probabilities; the new models pool embeddings and optimize a patient-bag loss. This intentionally tests a different objective and aggregation, not just a new tensor shape. MPS is seeded but not guaranteed bitwise deterministic.", "",
                  "## Class-level F1", "", "| Class | Patients | " + " | ".join(names[name] for name in metrics) + " |", "|---|---:|" + "---:|" * len(metrics)])
    for label in config["classes"]:
        lines.append(f"| {label} | {(frozen.label == label).sum()} | " + " | ".join(f"{saved['patient_metrics']['classification_report'][label]['f1-score']:.3f}" for saved in metrics.values()) + " |")
    lines += ["", "## Incremental tensor comparison", ""]
    if "tucker_attention" in metrics:
        for reference_name in ["bag_attention", "multiscale_attention"]:
            if reference_name in metrics:
                comparison = uncertainty["arms"]["tucker_attention"]["paired_comparisons"][reference_name]
                lo, hi = comparison["macro_f1_difference_95_interval"]
                lines.append(f"- Tucker vs {names[reference_name]}: Δ F1 {comparison['macro_f1_difference']:+.3f}, descriptive paired interval [{lo:+.3f}, {hi:+.3f}].")
    lines += ["", "## Error review", "", "The following are aggregate confusions. Errors should motivate signal/annotation review under consistent rules, not changes to labels to match predictions.", ""]
    for name in metrics:
        pairs = errors[name][:3]
        lines.append(f"- {names[name]}: " + ("; ".join(f"{row['truth']} → {row['prediction']}: {row['patients']} patients" for row in pairs) or "no errors") + ".")
    lines += ["", "## Limits and next step", "",
              "A more structured tensor is a testable hypothesis, not guaranteed extra information. Training-seed repeatability is reported separately; do not select a best seed, optimize an ensemble on outer predictions, or tune against PTB-XL's previously evaluated test split. Any improvement still needs a genuinely new validation cohort. No ECG + VCG model or self-supervised pretraining is included here.", "",
              "Research basis: [attention MIL](https://proceedings.mlr.press/v80/ilse18a.html) and [tensor contraction layers](https://www.jmlr.org/papers/v21/18-503.html). This is a VCG adaptation, not an exact reproduction or an established novel method.", "",
              "Reproduce with `.venv/bin/python -m mi_localization.patient_tensor` and `.venv/bin/python -m mi_localization.patient_tensor_report`. Patient prediction CSVs, fitted weights and feature arrays stay local. Only aggregate metrics/audits are published.", ""]
    destination = output / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/patient_tensor.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(summarize_patient_study(config, args.draws))
