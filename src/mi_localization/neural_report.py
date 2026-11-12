"""Report complete neural development runs against the matched fixed control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from .neural import _hash
from .relational_report import paired_bootstrap


def _read_predictions(path, saved, frozen, classes):
    frame = pd.read_csv(path).set_index("patient_id").sort_index()
    if not frame.index.is_unique or not frame.index.equals(frozen.index) or not frame.truth.equals(frozen.label.rename("truth")) or not frame.fold.equals(frozen.fold):
        raise ValueError("Report arms must match the frozen patient cohort, labels and folds")
    if saved["classes"] != classes or not saved["complete"] or saved["n_patients"] != len(frozen):
        raise ValueError("Report requires complete, matched classes and patients")
    probabilities = frame[[f"probability_{name}" for name in classes]].to_numpy()
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid report probabilities")
    mapping = {name: index for index, name in enumerate(classes)}
    truth, pred = frame.truth.map(mapping).to_numpy(), frame.prediction.map(mapping).to_numpy()
    if not np.array_equal(pred, probabilities.argmax(1)):
        raise ValueError("Predictions do not match probabilities")
    actual = f1_score(truth, pred, labels=np.arange(len(classes)), average="macro", zero_division=0)
    if not np.isclose(actual, saved["patient_metrics"]["macro_f1"], atol=1e-12):
        raise ValueError("Reported neural score does not match predictions")
    return truth, pred


def summarize_neural_study(config, draws=5000):
    output = Path(config["output_dir"]) / "complete"
    scope = json.loads((output / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope) != fingerprint or scope["config"] != config or scope["folds_sha256"] != hashlib.sha256(Path(config["folds_path"]).read_bytes()).hexdigest():
        raise ValueError("Neural report settings or frozen folds changed")
    frozen = pd.read_csv(config["folds_path"]).set_index("patient_id").sort_index()
    if not frozen.index.is_unique:
        raise ValueError("Duplicate frozen patient IDs")
    folds = list(range(1, int(config["outer_splits"]) + 1))
    if scope["evaluated_folds"] != folds:
        raise ValueError("Report requires a complete neural study")
    manifest = json.loads((Path(config["bank_dir"]) / "manifest.json").read_text())
    if scope["bank_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Neural feature bank changed")
    control = Path(config["control_dir"])
    baseline = json.loads((control / "metrics_statistics_short.json").read_text())
    truth, control_predictions = _read_predictions(control / "patient_predictions_statistics_short.csv", baseline, frozen, config["classes"])
    if baseline["evaluated_folds"] != folds or baseline["n_beats"] != manifest["n_beats"]:
        raise ValueError("Control does not match the neural study cohort/folds")
    predictions, metrics = {"statistics_short": control_predictions}, {"statistics_short": baseline}
    for model in config["models"]:
        saved = json.loads((output / f"metrics_{model}.json").read_text())
        if saved["study_fingerprint"] != fingerprint or saved["evaluated_folds"] != folds or saved["n_beats"] != manifest["n_beats"]:
            raise ValueError("Mismatched neural model study")
        audits = saved["folds"]
        if sorted(row["fold"] for row in audits) != folds or any(row["patient_overlap"] or row["study_fingerprint"] != fingerprint for row in audits):
            raise ValueError("Invalid neural fold audit")
        for audit in audits:
            inner = audit["inner_selection"]
            if len(inner) != int(config["inner_splits"]) or any(row["patient_overlap"] or not 1 <= row["selected_epoch"] <= config["max_epochs"] for row in inner):
                raise ValueError("Invalid nested early-stopping audit")
            expected_epoch = int(np.floor(np.median([row["selected_epoch"] for row in inner]) + 0.5))
            if audit["final_epochs"] != expected_epoch:
                raise ValueError("Final epochs do not match inner-only selection")
        _, pred = _read_predictions(output / f"patient_predictions_{model}.csv", saved, frozen, config["classes"])
        predictions[model], metrics[model] = pred, saved
    uncertainty = paired_bootstrap(truth, predictions, len(config["classes"]), draws=draws, seed=int(config["seed"]))
    uncertainty.update(study_fingerprint=fingerprint, n_patients=len(frozen), classes=config["classes"])
    for name, result in uncertainty["arms"].items():
        result["macro_f1"] = metrics[name]["patient_metrics"]["macro_f1"]
        if name != "statistics_short":
            result["paired_comparisons"]["statistics_short"]["macro_f1_difference"] = result["macro_f1"] - baseline["patient_metrics"]["macro_f1"]
    (output / "paired_bootstrap.json").write_text(json.dumps(uncertainty, indent=2))
    names = {"statistics_short": "Short-wavelet statistics + fixed XGBoost", "resnet": "Compact XYZ ResNet",
             "resnet_statistics": "Compact XYZ ResNet + short-wavelet statistics"}
    lines = ["# VCG neural development results", "",
        f"Complete pooled out-of-fold patient predictions: {len(frozen)} patients, {manifest['n_beats']:,} beats, {len(config['classes'])} classes, {config['outer_splits']} frozen patient folds.", "",
        "**Development only:** these PTB folds were inspected during earlier representation studies. This is not untouched confirmation, a clinical validation or an established SOTA result.", "",
        "| Model | Patient macro F1 | Accuracy | Δ F1 vs fixed control | Paired 95% interval for Δ |",
        "|---|---:|---:|---:|---:|"]
    for name, saved in metrics.items():
        value = saved["patient_metrics"]
        delta = value["macro_f1"] - baseline["patient_metrics"]["macro_f1"]
        interval = "—"
        if name != "statistics_short":
            low, high = uncertainty["arms"][name]["paired_comparisons"]["statistics_short"]["macro_f1_difference_95_interval"]
            interval = f"[{low:+.3f}, {high:+.3f}]"
        lines.append(f"| {names[name]} | {value['macro_f1']:.3f} | {value['accuracy']:.1%} | {delta:+.3f} | {interval} |")
    lines.extend(["", f"Descriptive uncertainty: {draws:,} paired, class-stratified patient bootstrap draws conditional on fitted predictions. No model refits; intervals do not adjust for repeated development, correlated cross-validation training folds or multiple comparisons.", "",
        "## Protocol", "",
        "- Measured Frank XYZ only, same denoised 651-sample beats and R positions. No ECG input or pretraining.",
        "- Small residual encoder (three stages, widths 16/32/64), GroupNorm, coarse temporal pooling, 64-dimensional waveform embedding. Fusion adds a 32-dimensional embedding of the existing 192 short-wavelet statistics.",
        "- Architecture and optimizer were fixed before the run; this is not a large hyperparameter search.",
        f"- Up to {config['max_epochs']} epochs with patience {config['patience']}. Early stopping uses patient macro F1 on {config['inner_splits']} inner patient folds only. The rounded median best epoch sets the outer-training refit length. Outer test patients are never used for stopping or normalization.",
        f"- Each training epoch draws {config['samples_per_epoch']:,} beats with replacement, balancing classes and patients within class. Unweighted cross entropy avoids applying the balancing twice. Validation/test predictions use every eligible beat.",
        "- Waveform scaling uses weighted training-set XYZ means/standard deviations, not per-beat amplitude normalization. Statistics are scaled using training-only moments and clipped to ±10 standardized units.",
        "- XGBoost trains on all eligible training beats; neural models use the declared sampling budget. Thus this is a practical model comparison, not an identical optimization budget.",
        "- Mean beat probabilities yield one prediction per patient. The historical nested depth-3/6 score of 0.487 is a different protocol from this fixed depth-3 control.",
        "- Full-record wavelet denoising is offline/noncausal, not a streaming predictor.", "",
        "## Class-level F1", "", "| Class | Patients | " + " | ".join(names[name] for name in metrics) + " |",
        "|---|---:|" + "---:|" * len(metrics)])
    for label in config["classes"]:
        values = [f"{saved['patient_metrics']['classification_report'][label]['f1-score']:.3f}" for saved in metrics.values()]
        lines.append(f"| {label} | {(frozen.label == label).sum()} | " + " | ".join(values) + " |")
    maximum = max(saved["patient_metrics"]["macro_f1"] for saved in metrics.values())
    lines.extend(["", "## Limits and next steps", "",
        "No arm reached 0.80 patient macro F1." if maximum < 0.8 else "A development score reached 0.80 macro F1, but it still needs independent confirmation.", "",
        "There are only 164 independent patients in the real PTB task; 55,863 beats do not constitute that many independent cases. A gain here must be confirmed on a genuinely new cohort. Do not repeatedly tune against the previously evaluated PTB-XL test split. ECG + VCG and self-supervised learning remain separate future extensions.", "",
        "## Reproduce", "", "```bash", ".venv/bin/python -m pip install -e '.[neural,dev]'",
        ".venv/bin/mi-localization neural-features", ".venv/bin/mi-localization neural-study",
        ".venv/bin/python -m mi_localization.neural_report", "```", "",
        "Checkpoints, fitted weights and patient predictions stay local; only aggregate audits, metrics and figures are published.", "",
        f"Runtime: `{scope['runtime']}`. CPU/CUDA deterministic algorithms are requested; seeded MPS is not guaranteed bitwise deterministic. Seeds do not guarantee identical results across platforms/PyTorch releases.", "",
        "Research motivation: Strodthoff et al., [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://arxiv.org/abs/2004.13701). That study motivates testing residual waveform models but does not establish their performance on this VCG MI localization task.", ""])
    destination = output / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neural_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(summarize_neural_study(config, args.draws))
