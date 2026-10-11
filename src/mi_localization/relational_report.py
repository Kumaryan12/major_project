"""Validate paired out-of-fold predictions and report relational development results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from .relational import ARMS


def paired_bootstrap(truth: np.ndarray, predictions: dict[str, np.ndarray],
                     n_classes: int, draws: int = 5000, seed: int = 42) -> dict:
    """Class-stratified, patient-paired percentile intervals; no model refits."""
    truth = np.asarray(truth, dtype=int)
    if draws < 100 or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("Need patient truths and at least 100 bootstrap draws")
    pools = [np.flatnonzero(truth == label) for label in range(n_classes)]
    if any(not len(pool) for pool in pools):
        raise ValueError("Every class must be represented")
    if any(np.asarray(pred).shape != truth.shape or np.any(np.asarray(pred) < 0)
           or np.any(np.asarray(pred) >= n_classes) for pred in predictions.values()):
        raise ValueError("Invalid paired patient predictions")
    rng = np.random.default_rng(seed)
    samples = np.column_stack([rng.choice(pool, (draws, len(pool))) for pool in pools])
    row_offsets = np.arange(draws)[:, None] * n_classes ** 2
    distributions = {}
    for name, prediction in predictions.items():
        encoded = n_classes * truth[samples] + np.asarray(prediction)[samples] + row_offsets
        counts = np.bincount(encoded.ravel(), minlength=draws * n_classes ** 2).reshape(draws, n_classes, n_classes)
        numerator = 2 * np.diagonal(counts, axis1=1, axis2=2)
        denominator = counts.sum(axis=1) + counts.sum(axis=2)
        distributions[name] = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                                        where=denominator != 0).mean(axis=1)
    output = {"draws": draws, "seed": seed, "method": "paired class-stratified patient bootstrap, percentile 95% intervals",
              "limitations": "Conditional on fitted cross-validation predictions; no model refits, no correction for repeated development or multiple comparisons. Not independent confirmation.",
              "arms": {}}
    for name, distribution in distributions.items():
        comparisons = {}
        for reference in ["statistics_short", "statistics_context"]:
            if reference in distributions and reference != name:
                differences = distribution - distributions[reference]
                comparisons[reference] = {"macro_f1_difference_95_interval": np.quantile(differences, [0.025, 0.975]).tolist()}
        output["arms"][name] = {"macro_f1_95_interval": np.quantile(distribution, [0.025, 0.975]).tolist(),
                                "paired_comparisons": comparisons}
    return output


def summarize_relational_study(config: dict, draws: int = 5000) -> Path:
    output = Path(config["output_dir"]) / "complete"
    bank_manifest = json.loads((Path(config["bank_dir"]) / "manifest.json").read_text())
    expected_hash = hashlib.sha256(json.dumps({"config": config, "bank": bank_manifest["fingerprint"]}, sort_keys=True).encode()).hexdigest()
    classes = config["classes"]
    mapping = {name: index for index, name in enumerate(classes)}
    study_manifest_path = output / "study_manifest.json"
    if study_manifest_path.exists():
        study_manifest = json.loads(study_manifest_path.read_text())
        if study_manifest["study_fingerprint"] != expected_hash or study_manifest["folds_sha256"] != hashlib.sha256(Path(config["folds_path"]).read_bytes()).hexdigest():
            raise ValueError("Frozen study settings changed")
    frozen = pd.read_csv(config["folds_path"]).set_index("patient_id").sort_index()
    if not frozen.index.is_unique:
        raise ValueError("Duplicate patients in frozen folds")
    metrics, predictions = {}, {}
    for arm in ARMS:
        saved = json.loads((output / f"metrics_{arm}.json").read_text())
        folds = list(range(1, int(config["outer_splits"]) + 1))
        if not saved["complete"] or saved["evaluated_folds"] != folds or saved["study_fingerprint"] != expected_hash:
            raise ValueError("Report requires all arms completed with the same study settings")
        if saved["classes"] != classes or saved["n_patients"] != len(frozen) or saved["n_beats"] != bank_manifest["n_beats"]:
            raise ValueError("Study cohort/classes changed")
        audits = saved["folds"]
        if sorted(row["fold"] for row in audits) != folds or any(row["patient_overlap"] != 0 or row["study_fingerprint"] != expected_hash for row in audits):
            raise ValueError("Invalid outer-fold audit")
        for row in audits:
            if arm.endswith("_shared"):
                if row["best_candidate"]["rank"] not in config["ranks"]:
                    raise ValueError("Selected rank is outside the prespecified candidates")
                if len(config["ranks"]) * len(config["model_candidates"]) > 1:
                    if any(len(candidate["inner_scores"]) != int(config["inner_splits"]) for candidate in row["inner_candidates"]):
                        raise ValueError("Incomplete inner rank selection")
        patient = pd.read_csv(output / f"patient_predictions_{arm}.csv").set_index("patient_id").sort_index()
        if not patient.index.is_unique or not patient.index.equals(frozen.index):
            raise ValueError("Paired arms must contain the same frozen patients")
        if not patient.truth.equals(frozen.label.rename("truth")) or not patient.fold.equals(frozen.fold):
            raise ValueError("Patient labels or fold assignments changed")
        probability = patient[[f"probability_{name}" for name in classes]].to_numpy()
        if not np.isfinite(probability).all() or np.any(probability < 0) or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Invalid patient probabilities")
        truth = patient.truth.map(mapping).to_numpy()
        pred = patient.prediction.map(mapping).to_numpy()
        if not np.array_equal(pred, probability.argmax(axis=1)):
            raise ValueError("Predictions do not match mean patient probabilities")
        score = f1_score(truth, pred, labels=range(len(classes)), average="macro", zero_division=0)
        if not np.isclose(score, saved["patient_metrics"]["macro_f1"], atol=1e-12):
            raise ValueError("Stored metrics do not match patient predictions")
        metrics[arm], predictions[arm] = saved, pred
    uncertainty = paired_bootstrap(truth, predictions, len(classes), draws=draws, seed=int(config["seed"]))
    uncertainty.update(study_fingerprint=expected_hash, n_patients=len(frozen),
                       classes=classes, class_counts=frozen.label.value_counts().to_dict(),
                       folds_sha256=hashlib.sha256(Path(config["folds_path"]).read_bytes()).hexdigest())
    for arm in ARMS:
        uncertainty["arms"][arm]["macro_f1"] = metrics[arm]["patient_metrics"]["macro_f1"]
        for reference, comparison in uncertainty["arms"][arm]["paired_comparisons"].items():
            comparison["macro_f1_difference"] = metrics[arm]["patient_metrics"]["macro_f1"] - metrics[reference]["patient_metrics"]["macro_f1"]
    (output / "paired_bootstrap.json").write_text(json.dumps(uncertainty, indent=2))
    names = {"statistics_short": "Short-beat statistics (control)", "statistics_context": "Full-record wavelet statistics",
             "correlation_flat": "Uncompressed correlations + context statistics",
             "correlation_shared": "Shared correlation cores + context statistics",
             "covariance_shared": "Shared covariance cores + context statistics"}
    maximum = max(saved["patient_metrics"]["macro_f1"] for saved in metrics.values())
    target_status = ("No arm reached 0.80 macro F1 in this study; 0.80–0.90 remains an aspirational target."
                     if maximum < 0.8 else "At least one arm reached 0.80 macro F1 on this development benchmark; this is not independent confirmation or a guaranteed generalization result.")
    lines = ["# VCG relational tensor development: completed results", "",
             "Measured Frank XYZ is the sole input. ECG + VCG remains a separate, **not yet implemented** extension.", "",
             f"All five arms completed all {config['outer_splits']} frozen patient folds: {len(frozen)} patients, {bank_manifest['n_beats']:,} beats, {len(classes)} classes. Scores below use pooled out-of-fold patient predictions, not an average of fold F1 scores.", "",
             f"These previously inspected PTB folds are a **development benchmark**, not untouched confirmation. {target_status}", "",
             "## Results", "", "| Representation | Patient macro F1 | Accuracy | Δ F1 vs short control | Paired 95% interval for Δ |", "|---|---:|---:|---:|---:|"]
    baseline = metrics["statistics_short"]["patient_metrics"]["macro_f1"]
    for arm in ARMS:
        value = metrics[arm]["patient_metrics"]
        if arm == "statistics_short":
            interval = "—"
        else:
            lo, hi = uncertainty["arms"][arm]["paired_comparisons"]["statistics_short"]["macro_f1_difference_95_interval"]
            interval = f"[{lo:+.3f}, {hi:+.3f}]"
        lines.append(f"| {names[arm]} | {value['macro_f1']:.3f} | {value['accuracy']:.1%} | {value['macro_f1'] - baseline:+.3f} | {interval} |")
    lines.extend(["", f"Intervals use {draws:,} class-stratified paired patient bootstrap draws. They are conditional on these fitted cross-validation predictions: models are not refitted, and intervals do not adjust for repeated development, correlated training folds or multiple comparisons. Do not treat them as independent evidence or a formal significance test.", "",
                  "## Class-level F1", "", "| Class | Patients | Short control | Context statistics | Flat correlation | Shared correlation | Shared covariance |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name in classes:
        values = [f"{metrics[arm]['patient_metrics']['classification_report'][name]['f1-score']:.3f}" for arm in ARMS]
        lines.append(f"| {name} | {(frozen.label == name).sum()} | " + " | ".join(values) + " |")
    lines.extend(["", "## Implementation and fair comparison", "",
                  "- Same R peaks, 651-sample beat windows, six classes and patient folds for every arm.",
                  "- Every arm uses fixed depth-3, 250-tree XGBoost, class/patient-balanced training weights and mean beat-probability patient aggregation.",
                  "- XYZ × (original + D3–D9) gives 24 channels. Four R-relative regions give 24 × 24 × 4 correlation/covariance tensors. Regions are approximate, not validated QRS/ST/T delineations.",
                  "- Bands are calculated over the full denoised recording before cropping beats. This is offline, noncausal processing; it is not a streaming system.",
                  "- A single shared channel basis compresses both matrix axes, leaving regions separate. Covariance scaling and bases are fitted on each training split only.",
                  "- Three inner patient folds select rank 4 or 8; no outer test patient enters a projection fit or rank selection. Ranks yield 40 or 144 core features, plus 192 context statistics.",
                  "- The uncompressed arm has 1,200 relational features + 192 context statistics. Statistics-only controls have 192 features.",
                  "- Correlation omits absolute amplitude; covariance retains it relative to training-only channel scales. All relational arms also retain amplitude-bearing statistics.",
                  "- Existing paper reproduction, nested depth-3/6 benchmark and PTB-XL external results were not overwritten or rescored. The historical 0.487 nested model result is a different classifier protocol, not this study's control.", "",
                  "## Inner-selected ranks", ""])
    for arm in ["correlation_shared", "covariance_shared"]:
        counts = Counter(row["best_candidate"]["rank"] for row in metrics[arm]["folds"])
        lines.append(f"- {names[arm]}: " + ", ".join(f"rank {rank} in {count}/{config['outer_splits']} outer folds" for rank, count in sorted(counts.items())) + ".")
    lines.extend(["", "## Interpretation and next experiments", "",
                  "This is a completed test of a sepsis-paper-inspired representation, not a reproduction of sepsis results or an established novelty/SOTA claim. A representation must improve the paired control before investing in a larger tuning budget. Retain unsuccessful results as ablations.", "",
                  "Next VCG-only hypotheses are morphology-preserving region features and multi-beat aggregation, then a regularized waveform encoder with training-patient-only pretraining. Prespecify each candidate and select it inside patient-grouped validation. Do not optimize against PTB-XL's previously evaluated test split. ECG + VCG must be reported separately with a matched VCG-only control.", "",
                  "An improvement claim ultimately needs a newly reserved independent cohort or external site. A fresh random split of these same repeatedly studied PTB patients is not independent confirmation.", "",
                  "## Reproducibility", "", "```bash", ".venv/bin/mi-localization relational-features", ".venv/bin/mi-localization relational-study",
                  ".venv/bin/python -m mi_localization.relational_report", "```", "",
                  "Configuration: `configs/relational_vcg.yaml`. Per-fold aggregate audits, confusion matrices, metrics and bootstrap intervals are in this directory. Patient prediction CSVs, feature banks and beat-probability checkpoints stay local and are not published to GitHub.", "",
                  f"Study fingerprint: `{expected_hash}`.", "",
                  "## Research basis", "",
                  "Nesaragi, Patidar and Thangaraj (2021), *A correlation matrix-based tensor decomposition method for early prediction of sepsis from clinical data*, [DOI: 10.1016/j.bbe.2021.06.009](https://doi.org/10.1016/j.bbe.2021.06.009). The proposed MI tensor and training-only shared projection are adaptations; the sepsis paper's metrics do not establish an achievable MI macro F1 target.", ""])
    destination = output / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/relational_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(summarize_relational_study(config, args.draws))
