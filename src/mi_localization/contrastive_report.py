"""Validate matched contrastive runs and summarize all prespecified seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .contrastive import ARMS, study_directory
from .neural import _hash, _save_json
from .neural_report import _read_predictions


def check_matching_fits(ce, supcon, config):
    for key in ["initial_state_sha256", "parameter_count", "normalization", "seed",
                "normalizer_fit_beats", "train_patients", "validation_patients"]:
        if ce[key] != supcon[key]:
            raise ValueError(f"Matched fit mismatch: {key}")
    if ce["contrastive_weight"] != 0 or supcon["contrastive_weight"] != config["contrastive_weight"]:
        raise ValueError("Incorrect control or contrastive coefficient")
    for audit in [ce, supcon]:
        if audit["patient_overlap"] != 0 or not audit["history"]:
            raise ValueError("Invalid training patient overlap/history")
        for epoch, row in enumerate(audit["history"], 1):
            if row["epoch"] != epoch or row["batches"] != config["batches_per_epoch"] or row["beat_presentations"] != config["batches_per_epoch"] * config["batch_size"] or row["positive_anchor_fraction"] != 1:
                raise ValueError("Structured batch budget or positive-anchor coverage changed")
            expected = row["cross_entropy"] + audit["contrastive_weight"] * row["cross_patient_supcon"]
            if not np.isfinite(row["training_loss"]) or not np.isclose(row["training_loss"], expected, atol=1e-5, rtol=1e-5):
                raise ValueError("Training objective does not match its coefficient")
    # Different objectives may stop at different epochs. Every shared epoch
    # must nevertheless contain the exact same sequence of training rows.
    for first, second in zip(ce["history"], supcon["history"]):
        if first["batch_indices_sha256"] != second["batch_indices_sha256"]:
            raise ValueError("Matched arms did not receive identical batch draws")


def paired_seed_bootstrap(truth, predictions, n_classes, draws=5000, seed=42):
    """Resample each patient once per draw, shared across all seeds and arms.

    Average F1 across the fixed seeds; NEVER flatten seed/patient pairs into
    independent patients or average probabilities into an unplanned ensemble.
    """
    truth = np.asarray(truth, dtype=int)
    if draws < 100 or truth.ndim != 1 or len(truth) == 0 or not predictions:
        raise ValueError("Need patient predictions and at least 100 bootstrap draws")
    pools = [np.flatnonzero(truth == label) for label in range(n_classes)]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError("Every class must be represented")
    rng = np.random.default_rng(seed)
    indices = np.column_stack([rng.choice(pool, (draws, len(pool))) for pool in pools])
    offsets = np.arange(draws)[:, None] * n_classes ** 2
    distributions = {arm: [] for arm in ARMS}
    per_seed = {}
    for training_seed, paired in predictions.items():
        seed_distributions = {}
        for arm in ARMS:
            pred = np.asarray(paired[arm])
            if pred.shape != truth.shape or not np.issubdtype(pred.dtype, np.integer) or np.any(pred < 0) or np.any(pred >= n_classes):
                raise ValueError("Invalid paired prediction labels")
            encoded = n_classes * truth[indices] + pred[indices] + offsets
            counts = np.bincount(encoded.ravel(), minlength=draws * n_classes ** 2).reshape(draws, n_classes, n_classes)
            numerator = 2 * np.diagonal(counts, axis1=1, axis2=2)
            denominator = counts.sum(1) + counts.sum(2)
            values = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0).mean(1)
            seed_distributions[arm] = values
            distributions[arm].append(values)
        difference = seed_distributions[ARMS[1]] - seed_distributions[ARMS[0]]
        per_seed[str(training_seed)] = {"paired_macro_f1_difference_95_interval": np.quantile(difference, [0.025, 0.975]).tolist()}
    means = {arm: np.mean(values, axis=0) for arm, values in distributions.items()}
    delta = means[ARMS[1]] - means[ARMS[0]]
    return {"draws": draws, "seed": seed, "unique_patients": len(truth), "fixed_training_seeds": list(predictions),
        "method": "class-stratified paired patient bootstrap, identical patient draws across ALL arms/seeds; mean of seed-wise macro F1 differences",
        "per_seed": per_seed, "mean_macro_f1_intervals": {arm: np.quantile(values, [0.025, 0.975]).tolist() for arm, values in means.items()},
        "mean_paired_macro_f1_difference_95_interval": np.quantile(delta, [0.025, 0.975]).tolist(),
        "limitations": "Conditional on fixed fitted predictions and these seeds. No model refits or seed resampling; no correction for correlated training folds, repeated development or multiple comparisons. Not independent confirmation."}


def summarize_study(config, draws=5000):
    root, chosen, partial = study_directory(config)
    if partial or len(config["training_seeds"]) < 3:
        raise ValueError("Final report requires the full study and at least three training seeds")
    scope = json.loads((root / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope) != fingerprint or scope["config"] != config or scope["evaluated_folds"] != chosen or scope["folds_sha256"] != hashlib.sha256(Path(config["folds_path"]).read_bytes()).hexdigest():
        raise ValueError("Contrastive report settings/frozen folds changed")
    manifest = json.loads((Path(config["bank_dir"]) / "manifest.json").read_text())
    if scope["bank_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Contrastive feature bank changed")
    frozen = pd.read_csv(config["folds_path"]).set_index("patient_id").sort_index()
    if not frozen.index.is_unique:
        raise ValueError("Duplicate frozen patient IDs")
    all_metrics, predictions, rows = {}, {}, []
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        metrics, predictions[seed] = {}, {}
        for arm in ARMS:
            saved = json.loads((output / f"metrics_{arm}.json").read_text())
            if saved["arm"] != arm or saved["training_seed"] != seed or saved["study_fingerprint"] != fingerprint or saved["evaluated_folds"] != chosen or saved["n_beats"] != manifest["n_beats"]:
                raise ValueError("Incomplete/mismatched contrastive study")
            truth, pred = _read_predictions(output / f"patient_predictions_{arm}.csv", saved, frozen, config["classes"])
            if sorted(row["fold"] for row in saved["folds"]) != chosen:
                raise ValueError("Missing contrastive outer-fold audits")
            for audit in saved["folds"]:
                selections = audit["inner_selection"]
                if audit["patient_overlap"] or audit["study_fingerprint"] != fingerprint or len(selections) != config["inner_splits"] or sorted(row["inner_fold"] for row in selections) != list(range(1, config["inner_splits"] + 1)):
                    raise ValueError("Invalid nested contrastive audit")
                for inner in selections:
                    if inner["seed"] != seed + audit["fold"] * 1000 + inner["inner_fold"] or not 1 <= inner["selected_epoch"] <= config["max_epochs"]:
                        raise ValueError("Invalid inner training seed or selected epoch")
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in selections]) + 0.5))
                final = audit["final_fit"]
                if audit["final_epochs"] != epochs or final["validation_patients"] != 0 or final["seed"] != seed + audit["fold"] or final["normalizer_fit_beats"] != audit["train_beats"] or any("validation_patient_macro_f1" in row for row in final["history"]):
                    raise ValueError("Final refit must use inner-selected epochs without test monitoring")
            metrics[arm], predictions[seed][arm] = saved, pred
        for first, second in zip(metrics[ARMS[0]]["folds"], metrics[ARMS[1]]["folds"]):
            if first["fold"] != second["fold"]:
                raise ValueError("Paired fold order changed")
            for ce, supcon in zip(first["inner_selection"], second["inner_selection"]):
                check_matching_fits(ce, supcon, config)
            check_matching_fits(first["final_fit"], second["final_fit"], config)
        ce, supcon = [metrics[arm]["patient_metrics"]["macro_f1"] for arm in ARMS]
        rows.append({"training_seed": seed, "matched_ce_macro_f1": ce, "cross_patient_supcon_macro_f1": supcon,
                     "paired_macro_f1_difference": supcon - ce})
        all_metrics[seed] = metrics
    table = pd.DataFrame(rows)
    uncertainty = paired_seed_bootstrap(truth, predictions, len(config["classes"]), draws, config["seed"])
    uncertainty["study_fingerprint"] = fingerprint
    _save_json(root / "paired_bootstrap.json", uncertainty)
    statistics = {}
    for arm in ARMS:
        scores = table[f"{arm}_macro_f1"].to_numpy()
        statistics[arm] = {"macro_f1_mean": float(scores.mean()), "macro_f1_sample_std": float(scores.std(ddof=1))}
    summary = {"complete": True, "study_fingerprint": fingerprint, "n_patients": len(frozen),
        "n_beats": manifest["n_beats"], "training_seeds": config["training_seeds"], "arms": statistics,
        "per_seed": rows, "mean_paired_macro_f1_difference": float(table.paired_macro_f1_difference.mean()),
        "matching_checks": "identical initial-state hashes, parameter counts, training-only normalization and shared-epoch batch-draw hashes for every paired inner/final fit",
        "interpretation": "Previously inspected PTB development cohort. No best-seed selection, unplanned ensemble, independent confirmation or SOTA claim."}
    _save_json(root / "summary.json", summary)
    table.to_csv(root / "paired_comparison.csv", index=False)
    lines = ["# Cross-patient contrastive VCG: matched experiment", "",
        f"Complete evaluation: {len(frozen)} unique patients, {manifest['n_beats']:,} beats, {len(config['classes'])} unchanged classes, {config['outer_splits']} frozen patient folds, training seeds {config['training_seeds']}.", "",
        "**Primary comparison: contrastive loss versus its matched cross-entropy control.** Both use the same structured batches, initialization, waveform model and maximum training budget. The historical ResNet mean of 0.550 is context, not the control for this changed sampler.", "",
        "| Training seed | Matched CE macro F1 | Cross-patient SupCon macro F1 | Paired Δ F1 |", "|---|---:|---:|---:|"]
    lines += [f"| {row['training_seed']} | {row['matched_ce_macro_f1']:.3f} | {row['cross_patient_supcon_macro_f1']:.3f} | {row['paired_macro_f1_difference']:+.3f} |" for row in rows]
    lines += ["", "## Across all prespecified seeds", ""]
    for arm in ARMS:
        values = statistics[arm]
        lines.append(f"- {arm}: mean macro F1 **{values['macro_f1_mean']:.3f} ± {values['macro_f1_sample_std']:.3f}** (sample SD across training seeds, not a generalization confidence interval).")
    lo, hi = uncertainty["mean_paired_macro_f1_difference_95_interval"]
    lines += [f"- Mean paired improvement: **{summary['mean_paired_macro_f1_difference']:+.3f}**; descriptive patient-bootstrap interval [{lo:+.3f}, {hi:+.3f}].", "",
        f"Bootstrap uses {draws:,} class-stratified patient draws shared across all seeds and both arms. Scores are averaged across fixed seeds; seed/patient repetitions are NOT treated as independent patients and probabilities are not ensembled. Conditional on fitted models: no training/seed resampling, or adjustment for correlated training folds, repeated development or multiple comparisons.", "",
        "## Exact experiment", "",
        f"- Same compact measured-XYZ ResNet and 64-dimensional encoder output in both arms; no extra projection head, tensor branch, augmentation, pretraining or ECG input.",
        f"- Every batch selects {config['patients_per_class']} distinct training patients per class and {config['beats_per_patient']} beats per selected patient: {config['batch_size']} beats. Training patients are selected uniformly within class. Beats use replacement only if a selected patient has fewer beats than requested.",
        f"- {config['batches_per_epoch']} batches = {config['batch_size'] * config['batches_per_epoch']:,} beat presentations per epoch in BOTH arms. This differs from the historical 8,192-beat epoch. Both have max {config['max_epochs']} epochs and patience {config['patience']}; actual epochs may differ due to independent inner-only stopping.",
        f"- Control loss: CE. Experimental loss: CE + {config['contrastive_weight']} × cross-patient SupCon, temperature {config['temperature']}. Contrastive embeddings use cosine normalization only inside that loss; the classifier retains its original embeddings and training-set-scaled XYZ input.",
        "- Same-class DIFFERENT-patient examples are positives. All same-patient pairs, including self-pairs, are excluded from both positives and denominator. No outer/validation examples enter a training contrastive batch or memory bank.",
        "- Audit checks verified identical initial weights, parameter counts, normalization and sampled-row hashes in every shared epoch of paired fits. The contrastive diagnostic is also recorded for CE, but contributes no gradient to that control.",
        "- Inner patient partitions are fixed across arms AND seeds. They select the epoch budget; a fresh outer-training fit uses the rounded median best epoch without outer test monitoring. All test beats are predicted; mean probabilities yield one patient prediction.",
        "- Existing labels, R positions, recordings and patient folds were not changed. Full-record denoising remains offline/noncausal. MPS is seeded but not guaranteed bitwise deterministic.", "",
        "## Mean class-level F1 across seeds", "", "| Class | Patients | Matched CE | Cross-patient SupCon |", "|---|---:|---:|---:|"]
    for label in config["classes"]:
        values = [np.mean([all_metrics[seed][arm]["patient_metrics"]["classification_report"][label]["f1-score"] for seed in config["training_seeds"]]) for arm in ARMS]
        lines.append(f"| {label} | {(frozen.label == label).sum()} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ["", "## Limits", "",
        "All these PTB patients have been repeatedly inspected. Neither nesting nor three seeds creates independent confirmation. No label was changed to agree with predictions, no favorable seed was selected, and no outer-test-optimized ensemble is reported. A 0.80–0.90 macro F1 target is not guaranteed. A genuinely new validation cohort is required for an improvement claim.", "",
        "Research basis: [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362). The cross-patient exclusion and structured batch policy are task-specific adaptations, not an exact reproduction or an established novelty/SOTA claim.", "",
        "Reproduce: `.venv/bin/python -m mi_localization.contrastive` followed by `.venv/bin/python -m mi_localization.contrastive_report`. Read saved progress without training: `.venv/bin/python -m mi_localization.contrastive --status`.", "",
        "Only aggregate metrics and audits are published. Patient prediction CSVs, waveform arrays and fitted weights stay local. Geometry-aware masked pretraining and ECG + VCG are not implemented in this phase.", ""]
    destination = root / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/contrastive_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(summarize_study(config, args.draws))
