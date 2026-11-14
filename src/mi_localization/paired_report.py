"""Validate every paired ECG/VCG fit and all prespecified development contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import paired_study as study
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle
from .contrastive_report import paired_seed_bootstrap
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .signal_audit import file_hash


def summarize_study(config, draws=5000):
    study.validate_config(config)
    root, chosen, partial = study.directory(config)
    if partial or len(config["training_seeds"]) < 3:
        raise ValueError("Final paired report requires all folds and three target seeds")
    scope = json.loads((root / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    _, _, y, groups, bank = study.paired_bank.load_paired(config)
    if _hash(scope) != fingerprint or scope["config"] != config or scope["bank"] != bank or scope["evaluated_folds"] != chosen or scope["folds_sha256"] != file_hash(config["folds_path"]) or scope["claim_scope"] != study.CLAIM_SCOPE or scope["runtime"]["code_sha256"] != file_hash(study.__file__) or scope["runtime"]["neural_code_sha256"] != file_hash(study.neural.__file__):
        raise ValueError("Paired report settings/data/code/folds changed")
    frozen = load_or_create_folds(y, groups, config).set_index("patient_id").sort_index()
    beat_folds = np.array([frozen.fold[str(patient)] for patient in groups])
    predictions, metrics_by_seed, rows, epoch_rows = {}, {}, [], []
    fit_count = 0
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        predictions[seed], metrics_by_seed[seed], audits = {}, {}, {}
        for arm in study.ARMS:
            saved = json.loads((output / f"metrics_{arm}.json").read_text())
            if saved["study_fingerprint"] != fingerprint or saved["arm"] != arm or saved["training_seed"] != seed or saved["n_beats"] != bank["target"]["n_beats"] or saved["evaluated_folds"] != chosen or saved["claim_scope"] != study.CLAIM_SCOPE:
                raise ValueError("Incomplete/mismatched paired metrics")
            truth, pred = _read_predictions(output / f"patient_predictions_{arm}.csv", saved, frozen, config["classes"])
            predictions[seed][arm], metrics_by_seed[seed][arm], audits[arm] = pred, saved["patient_metrics"], []
            pooled = np.full((len(y), len(config["classes"])), np.nan)
            for fold in chosen:
                train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
                audit = json.loads((output / f"fold_{fold:02d}_{arm}.json").read_text())
                if audit["study_fingerprint"] != fingerprint or audit["training_seed"] != seed or audit["arm"] != arm or audit["fold"] != fold or audit["patient_overlap"] or audit["train_beats"] != len(train) or audit["test_beats"] != len(test) or audit["train_patients"] != len(np.unique(groups[train])) or audit["test_patients"] != len(np.unique(groups[test])):
                    raise ValueError("Paired outer fold membership mismatch")
                splits = [(train[ti], train[vi]) for ti, vi in inner_patient_splits(y[train], groups[train], config["inner_splits"], config["split_seed"] + fold*1000)]
                if len(audit["inner_selection"]) != len(splits):
                    raise ValueError("Missing paired inner folds")
                for inner, (fit, (ti, vi)) in enumerate(zip(audit["inner_selection"], splits), 1):
                    if fit["study_fingerprint"] != fingerprint or fit["inner_fold"] != inner:
                        raise ValueError("Paired inner fold identity mismatch")
                    study.validate_fit(fit, config, seed + fold*1000 + inner, arm, ti, vi, groups)
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in audit["inner_selection"]]) + .5))
                if audit["final_epochs"] != epochs:
                    raise ValueError("Outer epochs must use inner-only selection")
                study.validate_fit(audit["final_fit"], config, seed + fold, arm, train, None, groups, epochs)
                checkpoint = output / f"fold_{fold:02d}_{arm}.npz"
                if audit["prediction_checkpoint_sha256"] != file_hash(checkpoint):
                    raise ValueError("Paired prediction checkpoint changed")
                with np.load(checkpoint) as values:
                    if not np.array_equal(values["indices"], test):
                        raise ValueError("Prediction/test membership changed")
                    p = values["probabilities"]
                if p.shape != (len(test), len(config["classes"])) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
                    raise ValueError("Invalid paired checkpoint probabilities")
                _, fold_truth, fold_scores = aggregate_patients(p, y[test], groups[test])
                if audit["patient_metrics"] != metric_bundle(fold_truth, fold_scores, config["classes"]):
                    raise ValueError("Paired fold metrics disagree with predictions")
                pooled[test] = p
                audits[arm].append(audit)
                epoch_rows.append({"training_seed": seed, "outer_fold": fold, "arm": arm, "refit_epochs": epochs})
                fit_count += config["inner_splits"] + 1
            if saved["folds"] != audits[arm]:
                raise ValueError("Metrics and saved fold audits differ")
            ids, pooled_truth, pooled_scores = aggregate_patients(pooled, y, groups)
            frame = pd.read_csv(output / f"patient_predictions_{arm}.csv").set_index("patient_id").sort_index()
            if not np.array_equal(ids, frame.index.to_numpy()) or not np.array_equal(pooled_truth, truth) or not np.allclose(pooled_scores, frame[[f"probability_{label}" for label in config["classes"]]].to_numpy(), atol=1e-12, rtol=1e-7) or saved["patient_metrics"] != metric_bundle(pooled_truth, pooled_scores, config["classes"]):
                raise ValueError("Paired patient results do not match all held-out beats")
            rows.append({"training_seed": seed, "arm": arm, "patient_macro_f1": saved["patient_metrics"]["macro_f1"], "patient_accuracy": saved["patient_metrics"]["accuracy"]})
        for fold_index, fold in enumerate(chosen):
            for arm in study.ARMS[1:]:
                first, second = audits["vcg"][fold_index], audits[arm][fold_index]
                for a, b in zip(first["inner_selection"], second["inner_selection"]):
                    study.check_pair(a, b)
                study.check_pair(first["final_fit"], second["final_fit"])
            for arm in ["joint_additive", "joint_bilinear"]:
                first, second = audits["vcg_twin"][fold_index], audits[arm][fold_index]
                for a, b in zip(first["inner_selection"], second["inner_selection"]):
                    study.check_pair(a, b)
                study.check_pair(first["final_fit"], second["final_fit"])
    table = pd.DataFrame(rows)
    statistics = {arm: {"macro_f1_mean": float(table.loc[table.arm == arm, "patient_macro_f1"].mean()),
        "macro_f1_sample_std": float(table.loc[table.arm == arm, "patient_macro_f1"].std(ddof=1)),
        "parameter_count": sum(p.numel() for p in study.initial_model(config, config["seed"], arm).parameters())} for arm in study.ARMS}
    contrasts, bootstrap = {}, {}
    for first, second in study.CONTRASTS:
        key = f"{second}_minus_{first}"
        bootstrap[key] = paired_seed_bootstrap(truth, predictions, len(config["classes"]), draws, config["seed"], arms=(first, second))
        contrasts[key] = {"mean_macro_f1_difference": statistics[second]["macro_f1_mean"] - statistics[first]["macro_f1_mean"],
            "descriptive_95_interval": bootstrap[key]["mean_paired_macro_f1_difference_95_interval"]}
    summary = {"complete": True, "study_fingerprint": fingerprint, "n_patients": len(frozen), "n_beats": len(y), "n_records": bank["target"]["n_records"],
        "training_seeds": config["training_seeds"], "arms": statistics, "prespecified_contrasts": contrasts, "unique_fit_count": fit_count,
        "claim_scope": study.CLAIM_SCOPE, "bootstrap_limits": "Conditional on fixed fitted predictions; no training/seed resampling or correction for correlated folds, repeated development or multiple contrasts."}
    _save_json(root / "summary.json", summary)
    _save_json(root / "paired_bootstrap.json", {"study_fingerprint": fingerprint, "contrasts": bootstrap})
    table.to_csv(root / "paired_comparison.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(root / "refit_epochs.csv", index=False)
    lines = ["# Measured ECG + VCG extension: low-rank fusion", "", study.CLAIM_SCOPE, "",
        f"Complete: {len(frozen)} unchanged patients, {bank['target']['n_records']} recordings, {len(y):,} original beats, {len(config['classes'])} classes, {config['outer_splits']} frozen patient folds, target seeds {config['training_seeds']}; {fit_count} validated fits.", "",
        "| Model | Patient macro F1, mean ± SD | Parameters |", "|---|---:|---:|"]
    for arm, values in statistics.items():
        lines.append(f"| {arm} | {values['macro_f1_mean']:.3f} ± {values['macro_f1_sample_std']:.3f} | {values['parameter_count']:,} |")
    lines += ["", "SD is across the three training seeds, not a generalization confidence interval. Every seed is retained; there is no best-seed selection or probability ensemble.", "",
        "## Prespecified paired comparisons", "", "| Comparison | Mean Δ macro F1 | Descriptive patient-bootstrap interval |", "|---|---:|---:|"]
    for key, value in contrasts.items():
        lo, hi = value["descriptive_95_interval"]
        lines.append(f"| {key} | {value['mean_macro_f1_difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", f"{draws:,} class-stratified patient draws shared across arms and all fixed seeds. Repeated seed/patient predictions are not independent patients. These are descriptive, unadjusted intervals conditional on fitted predictions, without refitting, seed resampling or correction for correlated CV folds, repeated development or multiple comparisons.", "",
        "## What was tested", "",
        "Measured Frank XYZ stays the VCG primary benchmark. This is a separately labeled ECG extension using the actual simultaneous twelve ECG leads, not ECG reconstructed from VCG or transferred source-dataset representations. No PTB-XL records, checkpoints or test folds enter this phase. See [bank audit](../bank_audit/REPORT.md).", "",
        "The single-stream models use the same compact residual architecture; ECG changes only the input convolution to twelve channels. The three dual models use the same two encoder architectures, exact initial weights, factor matrices, classifier and parameter counts. The VCG-only twin feeds a deterministic fourfold tiling of XYZ to its twelve-channel second branch; it adds capacity but no ECG information. All shared inputs use identical training-only per-lead normalization and all paired fits share exact sampled-row sequences in common epochs.", "",
        f"The dual embeddings a,b each have 64 dimensions. Rank-{config['fusion_rank']} factors U,V project them. Bilinear fusion uses tanh((Ua) ⊙ (Vb)), which compresses cross-embedding outer-product interactions without materializing the full 64 × 64 tensor. The additive control uses tanh((Ua + Vb)/√2) with the same trainable tensors. Both concatenate the interaction with the two original embeddings before classification. Tanh makes this a nonlinear low-rank bilinear adaptation, not an exact reproduction of the cited models. The additive comparison tests the interaction rule; the VCG twin comparison tests replacing the second VCG view with measured ECG at matched capacity. It is not a full factorial isolation of all possible fusion choices.", "",
        f"All models start from scratch with ordinary CE, AdamW rate {config['learning_rate']}, decay {config['weight_decay']}, dropout {config['dropout']}, {config['samples_per_epoch']:,} class/patient-balanced beat draws per epoch, max {config['max_epochs']} epochs and patience {config['patience']}. There is no external pretraining, auxiliary loss, augmentation, learned calibration or rank/rate search. Three inner patient splits choose epochs; rounded median best epoch sets a fresh outer-training refit with no outer-test monitoring. Every held-out beat contributes to mean patient probabilities.", "",
        "All patients, labels, cached R positions and crop boundaries are preserved. Denoising uses complete recordings and remains offline/noncausal. There is no clinical waveform delineation or validated attention explanation. Seeded MPS is not guaranteed bitwise deterministic.", "",
        "## All seed results", "", "| Training seed | " + " | ".join(study.ARMS) + " |", "|---|" + "---:|"*len(study.ARMS)]
    for seed in config["training_seeds"]:
        lines.append(f"| {seed} | " + " | ".join(f"{metrics_by_seed[seed][arm]['macro_f1']:.3f}" for arm in study.ARMS) + " |")
    lines += ["", "## Mean class F1 across seeds", "", "| Class | Patients | " + " | ".join(study.ARMS) + " |", "|---|---:|" + "---:|"*len(study.ARMS)]
    for label in config["classes"]:
        values = [np.mean([metrics_by_seed[seed][arm]["classification_report"][label]["f1-score"] for seed in config["training_seeds"]]) for arm in study.ARMS]
        lines.append(f"| {label} | {(frozen.label == label).sum()} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "## Interpretation", "",
        "These same PTB patients have been repeatedly inspected; added sensors, nesting and multiple seeds do not create independent confirmation. Larger models have additional compute and can overfit this small patient cohort. Similar initialization and bounded training are controls, not a guarantee of equally optimal tuning. No improvement, novelty, SOTA or 0.80–0.90 score follows merely from using a low-rank tensor. Any favorable development result still needs a genuinely new appropriate cohort and fixed protocol. Do not merge labels, remove hard patients or retune against old external test outputs to meet the target.", "",
        "Research basis: [Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325), [Efficient Low-rank Multimodal Fusion](https://aclanthology.org/P18-1209/), [PTB documentation](https://physionet.org/content/ptbdb/1.0.0/). These motivate the representation and data choice; they do not supply performance evidence for this adaptation.", "",
        "Reproduce: `.venv/bin/python -m mi_localization.paired_bank`, `.venv/bin/python -m mi_localization.paired_study`, then `.venv/bin/python -m mi_localization.paired_report`. Progress: `.venv/bin/python -m mi_localization.paired_study --status`. Patient predictions, source inventories, waveforms and fitted weights stay local; only aggregate reports/audits are public.", ""]
    destination = root / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paired_ecg_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    print(summarize_study(yaml.safe_load(Path(args.config).read_text()), args.draws))
