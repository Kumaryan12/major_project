"""Validate matched raw/denoised waveform runs across fixed training seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .contrastive_report import paired_seed_bootstrap
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .preprocessing_study import ARMS, directory
from .signal_audit import file_hash


def check_pair(first, second, config):
    for key in ["seed", "initial_state_sha256", "parameter_count", "normalizer_fit_beats", "train_patients", "validation_patients"]:
        if first[key] != second[key]:
            raise ValueError(f"Preprocessing matched-fit mismatch: {key}")
    # Different input signals MUST learn separate training-set scales. Do not
    # require identical normalization values, or fit pooled raw+clean scales.
    for fit in [first, second]:
        if fit["patient_overlap"] or not fit["history"]:
            raise ValueError("Invalid preprocessing patient overlap/history")
        if fit["normalization"]["stat_mean_"] or fit["normalization"]["stat_scale_"]:
            raise ValueError("Preprocessing study must remain waveform-only")
        for epoch, row in enumerate(fit["history"], 1):
            if row["epoch"] != epoch or row["beat_presentations"] != config["samples_per_epoch"] or not np.isfinite(row["training_loss"]):
                raise ValueError("Changed preprocessing training budget")
    for first_epoch, second_epoch in zip(first["history"], second["history"]):
        if first_epoch["sampled_rows_sha256"] != second_epoch["sampled_rows_sha256"]:
            raise ValueError("Preprocessing arms did not receive identical sampled rows")


def summarize_study(config, draws=5000):
    root, chosen, partial = directory(config)
    if partial or len(config["training_seeds"]) < 3:
        raise ValueError("Final preprocessing report requires full folds and at least three seeds")
    scope = json.loads((root / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope) != fingerprint or scope["config"] != config or scope["evaluated_folds"] != chosen or scope["folds_sha256"] != file_hash(config["folds_path"]):
        raise ValueError("Preprocessing report settings/folds changed")
    raw_bank, control = Path(config["raw_bank_dir"]), Path(config["denoised_bank_dir"])
    raw_manifest = json.loads((raw_bank / "manifest.json").read_text())
    control_manifest = json.loads((control / "manifest.json").read_text())
    if scope["raw_bank_fingerprint"] != raw_manifest["fingerprint"] or scope["denoised_bank_fingerprint"] != control_manifest["fingerprint"]:
        raise ValueError("Preprocessing bank fingerprints changed")
    if raw_manifest["waveforms_sha256"] != file_hash(raw_bank / "waveforms.npy") or raw_manifest["control_waveforms_sha256"] != file_hash(control / "waveforms.npy"):
        raise ValueError("Preprocessing waveform cache changed")
    frozen = pd.read_csv(config["folds_path"]).set_index("patient_id").sort_index()
    if not frozen.index.is_unique:
        raise ValueError("Duplicate frozen patients")
    predictions, all_metrics, rows = {}, {}, []
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        metrics, predictions[seed] = {}, {}
        for arm in ARMS:
            saved = json.loads((output / f"metrics_{arm}.json").read_text())
            if saved["study_fingerprint"] != fingerprint or saved["arm"] != arm or saved["training_seed"] != seed or saved["evaluated_folds"] != chosen or saved["n_beats"] != raw_manifest["n_beats"]:
                raise ValueError("Incomplete/mismatched preprocessing result")
            truth, pred = _read_predictions(output / f"patient_predictions_{arm}.csv", saved, frozen, config["classes"])
            if sorted(row["fold"] for row in saved["folds"]) != chosen:
                raise ValueError("Missing preprocessing outer-fold audits")
            for audit in saved["folds"]:
                inner = audit["inner_selection"]
                if audit["study_fingerprint"] != fingerprint or audit["patient_overlap"] or len(inner) != config["inner_splits"] or sorted(row["inner_fold"] for row in inner) != list(range(1, config["inner_splits"] + 1)):
                    raise ValueError("Invalid nested preprocessing audit")
                if any(row["seed"] != seed + audit["fold"] * 1000 + row["inner_fold"] or not 1 <= row["selected_epoch"] <= config["max_epochs"] for row in inner):
                    raise ValueError("Invalid preprocessing inner seed/epoch")
                for fit in inner:
                    values = [row.get("validation_patient_macro_f1", np.nan) for row in fit["history"]]
                    if not values or not np.isfinite(values).all() or len(values) > config["max_epochs"]:
                        raise ValueError("Missing/invalid preprocessing inner validation history")
                    # Training chooses the earliest improvement larger than 1e-8.
                    best, selected = -np.inf, 0
                    for epoch, value in enumerate(values, 1):
                        if value > best + 1e-8:
                            best, selected = value, epoch
                    if fit["selected_epoch"] != selected or not np.isclose(fit["best_validation_macro_f1"], best):
                        raise ValueError("Preprocessing stopping epoch disagrees with validation history")
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in inner]) + 0.5))
                final = audit["final_fit"]
                if audit["final_epochs"] != epochs or final["selected_epoch"] != epochs or len(final["history"]) != epochs or final["validation_patients"] != 0 or final["seed"] != seed + audit["fold"] or final["normalizer_fit_beats"] != audit["train_beats"] or final["train_patients"] != audit["train_patients"] or any("validation_patient_macro_f1" in row for row in final["history"]):
                    raise ValueError("Final preprocessing fit used invalid stopping or fit scope")
            metrics[arm], predictions[seed][arm] = saved, pred
        for first, second in zip(metrics[ARMS[0]]["folds"], metrics[ARMS[1]]["folds"]):
            if first["fold"] != second["fold"]:
                raise ValueError("Preprocessing fold order changed")
            for a, b in zip(first["inner_selection"], second["inner_selection"]):
                check_pair(a, b, config)
            check_pair(first["final_fit"], second["final_fit"], config)
        values = [metrics[arm]["patient_metrics"]["macro_f1"] for arm in ARMS]
        rows.append({"training_seed": seed, "denoised_macro_f1": values[0], "raw_macro_f1": values[1], "raw_minus_denoised_macro_f1": values[1] - values[0]})
        all_metrics[seed] = metrics
    table = pd.DataFrame(rows)
    uncertainty = paired_seed_bootstrap(truth, predictions, len(config["classes"]), draws, config["seed"], arms=ARMS)
    uncertainty["study_fingerprint"] = fingerprint
    _save_json(root / "paired_bootstrap.json", uncertainty)
    statistics = {arm: {"macro_f1_mean": float(table[f"{arm}_macro_f1"].mean()),
                        "macro_f1_sample_std": float(table[f"{arm}_macro_f1"].std(ddof=1))} for arm in ARMS}
    summary = {"complete": True, "study_fingerprint": fingerprint, "n_patients": len(frozen), "n_beats": raw_manifest["n_beats"],
        "training_seeds": config["training_seeds"], "arms": statistics, "per_seed": rows,
        "mean_raw_minus_denoised_macro_f1": float(table.raw_minus_denoised_macro_f1.mean()),
        "matching": "same architecture, initial weights, patient/beat membership and shared-epoch sampled-row hashes; separate training-only scaling per arm",
        "limits": "Previously inspected PTB development patients; fixed fitted predictions/seeds; no independent confirmation, best-seed selection or SOTA claim."}
    _save_json(root / "summary.json", summary)
    table.to_csv(root / "paired_comparison.csv", index=False)
    lines = ["# Raw versus denoised VCG: matched waveform experiment", "",
        f"{len(frozen)} unchanged patients, {raw_manifest['n_beats']:,} identical beat positions, {len(config['classes'])} classes, {config['outer_splits']} patient folds and seeds {config['training_seeds']}.", "",
        "| Training seed | Denoised macro F1 | Raw macro F1 | Raw − denoised |", "|---|---:|---:|---:|"]
    lines += [f"| {row['training_seed']} | {row['denoised_macro_f1']:.3f} | {row['raw_macro_f1']:.3f} | {row['raw_minus_denoised_macro_f1']:+.3f} |" for row in rows]
    lines += ["", "## Across all prespecified seeds", ""]
    for arm, value in statistics.items():
        lines.append(f"- {arm}: **{value['macro_f1_mean']:.3f} ± {value['macro_f1_sample_std']:.3f}** (mean ± sample SD across training seeds).")
    lo, hi = uncertainty["mean_paired_macro_f1_difference_95_interval"]
    lines += [f"- Mean paired raw − denoised difference: **{summary['mean_raw_minus_denoised_macro_f1']:+.3f}**, descriptive patient-bootstrap interval [{lo:+.3f}, {hi:+.3f}].", "",
        f"{draws:,} class-stratified patient draws are shared across both arms and all fixed seeds. Seed/patient repetitions are not independent cases. No model refits or seed resampling; no correction for correlated training folds, repeated development or multiple comparisons. These are development results, not independent confirmation or a clinical/SOTA claim.", "",
        "## Matched protocol", "",
        "The raw input is unfiltered measured XYZ cropped at the original cached R positions. The denoised input is the existing full-record wavelet-denoised cache, independently reproduced exactly before raw cache construction. No new R detection, beat exclusion, label change, longer crop, ECG input or statistical feature fusion is included.", "",
        "Both arms use the same compact ResNet, initial-weight hashes, class/patient-balanced weighted-random beat sampler and shared-epoch row sequences. Normalization is fit separately using only each training partition's signal; raw and denoised moments must NOT be forced to be equal. Waveform amplitudes are not normalized beat by beat.", "",
        f"Both have {config['samples_per_epoch']:,} beat presentations/epoch, batch size {config['batch_size']}, up to {config['max_epochs']} epochs and patience {config['patience']}. Architecture/optimizer are fixed. {config['inner_splits']} inner patient folds select the rounded-median epoch budget for a fresh outer-training refit; outer test patients never enter scaling or stopping. Actual epochs may differ between arms. Every outer-test beat is evaluated and probabilities are averaged per patient.", "",
        "The denoised control is freshly refit; the historical ResNet's 0.550 mean is not silently reused. Runtime/source/bank/fold fingerprints and private checkpoints permit compatible resumes. Seeded MPS is not guaranteed bitwise deterministic.", "",
        "## Mean class-level F1 across seeds", "", "| Class | Patients | Denoised | Raw |", "|---|---:|---:|---:|"]
    for label in config["classes"]:
        values = [np.mean([all_metrics[seed][arm]["patient_metrics"]["classification_report"][label]["f1-score"] for seed in config["training_seeds"]]) for arm in ARMS]
        lines.append(f"| {label} | {(frozen.label == label).sum()} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ["", "## Interpretation boundaries", "",
        "The accompanying signal audit verifies consistency and quantifies preprocessing changes; it does not validate R-peak accuracy or physiological coverage. No independent peak/delineation annotations or expert adjudication were supplied. Screening flags did not remove patients or beats. A lower signal-difference metric alone is not proof that denoising preserves all diagnostic information.", "",
        "This experiment evaluates this fixed processing/model combination. It neither establishes a PTB performance ceiling nor exhausts preprocessing choices. More architectures/longer windows should follow specific evidence; a 0.80–0.90 macro F1 is not promised. Independent validation remains necessary.", "",
        "Run `.venv/bin/python -m mi_localization.signal_audit`, `.venv/bin/python -m mi_localization.preprocessing_study`, then `.venv/bin/python -m mi_localization.preprocessing_report`. Read progress with `.venv/bin/python -m mi_localization.preprocessing_study --status`.", "",
        "Waveform arrays, patient predictions, fitted weights, raw review plots and record-level diagnostics remain local. Only aggregate metrics/audits are published. Existing results and PTB-XL tests are untouched.", ""]
    destination = root / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preprocessing_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(summarize_study(config, args.draws))
