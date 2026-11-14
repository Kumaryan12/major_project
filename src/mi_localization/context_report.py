"""Validate all seeds of matched short versus longer-context VCG runs."""
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
from .preprocessing_report import check_pair
from .context_study import ARMS, directory, load_banks
from .signal_audit import file_hash


def summarize_study(config, draws=5000):
    root, chosen, partial = directory(config)
    if partial or len(config["training_seeds"]) < 3:
        raise ValueError("Final context report requires full folds and at least three seeds")
    scope = json.loads((root / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope) != fingerprint or scope["config"] != config or scope["evaluated_folds"] != chosen or scope["folds_sha256"] != file_hash(config["folds_path"]):
        raise ValueError("Context report settings/folds changed")
    _, _, _, manifests, hashes = load_banks(config)
    if scope["bank_fingerprints"] != {arm: manifests[arm]["fingerprint"] for arm in ARMS} or scope["cache_hashes"] != hashes:
        raise ValueError("Context bank content/fingerprints changed")
    cache_audit = json.loads((Path(config["audit_dir"]) / "summary.json").read_text())
    if cache_audit["bank_fingerprint"] != manifests["long"]["fingerprint"] or not cache_audit["central_short_crop_verified_exactly"]:
        raise ValueError("Missing/mismatched context cache audit")
    frozen = pd.read_csv(config["folds_path"]).set_index("patient_id").sort_index()
    if not frozen.index.is_unique:
        raise ValueError("Duplicate frozen patients")
    predictions, all_metrics, rows = {}, {}, []
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        metrics, predictions[seed] = {}, {}
        for arm in ARMS:
            saved = json.loads((output / f"metrics_{arm}.json").read_text())
            if saved["study_fingerprint"] != fingerprint or saved["arm"] != arm or saved["training_seed"] != seed or saved["evaluated_folds"] != chosen or saved["n_beats"] != manifests["long"]["n_beats"]:
                raise ValueError("Incomplete/mismatched context result")
            truth, pred = _read_predictions(output / f"patient_predictions_{arm}.csv", saved, frozen, config["classes"])
            if sorted(row["fold"] for row in saved["folds"]) != chosen:
                raise ValueError("Missing context outer-fold audits")
            for audit in saved["folds"]:
                inner = audit["inner_selection"]
                if audit["study_fingerprint"] != fingerprint or audit["patient_overlap"] or len(inner) != config["inner_splits"] or sorted(row["inner_fold"] for row in inner) != list(range(1, config["inner_splits"] + 1)):
                    raise ValueError("Invalid nested context audit")
                if any(row["study_fingerprint"] != fingerprint or row["seed"] != seed + audit["fold"] * 1000 + row["inner_fold"] or not 1 <= row["selected_epoch"] <= config["max_epochs"] for row in inner):
                    raise ValueError("Invalid context inner seed/epoch")
                for fit in inner:
                    values = [row.get("validation_patient_macro_f1", np.nan) for row in fit["history"]]
                    if not values or not np.isfinite(values).all() or len(values) > config["max_epochs"]:
                        raise ValueError("Missing/invalid context inner validation history")
                    # Training chooses the earliest improvement larger than 1e-8.
                    best, selected = -np.inf, 0
                    for epoch, value in enumerate(values, 1):
                        if value > best + 1e-8:
                            best, selected = value, epoch
                    if fit["selected_epoch"] != selected or not np.isclose(fit["best_validation_macro_f1"], best):
                        raise ValueError("Context stopping epoch disagrees with validation history")
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in inner]) + 0.5))
                final = audit["final_fit"]
                if audit["final_epochs"] != epochs or final["selected_epoch"] != epochs or len(final["history"]) != epochs or final["validation_patients"] != 0 or final["seed"] != seed + audit["fold"] or final["normalizer_fit_beats"] != audit["train_beats"] or final["train_patients"] != audit["train_patients"] or any("validation_patient_macro_f1" in row for row in final["history"]):
                    raise ValueError("Final context fit used invalid stopping or fit scope")
            metrics[arm], predictions[seed][arm] = saved, pred
        for first, second in zip(metrics[ARMS[0]]["folds"], metrics[ARMS[1]]["folds"]):
            if first["fold"] != second["fold"]:
                raise ValueError("Context fold order changed")
            for a, b in zip(first["inner_selection"], second["inner_selection"]):
                check_pair(a, b, config)
            check_pair(first["final_fit"], second["final_fit"], config)
        values = [metrics[arm]["patient_metrics"]["macro_f1"] for arm in ARMS]
        rows.append({"training_seed": seed, "short_macro_f1": values[0], "long_macro_f1": values[1], "long_minus_short_macro_f1": values[1] - values[0]})
        all_metrics[seed] = metrics
    table = pd.DataFrame(rows)
    uncertainty = paired_seed_bootstrap(truth, predictions, len(config["classes"]), draws, config["seed"], arms=ARMS)
    uncertainty["study_fingerprint"] = fingerprint
    _save_json(root / "paired_bootstrap.json", uncertainty)
    statistics = {arm: {"macro_f1_mean": float(table[f"{arm}_macro_f1"].mean()),
                        "macro_f1_sample_std": float(table[f"{arm}_macro_f1"].std(ddof=1))} for arm in ARMS}
    summary = {"complete": True, "study_fingerprint": fingerprint, "n_patients": len(frozen), "n_beats": manifests["long"]["n_beats"],
        "training_seeds": config["training_seeds"], "arms": statistics, "per_seed": rows,
        "mean_long_minus_short_macro_f1": float(table.long_minus_short_macro_f1.mean()),
        "matching": "same architecture, initial weights, patient/beat membership and shared-epoch sampled-row hashes; separate training-only scaling per arm",
        "limits": "Previously inspected PTB development patients; fixed fitted predictions/seeds; no independent confirmation, best-seed selection or SOTA claim."}
    _save_json(root / "summary.json", summary)
    table.to_csv(root / "paired_comparison.csv", index=False)
    audit = cache_audit
    lines = ["# Short versus longer-context denoised VCG", "",
        f"{len(frozen)} unchanged patients, {summary['n_beats']:,} original beat positions, {len(config['classes'])} classes, {config['outer_splits']} patient folds, seeds {config['training_seeds']}.", "",
        "| Training seed | Short macro F1 | Long macro F1 | Long − short |", "|---|---:|---:|---:|"]
    lines += [f"| {row['training_seed']} | {row['short_macro_f1']:.3f} | {row['long_macro_f1']:.3f} | {row['long_minus_short_macro_f1']:+.3f} |" for row in rows]
    lines += ["", "## Across all prespecified seeds", ""]
    for arm, value in statistics.items():
        lines.append(f"- {arm}: **{value['macro_f1_mean']:.3f} ± {value['macro_f1_sample_std']:.3f}** (mean ± sample SD).")
    lo, hi = uncertainty["mean_paired_macro_f1_difference_95_interval"]
    lines += [f"- Mean paired long − short difference: **{summary['mean_long_minus_short_macro_f1']:+.3f}**, descriptive patient-bootstrap interval [{lo:+.3f}, {hi:+.3f}].", "",
        f"{draws:,} class-stratified patient draws are shared across both arms and all fixed seeds. Repeated seed/patient predictions are not independent patients. No refits or seed resampling; no correction for correlated training folds, repeated development or multiple comparisons. Development only; no independent confirmation or SOTA claim.", "",
        "## Matched protocol", "",
        f"Both inputs are measured Frank XYZ denoised over the full recording with the existing bior6.8 pipeline. Short crops use the original window; long crops span −{config['before_r']} to +{config['after_r']} ms, {config['before_r'] + config['after_r'] + 1} samples at 1000 Hz. All original central samples were verified bitwise equal. There is no new peak detection, time resizing, excluded beat/patient, ECG input, augmentation or statistical fusion.", "",
        f"{audit['padded_beats']:,} longer crops ({audit['padded_beats'] / audit['n_beats']:.3%}) across {audit['padded_patients']} patients require reflection at recording boundaries. Only the missing samples are synthetic; existing samples are unchanged. Padding metadata stays local and aggregate counts are in the cache audit. Reflected samples are not observed physiology.", "",
        "Both arms are freshly trained with the same ResNet parameter count and initial weights. Shared-epoch training rows match. Adaptive temporal pooling supports both lengths without time resampling. The longer arm processes more samples and changes the relative position of the central beat in the pooling bins; this is a window comparison, not an equal-FLOP experiment or isolation of a particular cardiac phase.", "",
        "Class/patient-weighted normalization is fit separately inside each training partition, over that arm's entire window (including reflected samples). There is no per-beat amplitude normalization. Different window moments are allowed and must not be pooled across arms or fit on held-out patients.", "",
        f"Each epoch presents {config['samples_per_epoch']:,} beats, batch size {config['batch_size']}. Fixed AdamW settings, up to {config['max_epochs']} epochs and patience {config['patience']}. {config['inner_splits']} frozen inner patient partitions select the rounded-median epoch count; fresh outer-training refits never monitor outer-test scores. Actual stopping epochs may differ. Every test beat is evaluated; mean probabilities produce one prediction per patient.", "",
        "The short control is refit within this study, rather than substituting a historical score. Source/runtime/fold/cache hashes guard compatible resumes; MPS is not guaranteed bitwise deterministic. The old raw-versus-denoised experiment remains unchanged.", "",
        "## Mean class-level F1 across seeds", "", "| Class | Patients | Short | Long |", "|---|---:|---:|---:|"]
    for label in config["classes"]:
        values = [np.mean([all_metrics[seed][arm]["patient_metrics"]["classification_report"][label]["f1-score"] for seed in config["training_seeds"]]) for arm in ARMS]
        lines.append(f"| {label} | {(frozen.label == label).sum()} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ["", "## Interpretation", "",
        "Longer windows may include neighboring beats. Neither window has independently adjudicated R positions or P/QRS/ST/T coverage; this experiment cannot attribute any gain to a specific phase. The same repeatedly inspected PTB cohort is development data. This does not establish a dataset ceiling, novelty, clinical validity or a promised 0.80–0.90 macro F1.", "",
        "Only aggregate metrics and audits are published. Patient predictions, waveforms, padding arrays and fitted weights stay local. Existing PTB-XL tests are not rescored.", ""]
    destination = root / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/context_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    print(summarize_study(yaml.safe_load(Path(args.config).read_text()), args.draws))
