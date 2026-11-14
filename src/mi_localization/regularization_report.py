"""Validate nested four-preset selection against its freshly fitted baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report

from .benchmark import inner_patient_splits, load_or_create_folds
from .contrastive_report import paired_seed_bootstrap
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .preprocessing_report import check_pair
from .regularization import ARMS, CANDIDATES, AUGMENTATION_SEED_OFFSET, choose_candidate, directory, has_augmentation, load_bank, validate_config
from .signal_audit import file_hash


def validate_fit(fit, candidate, seed, config, train_beats, train_patients, valid_patients, final_epochs=None):
    if fit["candidate"] != candidate or fit["seed"] != seed or fit["patient_overlap"] or fit["normalizer_fit_beats"] != train_beats or fit["train_patients"] != train_patients or fit["validation_patients"] != valid_patients or fit["validation_augmentation"] is not False:
        raise ValueError("Regularization fit scope/seed/augmentation mismatch")
    augmented = has_augmentation(candidate)
    if fit["augmentation_seed"] != (seed + AUGMENTATION_SEED_OFFSET if augmented else None):
        raise ValueError("Invalid independent augmentation seed")
    norm = fit["normalization"]
    if norm["stat_mean_"] or norm["stat_scale_"] or len(norm["wave_mean_"]) != 3 or len(norm["wave_scale_"]) != 3 or not np.isfinite(norm["wave_mean_"] + norm["wave_scale_"]).all() or any(value <= 0 for value in norm["wave_scale_"]):
        raise ValueError("Invalid waveform-only training normalization")
    history = fit["history"]
    if not history or len(history) > config["max_epochs"]:
        raise ValueError("Invalid regularization fit history")
    for epoch, row in enumerate(history, 1):
        state = row["augmentation_rng_state_sha256"]
        if row["epoch"] != epoch or row["beat_presentations"] != config["samples_per_epoch"] or not np.isfinite(row["training_loss"]) or row["augmented_presentations"] != (config["samples_per_epoch"] if augmented else 0):
            raise ValueError("Invalid training or augmentation presentation budget")
        if (augmented and (not isinstance(state, str) or len(state) != 64)) or (not augmented and state is not None):
            raise ValueError("Invalid augmentation RNG audit")
    if final_epochs is not None:
        if len(history) != final_epochs or fit["selected_epoch"] != final_epochs or fit["best_validation_macro_f1"] is not None or valid_patients != 0 or any("validation_patient_macro_f1" in row for row in history):
            raise ValueError("Final regularization refit used invalid stopping")
    else:
        scores = [row.get("validation_patient_macro_f1", np.nan) for row in history]
        if not np.isfinite(scores).all() or any(not 0 <= value <= 1 for value in scores):
            raise ValueError("Invalid inner patient validation history")
        best, selected = -np.inf, 0
        for epoch, score in enumerate(scores, 1):
            if score > best + 1e-8:
                best, selected = score, epoch
        if fit["selected_epoch"] != selected or not np.isclose(fit["best_validation_macro_f1"], best, atol=1e-12, rtol=0) or (len(history) < config["max_epochs"] and len(history) - selected != config["patience"]):
            raise ValueError("Inner epoch selection does not match validation history")


def matching(first, second, config, augmentation_stream=False):
    check_pair(first, second, config)
    if first["normalization"] != second["normalization"]:
        raise ValueError("Regularization candidates must fit identical training normalization")
    if augmentation_stream:
        for a, b in zip(first["history"], second["history"]):
            if a["augmentation_rng_state_sha256"] != b["augmentation_rng_state_sha256"]:
                raise ValueError("Augmented presets did not share the declared random stream")


def summarize_study(config, draws=5000):
    validate_config(config)
    root, chosen, partial = directory(config)
    if partial or len(config["training_seeds"]) < 3:
        raise ValueError("Final regularization report requires full folds and three seeds")
    scope = json.loads((root / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    _, y, groups, bank = load_bank(config)
    if _hash(scope) != fingerprint or scope["config"] != config or scope["bank"] != bank or scope["folds_sha256"] != file_hash(config["folds_path"]) or scope["evaluated_folds"] != chosen:
        raise ValueError("Regularization report settings/cohort/cache changed")
    frozen = load_or_create_folds(y, groups, config).set_index("patient_id").sort_index()
    if not frozen.index.is_unique:
        raise ValueError("Duplicate frozen patient IDs")
    lookup = frozen.fold.to_dict()
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    settings = {setting["id"]: setting for setting in config["candidates"]}
    splits = {}
    for fold in chosen:
        train = np.flatnonzero(beat_folds != fold)
        splits[fold] = [(train[ti], train[vi]) for ti, vi in inner_patient_splits(y[train], groups[train], config["inner_splits"], config["split_seed"] + fold * 1000)]
    rows, selection_rows, predictions, class_scores = [], [], {}, {}
    fit_count = 0
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        predictions[seed], class_scores[seed], metrics, patient_probabilities = {}, {}, {}, {}
        for arm in ARMS:
            saved = json.loads((output / f"metrics_{arm}.json").read_text())
            if saved["study_fingerprint"] != fingerprint or saved["arm"] != arm or saved["training_seed"] != seed or saved["n_beats"] != bank["n_beats"] or saved["evaluated_folds"] != chosen:
                raise ValueError("Incomplete/mismatched regularization result")
            truth, pred = _read_predictions(output / f"patient_predictions_{arm}.csv", saved, frozen, config["classes"])
            predictions[seed][arm], metrics[arm] = pred, saved["patient_metrics"]
            frame = pd.read_csv(output / f"patient_predictions_{arm}.csv").set_index("patient_id").sort_index()
            patient_probabilities[arm] = frame[[f"probability_{label}" for label in config["classes"]]].to_numpy()
            class_scores[seed][arm] = classification_report(truth, pred, labels=np.arange(len(config["classes"])), target_names=config["classes"], output_dict=True, zero_division=0)
        for fold in chosen:
            audit = json.loads((output / f"fold_{fold:02d}.json").read_text())
            train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            if audit["study_fingerprint"] != fingerprint or audit["training_seed"] != seed or audit["fold"] != fold or audit["patient_overlap"] or audit["train_beats"] != len(train) or audit["test_beats"] != len(test) or audit["train_patients"] != len(np.unique(groups[train])) or audit["test_patients"] != len(np.unique(groups[test])) or set(audit["inner_candidates"]) != set(CANDIDATES):
                raise ValueError("Invalid regularization fold membership/audit")
            for candidate_id, inner in audit["inner_candidates"].items():
                if len(inner) != config["inner_splits"]:
                    raise ValueError("Missing inner regularization fits")
                for number, (fit, (ti, vi)) in enumerate(zip(inner, splits[fold]), 1):
                    if fit["inner_fold"] != number or fit["study_fingerprint"] != fingerprint:
                        raise ValueError("Invalid inner-fold identity")
                    validate_fit(fit, settings[candidate_id], seed + fold * 1000 + number, config, len(ti), len(np.unique(groups[ti])), len(np.unique(groups[vi])))
                    matching(audit["inner_candidates"]["baseline"][number - 1], fit, config)
            for a, b in zip(audit["inner_candidates"]["augmented"], audit["inner_candidates"]["combined"]):
                matching(a, b, config, augmentation_stream=True)
            selection = choose_candidate(audit["inner_candidates"], config)
            if audit["selection"] != selection:
                raise ValueError("Selected candidate does not follow inner-only rule")
            selected = selection["selected_candidate"]
            if audit["selected_reuses_baseline_refit"] != (selected == "baseline"):
                raise ValueError("Invalid baseline fallback audit")
            budgets = {row["candidate_id"]: row["final_epochs"] for row in selection["candidate_scores"]}
            for arm, candidate_id in [("baseline", "baseline"), ("selected", selected)]:
                validate_fit(audit["final_fits"][arm], settings[candidate_id], seed + fold, config, len(train), len(np.unique(groups[train])), 0, budgets[candidate_id])
            matching(audit["final_fits"]["baseline"], audit["final_fits"]["selected"], config)
            if selected == "baseline":
                mask = frozen.fold.to_numpy() == fold
                if audit["final_fits"]["baseline"] != audit["final_fits"]["selected"] or not np.array_equal(patient_probabilities["baseline"][mask], patient_probabilities["selected"][mask]):
                    raise ValueError("Baseline fallback was not reused exactly")
            fit_count += len(CANDIDATES) * config["inner_splits"] + (1 if selected == "baseline" else 2)
            selection_rows.append({"training_seed": seed, "outer_fold": fold, "selected_candidate": selected,
                "baseline_final_epochs": budgets["baseline"], "selected_final_epochs": budgets[selected],
                **{f"inner_macro_f1_{row['candidate_id']}": row["mean_best_validation_macro_f1"] for row in selection["candidate_scores"]}})
        a, b = [metrics[arm]["macro_f1"] for arm in ARMS]
        rows.append({"training_seed": seed, "baseline_macro_f1": a, "selected_macro_f1": b, "selected_minus_baseline_macro_f1": b - a})
    table = pd.DataFrame(rows)
    uncertainty = paired_seed_bootstrap(truth, predictions, len(config["classes"]), draws, config["seed"], arms=ARMS)
    uncertainty["study_fingerprint"] = fingerprint
    statistics = {arm: {"macro_f1_mean": float(table[f"{arm}_macro_f1"].mean()), "macro_f1_sample_std": float(table[f"{arm}_macro_f1"].std(ddof=1))} for arm in ARMS}
    counts = {candidate: sum(row["selected_candidate"] == candidate for row in selection_rows) for candidate in CANDIDATES}
    summary = {"complete": True, "study_fingerprint": fingerprint, "n_patients": len(frozen), "n_records": bank["n_records"], "n_beats": bank["n_beats"],
        "training_seeds": config["training_seeds"], "arms": statistics, "per_seed": rows, "selection_counts": counts, "unique_fit_count": fit_count,
        "mean_selected_minus_baseline_macro_f1": float(table.selected_minus_baseline_macro_f1.mean()),
        "limits": "Fixed baseline vs four-preset nested selection with extra tuning compute; repeated PTB development, not independent confirmation or an isolated augmentation-effect estimate."}
    _save_json(root / "paired_bootstrap.json", uncertainty)
    _save_json(root / "summary.json", summary)
    table.to_csv(root / "paired_comparison.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(root / "selection_summary.csv", index=False)
    lines = ["# Bounded regularization and augmentation: nested patient selection", "",
        f"{len(frozen)} unchanged patients, {bank['n_records']} records, {bank['n_beats']:,} original denoised beats, {len(config['classes'])} classes, {config['outer_splits']} frozen patient folds and seeds {config['training_seeds']}.", "",
        "| Training seed | Fixed baseline macro F1 | Inner-selected macro F1 | Selected − baseline |", "|---|---:|---:|---:|"]
    lines += [f"| {row['training_seed']} | {row['baseline_macro_f1']:.3f} | {row['selected_macro_f1']:.3f} | {row['selected_minus_baseline_macro_f1']:+.3f} |" for row in rows]
    lines += ["", "## Across all prespecified seeds", ""]
    for arm, value in statistics.items():
        lines.append(f"- {arm}: **{value['macro_f1_mean']:.3f} ± {value['macro_f1_sample_std']:.3f}** (mean ± sample SD).")
    lo, hi = uncertainty["mean_paired_macro_f1_difference_95_interval"]
    lines += [f"- Mean selected − baseline difference: **{summary['mean_selected_minus_baseline_macro_f1']:+.3f}**, descriptive patient-bootstrap interval [{lo:+.3f}, {hi:+.3f}].", "",
        f"{draws:,} paired class-stratified patient draws, shared across arms and fixed seeds. Repeated seed/patient predictions are not independent patients. No refits, seed resampling or correction for correlated training folds, repeated development or multiple comparisons.", "",
        "## Four bounded presets", "", "| Preset | Dropout | AdamW decay | Label smoothing | Common XYZ gain | Noise SD / training lead SD |", "|---|---:|---:|---:|---|---:|"]
    for setting in config["candidates"]:
        lines.append(f"| {setting['id']} | {setting['dropout']} | {setting['weight_decay']} | {setting['label_smoothing']} | {setting['gain_min']}–{setting['gain_max']} | {setting['noise_std']} |")
    lines += ["", "## Nested selection and matching", "",
        f"Each outer training set evaluates exactly four presets on the same {config['inner_splits']} inner patient folds. Highest arithmetic mean of inner best patient macro F1 selects the preset; exact ties follow configured order, starting with baseline. Its rounded-median inner selected epoch budget controls a fresh outer-training refit. All selected settings and scores are in `selection_summary.csv`.", "",
        "The fixed baseline is freshly trained using the original trainer. Its inner fits also serve as the baseline candidate in the bounded search. If baseline wins selection, the same final model/predictions are reused exactly; this is not an additional independent fit. Otherwise both final models start from matching weights. Outer-test patients never enter normalization, stopping, candidate selection or augmentation.", "",
        f"All models use the same compact waveform-only ResNet, fixed optimizer learning rate, {config['samples_per_epoch']:,} weighted-random beat presentations/epoch, batch size {config['batch_size']}, at most {config['max_epochs']} epochs and patience {config['patience']}. Shared-epoch row hashes, initialization hashes and training normalization match. The selected procedure uses extra tuning compute: {fit_count} unique inner/final fits in total. This compares a fixed model with a bounded selection procedure; it does not isolate a single regularizer or augmentation effect.", "",
        "Normalization is fitted to unaugmented training beats only, balancing classes and patients. Augmentation draws use a separate seeded CPU stream, shared between the augmented/combined presets. A single positive physical-unit gain per beat is shared by XYZ; low-level Gaussian noise is scaled by training-only lead SD. No per-beat renormalization is imposed. Original arrays are unchanged; each sampled training occurrence receives a fresh draw. Validation/test prediction never calls augmentation or test-time averaging over augmentations.", "",
        "The gain bounds and noise strength are exploratory choices, not clinically validated label invariances. Shared gain preserves vector direction before noise, but amplitude can contain diagnostic information. These operations do not create new patients or confirm physiological equivalence. No lead flipping/rotation, time warping, segment masking, longer windows, ECG input, pretraining or new feature tensor is included.", "",
        "## Selection frequencies", "", "| Preset | Outer seed/fold selections |", "|---|---:|"]
    lines += [f"| {candidate} | {count} |" for candidate, count in counts.items()]
    lines += ["", "These are repeated selections on the same patients, not independent clinical cases or proof that an individual preset is superior.", "",
        "## Mean class-level F1 across seeds", "", "| Class | Patients | Baseline | Selected |", "|---|---:|---:|---:|"]
    for label in config["classes"]:
        values = [np.mean([class_scores[seed][arm][label]["f1-score"] for seed in config["training_seeds"]]) for arm in ARMS]
        lines.append(f"| {label} | {(frozen.label == label).sum()} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ["", "## Interpretation boundaries", "",
        "Repeatedly inspected PTB patients remain development data. Neither candidate selection nor patient bootstrap makes these untouched confirmation. No best-seed selection or unplanned ensemble is used. This experiment does not establish novelty, SOTA, clinical validity, a dataset ceiling or a promised 0.80–0.90 macro F1. MPS is not guaranteed bitwise deterministic.", "",
        "Existing experiments and PTB-XL tests are unchanged. Patient predictions, weights and waveform arrays stay local; only aggregate reports/metrics/audits are published. The full method and primary sources are in `docs/regularization_vcg_plan.md`.", ""]
    destination = root / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regularization_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    print(summarize_study(yaml.safe_load(Path(args.config).read_text()), args.draws))
