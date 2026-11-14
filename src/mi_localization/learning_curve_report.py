"""Validate and plot complete patient-level training-size sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, classification_report

from .benchmark import aggregate_patients
from .contrastive_report import paired_seed_bootstrap
from .learning_curve import directory, fraction_key, inputs_and_plan, validate_config
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .regularization_report import validate_fit
from .signal_audit import file_hash


def check_fit(fit, seed, config, train_beats, train_patients, valid_patients, final_epochs=None):
    # Adapt frozen waveform-only audits to the existing strict history validator.
    candidate = {"id": "baseline", "dropout": config["dropout"], "weight_decay": config["weight_decay"],
                 "label_smoothing": 0, "gain_min": 1, "gain_max": 1, "noise_std": 0}
    adapted = dict(fit, candidate=candidate, augmentation_seed=None, validation_augmentation=False,
                   history=[dict(row, augmented_presentations=0, augmentation_rng_state_sha256=None) for row in fit["history"]])
    validate_fit(adapted, candidate, seed, config, train_beats, train_patients, valid_patients, final_epochs)


def draw_curve(rows, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = [row["mean_training_patients"] for row in rows]
    ax.errorbar(x, [row["heldout_macro_f1_mean"] for row in rows], yerr=[row["heldout_macro_f1_sample_std"] for row in rows],
                marker="o", color="black", capsize=4, label="Pooled held-out patient F1; mean ± seed SD")
    ax.plot(x, [row["mean_training_macro_f1_across_fits_and_seeds"] for row in rows], marker="s", linestyle="--", color="0.6", label="Training resubstitution F1; mean across fits/seeds")
    ax.set(xlabel="Mean unique training patients per outer fit", ylabel="Six-class patient macro F1", ylim=(0, 1.04),
           title="Training-patient sensitivity: fixed ResNet and outer test folds")
    ax.set_xticks(x, [f"{value:.1f}\n({100 * row['training_fraction']:g}% nominal)" for value, row in zip(x, rows)])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output / "learning_curve.png", dpi=180)
    fig.savefig(output / "learning_curve.svg")
    plt.close(fig)


def summarize_study(config, draws=5000):
    validate_config(config)
    root, chosen, partial = directory(config)
    if partial or len(config["training_seeds"]) < 3:
        raise ValueError("Final learning curve requires every fold and at least three seeds")
    _, y, groups, bank, frozen, plan, descriptions = inputs_and_plan(config, chosen)
    scope = json.loads((root / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope) != fingerprint or scope["config"] != config or scope["bank"] != bank or scope["subset_plan"] != descriptions or scope["folds_sha256"] != file_hash(config["folds_path"]) or scope["evaluated_folds"] != chosen:
        raise ValueError("Learning-curve report settings/cache/subsets changed")
    frozen = frozen.set_index("patient_id").sort_index()
    predictions, class_scores, per_seed, fit_audits = {}, {}, [], {}
    for seed in config["training_seeds"]:
        predictions[seed], class_scores[seed], fit_audits[seed] = {}, {}, {}
        for fraction in config["training_fractions"]:
            key = fraction_key(fraction)
            output = root / f"seed_{seed}" / key
            saved = json.loads((output / "metrics_resnet.json").read_text())
            if saved["study_fingerprint"] != fingerprint or saved["training_seed"] != seed or saved["training_fraction"] != fraction or saved["arm"] != key or saved["evaluated_folds"] != chosen or saved["n_beats"] != bank["n_beats"]:
                raise ValueError("Incomplete/mismatched learning-curve result")
            truth, prediction = _read_predictions(output / "patient_predictions_resnet.csv", saved, frozen, config["classes"])
            predictions[seed][key] = prediction
            class_scores[seed][key] = classification_report(truth, prediction, labels=np.arange(len(config["classes"])), target_names=config["classes"], output_dict=True, zero_division=0)
            training_scores, fit_audits[seed][key] = [], []
            all_test_probabilities = np.full((len(y), len(config["classes"])), np.nan)
            for fold in chosen:
                train, test, inner = plan[key][fold]
                audit = json.loads((output / f"fold_{fold:02d}.json").read_text())
                if audit["study_fingerprint"] != fingerprint or audit["training_seed"] != seed or audit["training_fraction"] != fraction or audit["fold"] != fold or audit["patient_overlap"] or audit["subset"] != descriptions[key][str(fold)] or len(audit["inner_selection"]) != config["inner_splits"]:
                    raise ValueError("Invalid learning-curve fold/subset audit")
                for number, (fit, (ti, vi)) in enumerate(zip(audit["inner_selection"], inner), 1):
                    if fit["study_fingerprint"] != fingerprint or fit["inner_fold"] != number or fit["training_seed"] != seed or fit["training_fraction"] != fraction or fit["outer_fold"] != fold:
                        raise ValueError("Invalid learning-curve inner identity")
                    check_fit(fit, seed + fold * 1000 + number, config, len(ti), len(np.unique(groups[ti])), len(np.unique(groups[vi])))
                epochs = int(np.floor(np.median([fit["selected_epoch"] for fit in audit["inner_selection"]]) + 0.5))
                if audit["final_epochs"] != epochs:
                    raise ValueError("Learning-curve epochs did not follow inner-only selection")
                check_fit(audit["final_fit"], seed + fold, config, len(train), len(np.unique(groups[train])), 0, epochs)
                with np.load(output / f"fold_{fold:02d}.npz") as checkpoint:
                    if not np.array_equal(checkpoint["indices"], test):
                        raise ValueError("Learning-curve test checkpoint membership changed")
                    test_probs = checkpoint["probabilities"]
                    if test_probs.shape != (len(test), len(config["classes"])) or not np.isfinite(test_probs).all() or np.any(test_probs < 0) or not np.allclose(test_probs.sum(1), 1, atol=1e-5):
                        raise ValueError("Invalid learning-curve test probabilities")
                    all_test_probabilities[test] = test_probs
                    train_ids, train_truth, train_probs = checkpoint["train_patient_ids"], checkpoint["train_truth"], checkpoint["train_probabilities"]
                    ids, expected_truth, _ = aggregate_patients(np.zeros((len(train), len(config["classes"]))), y[train], groups[train])
                    if not np.array_equal(train_ids, ids) or not np.array_equal(train_truth, expected_truth) or train_probs.shape != (len(ids), len(config["classes"])) or not np.isfinite(train_probs).all() or np.any(train_probs < 0) or not np.allclose(train_probs.sum(1), 1, atol=1e-5):
                        raise ValueError("Training resubstitution checkpoint contains invalid patients/probabilities")
                    score = float(f1_score(train_truth, train_probs.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
                if not np.isclose(score, audit["training_patient_metrics"]["macro_f1"], atol=1e-12, rtol=0):
                    raise ValueError("Training resubstitution score does not match checkpoint")
                training_scores.append(score)
                fit_audits[seed][key].append(audit)
            if not np.isclose(np.mean(training_scores), saved["mean_training_patient_macro_f1_across_fits"], atol=1e-12, rtol=0):
                raise ValueError("Training learning-curve summary does not match fits")
            ids, expected_truth, expected_scores = aggregate_patients(all_test_probabilities, y, groups)
            csv = pd.read_csv(output / "patient_predictions_resnet.csv").set_index("patient_id").sort_index()
            if not np.array_equal(ids, csv.index.to_numpy()) or not np.allclose(expected_scores, csv[[f"probability_{label}" for label in config["classes"]]].to_numpy(), atol=1e-12, rtol=0):
                raise ValueError("Pooled learning-curve predictions do not match fold checkpoints")
            per_seed.append({"training_seed": seed, "training_fraction": fraction,
                             "heldout_macro_f1": saved["patient_metrics"]["macro_f1"], "mean_training_macro_f1_across_fits": float(np.mean(training_scores))})
    # The architecture/initial weights match, but subset-specific normalizers,
    # sampled rows and inner partitions MUST be allowed to change with size.
    for seed in config["training_seeds"]:
        for fold_offset, fold in enumerate(chosen):
            reference = fit_audits[seed][fraction_key(1.0)][fold_offset]
            for fraction in config["training_fractions"]:
                audit = fit_audits[seed][fraction_key(fraction)][fold_offset]
                for first, second in zip(reference["inner_selection"] + [reference["final_fit"]], audit["inner_selection"] + [audit["final_fit"]]):
                    if any(first[name] != second[name] for name in ["initial_state_sha256", "parameter_count", "seed"]):
                        raise ValueError("Learning-curve architecture/initialization changed")
    rows, uncertainty = [], {}
    frame = pd.DataFrame(per_seed)
    for fraction in config["training_fractions"]:
        key = fraction_key(fraction)
        counts = [row["train_patients"] for row in descriptions[key].values()]
        part = frame.loc[frame.training_fraction == fraction]
        rows.append({"training_fraction": fraction, "mean_training_patients": float(np.mean(counts)), "min_training_patients": min(counts),
            "max_training_patients": max(counts), "heldout_macro_f1_mean": float(part.heldout_macro_f1.mean()),
            "heldout_macro_f1_sample_std": float(part.heldout_macro_f1.std(ddof=1)),
            "mean_training_macro_f1_across_fits_and_seeds": float(part.mean_training_macro_f1_across_fits.mean())})
        if fraction != 1.0:
            paired = {seed: {key: predictions[seed][key], fraction_key(1.0): predictions[seed][fraction_key(1.0)]} for seed in config["training_seeds"]}
            uncertainty[f"full_minus_{key}"] = paired_seed_bootstrap(truth, paired, len(config["classes"]), draws, config["seed"], arms=(key, fraction_key(1.0)))
    summary = {"complete": True, "study_fingerprint": fingerprint, "n_patients": len(frozen), "n_records": bank["n_records"], "n_beats": bank["n_beats"],
        "training_seeds": config["training_seeds"], "curve": rows, "per_seed": per_seed,
        "fit_count": len(chosen) * len(config["training_seeds"]) * len(config["training_fractions"]) * (config["inner_splits"] + 1),
        "limits": "One nested class-stratified subset realization per outer fold, shared across model seeds; same maximum training budget. No extrapolation, data-size law, independent confirmation or dataset-ceiling claim."}
    _save_json(root / "summary.json", summary)
    _save_json(root / "paired_bootstrap.json", {"study_fingerprint": fingerprint, "comparisons": uncertainty})
    frame.to_csv(root / "per_seed_curve.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "learning_curve.csv", index=False)
    draw_curve(rows, root)
    lines = ["# Patient-training-size sensitivity", "", f"{len(frozen)} identical held-out patients, {bank['n_records']} records, {bank['n_beats']:,} original beats; seeds {config['training_seeds']}; {config['outer_splits']} frozen outer patient folds.", "",
        "![Learning curve](learning_curve.png)", "", "| Nominal per-class fraction | Actual training patients per fit, mean [range] | Held-out macro F1, mean ± seed SD | Mean training resubstitution F1 |", "|---|---|---:|---:|"]
    lines += [f"| {100 * row['training_fraction']:g}% | {row['mean_training_patients']:.1f} [{row['min_training_patients']}, {row['max_training_patients']}] | {row['heldout_macro_f1_mean']:.3f} ± {row['heldout_macro_f1_sample_std']:.3f} | {row['mean_training_macro_f1_across_fits_and_seeds']:.3f} |" for row in rows]
    lines += ["", "## Protocol", "",
        "Every point is freshly trained, including 100%; no historical score is substituted. Class-stratified training-patient prefixes are nested across sizes, chosen by a separate fixed subset seed and class labels/IDs only, never error-audit flags. All beats of selected patients enter the eligible training pool. Per-class counts are rounded up, so actual patient counts differ from nominal percentages. All outer-test patients and beats remain identical across sizes.", "",
        "Inner patient partitions are rebuilt within each selected subset using the same split-seed rule. Normalization and stopping use only that subset's inner-training/validation patients; excluded training-pool patients do not supply validation labels. At 100%, original inner memberships and model-seed rules are retained. Selected epoch budgets are rounded medians of inner best epochs; fresh outer-training refits never monitor outer-test results.", "",
        f"The original 651-sample denoised measured XYZ ResNet is unchanged. No augmentation, label changes, beat exclusions, new detector, ECG input or pretraining. All sizes have {config['samples_per_epoch']:,} class/patient-balanced beat presentations/epoch, batch size {config['batch_size']}, up to {config['max_epochs']} epochs and patience {config['patience']}. Initial weights match across sizes; sampled rows, normalizers, stopping epochs and inner membership can differ. Total fresh fits: {summary['fit_count']}.", "",
        "Training scores are resubstitution on all selected training patients after fitting, averaged across outer fits and then seeds. These repeatedly occurring training patients are not independent observations. Held-out scores are pooled out-of-fold patient macro F1, then averaged across fixed training seeds. These quantities describe different populations/averaging rules; their gap is not a formal bias/variance decomposition. Error bars show three-seed SD, not confidence intervals or training-subset variability.", "",
        "## Paired held-out comparisons", "", "| Comparison | Mean macro F1 difference | Descriptive paired patient interval |", "|---|---:|---|"]
    full = frame.loc[frame.training_fraction == 1].heldout_macro_f1.mean()
    for row in rows[:-1]:
        key = fraction_key(row["training_fraction"])
        lo, hi = uncertainty[f"full_minus_{key}"]["mean_paired_macro_f1_difference_95_interval"]
        lines.append(f"| 100% − {100 * row['training_fraction']:g}% | {full - row['heldout_macro_f1_mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", f"{draws:,} class-stratified patient draws shared across fractions/seeds in each comparison. Conditional on fitted predictions and fixed seeds; no refits, training-subset resampling, correction for shared training folds, repeated development or multiple comparisons.", "",
        "## Mean held-out class F1 across seeds", "", "| Class | " + " | ".join(f"{100 * fraction:g}%" for fraction in config["training_fractions"]) + " |", "|---|" + "---:|" * len(rows)]
    for label in config["classes"]:
        values = [np.mean([class_scores[seed][fraction_key(fraction)][label]["f1-score"] for seed in config["training_seeds"]]) for fraction in config["training_fractions"]]
        lines.append("| " + label + " | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "## Interpretation boundaries", "",
        "Only one nested subset realization per outer fold is used, shared across training seeds. Small class counts make inner stopping noisy. Equal presentations/epoch expose fewer-patient subsets to more repeats per patient, while larger subsets have fewer repeats; a flat or rising curve applies to this fixed model and compute policy. Do not extrapolate a target score, sample-size requirement, physiological cause or PTB performance ceiling. Repeatedly inspected PTB folds are development data, not independent confirmation. MPS is not guaranteed bitwise deterministic.", "",
        "Persistent-error flags do not alter subsets, labels or patient eligibility. No unplanned ensemble or best fraction/seed is selected for a new claimed result. Existing studies and PTB-XL tests are unchanged. Waveforms, weights and individual training/test probabilities remain local. This graph is a diagnostic, not a model-selection sweep.", "",
        "Method background: [scikit-learn learning-curve documentation](https://scikit-learn.org/stable/modules/learning_curve.html). This implementation uses explicit patient groups and inner stopping rather than beat-level default learning-curve splits.", ""]
    destination = root / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/learning_curve_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    print(summarize_study(yaml.safe_load(Path(args.config).read_text()), args.draws))
