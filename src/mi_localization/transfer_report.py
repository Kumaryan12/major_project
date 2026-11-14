"""Validate all prespecified transfer fits and report exploratory patient scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import transfer
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle
from .contrastive_report import paired_seed_bootstrap
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .signal_audit import file_hash


def summarize_study(config, draws=5000):
    transfer.validate_config(config)
    root, chosen, partial = transfer.directory(config)
    if partial or len(config["training_seeds"]) < 3:
        raise ValueError("Final transfer report requires all folds and at least three target seeds")
    scope = json.loads((root / "study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    _, y, groups, bank = transfer.load_bank(config)
    encoder, source = transfer.load_source(config, bank)
    encoder_hash = transfer.initial_state_hash(transfer.initial_model(config, config["seed"], "pretrained", encoder).encoder)
    if _hash(scope) != fingerprint or scope["config"] != config or scope["bank"] != bank or scope["source"] != source or scope["source_encoder_state_sha256"] != encoder_hash or scope["folds_sha256"] != file_hash(config["folds_path"]) or scope["evaluated_folds"] != chosen or scope["claim_scope"] != transfer.CLAIM_SCOPE:
        raise ValueError("Transfer report settings/cohort/source changed")
    if scope["runtime"]["code_sha256"] != file_hash(transfer.__file__):
        raise ValueError("Transfer trainer code changed")
    frozen = load_or_create_folds(y, groups, config).set_index("patient_id").sort_index()
    lookup = frozen.fold.to_dict()
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    predictions, scores_by_seed, rows, selections = {}, {}, [], []
    fit_count = 0
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        predictions[seed], metrics, audits = {}, {}, {}
        for arm in transfer.ARMS:
            saved = json.loads((output / f"metrics_{arm}.json").read_text())
            if saved["study_fingerprint"] != fingerprint or saved["arm"] != arm or saved["training_seed"] != seed or saved["n_beats"] != bank["n_beats"] or saved["evaluated_folds"] != chosen or saved["patient_identity_independence_cleared"] is not False or saved["claim_scope"] != transfer.CLAIM_SCOPE:
                raise ValueError("Incomplete/mismatched transfer metrics or claim scope")
            truth, pred = _read_predictions(output / f"patient_predictions_{arm}.csv", saved, frozen, config["classes"])
            predictions[seed][arm], metrics[arm], audits[arm] = pred, saved["patient_metrics"], []
            pooled = np.full((len(y), len(config["classes"])), np.nan)
            for fold in chosen:
                train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
                path = output / f"fold_{fold:02d}_{arm}.json"
                audit = json.loads(path.read_text())
                if audit["study_fingerprint"] != fingerprint or audit["arm"] != arm or audit["training_seed"] != seed or audit["fold"] != fold or audit["patient_overlap"] or audit["train_beats"] != len(train) or audit["test_beats"] != len(test) or audit["train_patients"] != len(np.unique(groups[train])) or audit["test_patients"] != len(np.unique(groups[test])):
                    raise ValueError("Invalid transfer outer-fold membership/audit")
                splits = [(train[ti], train[vi]) for ti, vi in inner_patient_splits(y[train], groups[train], config["inner_splits"], config["split_seed"] + fold * 1000)]
                candidates = audit["inner_candidates"]
                if set(candidates) != {f"rate_{i}" for i in range(len(config["learning_rates"]))}:
                    raise ValueError("Missing learning-rate candidates")
                for index, rate in enumerate(config["learning_rates"]):
                    fits = candidates[f"rate_{index}"]
                    if len(fits) != len(splits):
                        raise ValueError("Missing transfer inner fits")
                    for inner, (fit, (ti, vi)) in enumerate(zip(fits, splits), 1):
                        if fit["study_fingerprint"] != fingerprint or fit["inner_fold"] != inner:
                            raise ValueError("Transfer inner checkpoint provenance mismatch")
                        transfer.validate_fit(fit, config, seed + fold * 1000 + inner, arm, rate, len(ti), len(np.unique(groups[ti])), len(np.unique(groups[vi])), encoder_hash)
                        transfer.check_pair(candidates["rate_0"][inner - 1], fit)
                        if fit["initial_state_sha256"] != candidates["rate_0"][inner - 1]["initial_state_sha256"]:
                            raise ValueError("Learning-rate candidates changed initial weights")
                selection = transfer.choose_rate(candidates, config)
                if audit["selection"] != selection:
                    raise ValueError("Learning rate must follow inner-only selection")
                selected = selection["selected"]
                transfer.validate_fit(audit["final_fit"], config, seed + fold, arm, selected["learning_rate"], len(train), len(np.unique(groups[train])), 0, encoder_hash, selected["final_epochs"])
                checkpoint = output / f"fold_{fold:02d}_{arm}.npz"
                if audit["prediction_checkpoint_sha256"] != file_hash(checkpoint):
                    raise ValueError("Transfer prediction checkpoint changed")
                with np.load(checkpoint) as values:
                    if not np.array_equal(values["indices"], test):
                        raise ValueError("Transfer prediction membership mismatch")
                    p = values["probabilities"]
                if p.shape != (len(test), len(config["classes"])) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
                    raise ValueError("Invalid saved transfer probabilities")
                _, fold_truth, fold_scores = aggregate_patients(p, y[test], groups[test])
                if audit["patient_metrics"] != metric_bundle(fold_truth, fold_scores, config["classes"]):
                    raise ValueError("Fold metrics do not match saved predictions")
                pooled[test] = p
                audits[arm].append(audit)
                selections.append({"training_seed": seed, "arm": arm, "outer_fold": fold, "selected_learning_rate": selected["learning_rate"], "refit_epochs": selected["final_epochs"],
                    **{f"mean_inner_f1_{row['candidate']}": row["mean_best_validation_macro_f1"] for row in selection["candidates"]}})
                fit_count += len(config["learning_rates"]) * config["inner_splits"] + 1
            if saved["folds"] != audits[arm]:
                raise ValueError("Metrics and per-fold audits differ")
            ids, pooled_truth, pooled_scores = aggregate_patients(pooled, y, groups)
            frame = pd.read_csv(output / f"patient_predictions_{arm}.csv").set_index("patient_id").sort_index()
            if not np.array_equal(ids, frame.index.to_numpy()) or not np.array_equal(pooled_truth, truth) or not np.allclose(pooled_scores, frame[[f"probability_{label}" for label in config["classes"]]].to_numpy(), atol=1e-12, rtol=1e-7) or metrics[arm] != metric_bundle(pooled_truth, pooled_scores, config["classes"]):
                raise ValueError("Pooled transfer results do not match all held-out beats")
        for first, second in zip(audits["scratch"], audits["pretrained"]):
            for index in range(len(config["learning_rates"])):
                for a, b in zip(first["inner_candidates"][f"rate_{index}"], second["inner_candidates"][f"rate_{index}"]):
                    transfer.check_pair(a, b)
                    if a["initial_state_sha256"] == b["initial_state_sha256"]:
                        raise ValueError("Transfer arms unexpectedly used the same encoder initialization")
            transfer.check_pair(first["final_fit"], second["final_fit"])
        scores_by_seed[seed] = metrics
        a, b = [metrics[arm]["macro_f1"] for arm in transfer.ARMS]
        rows.append({"training_seed": seed, "scratch_macro_f1": a, "pretrained_macro_f1": b, "pretrained_minus_scratch_macro_f1": b - a})
    table = pd.DataFrame(rows)
    statistics = {arm: {"macro_f1_mean": float(table[f"{arm}_macro_f1"].mean()), "macro_f1_sample_std": float(table[f"{arm}_macro_f1"].std(ddof=1))} for arm in transfer.ARMS}
    uncertainty = paired_seed_bootstrap(truth, predictions, len(config["classes"]), draws, config["seed"], arms=transfer.ARMS)
    uncertainty["study_fingerprint"] = fingerprint
    summary = {"complete": True, "study_fingerprint": fingerprint, "n_patients": len(frozen), "n_records": bank["n_records"], "n_beats": bank["n_beats"],
        "training_seeds": config["training_seeds"], "pretraining_seeds": source["pretraining_seeds"], "arms": statistics, "per_seed": rows,
        "unique_target_fit_count": fit_count, "mean_pretrained_minus_scratch_macro_f1": float(table.pretrained_minus_scratch_macro_f1.mean()),
        "patient_identity_independence_cleared": False, "claim_scope": transfer.CLAIM_SCOPE}
    _save_json(root / "summary.json", summary)
    _save_json(root / "paired_bootstrap.json", uncertainty)
    table.to_csv(root / "paired_comparison.csv", index=False)
    pd.DataFrame(selections).to_csv(root / "selection_summary.csv", index=False)
    lines = ["# Exploratory masked-encoder transfer to measured VCG", "", transfer.CLAIM_SCOPE, "",
        f"{len(frozen)} patients, {bank['n_records']} records, {bank['n_beats']:,} denoised measured-Frank XYZ beats, {len(config['classes'])} classes; {config['outer_splits']} frozen patient folds; target training seeds {config['training_seeds']}.", "",
        "| Target training seed | Scratch macro F1 | Pretrained macro F1 | Pretrained − scratch |", "|---|---:|---:|---:|"]
    lines += [f"| {row['training_seed']} | {row['scratch_macro_f1']:.3f} | {row['pretrained_macro_f1']:.3f} | {row['pretrained_minus_scratch_macro_f1']:+.3f} |" for row in rows]
    for arm, values in statistics.items():
        lines += ["", f"{arm}: **{values['macro_f1_mean']:.3f} ± {values['macro_f1_sample_std']:.3f}** (mean ± sample SD across target training seeds)."]
    lo, hi = uncertainty["mean_paired_macro_f1_difference_95_interval"]
    lines += ["", f"Mean paired difference **{summary['mean_pretrained_minus_scratch_macro_f1']:+.3f}**, descriptive paired patient-bootstrap interval [{lo:+.3f}, {hi:+.3f}] ({draws:,} class-stratified draws shared across both arms and all fixed seeds).", "",
        "## Exact comparison", "",
        f"Source encoder: one fixed masked-reconstruction pilot, seed {source['pretraining_seeds']}, {source['source_training_patients']:,} source-training and {source['source_validation_patients']} source-validation PTB-XL patients. PTB-XL fold 10 was excluded. Source validation selects the source encoder; no PTB target examples enter source training or source stopping. Three target seeds are not three independent pretraining replications.", "",
        "Only encoder weights transfer. The six-class classifier is freshly initialized with the same weights in paired fits; source reconstruction decoder, normalization and optimizer are discarded. Both arms fine-tune the entire same compact ResNet with ordinary CE, dropout and AdamW. Target normalization is fitted only on each target-training partition with the same patient/class weights in both arms.", "",
        f"Both arms receive the SAME two target learning-rate candidates {config['learning_rates']}, up to {config['max_epochs']} epochs, patience {config['patience']}, {config['samples_per_epoch']:,} sampled beats per epoch, and {config['inner_splits']} fixed inner patient splits per outer fold. The maximum mean best inner-patient macro F1 selects the rate separately per arm; exact ties use configured order. Rounded median best inner epoch sets a fresh outer-training refit. Outer test patients never select rates, epochs or normalization. Historical 0.550 ± 0.040 is context, not the control for this new learning-rate search.", "",
        f"Validated {fit_count} target fits: initialization provenance, fresh matching classifier, equal parameter counts, training normalization, seeds and shared-epoch sampled-row digests. Initial encoders deliberately differ across arms. Actual early-stopped training lengths may differ. Pretraining adds compute beyond the equal target tuning budgets. All held-out beats contribute to mean patient probabilities; there is no best-seed selection or probability ensemble.", "",
        "## Mean class F1 across target seeds", "", "| Class | Patients | Scratch | Pretrained |", "|---|---:|---:|---:|"]
    for label in config["classes"]:
        values = [np.mean([scores_by_seed[seed][arm]["classification_report"][label]["f1-score"] for seed in config["training_seeds"]]) for arm in transfer.ARMS]
        lines.append(f"| {label} | {(frozen.label == label).sum()} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ["", "## Interpretation limits", "",
        "The source–target recording screen passed its bounded top-k continuous-ECG design with synthetic copied-segment controls. This is not proof of disjoint patient identities. See [screen report](../overlap_audit/REPORT.md). Different recordings of the same person, shorter/modified copies and candidates outside retrieval can be missed. No target patient was removed or relabeled.", "",
        "Kors-derived source XYZ differs from directly measured Frank target XYZ; better source reconstruction does not guarantee target improvement. All PTB folds have been repeatedly inspected during development. These bootstrap intervals are conditional on fixed fitted predictions and one fixed source encoder, without model refits, seed resampling or correction for correlated folds, repeated development or multiple testing. The study cannot establish patient-independent external confirmation, clinical utility or SOTA. It does not guarantee 0.80–0.90 macro F1.", "",
        "Cached full-record denoising remains offline/noncausal. Seeded MPS is not guaranteed bitwise deterministic. Raw waveforms, recording-pair identities, patient predictions and weights remain local; only aggregate results and audits are published.", "",
        "Reproduce: `.venv/bin/python -m mi_localization.overlap_screen`, then `.venv/bin/python -m mi_localization.transfer`, then `.venv/bin/python -m mi_localization.transfer_report`. Read progress: `.venv/bin/python -m mi_localization.transfer --status`.", ""]
    destination = root / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transfer_vcg.yaml")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    print(summarize_study(yaml.safe_load(Path(args.config).read_text()), args.draws))
