"""Patient-training-size sensitivity with frozen outer test folds."""
from __future__ import annotations

import argparse
import json
import platform
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import benchmark, neural, preprocessing_study, regularization
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle, patient_table, save_confusion
from .contrastive import _progress
from .neural import _device, _hash, _save_json
from .preprocessing_study import fit_waves, predict_waves
from .regularization import load_bank
from .signal_audit import file_hash


def fraction_key(fraction):
    return f"fraction_{round(100 * fraction):03d}"


def select_subset(y, groups, eligible, fraction, classes, inner_splits, seed):
    """Nested class-stratified patient prefixes; never consume model errors."""
    eligible = np.sort(np.asarray(eligible, dtype=int))
    patients = patient_table(y[eligible], groups[eligible], classes)
    complete = np.flatnonzero(np.isin(groups, patients.patient_id.to_numpy()))
    if not np.array_equal(eligible, complete) or not 0 < fraction <= 1:
        raise ValueError("Eligible rows must contain complete patients and a valid fraction")
    rng, chosen, counts = np.random.default_rng(seed), [], {}
    for label in classes:
        ids = patients.loc[patients.label == label, "patient_id"].sort_values().to_numpy()
        count = int(np.ceil(fraction * len(ids)))
        if count < inner_splits:
            raise ValueError("Too few patients per class for inner validation at this training fraction")
        chosen.extend(rng.permutation(ids)[:count].tolist())
        counts[label] = count
    indices = eligible[np.isin(groups[eligible], chosen)]
    return indices, {"fraction": fraction, "available_train_patients": len(patients), "train_patients": len(chosen),
                     "train_beats": len(indices), "class_patient_counts": counts, "train_patients_sha256": _hash(sorted(chosen))}


def make_plan(y, groups, folds, config, chosen):
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    plan, descriptions = {}, {}
    for fraction in config["training_fractions"]:
        key = fraction_key(fraction)
        plan[key], descriptions[key] = {}, {}
        for fold in chosen:
            eligible, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            train, description = select_subset(y, groups, eligible, fraction, config["classes"], config["inner_splits"], config["subset_seed"] + fold * 1000)
            inner = [(train[ti], train[vi]) for ti, vi in inner_patient_splits(y[train], groups[train], config["inner_splits"], config["split_seed"] + fold * 1000)]
            description.update(test_patients=len(np.unique(groups[test])), test_beats=len(test),
                inner_folds=[{"inner_fold": number, "train_beats": len(ti), "validation_beats": len(vi),
                    "train_patients": len(np.unique(groups[ti])), "validation_patients": len(np.unique(groups[vi])),
                    "train_patients_sha256": _hash(sorted(np.unique(groups[ti]).tolist())),
                    "validation_patients_sha256": _hash(sorted(np.unique(groups[vi]).tolist()))} for number, (ti, vi) in enumerate(inner, 1)])
            if set(groups[train]) & set(groups[test]) or any(set(groups[ti]) & set(groups[vi]) or set(groups[test]) & (set(groups[ti]) | set(groups[vi])) for ti, vi in inner):
                raise ValueError("Learning-curve patient leakage")
            plan[key][fold] = (train, test, inner)
            descriptions[key][str(fold)] = description
    # The same seeded per-class permutation guarantees nested memberships.
    for fold in chosen:
        previous = set()
        for fraction in config["training_fractions"]:
            current = set(groups[plan[fraction_key(fraction)][fold][0]])
            if not previous <= current:
                raise ValueError("Learning-curve training memberships are not nested")
            previous = current
    return plan, descriptions


def validate_config(config):
    fractions, seeds = config["training_fractions"], config["training_seeds"]
    if len(fractions) < 2 or not np.isfinite(fractions).all() or fractions != sorted(set(fractions)) or fractions[-1] != 1 or any(not 0 < value <= 1 or abs(100 * value - round(100 * value)) > 1e-9 for value in fractions):
        raise ValueError("Require increasing distinct whole-percent training fractions ending at 1")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require distinct training seeds")
    for key in ["inner_splits", "outer_splits", "batch_size", "samples_per_epoch", "max_epochs", "patience", "torch_threads"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < (2 if key in {"inner_splits", "outer_splits"} else 1):
            raise ValueError(f"Invalid positive learning-curve setting: {key}")
    if not np.isfinite([config["dropout"], config["weight_decay"], config["learning_rate"]]).all() or not 0 <= config["dropout"] < 1 or config["weight_decay"] < 0 or config["learning_rate"] <= 0:
        raise ValueError("Invalid learning-curve optimizer settings")


def directory(config, outer_folds=None):
    full = list(range(1, config["outer_splits"] + 1))
    chosen = full if outer_folds is None else sorted(set(outer_folds))
    if not chosen or set(chosen) - set(full):
        raise ValueError("Invalid requested learning-curve folds")
    partial = chosen != full
    return Path(config["output_dir"]) / ("partial_folds_" + "_".join(map(str, chosen)) if partial else "complete"), chosen, partial


def inputs_and_plan(config, chosen):
    waves, y, groups, bank = load_bank(config)
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require the existing frozen patient folds")
    folds = load_or_create_folds(y, groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"] + 1)):
        raise ValueError("Learning-curve frozen fold IDs changed")
    plan, descriptions = make_plan(y, groups, folds, config, chosen)
    return waves, y, groups, bank, folds, plan, descriptions


def status(config, outer_folds=None):
    root, chosen, partial = directory(config, outer_folds)
    lines = [f"{'Partial' if partial else 'Full'} learning curve: {root}"]
    for seed in config["training_seeds"]:
        for fraction in config["training_fractions"]:
            count = len(list((root / f"seed_{seed}" / fraction_key(fraction)).glob("fold_*.json")))
            lines.append(f"seed {seed}, {100 * fraction:g}% training patients: {count}/{len(chosen)} outer folds complete")
    for name in ["progress.json", "last_error.json"]:
        if (root / name).exists():
            lines.append(f"Last saved {name} (not a live process check): " + (root / name).read_text())
    lines.append(f"Final report ready: {(root / 'REPORT.md').exists()}")
    return "\n".join(lines)


def run_study(config, outer_folds=None):
    validate_config(config)
    root, chosen, partial = directory(config, outer_folds)
    waves, y, groups, bank, folds, plan, descriptions = inputs_and_plan(config, chosen)
    scope = {"config": config, "bank": bank, "folds_sha256": file_hash(config["folds_path"]), "evaluated_folds": chosen, "subset_plan": descriptions,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "device": str(_device(config)),
                    "deterministic_kernels": _device(config).type != "mps", "code_sha256": file_hash(__file__),
                    "fit_code_sha256": file_hash(preprocessing_study.__file__), "neural_code_sha256": file_hash(neural.__file__),
                    "loader_code_sha256": file_hash(regularization.__file__), "benchmark_code_sha256": file_hash(benchmark.__file__)}}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root / "study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Learning-curve settings/cache/runtime/subsets changed; use a new output_dir")
    _save_json(guard, scope)
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    summary = []
    for seed in config["training_seeds"]:
        for fraction in config["training_fractions"]:
            key = fraction_key(fraction)
            output = root / f"seed_{seed}" / key
            output.mkdir(parents=True, exist_ok=True)
            probabilities, audits = np.full((len(y), len(config["classes"])), np.nan), []
            for fold in chosen:
                train, test, inner = plan[key][fold]
                checkpoint, audit_path = output / f"fold_{fold:02d}.npz", output / f"fold_{fold:02d}.json"
                if checkpoint.exists() and audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    with np.load(checkpoint) as saved:
                        scores = saved["probabilities"]
                        if audit["study_fingerprint"] != fingerprint or audit["training_seed"] != seed or audit["training_fraction"] != fraction or audit["fold"] != fold or audit["subset"] != descriptions[key][str(fold)] or not np.array_equal(saved["indices"], test):
                            raise ValueError("Learning-curve checkpoint settings/cohort changed")
                        if scores.shape != (len(test), len(config["classes"])) or not np.isfinite(scores).all() or np.any(scores < 0) or not np.allclose(scores.sum(1), 1, atol=1e-5):
                            raise ValueError("Invalid resumed learning-curve predictions")
                        probabilities[test] = scores
                    audits.append(audit)
                    print(f"seed {seed} {key} fold {fold}: resumed checkpoint", flush=True)
                    continue
                selections = []
                for number, (ti, vi) in enumerate(inner, 1):
                    path = output / f"inner_{fold:02d}_{number}.json"
                    if path.exists():
                        fit = json.loads(path.read_text())
                        if fit["study_fingerprint"] != fingerprint or fit["training_seed"] != seed or fit["training_fraction"] != fraction or fit["outer_fold"] != fold or fit["inner_fold"] != number:
                            raise ValueError("Inner learning-curve identity/settings changed")
                    else:
                        callback = lambda row, number=number: _progress(root, state="training", training_seed=seed, training_fraction=fraction, outer_fold=fold, inner_fold=number, **row)
                        model, normalizer, fit = fit_waves(waves, y, groups, ti, vi, config["max_epochs"], config, seed + fold * 1000 + number, callback)
                        fit.update(study_fingerprint=fingerprint, inner_fold=number, training_seed=seed, training_fraction=fraction, outer_fold=fold)
                        _save_json(path, fit)
                        del model, normalizer
                    selections.append(fit)
                    print(f"seed {seed} {key} outer {fold}/{config['outer_splits']} inner {number}: epoch {fit['selected_epoch']}, macro F1={fit['best_validation_macro_f1']:.4f}", flush=True)
                epochs = int(np.floor(np.median([fit["selected_epoch"] for fit in selections]) + 0.5))
                callback = lambda row: _progress(root, state="training", training_seed=seed, training_fraction=fraction, outer_fold=fold, inner_fold=None, **row)
                model, normalizer, final = fit_waves(waves, y, groups, train, None, epochs, config, seed + fold, callback)
                # Resubstitution is descriptive, computed after fitting; never
                # used to choose epochs, fractions, architecture or test cases.
                train_probabilities = predict_waves(model, normalizer, waves, y, train, config)
                train_ids, train_truth, train_scores = aggregate_patients(train_probabilities, y[train], groups[train])
                probabilities[test] = predict_waves(model, normalizer, waves, y, test, config)
                audit = {"study_fingerprint": fingerprint, "training_seed": seed, "training_fraction": fraction, "fold": fold,
                    "patient_overlap": 0, "subset": descriptions[key][str(fold)], "inner_selection": selections,
                    "final_epochs": epochs, "final_fit": final, "training_patient_metrics": metric_bundle(train_truth, train_scores, config["classes"])}
                weight_path = output / f"model_{fold:02d}.pt"
                temporary = weight_path.with_suffix(".pt.partial")
                torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}, "normalization": normalizer.audit(),
                            "study_fingerprint": fingerprint, "training_fraction": fraction}, temporary)
                temporary.replace(weight_path)
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez_compressed(temporary, indices=test, probabilities=probabilities[test],
                                    train_patient_ids=train_ids, train_truth=train_truth, train_probabilities=train_scores)
                temporary.replace(checkpoint)
                _save_json(audit_path, audit)
                del model, normalizer
                audits.append(audit)
                _progress(root, state="outer_fold_complete", training_seed=seed, training_fraction=fraction, outer_fold=fold)
                print(f"seed {seed} {key} fold {fold}/{config['outer_splits']}: {len(np.unique(groups[train]))} training patients", flush=True)
            evaluated = np.isfinite(probabilities).all(1)
            if not partial and not evaluated.all():
                raise RuntimeError("Missing complete learning-curve predictions")
            ids, truth, scores = aggregate_patients(probabilities[evaluated], y[evaluated], groups[evaluated])
            saved = {"study_fingerprint": fingerprint, "training_seed": seed, "training_fraction": fraction, "arm": key,
                "classes": config["classes"], "complete": not partial, "evaluated_folds": chosen, "n_patients": len(ids), "n_beats": int(evaluated.sum()),
                "patient_metrics": metric_bundle(truth, scores, config["classes"]),
                "mean_training_patient_macro_f1_across_fits": float(np.mean([audit["training_patient_metrics"]["macro_f1"] for audit in audits]))}
            _save_json(output / "metrics_resnet.json", saved)
            frame = pd.DataFrame({"patient_id": ids, "truth": [config["classes"][i] for i in truth], "prediction": [config["classes"][i] for i in scores.argmax(1)], "fold": [lookup[str(patient)] for patient in ids]})
            for index, label in enumerate(config["classes"]):
                frame[f"probability_{label}"] = scores[:, index]
            frame.to_csv(output / "patient_predictions_resnet.csv", index=False)
            save_confusion(truth, scores.argmax(1), config["classes"], "resnet", output)
            summary.append({"training_seed": seed, "training_fraction": fraction, "complete": not partial,
                "patient_macro_f1": saved["patient_metrics"]["macro_f1"], "mean_training_patient_macro_f1_across_fits": saved["mean_training_patient_macro_f1_across_fits"]})
            pd.DataFrame(summary).to_csv(root / "comparison.csv", index=False)
    _progress(root, state="partial_complete" if partial else "complete")
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/learning_curve_vcg.yaml")
    parser.add_argument("--outer-folds", nargs="+", type=int)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Validate subsets and print actual counts without fitting or writing results")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        print(status(config, args.outer_folds))
    elif args.plan:
        validate_config(config)
        _, chosen, _ = directory(config, args.outer_folds)
        *_, descriptions = inputs_and_plan(config, chosen)
        print(pd.DataFrame([{"fraction": float(key.split('_')[1]) / 100, "outer_fold": int(fold), "training_patients": row["train_patients"], "available_training_patients": row["available_train_patients"], "class_patient_counts": row["class_patient_counts"]} for key, folds in descriptions.items() for fold, row in folds.items()]).to_string(index=False))
    else:
        try:
            print(run_study(config, args.outer_folds).to_string(index=False))
        except BaseException as error:
            root, _, _ = directory(config, args.outer_folds)
            if root.exists():
                _save_json(root / "last_error.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise
