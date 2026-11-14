"""Matched short versus longer-context denoised measured XYZ experiment."""
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

from . import neural, preprocessing_study
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle, save_confusion
from .contrastive import _progress
from .neural import _device, _hash, _save_json
from .preprocessing_study import fit_waves, predict_waves
from .signal_audit import file_hash

ARMS = ("short", "long")


def load_banks(config):
    short, long = Path(config["short_bank_dir"]), Path(config["long_bank_dir"])
    manifests = {arm: json.loads((root / "manifest.json").read_text()) for arm, root in [("short", short), ("long", long)]}
    control, extended = manifests["short"], manifests["long"]
    provenance = extended["provenance"]
    if _hash(provenance) != extended["fingerprint"] or provenance["control_fingerprint"] != control["fingerprint"]:
        raise ValueError("Context bank provenance/control changed")
    if any(provenance[key] != config[key] for key in ["classes", "before_r", "after_r", "boundary_mode"]):
        raise ValueError("Context window settings changed")
    paper = provenance["paper"]
    expected = {"short": paper["data"]["before_r"] + paper["data"]["after_r"] + 1,
                "long": config["before_r"] + config["after_r"] + 1}
    hashes = {}
    for arm, root, saved in [("short", short, provenance["control_files"]), ("long", long, extended["files"])]:
        hashes[arm] = {name: file_hash(root / f"{name}.npy") for name in saved}
        if hashes[arm] != saved or manifests[arm]["classes"] != config["classes"]:
            raise ValueError("Context waveform cache content/classes changed")
        hashes[arm]["manifest"] = file_hash(root / "manifest.json")
    labels, groups = [np.load(short / f"{name}.npy") for name in ["y", "groups"]]
    for name, values in [("y", labels), ("groups", groups)]:
        if not np.array_equal(values, np.load(long / f"{name}.npy")):
            raise ValueError("Context patient/label rows do not align")
    waves = {arm: np.load(root / "waveforms.npy", mmap_mode="r") for arm, root in [("short", short), ("long", long)]}
    if len(groups) != len(labels) or any(list(values.shape) != manifests[arm]["shape"] or values.shape != (len(labels), 3, expected[arm]) for arm, values in waves.items()):
        raise ValueError("Context bank shape/cohort changed")
    if extended["n_beats"] != len(labels) or extended["n_patients"] != len(np.unique(groups)):
        raise ValueError("Context bank cohort counts changed")
    return waves, labels, groups, manifests, hashes


def directory(config, outer_folds=None):
    all_folds = list(range(1, config["outer_splits"] + 1))
    chosen = all_folds if outer_folds is None else sorted(set(outer_folds))
    if not chosen or set(chosen) - set(all_folds):
        raise ValueError("Invalid requested context folds")
    partial = chosen != all_folds
    root = Path(config["output_dir"]) / ("partial_folds_" + "_".join(map(str, chosen)) if partial else "complete")
    return root, chosen, partial


def status(config, outer_folds=None):
    root, _, partial = directory(config, outer_folds)
    lines = [f"{'Partial' if partial else 'Full'} context study: {root}"]
    for seed in config["training_seeds"]:
        for arm in ARMS:
            count = len(list((root / f"seed_{seed}").glob(f"fold_*_{arm}.json")))
            lines.append(f"seed {seed:4} {arm:10} {count}/{config['outer_splits']} completed outer folds")
    progress = root / "progress.json"
    if progress.exists():
        lines.append("Last saved progress, not a live process check: " + progress.read_text())
    error = root / "last_error.json"
    if error.exists() and (not progress.exists() or error.stat().st_mtime > progress.stat().st_mtime):
        lines.append("Latest recorded error: " + error.read_text())
    lines.append(f"Final report ready: {(root / 'REPORT.md').exists()}")
    return "\n".join(lines)


def validate_config(config):
    if config["arms"] != list(ARMS) or not config["training_seeds"] or len(set(config["training_seeds"])) != len(config["training_seeds"]):
        raise ValueError("Require both context arms and unique training seeds")
    for key in ["batch_size", "samples_per_epoch", "max_epochs", "patience", "torch_threads"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["inner_splits"] < 2 or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid context training settings")


def run_study(config, outer_folds=None):
    validate_config(config)
    root, chosen, partial = directory(config, outer_folds)
    waves, labels, groups, manifests, cache_hashes = load_banks(config)
    mapping = {name: index for index, name in enumerate(config["classes"])}
    y = np.array([mapping[str(label)] for label in labels])
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require the existing frozen patient folds")
    folds = load_or_create_folds(y, groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"] + 1)):
        raise ValueError("Frozen context fold IDs changed")
    scope = {"config": config, "bank_fingerprints": {arm: manifests[arm]["fingerprint"] for arm in ARMS},
        "cache_hashes": cache_hashes, "folds_sha256": file_hash(config["folds_path"]), "evaluated_folds": chosen,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(),
            "device": str(_device(config)), "deterministic_kernels": _device(config).type != "mps",
            "code_sha256": file_hash(__file__), "fit_code_sha256": file_hash(preprocessing_study.__file__),
            "neural_code_sha256": file_hash(neural.__file__)}}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root / "study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Context study settings/runtime changed; choose a new output_dir")
    _save_json(guard, scope)
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    splits = {fold: list(inner_patient_splits(y[beat_folds != fold], groups[beat_folds != fold], config["inner_splits"], config["split_seed"] + fold * 1000)) for fold in chosen}
    summary = []
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        output.mkdir(exist_ok=True)
        for arm in ARMS:
            probabilities, audits = np.full((len(y), len(mapping)), np.nan), []
            for fold in chosen:
                train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
                if set(groups[train]) & set(groups[test]):
                    raise RuntimeError("Outer context patient leakage")
                checkpoint, audit_path = output / f"fold_{fold:02d}_{arm}.npz", output / f"fold_{fold:02d}_{arm}.json"
                if checkpoint.exists() and audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    with np.load(checkpoint) as saved:
                        if audit["study_fingerprint"] != fingerprint or not np.array_equal(saved["indices"], test):
                            raise ValueError("Context checkpoint settings/cohort changed")
                        probabilities[test] = saved["probabilities"]
                    if not np.isfinite(probabilities[test]).all() or np.any(probabilities[test] < 0) or not np.allclose(probabilities[test].sum(1), 1, atol=1e-5):
                        raise ValueError("Invalid resumed context probabilities")
                    audits.append(audit)
                    print(f"seed {seed} {arm} fold {fold}: resumed completed checkpoint", flush=True)
                    continue
                selections = []
                for inner, (ti, vi) in enumerate(splits[fold], 1):
                    path = output / f"inner_{fold:02d}_{inner}_{arm}.json"
                    if path.exists():
                        selected = json.loads(path.read_text())
                        if selected["study_fingerprint"] != fingerprint:
                            raise ValueError("Inner context settings changed")
                    else:
                        callback = lambda row, inner=inner: _progress(root, state="training", training_seed=seed, arm=arm, outer_fold=fold, inner_fold=inner, **row)
                        model, normalizer, selected = fit_waves(waves[arm], y, groups, train[ti], train[vi], config["max_epochs"], config, seed + fold * 1000 + inner, callback)
                        selected.update(study_fingerprint=fingerprint, inner_fold=inner)
                        _save_json(path, selected)
                        del model, normalizer
                    selections.append(selected)
                    print(f"seed {seed} {arm} outer {fold}/{config['outer_splits']} inner {inner}/{config['inner_splits']}: epoch {selected['selected_epoch']}, macro F1={selected['best_validation_macro_f1']:.4f}", flush=True)
                epochs = int(np.clip(np.floor(np.median([row["selected_epoch"] for row in selections]) + 0.5), 1, config["max_epochs"]))
                callback = lambda row: _progress(root, state="training", training_seed=seed, arm=arm, outer_fold=fold, inner_fold=None, **row)
                model, normalizer, final = fit_waves(waves[arm], y, groups, train, None, epochs, config, seed + fold, callback)
                probabilities[test] = predict_waves(model, normalizer, waves[arm], y, test, config)
                _, truth, scores = aggregate_patients(probabilities[test], y[test], groups[test])
                audit = {"study_fingerprint": fingerprint, "training_seed": seed, "arm": arm, "fold": fold, "patient_overlap": 0,
                    "train_patients": len(np.unique(groups[train])), "test_patients": len(np.unique(groups[test])), "train_beats": len(train), "test_beats": len(test),
                    "inner_selection": selections, "final_epochs": epochs, "final_fit": final, "patient_metrics": metric_bundle(truth, scores, config["classes"])}
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez_compressed(temporary, indices=test, probabilities=probabilities[test])
                temporary.replace(checkpoint)
                _save_json(audit_path, audit)
                weight_path = output / f"model_{fold:02d}_{arm}.pt"
                temporary = weight_path.with_suffix(".pt.partial")
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "normalization": normalizer.audit(), "study_fingerprint": fingerprint}, temporary)
                temporary.replace(weight_path)
                del model, normalizer
                audits.append(audit)
                _progress(root, state="outer_fold_complete", training_seed=seed, arm=arm, outer_fold=fold)
                print(f"seed {seed} {arm} fold {fold}/{config['outer_splits']}: patient macro F1={audit['patient_metrics']['macro_f1']:.4f}", flush=True)
            evaluated = np.isfinite(probabilities).all(1)
            if not partial and not evaluated.all():
                raise RuntimeError("Missing complete context predictions")
            ids, truth, scores = aggregate_patients(probabilities[evaluated], y[evaluated], groups[evaluated])
            saved = {"study_fingerprint": fingerprint, "arm": arm, "training_seed": seed, "classes": config["classes"],
                "complete": not partial, "evaluated_folds": chosen, "n_patients": len(ids), "n_beats": int(evaluated.sum()),
                "patient_metrics": metric_bundle(truth, scores, config["classes"]), "folds": audits,
                "protocol": "Short versus longer denoised crops at identical cached R positions; waveform-only CE ResNet, matched initializations and weighted-random sampling; nested patient-only normalization/stopping; mean beat probabilities per patient. Repeated PTB development folds."}
            _save_json(output / f"metrics_{arm}.json", saved)
            frame = pd.DataFrame({"patient_id": ids, "truth": [config["classes"][i] for i in truth], "prediction": [config["classes"][i] for i in scores.argmax(1)], "fold": [lookup[str(patient)] for patient in ids]})
            for i, label in enumerate(config["classes"]):
                frame[f"probability_{label}"] = scores[:, i]
            frame.to_csv(output / f"patient_predictions_{arm}.csv", index=False)
            save_confusion(truth, scores.argmax(1), config["classes"], arm, output)
            summary.append({"training_seed": seed, "arm": arm, "complete": not partial,
                "patient_macro_f1": saved["patient_metrics"]["macro_f1"], "patient_accuracy": saved["patient_metrics"]["accuracy"]})
            pd.DataFrame(summary).to_csv(root / "comparison.csv", index=False)
    _progress(root, state="partial_complete" if partial else "complete")
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/context_vcg.yaml")
    parser.add_argument("--outer-folds", nargs="+", type=int)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    if args.status:
        print(status(config, args.outer_folds))
    else:
        try:
            print(run_study(config, args.outer_folds).to_string(index=False))
        except BaseException as error:
            root, _, _ = directory(config, args.outer_folds)
            if root.exists():
                _save_json(root / "last_error.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise
