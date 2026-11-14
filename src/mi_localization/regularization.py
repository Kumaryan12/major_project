"""Bounded inner-patient selection of regularization and signal augmentation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import benchmark, neural, preprocessing_study
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights, save_confusion
from .contrastive import _progress, initial_state_hash
from .neural import FoldNormalizer, VCGResNet, _device, _hash, _save_json, _seed
from .preprocessing_study import WaveRows, predict_waves
from .signal_audit import file_hash

ARMS = ("baseline", "selected")
CANDIDATES = ("baseline", "regularized", "augmented", "combined")
AUGMENTATION_SEED_OFFSET = 10_000_019


def has_augmentation(candidate):
    return candidate["gain_min"] != 1 or candidate["gain_max"] != 1 or candidate["noise_std"] != 0


def augment_training_batch(x, normalizer, candidate, generator):
    """CPU-only random stream, independent of model/dropout and row sampling.

    A scalar gain multiplies all physical XYZ samples of each beat equally.
    After normalization: x' = g*x + (g-1)*mean/scale + epsilon.
    epsilon has SD noise_std in training-standardized units per XYZ lead.
    """
    if x.device.type != "cpu":
        raise ValueError("Augmentation draws must use the independent CPU stream")
    if not has_augmentation(candidate):
        return x
    gain = torch.empty((len(x), 1, 1), dtype=x.dtype).uniform_(candidate["gain_min"], candidate["gain_max"], generator=generator)
    offset = torch.as_tensor(normalizer.wave_mean_ / normalizer.wave_scale_, dtype=x.dtype)[None, :, None]
    result = gain * x + (gain - 1) * offset
    if candidate["noise_std"]:
        result = result + candidate["noise_std"] * torch.randn(x.shape, dtype=x.dtype, generator=generator)
    return result


def fit_candidate(waves, y, groups, train, valid, epochs, config, seed, candidate, progress=None):
    settings = dict(config, dropout=candidate["dropout"], weight_decay=candidate["weight_decay"])
    if candidate["id"] == "baseline":
        # Use the frozen original trainer directly for the fresh control.
        model, normalizer, audit = preprocessing_study.fit_waves(waves, y, groups, train, valid, epochs, settings, seed, progress)
        for row in audit["history"]:
            row.update(augmented_presentations=0, augmentation_rng_state_sha256=None)
        audit.update(candidate=candidate, augmentation_seed=None, validation_augmentation=False)
        return model, normalizer, audit
    if valid is not None and set(groups[train]) & set(groups[valid]):
        raise ValueError("Regularization training/validation patients overlap")
    _seed(seed, settings)
    weights = patient_balanced_weights(y[train], groups[train])
    normalizer = FoldNormalizer().fit(waves, np.empty((len(waves), 0), dtype=np.float32), train, weights)
    model = VCGResNet(n_classes=len(config["classes"]), widths=tuple(config["widths"]), pool_bins=config["pool_bins"], dropout=candidate["dropout"]).to(_device(config))
    initial_hash = initial_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=candidate["weight_decay"])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), config["samples_per_epoch"], replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(WaveRows(waves, y, train, normalizer), batch_size=config["batch_size"], sampler=sampler, num_workers=0)
    augmented = has_augmentation(candidate)
    aug_seed = seed + AUGMENTATION_SEED_OFFSET if augmented else None
    aug_rng = torch.Generator().manual_seed(aug_seed) if augmented else None
    best, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses, count, digest = [], 0, hashlib.sha256()
        for x, labels, rows in loader:
            digest.update(rows.numpy().astype("<i8").tobytes())
            if augmented:
                x = augment_training_batch(x, normalizer, candidate, aug_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(_device(config))), labels.to(_device(config)), label_smoothing=candidate["label_smoothing"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite regularization-study loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            count += len(labels)
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)), "sampled_rows_sha256": digest.hexdigest(),
            "beat_presentations": count, "augmented_presentations": count if augmented else 0,
            "augmentation_rng_state_sha256": hashlib.sha256(aug_rng.get_state().numpy().tobytes()).hexdigest() if augmented else None}
        if valid is not None:
            probabilities = predict_waves(model, normalizer, waves, y, valid, config)
            _, truth, scores = aggregate_patients(probabilities, y[valid], groups[valid])
            score = float(f1_score(truth, scores.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
            row["validation_patient_macro_f1"] = score
            if score > best + 1e-8:
                best, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
        if progress:
            progress(row)
        if valid is not None and epoch - best_epoch >= config["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, normalizer, {"seed": seed, "candidate": candidate, "augmentation_seed": aug_seed, "validation_augmentation": False,
        "initial_state_sha256": initial_hash, "selected_epoch": best_epoch if valid is not None else epochs,
        "best_validation_macro_f1": best if valid is not None else None, "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()), "patient_overlap": 0,
        "train_patients": len(np.unique(groups[train])), "validation_patients": 0 if valid is None else len(np.unique(groups[valid])),
        "normalizer_fit_beats": len(train), "normalization": normalizer.audit()}


def choose_candidate(candidates, config):
    scores = []
    for setting in config["candidates"]:
        fits = candidates[setting["id"]]
        if len(fits) != config["inner_splits"]:
            raise ValueError("Missing candidate inner fits")
        values = [fit["best_validation_macro_f1"] for fit in fits]
        epochs = [fit["selected_epoch"] for fit in fits]
        if not np.isfinite(values).all() or any(not 0 <= value <= 1 for value in values) or any(not 1 <= value <= config["max_epochs"] for value in epochs):
            raise ValueError("Invalid inner candidate scores/epochs")
        scores.append({"candidate_id": setting["id"], "mean_best_validation_macro_f1": float(np.mean(values)),
                       "final_epochs": int(np.floor(np.median(epochs) + 0.5))})
    # Exact ties favor the first configured preset, beginning with baseline.
    best = max(range(len(scores)), key=lambda index: scores[index]["mean_best_validation_macro_f1"])
    return {"selected_candidate": scores[best]["candidate_id"], "candidate_scores": scores,
            "rule": "maximum arithmetic mean inner-patient best macro F1; exact ties use configured order; rounded-median selected epochs"}


def validate_config(config):
    if [setting["id"] for setting in config["candidates"]] != list(CANDIDATES):
        raise ValueError("Require exactly four ordered regularization candidates")
    if len(set(config["training_seeds"])) != len(config["training_seeds"]) or not config["training_seeds"]:
        raise ValueError("Require unique training seeds")
    for key in ["batch_size", "samples_per_epoch", "max_epochs", "patience", "torch_threads", "inner_splits", "outer_splits"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < (2 if key in {"inner_splits", "outer_splits"} else 1):
            raise ValueError(f"Invalid positive training setting: {key}")
    for setting in config["candidates"]:
        values = [setting[key] for key in ["dropout", "weight_decay", "label_smoothing", "gain_min", "gain_max", "noise_std"]]
        if not np.isfinite(values).all() or not 0 <= setting["dropout"] < 1 or not 0 <= setting["label_smoothing"] < 1 or setting["weight_decay"] < 0 or not 0 < setting["gain_min"] <= setting["gain_max"] or setting["noise_std"] < 0:
            raise ValueError("Invalid regularization/augmentation settings")
    if config["candidates"][0]["label_smoothing"] != 0 or has_augmentation(config["candidates"][0]) or not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("Baseline must use ordinary CE without augmentation; learning rate must be positive")
    baseline, regularized, augmented, combined = config["candidates"]
    for keys, first, second in [(["dropout", "weight_decay", "label_smoothing"], baseline, augmented),
                                (["dropout", "weight_decay", "label_smoothing"], regularized, combined),
                                (["gain_min", "gain_max", "noise_std"], augmented, combined)]:
        if any(first[key] != second[key] for key in keys):
            raise ValueError("Candidates must preserve the declared two-by-two preset design")
    if has_augmentation(regularized) or not has_augmentation(augmented) or all(regularized[key] == baseline[key] for key in ["dropout", "weight_decay", "label_smoothing"]):
        raise ValueError("Require distinct regularization and augmentation presets")


def load_bank(config):
    root = Path(config["bank_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    paper = yaml.safe_load(Path(config["paper_config"]).read_text())
    if paper["data"]["leads"] != ["vx", "vy", "vz"] or paper["data"]["sampling_rate"] != 1000:
        raise ValueError("Require measured XYZ at 1000 Hz")
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    labels, groups = [np.load(root / f"{name}.npy") for name in ["y", "groups"]]
    if beats.empty or not np.array_equal(labels, beats.label.astype(str).to_numpy()) or not np.array_equal(groups, beats.patient_id.astype(str).to_numpy()):
        raise ValueError("Regularization cohort/beat row alignment changed")
    waves = np.load(root / "waveforms.npy", mmap_mode="r")
    shape = [len(beats), 3, paper["data"]["before_r"] + paper["data"]["after_r"] + 1]
    if manifest["classes"] != config["classes"] or manifest["shape"] != shape or list(waves.shape) != shape or not np.isfinite(waves).all():
        raise ValueError("Regularization bank classes/window/content changed")
    mapping = {label: index for index, label in enumerate(config["classes"])}
    scope = {"fingerprint": manifest["fingerprint"], "files": {name: file_hash(root / name) for name in ["manifest.json", "waveforms.npy", "y.npy", "groups.npy"]},
             "beats_sha256": file_hash(config["beats_path"]), "paper_sha256": file_hash(config["paper_config"]),
             "n_beats": len(beats), "n_patients": int(beats.patient_id.nunique()), "n_records": int(beats.record.nunique()), "shape": shape}
    return waves, np.array([mapping[str(label)] for label in labels]), groups, scope


def directory(config, outer_folds=None):
    full = list(range(1, config["outer_splits"] + 1))
    chosen = full if outer_folds is None else sorted(set(outer_folds))
    if not chosen or set(chosen) - set(full):
        raise ValueError("Invalid requested regularization folds")
    partial = chosen != full
    return Path(config["output_dir"]) / ("partial_folds_" + "_".join(map(str, chosen)) if partial else "complete"), chosen, partial


def status(config, outer_folds=None):
    root, chosen, partial = directory(config, outer_folds)
    lines = [f"{'Partial' if partial else 'Full'} regularization study: {root}"]
    for seed in config["training_seeds"]:
        count = len(list((root / f"seed_{seed}").glob("fold_*.json")))
        lines.append(f"seed {seed}: {count}/{len(chosen)} paired outer folds complete")
    for name in ["progress.json", "last_error.json"]:
        if (root / name).exists():
            lines.append(f"Last saved {name} (not a live process check): " + (root / name).read_text())
    lines.append(f"Final report ready: {(root / 'REPORT.md').exists()}")
    return "\n".join(lines)


def _save_weights(path, model, normalizer, fingerprint, candidate):
    temporary = path.with_suffix(".pt.partial")
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "normalization": normalizer.audit(), "study_fingerprint": fingerprint, "candidate": candidate}, temporary)
    temporary.replace(path)


def run_study(config, outer_folds=None):
    validate_config(config)
    root, chosen, partial = directory(config, outer_folds)
    waves, y, groups, bank = load_bank(config)
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require existing frozen patient folds")
    folds = load_or_create_folds(y, groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"] + 1)):
        raise ValueError("Frozen regularization fold IDs changed")
    scope = {"config": config, "bank": bank, "folds_sha256": file_hash(config["folds_path"]), "evaluated_folds": chosen,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "device": str(_device(config)),
                    "deterministic_kernels": _device(config).type != "mps",
                    "code_sha256": file_hash(__file__), "fit_code_sha256": file_hash(preprocessing_study.__file__),
                    "neural_code_sha256": file_hash(neural.__file__), "benchmark_code_sha256": file_hash(benchmark.__file__)}}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root / "study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Regularization settings/cache/runtime changed; choose a new output_dir")
    _save_json(guard, scope)
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    splits = {fold: list(inner_patient_splits(y[beat_folds != fold], groups[beat_folds != fold], config["inner_splits"], config["split_seed"] + fold * 1000)) for fold in chosen}
    settings = {candidate["id"]: candidate for candidate in config["candidates"]}
    summary = []
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        output.mkdir(exist_ok=True)
        probabilities = {arm: np.full((len(y), len(config["classes"])), np.nan) for arm in ARMS}
        for fold in chosen:
            train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            if set(groups[train]) & set(groups[test]):
                raise RuntimeError("Outer regularization patient leakage")
            checkpoint, audit_path = output / f"fold_{fold:02d}.npz", output / f"fold_{fold:02d}.json"
            if checkpoint.exists() and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                with np.load(checkpoint) as saved:
                    if audit["study_fingerprint"] != fingerprint or not np.array_equal(saved["indices"], test):
                        raise ValueError("Regularization checkpoint cohort/settings changed")
                    for arm in ARMS:
                        scores = saved[arm]
                        if scores.shape != (len(test), len(config["classes"])) or not np.isfinite(scores).all() or np.any(scores < 0) or not np.allclose(scores.sum(1), 1, atol=1e-5):
                            raise ValueError("Invalid resumed regularization probabilities")
                        probabilities[arm][test] = scores
                print(f"seed {seed} fold {fold}: resumed paired checkpoint", flush=True)
                continue
            inner = {}
            for candidate in config["candidates"]:
                candidate_id = candidate["id"]
                inner[candidate_id] = []
                for number, (ti, vi) in enumerate(splits[fold], 1):
                    path = output / f"inner_{fold:02d}_{number}_{candidate_id}.json"
                    if path.exists():
                        fit = json.loads(path.read_text())
                        if fit["study_fingerprint"] != fingerprint or fit["candidate"] != candidate:
                            raise ValueError("Inner regularization settings changed")
                    else:
                        callback = lambda row, number=number: _progress(root, state="training", training_seed=seed, candidate=candidate_id, outer_fold=fold, inner_fold=number, **row)
                        model, normalizer, fit = fit_candidate(waves, y, groups, train[ti], train[vi], config["max_epochs"], config, seed + fold * 1000 + number, candidate, callback)
                        fit.update(study_fingerprint=fingerprint, inner_fold=number)
                        _save_json(path, fit)
                        del model, normalizer
                    inner[candidate_id].append(fit)
                    print(f"seed {seed} outer {fold}/{config['outer_splits']} {candidate_id} inner {number}/{config['inner_splits']}: epoch {fit['selected_epoch']}, macro F1={fit['best_validation_macro_f1']:.4f}", flush=True)
            selection = choose_candidate(inner, config)
            selected = selection["selected_candidate"]
            budgets = {row["candidate_id"]: row["final_epochs"] for row in selection["candidate_scores"]}
            final_fits = {}
            for arm, candidate_id in [("baseline", "baseline"), ("selected", selected)]:
                if arm == "selected" and selected == "baseline":
                    final_fits[arm] = final_fits["baseline"]
                    probabilities[arm][test] = probabilities["baseline"][test]
                    continue
                callback = lambda row: _progress(root, state="training", training_seed=seed, candidate=candidate_id, arm=arm, outer_fold=fold, inner_fold=None, **row)
                model, normalizer, fit = fit_candidate(waves, y, groups, train, None, budgets[candidate_id], config, seed + fold, settings[candidate_id], callback)
                probabilities[arm][test] = predict_waves(model, normalizer, waves, y, test, config)
                final_fits[arm] = fit
                _save_weights(output / f"model_{fold:02d}_{arm}.pt", model, normalizer, fingerprint, settings[candidate_id])
                del model, normalizer
            audit = {"study_fingerprint": fingerprint, "training_seed": seed, "fold": fold, "patient_overlap": 0,
                "train_patients": len(np.unique(groups[train])), "test_patients": len(np.unique(groups[test])), "train_beats": len(train), "test_beats": len(test),
                "inner_candidates": inner, "selection": selection, "final_fits": final_fits, "selected_reuses_baseline_refit": selected == "baseline"}
            temporary = checkpoint.with_suffix(".partial.npz")
            np.savez_compressed(temporary, indices=test, **{arm: probabilities[arm][test] for arm in ARMS})
            temporary.replace(checkpoint)
            _save_json(audit_path, audit)
            _progress(root, state="outer_fold_complete", training_seed=seed, outer_fold=fold, selected_candidate=selected)
            print(f"seed {seed} fold {fold}/{config['outer_splits']}: completed paired predictions; selected {selected}", flush=True)
        for arm in ARMS:
            evaluated = np.isfinite(probabilities[arm]).all(1)
            if not partial and not evaluated.all():
                raise RuntimeError("Missing complete regularization predictions")
            ids, truth, scores = aggregate_patients(probabilities[arm][evaluated], y[evaluated], groups[evaluated])
            saved = {"study_fingerprint": fingerprint, "arm": arm, "training_seed": seed, "classes": config["classes"], "complete": not partial,
                "evaluated_folds": chosen, "n_patients": len(ids), "n_beats": int(evaluated.sum()), "patient_metrics": metric_bundle(truth, scores, config["classes"]),
                "protocol": "Fixed baseline versus four-preset inner-patient selection; same measured denoised XYZ and cached R positions; fresh outer refits, training-only augmentation and normalization; mean all beat probabilities per patient. Repeated PTB development."}
            _save_json(output / f"metrics_{arm}.json", saved)
            frame = pd.DataFrame({"patient_id": ids, "truth": [config["classes"][i] for i in truth], "prediction": [config["classes"][i] for i in scores.argmax(1)], "fold": [lookup[str(patient)] for patient in ids]})
            for i, label in enumerate(config["classes"]):
                frame[f"probability_{label}"] = scores[:, i]
            frame.to_csv(output / f"patient_predictions_{arm}.csv", index=False)
            save_confusion(truth, scores.argmax(1), config["classes"], arm, output)
            summary.append({"training_seed": seed, "arm": arm, "complete": not partial, "patient_macro_f1": saved["patient_metrics"]["macro_f1"], "patient_accuracy": saved["patient_metrics"]["accuracy"]})
        pd.DataFrame(summary).to_csv(root / "comparison.csv", index=False)
    _progress(root, state="partial_complete" if partial else "complete")
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regularization_vcg.yaml")
    parser.add_argument("--outer-folds", nargs="+", type=int)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
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
