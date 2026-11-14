"""Matched CE vs cross-patient supervised contrastive VCG development.

Both arms use identical architecture, initial weights and structured batches.
No projection head, augmentation, pretraining or held-out feature memory bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from . import neural
from .benchmark import (aggregate_patients, inner_patient_splits, load_or_create_folds,
                        metric_bundle, patient_balanced_weights, patient_table, save_confusion)
from .neural import FoldNormalizer, VCGResNet, _device, _hash, _predict_network, _save_json, _seed


ARMS = ("matched_ce", "cross_patient_supcon")


def cross_patient_masks(labels, patients):
    if labels.ndim != 1 or patients.shape != labels.shape:
        raise ValueError("Expected aligned one-dimensional labels and patient codes")
    allowed = patients[:, None] != patients[None, :]
    positives = allowed & (labels[:, None] == labels[None, :])
    return allowed, positives


def cross_patient_loss(features, labels, patients, temperature=0.1):
    """Mean positive log probability, excluding same-patient pairs entirely.

    This is a task-specific SupCon adaptation on the 64-D encoder output.
    Anchors lacking cross-patient positives contribute no loss. Structured
    batches in the real study guarantee such positives for every anchor.
    """
    if features.ndim != 2 or len(features) != len(labels) or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Invalid contrastive feature shape or temperature")
    allowed, positives = cross_patient_masks(labels, patients)
    valid = positives.any(dim=1)
    if not valid.any():
        return features.sum() * 0
    normalized = F.normalize(features, dim=1, eps=1e-8)
    logits = normalized @ normalized.T / temperature
    # A valid positive implies at least two patients globally, so every row
    # has an allowed denominator. Avoid boolean row indexing/scatter backward
    # to support deterministic masked reductions on the MPS backend.
    denominator = torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)
    log_probability = logits - denominator[:, None]
    positive_mean = torch.where(positives, log_probability, 0).sum(1) / positives.sum(1).clamp_min(1)
    return -(positive_mean * valid).sum() / valid.sum()


class StructuredPatientBatches:
    """Uniform classes/patients, with P distinct patients per class per batch.

    Yields positions in the training-row dataset, not global held-out rows.
    A dedicated NumPy generator makes draws independent of model gradients.
    """
    def __init__(self, y, groups, indices, n_classes, patients_per_class,
                 beats_per_patient, batches, seed):
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError("Training row indices must be nonempty and unique")
        if patients_per_class < 2 or beats_per_patient < 1 or batches < 1:
            raise ValueError("Need at least two patients per class and positive batch settings")
        patient_table(y[indices], groups[indices], [str(i) for i in range(n_classes)])
        self.pools = []
        for label in range(n_classes):
            positions = np.flatnonzero(y[indices] == label)
            patients = np.unique(groups[indices[positions]])
            if len(patients) < patients_per_class:
                raise ValueError(f"Class {label} has fewer than {patients_per_class} training patients")
            self.pools.append([positions[groups[indices[positions]] == patient] for patient in patients])
        self.patients_per_class, self.beats_per_patient = patients_per_class, beats_per_patient
        self.batches, self.seed, self.epoch = batches, seed, 0

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1000003)
        self.epoch += 1
        for _ in range(self.batches):
            batch = []
            for patient_pools in self.pools:
                selected = rng.choice(len(patient_pools), self.patients_per_class, replace=False)
                for patient in selected:
                    rows = patient_pools[patient]
                    batch.extend(rng.choice(rows, self.beats_per_patient,
                                            replace=len(rows) < self.beats_per_patient).tolist())
            rng.shuffle(batch)
            yield batch


class ContrastiveBeats(Dataset):
    def __init__(self, waves, y, patient_codes, indices, normalizer):
        self.waves, self.y, self.patient_codes = waves, y, patient_codes
        self.indices, self.normalizer = np.asarray(indices), normalizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        x = (np.asarray(self.waves[row], dtype=np.float32) - self.normalizer.wave_mean_[:, None]) / self.normalizer.wave_scale_[:, None]
        return torch.from_numpy(x.copy()), int(self.y[row]), int(self.patient_codes[row]), int(row)


def initial_state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_contrastive(arm, waves, statistics, y, groups, train, valid, epochs,
                    config, seed, progress=None):
    if arm not in ARMS:
        raise ValueError("Unknown contrastive arm")
    if valid is not None and set(groups[train]) & set(groups[valid]):
        raise ValueError("Training and early-stopping patients overlap")
    _seed(seed, config)
    device = _device(config)
    normalizer = FoldNormalizer().fit(waves, statistics, train, patient_balanced_weights(y[train], groups[train]))
    codes, _ = pd.factorize(groups, sort=True)
    sampler = StructuredPatientBatches(y, groups, train, len(config["classes"]),
        config["patients_per_class"], config["beats_per_patient"], config["batches_per_epoch"], seed)
    loader = DataLoader(ContrastiveBeats(waves, y, codes, train, normalizer), batch_sampler=sampler, num_workers=0)
    model = VCGResNet(n_classes=len(config["classes"]), widths=tuple(config["widths"]),
                     pool_bins=config["pool_bins"], dropout=config["dropout"]).to(device)
    initial_hash = initial_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    coefficient = 0.0 if arm == "matched_ce" else float(config["contrastive_weight"])
    best_score, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses, ce_losses, cp_losses = [], [], []
        draw_hash = hashlib.sha256()
        beats_seen, anchors_seen, positive_anchors = 0, 0, 0
        for x, labels, patients, rows in loader:
            draw_hash.update(rows.numpy().astype("<i8").tobytes())
            x, labels, patients = x.to(device), labels.to(device), patients.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model.encoder(x)
            ce = F.cross_entropy(model.classifier(embeddings), labels)
            cp = cross_patient_loss(embeddings, labels, patients, config["temperature"])
            loss = ce if coefficient == 0 else ce + coefficient * cp
            if not torch.isfinite(loss) or not torch.isfinite(cp):
                raise RuntimeError("Nonfinite matched contrastive loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            _, positives = cross_patient_masks(labels, patients)
            positive_anchors += int(positives.any(1).sum().item())
            anchors_seen += len(labels)
            beats_seen += len(labels)
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(ce.detach().cpu()))
            cp_losses.append(float(cp.detach().cpu()))
        if positive_anchors != anchors_seen:
            raise RuntimeError("Structured batches failed to provide cross-patient positives")
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)),
            "cross_entropy": float(np.mean(ce_losses)), "cross_patient_supcon": float(np.mean(cp_losses)),
            "batch_indices_sha256": draw_hash.hexdigest(), "batches": len(sampler),
            "beat_presentations": beats_seen, "positive_anchor_fraction": positive_anchors / anchors_seen}
        if valid is not None:
            probabilities = _predict_network(model, normalizer, waves, statistics, y, valid, config)
            _, truth, patient_scores = aggregate_patients(probabilities, y[valid], groups[valid])
            score = float(f1_score(truth, patient_scores.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
            row["validation_patient_macro_f1"] = score
            if score > best_score + 1e-8:
                best_score, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
        if progress is not None:
            progress(row)
        if valid is not None and epoch - best_epoch >= config["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    audit = {"seed": seed, "selected_epoch": best_epoch if valid is not None else epochs,
        "best_validation_macro_f1": best_score if valid is not None else None, "history": history,
        "epochs_run": len(history), "initial_state_sha256": initial_hash,
        "contrastive_weight": coefficient, "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_patients": len(np.unique(groups[train])), "validation_patients": 0 if valid is None else len(np.unique(groups[valid])),
        "normalizer_fit_beats": len(train), "normalization": normalizer.audit(), "patient_overlap": 0}
    return model, normalizer, audit


def validate_config(config):
    if config["arms"] != list(ARMS):
        raise ValueError("This matched study requires both ordered CE and contrastive arms")
    if not config["training_seeds"] or len(set(config["training_seeds"])) != len(config["training_seeds"]):
        raise ValueError("Choose unique training seeds")
    for name in ["patients_per_class", "beats_per_patient", "batches_per_epoch", "batch_size", "max_epochs", "patience", "torch_threads"]:
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["patients_per_class"] < 2:
        raise ValueError("Need at least two patients per class")
    if config["batch_size"] != len(config["classes"]) * config["patients_per_class"] * config["beats_per_patient"]:
        raise ValueError("batch_size must match the structured class/patient/beat product")
    if config["inner_splits"] < 2 or len(set(config["classes"])) != len(config["classes"]):
        raise ValueError("Invalid patient splits or class list")
    for name in ["temperature", "contrastive_weight", "learning_rate"]:
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")


def study_directory(config, outer_folds=None):
    all_folds = list(range(1, config["outer_splits"] + 1))
    chosen = all_folds if outer_folds is None else sorted(set(outer_folds))
    if not chosen or set(chosen) - set(all_folds):
        raise ValueError("Invalid requested outer folds")
    partial = chosen != all_folds
    directory = Path(config["output_dir"]) / ("partial_folds_" + "_".join(map(str, chosen)) if partial else "complete")
    return directory, chosen, partial


def _progress(root, **values):
    _save_json(root / "progress.json", {"updated_utc": datetime.now(timezone.utc).isoformat(),
                                      "pid": os.getpid(), **values})


def status(config, outer_folds=None):
    root, _, partial = study_directory(config, outer_folds)
    lines = [f"{'Partial' if partial else 'Full'} study: {root}"]
    for seed in config["training_seeds"]:
        for arm in ARMS:
            count = len(list((root / f"seed_{seed}").glob(f"fold_*_{arm}.json")))
            lines.append(f"seed {seed:4}  {arm:24} {count}/{config['outer_splits']} completed outer folds")
    path = root / "progress.json"
    if path.exists():
        lines.append("Last saved progress (not a live process check): " + path.read_text())
    error_path = root / "last_error.json"
    if error_path.exists() and (not path.exists() or error_path.stat().st_mtime > path.stat().st_mtime):
        error = json.loads(error_path.read_text())
        lines.append(f"Last recorded error: {error['type']}: {error['error']}")
    lines.append(f"Final report ready: {(root / 'REPORT.md').exists()}")
    return "\n".join(lines)


def run_study(config, outer_folds=None):
    validate_config(config)
    root, chosen, partial = study_directory(config, outer_folds)
    bank = Path(config["bank_dir"])
    manifest = json.loads((bank / "manifest.json").read_text())
    if manifest["classes"] != config["classes"] or not Path(config["folds_path"]).exists():
        raise ValueError("Study requires the unchanged cohort and frozen patient folds")
    waves, statistics = [np.load(bank / f"{name}.npy", mmap_mode="r") for name in ["waveforms", "statistics"]]
    labels, groups = [np.load(bank / f"{name}.npy") for name in ["y", "groups"]]
    if waves.shape != tuple(manifest["shape"]) or len(labels) != len(waves) or len(groups) != len(waves) or statistics.shape != (len(waves), manifest["statistics"]):
        raise ValueError("Incomplete waveform bank")
    mapping = {label: index for index, label in enumerate(config["classes"])}
    y = np.array([mapping[str(label)] for label in labels])
    folds = load_or_create_folds(y, groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"] + 1)):
        raise ValueError("Frozen patient fold IDs changed")
    runtime = {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(),
        "device": str(_device(config)), "deterministic_kernels": _device(config).type != "mps",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "neural_code_sha256": hashlib.sha256(Path(neural.__file__).read_bytes()).hexdigest()}
    scope = {"config": config, "runtime": runtime, "bank_fingerprint": manifest["fingerprint"],
        "folds_sha256": hashlib.sha256(Path(config["folds_path"]).read_bytes()).hexdigest(), "evaluated_folds": chosen}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root / "study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Contrastive settings/runtime changed; choose a new output_dir")
    _save_json(guard, scope)
    # Check EVERY inner training pool before launching an expensive run.
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    splits = {}
    for fold in chosen:
        train = np.flatnonzero(beat_folds != fold)
        splits[fold] = list(inner_patient_splits(y[train], groups[train], config["inner_splits"], config["split_seed"] + fold * 1000))
        for inner_train, _ in splits[fold]:
            StructuredPatientBatches(y, groups, train[inner_train], len(mapping), config["patients_per_class"], config["beats_per_patient"], 1, config["seed"])
    summary = []
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        output.mkdir(exist_ok=True)
        for arm in ARMS:
            probabilities, audits = np.full((len(y), len(mapping)), np.nan), []
            for fold in chosen:
                train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
                if set(groups[train]) & set(groups[test]):
                    raise RuntimeError("Outer patient leakage")
                checkpoint, audit_path = output / f"fold_{fold:02d}_{arm}.npz", output / f"fold_{fold:02d}_{arm}.json"
                if checkpoint.exists() and audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    with np.load(checkpoint) as saved:
                        if audit["study_fingerprint"] != fingerprint or not np.array_equal(saved["indices"], test):
                            raise ValueError("Contrastive checkpoint cohort/settings changed")
                        probabilities[test] = saved["probabilities"]
                    if not np.isfinite(probabilities[test]).all() or np.any(probabilities[test] < 0) or not np.allclose(probabilities[test].sum(1), 1, atol=1e-5):
                        raise ValueError("Invalid resumed probabilities")
                    audits.append(audit)
                    print(f"seed {seed} {arm} outer {fold}: resumed completed checkpoint", flush=True)
                    continue
                selections = []
                for inner, (itrain, ivalid) in enumerate(splits[fold], 1):
                    path = output / f"inner_{fold:02d}_{inner}_{arm}.json"
                    if path.exists():
                        selection = json.loads(path.read_text())
                        if selection["study_fingerprint"] != fingerprint:
                            raise ValueError("Inner contrastive settings changed")
                    else:
                        callback = lambda row, inner=inner: _progress(root, state="training", training_seed=seed,
                            arm=arm, outer_fold=fold, inner_fold=inner, **row)
                        model, normalizer, selection = fit_contrastive(arm, waves, statistics, y, groups,
                            train[itrain], train[ivalid], config["max_epochs"], config, seed + fold * 1000 + inner, callback)
                        selection.update(study_fingerprint=fingerprint, inner_fold=inner)
                        _save_json(path, selection)
                        del model, normalizer
                    selections.append(selection)
                    print(f"seed {seed} {arm} outer {fold}/{config['outer_splits']} inner {inner}/{config['inner_splits']}: epoch {selection['selected_epoch']}, macro F1={selection['best_validation_macro_f1']:.4f}", flush=True)
                epochs = int(np.clip(np.floor(np.median([row["selected_epoch"] for row in selections]) + 0.5), 1, config["max_epochs"]))
                callback = lambda row: _progress(root, state="training", training_seed=seed,
                    arm=arm, outer_fold=fold, inner_fold=None, **row)
                model, normalizer, final = fit_contrastive(arm, waves, statistics, y, groups, train,
                    None, epochs, config, seed + fold, callback)
                probabilities[test] = _predict_network(model, normalizer, waves, statistics, y, test, config)
                _, truth, scores = aggregate_patients(probabilities[test], y[test], groups[test])
                audit = {"study_fingerprint": fingerprint, "training_seed": seed, "arm": arm, "fold": fold,
                    "patient_overlap": 0, "train_patients": len(np.unique(groups[train])), "test_patients": len(np.unique(groups[test])),
                    "train_beats": len(train), "test_beats": len(test), "inner_selection": selections,
                    "final_epochs": epochs, "final_fit": final, "patient_metrics": metric_bundle(truth, scores, config["classes"])}
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez_compressed(temporary, indices=test, probabilities=probabilities[test])
                temporary.replace(checkpoint)
                _save_json(audit_path, audit)
                weight_path = output / f"model_{fold:02d}_{arm}.pt"
                weight_temporary = weight_path.with_suffix(".pt.partial")
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "normalization": normalizer.audit(), "study_fingerprint": fingerprint}, weight_temporary)
                weight_temporary.replace(weight_path)
                del model, normalizer
                audits.append(audit)
                _progress(root, state="outer_fold_complete", training_seed=seed, arm=arm, outer_fold=fold)
                print(f"seed {seed} {arm} outer {fold}/{config['outer_splits']}: patient macro F1={audit['patient_metrics']['macro_f1']:.4f}", flush=True)
            evaluated = np.isfinite(probabilities).all(1)
            if not partial and not evaluated.all():
                raise RuntimeError("Missing complete out-of-fold predictions")
            ids, truth, scores = aggregate_patients(probabilities[evaluated], y[evaluated], groups[evaluated])
            result = {"study_fingerprint": fingerprint, "training_seed": seed, "arm": arm,
                "classes": config["classes"], "complete": not partial, "evaluated_folds": chosen,
                "n_patients": len(ids), "n_beats": int(evaluated.sum()), "patient_metrics": metric_bundle(truth, scores, config["classes"]),
                "protocol": "Matched structured batches and initial weights; inner-patient stopping; all test beats mean-probability patient aggregation. Previously inspected PTB development folds.",
                "folds": audits}
            _save_json(output / f"metrics_{arm}.json", result)
            frame = pd.DataFrame({"patient_id": ids, "truth": [config["classes"][i] for i in truth],
                "prediction": [config["classes"][i] for i in scores.argmax(1)], "fold": [lookup[str(patient)] for patient in ids]})
            for i, label in enumerate(config["classes"]):
                frame[f"probability_{label}"] = scores[:, i]
            frame.to_csv(output / f"patient_predictions_{arm}.csv", index=False)
            save_confusion(truth, scores.argmax(1), config["classes"], f"{arm} seed {seed}", output)
            summary.append({"training_seed": seed, "arm": arm, "complete": not partial,
                "patient_macro_f1": result["patient_metrics"]["macro_f1"], "patient_accuracy": result["patient_metrics"]["accuracy"]})
            pd.DataFrame(summary).to_csv(root / "comparison.csv", index=False)
    _progress(root, state="partial_complete" if partial else "complete")
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/contrastive_vcg.yaml")
    parser.add_argument("--outer-folds", nargs="+", type=int)
    parser.add_argument("--status", action="store_true", help="Read saved progress without starting training")
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    if args.status:
        print(status(config, args.outer_folds))
    else:
        try:
            print(run_study(config, args.outer_folds).to_string(index=False))
        except BaseException as error:
            root, _, _ = study_directory(config, args.outer_folds)
            if root.exists():
                _save_json(root / "last_error.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise
