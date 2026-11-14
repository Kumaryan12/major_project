"""Separate measured ECG + VCG extension with matched tensor-fusion controls."""
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from . import neural, paired_bank
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights, save_confusion
from .contrastive import initial_state_hash, _progress
from .neural import VCGResNet, _device, _hash, _save_json, _seed
from .preprocessing_study import directory
from .signal_audit import file_hash

ARMS = ("vcg", "ecg", "vcg_twin", "joint_additive", "joint_bilinear")
DUAL = ARMS[2:]
ECG_INPUT = ("ecg", "joint_additive", "joint_bilinear")
CONTRASTS = (("vcg", "joint_bilinear"), ("vcg_twin", "joint_bilinear"), ("joint_additive", "joint_bilinear"), ("ecg", "joint_bilinear"))
CLAIM_SCOPE = "Separate measured ECG extension; repeatedly inspected PTB patient folds are development data, not independent confirmation or SOTA."


class LeadNormalizer:
    """Patient/class weighted TRAINING moments for either 3 or 12 leads."""
    def fit(self, waves, train, weights):
        weights = np.asarray(weights, dtype=float)
        train = np.asarray(train)
        if len(train) == 0 or weights.shape != train.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid training-only normalization rows/weights")
        weights = weights / weights.sum()
        mean, second = np.zeros(waves.shape[1]), np.zeros(waves.shape[1])
        for start in range(0, len(train), 256):
            x = np.asarray(waves[train[start:start+256]], dtype=np.float64)
            weight = weights[start:start+256]
            if x.ndim != 3 or not np.isfinite(x).all():
                raise ValueError("Invalid training lead waveforms")
            mean += weight @ x.mean(2)
            second += weight @ np.square(x).mean(2)
        self.wave_mean_ = mean.astype(np.float32)
        self.wave_scale_ = np.sqrt(np.maximum(second - mean**2, 1e-12)).astype(np.float32)
        return self

    def transform(self, wave):
        return (np.asarray(wave, dtype=np.float32) - self.wave_mean_[:, None]) / self.wave_scale_[:, None]

    def audit(self):
        return {"wave_mean_": self.wave_mean_.tolist(), "wave_scale_": self.wave_scale_.tolist()}


class PairedNet(nn.Module):
    def __init__(self, config, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown paired input arm")
        self.arm = arm
        base = VCGResNet(len(config["classes"]), tuple(config["widths"]), config["pool_bins"], config["dropout"])
        self.encoder, self.classifier = base.encoder, base.classifier
        if arm == "ecg":
            self.encoder[0] = nn.Conv1d(12, config["widths"][0], 15, stride=2, padding=7, bias=False)
        if arm in DUAL:
            second = VCGResNet(len(config["classes"]), tuple(config["widths"]), config["pool_bins"], config["dropout"])
            second.encoder[0] = nn.Conv1d(12, config["widths"][0], 15, stride=2, padding=7, bias=False)
            self.secondary_encoder = second.encoder
            self.factor_vcg = nn.Linear(64, config["fusion_rank"], bias=False)
            self.factor_ecg = nn.Linear(64, config["fusion_rank"], bias=False)
            self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(128 + config["fusion_rank"], len(config["classes"])))

    def interaction(self, a, b):
        left, right = self.factor_vcg(a), self.factor_ecg(b)
        return torch.tanh((left + right) / np.sqrt(2)) if self.arm == "joint_additive" else torch.tanh(left * right)

    def forward(self, vcg, ecg):
        if self.arm == "ecg":
            return self.classifier(self.encoder(ecg))
        a = self.encoder(vcg)
        if self.arm == "vcg":
            return self.classifier(a)
        # Tiling XYZ changes no information. This control has exactly the same
        # trainable tensors/input dimensions as the measured-ECG fusion model.
        second = vcg.repeat(1, 4, 1) if self.arm == "vcg_twin" else ecg
        b = self.secondary_encoder(second)
        return self.classifier(torch.cat([a, b, self.interaction(a, b)], 1))


def initial_model(config, seed, arm):
    _seed(seed, config)
    return PairedNet(config, arm)


class PairedRows(Dataset):
    def __init__(self, vcg, ecg, y, indices, normalizers):
        self.vcg, self.ecg, self.y, self.indices, self.normalizers = vcg, ecg, y, np.asarray(indices), normalizers

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        streams = []
        for key, values in [("vcg", self.vcg), ("ecg", self.ecg)]:
            norm = self.normalizers[key]
            # Do not even index ECG samples in VCG-only arms (or vice versa).
            x = np.empty((0, 0), dtype=np.float32) if norm is None else norm.transform(values[row])
            streams.append(torch.from_numpy(x.copy()))
        return *streams, int(self.y[row]), int(row)


def predict(model, normalizers, vcg, ecg, y, indices, config):
    model.eval()
    loader = DataLoader(PairedRows(vcg, ecg, y, indices, normalizers), batch_size=config["batch_size"], shuffle=False, num_workers=0)
    output = []
    with torch.no_grad():
        for x, e, _, _ in loader:
            output.append(model(x.to(_device(config)), e.to(_device(config))).softmax(1).cpu().numpy())
    probabilities = np.concatenate(output)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid paired model probabilities")
    return probabilities


def fit(vcg, ecg, y, groups, train, valid, epochs, config, seed, arm, progress=None):
    if valid is not None and set(groups[train]) & set(groups[valid]):
        raise ValueError("Paired training/validation patients overlap")
    weights = patient_balanced_weights(y[train], groups[train])
    normalizers = {"vcg": None if arm == "ecg" else LeadNormalizer().fit(vcg, train, weights),
        "ecg": LeadNormalizer().fit(ecg, train, weights) if arm in ECG_INPUT else None}
    model = initial_model(config, seed, arm).to(_device(config))
    initial = {"initial_state_sha256": initial_state_hash(model), "initial_classifier_sha256": initial_state_hash(model.classifier),
        "initial_primary_encoder_sha256": initial_state_hash(model.encoder)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), config["samples_per_epoch"], replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(PairedRows(vcg, ecg, y, train, normalizers), batch_size=config["batch_size"], sampler=sampler, num_workers=0)
    best, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, epochs+1):
        model.train()
        losses, count, digest = [], 0, hashlib.sha256()
        for x, e, labels, rows in loader:
            digest.update(rows.numpy().astype("<i8").tobytes())
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(_device(config)), e.to(_device(config))), labels.to(_device(config)))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite paired training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            count += len(labels)
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)), "sampled_rows_sha256": digest.hexdigest(), "beat_presentations": count}
        if valid is not None:
            probabilities = predict(model, normalizers, vcg, ecg, y, valid, config)
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
    return model, normalizers, dict(initial, seed=seed, arm=arm, patient_overlap=0, parameter_count=sum(p.numel() for p in model.parameters()),
        selected_epoch=best_epoch if valid is not None else epochs, best_validation_macro_f1=best if valid is not None else None, history=history,
        train_patients=len(np.unique(groups[train])), validation_patients=0 if valid is None else len(np.unique(groups[valid])), normalizer_fit_beats=len(train),
        normalization={key: None if norm is None else norm.audit() for key, norm in normalizers.items()})


def validate_config(config):
    if config["arms"] != list(ARMS) or not config["training_seeds"] or len(set(config["training_seeds"])) != len(config["training_seeds"]):
        raise ValueError("Require all five declared paired arms and unique target seeds")
    for key in ["fusion_rank", "outer_splits", "inner_splits", "batch_size", "samples_per_epoch", "max_epochs", "patience", "torch_threads", "conversion_workers"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < (2 if key in {"outer_splits", "inner_splits"} else 1):
            raise ValueError("Invalid positive paired study budget")
    if not np.isfinite([config["learning_rate"], config["weight_decay"], config["dropout"]]).all() or config["learning_rate"] <= 0 or config["weight_decay"] < 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("Invalid paired optimization settings")


def validate_fit(audit, config, seed, arm, train, valid, groups, final_epochs=None):
    expected = initial_model(config, seed, arm)
    if audit["seed"] != seed or audit["arm"] != arm or audit["patient_overlap"] or audit["normalizer_fit_beats"] != len(train) or audit["train_patients"] != len(np.unique(groups[train])) or audit["validation_patients"] != (0 if valid is None else len(np.unique(groups[valid]))) or audit["parameter_count"] != sum(p.numel() for p in expected.parameters()):
        raise ValueError("Paired fit scope/membership mismatch")
    for key, module in [("initial_state_sha256", expected), ("initial_classifier_sha256", expected.classifier), ("initial_primary_encoder_sha256", expected.encoder)]:
        if audit[key] != initial_state_hash(module):
            raise ValueError("Paired fit initialization mismatch")
    for key, leads, used in [("vcg", 3, arm != "ecg"), ("ecg", 12, arm in ECG_INPUT)]:
        norm = audit["normalization"][key]
        if not used:
            if norm is not None:
                raise ValueError("Unused input must not fit a normalizer")
        elif norm is None or len(norm["wave_mean_"]) != leads or len(norm["wave_scale_"]) != leads or not np.isfinite(norm["wave_mean_"] + norm["wave_scale_"]).all() or min(norm["wave_scale_"]) <= 0:
            raise ValueError("Invalid training-only lead normalizer")
    history = audit["history"]
    if not history or len(history) > config["max_epochs"]:
        raise ValueError("Missing/invalid paired training history")
    for epoch, row in enumerate(history, 1):
        if row["epoch"] != epoch or row["beat_presentations"] != config["samples_per_epoch"] or not np.isfinite(row["training_loss"]) or len(row["sampled_rows_sha256"]) != 64:
            raise ValueError("Invalid paired training budget/history")
    if final_epochs is not None:
        if len(history) != final_epochs or audit["selected_epoch"] != final_epochs or audit["best_validation_macro_f1"] is not None or valid is not None or any("validation_patient_macro_f1" in row for row in history):
            raise ValueError("Outer refit must not monitor test/validation patients")
    else:
        values = [row.get("validation_patient_macro_f1", np.nan) for row in history]
        if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
            raise ValueError("Invalid inner patient scores")
        best, selected = -np.inf, 0
        for epoch, value in enumerate(values, 1):
            if value > best + 1e-8:
                best, selected = value, epoch
        if audit["selected_epoch"] != selected or audit["best_validation_macro_f1"] != best or (len(history) < config["max_epochs"] and len(history) - selected != config["patience"]):
            raise ValueError("Paired inner stopping mismatch")


def check_pair(first, second):
    for key in ["seed", "normalizer_fit_beats", "train_patients", "validation_patients"]:
        if first[key] != second[key]:
            raise ValueError("Paired fits changed patient/sample membership")
    for key in ["vcg", "ecg"]:
        a, b = first["normalization"][key], second["normalization"][key]
        if a is not None and b is not None and a != b:
            raise ValueError("Shared-input training normalization differs")
    if first["arm"] in DUAL and second["arm"] in DUAL:
        for key in ["initial_state_sha256", "initial_classifier_sha256", "parameter_count"]:
            if first[key] != second[key]:
                raise ValueError("Capacity-matched dual model initialization differs")
    for a, b in zip(first["history"], second["history"]):
        if a["sampled_rows_sha256"] != b["sampled_rows_sha256"]:
            raise ValueError("Paired fits received different sampled rows")


def run_study(config, outer_folds=None):
    validate_config(config)
    vcg, ecg, y, groups, bank = paired_bank.load_paired(config)
    root, chosen, partial = directory(config, outer_folds)
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require existing frozen target patient folds")
    folds = load_or_create_folds(y, groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"]+1)):
        raise ValueError("Patient fold IDs changed")
    scope = {"config": config, "bank": bank, "evaluated_folds": chosen, "folds_sha256": file_hash(config["folds_path"]), "claim_scope": CLAIM_SCOPE,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "device": str(_device(config)),
            "deterministic_kernels": _device(config).type != "mps", "code_sha256": file_hash(__file__), "neural_code_sha256": file_hash(neural.__file__)}}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root / "study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Paired study settings/runtime/data changed; use a new output_dir")
    _save_json(guard, scope)
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    summary = []
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        output.mkdir(exist_ok=True)
        probabilities = {arm: np.full((len(y), len(config["classes"])), np.nan) for arm in ARMS}
        audits = {arm: [] for arm in ARMS}
        for fold in chosen:
            train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            if set(groups[train]) & set(groups[test]):
                raise ValueError("Paired outer patient leakage")
            splits = [(train[ti], train[vi]) for ti, vi in inner_patient_splits(y[train], groups[train], config["inner_splits"], config["split_seed"] + fold*1000)]
            for arm in ARMS:
                checkpoint, audit_path = output / f"fold_{fold:02d}_{arm}.npz", output / f"fold_{fold:02d}_{arm}.json"
                if checkpoint.exists() and audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    if audit["study_fingerprint"] != fingerprint or audit["prediction_checkpoint_sha256"] != file_hash(checkpoint):
                        raise ValueError("Paired completed checkpoint changed")
                    with np.load(checkpoint) as saved:
                        if not np.array_equal(saved["indices"], test):
                            raise ValueError("Paired checkpoint membership changed")
                        probabilities[arm][test] = saved["probabilities"]
                    audits[arm].append(audit)
                    continue
                selections = []
                for inner, (ti, vi) in enumerate(splits, 1):
                    path = output / f"inner_{fold:02d}_{arm}_{inner}.json"
                    if path.exists():
                        fit_audit = json.loads(path.read_text())
                        if fit_audit["study_fingerprint"] != fingerprint or fit_audit["inner_fold"] != inner:
                            raise ValueError("Paired inner checkpoint changed")
                    else:
                        callback = lambda row, inner=inner: _progress(root, state="training", training_seed=seed, arm=arm, outer_fold=fold, inner_fold=inner, **row)
                        model, norms, fit_audit = fit(vcg, ecg, y, groups, ti, vi, config["max_epochs"], config, seed + fold*1000 + inner, arm, callback)
                        fit_audit.update(study_fingerprint=fingerprint, inner_fold=inner)
                        del model, norms
                    validate_fit(fit_audit, config, seed + fold*1000 + inner, arm, ti, vi, groups)
                    _save_json(path, fit_audit)
                    selections.append(fit_audit)
                    print(f"seed {seed} {arm} outer {fold} inner {inner}: best epoch {fit_audit['selected_epoch']}, F1 {fit_audit['best_validation_macro_f1']:.4f}", flush=True)
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in selections]) + .5))
                callback = lambda row: _progress(root, state="training", training_seed=seed, arm=arm, outer_fold=fold, inner_fold=None, **row)
                model, norms, final = fit(vcg, ecg, y, groups, train, None, epochs, config, seed + fold, arm, callback)
                validate_fit(final, config, seed + fold, arm, train, None, groups, epochs)
                probabilities[arm][test] = predict(model, norms, vcg, ecg, y, test, config)
                _, truth, scores = aggregate_patients(probabilities[arm][test], y[test], groups[test])
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez_compressed(temporary, indices=test, probabilities=probabilities[arm][test])
                temporary.replace(checkpoint)
                audit = {"study_fingerprint": fingerprint, "arm": arm, "training_seed": seed, "fold": fold, "patient_overlap": 0,
                    "train_beats": len(train), "test_beats": len(test), "train_patients": len(np.unique(groups[train])), "test_patients": len(np.unique(groups[test])),
                    "inner_selection": selections, "final_epochs": epochs, "final_fit": final, "prediction_checkpoint_sha256": file_hash(checkpoint),
                    "patient_metrics": metric_bundle(truth, scores, config["classes"])}
                _save_json(audit_path, audit)
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "normalization": final["normalization"], "arm": arm, "study_fingerprint": fingerprint}, output / f"model_{fold:02d}_{arm}.pt")
                audits[arm].append(audit)
                del model, norms
                _progress(root, state="outer_fold_complete", training_seed=seed, arm=arm, outer_fold=fold)
                print(f"seed {seed} {arm} outer {fold}/{config['outer_splits']} complete: F1 {audit['patient_metrics']['macro_f1']:.4f}", flush=True)
            for arm in ARMS[1:]:
                for a, b in zip(audits["vcg"][-1]["inner_selection"], audits[arm][-1]["inner_selection"]):
                    check_pair(a, b)
                check_pair(audits["vcg"][-1]["final_fit"], audits[arm][-1]["final_fit"])
            for arm in ["joint_additive", "joint_bilinear"]:
                for a, b in zip(audits["vcg_twin"][-1]["inner_selection"], audits[arm][-1]["inner_selection"]):
                    check_pair(a, b)
                check_pair(audits["vcg_twin"][-1]["final_fit"], audits[arm][-1]["final_fit"])
        for arm in ARMS:
            mask = np.isfinite(probabilities[arm]).all(1)
            if not partial and not mask.all():
                raise ValueError("Incomplete paired study predictions")
            p = probabilities[arm][mask]
            if np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
                raise ValueError("Invalid paired study probabilities")
            ids, truth, scores = aggregate_patients(p, y[mask], groups[mask])
            saved = {"study_fingerprint": fingerprint, "arm": arm, "training_seed": seed, "classes": config["classes"], "complete": not partial,
                "evaluated_folds": chosen, "n_patients": len(ids), "n_beats": int(mask.sum()), "patient_metrics": metric_bundle(truth, scores, config["classes"]), "folds": audits[arm], "claim_scope": CLAIM_SCOPE}
            _save_json(output / f"metrics_{arm}.json", saved)
            frame = pd.DataFrame({"patient_id": ids, "truth": [config["classes"][i] for i in truth], "prediction": [config["classes"][i] for i in scores.argmax(1)], "fold": [lookup[str(patient)] for patient in ids]})
            for i, label in enumerate(config["classes"]):
                frame[f"probability_{label}"] = scores[:, i]
            frame.to_csv(output / f"patient_predictions_{arm}.csv", index=False)
            save_confusion(truth, scores.argmax(1), config["classes"], arm, output)
            summary.append({"training_seed": seed, "arm": arm, "complete": not partial, "patient_macro_f1": saved["patient_metrics"]["macro_f1"]})
            pd.DataFrame(summary).to_csv(root / "comparison.csv", index=False)
    _progress(root, state="partial_complete" if partial else "complete")
    return pd.DataFrame(summary)


def status(config):
    root, _, _ = directory(config)
    lines = [f"Separate paired ECG + VCG study: {root}"]
    for seed in config["training_seeds"]:
        for arm in ARMS:
            lines.append(f"seed {seed} {arm}: {len(list((root / f'seed_{seed}').glob(f'fold_*_{arm}.json')))}/{config['outer_splits']} outer folds")
    for name in ["progress.json", "last_error.json"]:
        if (root / name).exists():
            lines.append(f"Last saved {name} (not a live process check): " + (root / name).read_text())
    lines.append(f"Validated final report: {(root / 'REPORT.md').exists()}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paired_ecg_vcg.yaml")
    parser.add_argument("--outer-folds", nargs="+", type=int)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        print(status(config))
    else:
        try:
            print(run_study(config, args.outer_folds).to_string(index=False))
        except BaseException as error:
            root, _, _ = directory(config, args.outer_folds)
            root.mkdir(parents=True, exist_ok=True)
            _save_json(root / "last_error.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise
