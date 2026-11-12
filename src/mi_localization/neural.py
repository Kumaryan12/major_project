"""Compact measured-VCG ResNet development with nested patient-only stopping.

PyTorch is optional: install the project's `neural` extra for this module.
No external ECG data, pretraining, augmentation or test-set early stopping.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .benchmark import (aggregate_patients, inner_patient_splits, load_or_create_folds,
                        metric_bundle, patient_balanced_weights, save_confusion)
from .data import read_vcg
from .signal import denoise_vcg


MODELS = ("resnet", "resnet_statistics")
SCHEMA = 1


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _save_json(path: Path, value) -> None:
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _waveform_rows(record, rows, paper):
    data, signal = paper["data"], paper["signal"]
    raw, fs = read_vcg(Path(data["raw_dir"]), record, list(data["leads"]))
    if fs != int(data["sampling_rate"]):
        raise ValueError("Unexpected VCG sampling rate")
    clean = denoise_vcg(raw, signal["denoise_wavelet"], int(signal["denoise_level"]))
    before, after = int(data["before_r"]), int(data["after_r"])
    values = []
    for peak in rows.r_peak.astype(int):
        if peak - before < 0 or peak + after >= len(clean):
            raise ValueError("Incomplete cached beat")
        values.append(clean[peak - before:peak + after + 1].T)
    return rows.index.to_numpy(), np.asarray(values, dtype=np.float32)


def build_waveform_bank(config: dict) -> Path:
    """Cache denoised XYZ at existing R positions, without fitting normalization."""
    with Path(config["paper_config"]).open() as stream:
        paper = yaml.safe_load(stream)
    data = paper["data"]
    if data["leads"] != ["vx", "vy", "vz"]:
        raise ValueError("Primary neural benchmark requires measured Frank XYZ")
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    if beats.empty:
        raise ValueError("Empty waveform cohort")
    relational = Path(config["statistics_bank_dir"])
    for name, expected in [("groups", beats.patient_id.astype(str)), ("y", beats.label.astype(str))]:
        if not np.array_equal(np.load(relational / f"{name}.npy"), expected.to_numpy()):
            raise ValueError("Waveform and statistics row order do not match")
    statistics_path = relational / "statistics_short.npy"
    statistics = np.load(statistics_path, mmap_mode="r")
    sources = []
    for record in beats.record.unique():
        for extension in ["hea", "xyz"]:
            source = Path(data["raw_dir"]) / f"{record}.{extension}"
            stat = source.stat()
            sources.append((str(source), stat.st_size, stat.st_mtime_ns))
    fingerprint = _hash({"schema": SCHEMA, "data": data, "signal": paper["signal"],
        "classes": config["classes"], "sources": sources,
        "beats_sha256": hashlib.sha256(Path(config["beats_path"]).read_bytes()).hexdigest(),
        "statistics_sha256": hashlib.sha256(statistics_path.read_bytes()).hexdigest()})
    destination = Path(config["bank_dir"])
    shape = (len(beats), 3, int(data["before_r"]) + int(data["after_r"]) + 1)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Waveform sources/settings changed; choose a new bank_dir")
        waveforms = np.load(destination / "waveforms.npy", mmap_mode="r")
        if waveforms.shape != shape or not np.isfinite(waveforms).all():
            raise ValueError("Waveform bank is incomplete or nonfinite")
        for name in ["statistics", "y", "groups"]:
            if np.load(destination / f"{name}.npy", mmap_mode="r").shape[0] != len(beats):
                raise ValueError("Waveform bank is incomplete")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "waveforms.partial.npy"
    waveforms = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    jobs = Parallel(n_jobs=int(config.get("feature_n_jobs", 2)), return_as="generator_unordered")(
        delayed(_waveform_rows)(record, rows, paper) for record, rows in beats.groupby("record", sort=False))
    for completed, (indices, values) in enumerate(jobs, 1):
        waveforms[indices] = values
        if completed % 25 == 0 or completed == beats.record.nunique():
            print(f"Waveform bank: {completed}/{beats.record.nunique()} records", flush=True)
    waveforms.flush()
    if not np.isfinite(waveforms).all() or statistics.ndim != 2 or not np.isfinite(statistics).all():
        raise ValueError("Nonfinite waveform/statistics cache")
    del waveforms
    temporary.replace(destination / "waveforms.npy")
    np.save(destination / "statistics.npy", statistics)
    np.save(destination / "y.npy", beats.label.astype(str).to_numpy(dtype="U"))
    np.save(destination / "groups.npy", beats.patient_id.astype(str).to_numpy(dtype="U"))
    _save_json(manifest_path, {"schema": SCHEMA, "fingerprint": fingerprint,
        "n_beats": len(beats), "n_patients": int(beats.patient_id.nunique()),
        "n_records": int(beats.record.nunique()), "shape": shape, "statistics": int(statistics.shape[1]),
        "input": "measured Frank XYZ only; denoised complete recordings, original cached R positions",
        "processing": "offline/noncausal; no fitted dataset normalization in feature cache",
        "classes": config["classes"]})
    return destination


class FoldNormalizer:
    """Patient/class-weighted training-set scaling; never normalize each beat."""

    def fit(self, waves, statistics, indices, weights):
        indices, weights = np.asarray(indices), np.asarray(weights, dtype=float)
        if len(indices) == 0 or weights.shape != indices.shape or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Invalid training rows/weights")
        weights = weights / weights.sum()
        mean, second = np.zeros(3), np.zeros(3)
        sm, ss = np.zeros(statistics.shape[1]), np.zeros(statistics.shape[1])
        for start in range(0, len(indices), 256):
            rows, weight = indices[start:start + 256], weights[start:start + 256]
            x, s = np.asarray(waves[rows], dtype=np.float64), np.asarray(statistics[rows], dtype=np.float64)
            if x.ndim != 3 or x.shape[1] != 3 or not np.isfinite(x).all() or not np.isfinite(s).all():
                raise ValueError("Expected finite XYZ waveforms/statistics")
            mean += weight @ x.mean(axis=2)
            second += weight @ np.square(x).mean(axis=2)
            sm += weight @ s
            ss += weight @ np.square(s)
        self.wave_mean_ = mean.astype(np.float32)
        self.wave_scale_ = np.sqrt(np.maximum(second - mean ** 2, 1e-12)).astype(np.float32)
        self.stat_mean_ = sm.astype(np.float32)
        self.stat_scale_ = np.sqrt(np.maximum(ss - sm ** 2, 1e-12)).astype(np.float32)
        self.n_fit_samples_ = len(indices)
        return self

    def transform(self, waves, statistics):
        x = (np.asarray(waves, dtype=np.float32) - self.wave_mean_[:, None]) / self.wave_scale_[:, None]
        s = np.clip((np.asarray(statistics, dtype=np.float32) - self.stat_mean_) / self.stat_scale_, -10, 10)
        return x, s

    def audit(self):
        return {name: getattr(self, name).tolist() for name in ["wave_mean_", "wave_scale_", "stat_mean_", "stat_scale_"]}


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(incoming, outgoing, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(4, outgoing), nn.GELU(), nn.Conv1d(outgoing, outgoing, 7, padding=3, bias=False),
            nn.GroupNorm(4, outgoing))
        self.skip = nn.Conv1d(incoming, outgoing, 1, stride=2, bias=False)

    def forward(self, x):
        return torch.nn.functional.gelu(self.main(x) + self.skip(x))


class TemporalBinPool(nn.Module):
    """Adaptive-average-pool semantics, including non-divisible MPS lengths."""
    def __init__(self, bins):
        super().__init__()
        self.bins = bins

    def forward(self, x):
        length = x.shape[-1]
        return torch.stack([x[..., (index * length) // self.bins:
            ((index + 1) * length + self.bins - 1) // self.bins].mean(dim=-1)
            for index in range(self.bins)], dim=-1)


class VCGResNet(nn.Module):
    """Small residual encoder with coarse time bins and optional statistics."""
    def __init__(self, n_classes=6, widths=(16, 32, 64), pool_bins=4, dropout=0.2, statistics=0):
        super().__init__()
        if len(widths) != 3 or any(width < 4 or width % 4 for width in widths) or pool_bins < 1:
            raise ValueError("Three widths divisible by four and positive pool_bins are required")
        self.statistics = statistics
        self.encoder = nn.Sequential(nn.Conv1d(3, widths[0], 15, stride=2, padding=7, bias=False),
            nn.GroupNorm(4, widths[0]), nn.GELU(), ResidualBlock(widths[0], widths[0]),
            ResidualBlock(widths[0], widths[1]), ResidualBlock(widths[1], widths[2]),
            TemporalBinPool(pool_bins), nn.Flatten(), nn.Linear(widths[2] * pool_bins, 64), nn.GELU())
        self.stat_encoder = nn.Sequential(nn.Linear(statistics, 32), nn.GELU()) if statistics else None
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(96 if statistics else 64, n_classes))

    def forward(self, waveform, statistics=None):
        representation = self.encoder(waveform)
        if self.stat_encoder is not None:
            if statistics is None:
                raise ValueError("Fusion model requires statistics")
            representation = torch.cat([representation, self.stat_encoder(statistics)], dim=1)
        return self.classifier(representation)


class BeatDataset(Dataset):
    def __init__(self, waves, statistics, labels, indices, normalizer):
        self.waves, self.statistics, self.labels = waves, statistics, labels
        self.indices, self.normalizer = np.asarray(indices), normalizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        x, s = self.normalizer.transform(self.waves[row], self.statistics[row])
        return torch.from_numpy(x.copy()), torch.from_numpy(s.copy()), int(self.labels[row])


def _device(config):
    requested = config.get("device", "cpu")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto/cpu/mps/cuda")
    return torch.device(requested)


def _seed(seed, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    # Enforce deterministic kernels on CPU/CUDA. MPS is seeded but its full
    # operation set does not provide this guarantee; record that limitation.
    torch.use_deterministic_algorithms(_device(config).type != "mps")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False


def _predict_network(model, normalizer, waves, statistics, y, indices, config):
    loader = DataLoader(BeatDataset(waves, statistics, y, indices, normalizer),
                        batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    device = _device(config)
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, s, _ in loader:
            outputs.append(model(x.to(device), s.to(device)).softmax(dim=1).cpu().numpy())
    probabilities = np.concatenate(outputs)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise RuntimeError("Network generated invalid probabilities")
    return probabilities


def _fit_network(name, waves, statistics, y, groups, train, valid, epochs, config, seed):
    if valid is not None and set(groups[train]) & set(groups[valid]):
        raise ValueError("Training and early-stopping patients overlap")
    _seed(seed, config)
    weights = patient_balanced_weights(y[train], groups[train])
    normalizer = FoldNormalizer().fit(waves, statistics, train, weights)
    model = VCGResNet(n_classes=len(config["classes"]), widths=tuple(config["widths"]),
        pool_bins=int(config["pool_bins"]), dropout=float(config["dropout"]),
        statistics=statistics.shape[1] if name == "resnet_statistics" else 0).to(_device(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(config["samples_per_epoch"]), replacement=True, generator=generator)
    loader = DataLoader(BeatDataset(waves, statistics, y, train, normalizer),
        batch_size=int(config["batch_size"]), sampler=sampler, num_workers=0)
    best_score, best_epoch, best_state = -np.inf, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, s, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x.to(_device(config)), s.to(_device(config))), labels.to(_device(config)))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite neural training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "training_loss": float(np.mean(losses))}
        if valid is not None:
            probabilities = _predict_network(model, normalizer, waves, statistics, y, valid, config)
            _, truth, patient_probabilities = aggregate_patients(probabilities, y[valid], groups[valid])
            score = float(f1_score(truth, patient_probabilities.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
            row["validation_patient_macro_f1"] = score
            if score > best_score + 1e-8:
                best_score, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
        if config.get("verbose_epochs", False):
            print(f"{name} seed {seed} epoch {epoch}/{epochs}: {row}", flush=True)
        if valid is not None and epoch - best_epoch >= int(config["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    audit = {"seed": seed, "epochs_run": len(history), "selected_epoch": best_epoch if valid is not None else epochs,
        "best_validation_macro_f1": best_score if valid is not None else None, "history": history,
        "train_patients": len(np.unique(groups[train])), "validation_patients": 0 if valid is None else len(np.unique(groups[valid])),
        "normalizer_fit_beats": normalizer.n_fit_samples_, "parameter_count": sum(p.numel() for p in model.parameters()),
        "samples_per_epoch": int(config["samples_per_epoch"]), "normalization": normalizer.audit()}
    return model, normalizer, audit


def _validate_config(config):
    if not config["models"] or set(config["models"]) - set(MODELS) or len(set(config["models"])) != len(config["models"]):
        raise ValueError("Choose unique resnet/resnet_statistics models")
    for name in ["max_epochs", "patience", "samples_per_epoch", "batch_size", "torch_threads"]:
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if int(config["inner_splits"]) < 2 or float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("Invalid inner splitting or optimizer settings")


def run_neural_study(config: dict, outer_folds: list[int] | None = None) -> pd.DataFrame:
    _validate_config(config)
    bank = Path(config["bank_dir"])
    manifest = json.loads((bank / "manifest.json").read_text())
    if manifest["classes"] != config["classes"]:
        raise ValueError("Neural bank classes/order changed")
    waves, statistics = [np.load(bank / f"{name}.npy", mmap_mode="r") for name in ["waveforms", "statistics"]]
    labels, groups = [np.load(bank / f"{name}.npy") for name in ["y", "groups"]]
    if waves.shape != tuple(manifest["shape"]) or len(groups) != len(labels) or len(groups) != len(waves) or statistics.shape != (len(waves), manifest["statistics"]):
        raise ValueError("Incomplete neural bank")
    mapping = {name: i for i, name in enumerate(config["classes"])}
    y = np.array([mapping[str(label)] for label in labels])
    # Do not silently create new folds for this follow-on development study.
    if not Path(config["folds_path"]).exists():
        raise ValueError("Neural study requires the existing frozen patient folds")
    folds = load_or_create_folds(y, groups, config)
    all_folds = list(range(1, int(config["outer_splits"]) + 1))
    chosen = all_folds if outer_folds is None else sorted(set(outer_folds))
    if sorted(folds.fold.unique()) != all_folds or not chosen or set(chosen) - set(all_folds):
        raise ValueError("Invalid frozen or requested fold IDs")
    partial = chosen != all_folds
    output = Path(config["output_dir"]) / ("partial_folds_" + "_".join(map(str, chosen)) if partial else "complete")
    output.mkdir(parents=True, exist_ok=True)
    runtime = {"torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "device": str(_device(config)), "platform": platform.platform(), "torch_threads": int(config["torch_threads"]),
        "deterministic_kernels": _device(config).type != "mps",
        "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    scope = {"schema": SCHEMA, "config": config, "bank_fingerprint": manifest["fingerprint"],
        "folds_sha256": hashlib.sha256(Path(config["folds_path"]).read_bytes()).hexdigest(), "runtime": runtime,
        "evaluated_folds": chosen}
    study_hash = _hash(scope)
    scope["study_fingerprint"] = study_hash
    scope_path = output / "study_manifest.json"
    if scope_path.exists() and json.loads(scope_path.read_text()) != scope:
        raise ValueError("Neural study settings/runtime changed; choose a new output_dir")
    _save_json(scope_path, scope)
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(group)] for group in groups])
    summary = []
    for name in config["models"]:
        probabilities = np.full((len(y), len(mapping)), np.nan)
        audits = []
        for fold in chosen:
            train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            if set(groups[train]) & set(groups[test]):
                raise RuntimeError("Outer patient leakage")
            checkpoint, audit_path = output / f"fold_{fold:02d}_{name}.npz", output / f"fold_{fold:02d}_{name}.json"
            if checkpoint.exists() and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                if audit["study_fingerprint"] != study_hash:
                    raise ValueError("Neural checkpoint settings changed")
                with np.load(checkpoint) as saved:
                    if not np.array_equal(saved["indices"], test):
                        raise ValueError("Neural checkpoint test rows changed")
                    probabilities[test] = saved["probabilities"]
                if not np.isfinite(probabilities[test]).all() or not np.allclose(probabilities[test].sum(1), 1, atol=1e-5):
                    raise ValueError("Invalid resumed neural probabilities")
                audits.append(audit)
                print(f"{name} fold {fold}: resumed completed checkpoint", flush=True)
                continue
            selections = []
            splits = inner_patient_splits(y[train], groups[train], int(config["inner_splits"]), int(config["seed"]) + fold * 1000)
            for inner, (itrain, ivalid) in enumerate(splits, 1):
                ti, vi = train[itrain], train[ivalid]
                selection_path = output / f"inner_{fold:02d}_{inner}_{name}.json"
                if selection_path.exists():
                    selected = json.loads(selection_path.read_text())
                    if selected["study_fingerprint"] != study_hash:
                        raise ValueError("Inner selection settings changed")
                else:
                    model, normalizer, selected = _fit_network(name, waves, statistics, y, groups, ti, vi,
                        int(config["max_epochs"]), config, int(config["seed"]) + fold * 1000 + inner)
                    selected.update(study_fingerprint=study_hash, inner_fold=inner, patient_overlap=0)
                    _save_json(selection_path, selected)
                    del model, normalizer
                selections.append(selected)
                print(f"{name} outer {fold}/{config['outer_splits']} inner {inner}/{config['inner_splits']}: selected epoch {selected['selected_epoch']}, patient macro F1={selected['best_validation_macro_f1']:.4f}", flush=True)
            epochs = int(np.clip(np.floor(np.median([row["selected_epoch"] for row in selections]) + 0.5), 1, int(config["max_epochs"])))
            model, normalizer, final_fit = _fit_network(name, waves, statistics, y, groups, train, None,
                epochs, config, int(config["seed"]) + fold)
            probabilities[test] = _predict_network(model, normalizer, waves, statistics, y, test, config)
            _, truth, patient_scores = aggregate_patients(probabilities[test], y[test], groups[test])
            audit = {"study_fingerprint": study_hash, "model": name, "fold": fold,
                "patient_overlap": 0, "train_patients": len(np.unique(groups[train])), "test_patients": len(np.unique(groups[test])),
                "train_beats": len(train), "test_beats": len(test), "inner_selection": selections,
                "final_epochs": epochs, "epoch_selection": "rounded median of inner-validation best patient-macro-F1 epochs",
                "final_fit": final_fit, "patient_metrics": metric_bundle(truth, patient_scores, config["classes"])}
            temporary = checkpoint.with_suffix(".partial.npz")
            np.savez_compressed(temporary, indices=test, probabilities=probabilities[test])
            temporary.replace(checkpoint)
            _save_json(audit_path, audit)
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "normalization": normalizer.audit(), "study_fingerprint": study_hash}, output / f"model_{fold:02d}_{name}.pt")
            del model, normalizer
            audits.append(audit)
            print(f"{name} fold {fold}/{config['outer_splits']}: patient macro F1={audit['patient_metrics']['macro_f1']:.4f}", flush=True)
        evaluated = np.isfinite(probabilities).all(axis=1)
        if not partial and not evaluated.all():
            raise RuntimeError("Missing neural out-of-fold predictions")
        ids, truth, patient_scores = aggregate_patients(probabilities[evaluated], y[evaluated], groups[evaluated])
        result = {"model": name, "classes": config["classes"], "study_fingerprint": study_hash,
            "complete": not partial, "evaluated_folds": chosen, "n_patients": len(ids), "n_beats": int(evaluated.sum()),
            "protocol": "Previously inspected PTB development folds; fixed architecture, nested patient early stopping, final refit with no outer test monitoring",
            "weighting": "class/patient-balanced sampling with replacement; unweighted cross entropy (no double weighting)",
            "aggregation": "mean beat probabilities per patient; all validation/test beats evaluated",
            "runtime": runtime, "patient_metrics": metric_bundle(truth, patient_scores, config["classes"]), "folds": audits}
        _save_json(output / f"metrics_{name}.json", result)
        frame = pd.DataFrame({"patient_id": ids, "truth": [config["classes"][i] for i in truth],
            "prediction": [config["classes"][i] for i in patient_scores.argmax(1)], "fold": [lookup[str(pid)] for pid in ids]})
        for i, label in enumerate(config["classes"]):
            frame[f"probability_{label}"] = patient_scores[:, i]
        frame.to_csv(output / f"patient_predictions_{name}.csv", index=False)
        save_confusion(truth, patient_scores.argmax(1), config["classes"], name, output)
        summary.append({"model": name, "complete": not partial, "n_patients": len(ids),
            "patient_macro_f1": result["patient_metrics"]["macro_f1"], "patient_accuracy": result["patient_metrics"]["accuracy"]})
        pd.DataFrame(summary).to_csv(output / "comparison.csv", index=False)
    return pd.DataFrame(summary)
