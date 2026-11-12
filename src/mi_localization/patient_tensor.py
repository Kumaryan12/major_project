"""Patient-level MIL and supervised multi-scale tensor contraction for VCG."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from . import neural
from .benchmark import inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights, patient_table, save_confusion
from .neural import FoldNormalizer, TemporalBinPool, VCGResNet, _device, _hash, _save_json, _seed


MODELS = ("bag_mean", "bag_attention", "multiscale_attention", "tucker_attention")


def cosine_basis(size, rank):
    position = np.arange(size)[:, None] + 0.5
    frequency = np.arange(rank)[None, :]
    basis = np.cos(np.pi * position * frequency / size) * np.sqrt(2 / size)
    basis[:, 0] /= np.sqrt(2)
    return torch.tensor(basis, dtype=torch.float32)


class MultiscaleViews(nn.Module):
    """Preserve XYZ; original view plus shared trainable temporal filters.

    Filters initialize as moving averages but may learn signed responses;
    their learned scale axis is not a guaranteed physiological frequency band.
    """
    def __init__(self, windows=(9, 33, 129)):
        super().__init__()
        if not windows or any(window < 1 or window % 2 == 0 for window in windows):
            raise ValueError("Scale windows must be positive odd lengths")
        width = max(windows)
        kernels = torch.zeros(len(windows), 1, width)
        for index, window in enumerate(windows):
            start = (width - window) // 2
            kernels[index, 0, start:start + window] = 1 / window
        self.kernels = nn.Parameter(kernels)
        self.padding = width // 2

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 3 or x.shape[-1] <= self.padding:
            raise ValueError("Expected XYZ beats longer than the filter padding")
        n, leads, time = x.shape
        flat = x.reshape(n * leads, 1, time)
        filtered = F.conv1d(F.pad(flat, (self.padding, self.padding), mode="reflect"), self.kernels)
        return torch.cat([flat, filtered], dim=1).reshape(n, leads, -1, time).permute(0, 1, 3, 2)


class TensorFeatures(nn.Module):
    """Lead x time x scale features: fixed pooling or supervised Tucker maps."""
    def __init__(self, config, tucker):
        super().__init__()
        self.views = MultiscaleViews(tuple(config["scale_windows"]))
        self.time_bins = int(config["tensor_time_bins"])
        self.time_pool = TemporalBinPool(self.time_bins)
        lead_rank, time_rank, scale_rank = map(int, config["tensor_ranks"])
        scales = 1 + len(config["scale_windows"])
        if not 1 <= lead_rank <= 3 or not 1 <= time_rank <= self.time_bins or not 1 <= scale_rank <= scales:
            raise ValueError("Tensor ranks exceed their physical mode dimensions")
        self.tucker = tucker
        if tucker:
            self.lead_factor = nn.Parameter(cosine_basis(3, lead_rank))
            self.time_factor = nn.Parameter(cosine_basis(self.time_bins, time_rank))
            self.scale_factor = nn.Parameter(cosine_basis(scales, scale_rank))
            width = lead_rank * time_rank * scale_rank
        else:
            self.fixed_pool = TemporalBinPool(time_rank)
            width = 3 * time_rank * scales
        self.projection = nn.Sequential(nn.Linear(width, int(config["tensor_embedding"])), nn.GELU())

    def core(self, x):
        views = self.views(x)  # n x lead x time x scale, raw view never dropped
        views = self.time_pool(views.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        if not self.tucker:
            return self.fixed_pool(views.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # Sequential mode products are easier to audit than one flattened map.
        core = views.permute(0, 2, 3, 1) @ self.lead_factor  # n,t,s,a
        core = core.permute(0, 2, 3, 1) @ self.time_factor   # n,s,a,b
        return core.permute(0, 2, 3, 1) @ self.scale_factor  # n,a,b,c

    def forward(self, x):
        return self.projection(self.core(x).flatten(1))

    def penalty(self):
        if not self.tucker:
            return self.views.kernels.new_zeros(())
        return sum(((factor.T @ factor - torch.eye(factor.shape[1], device=factor.device)) ** 2).mean()
                   for factor in [self.lead_factor, self.time_factor, self.scale_factor])


class PatientTensorModel(nn.Module):
    def __init__(self, name, config):
        super().__init__()
        if name not in MODELS:
            raise ValueError("Unknown patient tensor model")
        base = VCGResNet(n_classes=len(config["classes"]), widths=tuple(config["widths"]),
                         pool_bins=int(config["pool_bins"]), dropout=float(config["dropout"]))
        self.encoder = base.encoder
        self.tensor = TensorFeatures(config, name == "tucker_attention") if name in {"multiscale_attention", "tucker_attention"} else None
        width = 64 + (int(config["tensor_embedding"]) if self.tensor is not None else 0)
        self.head = base.classifier if width == 64 else nn.Sequential(nn.Dropout(float(config["dropout"])), nn.Linear(width, len(config["classes"])))
        self.attention = name != "bag_mean"
        if self.attention:
            hidden = int(config["attention_hidden"])
            self.gate_v, self.gate_u, self.gate_w = nn.Linear(width, hidden), nn.Linear(width, hidden), nn.Linear(hidden, 1, bias=False)

    def encode(self, waveform):
        features = self.encoder(waveform)
        return features if self.tensor is None else torch.cat([features, self.tensor(waveform)], dim=1)

    def pool(self, embeddings, mask):
        if mask.shape != embeddings.shape[:2] or not mask.any(dim=1).all():
            raise ValueError("Every patient bag needs at least one valid beat")
        if self.attention:
            scores = self.gate_w(torch.tanh(self.gate_v(embeddings)) * torch.sigmoid(self.gate_u(embeddings))).squeeze(-1)
            weights = scores.masked_fill(~mask, -torch.inf).softmax(dim=1)
        else:
            weights = mask.to(embeddings.dtype) / mask.sum(1, keepdim=True)
        return (embeddings * weights[..., None]).sum(1), weights

    def forward(self, waveforms, mask):
        b, k, leads, time = waveforms.shape
        embeddings = self.encode(waveforms.reshape(b * k, leads, time)).reshape(b, k, -1)
        pooled, _ = self.pool(embeddings, mask)
        return self.head(pooled)

    def penalty(self):
        return self.tensor.penalty() if self.tensor is not None else next(self.parameters()).new_zeros(())


class PatientBags(Dataset):
    """Random sets of beats; never imply chronological continuity across records."""
    def __init__(self, waves, y, groups, train, normalizer, bag_size, seed):
        self.waves, self.y, self.groups, self.normalizer = waves, y, groups, normalizer
        self.ids = np.unique(groups[train])
        self.rows = [train[groups[train] == patient] for patient in self.ids]
        self.labels = np.array([int(y[rows[0]]) for rows in self.rows])
        if any(len(np.unique(y[rows])) != 1 for rows in self.rows):
            raise ValueError("Patient bag has inconsistent labels")
        self.bag_size, self.rng = int(bag_size), np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        rows = self.rows[index]
        chosen = self.rng.choice(rows, min(len(rows), self.bag_size), replace=False)
        mask = np.arange(self.bag_size) < len(chosen)
        padded = np.pad(chosen, (0, self.bag_size - len(chosen)), mode="edge")
        wave = (np.asarray(self.waves[padded], dtype=np.float32) - self.normalizer.wave_mean_[:, None]) / self.normalizer.wave_scale_[:, None]
        return torch.from_numpy(wave.copy()), torch.from_numpy(mask), int(self.labels[index])


def predict_patients(model, normalizer, waves, y, groups, indices, config):
    """Use ALL beats once per patient, in bounded encoder batches."""
    ids, truth, scores = np.unique(groups[indices]), [], []
    device = _device(config)
    model.eval()
    with torch.no_grad():
        for patient in ids:
            rows = indices[groups[indices] == patient]
            if len(np.unique(y[rows])) != 1:
                raise ValueError("Inconsistent patient labels")
            chunks = []
            for start in range(0, len(rows), int(config["inference_batch_size"])):
                x = (np.asarray(waves[rows[start:start + int(config["inference_batch_size"])]], dtype=np.float32) - normalizer.wave_mean_[:, None]) / normalizer.wave_scale_[:, None]
                chunks.append(model.encode(torch.from_numpy(x.copy()).to(device)))
            embeddings = torch.cat(chunks).unsqueeze(0)
            pooled, _ = model.pool(embeddings, torch.ones(embeddings.shape[:2], dtype=torch.bool, device=device))
            scores.append(model.head(pooled).softmax(1).cpu().numpy()[0])
            truth.append(int(y[rows[0]]))
    probabilities = np.asarray(scores)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise RuntimeError("Invalid patient bag probabilities")
    return ids, np.asarray(truth), probabilities


def fit_patient_model(name, waves, statistics, y, groups, train, valid, epochs, config, seed):
    if valid is not None and set(groups[train]) & set(groups[valid]):
        raise ValueError("Patient training/validation overlap")
    _seed(seed, config)
    normalizer = FoldNormalizer().fit(waves, statistics, train, patient_balanced_weights(y[train], groups[train]))
    dataset = PatientBags(waves, y, groups, train, normalizer, config["bag_size"], seed)
    weights = patient_balanced_weights(dataset.labels, dataset.ids)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), int(config["bags_per_epoch"]),
                                    replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), sampler=sampler, num_workers=0)
    model = PatientTensorModel(name, config).to(_device(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    best, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(_device(config)), mask.to(_device(config))), labels.to(_device(config)))
            loss = loss + float(config["orthogonality_weight"]) * model.penalty()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite patient tensor training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "training_loss": float(np.mean(losses))}
        if valid is not None:
            _, truth, probabilities = predict_patients(model, normalizer, waves, y, groups, valid, config)
            score = float(f1_score(truth, probabilities.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
            row["validation_patient_macro_f1"] = score
            if score > best + 1e-8:
                best, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
        if valid is not None and epoch - best_epoch >= int(config["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    audit = {"seed": seed, "selected_epoch": best_epoch if valid is not None else epochs,
        "best_validation_macro_f1": best if valid is not None else None, "history": history,
        "train_patients": len(dataset), "validation_patients": len(np.unique(groups[valid])) if valid is not None else 0,
        "normalizer_fit_beats": len(train), "normalization": normalizer.audit(),
        "parameter_count": sum(p.numel() for p in model.parameters()), "bags_per_epoch": config["bags_per_epoch"],
        "bag_size": config["bag_size"], "patient_overlap": 0}
    return model, normalizer, audit


def validate_config(config):
    if not config["models"] or set(config["models"]) - set(MODELS) or len(set(config["models"])) != len(config["models"]):
        raise ValueError("Choose unique patient tensor models")
    for name in ["bag_size", "bags_per_epoch", "batch_size", "inference_batch_size", "max_epochs", "patience", "torch_threads"]:
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if config["inner_splits"] < 2 or config["learning_rate"] <= 0 or config["orthogonality_weight"] < 0:
        raise ValueError("Invalid patient training settings")


def run_patient_study(config, outer_folds=None):
    validate_config(config)
    bank = Path(config["bank_dir"])
    manifest = json.loads((bank / "manifest.json").read_text())
    if manifest["classes"] != config["classes"] or not Path(config["folds_path"]).exists():
        raise ValueError("Patient study requires the existing cohort and frozen folds")
    waves, statistics = [np.load(bank / f"{name}.npy", mmap_mode="r") for name in ["waveforms", "statistics"]]
    labels, groups = [np.load(bank / f"{name}.npy") for name in ["y", "groups"]]
    mapping = {label: i for i, label in enumerate(config["classes"])}
    y = np.array([mapping[str(label)] for label in labels])
    if waves.shape != tuple(manifest["shape"]) or len(groups) != len(waves) or statistics.shape[0] != len(waves):
        raise ValueError("Incomplete patient waveform bank")
    patients = patient_table(y, groups, config["classes"])
    folds = load_or_create_folds(y, groups, config).set_index("patient_id")
    all_folds = list(range(1, int(config["outer_splits"]) + 1))
    chosen = all_folds if outer_folds is None else sorted(set(outer_folds))
    if sorted(folds.fold.unique()) != all_folds or not chosen or set(chosen) - set(all_folds):
        raise ValueError("Invalid patient fold IDs")
    partial = chosen != all_folds
    output = Path(config["output_dir"]) / ("partial_folds_" + "_".join(map(str, chosen)) if partial else "complete")
    output.mkdir(parents=True, exist_ok=True)
    runtime = {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(_device(config)),
        "platform": platform.platform(), "deterministic_kernels": _device(config).type != "mps",
        "patient_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "neural_code_sha256": hashlib.sha256(Path(neural.__file__).read_bytes()).hexdigest()}
    scope = {"config": config, "bank_fingerprint": manifest["fingerprint"], "runtime": runtime,
        "folds_sha256": hashlib.sha256(Path(config["folds_path"]).read_bytes()).hexdigest(), "evaluated_folds": chosen}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    guard = output / "study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Patient study settings/runtime changed; use a new output_dir")
    _save_json(guard, scope)
    beat_folds = np.array([int(folds.loc[str(patient), "fold"]) for patient in groups])
    ids = patients.patient_id.astype(str).to_numpy()
    truth = patients.label_index.to_numpy()
    patient_folds = np.array([int(folds.loc[patient, "fold"]) for patient in ids])
    summary = []
    for name in config["models"]:
        probabilities, audits = np.full((len(ids), len(mapping)), np.nan), []
        for fold in chosen:
            train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            locations = np.flatnonzero(patient_folds == fold)
            if set(groups[train]) & set(groups[test]):
                raise RuntimeError("Patient outer-fold leakage")
            checkpoint, audit_path = output / f"fold_{fold:02d}_{name}.npz", output / f"fold_{fold:02d}_{name}.json"
            if checkpoint.exists() and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                with np.load(checkpoint) as saved:
                    if audit["study_fingerprint"] != fingerprint or not np.array_equal(saved["patient_ids"], ids[locations]):
                        raise ValueError("Patient checkpoint cohort/settings changed")
                    probabilities[locations] = saved["probabilities"]
                if not np.isfinite(probabilities[locations]).all() or np.any(probabilities[locations] < 0) or not np.allclose(probabilities[locations].sum(1), 1, atol=1e-5):
                    raise ValueError("Invalid resumed patient probabilities")
                audits.append(audit)
                print(f"{name} fold {fold}: resumed completed checkpoint", flush=True)
                continue
            selections = []
            for inner, (ti, vi) in enumerate(inner_patient_splits(y[train], groups[train], config["inner_splits"], config["split_seed"] + fold * 1000), 1):
                inner_path = output / f"inner_{fold:02d}_{inner}_{name}.json"
                if inner_path.exists():
                    selection = json.loads(inner_path.read_text())
                    if selection["study_fingerprint"] != fingerprint:
                        raise ValueError("Patient inner checkpoint changed")
                else:
                    model, normalizer, selection = fit_patient_model(name, waves, statistics, y, groups,
                        train[ti], train[vi], config["max_epochs"], config, config["training_seed"] + fold * 1000 + inner)
                    selection.update(study_fingerprint=fingerprint, inner_fold=inner)
                    _save_json(inner_path, selection)
                    del model, normalizer
                selections.append(selection)
                print(f"{name} outer {fold}/{config['outer_splits']} inner {inner}/{config['inner_splits']}: epoch {selection['selected_epoch']}, macro F1={selection['best_validation_macro_f1']:.4f}", flush=True)
            epochs = int(np.clip(np.floor(np.median([row["selected_epoch"] for row in selections]) + 0.5), 1, config["max_epochs"]))
            model, normalizer, final = fit_patient_model(name, waves, statistics, y, groups,
                train, None, epochs, config, config["training_seed"] + fold)
            predicted_ids, test_truth, values = predict_patients(model, normalizer, waves, y, groups, test, config)
            if not np.array_equal(predicted_ids, ids[locations]) or not np.array_equal(test_truth, truth[locations]):
                raise RuntimeError("Patient predictions do not align with frozen folds")
            probabilities[locations] = values
            audit = {"model": name, "fold": fold, "study_fingerprint": fingerprint, "patient_overlap": 0,
                "train_patients": len(np.unique(groups[train])), "test_patients": len(locations),
                "train_beats": len(train), "test_beats": len(test), "inner_selection": selections,
                "final_epochs": epochs, "final_fit": final, "patient_metrics": metric_bundle(test_truth, values, config["classes"])}
            temporary = checkpoint.with_suffix(".partial.npz")
            np.savez_compressed(temporary, patient_ids=predicted_ids, probabilities=values)
            temporary.replace(checkpoint)
            _save_json(audit_path, audit)
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "normalization": normalizer.audit(), "study_fingerprint": fingerprint}, output / f"model_{fold:02d}_{name}.pt")
            del model, normalizer
            audits.append(audit)
            print(f"{name} fold {fold}/{config['outer_splits']}: patient macro F1={audit['patient_metrics']['macro_f1']:.4f}", flush=True)
        evaluated = np.isfinite(probabilities).all(1)
        if not partial and not evaluated.all():
            raise RuntimeError("Incomplete patient out-of-fold predictions")
        if np.any(probabilities[evaluated] < 0) or not np.allclose(probabilities[evaluated].sum(1), 1, atol=1e-5):
            raise RuntimeError("Invalid patient out-of-fold probabilities")
        result = {"model": name, "study_fingerprint": fingerprint, "classes": config["classes"],
            "complete": not partial, "evaluated_folds": chosen, "n_patients": int(evaluated.sum()),
            "n_beats": int(np.isin(beat_folds, chosen).sum()), "patient_metrics": metric_bundle(truth[evaluated], probabilities[evaluated], config["classes"]),
            "protocol": "Patient-bag loss; nested patient stopping; inference pools embeddings from ALL eligible patient beats. Previously inspected PTB development folds, not independent confirmation.",
            "runtime": runtime, "folds": audits}
        _save_json(output / f"metrics_{name}.json", result)
        frame = pd.DataFrame({"patient_id": ids[evaluated], "truth": [config["classes"][i] for i in truth[evaluated]],
            "prediction": [config["classes"][i] for i in probabilities[evaluated].argmax(1)], "fold": patient_folds[evaluated]})
        for i, label in enumerate(config["classes"]):
            frame[f"probability_{label}"] = probabilities[evaluated, i]
        frame.to_csv(output / f"patient_predictions_{name}.csv", index=False)
        save_confusion(truth[evaluated], probabilities[evaluated].argmax(1), config["classes"], name, output)
        summary.append({"model": name, "complete": not partial, "n_patients": int(evaluated.sum()),
            "patient_macro_f1": result["patient_metrics"]["macro_f1"], "patient_accuracy": result["patient_metrics"]["accuracy"]})
        pd.DataFrame(summary).to_csv(output / "comparison.csv", index=False)
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/patient_tensor.yaml")
    parser.add_argument("--outer-folds", nargs="+", type=int)
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    print(run_patient_study(config, args.outer_folds).to_string(index=False))
