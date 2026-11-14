"""Source-only masked-XYZ feasibility pilot; no PTB localization evaluation."""
from __future__ import annotations

import argparse
import json
import platform
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
import yaml

from . import neural, pretraining_source
from .contrastive import _progress, initial_state_hash
from .neural import FoldNormalizer, VCGResNet, _device, _hash, _save_json, _seed
from .preprocessing_study import WaveRows
from .signal_audit import file_hash


class MaskedVCG(nn.Module):
    def __init__(self, config, length=651):
        super().__init__()
        base = VCGResNet(n_classes=6, widths=tuple(config["widths"]), pool_bins=config["pool_bins"], dropout=0)
        self.encoder = base.encoder
        self.decoder = nn.Linear(64, 3 * length)
        self.length = length

    def forward(self, x):
        return self.decoder(self.encoder(x)).reshape(-1, 3, self.length)


def make_masks(count, length, patch_samples, masked_patches, generator):
    if length % patch_samples or count < 1 or not 1 <= masked_patches < length // patch_samples:
        raise ValueError("Need positive batches and a proper subset of complete time patches")
    patches = length // patch_samples
    selected = torch.rand((count, patches), generator=generator).argsort(1)[:, :masked_patches]
    mask = torch.zeros((count, patches), dtype=torch.bool).scatter_(1, selected, True)
    return mask.repeat_interleave(patch_samples, 1)[:, None, :].expand(-1, 3, -1)


def masked_mse(prediction, target, mask):
    weights = mask.to(prediction.dtype)
    return ((prediction - target).square() * weights).sum() / weights.sum()


def evaluate(model, normalizer, waves, indices, masks, config, mode="model", template=None):
    if len(indices) != len(masks) or mode not in {"model", "zero_input", "zero", "template"}:
        raise ValueError("Invalid fixed validation masks/predictor")
    device = _device(config)
    loader = DataLoader(WaveRows(waves, np.zeros(len(waves), dtype=int), indices, normalizer), batch_size=config["batch_size"],
                        num_workers=0, shuffle=False, generator=torch.Generator().manual_seed(config["seed"] + 113))
    model.eval()
    total, coordinates, position = 0.0, 0, 0
    with torch.no_grad():
        for x, _, _ in loader:
            mask = masks[position:position + len(x)].to(device)
            target = x.to(device)
            if mode == "zero":
                prediction = torch.zeros_like(target)
            elif mode == "template":
                prediction = torch.as_tensor(template, dtype=target.dtype, device=device)[None].expand_as(target)
            else:
                inputs = torch.zeros_like(target) if mode == "zero_input" else target.masked_fill(mask, 0)
                prediction = model(inputs)
            count = int(mask.sum().cpu())
            loss = masked_mse(prediction, target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite masked validation loss")
            total += float(loss.cpu()) * count
            coordinates += count
            position += len(x)
    return total / coordinates


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _save_checkpoint(path, value):
    temporary = path.with_suffix(".pt.partial")
    torch.save(_cpu(value), temporary)
    temporary.replace(path)


def validate_config(config):
    pretraining_source.validate_config(config)
    for key in ["batch_size", "samples_per_epoch", "max_epochs", "patience", "patch_samples", "masked_patches", "torch_threads"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"Invalid positive pilot setting: {key}")
    make_masks(1, 651, config["patch_samples"], config["masked_patches"], torch.Generator().manual_seed(0))
    if not np.isfinite([config["learning_rate"], config["weight_decay"]]).all() or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid pilot optimizer settings")


def status(config):
    root = Path(config["output_dir"])
    lines = [f"Source-only masked pilot: {root}"]
    for name in ["progress.json", "last_error.json"]:
        if (root / name).exists():
            lines.append(f"Last saved {name}, not a live process check: " + (root / name).read_text())
    lines.append(f"Source-pilot report ready: {(root / 'REPORT.md').exists()}")
    lines.append("No PTB classification run is part of this pilot.")
    return "\n".join(lines)


def run_pilot(config):
    validate_config(config)
    bank = pretraining_source.prepare_bank(config)
    manifest = json.loads((bank / "manifest.json").read_text())
    waves, groups, roles = [np.load(bank / f"{name}.npy", mmap_mode="r") for name in ["waveforms", "groups", "roles"]]
    train, valid = np.flatnonzero(roles == "train"), np.flatnonzero(roles == "validation")
    if list(waves.shape) != manifest["shape"] or waves.shape[1:] != (3, 651) or not np.isfinite(waves).all() or len(groups) != len(waves) or len(roles) != len(waves) or not len(train) or not len(valid) or len(train) + len(valid) != len(waves) or len(np.unique(groups)) != len(groups) or set(groups[train]) & set(groups[valid]):
        raise ValueError("Invalid source-only bank patient roles/shape")
    root = Path(config["output_dir"])
    scope = {"config": config, "bank_manifest_sha256": file_hash(bank / "manifest.json"), "bank_fingerprint": manifest["fingerprint"],
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "device": str(_device(config)),
                    "deterministic_kernels": _device(config).type != "mps", "code_sha256": file_hash(__file__), "neural_code_sha256": file_hash(neural.__file__),
                    "source_code_sha256": file_hash(pretraining_source.__file__)},
        "cross_dataset_identity_overlap": pretraining_source.IDENTITY_STATUS, "ptb_patient_independence_cleared": False}
    fingerprint = _hash(scope)
    scope["pilot_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root / "pilot_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Masked pilot data/settings/runtime changed; use a new output_dir")
    _save_json(guard, scope)
    _seed(config["seed"], config)
    device = _device(config)
    normalizer = FoldNormalizer().fit(waves, np.empty((len(waves), 0), dtype=np.float32), train, np.ones(len(train)))
    model = MaskedVCG(config).to(device)
    initial_hash = initial_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    sample_rng = torch.Generator().manual_seed(config["seed"] + 17)
    mask_rng = torch.Generator().manual_seed(config["seed"] + 19)
    loader_rng = torch.Generator().manual_seed(config["seed"] + 23)
    sampler = RandomSampler(range(len(train)), replacement=True, num_samples=config["samples_per_epoch"], generator=sample_rng)
    loader = DataLoader(WaveRows(waves, np.zeros(len(waves), dtype=int), train, normalizer), batch_size=config["batch_size"], sampler=sampler, num_workers=0, generator=loader_rng)
    validation_masks = make_masks(len(valid), 651, config["patch_samples"], config["masked_patches"], torch.Generator().manual_seed(config["seed"] + 29))
    template = ((np.asarray(waves[train]) - normalizer.wave_mean_[None, :, None]) / normalizer.wave_scale_[None, :, None]).mean(0)
    zero_baseline = evaluate(model, normalizer, waves, valid, validation_masks, config, "zero")
    template_baseline = evaluate(model, normalizer, waves, valid, validation_masks, config, "template", template)
    checkpoint = root / "checkpoint.pt"
    history, best, best_epoch, best_state = [], np.inf, 0, None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["pilot_fingerprint"] != fingerprint or saved["initial_state_sha256"] != initial_hash or saved["normalization"] != normalizer.audit():
            raise ValueError("Masked pilot checkpoint scope/normalization changed")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        sample_rng.set_state(saved["sample_rng"])
        mask_rng.set_state(saved["mask_rng"])
        loader_rng.set_state(saved["loader_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        history, best, best_epoch, best_state = saved["history"], saved["best"], saved["best_epoch"], saved["best_state_dict"]
        print(f"Resumed source-only pilot after epoch {len(history)}", flush=True)
    for epoch in range(len(history) + 1, config["max_epochs"] + 1):
        if history and len(history) - best_epoch >= config["patience"]:
            break
        model.train()
        total, coordinates, presentations = 0.0, 0, 0
        for x, _, _ in loader:
            target = x.to(device)
            mask = make_masks(len(x), 651, config["patch_samples"], config["masked_patches"], mask_rng).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(target.masked_fill(mask, 0)), target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite source-only training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            count = int(mask.sum().cpu())
            total += float(loss.detach().cpu()) * count
            coordinates += count
            presentations += len(x)
        value = evaluate(model, normalizer, waves, valid, validation_masks, config)
        if value < best - 1e-8:
            best, best_epoch, best_state = value, epoch, _cpu(model.state_dict())
        row = {"epoch": epoch, "training_masked_mse": total / coordinates, "validation_masked_mse": value, "presentations": presentations}
        history.append(row)
        _save_checkpoint(checkpoint, {"pilot_fingerprint": fingerprint, "initial_state_sha256": initial_hash,
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "normalization": normalizer.audit(),
            "sample_rng": sample_rng.get_state(), "mask_rng": mask_rng.get_state(), "loader_rng": loader_rng.get_state(),
            "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
            "history": history, "best": best, "best_epoch": best_epoch, "best_state_dict": best_state})
        _progress(root, state="training_source_only", **row)
        print(f"Source-only epoch {epoch}/{config['max_epochs']}: train masked MSE={row['training_masked_mse']:.5f}, fixed validation MSE={value:.5f}", flush=True)
    model.load_state_dict(best_state)
    final_loss = evaluate(model, normalizer, waves, valid, validation_masks, config)
    if not np.isclose(final_loss, best, atol=1e-6, rtol=1e-6):
        raise ValueError("Best source-only checkpoint does not reproduce its validation loss")
    zero_input = evaluate(model, normalizer, waves, valid, validation_masks, config, "zero_input")
    export = root / "encoder_source_only.pt"
    _save_checkpoint(export, {"encoder_state_dict": model.encoder.state_dict(), "normalization": normalizer.audit(), "pilot_fingerprint": fingerprint,
        "widths": config["widths"], "pool_bins": config["pool_bins"], "ptb_patient_independence_cleared": False,
        "cross_dataset_identity_overlap": pretraining_source.IDENTITY_STATUS, "endpoint": "source-only reconstruction; not evaluated on PTB"})
    summary = {"complete": True, "pilot_fingerprint": fingerprint, "source_training_patients": len(train), "source_validation_patients": len(valid),
        "best_epoch": best_epoch, "epochs_completed": len(history), "training_presentations": sum(row["presentations"] for row in history),
        "best_masked_validation_mse": final_loss, "training_mean_beat_template_masked_validation_mse": template_baseline,
        "zero_predictor_masked_validation_mse": zero_baseline, "trained_model_zero_input_masked_validation_mse": zero_input,
        "masked_time_fraction": config["masked_patches"] * config["patch_samples"] / 651,
        "encoder_parameters": sum(value.numel() for value in model.encoder.parameters()), "decoder_parameters": sum(value.numel() for value in model.decoder.parameters()),
        "normalization_fit_patients": normalizer.n_fit_samples_, "encoder_export_sha256": file_hash(export), "history": history,
        "ptb_classification_evaluated": False, "ptb_patient_independence_cleared": False,
        "limits": "One pretraining seed, one crop per source patient, one fixed validation mask realization; reconstruction metrics do not establish useful transfer, MI accuracy or SOTA. Cross-dataset identity overlap unresolved."}
    _save_json(root / "summary.json", summary)
    pd.DataFrame(history).to_csv(root / "history.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([row["epoch"] for row in history], [row["training_masked_mse"] for row in history], color="0.6", label="Training; fresh masks")
    ax.plot([row["epoch"] for row in history], [row["validation_masked_mse"] for row in history], color="black", label="Held-out source patients; fixed masks")
    ax.axhline(template_baseline, linestyle="--", color="0.3", label="Training mean-beat template")
    ax.set(xlabel="Pretraining epoch", ylabel="Masked standardized-coordinate MSE", title="Source-only reconstruction feasibility")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "reconstruction_curve.png", dpi=160)
    plt.close(fig)
    lines = ["# Source-only masked-XYZ pretraining pilot", "", f"{len(train):,} unique PTB-XL training patients; {len(valid):,} disjoint source validation patients. One 651-sample Kors-derived XYZ crop per retained patient. No diagnostic labels used.", "",
        f"Best fixed-mask validation MSE: **{final_loss:.5f}**, epoch {best_epoch}; {len(history)} epochs completed.", "",
        "| Validation predictor | Masked MSE |", "|---|---:|", f"| Pretrained encoder + reconstruction decoder | {final_loss:.5f} |",
        f"| Training mean-beat template | {template_baseline:.5f} |", f"| Zero predictor (training lead means) | {zero_baseline:.5f} |",
        f"| Trained model with all-zero input | {zero_input:.5f} |", "", "![Reconstruction curve](reconstruction_curve.png)", "",
        f"{config['masked_patches']} of {651 // config['patch_samples']} contiguous {config['patch_samples']}-sample time patches are masked jointly across XYZ ({100 * summary['masked_time_fraction']:.2f}% of time). Masks are fresh during training; validation masks are fixed for the entire run. Input is zeroed in training-standardized units, and MSE is computed only on masked coordinates. Source normalization and mean-beat template use training patients only.", "",
        f"The encoder reuses the existing ResNet encoder ({summary['encoder_parameters']:,} parameters); a temporary linear decoder adds {summary['decoder_parameters']:,} parameters. The decoder is not part of a future target classifier. Every epoch samples {config['samples_per_epoch']:,} source training patients with replacement. This creates additional masked views, not additional independent patients.", "",
        "Checkpoints preserve model, optimizer, best state and sampler/mask/loader/torch RNG states. Compatible interruptions resume; MPS is not guaranteed bitwise deterministic. The local encoder-only export is architecture-compatible with the existing VCG ResNet, but carries an explicit unresolved cross-dataset identity flag.", "",
        "## What this result permits", "",
        "This measures source-only reconstruction feasibility against simple baselines. A lower MSE may reflect learning an aligned average waveform; the training-template and zero-input controls help expose that limitation. Even a good reconstruction result does not establish useful localization features or downstream transfer benefit. One seed, one selected crop/patient and one validation mask realization are used.", "",
        "No PTB target labels, folds, waveforms or predictions enter this pilot. No PTB localization score is reported. PTB-versus-PTB-XL patient identity overlap and comprehensive aligned waveform-overlap screening remain unresolved, so the export is not cleared for a clean patient-independent target claim. The next step must address that provenance limitation and then run a separately declared matched transfer comparison. No clinical, 0.80–0.90 F1 or SOTA claim is made.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    _progress(root, state="source_only_complete", best_epoch=best_epoch, best_masked_validation_mse=final_loss)
    return root / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/source_pretraining_vcg.yaml")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        print(status(config))
    else:
        try:
            print(run_pilot(config))
        except BaseException as error:
            root = Path(config["output_dir"])
            if root.exists():
                _save_json(root / "last_error.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise
