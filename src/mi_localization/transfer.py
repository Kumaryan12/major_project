"""Exploratory source-only masked pretraining versus matched target scratch fits."""
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

from . import neural, preprocessing_study, pretraining_source, overlap_screen
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights, save_confusion
from .contrastive import initial_state_hash, _progress
from .neural import FoldNormalizer, VCGResNet, _device, _hash, _save_json, _seed
from .preprocessing_study import WaveRows, predict_waves, directory
from .regularization import load_bank
from .signal_audit import file_hash

ARMS = ("scratch", "pretrained")
CLAIM_SCOPE = "Exploratory PTB development; source-target patient identity unresolved; no independent confirmation or SOTA claim."


def validate_config(config):
    if config["arms"] != list(ARMS) or not config["training_seeds"] or len(set(config["training_seeds"])) != len(config["training_seeds"]):
        raise ValueError("Require both transfer arms and unique target training seeds")
    for key in ["batch_size", "samples_per_epoch", "max_epochs", "patience", "torch_threads", "inner_splits", "outer_splits"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < (2 if key in {"inner_splits", "outer_splits"} else 1):
            raise ValueError("Invalid positive transfer training budget")
    rates = config["learning_rates"]
    if len(rates) != 2 or len(set(rates)) != 2 or not np.isfinite(rates).all() or min(rates) <= 0 or not 0 <= config["dropout"] < 1 or not np.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("Require two positive target learning rates and valid regularization")


def load_source(config, target_bank):
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    bank = pretraining_source.prepare_bank(source)
    pilot_dir = Path(source["output_dir"])
    pilot = json.loads((pilot_dir / "pilot_manifest.json").read_text())
    pilot_fingerprint = pilot.pop("pilot_fingerprint")
    summary = json.loads((pilot_dir / "summary.json").read_text())
    if _hash(pilot) != pilot_fingerprint or pilot["config"] != source or pilot["bank_manifest_sha256"] != file_hash(bank / "manifest.json") or not summary["complete"] or summary["pilot_fingerprint"] != pilot_fingerprint:
        raise ValueError("Source pilot provenance is incomplete or changed")
    encoder_sha = file_hash(config["encoder_path"])
    if summary["encoder_export_sha256"] != encoder_sha:
        raise ValueError("Source encoder export changed")
    export = torch.load(config["encoder_path"], map_location="cpu", weights_only=True)
    if export["pilot_fingerprint"] != pilot_fingerprint or export["widths"] != config["widths"] or export["pool_bins"] != config["pool_bins"] or export["ptb_patient_independence_cleared"] is not False:
        raise ValueError("Source encoder provenance/architecture mismatch")
    state = export["encoder_state_dict"]
    if not state or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite/empty source encoder")
    screen_config = yaml.safe_load(Path(config["overlap_config"]).read_text())
    screen_dir = Path(screen_config["output_dir"])
    screen = json.loads((screen_dir / "screen_manifest.json").read_text())
    screen_fingerprint = screen.pop("screen_fingerprint")
    screened = json.loads((screen_dir / "summary.json").read_text())
    if _hash(screen) != screen_fingerprint or screen["config"] != screen_config or screen_config["source_config"] != config["source_config"] or screen["code_sha256"] != file_hash(overlap_screen.__file__) or screen["source_bank_manifest_sha256"] != file_hash(bank / "manifest.json") or screen["target_bank"] != target_bank or screened["screen_fingerprint"] != screen_fingerprint:
        raise ValueError("Source-target recording screen provenance changed")
    if screened["recording_screen_passed"] is not True or screened["patient_identity_independence_cleared"] is not False or screened["near_recording_candidate_pairs"] or screened["exact_encoded_data_matches"] or screened["incomplete_active_lead_pairs"] or screened["positive_controls"] < 1 or screened["positive_controls_retrieved_and_confirmed"] != screened["positive_controls"]:
        raise ValueError("Recording screen did not pass; review/quarantine source before target transfer")
    inventories = json.loads((screen_dir / "review/input_provenance.json").read_text())
    for role in ["source", "target"]:
        if _hash(inventories[role]) != screen[f"{role}_inventory_sha256"]:
            raise ValueError("Screen inventory changed")
        for item in inventories[role]:
            if file_hash(item["path"] + ".dat") != item["data_sha256"] or file_hash(item["path"] + ".hea") != item["header_sha256"]:
                raise ValueError("Screened raw waveform/header changed")
    provenance = {"encoder_export_sha256": encoder_sha, "pilot_fingerprint": pilot_fingerprint,
        "pilot_manifest_sha256": file_hash(pilot_dir / "pilot_manifest.json"), "source_bank_manifest_sha256": file_hash(bank / "manifest.json"),
        "screen_fingerprint": screen_fingerprint, "screen_summary_sha256": file_hash(screen_dir / "summary.json"),
        "source_training_patients": summary["source_training_patients"], "source_validation_patients": summary["source_validation_patients"],
        "pretraining_seeds": [source["seed"]], "patient_identity_independence_cleared": False, "claim_scope": CLAIM_SCOPE}
    return state, provenance


def initial_model(config, seed, arm, encoder):
    if arm not in ARMS:
        raise ValueError("Unknown transfer arm")
    _seed(seed, config)
    model = VCGResNet(len(config["classes"]), tuple(config["widths"]), config["pool_bins"], config["dropout"])
    if arm == "pretrained":
        model.encoder.load_state_dict(encoder, strict=True)
    return model


def fit_transfer(waves, y, groups, train, valid, epochs, config, seed, arm, encoder, progress=None):
    if valid is not None and set(groups[train]) & set(groups[valid]):
        raise ValueError("Transfer training/validation patients overlap")
    model = initial_model(config, seed, arm, encoder)
    initial = {"initial_state_sha256": initial_state_hash(model), "initial_classifier_sha256": initial_state_hash(model.classifier),
        "initial_encoder_sha256": initial_state_hash(model.encoder), "arm": arm, "learning_rate": config["learning_rate"]}
    if arm == "scratch":
        del model
        model, normalizer, audit = preprocessing_study.fit_waves(waves, y, groups, train, valid, epochs, config, seed, progress)
        if audit["initial_state_sha256"] != initial["initial_state_sha256"]:
            raise RuntimeError("Fresh scratch initialization changed")
        audit.update(initial)
        return model, normalizer, audit
    # Load only the encoder. Source normalization/decoder/optimizer never transfer.
    weights = patient_balanced_weights(y[train], groups[train])
    normalizer = FoldNormalizer().fit(waves, np.empty((len(waves), 0), np.float32), train, weights)
    model = model.to(_device(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), config["samples_per_epoch"], replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(WaveRows(waves, y, train, normalizer), batch_size=config["batch_size"], sampler=sampler, num_workers=0)
    best, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses, count, digest = [], 0, hashlib.sha256()
        for x, labels, rows in loader:
            digest.update(rows.numpy().astype("<i8").tobytes())
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(_device(config))), labels.to(_device(config)))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite transfer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            count += len(labels)
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)), "sampled_rows_sha256": digest.hexdigest(), "beat_presentations": count}
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
    return model, normalizer, dict(initial, seed=seed, selected_epoch=best_epoch if valid is not None else epochs,
        best_validation_macro_f1=best if valid is not None else None, history=history,
        parameter_count=sum(p.numel() for p in model.parameters()), patient_overlap=0,
        train_patients=len(np.unique(groups[train])), validation_patients=0 if valid is None else len(np.unique(groups[valid])),
        normalizer_fit_beats=len(train), normalization=normalizer.audit())


def choose_rate(candidates, config):
    scores = []
    if len(candidates) != len(config["learning_rates"]):
        raise ValueError("Missing learning-rate candidates")
    for index, rate in enumerate(config["learning_rates"]):
        fits = candidates[f"rate_{index}"]
        if len(fits) != config["inner_splits"]:
            raise ValueError("Missing inner learning-rate fits")
        values = [row["best_validation_macro_f1"] for row in fits]
        epochs = [row["selected_epoch"] for row in fits]
        if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1 or min(epochs) < 1 or max(epochs) > config["max_epochs"]:
            raise ValueError("Invalid inner learning-rate scores/epochs")
        scores.append({"candidate": f"rate_{index}", "learning_rate": rate, "mean_best_validation_macro_f1": float(np.mean(values)),
            "final_epochs": int(np.floor(np.median(epochs) + .5))})
    selected = max(range(len(scores)), key=lambda index: scores[index]["mean_best_validation_macro_f1"])
    return {"selected": scores[selected], "candidates": scores, "rule": "maximum mean inner patient macro F1; exact ties use configured order; rounded-median best epochs"}


def validate_fit(fit, config, seed, arm, rate, train_beats, train_patients, valid_patients, encoder_hash, final_epochs=None):
    if fit["seed"] != seed or fit["arm"] != arm or fit["learning_rate"] != rate or fit["patient_overlap"] or fit["normalizer_fit_beats"] != train_beats or fit["train_patients"] != train_patients or fit["validation_patients"] != valid_patients:
        raise ValueError("Transfer fit membership/seed/rate mismatch")
    expected = initial_model(config, seed, "scratch", None)
    if fit["initial_classifier_sha256"] != initial_state_hash(expected.classifier):
        raise ValueError("Target classifier initialization mismatch")
    if arm == "scratch" and (fit["initial_encoder_sha256"] != initial_state_hash(expected.encoder) or fit["initial_state_sha256"] != initial_state_hash(expected)):
        raise ValueError("Scratch initialization mismatch")
    if arm == "pretrained" and fit["initial_encoder_sha256"] != encoder_hash:
        raise ValueError("Pretrained encoder initialization mismatch")
    if fit["parameter_count"] != sum(p.numel() for p in expected.parameters()):
        raise ValueError("Transfer architecture changed")
    norm = fit["normalization"]
    if norm["stat_mean_"] or norm["stat_scale_"] or len(norm["wave_mean_"]) != 3 or len(norm["wave_scale_"]) != 3 or not np.isfinite(norm["wave_mean_"] + norm["wave_scale_"]).all() or min(norm["wave_scale_"]) <= 0:
        raise ValueError("Invalid target-only waveform normalization")
    history = fit["history"]
    if not history or len(history) > config["max_epochs"]:
        raise ValueError("Invalid transfer history")
    for epoch, row in enumerate(history, 1):
        digest = row["sampled_rows_sha256"]
        if row["epoch"] != epoch or row["beat_presentations"] != config["samples_per_epoch"] or not np.isfinite(row["training_loss"]) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid transfer history/budget")
    if final_epochs is not None:
        if len(history) != final_epochs or fit["selected_epoch"] != final_epochs or fit["best_validation_macro_f1"] is not None or valid_patients or any("validation_patient_macro_f1" in row for row in history):
            raise ValueError("Outer refit must not monitor target test scores")
    else:
        values = [row.get("validation_patient_macro_f1", np.nan) for row in history]
        if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
            raise ValueError("Invalid inner validation scores")
        best, selected = -np.inf, 0
        for epoch, value in enumerate(values, 1):
            if value > best + 1e-8:
                best, selected = value, epoch
        if fit["selected_epoch"] != selected or fit["best_validation_macro_f1"] != best or (len(history) < config["max_epochs"] and len(history) - selected != config["patience"]):
            raise ValueError("Invalid inner epoch selection")


def check_pair(first, second):
    for key in ["initial_classifier_sha256", "seed", "parameter_count", "normalization", "normalizer_fit_beats", "train_patients", "validation_patients"]:
        if first[key] != second[key]:
            raise ValueError(f"Matched transfer fits differ: {key}")
    for a, b in zip(first["history"], second["history"]):
        if a["sampled_rows_sha256"] != b["sampled_rows_sha256"]:
            raise ValueError("Matched transfer sample draws differ")


def run_study(config, outer_folds=None):
    validate_config(config)
    root, chosen, partial = directory(config, outer_folds)
    waves, y, groups, bank = load_bank(config)
    encoder, source = load_source(config, bank)
    encoder_hash = initial_state_hash(initial_model(config, config["seed"], "pretrained", encoder).encoder)
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require frozen target patient folds")
    folds = load_or_create_folds(y, groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"] + 1)):
        raise ValueError("Target fold IDs changed")
    scope = {"config": config, "bank": bank, "source": source, "source_encoder_state_sha256": encoder_hash,
        "folds_sha256": file_hash(config["folds_path"]), "evaluated_folds": chosen, "claim_scope": CLAIM_SCOPE,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "device": str(_device(config)),
            "deterministic_kernels": _device(config).type != "mps", "code_sha256": file_hash(__file__),
            "neural_code_sha256": file_hash(neural.__file__), "scratch_trainer_code_sha256": file_hash(preprocessing_study.__file__)}}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root / "study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Transfer settings/code/data/source changed; choose a new output_dir")
    _save_json(guard, scope)
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    summary_rows = []
    for seed in config["training_seeds"]:
        output = root / f"seed_{seed}"
        output.mkdir(exist_ok=True)
        probabilities = {arm: np.full((len(y), len(config["classes"])), np.nan) for arm in ARMS}
        audits = {arm: [] for arm in ARMS}
        for fold in chosen:
            train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            if set(groups[train]) & set(groups[test]):
                raise ValueError("Outer target patient overlap")
            splits = [(train[ti], train[vi]) for ti, vi in inner_patient_splits(y[train], groups[train], config["inner_splits"], config["split_seed"] + fold * 1000)]
            for arm in ARMS:
                checkpoint, audit_path = output / f"fold_{fold:02d}_{arm}.npz", output / f"fold_{fold:02d}_{arm}.json"
                if checkpoint.exists() and audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    if audit["study_fingerprint"] != fingerprint or audit["prediction_checkpoint_sha256"] != file_hash(checkpoint):
                        raise ValueError("Transfer completed checkpoint changed")
                    with np.load(checkpoint) as saved:
                        if not np.array_equal(saved["indices"], test):
                            raise ValueError("Target checkpoint membership changed")
                        probabilities[arm][test] = saved["probabilities"]
                    audits[arm].append(audit)
                    continue
                candidates = {}
                for index, rate in enumerate(config["learning_rates"]):
                    candidate = f"rate_{index}"
                    settings = dict(config, learning_rate=rate)
                    candidates[candidate] = []
                    for inner, (ti, vi) in enumerate(splits, 1):
                        path = output / f"inner_{fold:02d}_{arm}_{candidate}_{inner}.json"
                        if path.exists():
                            fit = json.loads(path.read_text())
                            if fit["study_fingerprint"] != fingerprint or fit["inner_fold"] != inner:
                                raise ValueError("Transfer inner checkpoint changed")
                        else:
                            callback = lambda row, inner=inner: _progress(root, state="training", training_seed=seed, arm=arm, learning_rate=rate, outer_fold=fold, inner_fold=inner, **row)
                            model, norm, fit = fit_transfer(waves, y, groups, ti, vi, config["max_epochs"], settings, seed + fold * 1000 + inner, arm, encoder, callback)
                            fit.update(study_fingerprint=fingerprint, inner_fold=inner)
                            del model, norm
                        validate_fit(fit, config, seed + fold * 1000 + inner, arm, rate, len(ti), len(np.unique(groups[ti])), len(np.unique(groups[vi])), encoder_hash)
                        _save_json(path, fit)
                        candidates[candidate].append(fit)
                        print(f"seed {seed} {arm} fold {fold} lr {rate:g} inner {inner}: epoch {fit['selected_epoch']}, F1 {fit['best_validation_macro_f1']:.4f}", flush=True)
                selection = choose_rate(candidates, config)
                selected = selection["selected"]
                callback = lambda row: _progress(root, state="training", training_seed=seed, arm=arm, learning_rate=selected["learning_rate"], outer_fold=fold, inner_fold=None, **row)
                settings = dict(config, learning_rate=selected["learning_rate"])
                model, norm, final = fit_transfer(waves, y, groups, train, None, selected["final_epochs"], settings, seed + fold, arm, encoder, callback)
                validate_fit(final, config, seed + fold, arm, selected["learning_rate"], len(train), len(np.unique(groups[train])), 0, encoder_hash, selected["final_epochs"])
                probabilities[arm][test] = predict_waves(model, norm, waves, y, test, config)
                _, truth, scores = aggregate_patients(probabilities[arm][test], y[test], groups[test])
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez_compressed(temporary, indices=test, probabilities=probabilities[arm][test])
                temporary.replace(checkpoint)
                audit = {"study_fingerprint": fingerprint, "training_seed": seed, "arm": arm, "fold": fold, "patient_overlap": 0,
                    "train_beats": len(train), "test_beats": len(test), "train_patients": len(np.unique(groups[train])), "test_patients": len(np.unique(groups[test])),
                    "inner_candidates": candidates, "selection": selection, "final_fit": final,
                    "prediction_checkpoint_sha256": file_hash(checkpoint), "patient_metrics": metric_bundle(truth, scores, config["classes"])}
                _save_json(audit_path, audit)
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "normalization": norm.audit(), "study_fingerprint": fingerprint}, output / f"model_{fold:02d}_{arm}.pt")
                audits[arm].append(audit)
                del model, norm
                _progress(root, state="outer_fold_complete", training_seed=seed, arm=arm, outer_fold=fold)
                print(f"seed {seed} {arm} outer {fold}/{config['outer_splits']} complete: F1 {audit['patient_metrics']['macro_f1']:.4f}", flush=True)
            for index in range(len(config["learning_rates"])):
                for a, b in zip(audits["scratch"][-1]["inner_candidates"][f"rate_{index}"], audits["pretrained"][-1]["inner_candidates"][f"rate_{index}"]):
                    check_pair(a, b)
            check_pair(audits["scratch"][-1]["final_fit"], audits["pretrained"][-1]["final_fit"])
        for arm in ARMS:
            evaluated = np.isfinite(probabilities[arm]).all(1)
            if not partial and not evaluated.all():
                raise ValueError("Incomplete target predictions")
            values = probabilities[arm][evaluated]
            if np.any(values < 0) or not np.allclose(values.sum(1), 1, atol=1e-5):
                raise ValueError("Invalid target probabilities")
            ids, truth, scores = aggregate_patients(values, y[evaluated], groups[evaluated])
            saved = {"study_fingerprint": fingerprint, "arm": arm, "training_seed": seed, "classes": config["classes"], "complete": not partial,
                "evaluated_folds": chosen, "n_patients": len(ids), "n_beats": int(evaluated.sum()), "folds": audits[arm], "patient_metrics": metric_bundle(truth, scores, config["classes"]),
                "patient_identity_independence_cleared": False, "claim_scope": CLAIM_SCOPE}
            _save_json(output / f"metrics_{arm}.json", saved)
            frame = pd.DataFrame({"patient_id": ids, "truth": [config["classes"][i] for i in truth], "prediction": [config["classes"][i] for i in scores.argmax(1)], "fold": [lookup[str(patient)] for patient in ids]})
            for index, label in enumerate(config["classes"]):
                frame[f"probability_{label}"] = scores[:, index]
            frame.to_csv(output / f"patient_predictions_{arm}.csv", index=False)
            save_confusion(truth, scores.argmax(1), config["classes"], arm, output)
            summary_rows.append({"training_seed": seed, "arm": arm, "complete": not partial, "patient_macro_f1": saved["patient_metrics"]["macro_f1"]})
            pd.DataFrame(summary_rows).to_csv(root / "comparison.csv", index=False)
    _progress(root, state="partial_complete" if partial else "complete")
    return pd.DataFrame(summary_rows)


def status(config):
    root, _, _ = directory(config)
    rows = [f"Exploratory transfer study: {root}"]
    for seed in config["training_seeds"]:
        for arm in ARMS:
            rows.append(f"seed {seed} {arm}: {len(list((root / f'seed_{seed}').glob(f'fold_*_{arm}.json')))}/{config['outer_splits']} outer folds")
    for name in ["progress.json", "last_error.json"]:
        if (root / name).exists():
            rows.append(f"Last saved {name} (not a live process check): " + (root / name).read_text())
    rows.append(f"Validated final report: {(root / 'REPORT.md').exists()}")
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transfer_vcg.yaml")
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
