"""Nested 4/8/16-beat comparison with a shared, symmetric inner length choice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from . import chronology_study as legacy, chronology_model as models
from .benchmark import inner_patient_splits, metric_bundle
from .contrastive import initial_state_hash, _progress
from .neural import _device, _hash, _save_json
from .signal_audit import file_hash

CLAIM = "Nested chronological length selection on repeatedly inspected PTB patient folds; shared length chosen symmetrically from ordered/shuffled inner F1, not outer scores or independent confirmation."


def context_config(grid, base, length):
    if length not in grid["lengths"] or type(length) != int or length < 2:
        raise ValueError("Invalid candidate beat length")
    for key in ("beat_slots_per_epoch", "beat_slots_per_batch", "inference_beat_slots"):
        if type(grid[key]) != int or grid[key] < length or grid[key] % length:
            raise ValueError("Beat budgets must divide every candidate length")
    if grid["beat_slots_per_epoch"] % grid["beat_slots_per_batch"]:
        raise ValueError("Epoch budget must divide into full optimizer batches")
    return dict(base, output_dir=grid["output_dir"], sequence_length=length,
                windows_per_epoch=grid["beat_slots_per_epoch"]//length,
                batch_size=grid["beat_slots_per_batch"]//length,
                inference_batch_size=grid["inference_beat_slots"]//length)


def load_inputs(grid):
    base = yaml.safe_load(Path(grid["base_config"]).read_text())
    if grid["length_selection"] != "paired_inner_mean" or len(grid["lengths"]) < 2 or len(set(grid["lengths"])) != len(grid["lengths"]):
        raise ValueError("Require a declared unique length grid and symmetric selection")
    for length in grid["lengths"]: context_config(grid, base, length)
    bank, assigned = legacy.load_inputs(base)
    return base, bank, assigned


def make_scope(grid, base, bank):
    scope = {"config": grid, "base_protocol": legacy.make_scope(base, bank), "claim_scope": CLAIM,
             "code_sha256": file_hash(__file__), "legacy_study_code_sha256": file_hash(legacy.__file__)}
    return dict(scope, study_fingerprint=_hash(scope))


def verify_fit(audit, bank, train, config, arm, seed, valid, norm_memo):
    key = _hash(train.tolist())
    if key not in norm_memo: norm_memo[key] = models.SequenceNormalizer().fit(bank, train)
    norm = norm_memo[key]
    model = models.initial_model(config, seed, arm)
    if audit["normalization"] != norm.audit() or audit["initial_state_sha256"] != initial_state_hash(model) or audit["seed"] != seed or audit["arm"] != arm or audit["parameter_count"] != sum(x.numel() for x in model.parameters()):
        raise ValueError("Length-study normalization/initialization changed")
    del model
    history = audit["history"]
    if not history or [r["epoch"] for r in history] != list(range(1, len(history)+1)) or len(history) > config["max_epochs"]:
        raise ValueError("Length-study history changed")
    best, epoch = -np.inf, 0
    for row in history:
        plan = models.training_schedule(bank, train, config["sequence_length"], config["windows_per_epoch"], seed, row["epoch"])
        expected = models.WindowDataset(bank, norm, plan, config["sequence_length"], arm, seed, row["epoch"]).audit()
        if any(row.get(k) != v for k, v in expected.items()) or not np.isfinite(row["training_loss"]):
            raise ValueError("Length-study window/permutation/budget changed")
        if valid is not None:
            score = row["validation_patient_macro_f1"]
            if not np.isfinite(score) or not 0 <= score <= 1: raise ValueError("Invalid inner score")
            if score > best+1e-8: best, epoch = score, row["epoch"]
        elif "validation_patient_macro_f1" in row:
            raise ValueError("Outer refit monitored validation")
    if valid is not None:
        if audit["selected_epoch"] != epoch or audit["best_validation_macro_f1"] != best or (len(history) < config["max_epochs"] and len(history)-epoch != config["patience"]):
            raise ValueError("Length-study stopping changed")
    elif audit["best_validation_macro_f1"] is not None or audit["selected_epoch"] != len(history):
        raise ValueError("Length-study outer stopping changed")
    return norm


def inner_fit(grid, base, bank, scope, root, arm, seed, fold, inner, length, train, valid, verify, norm_memo):
    config = context_config(grid, base, length)
    fit_seed = seed+fold*1000+inner
    expected = legacy.expected_fit(bank, scope, arm, seed, fold, inner, train, valid)
    expected.update(sequence_length=length, training_config_sha256=_hash(config))
    path = root/f"inner_{fold:02d}_{arm}_s{seed}_l{length}_i{inner}.json"
    if verify or path.exists():
        audit, p = legacy.read_result(path, expected, valid, len(base["classes"]))
    else:
        def progress(row):
            _progress(root, state="inner_training", arm=arm, training_seed=seed, outer_fold=fold, inner_fold=inner,
                      sequence_length=length, epoch=row["epoch"], max_epochs=config["max_epochs"], training_loss=row["training_loss"],
                      inner_validation_macro_f1=row["validation_patient_macro_f1"])
        model, norm, fit, p, coverage = models.fit(bank, train, valid, config, fit_seed, arm, config["max_epochs"], progress)
        audit = dict(expected, fit=fit, coverage=coverage, probabilities_sha256=legacy.save_result(path, valid, p))
        _save_json(path, audit);del model, norm
    score = metric_bundle(bank.labels[np.searchsorted(bank.ids, valid)], p, base["classes"])["macro_f1"]
    if abs(score-audit["fit"]["best_validation_macro_f1"]) > 1e-12:
        raise ValueError("Length-study inner predictions disagree with selected score")
    if verify:
        norm = verify_fit(audit["fit"], bank, train, config, arm, fit_seed, valid, norm_memo)
        plan, coverage = models.inference_schedule(bank, valid, length)
        coverage["token_order_sha256"] = models.WindowDataset(bank, norm, plan, length, arm, fit_seed, 0).audit()["token_order_sha256"]
        if coverage != audit["coverage"]: raise ValueError("Length-study inner coverage changed")
    return {"selected_epoch": audit["fit"]["selected_epoch"], "inner_macro_f1": score, "audit_sha256": file_hash(path)}


def choose_length(lengths, candidates):
    rows = []
    for length in lengths:
        ordered = candidates["ordered_rr"][str(length)]["mean_inner_macro_f1"]
        shuffled = candidates["shuffled_rr"][str(length)]["mean_inner_macro_f1"]
        rows.append({"length": length, "ordered_rr_mean_inner_f1": ordered, "shuffled_rr_mean_inner_f1": shuffled,
                     "paired_inner_mean": float((ordered+shuffled)/2)})
    index = max(range(len(rows)), key=lambda i: rows[i]["paired_inner_mean"])
    return rows[index]["length"], rows


def outer_fit(grid, base, bank, scope, root, arm, seed, fold, length, train, query, choices, selection_path, verify, norm_memo):
    config = context_config(grid, base, length)
    epochs = int(np.clip(np.floor(np.median([x["selected_epoch"] for x in choices])+0.5), 1, config["max_epochs"]))
    expected = legacy.expected_fit(bank, scope, arm, seed, fold, 0, train, query)
    expected.update(sequence_length=length, final_epochs=epochs, inner_choices=choices,
                    selection_sha256=file_hash(selection_path), training_config_sha256=_hash(config))
    path = root/f"outer_{fold:02d}_{arm}_s{seed}_l{length}.json"
    fit_seed = seed+fold
    if verify or path.exists():
        audit, p = legacy.read_result(path, expected, query, len(base["classes"]), outer=True)
    else:
        def progress(row):
            _progress(root, state="outer_training", arm=arm, training_seed=seed, outer_fold=fold,
                      sequence_length=length, epoch=row["epoch"], final_epochs=epochs, training_loss=row["training_loss"])
        model, norm, fit, _, _ = models.fit(bank, train, None, config, fit_seed, arm, epochs, progress)
        p, coverage = models.predict(model, norm, bank, query, config, fit_seed)
        temporary = path.with_suffix(".partial.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "normalization": norm.audit(), "expected": expected}, temporary)
        temporary.replace(path.with_suffix(".pt"))
        audit = dict(expected, fit=fit, coverage=coverage, probabilities_sha256=legacy.save_result(path, query, p),
                     model_sha256=file_hash(path.with_suffix(".pt")), patient_metrics=metric_bundle(bank.labels[np.searchsorted(bank.ids, query)], p, base["classes"]))
        _save_json(path, audit);del model, norm
    if verify:
        norm = verify_fit(audit["fit"], bank, train, config, arm, fit_seed, None, norm_memo)
        saved = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        if saved["expected"] != expected or saved["normalization"] != norm.audit() or len(audit["fit"]["history"]) != epochs:
            raise ValueError("Length-study restored outer model changed")
        model = models.initial_model(config, fit_seed, arm);model.load_state_dict(saved["state_dict"]);model.to(_device(config))
        reproduced, coverage = models.predict(model, norm, bank, query, config, fit_seed)
        if coverage != audit["coverage"] or not np.array_equal(p.argmax(1), reproduced.argmax(1)) or not np.allclose(p, reproduced, rtol=1e-5, atol=1e-6):
            raise ValueError("Length-study restored predictions/coverage changed")
        if audit["patient_metrics"] != metric_bundle(bank.labels[np.searchsorted(bank.ids, query)], p, base["classes"]):
            raise ValueError("Length-study outer metrics changed")
        del model
    else:
        _progress(root, state="outer_complete", arm=arm, training_seed=seed, outer_fold=fold, sequence_length=length)
        print(f"chronological length study: {arm}, seed {seed}, outer {fold}/{base['outer_splits']}, shared length {length}: complete", flush=True)
    return p


def evaluate_fold(grid, base, bank, assigned, scope, root, seed, fold, verify=False):
    train, query = bank.ids[assigned != fold], bank.ids[assigned == fold]
    truth = bank.labels[assigned != fold]
    splits = [(train[a], train[b]) for a, b in inner_patient_splits(truth, train, base["inner_splits"], base["split_seed"]+fold*1000)]
    candidates, norm_memo = {}, {}
    for length in grid["lengths"]:
        for arm in base["arms"]:
            rows = [inner_fit(grid, base, bank, scope, root, arm, seed, fold, inner, length, fit, valid, verify, norm_memo)
                    for inner, (fit, valid) in enumerate(splits, 1)]
            candidates.setdefault(arm, {})[str(length)] = {"mean_inner_macro_f1": float(np.mean([r["inner_macro_f1"] for r in rows])), "inner": rows}
    length, length_scores = choose_length(grid["lengths"], candidates)
    decision = {"study_fingerprint": scope["study_fingerprint"], "outer_fold": fold, "training_seed": seed,
                "length_selection": grid["length_selection"], "length_scores": length_scores,
                "selected_length": length, "candidates": candidates}
    path = root/f"selection_{fold:02d}_s{seed}.json"
    if verify:
        if json.loads(path.read_text()) != decision: raise ValueError("Shared inner length choice changed")
    else: _save_json(path, decision)
    values = {arm: outer_fit(grid, base, bank, scope, root, arm, seed, fold, length, train, query,
                            candidates[arm][str(length)]["inner"], path, verify, norm_memo) for arm in base["arms"]}
    return np.flatnonzero(assigned == fold), values, decision


def run_study(grid):
    base, bank, assigned = load_inputs(grid);scope = make_scope(grid, base, bank)
    root = Path(grid["output_dir"])/"complete";root.mkdir(parents=True, exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Length-study source/config/runtime changed; use a new namespace")
    _save_json(guard, scope)
    for fold in range(1, base["outer_splits"]+1):
        for seed in base["training_seeds"]: evaluate_fold(grid, base, bank, assigned, scope, root, seed, fold)
    _progress(root, state="fits_complete_report_pending")
    return root


def smoke(grid):
    base, bank, assigned = load_inputs(grid)
    train, query = bank.ids[assigned != 1], bank.ids[assigned == 1][:2]
    audits = {}
    for length in grid["lengths"]:
        config = dict(context_config(grid, base, length), windows_per_epoch=8, max_epochs=1)
        model, norm, audit, _, _ = models.fit(bank, train, None, config, 42, "ordered_rr", 1)
        p, coverage = models.predict(model, norm, bank, query, config, 42)
        legacy.check_p(p, len(query), len(base["classes"]))
        root = Path(grid["output_dir"])/"smoke";root.mkdir(parents=True, exist_ok=True)
        path = root/f"length{length}.pt"
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "normalization": norm.audit()}, path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        restored = models.initial_model(config, 42, "ordered_rr");restored.load_state_dict(state["state_dict"]);restored.to(_device(base))
        after, repeated = models.predict(restored, models.SequenceNormalizer.restore(state["normalization"]), bank, query, config, 42)
        if coverage != repeated or not np.array_equal(p.argmax(1), after.argmax(1)) or not np.allclose(p, after, rtol=1e-5, atol=1e-6):
            raise ValueError("Length smoke failed model restoration")
        audits[str(length)] = {"fit": audit, "coverage": coverage, "restore_verified": True}
        del model, restored
    _save_json(root/"summary.json", {"complete": True, "claim": "Finite device/gradient/restore checks, not classification results",
                                   "scope": make_scope(grid, base, bank), "device": str(_device(base)), "lengths": audits})
    print("Actual-data chronological length smoke passed for every candidate.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--config", default="configs/chronological_lengths_v1.yaml")
    parser.add_argument("--smoke", action="store_true");args = parser.parse_args()
    grid = yaml.safe_load(Path(args.config).read_text())
    try: smoke(grid) if args.smoke else print(run_study(grid))
    except Exception as error:
        root = Path(grid["output_dir"])/"complete";root.mkdir(parents=True, exist_ok=True)
        _progress(root, state="failed", error_type=type(error).__name__, error=str(error));raise
