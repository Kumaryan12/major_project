"""Patient-nested fixed-four-beat residual-context and supervision ablations."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

from . import chronology_bank, chronology_model, chronology_study as cache
from . import residual_context_model as models
from .benchmark import inner_patient_splits, load_or_create_folds, metric_bundle
from .contrastive import initial_state_hash, _progress
from .neural import _device, _hash, _save_json
from .signal_audit import file_hash

CLAIM = "Fixed-four-beat direct morphology plus gated context development on repeatedly inspected patient folds; no independent confirmation or established SOTA."


def validate_config(config):
    if config["arms"] != list(models.ARMS) or len(set(config["training_seeds"])) != len(config["training_seeds"]) or not config["training_seeds"]:
        raise ValueError("Require all declared residual-context controls and unique seeds")
    for key in ("sequence_length", "windows_per_epoch", "batch_size", "inference_batch_size", "max_epochs", "patience", "torch_threads", "rr_embedding", "hidden_size", "outer_splits", "inner_splits"):
        if type(config[key]) != int or config[key] < 1: raise ValueError("Invalid residual-context budget")
    if config["windows_per_epoch"] % config["batch_size"] or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Require full optimizer batches and a valid optimizer")
    if not np.isfinite(config["auxiliary_weight"]) or config["auxiliary_weight"] <= 0 or not np.isfinite(config["gate_initial_bias"]):
        raise ValueError("Invalid fixed auxiliary weight/gate initialization")
    if config["bootstrap_draws"] < 100: raise ValueError("Insufficient descriptive bootstrap draws")


def load_inputs(config):
    validate_config(config)
    bank = chronology_bank.prepare_bank(config)
    if not Path(config["folds_path"]).is_file(): raise ValueError("Require original frozen patient folds")
    folds = load_or_create_folds(bank.y, bank.groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"]+1)):
        raise ValueError("Residual-context patient folds changed")
    assigned = folds.set_index("patient_id").loc[bank.ids, "fold"].to_numpy()
    return bank, assigned


def make_scope(config, bank):
    scope = {"config": config, "source": bank.provenance, "claim_scope": CLAIM,
             "folds_sha256": file_hash(config["folds_path"]), "code_sha256": file_hash(__file__),
             "model_code_sha256": file_hash(models.__file__), "metadata_code_sha256": file_hash(chronology_bank.__file__),
             "shared_model_code_sha256": file_hash(chronology_model.__file__), "cache_code_sha256": file_hash(cache.__file__),
             "neural_code_sha256": file_hash(Path(__file__).with_name("neural.py")),
             "metric_code_sha256": file_hash(Path(__file__).with_name("benchmark.py")),
             "report_code_sha256": file_hash(Path(__file__).with_name("residual_context_report.py")),
             "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(),
                         "device": str(_device(config)), "deterministic_kernels": _device(config).type != "mps"}}
    return dict(scope, study_fingerprint=_hash(scope))


def expected_fit(bank, scope, arm, seed, fold, inner, train, query):
    expected = cache.expected_fit(bank, scope, arm, seed, fold, inner, train, query)
    config = scope["config"]
    return dict(expected, sequence_length=config["sequence_length"], auxiliary_weight=models.auxiliary_weight(config, arm))


def verify_fit(audit, bank, train, config, arm, seed, valid, norm_memo):
    key = _hash(np.asarray(train).tolist())
    if key not in norm_memo: norm_memo[key] = chronology_model.SequenceNormalizer().fit(bank, train)
    norm = norm_memo[key]
    model = models.initial_model(config, seed, arm)
    if (audit["normalization"] != norm.audit() or audit["initial_state_sha256"] != initial_state_hash(model)
            or audit["seed"] != seed or audit["arm"] != arm or audit["patient_overlap"]
            or audit["train_patients"] != len(train) or audit["validation_patients"] != (len(valid) if valid is not None else 0)
            or audit["parameter_count"] != sum(x.numel() for x in model.parameters())
            or audit["auxiliary_weight"] != models.auxiliary_weight(config, arm) or audit["gate_initial_bias"] != config["gate_initial_bias"]):
        raise ValueError("Residual-context normalization/initialization/objective changed")
    del model
    history = audit["history"]
    if not history or [r["epoch"] for r in history] != list(range(1, len(history)+1)) or len(history) > config["max_epochs"]:
        raise ValueError("Residual-context history changed")
    best, epoch = -np.inf, 0
    for row in history:
        plan = chronology_model.training_schedule(bank, train, config["sequence_length"], config["windows_per_epoch"], seed, row["epoch"])
        expected = models.ResidualDataset(bank, norm, plan, config["sequence_length"], arm, seed, row["epoch"]).audit()
        if any(row.get(k) != v for k, v in expected.items()): raise ValueError("Residual-context windows/permutations changed")
        components = [row[k] for k in ("training_loss", "window_loss", "beat_loss", "gate_mean")]
        if (not np.isfinite(components).all() or min(components) < 0 or not 0 <= row["gate_mean"] <= 1
                or abs(row["training_loss"]-row["window_loss"]-audit["auxiliary_weight"]*row["beat_loss"]) > 2e-6
                or (arm.startswith("beat_") and row["gate_mean"] != 0)):
            raise ValueError("Residual-context loss/gate audit changed")
        if valid is not None:
            score = row["validation_patient_macro_f1"]
            if not np.isfinite(score) or not 0 <= score <= 1: raise ValueError("Invalid inner score")
            if score > best+1e-8: best, epoch = score, row["epoch"]
        elif "validation_patient_macro_f1" in row:
            raise ValueError("Outer refit monitored query outcomes")
    if valid is not None:
        if (audit["selected_epoch"] != epoch or audit["best_validation_macro_f1"] != best
                or (len(history) < config["max_epochs"] and len(history)-epoch != config["patience"])):
            raise ValueError("Residual-context stopping changed")
    elif audit["best_validation_macro_f1"] is not None or audit["selected_epoch"] != len(history):
        raise ValueError("Residual-context outer duration changed")
    return norm


def progress_callback(root, config, arm, seed, fold, inner, epochs):
    def progress(row):
        values = {key: row[key] for key in ("epoch", "training_loss", "window_loss", "beat_loss", "gate_mean")}
        if inner: values.update(inner_validation_macro_f1=row["validation_patient_macro_f1"], max_epochs=config["max_epochs"])
        else: values["final_epochs"] = epochs
        _progress(root, state="inner_training" if inner else "outer_training", arm=arm, training_seed=seed,
                  outer_fold=fold, inner_fold=inner, sequence_length=config["sequence_length"], **values)
    return progress


def evaluate_arm(bank, assigned, config, scope, root, arm, seed, fold, verify=False, norm_memo=None):
    if norm_memo is None: norm_memo = {}
    train, query = bank.ids[assigned != fold], bank.ids[assigned == fold]
    truth = bank.labels[assigned != fold]
    choices = []
    splits = inner_patient_splits(truth, train, config["inner_splits"], config["split_seed"]+fold*1000)
    for inner, (a, b) in enumerate(splits, 1):
        fit_ids, valid = train[a], train[b]
        expected = expected_fit(bank, scope, arm, seed, fold, inner, fit_ids, valid)
        path = root/f"inner_{fold:02d}_{arm}_s{seed}_i{inner}.json"
        fit_seed = seed+fold*1000+inner
        if verify or path.exists():
            audit, p = cache.read_result(path, expected, valid, len(config["classes"]))
        else:
            model, norm, fit, p, coverage = models.fit(bank, fit_ids, valid, config, fit_seed, arm, config["max_epochs"],
                progress_callback(root, config, arm, seed, fold, inner, config["max_epochs"]))
            audit = dict(expected, fit=fit, coverage=coverage, probabilities_sha256=cache.save_result(path, valid, p))
            _save_json(path, audit);del model, norm
        score = metric_bundle(bank.labels[np.searchsorted(bank.ids, valid)], p, config["classes"])["macro_f1"]
        if abs(score-audit["fit"]["best_validation_macro_f1"]) > 1e-12:
            raise ValueError("Residual-context inner predictions disagree with stopping")
        if verify:
            norm = verify_fit(audit["fit"], bank, fit_ids, config, arm, fit_seed, valid, norm_memo)
            plan, coverage = chronology_model.inference_schedule(bank, valid, config["sequence_length"])
            dataset = models.ResidualDataset(bank, norm, plan, config["sequence_length"], arm, fit_seed, 0)
            coverage.update({k: v for k, v in dataset.audit().items() if "order_sha256" in k})
            if coverage != audit["coverage"]: raise ValueError("Residual-context inner coverage changed")
        choices.append({"inner_fold": inner, "selected_epoch": audit["fit"]["selected_epoch"],
                        "inner_macro_f1": score, "audit_sha256": file_hash(path)})
    epochs = int(np.clip(np.floor(np.median([v["selected_epoch"] for v in choices])+0.5), 1, config["max_epochs"]))
    expected = dict(expected_fit(bank, scope, arm, seed, fold, 0, train, query), final_epochs=epochs, inner_choices=choices)
    path = root/f"outer_{fold:02d}_{arm}_s{seed}.json"
    fit_seed = seed+fold
    if verify or path.exists():
        audit, p = cache.read_result(path, expected, query, len(config["classes"]), outer=True)
    else:
        model, norm, fit, _, _ = models.fit(bank, train, None, config, fit_seed, arm, epochs,
            progress_callback(root, config, arm, seed, fold, 0, epochs))
        p, coverage = models.predict(model, norm, bank, query, config, fit_seed)
        temporary = path.with_suffix(".partial.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "normalization": norm.audit(), "expected": expected}, temporary)
        temporary.replace(path.with_suffix(".pt"))
        audit = dict(expected, fit=fit, coverage=coverage, probabilities_sha256=cache.save_result(path, query, p),
                     model_sha256=file_hash(path.with_suffix(".pt")),
                     patient_metrics=metric_bundle(bank.labels[np.searchsorted(bank.ids, query)], p, config["classes"]))
        _save_json(path, audit);del model, norm
    if verify:
        norm = verify_fit(audit["fit"], bank, train, config, arm, fit_seed, None, norm_memo)
        state = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        if state["expected"] != expected or state["normalization"] != norm.audit() or len(audit["fit"]["history"]) != epochs:
            raise ValueError("Residual-context checkpoint metadata changed")
        model = models.initial_model(config, fit_seed, arm);model.load_state_dict(state["state_dict"]);model.to(_device(config))
        reproduced, coverage = models.predict(model, norm, bank, query, config, fit_seed)
        if (coverage != audit["coverage"] or not np.array_equal(p.argmax(1), reproduced.argmax(1))
                or not np.allclose(p, reproduced, rtol=1e-5, atol=1e-6)):
            raise ValueError("Residual-context restored predictions/coverage changed")
        if audit["patient_metrics"] != metric_bundle(bank.labels[np.searchsorted(bank.ids, query)], p, config["classes"]):
            raise ValueError("Residual-context outer metrics changed")
        del model
    else:
        _progress(root, state="outer_complete", arm=arm, training_seed=seed, outer_fold=fold)
        print(f"residual context: {arm}, seed {seed}, outer {fold}/{config['outer_splits']}: complete; partial scores are not final performance", flush=True)
    return np.flatnonzero(assigned == fold), p


def run_study(config):
    bank, assigned = load_inputs(config);scope = make_scope(config, bank)
    root = Path(config["output_dir"])/"complete";root.mkdir(parents=True, exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Residual-context source/config/runtime changed; use a new namespace")
    _save_json(guard, scope)
    for fold in range(1, config["outer_splits"]+1):
        for seed in config["training_seeds"]:
            for arm in config["arms"]: evaluate_arm(bank, assigned, config, scope, root, arm, seed, fold)
    _progress(root, state="fits_complete_report_pending")
    return root


def smoke(config, ensure=False):
    bank, assigned = load_inputs(config);scope = make_scope(config, bank)
    root = Path(config["output_dir"])/"smoke";root.mkdir(parents=True, exist_ok=True)
    path = root/"summary.json"
    if ensure and path.is_file():
        saved = json.loads(path.read_text())
        if saved.get("complete") is True and saved.get("scope") == scope and set(saved.get("arms", {})) == set(config["arms"]):
            print("Matching six-arm device smoke already passed.", flush=True);return
    train, query = bank.ids[assigned != 1], bank.ids[assigned == 1][:2]
    small = dict(config, windows_per_epoch=8, batch_size=4, max_epochs=1)
    audits = {}
    for arm in config["arms"]:
        model, norm, audit, _, _ = models.fit(bank, train, None, small, 42, arm, 1)
        p, coverage = models.predict(model, norm, bank, query, small, 42)
        target = root/f"{arm}.pt"
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "normalization": norm.audit()}, target)
        state = torch.load(target, map_location="cpu", weights_only=True)
        restored = models.initial_model(small, 42, arm);restored.load_state_dict(state["state_dict"]);restored.to(_device(small))
        after, repeated = models.predict(restored, chronology_model.SequenceNormalizer.restore(state["normalization"]), bank, query, small, 42)
        if coverage != repeated or not np.array_equal(p.argmax(1), after.argmax(1)) or not np.allclose(p, after, rtol=1e-5, atol=1e-6):
            raise ValueError("Residual-context device smoke restoration failed")
        audits[arm] = {"fit": audit, "coverage": coverage, "restore_verified": True}
        del model, restored
    if len({a["fit"]["initial_state_sha256"] for a in audits.values()}) != 1:
        raise ValueError("Residual-context arm initializations are not matched")
    _save_json(path, {"complete": True, "scope": scope, "claim": "Six-arm device/gradient/checkpoint checks, not classification performance", "arms": audits})
    print("Actual-data residual-context smoke passed for all six arms.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--config", default="configs/residual_context_v1.yaml")
    parser.add_argument("--smoke", action="store_true");parser.add_argument("--ensure-smoke", action="store_true")
    args = parser.parse_args();config = yaml.safe_load(Path(args.config).read_text())
    try:
        if args.smoke or args.ensure_smoke: smoke(config, args.ensure_smoke)
        else: print(run_study(config))
    except Exception as error:
        root = Path(config["output_dir"])/("smoke" if args.smoke or args.ensure_smoke else "complete");root.mkdir(parents=True, exist_ok=True)
        _progress(root, state="failed", error_type=type(error).__name__, error=str(error));raise
