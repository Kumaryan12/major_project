"""Patient-nested chronological VCG/RR ablations; fresh initialization per fit."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

from . import chronology_bank as bank_module, chronology_model as models, neural
from .benchmark import inner_patient_splits, load_or_create_folds, metric_bundle
from .contrastive import initial_state_hash, _progress
from .multiview_bank import array_hash
from .neural import _device, _hash, _save_json
from .signal_audit import file_hash

CLAIM = "Chronological VCG development on repeatedly inspected patient folds; detector-based RR, no independent confirmation or demonstrated SOTA."


def validate_config(config):
    if config["arms"] != list(models.ARMS) or len(set(config["training_seeds"])) != len(config["training_seeds"]) or not config["training_seeds"]:
        raise ValueError("Require declared chronology/RR controls and unique training seeds")
    for key in ("sequence_length", "windows_per_epoch", "batch_size", "inference_batch_size", "max_epochs", "patience", "torch_threads", "rr_embedding", "hidden_size"):
        if type(config[key]) != int or config[key] < 1:
            raise ValueError("Invalid chronological training budget")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0 or config["bootstrap_draws"] < 100:
        raise ValueError("Invalid optimizer/report budget")


def load_inputs(config):
    validate_config(config)
    bank = bank_module.prepare_bank(config)
    if not Path(config["folds_path"]).is_file():
        raise ValueError("Require existing frozen patient folds")
    folds = load_or_create_folds(bank.y, bank.groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"]+1)):
        raise ValueError("Chronology outer patient folds changed")
    assigned = folds.set_index("patient_id").loc[bank.ids, "fold"].to_numpy()
    return bank, assigned


def make_scope(config, bank):
    scope = {"config": config, "source": bank.provenance, "claim_scope": CLAIM,
             "folds_sha256": file_hash(config["folds_path"]), "code_sha256": file_hash(__file__),
             "model_code_sha256": file_hash(models.__file__), "metadata_code_sha256": file_hash(bank_module.__file__),
             "neural_code_sha256": file_hash(neural.__file__),
             "metric_helper_code_sha256": file_hash(Path(__file__).with_name("deviation_report.py")),
             "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(),
                         "device": str(_device(config)), "deterministic_kernels": _device(config).type != "mps"}}
    return dict(scope, study_fingerprint=_hash(scope))


def check_p(p, n, classes):
    if p.shape != (n, classes) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid chronological probabilities")


def save_result(path, query, p):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, patient_ids=np.asarray(query), probabilities=p);temporary.replace(path.with_suffix(".npz"))
    return file_hash(path.with_suffix(".npz"))


def read_result(path, expected, query, classes, outer=False):
    audit = json.loads(path.read_text())
    if any(audit.get(key) != value for key, value in expected.items()) or audit["probabilities_sha256"] != file_hash(path.with_suffix(".npz")):
        raise ValueError("Chronology cached fit/probabilities changed")
    with np.load(path.with_suffix(".npz")) as saved:
        if not np.array_equal(saved["patient_ids"], query):
            raise ValueError("Chronology cached patient membership changed")
        p = saved["probabilities"]
    check_p(p, len(query), classes)
    if outer and audit["model_sha256"] != file_hash(path.with_suffix(".pt")):
        raise ValueError("Chronology model checkpoint changed")
    return audit, p


def expected_fit(bank, scope, arm, seed, fold, inner, train, query):
    if set(train) & set(query):
        raise ValueError("Chronological fit/query overlap")
    return {"study_fingerprint": scope["study_fingerprint"], "arm": arm, "training_seed": seed, "outer_fold": fold, "inner_fold": inner,
            "train_membership_sha256": _hash(train.tolist()), "query_membership_sha256": _hash(query.tolist()),
            "training_labels_sha256": array_hash(bank.labels[np.searchsorted(bank.ids, train)]),
            "train_beats": len(bank.beats_for(train)), "query_beats": len(bank.beats_for(query)), "patient_overlap": 0}


def evaluate_arm(bank, assigned, config, scope, root, arm, seed, fold, verify=False):
    train, query = bank.ids[assigned != fold], bank.ids[assigned == fold]
    truth_train = bank.labels[assigned != fold]
    choices = []
    for inner, (ti, vi) in enumerate(inner_patient_splits(truth_train, train, config["inner_splits"], config["split_seed"]+fold*1000), 1):
        fit, valid = train[ti], train[vi]
        expected = expected_fit(bank, scope, arm, seed, fold, inner, fit, valid)
        path = root/f"inner_{fold:02d}_{arm}_s{seed}_i{inner}.json"
        fit_seed = seed+fold*1000+inner
        if verify or path.exists():
            audit, p = read_result(path, expected, valid, len(config["classes"]))
        else:
            def progress(row):
                _progress(root, state="inner_training", arm=arm, training_seed=seed, outer_fold=fold,
                          inner_fold=inner, epoch=row["epoch"], training_loss=row["training_loss"], validation_macro_f1=row["validation_patient_macro_f1"])
            model, norm, result, p, coverage = models.fit(bank, fit, valid, config, fit_seed, arm, config["max_epochs"], progress)
            audit = dict(expected, fit=result, coverage=coverage, probabilities_sha256=save_result(path, valid, p))
            _save_json(path, audit);del model, norm
        if verify:
            norm = models.SequenceNormalizer().fit(bank, fit)
            model = models.initial_model(config, fit_seed, arm)
            if audit["fit"]["normalization"] != norm.audit() or audit["fit"]["initial_state_sha256"] != initial_state_hash(model) or audit["fit"]["seed"] != fit_seed or audit["fit"]["arm"] != arm:
                raise ValueError("Chronology inner normalization/initialization changed")
            del model
            history = audit["fit"]["history"]
            if not history or [row["epoch"] for row in history] != list(range(1, len(history)+1)) or len(history) > config["max_epochs"]:
                raise ValueError("Chronology inner history is invalid")
            best, epoch = -np.inf, 0
            for row in audit["fit"]["history"]:
                plan = models.training_schedule(bank, fit, config["sequence_length"], config["windows_per_epoch"], fit_seed, row["epoch"])
                expected_schedule = models.WindowDataset(bank, norm, plan, config["sequence_length"], arm, fit_seed, row["epoch"]).audit()
                if any(row.get(key) != value for key, value in expected_schedule.items()) or not np.isfinite(row["training_loss"]):
                    raise ValueError("Chronology training windows/permutations changed")
                if row["validation_patient_macro_f1"] > best+1e-8:
                    best, epoch = row["validation_patient_macro_f1"], row["epoch"]
            score = metric_bundle(bank.labels[np.searchsorted(bank.ids, valid)], p, config["classes"])["macro_f1"]
            if epoch != audit["fit"]["selected_epoch"] or best != audit["fit"]["best_validation_macro_f1"] or abs(score-best) > 1e-12:
                raise ValueError("Chronology selected epoch/probability score changed")
            if len(history) < config["max_epochs"] and len(history)-epoch != config["patience"]:
                raise ValueError("Chronology early-stopping patience changed")
            plan, coverage = models.inference_schedule(bank, valid, config["sequence_length"])
            coverage["token_order_sha256"] = models.WindowDataset(bank, norm, plan, config["sequence_length"], arm, fit_seed, 0).audit()["token_order_sha256"]
            if coverage != audit["coverage"]:
                raise ValueError("Chronology inner query coverage changed")
        choices.append({"selected_epoch": audit["fit"]["selected_epoch"], "inner_audit_sha256": file_hash(path)})
    epochs = int(np.clip(np.floor(np.median([x["selected_epoch"] for x in choices])+.5), 1, config["max_epochs"]))
    expected = expected_fit(bank, scope, arm, seed, fold, 0, train, query)
    expected.update(inner_choices=choices, final_epochs=epochs)
    path = root/f"outer_{fold:02d}_{arm}_s{seed}.json"
    fit_seed = seed+fold
    if verify or path.exists():
        audit, p = read_result(path, expected, query, len(config["classes"]), outer=True)
    else:
        def progress(row):
            _progress(root, state="outer_training", arm=arm, training_seed=seed, outer_fold=fold, epoch=row["epoch"], final_epochs=epochs, training_loss=row["training_loss"])
        model, norm, result, _, _ = models.fit(bank, train, None, config, fit_seed, arm, epochs, progress)
        p, coverage = models.predict(model, norm, bank, query, config, fit_seed)
        temporary = path.with_suffix(".partial.pt")
        torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "normalization": norm.audit(), "expected": expected}, temporary)
        temporary.replace(path.with_suffix(".pt"))
        audit = dict(expected, fit=result, coverage=coverage, probabilities_sha256=save_result(path, query, p), model_sha256=file_hash(path.with_suffix(".pt")),
                     patient_metrics=metric_bundle(bank.labels[assigned == fold], p, config["classes"]))
        _save_json(path, audit);del model, norm
    if verify:
        saved = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        norm = models.SequenceNormalizer().fit(bank, train)
        model = models.initial_model(config, fit_seed, arm)
        if saved["expected"] != expected or saved["normalization"] != norm.audit() or audit["fit"]["normalization"] != norm.audit() or audit["fit"]["initial_state_sha256"] != initial_state_hash(model) or audit["fit"]["seed"] != fit_seed or audit["fit"]["arm"] != arm:
            raise ValueError("Chronology outer model/normalization changed")
        model.load_state_dict(saved["state_dict"]);model.to(_device(config))
        restored, coverage = models.predict(model, norm, bank, query, config, fit_seed)
        if coverage != audit["coverage"] or not np.array_equal(p.argmax(1), restored.argmax(1)) or not np.allclose(p, restored, rtol=1e-5, atol=1e-6):
            raise ValueError("Restored chronology predictions/coverage changed")
        if audit["patient_metrics"] != metric_bundle(bank.labels[assigned == fold], p, config["classes"]):
            raise ValueError("Chronology outer metrics changed")
        for row in audit["fit"]["history"]:
            plan = models.training_schedule(bank, train, config["sequence_length"], config["windows_per_epoch"], fit_seed, row["epoch"])
            schedule = models.WindowDataset(bank, norm, plan, config["sequence_length"], arm, fit_seed, row["epoch"]).audit()
            if any(row.get(key) != value for key, value in schedule.items()):
                raise ValueError("Chronology outer schedule changed")
        if audit["fit"]["selected_epoch"] != epochs or [r["epoch"] for r in audit["fit"]["history"]] != list(range(1, epochs+1)) or audit["fit"]["best_validation_macro_f1"] is not None:
            raise ValueError("Chronology outer training used an invalid stopping policy")
        del model
    else:
        print(f"chronology {arm}, training seed {seed}, outer {fold}/{config['outer_splits']}: complete; partial scores are not final", flush=True)
        _progress(root, state="outer_complete", arm=arm, training_seed=seed, outer_fold=fold)
    return np.flatnonzero(assigned == fold), p


def run_study(config):
    bank, assigned = load_inputs(config);scope = make_scope(config, bank)
    root = Path(config["output_dir"])/"complete";root.mkdir(parents=True, exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Chronology data/code/runtime changed; use a new output directory")
    _save_json(guard, scope)
    for fold in range(1, config["outer_splits"]+1):
        for seed in config["training_seeds"]:
            for arm in config["arms"]:
                evaluate_arm(bank, assigned, config, scope, root, arm, seed, fold)
    _progress(root, state="fits_complete_report_pending")
    return root


def smoke(config):
    bank, assigned = load_inputs(config)
    smoke_config = dict(config, windows_per_epoch=8, max_epochs=1, patience=1)
    train, query = bank.ids[assigned != 1], bank.ids[assigned == 1]
    model, norm, audit, _, _ = models.fit(bank, train, None, smoke_config, 42, "ordered_rr", 1)
    p, coverage = models.predict(model, norm, bank, query[:2], smoke_config, 42)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    restored = models.initial_model(smoke_config, 42, "ordered_rr");restored.load_state_dict(state);restored.to(_device(config))
    reproduced, repeated = models.predict(restored, models.SequenceNormalizer.restore(norm.audit()), bank, query[:2], smoke_config, 42)
    if repeated != coverage or not np.allclose(p, reproduced, rtol=1e-5, atol=1e-6) or not np.array_equal(p.argmax(1), reproduced.argmax(1)):
        raise ValueError("Chronology device smoke did not reproduce predictions")
    root = Path(config["output_dir"])/"smoke";root.mkdir(parents=True, exist_ok=True)
    path = root/"smoke.pt"
    torch.save({"state_dict": state, "normalization": norm.audit()}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    restored.load_state_dict(loaded["state_dict"])
    disk_p, disk_coverage = models.predict(restored, models.SequenceNormalizer.restore(loaded["normalization"]), bank, query[:2], smoke_config, 42)
    if disk_coverage != coverage or not np.allclose(disk_p, p, rtol=1e-5, atol=1e-6):
        raise ValueError("Chronology disk checkpoint smoke did not restore")
    _save_json(root/"summary.json", {"complete": True, "claim": "Finite actual-data/device gradient/restore smoke only; no score claim",
                "device": str(_device(config)), "scope": make_scope(config, bank), "fit": audit, "coverage": coverage, "state_restore_verified": True})
    print("Chronology actual-data forward/backward/restore smoke passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chronological_vcg.yaml")
    group = parser.add_mutually_exclusive_group();group.add_argument("--smoke", action="store_true");group.add_argument("--status", action="store_true")
    args = parser.parse_args();config = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        root = Path(config["output_dir"])/"complete"
        print(f"Saved fits: {len(list(root.glob('inner_*.json')))+len(list(root.glob('outer_*.json')))}/480")
        print(f"Outer fits: {len(list(root.glob('outer_*.json')))}/120")
        if (root/"progress.json").exists(): print("Saved progress, not a live-process check:", (root/"progress.json").read_text())
        print("Report present:", (root/"REPORT.md").exists())
    else:
        try: smoke(config) if args.smoke else print(run_study(config))
        except Exception as error:
            root = Path(config["output_dir"])/"complete";root.mkdir(parents=True, exist_ok=True)
            _progress(root, state="failed", error_type=type(error).__name__, error=str(error));raise
