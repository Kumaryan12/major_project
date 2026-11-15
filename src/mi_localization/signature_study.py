"""Patient-nested ordered-trajectory feature ablations with pinned local PFN-v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import f1_score

from . import signature_bank as bank, signature_tensor as tensor, feature_stability as selection
from . import tabpfn_backend as backend, tabpfn_study as pfn_checks
from .benchmark import inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

ARMS = ["morphology", *tensor.FAMILIES]
CLAIM = "Ordered-trajectory representation development on repeatedly inspected PTB patient folds; no independent confirmation, clinical validity, novel signature theory or demonstrated MI SOTA."


def structural_mask(names):
    """Drop fixed-clock-only words by algebra/schema, without inspecting data.

    All beats use the same clock per window. Pure time words (and their IQRs)
    are constants; weighted variance on them can become slightly negative by
    rounding. Keep time-XYZ interactions, signed areas and physical means.
    """
    return np.array([not (name.split("/")[-2] in {"level1", "level2", "level3"}
                         and set(name.split("/")[-1].split("_")) == {"time"}) for name in names])


def load_inputs(config):
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    pfn = yaml.safe_load(Path(config["tabpfn_config"]).read_text())
    if config["arms"] != ARMS or config["sampling_seeds"] != [42, 123, 2026] or config["feature_counts"] != [24, 64]:
        raise ValueError("Require declared signature controls, resampling seeds and budgets")
    if config["seed"] != source["seed"] or config["split_seed"] != source["split_seed"] or config["bootstrap_replicates"] != 64 or config["audit_replicates"] != 32 or config["bootstrap_draws"] < 100 or config["exclude_clock_only"] is not True:
        raise ValueError("Signature protocol seeds/bootstrap policy changed")
    root = bank.prepare_bank(config)
    schema = json.loads((root/"feature_names.json").read_text())
    masks = {arm: structural_mask(schema[arm]) for arm in tensor.FAMILIES}
    with np.load(root/"patient_features.npz") as saved:
        ids, y = saved["patient_id"], saved["y"]
        features = {arm: saved[arm] if arm == "morphology" else np.concatenate([saved["morphology"], saved[arm][:, masks[arm]]], axis=1) for arm in ARMS}
    names = {arm: schema[arm] if arm == "morphology" else list(schema["morphology"])+[f"{arm}/{name}" for name, keep in zip(schema[arm], masks[arm], strict=True) if keep] for arm in ARMS}
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, np.sort(ids)) or any(X.shape != (len(ids), len(names[arm])) or not np.isfinite(X).all() for arm, X in features.items()):
        raise ValueError("Signature feature patient schema changed")
    if not Path(source["folds_path"]).is_file():
        raise ValueError("Require existing frozen patient folds")
    folds = load_or_create_folds(y, ids, source)
    if sorted(folds.fold.unique()) != list(range(1, source["outer_splits"]+1)):
        raise ValueError("Frozen signature patient folds changed")
    assigned = folds.set_index("patient_id").loc[ids, "fold"].to_numpy()
    data = dict(source=source, pfn=pfn, ids=ids, y=y, features=features, names=names, assigned=assigned)
    data["provenance"] = {"bank_manifest_sha256": file_hash(root/"manifest.json"),
                          "bank_build_sha256": file_hash(root/"build_manifest.json"),
                          "excluded_structural_clock_columns": {arm: int((~mask).sum()) for arm, mask in masks.items()},
                          "source_config": source, "folds_sha256": file_hash(source["folds_path"]),
                          "pfn_config": pfn, "checkpoint": backend.prepare_checkpoint(pfn), "backend_runtime": backend.runtime_info()}
    return data


def make_scope(config, data):
    scope = {"config": config, "source": data["provenance"], "claim_scope": CLAIM,
             "code_sha256": file_hash(__file__), "bank_code_sha256": file_hash(bank.__file__),
             "tensor_code_sha256": file_hash(tensor.__file__), "selector_code_sha256": file_hash(selection.__file__),
             "base_selector_code_sha256": file_hash(selection.original.__file__), "backend_code_sha256": file_hash(backend.__file__),
             "metric_helper_code_sha256": file_hash(Path(__file__).with_name("deviation_report.py"))}
    return dict(scope, study_fingerprint=_hash(scope))


def prepare(data, config, root, arm, train, query, k, sampling_seed, fold, inner, memo, verify=False):
    if set(data["ids"][train]) & set(data["ids"][query]):
        raise ValueError("Signature training/query patient overlap")
    X, names = data["features"][arm], data["names"][arm]
    # Match the previous full-morphology control at seed 42; no reference split.
    selector_seed = sampling_seed+fold*1000+inner*10+1
    cfg = dict(config, _selector_code_sha256=file_hash(selection.__file__))
    source = dict(data["source"], feature_counts=config["feature_counts"])
    payload = selection.bootstrap_cache(X, data["y"], data["ids"], train, names, arm,
                                        cfg, source, selector_seed, root, memo, verify)
    weights = patient_balanced_weights(data["y"][train], data["ids"][train])
    selector = selection.ConsensusSelector().fit(X[train], data["y"][train], weights, k,
                source["correlation_threshold"], names, arm, "bootstrap", payload)
    audit = {"patient_overlap": 0, "training_patients": len(train), "query_patients": len(query),
             "training_membership_sha256": _hash(data["ids"][train].tolist()),
             "query_membership_sha256": _hash(data["ids"][query].tolist()),
             "training_labels_sha256": array_hash(data["y"][train]),
             "training_features_sha256": array_hash(X[train]), "query_features_sha256": array_hash(X[query]),
             "raw_training_context_sha256": array_hash(X[train][:, selector.selected]),
             "selector_seed": selector_seed, "selector": selector.audit(), "selected_names": [names[i] for i in selector.selected]}
    return selector, audit


def check_p(p, n, classes):
    pfn_checks.prior.reference.check_probabilities(p, n, classes)


def context(data, config, scope, root, arm, sampling_seed, fold, inner, train, query, k, memo, verify=False):
    selector, expected = prepare(data, config, root, arm, train, query, k, sampling_seed, fold, inner, memo, verify)
    seed = config["seed"]+fold*1000+inner*10+1
    expected.update(study_fingerprint=scope["study_fingerprint"], arm=arm, sampling_seed=sampling_seed,
                    fold=fold, inner=inner, k=k, inference_seed=seed)
    path = root/f"{'inner' if inner else 'outer'}_{fold:02d}_{arm}_s{sampling_seed}_i{inner}_k{k}.json"
    X, y = data["features"][arm], data["y"]
    n_classes = len(data["source"]["classes"])
    if verify or path.exists():
        audit = json.loads(path.read_text())
        if any(audit.get(key) != value for key, value in expected.items()) or audit["probabilities_sha256"] != file_hash(path.with_suffix(".npz")):
            raise ValueError("Signature cached context/selector changed")
        with np.load(path.with_suffix(".npz")) as saved:
            if not np.array_equal(saved["indices"], query):
                raise ValueError("Signature prediction membership changed")
            p = saved["probabilities"]
        if inner == 0 and (audit["model_sha256"] != file_hash(path.with_suffix(".joblib")) or audit["state_sha256"] != file_hash(path.with_suffix(".tabpfn_fit"))):
            raise ValueError("Signature saved model changed")
    else:
        state = path.with_suffix(".tabpfn_fit") if inner == 0 else None
        p, actual = backend.run_fit(data["pfn"], seed, X[train][:, selector.selected], y[train], X[query][:, selector.selected], state)
        pfn_checks.check_backend(actual, selector, n_classes, data["pfn"], seed)
        check_p(p, len(query), n_classes)
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, indices=query, probabilities=p)
        temporary.replace(path.with_suffix(".npz"))
        audit = dict(expected, backend=actual, probabilities_sha256=file_hash(path.with_suffix(".npz")))
        if inner == 0:
            temporary = path.with_suffix(".partial.joblib")
            joblib.dump({"selector": selector, "expected": expected}, temporary)
            temporary.replace(path.with_suffix(".joblib"))
            audit.update(model_sha256=file_hash(path.with_suffix(".joblib")), state_sha256=file_hash(state))
        _save_json(path, audit)
    check_p(p, len(query), n_classes)
    pfn_checks.check_backend(audit["backend"], selector, n_classes, data["pfn"], seed)
    if verify and inner == 0:
        model = joblib.load(path.with_suffix(".joblib"))
        if model["expected"] != expected or model["selector"].audit() != selector.audit():
            raise ValueError("Restored signature selector changed")
        restored, actual = backend.run_predict(data["pfn"], path.with_suffix(".tabpfn_fit"), X[query][:, selector.selected])
        pfn_checks.check_backend(actual, selector, n_classes, data["pfn"], seed)
        if not np.array_equal(p.argmax(1), restored.argmax(1)) or not np.allclose(p, restored, rtol=1e-5, atol=1e-6):
            raise ValueError("Restored signature probabilities changed")
    if not verify:
        _progress(root, state="context_complete", arm=arm, sampling_seed=sampling_seed, outer_fold=fold, inner_fold=inner, feature_budget=k)
    return p, path


def evaluate_arm(data, config, scope, root, arm, sampling_seed, fold, verify=False):
    train, test = np.flatnonzero(data["assigned"] != fold), np.flatnonzero(data["assigned"] == fold)
    splits = [(train[a], train[b]) for a, b in inner_patient_splits(data["y"][train], data["ids"][train],
                data["source"]["inner_splits"], config["split_seed"]+fold*1000)]
    choices, memo = [], {}
    for k in config["feature_counts"]:
        scores, digests = [], []
        for inner, (fit, valid) in enumerate(splits, 1):
            p, path = context(data, config, scope, root, arm, sampling_seed, fold, inner, fit, valid, k, memo, verify)
            scores.append(float(f1_score(data["y"][valid], p.argmax(1), labels=np.arange(len(data["source"]["classes"])), average="macro", zero_division=0)))
            digests.append(file_hash(path))
        choices.append({"k": k, "inner_scores": scores, "mean_inner_macro_f1": float(np.mean(scores)), "inner_audit_sha256": digests})
    chosen = max(range(len(choices)), key=lambda i: choices[i]["mean_inner_macro_f1"])
    decision = {"study_fingerprint": scope["study_fingerprint"], "arm": arm, "sampling_seed": sampling_seed,
                "fold": fold, "choices": choices, "selected_k": choices[chosen]["k"]}
    path = root/f"selection_{fold:02d}_{arm}_s{sampling_seed}.json"
    if verify:
        if json.loads(path.read_text()) != decision:
            raise ValueError("Signature inner budget choice changed")
    else:
        _save_json(path, decision)
    p, checkpoint = context(data, config, scope, root, arm, sampling_seed, fold, 0, train, test, decision["selected_k"], memo, verify)
    result = dict(decision, outer_audit_sha256=file_hash(checkpoint), patient_metrics=metric_bundle(data["y"][test], p, data["source"]["classes"]))
    path = root/f"fold_{fold:02d}_{arm}_s{sampling_seed}.json"
    if verify:
        if json.loads(path.read_text()) != result:
            raise ValueError("Signature outer metrics/choice changed")
    else:
        _save_json(path, result)
        print(f"signature {arm}, sampling seed {sampling_seed}, outer {fold}/{data['source']['outer_splits']}: complete; partial scores are not the study result", flush=True)
    return test, p


def run_study(config):
    data = load_inputs(config)
    scope = make_scope(config, data)
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/"study_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != scope:
        raise ValueError("Signature study data/code/runtime changed; use a new output directory")
    _save_json(manifest, scope)
    for fold in range(1, data["source"]["outer_splits"]+1):
        for seed in config["sampling_seeds"]:
            for arm in config["arms"]:
                evaluate_arm(data, config, scope, root, arm, seed, fold)
    _progress(root, state="fits_complete_report_pending")
    return root


def smoke(config):
    data = load_inputs(config)
    root = Path(config["output_dir"])/"smoke"
    root.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(data["assigned"] != 1)
    fit, valid = next(inner_patient_splits(data["y"][train], data["ids"][train], data["source"]["inner_splits"], config["split_seed"]+1000))
    fit, valid = train[fit], train[valid]
    selector, audit = prepare(data, config, root, "signature3", fit, valid, 24, 42, 1, 1, {})
    X = data["features"]["signature3"]
    state = root/"smoke.tabpfn_fit"
    p, actual = backend.run_fit(data["pfn"], 1053, X[fit][:, selector.selected], data["y"][fit], X[valid][:, selector.selected], state)
    check_p(p, len(valid), len(data["source"]["classes"]))
    pfn_checks.check_backend(actual, selector, len(data["source"]["classes"]), data["pfn"], 1053)
    restored, _ = backend.run_predict(data["pfn"], state, X[valid][:, selector.selected])
    if not np.array_equal(p.argmax(1), restored.argmax(1)) or not np.allclose(p, restored, rtol=1e-5, atol=1e-6):
        raise ValueError("Signature real-data smoke did not restore")
    _save_json(root/"summary.json", {"complete": True, "claim": "Numerical backend/selection smoke only; no score claim", "scope": make_scope(config, data),
                                   "audit": audit, "backend": actual, "save_load_verified": True})
    print("Signature real-data smoke passed; saved PFN context reproduces predictions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/signature_vcg.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        root = Path(config["output_dir"])/"complete"
        print(f"Saved contexts: {len(list(root.glob('inner_*.json')))+len(list(root.glob('outer_*.json')))}/1050")
        print(f"Saved outer evaluations: {len(list(root.glob('fold_*.json')))}/150")
        if (root/"progress.json").exists():
            print("Saved progress (not a live-process check):", (root/"progress.json").read_text())
        print("Report present:", (root/"REPORT.md").exists())
    else:
        try:
            smoke(config) if args.smoke else print(run_study(config))
        except Exception as error:
            failed_root = Path(config["output_dir"])/"complete"
            failed_root.mkdir(parents=True, exist_ok=True)
            _progress(failed_root, state="failed", error_type=type(error).__name__, error=str(error))
            raise
