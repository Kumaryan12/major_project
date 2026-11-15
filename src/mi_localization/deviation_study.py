"""Nested reference-disjoint VCG deviation study with local pinned TabPFN-v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import deviation_tensor as tensor, feature_stability as selection
from . import tabpfn_backend as backend, tabpfn_study as previous
from .benchmark import inner_patient_splits, metric_bundle, patient_balanced_weights
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

ARMS = ["full_morphology", "ensemble_morphology", "healthy", "mixed"]
CLAIM = "Healthy-reference deviation development on repeatedly inspected PTB folds; supervised reference construction, not clinical counterfactuals, independent confirmation or demonstrated SOTA."


def load_inputs(config):
    pfn = yaml.safe_load(Path(config["tabpfn_config"]).read_text())
    inherited = previous.load_inputs(pfn)
    stability, source, _, features, y, ids, names, folds, provenance = inherited
    if config["arms"] != ARMS or config["reference_parts"] != 3 or config["medoids"] != 2 or config["reference_dct"] != 32:
        raise ValueError("Require the predeclared deviation representation/controls")
    if config["reference_seeds"] != [42, 123, 2026] or config["feature_counts"] != [24, 64] or config["seed"] != source["seed"] or config["split_seed"] != source["split_seed"]:
        raise ValueError("Require fixed reference seeds and matched feature budgets")
    if config["bootstrap_replicates"] != stability["bootstrap_replicates"] or config["audit_replicates"] != stability["audit_replicates"] or config["bootstrap_draws"] < 100:
        raise ValueError("Invalid deviation bootstrap policy")
    template_root = Path(source["template_bank_dir"])
    manifest = json.loads((template_root/"manifest.json").read_text())
    if any(file_hash(template_root/name) != digest for name, digest in manifest["files"].items()):
        raise ValueError("Source recording templates changed")
    waves = np.asarray(np.load(template_root/"templates.npy"), dtype=np.float64)
    indices = np.load(template_root/"record_indices.npy")
    beats = pd.read_csv(source["beats_path"])
    beats = beats.loc[beats.label.isin(source["classes"])].reset_index(drop=True)
    if indices.shape != (len(beats),) or waves.shape != (manifest["n_records"], 3, 651) or not np.isfinite(waves).all():
        raise ValueError("Source recording/beat mapping changed")
    record_patients = np.full(len(waves), "", dtype=ids.dtype)
    for record, part in beats.groupby("record", sort=True):
        mapped = np.unique(indices[part.index])
        patients = part.patient_id.unique()
        if len(mapped) != 1 or len(patients) != 1 or not 0 <= mapped[0] < len(waves) or record_patients[mapped[0]]:
            raise ValueError("Recording templates do not map one-to-one to original records")
        record_patients[mapped[0]] = patients[0]
    if set(record_patients) != set(ids) or np.any(record_patients == ""):
        raise ValueError("Recording template patient membership changed")
    patient_waves = np.stack([np.median(waves[record_patients == patient], axis=0) for patient in ids])
    assigned = folds.set_index("patient_id").loc[ids, "fold"].to_numpy()
    if not np.array_equal(folds.set_index("patient_id").loc[ids, "label"].to_numpy(), np.asarray(source["classes"])[y]):
        raise ValueError("Original patient labels disagree")
    data = dict(source=source, pfn=pfn, X=features["morphology"], y=y, ids=ids,
                names=names["morphology"], assigned=assigned, waves=waves,
                record_patients=record_patients, patient_waves=patient_waves)
    data["provenance"] = {"prior": provenance, "source_config": source, "pfn_config": pfn,
                          "template_manifest_sha256": file_hash(template_root/"manifest.json"),
                          "record_membership_sha256": _hash(record_patients.tolist()),
                          "patient_waveforms_sha256": array_hash(patient_waves),
                          "morphology_sha256": array_hash(data["X"]), "names_sha256": _hash(data["names"]),
                          "labels_sha256": array_hash(y), "patients_sha256": _hash(ids.tolist()),
                          "folds_sha256": array_hash(assigned)}
    return data


def make_scope(config, data):
    scope = {"config": config, "source": data["provenance"], "claim_scope": CLAIM,
             "code_sha256": file_hash(__file__), "tensor_code_sha256": file_hash(tensor.__file__),
             "selector_code_sha256": file_hash(selection.__file__),
             "base_selector_code_sha256": file_hash(selection.original.__file__),
             "backend_code_sha256": file_hash(backend.__file__)}
    return dict(scope, study_fingerprint=_hash(scope))


def splits(data, config, fold):
    train = np.flatnonzero(data["assigned"] != fold)
    test = np.flatnonzero(data["assigned"] == fold)
    inner = [(train[a], train[b]) for a, b in inner_patient_splits(
        data["y"][train], data["ids"][train], data["source"]["inner_splits"], config["split_seed"]+fold*1000)]
    return train, test, inner


def members(data, config, arm, train, query, reference_seed, fold, inner):
    if arm == "full_morphology":
        tensor.check_partition(data["ids"], train, [], query)
        return [(train, np.array([], dtype=int))]
    return tensor.reference_members(data["y"], data["ids"], train, query,
                                    reference_seed+fold*1000+inner*10, config["reference_parts"])


def feature_context(data, config, arm, train, reference, query, seed):
    ids, y = data["ids"], data["y"]
    tensor.check_partition(ids, train, reference, query)
    X, names = data["X"], data["names"]
    atlas, atlas_audit = None, None
    if arm in {"healthy", "mixed"}:
        atlas, atlas_audit = tensor.fit_atlas(data["patient_waves"], y, ids, train, reference, query,
                                            data["source"]["classes"].index("HC"), arm, seed, config["reference_dct"])
        rows = np.concatenate([train, query])
        values = tensor.deviation_features(data["waves"], data["record_patients"], ids[rows], atlas, data["source"], config["reference_dct"])
        extra = np.zeros((len(ids), values.shape[1]), dtype=np.float64)
        extra[rows] = values
        X = np.concatenate([X, extra], axis=1)
        names = list(names)+[f"deviation/{name}" for name in tensor.feature_names(data["source"])]
    audit = {"training_patients": len(train), "query_patients": len(query), "reference_patients": len(reference),
             "training_membership_sha256": _hash(ids[train].tolist()), "query_membership_sha256": _hash(ids[query].tolist()),
             "reference_membership_sha256": _hash(ids[reference].tolist()),
             "training_features_sha256": array_hash(X[train]), "query_features_sha256": array_hash(X[query]),
             "training_labels_sha256": array_hash(y[train]), "feature_names_sha256": _hash(names),
             "atlas": atlas_audit, "patient_overlap": 0}
    return X, names, atlas, audit


def prepare_selector(data, config, X, names, arm, train, query, k, seed, root, memo, verify=False):
    cfg = dict(config, _selector_code_sha256=file_hash(selection.__file__))
    family = "morphology" if arm in {"full_morphology", "ensemble_morphology"} else arm
    source = dict(data["source"], feature_counts=config["feature_counts"])
    payload = selection.bootstrap_cache(X, data["y"], data["ids"], train, names, family, cfg, source, seed, root, memo, verify)
    weights = patient_balanced_weights(data["y"][train], data["ids"][train])
    selector = selection.ConsensusSelector().fit(X[train], data["y"][train], weights, k,
                source["correlation_threshold"], names, family, "bootstrap", payload)
    return selector


def check_p(p, n, classes):
    previous.prior.reference.check_probabilities(p, n, classes)


def read_result(path, expected, query, classes, outer=False):
    audit = json.loads(path.read_text())
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Deviation cached member scope changed")
    npz = path.with_suffix(".npz")
    if audit["probabilities_sha256"] != file_hash(npz):
        raise ValueError("Deviation member probabilities changed")
    with np.load(npz) as saved:
        if not np.array_equal(saved["indices"], query):
            raise ValueError("Deviation prediction patients changed")
        p = saved["probabilities"]
    check_p(p, len(query), classes)
    if outer and (audit["model_sha256"] != file_hash(path.with_suffix(".joblib")) or
                  audit["state_sha256"] != file_hash(path.with_suffix(".tabpfn_fit"))):
        raise ValueError("Deviation saved member changed")
    return audit, p


def member_fit(data, config, scope, root, arm, refseed, fold, inner, member, train, reference, query, k, memo, verify=False):
    # Classifier/inference seed is shared across arms and reference-seed repeats.
    seed = config["seed"]+fold*1000+inner*10+member
    atlas_seed = refseed+fold*1000+inner*10+member
    X, names, atlas, expected = feature_context(data, config, arm, train, reference, query, atlas_seed)
    selector = prepare_selector(data, config, X, names, arm, train, query, k, seed, root, memo, verify)
    expected.update(study_fingerprint=scope["study_fingerprint"], arm=arm, reference_seed=refseed,
                    fold=fold, inner=inner, member=member, k=k, seed=seed, selector=selector.audit(),
                    selected_names=[names[i] for i in selector.selected],
                    raw_training_context_sha256=array_hash(X[train][:, selector.selected]))
    prefix = f"{'outer' if inner == 0 else 'inner'}_{fold:02d}_{arm}_s{refseed}_i{inner}_k{k}_m{member}"
    path = root/f"{prefix}.json"
    if verify or path.exists():
        audit, p = read_result(path, expected, query, len(data["source"]["classes"]), outer=inner == 0)
    else:
        state = path.with_suffix(".tabpfn_fit") if inner == 0 else None
        p, actual = backend.run_fit(data["pfn"], seed, X[train][:, selector.selected], data["y"][train], X[query][:, selector.selected], state)
        previous.check_backend(actual, selector, len(data["source"]["classes"]), data["pfn"], seed)
        check_p(p, len(query), len(data["source"]["classes"]))
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, indices=query, probabilities=p)
        temporary.replace(path.with_suffix(".npz"))
        audit = dict(expected, backend=actual, probabilities_sha256=file_hash(path.with_suffix(".npz")))
        if inner == 0:
            temporary = path.with_suffix(".partial.joblib")
            joblib.dump({"selector": selector, "atlas": atlas, "names": names, "scope": expected}, temporary)
            temporary.replace(path.with_suffix(".joblib"))
            audit.update(model_sha256=file_hash(path.with_suffix(".joblib")), state_sha256=file_hash(state))
        _save_json(path, audit)
    previous.check_backend(audit["backend"], selector, len(data["source"]["classes"]), data["pfn"], seed)
    if verify and inner == 0:
        model = joblib.load(path.with_suffix(".joblib"))
        if model["scope"] != expected or model["names"] != names or model["selector"].audit() != selector.audit() or ((atlas is None) != (model["atlas"] is None)):
            raise ValueError("Restored deviation transformer changed")
        if atlas is not None and not np.array_equal(atlas, model["atlas"]):
            raise ValueError("Restored atlas changed")
        restored, actual = backend.run_predict(data["pfn"], path.with_suffix(".tabpfn_fit"), X[query][:, selector.selected])
        previous.check_backend(actual, selector, len(data["source"]["classes"]), data["pfn"], seed)
        if not np.array_equal(p.argmax(1), restored.argmax(1)) or not np.allclose(p, restored, rtol=1e-5, atol=1e-6):
            raise ValueError("Restored deviation predictions changed")
    return p, path


def ensemble(data, config, scope, root, arm, refseed, fold, inner, train, query, k, memo, verify=False):
    probabilities, paths = [], []
    for number, (fit, reference) in enumerate(members(data, config, arm, train, query, refseed, fold, inner), 1):
        p, path = member_fit(data, config, scope, root, arm, refseed, fold, inner, number, fit, reference, query, k, memo, verify)
        probabilities.append(p); paths.append(path)
        if not verify:
            _progress(root, state="member_complete", arm=arm, reference_seed=refseed, outer_fold=fold,
                      inner_fold=inner, feature_budget=k, member=number)
    return np.mean(probabilities, axis=0), paths


def evaluate_arm(data, config, scope, root, arm, refseed, fold, memo, verify=False):
    train, test, inner_splits = splits(data, config, fold)
    choices = []
    for k in config["feature_counts"]:
        scores, digests = [], []
        for inner, (fit, valid) in enumerate(inner_splits, 1):
            p, paths = ensemble(data, config, scope, root, arm, refseed, fold, inner, fit, valid, k, memo, verify)
            scores.append(float(f1_score(data["y"][valid], p.argmax(1), labels=np.arange(len(data["source"]["classes"])), average="macro", zero_division=0)))
            digests.append([file_hash(path) for path in paths])
        choices.append({"k": k, "mean_inner_macro_f1": float(np.mean(scores)), "inner_scores": scores, "member_audits_sha256": digests})
    # Stable configured-order ties; outer predictions never choose the budget.
    chosen = max(range(len(choices)), key=lambda i: choices[i]["mean_inner_macro_f1"])
    decision = {"study_fingerprint": scope["study_fingerprint"], "arm": arm, "reference_seed": refseed,
                "fold": fold, "choices": choices, "selected_k": choices[chosen]["k"]}
    selection_path = root/f"selection_{fold:02d}_{arm}_s{refseed}.json"
    if verify:
        if json.loads(selection_path.read_text()) != decision:
            raise ValueError("Deviation inner selection changed")
    else:
        _save_json(selection_path, decision)
    p, paths = ensemble(data, config, scope, root, arm, refseed, fold, 0, train, test, decision["selected_k"], memo, verify)
    metrics = metric_bundle(data["y"][test], p, data["source"]["classes"])
    result = dict(decision, member_audits_sha256=[file_hash(path) for path in paths], patient_metrics=metrics)
    result_path = root/f"fold_{fold:02d}_{arm}_s{refseed}.json"
    if verify:
        if json.loads(result_path.read_text()) != result:
            raise ValueError("Deviation outer ensemble changed")
    else:
        _save_json(result_path, result)
        print(f"{arm} reference seed {refseed}, outer {fold}/{data['source']['outer_splits']}: k={decision['selected_k']}; completed (partial scores are not study results)", flush=True)
    memo.clear()
    return test, p


def run_study(config):
    data = load_inputs(config)
    scope = make_scope(config, data)
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/"study_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != scope:
        raise ValueError("Deviation data/code/config changed; use a new output directory")
    _save_json(manifest, scope)
    for fold in range(1, data["source"]["outer_splits"]+1):
        evaluate_arm(data, config, scope, root, "full_morphology", config["seed"], fold, {})
        for refseed in config["reference_seeds"]:
            for arm in config["arms"][1:]:
                evaluate_arm(data, config, scope, root, arm, refseed, fold, {})
    _progress(root, state="fits_complete_report_pending")
    return root


def smoke(config):
    data = load_inputs(config)
    root = Path(config["output_dir"])/"smoke"
    root.mkdir(parents=True, exist_ok=True)
    _, _, inner = splits(data, config, 1)
    train, query = inner[0]
    fit, reference = members(data, config, "healthy", train, query, 42, 1, 1)[0]
    X, names, atlas, audit = feature_context(data, config, "healthy", fit, reference, query, 1053)
    selector = prepare_selector(data, config, X, names, "healthy", fit, query, 24, 1053, root, {})
    state = root/"smoke.tabpfn_fit"
    p, actual = backend.run_fit(data["pfn"], 1053, X[fit][:, selector.selected], data["y"][fit], X[query][:, selector.selected], state)
    previous.check_backend(actual, selector, len(data["source"]["classes"]), data["pfn"], 1053)
    check_p(p, len(query), len(data["source"]["classes"]))
    restored, _ = backend.run_predict(data["pfn"], state, X[query][:, selector.selected])
    if not np.array_equal(p.argmax(1), restored.argmax(1)) or not np.allclose(p, restored, rtol=1e-5, atol=1e-6):
        raise ValueError("Smoke context did not restore")
    _save_json(root/"summary.json", {"complete": True, "scope": "Finite local backend/membership smoke only; no score claim", "study_fingerprint": make_scope(config, data)["study_fingerprint"],
                "context": audit, "backend": actual, "selector": selector.audit(), "save_load_verified": True})
    print("Healthy-reference real-data smoke passed; saved context reproduces predictions.", flush=True)


def status(config):
    root = Path(config["output_dir"])/"complete"
    expected = 10*(1+3*3)
    print(f"Completed outer arm/seed evaluations: {len(list(root.glob('fold_*.json')))}/{expected}")
    print(f"Saved member contexts: {len(list(root.glob('inner_*.json')))+len(list(root.glob('outer_*.json')))}/1960")
    if (root/"progress.json").exists():
        print("Saved progress (not proof of a live process):", (root/"progress.json").read_text())
    print("Independently verified report:", (root/"REPORT.md").exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deviation_vcg.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        status(cfg)
    elif args.smoke:
        smoke(cfg)
    else:
        try:
            print(run_study(cfg))
        except Exception as error:
            _progress(Path(cfg["output_dir"])/"complete", state="failed", error_type=type(error).__name__, error=str(error))
            raise
