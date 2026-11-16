"""Bounded nested random-convolution feature study with identical ridge controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import scipy
from scipy.special import softmax
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import torch
from threadpoolctl import threadpool_limits
import yaml

from . import rocket_bank, rocket_features as features
from . import benchmark
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

CLAIM = "Bounded record-template random-convolution development on repeatedly inspected PTB patient folds; no independent confirmation, exact upstream reproduction or established MI SOTA."


def validate(config):
    if config["arms"] != list(features.ARMS) or not config["transform_seeds"] or len(set(config["transform_seeds"])) != len(config["transform_seeds"]):
        raise ValueError("Require all four arms and unique transform seeds")
    alphas = config["ridge_alphas"]
    if not alphas or len(set(alphas)) != len(alphas) or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Require fixed positive ridge candidates")
    for key in ["dct_coefficients", "multirocket_dilations", "hydra_groups", "hydra_kernels", "chunk_size", "torch_threads", "bootstrap_draws"]:
        if type(config[key]) != int or config[key] < 1: raise ValueError("Invalid feature budget")
    if config["multirocket_dilations"] > 32 or config["hydra_groups"] % 2 or config["outer_splits"] < 2 or config["inner_splits"] < 2:
        raise ValueError("Invalid bounded groups/folds/dilations")


def load_inputs(config):
    validate(config)
    bank = rocket_bank.prepare(config)
    if config["dct_coefficients"] > bank.templates.shape[-1]: raise ValueError("Too many DCT coefficients")
    if not Path(config["folds_path"]).is_file(): raise ValueError("Require existing frozen patient folds")
    folds = benchmark.load_or_create_folds(bank.labels, bank.ids, config)
    assigned = folds.set_index("patient_id").loc[bank.ids, "fold"].to_numpy()
    if sorted(set(assigned)) != list(range(1, config["outer_splits"]+1)): raise ValueError("Frozen fold count changed")
    return bank, assigned


def make_scope(config, bank):
    source_files = [__file__, features.__file__, rocket_bank.__file__]
    # Explicit paths avoid reliance on package layout or importing a second copy.
    import inspect
    code = {Path(p).name: file_hash(p) for p in source_files+[inspect.getfile(features.HydraMultivariate),
            Path(__file__).with_name("rocket_report.py"), benchmark.__file__]}
    from . import residual_context_report as uncertainty
    code["bootstrap_code"] = file_hash(uncertainty.__file__)
    scope = {"config": config, "bank": bank.provenance, "folds_sha256": file_hash(config["folds_path"]),
             "code": code, "claim_scope": CLAIM,
             "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                         "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu", "blas_threads": 1}}
    return dict(scope, study_fingerprint=_hash(scope))


def budget(config):
    groups = config["outer_splits"]*len(config["transform_seeds"])
    return {"feature_contexts": groups*(config["inner_splits"]+1),
            "fits": groups*len(config["arms"])*(config["inner_splits"]*len(config["ridge_alphas"])+1),
            "outer_models": groups*len(config["arms"])}


def dump_model(path, value):
    temporary = path.with_suffix(".partial.joblib")
    joblib.dump(value, temporary)
    temporary.replace(path)


def expected_membership(bank, scope, train, query):
    train, query = np.asarray(train), np.asarray(query)
    if (not len(train) or not len(query) or len(set(train)) != len(train) or len(set(query)) != len(query)
            or set(train) & set(query) or np.any(np.r_[train, query] < 0) or np.any(np.r_[train, query] >= len(bank.ids))):
        raise ValueError("Invalid/overlapping training/query patients")
    return {"study_fingerprint": scope["study_fingerprint"], "train_membership_sha256": array_hash(bank.ids[train]),
            "query_membership_sha256": array_hash(bank.ids[query]), "training_labels_sha256": array_hash(bank.labels[train]),
            "n_train": len(train), "n_query": len(query), "patient_overlap": 0}


def feature_context(bank, config, scope, root, seed, fold, inner, train, query, verify=False):
    expected = expected_membership(bank, scope, train, query)
    expected.update(transform_seed=seed, fit_seed=seed+fold*1000+inner, outer_fold=fold, inner_fold=inner)
    path = root/f"context_{fold:02d}_s{seed}_i{inner}.json"
    model_path, values_path = path.with_suffix(".joblib"), path.with_suffix(".npz")
    rows = np.r_[train, query]
    train_records = np.flatnonzero(np.isin(bank.record_patients, bank.ids[train]))
    selected_records = np.flatnonzero(np.isin(bank.record_patients, bank.ids[rows]))
    expected.update(train_records_sha256=array_hash(train_records), n_train_records=len(train_records))
    if verify or path.exists():
        audit = json.loads(path.read_text())
        if (any(audit.get(k) != v for k, v in expected.items()) or audit["transform_sha256"] != file_hash(model_path)
                or audit["features_sha256"] != file_hash(values_path)):
            raise ValueError("Convolutional feature context changed")
        transform = joblib.load(model_path)
        with np.load(values_path) as saved:
            if not np.array_equal(saved["rows"], rows): raise ValueError("Feature row order changed")
            values = {a: saved[a] for a in config["arms"]}
        if transform.audit() != audit["fit"]: raise ValueError("Saved convolutional transform changed")
        if verify:
            fresh = features.FeatureTransform(config, expected["fit_seed"]).fit(bank.templates[train_records], bank.record_patients[train_records])
            if fresh.audit() != audit["fit"]: raise ValueError("Transform is not reproducible from training patients")
            restored = features.aggregate(transform.transform(bank.templates[selected_records]), bank.record_patients[selected_records], bank.ids[rows])
            if any(not np.allclose(restored[a], values[a], rtol=1e-6, atol=1e-6) for a in config["arms"]):
                raise ValueError("Restored convolutional features disagree")
    else:
        _progress(root, state="extracting_features", transform_seed=seed, outer_fold=fold, inner_fold=inner)
        transform = features.FeatureTransform(config, expected["fit_seed"]).fit(bank.templates[train_records], bank.record_patients[train_records])
        values = features.aggregate(transform.transform(bank.templates[selected_records]), bank.record_patients[selected_records], bank.ids[rows])
        dump_model(model_path, transform)
        temporary = values_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, rows=rows, **values)
        temporary.replace(values_path)
        audit = dict(expected, fit=transform.audit(), transform_sha256=file_hash(model_path), features_sha256=file_hash(values_path),
                     dimensions={a: int(x.shape[1]) for a, x in values.items()})
        _save_json(path, audit)
    if any(x.shape[0] != len(rows) or not np.isfinite(x).all() for x in values.values()):
        raise ValueError("Invalid patient convolutional features")
    return values, path


def predict(model, x, classes):
    if not np.array_equal(model[-1].classes_, np.arange(classes)): raise ValueError("Ridge class order changed")
    scores = model.decision_function(x)
    if scores.shape != (len(x), classes) or not np.isfinite(scores).all(): raise ValueError("Invalid ridge scores")
    # Fixed score-to-probability mapping, not fitted/calibrated probabilities.
    return softmax(scores, axis=1)


def classifier_fit(bank, config, scope, root, seed, fold, inner, arm, candidate, train, query, values, context_path, verify=False):
    alpha = config["ridge_alphas"][candidate]
    expected = expected_membership(bank, scope, train, query)
    x = values[arm]
    expected.update(transform_seed=seed, outer_fold=fold, inner_fold=inner, arm=arm, candidate_id=candidate, alpha=alpha,
                    context_sha256=file_hash(context_path), training_features_sha256=array_hash(x[:len(train)]),
                    query_features_sha256=array_hash(x[len(train):]), n_features=x.shape[1])
    suffix = f"_i{inner}_a{candidate}" if inner else ""
    path = root/f"{'inner' if inner else 'outer'}_{fold:02d}_{arm}_s{seed}{suffix}.json"
    model_path, p_path = path.with_suffix(".joblib"), path.with_suffix(".npz")
    if verify or path.exists():
        audit = json.loads(path.read_text())
        if (any(audit.get(k) != v for k, v in expected.items()) or audit["model_sha256"] != file_hash(model_path)
                or audit["probabilities_sha256"] != file_hash(p_path)):
            raise ValueError("Ridge model/probability context changed")
        model = joblib.load(model_path)
        with np.load(p_path) as saved:
            if not np.array_equal(saved["rows"], query): raise ValueError("Ridge query membership changed")
            p = saved["probabilities"]
        if verify:
            scaler = StandardScaler().fit(x[:len(train)])
            if (not np.array_equal(scaler.mean_, model[0].mean_) or not np.array_equal(scaler.scale_, model[0].scale_)
                    or model[-1].alpha != alpha or model[-1].class_weight != "balanced" or model[-1].solver != "cholesky"):
                raise ValueError("Ridge training scaler/settings changed")
            if not np.allclose(predict(model, x[len(train):], len(config["classes"])), p, rtol=1e-10, atol=1e-12):
                raise ValueError("Restored ridge probabilities disagree")
    else:
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, class_weight="balanced", solver="cholesky"))
        with threadpool_limits(limits=1): model.fit(x[:len(train)], bank.labels[train])
        p = predict(model, x[len(train):], len(config["classes"]))
        dump_model(model_path, model)
        temporary = p_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, rows=query, probabilities=p)
        temporary.replace(p_path)
        audit = dict(expected, model_sha256=file_hash(model_path), probabilities_sha256=file_hash(p_path))
        if inner:
            audit["validation_macro_f1"] = float(f1_score(bank.labels[query], p.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
        _save_json(path, audit)
    if p.shape != (len(query), len(config["classes"])) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1):
        raise ValueError("Invalid saved ridge probabilities")
    if inner:
        score = float(f1_score(bank.labels[query], p.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
        if audit["validation_macro_f1"] != score: raise ValueError("Ridge inner score changed")
    _progress(root, state="verifying" if verify else "fitting_classifiers", arm=arm, transform_seed=seed, outer_fold=fold, inner_fold=inner, candidate_id=candidate)
    return p, audit


def evaluate_fold(bank, assigned, config, scope, root, seed, fold, verify=False):
    train, test = np.flatnonzero(assigned != fold), np.flatnonzero(assigned == fold)
    splits = [(train[t], train[v]) for t, v in benchmark.inner_patient_splits(bank.labels[train], bank.ids[train], config["inner_splits"], config["split_seed"]+fold*1000)]
    choices = {a: [[] for _ in config["ridge_alphas"]] for a in config["arms"]}
    for inner, (ti, vi) in enumerate(splits, 1):
        values, context_path = feature_context(bank, config, scope, root, seed, fold, inner, ti, vi, verify)
        for arm in config["arms"]:
            for candidate in range(len(config["ridge_alphas"])):
                _, audit = classifier_fit(bank, config, scope, root, seed, fold, inner, arm, candidate, ti, vi, values, context_path, verify)
                choices[arm][candidate].append(audit["validation_macro_f1"])
    values, context_path = feature_context(bank, config, scope, root, seed, fold, 0, train, test, verify)
    result = {}
    for arm in config["arms"]:
        means = [float(np.mean(x)) for x in choices[arm]]
        selected = max(range(len(means)), key=means.__getitem__)
        decision = dict(study_fingerprint=scope["study_fingerprint"], arm=arm, transform_seed=seed, outer_fold=fold,
                        inner_scores=choices[arm], mean_inner_macro_f1=means, selected_candidate=selected,
                        selected_alpha=config["ridge_alphas"][selected], tie_rule="first configured alpha")
        path = root/f"selection_{fold:02d}_{arm}_s{seed}.json"
        if verify or path.exists():
            if json.loads(path.read_text()) != decision: raise ValueError("Ridge nested selection changed")
        else: _save_json(path, decision)
        result[arm], _ = classifier_fit(bank, config, scope, root, seed, fold, 0, arm, selected, train, test, values, context_path, verify)
    return test, result


def run(config):
    bank, assigned = load_inputs(config)
    scope = make_scope(config, bank)
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True, exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope: raise ValueError("Random-convolution scope changed; use a new namespace")
    _save_json(guard, scope)
    for fold in range(1, config["outer_splits"]+1):
        for seed in config["transform_seeds"]:
            evaluate_fold(bank, assigned, config, scope, root, seed, fold)
            print(f"Random-convolution seed {seed}, fold {fold}/{config['outer_splits']}: fits ready; not final performance", flush=True)
    _progress(root, state="awaiting_independent_verification", **budget(config))


def smoke(config):
    bank, assigned = load_inputs(config)
    scope = make_scope(config, bank)
    root = Path(config["output_dir"])/"smoke"
    root.mkdir(parents=True, exist_ok=True)
    train, query = np.flatnonzero(assigned != 1), np.flatnonzero(assigned == 1)
    values, context = feature_context(bank, config, scope, root, config["transform_seeds"][0], 1, 0, train, query)
    for arm in config["arms"]:
        classifier_fit(bank, config, scope, root, config["transform_seeds"][0], 1, 0, arm, 0, train, query, values, context)
        classifier_fit(bank, config, scope, root, config["transform_seeds"][0], 1, 0, arm, 0, train, query, values, context, True)
    restored, _ = feature_context(bank, config, scope, root, config["transform_seeds"][0], 1, 0, train, query, True)
    if any(not np.array_equal(values[a], restored[a]) for a in config["arms"]): raise ValueError("Feature smoke restoration changed")
    summary = dict(complete=True, study_fingerprint=scope["study_fingerprint"], scope=scope, arms=config["arms"],
                   dimensions={a: x.shape[1] for a, x in values.items()}, transform_restored=True, classifier_restorations=4,
                   claim="Actual-data numerical/context/restore smoke only; no reported classification score")
    _save_json(root/"summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rocket_vcg_v1.yaml")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config["torch_threads"])
    try:
        with threadpool_limits(limits=1):
            if args.smoke:
                s = smoke(config)
                print(json.dumps({k: s[k] for k in ["complete", "dimensions", "transform_restored", "classifier_restorations"]}))
            else: run(config)
    except Exception as error:
        target = Path(config["output_dir"])/("smoke" if args.smoke else "complete")
        target.mkdir(parents=True, exist_ok=True)
        _progress(target, state="failed", error=str(error))
        raise


if __name__ == "__main__": main()
