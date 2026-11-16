"""Nested four-arm morphology distribution study, with immutable fit artifacts."""
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
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import benchmark, distribution_bank, distribution_features as features
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

CLAIM = "Prespecified measured-VCG distribution-pooling development on repeatedly inspected PTB patients; no independent confirmation, physiological distribution model or established SOTA."


def validate(config):
    if config["arms"] != list(features.ARMS): raise ValueError("Require all four matched arms")
    for key in ("projection_seeds", "ridge_alphas"):
        x = config[key]
        if not x or len(set(x)) != len(x): raise ValueError("Require unique declared seeds/candidates")
    if any(type(s) != int or s < 0 for s in config["projection_seeds"]): raise ValueError("Invalid projection seed")
    if any(not np.isfinite(x) or x <= 0 for x in config["ridge_alphas"]): raise ValueError("Invalid ridge candidates")
    for key in ("dct_coefficients", "projections", "chunk_size", "torch_threads", "bootstrap_draws"):
        if type(config[key]) != int or config[key] < 1: raise ValueError("Invalid bounded budget")
    if config["outer_splits"] < 2 or config["inner_splits"] < 2: raise ValueError("Invalid nested folds")
    windows = config["windows"]
    if (not windows or len({tuple(w) for w in windows}) != len(windows)
            or any(len(w) != 2 or any(type(i) != int for i in w) or not 0 <= w[0] < w[1]
                   or w[1]-w[0] < config["dct_coefficients"] for w in windows)):
        raise ValueError("Invalid R-relative feature windows")
    q = np.asarray(config["quantiles"], dtype=float)
    if len(q) < 3 or not np.isfinite(q).all() or np.any(q <= 0) or np.any(q >= 1) or np.any(np.diff(q) <= 0):
        raise ValueError("Invalid fixed quantile grid")
    if not np.isfinite(config["feature_scale_floor"]) or config["feature_scale_floor"] <= 0:
        raise ValueError("Invalid constant-feature threshold")


def load_inputs(config, verify=False):
    validate(config)
    if not Path(config["folds_path"]).is_file(): raise ValueError("Require existing frozen patient folds")
    bank = distribution_bank.prepare(config, verify=verify)
    folds = benchmark.load_or_create_folds(bank.labels, bank.ids, config)
    assigned = folds.set_index("patient_id").loc[bank.ids, "fold"].to_numpy()
    if sorted(set(assigned)) != list(range(1, config["outer_splits"]+1)):
        raise ValueError("Frozen fold count changed")
    return bank, assigned


def make_scope(config, bank):
    from . import chronology_bank, regularization, neural, multiview_bank, signal_audit, residual_context_report
    modules = [features, distribution_bank, benchmark, chronology_bank, regularization,
               neural, multiview_bank, signal_audit, residual_context_report]
    files = [Path(m.__file__) for m in modules]+[Path(__file__), Path(__file__).with_name("distribution_report.py")]
    scope = dict(config=config, bank=bank.provenance, folds_sha256=file_hash(config["folds_path"]),
                 code={p.name: file_hash(p) for p in files}, claim_scope=CLAIM,
                 runtime=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                              joblib=joblib.__version__, torch=str(torch.__version__), platform=platform.platform(),
                              device="cpu", blas_threads=1))
    return dict(scope, study_fingerprint=_hash(scope))


def budget(config):
    groups = config["outer_splits"]*len(config["projection_seeds"])
    return dict(feature_contexts=groups*(config["inner_splits"]+1),
                fits=groups*len(config["arms"])*(config["inner_splits"]*len(config["ridge_alphas"])+1),
                outer_models=groups*len(config["arms"]))


def dump_model(path, value):
    temporary = path.with_suffix(".partial.joblib")
    joblib.dump(value, temporary); temporary.replace(path)


def membership(bank, scope, train, query):
    train, query = np.asarray(train), np.asarray(query)
    if (not len(train) or not len(query) or len(set(train)) != len(train) or len(set(query)) != len(query)
            or set(train) & set(query) or np.any(np.r_[train, query] < 0) or np.any(np.r_[train, query] >= len(bank.ids))):
        raise ValueError("Invalid/overlapping patient partitions")
    return dict(study_fingerprint=scope["study_fingerprint"], train_membership_sha256=array_hash(bank.ids[train]),
                query_membership_sha256=array_hash(bank.ids[query]), training_labels_sha256=array_hash(bank.labels[train]),
                n_train=len(train), n_query=len(query), patient_overlap=0)


def feature_context(bank, config, scope, root, seed, fold, inner, train, query, verify=False):
    expected = membership(bank, scope, train, query)
    expected.update(projection_seed=seed, outer_fold=fold, inner_fold=inner,
                    train_records_sha256=array_hash(bank.records_for(bank.ids[train])),
                    query_records_sha256=array_hash(bank.records_for(bank.ids[query])))
    context_dir = root/"review/contexts"; context_dir.mkdir(parents=True, exist_ok=True)
    path = context_dir/f"context_{fold:02d}_s{seed}_i{inner}.json"
    model_path, values_path = path.with_suffix(".joblib"), path.with_suffix(".npz")
    rows = np.r_[train, query]
    if verify or path.exists():
        audit = json.loads(path.read_text())
        if (any(audit.get(k) != v for k, v in expected.items()) or audit["transform_sha256"] != file_hash(model_path)
                or audit["features_sha256"] != file_hash(values_path)):
            raise ValueError("Distribution feature context changed")
        transform = joblib.load(model_path)
        with np.load(values_path) as saved:
            if not np.array_equal(saved["rows"], rows): raise ValueError("Feature row order changed")
            values = {a: saved[a] for a in config["arms"]}
        if transform.audit() != audit["fit"]: raise ValueError("Saved training-only scaling/projections changed")
        if verify:
            fresh = features.DistributionTransform(config, seed).fit(bank, bank.ids[train])
            if fresh.audit() != audit["fit"]: raise ValueError("Transform not reproducible from exact training patients")
            restored = transform.transform(bank, bank.ids[rows])
            rebuilt = fresh.transform(bank, bank.ids[rows])
            if any(not np.allclose(v[a], values[a], rtol=1e-10, atol=1e-12) for v in (restored, rebuilt) for a in config["arms"]):
                raise ValueError("Restored/reconstructed distribution tensor disagrees")
    else:
        _progress(root, state="extracting_features", projection_seed=seed, outer_fold=fold, inner_fold=inner)
        transform = features.DistributionTransform(config, seed).fit(bank, bank.ids[train])
        values = transform.transform(bank, bank.ids[rows])
        dump_model(model_path, transform)
        temporary = values_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, rows=rows, **values); temporary.replace(values_path)
        audit = dict(expected, fit=transform.audit(), transform_sha256=file_hash(model_path), features_sha256=file_hash(values_path),
                     dimensions={a: int(v.shape[1]) for a, v in values.items()})
        _save_json(path, audit)
    size = len(config["windows"])*config["projections"]*len(config["quantiles"])
    if any(x.shape != (len(rows), size) or not np.isfinite(x).all() for x in values.values()):
        raise ValueError("Invalid/mismatched tensor dimensions")
    return values, path


def predict(model, x, classes):
    if not np.array_equal(model.classes_, np.arange(classes)): raise ValueError("Class order changed")
    scores = model.decision_function(x)
    if scores.shape != (len(x), classes) or not np.isfinite(scores).all(): raise ValueError("Invalid ridge logits")
    return softmax(scores, axis=1)


def classifier_fit(bank, config, scope, root, seed, fold, inner, arm, candidate, train, query, values, context, verify=False):
    x = values[arm]; alpha = config["ridge_alphas"][candidate]
    expected = membership(bank, scope, train, query)
    expected.update(projection_seed=seed, outer_fold=fold, inner_fold=inner, arm=arm, candidate_id=candidate, alpha=alpha,
                    context_sha256=file_hash(context), training_features_sha256=array_hash(x[:len(train)]),
                    query_features_sha256=array_hash(x[len(train):]), n_features=x.shape[1],
                    classifier="class-balanced ridge/cholesky; intercept; no per-arm scaler")
    directory = root/"review/fits"; directory.mkdir(parents=True, exist_ok=True)
    suffix = f"_i{inner}_a{candidate}" if inner else ""
    path = directory/f"{'inner' if inner else 'outer'}_{fold:02d}_{arm}_s{seed}{suffix}.json"
    model_path, p_path = path.with_suffix(".joblib"), path.with_suffix(".npz")
    if verify or path.exists():
        audit = json.loads(path.read_text())
        if (any(audit.get(k) != v for k, v in expected.items()) or audit["model_sha256"] != file_hash(model_path)
                or audit["probabilities_sha256"] != file_hash(p_path)):
            raise ValueError("Ridge context/model/probability hashes changed")
        model = joblib.load(model_path)
        with np.load(p_path) as saved:
            if not np.array_equal(saved["rows"], query): raise ValueError("Query rows changed")
            p = saved["probabilities"]
        if verify:
            if model.alpha != alpha or model.class_weight != "balanced" or model.solver != "cholesky" or model.fit_intercept is not True:
                raise ValueError("Saved ridge settings changed")
            if not np.allclose(predict(model, x[len(train):], len(config["classes"])), p, rtol=1e-10, atol=1e-12):
                raise ValueError("Restored classifier predictions disagree")
    else:
        model = RidgeClassifier(alpha=alpha, class_weight="balanced", solver="cholesky", fit_intercept=True)
        with threadpool_limits(limits=1): model.fit(x[:len(train)], bank.labels[train])
        p = predict(model, x[len(train):], len(config["classes"]))
        dump_model(model_path, model)
        temporary = p_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, rows=query, probabilities=p); temporary.replace(p_path)
        audit = dict(expected, model_sha256=file_hash(model_path), probabilities_sha256=file_hash(p_path))
        if inner: audit["validation_macro_f1"] = float(f1_score(bank.labels[query], p.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
        _save_json(path, audit)
    if p.shape != (len(query), len(config["classes"])) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1):
        raise ValueError("Invalid probabilities")
    if inner:
        score = float(f1_score(bank.labels[query], p.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
        if audit["validation_macro_f1"] != score: raise ValueError("Saved inner F1 changed")
    _progress(root, state="verifying" if verify else "fitting_classifiers", projection_seed=seed, outer_fold=fold,
              inner_fold=inner, arm=arm, candidate_id=candidate)
    return p, audit


def select(scores):
    means = [float(np.mean(x)) for x in scores]
    return max(range(len(means)), key=means.__getitem__), means


def evaluate_fold(bank, assigned, config, scope, root, seed, fold, verify=False):
    train, test = np.flatnonzero(assigned != fold), np.flatnonzero(assigned == fold)
    splits = [(train[t], train[v]) for t, v in benchmark.inner_patient_splits(bank.labels[train], bank.ids[train], config["inner_splits"], config["split_seed"]+fold*1000)]
    scores = {a: [[] for _ in config["ridge_alphas"]] for a in config["arms"]}
    for inner, (ti, vi) in enumerate(splits, 1):
        values, context = feature_context(bank, config, scope, root, seed, fold, inner, ti, vi, verify)
        for arm in config["arms"]:
            for candidate in range(len(config["ridge_alphas"])):
                _, audit = classifier_fit(bank, config, scope, root, seed, fold, inner, arm, candidate, ti, vi, values, context, verify)
                scores[arm][candidate].append(audit["validation_macro_f1"])
    values, context = feature_context(bank, config, scope, root, seed, fold, 0, train, test, verify)
    result = {}
    for arm in config["arms"]:
        candidate, means = select(scores[arm])
        decision = dict(study_fingerprint=scope["study_fingerprint"], arm=arm, projection_seed=seed, outer_fold=fold,
                        inner_scores=scores[arm], mean_inner_macro_f1=means, selected_candidate=candidate,
                        selected_alpha=config["ridge_alphas"][candidate], tie_rule="first configured alpha")
        path = root/"review"/f"selection_{fold:02d}_{arm}_s{seed}.json"
        if verify or path.exists():
            if json.loads(path.read_text()) != decision: raise ValueError("Training-only selection changed")
        else: _save_json(path, decision)
        result[arm], _ = classifier_fit(bank, config, scope, root, seed, fold, 0, arm, candidate, train, test, values, context, verify)
    return test, result


def run(config):
    bank, assigned = load_inputs(config)
    scope = make_scope(config, bank)
    root = Path(config["output_dir"])/"complete"; root.mkdir(parents=True, exist_ok=True)
    path = root/"study_manifest.json"
    if path.exists() and json.loads(path.read_text()) != scope: raise ValueError("Distribution study scope changed; use a new namespace")
    _save_json(path, scope)
    for fold in range(1, config["outer_splits"]+1):
        for seed in config["projection_seeds"]:
            evaluate_fold(bank, assigned, config, scope, root, seed, fold)
            print(f"Distribution seed {seed}, fold {fold}/{config['outer_splits']}: saved; not final performance", flush=True)
    _progress(root, state="awaiting_independent_verification", **budget(config))


def smoke(config):
    bank, assigned = load_inputs(config)
    scope = make_scope(config, bank)
    root = Path(config["output_dir"])/"smoke"; root.mkdir(parents=True, exist_ok=True)
    outer_train = np.flatnonzero(assigned != 1)
    ti, vi = next(benchmark.inner_patient_splits(bank.labels[outer_train], bank.ids[outer_train], config["inner_splits"], config["split_seed"]+1000))
    train, query = outer_train[ti], outer_train[vi]
    seed = config["projection_seeds"][0]
    values, context = feature_context(bank, config, scope, root, seed, 1, 1, train, query)
    for arm in config["arms"]:
        classifier_fit(bank, config, scope, root, seed, 1, 1, arm, 0, train, query, values, context)
        classifier_fit(bank, config, scope, root, seed, 1, 1, arm, 0, train, query, values, context, True)
    feature_context(bank, config, scope, root, seed, 1, 1, train, query, True)
    result = dict(complete=True, study_fingerprint=scope["study_fingerprint"], four_classifiers_restored=True,
                  exact_training_transform_reconstructed=True, outer_test_patients_used=False,
                  dimensions={a: x.shape[1] for a, x in values.items()}, claim="Actual-data inner-partition smoke only; not final classification performance")
    _save_json(root/"summary.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", default="configs/distribution_pooling_v1.yaml"); p.add_argument("--smoke", action="store_true")
    args = p.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); torch.set_num_threads(config["torch_threads"])
    try:
        with threadpool_limits(limits=1): result = smoke(config) if args.smoke else run(config)
        if result: print(json.dumps(result))
    except Exception as error:
        _progress(Path(config["output_dir"])/("smoke" if args.smoke else "complete"), state="failed", error=str(error)); raise
