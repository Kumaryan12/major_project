"""Training-only bootstrap selection and local PFN on pinned convolutional features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import rocket_study as source_study, feature_stability as selection
from . import tabpfn_backend as backend, tabpfn_study as pfn_checks
from .benchmark import inner_patient_splits, patient_balanced_weights
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

CLAIM = "Feature-selected ridge versus pinned local TabPFN-v2 on repeatedly inspected PTB convolutional features; adaptive development, no independent confirmation or established MI SOTA."


def budget(config, source):
    g = source["outer_splits"]*len(source["transform_seeds"])
    k, a, inner = len(config["feature_counts"]), len(config["ridge_alphas"]), source["inner_splits"]
    ridge, pfn = g*(inner*k*a+k), g*(inner*k+k)
    return dict(fits=ridge+pfn, ridge_fits=ridge, tabpfn_fits=pfn, outer_models=g*k*2,
                source_feature_contexts=g*(inner+1), source_ridge_contexts=g*(inner*a+1),
                selector_contexts=g*(inner+1)*k, reference_outer_models=g)


def validate(config, source):
    counts = config["feature_counts"]
    if (len(counts) != 2 or len(set(counts)) != 2 or any(type(k) != int or k < 1 for k in counts)
            or config["ridge_alphas"] != source["ridge_alphas"] or config["seed"] != source["seed"]
            or config["split_seed"] != source["split_seed"] or not 0 < config["correlation_threshold"] <= 1):
        raise ValueError("Require fixed source folds/seeds, two budgets and matching ridge alphas")
    for key in ["bootstrap_replicates", "audit_replicates", "selector_seed_offset", "bootstrap_draws"]:
        if type(config[key]) != int or config[key] < 1: raise ValueError("Invalid bounded selector budget")


def source_artifacts(root):
    paths = [root/p for p in ["study_manifest.json", "summary.json", "REPORT.md"]]
    for path in root.glob("context_*.json"):
        paths.extend([path, path.with_suffix(".npz"), path.with_suffix(".joblib")])
    for pattern in ["inner_*_combined_*.json", "outer_*_combined_*.json"]:
        for path in root.glob(pattern): paths.extend([path, path.with_suffix(".npz"), path.with_suffix(".joblib")])
    paths.extend(root.glob("selection_*_combined_*.json"))
    return {str(p): file_hash(p) for p in sorted(paths)}


def load_inputs(config):
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    validate(config, source)
    bank, assigned = source_study.load_inputs(source)
    original_scope = source_study.make_scope(source, bank)
    root = Path(source["output_dir"])/"complete"
    summary = json.loads((root/"summary.json").read_text())
    expected = source_study.budget(source)
    if (json.loads((root/"study_manifest.json").read_text()) != original_scope or summary.get("complete") is not True
            or summary.get("study_fingerprint") != original_scope["study_fingerprint"]
            or summary.get("all_models_restored") != expected["fits"]
            or summary.get("feature_contexts_reconstructed") != expected["feature_contexts"]
            or summary.get("outer_models_restored") != expected["outer_models"]):
        raise ValueError("Require completed unchanged convolutional source study")
    pfn = yaml.safe_load(Path(config["tabpfn_config"]).read_text())
    data = dict(source=source, bank=bank, assigned=assigned, original_scope=original_scope, root=root, pfn=pfn)
    artifacts = source_artifacts(root)
    counts = budget(config, source)
    expected_files = 3*(counts["source_feature_contexts"]+counts["source_ridge_contexts"])+counts["reference_outer_models"]+3
    if len(artifacts) != expected_files: raise ValueError("Incomplete or extra pinned source context files")
    data["provenance"] = dict(source_scope=original_scope, source_summary_sha256=file_hash(root/"summary.json"),
        source_artifacts_sha256=_hash(artifacts), source_artifacts_count=len(artifacts), pfn_config=pfn,
        checkpoint=backend.prepare_checkpoint(pfn), backend_runtime=backend.runtime_info())
    return data


def make_scope(config, data):
    files = [__file__, Path(__file__).with_name("rocket_tabpfn_report.py"), selection.__file__, selection.original.__file__,
             backend.__file__, pfn_checks.__file__]
    from . import residual_context_report, benchmark
    files += [residual_context_report.__file__, benchmark.__file__]
    scope = dict(config=config, source=data["provenance"], claim_scope=CLAIM,
                 code={Path(p).name: file_hash(p) for p in files})
    return dict(scope, study_fingerprint=_hash(scope))


def unchanged_source(data):
    artifacts = source_artifacts(data["root"])
    if (_hash(artifacts) != data["provenance"]["source_artifacts_sha256"]
            or len(artifacts) != data["provenance"]["source_artifacts_count"]):
        raise ValueError("Pinned source artifacts changed")


def get_features(data, seed, fold, inner, train, query, verify=False):
    path = data["root"]/f"context_{fold:02d}_s{seed}_i{inner}.json"
    if not all(path.with_suffix(s).is_file() for s in [".json", ".npz", ".joblib"]):
        raise ValueError("Missing source feature partition; do not fit an outer transform for it")
    values, path = source_study.feature_context(data["bank"], data["source"], data["original_scope"], data["root"],
                                               seed, fold, inner, train, query, verify=verify)
    return values["combined"], path


def prepare(data, config, scope, root, seed, fold, inner, train, query, x, source_path, memo, verify=False):
    bank = data["bank"]
    members = source_study.expected_membership(bank, scope, train, query)
    if x.shape[0] != len(train)+len(query) or not np.isfinite(x).all(): raise ValueError("Invalid scoped source features")
    full = np.zeros((len(bank.ids), x.shape[1]))
    full[np.r_[train, query]] = x
    names = [f"convolution/column_{i:04d}" for i in range(x.shape[1])]
    selector_seed = config["selector_seed_offset"]+seed+fold*1000+inner*10
    settings = dict(config, _selector_code_sha256=file_hash(selection.__file__))
    payload = selection.bootstrap_cache(full, bank.labels, bank.ids, train, names, "convolution", settings,
        config, selector_seed, root, memo, verify=verify)
    weights = patient_balanced_weights(bank.labels[train], bank.ids[train])
    result = {}
    for k in config["feature_counts"]:
        selector = selection.ConsensusSelector().fit(x[:len(train)], bank.labels[train], weights, k,
            config["correlation_threshold"], names, "convolution", "bootstrap", payload)
        if not 0 < len(selector.selected) <= k: raise ValueError("Invalid selected feature count")
        expected = dict(members, transform_seed=seed, outer_fold=fold, inner_fold=inner, feature_budget=k,
            selector_seed=selector_seed, source_context_sha256=file_hash(source_path), selector=selector.audit(),
            training_features_sha256=array_hash(x[:len(train)]), query_features_sha256=array_hash(x[len(train):]),
            selected_training_features_sha256=array_hash(x[:len(train), selector.selected]),
            selected_query_features_sha256=array_hash(x[len(train):, selector.selected]))
        path = root/f"selector_{fold:02d}_s{seed}_i{inner}_k{k}.json"
        model_path = path.with_suffix(".joblib")
        if verify or path.exists():
            audit = json.loads(path.read_text())
            if any(audit.get(a) != b for a, b in expected.items()) or audit["selector_model_sha256"] != file_hash(model_path):
                raise ValueError("Scoped bootstrap selector changed")
            saved = joblib.load(model_path)
            if saved.audit() != selector.audit(): raise ValueError("Restored bootstrap selector changed")
        else:
            source_study.dump_model(model_path, selector)
            _save_json(path, dict(expected, selector_model_sha256=file_hash(model_path)))
        result[k] = dict(selector=selector, audit=expected, path=path,
                         train=x[:len(train), selector.selected], query=x[len(train):, selector.selected])
    return result


def check_p(p, n, classes):
    if (p.shape != (n, classes) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(1), 1, atol=1e-6)):
        raise ValueError("Invalid scoped classifier probabilities")


def score(data, query, p):
    return float(f1_score(data["bank"].labels[query], p.argmax(1), labels=np.arange(len(data["source"]["classes"])), average="macro", zero_division=0))


def new_fit(data, config, scope, root, seed, fold, inner, train, query, learner, k, candidate, prepared, verify=False):
    if learner not in ["ridge", "tabpfn"] or (learner == "tabpfn" and candidate != 0):
        raise ValueError("Invalid declared classifier candidate")
    context = prepared[k]
    expected = dict(context["audit"], learner=learner, candidate_id=candidate,
        alpha=config["ridge_alphas"][candidate] if learner == "ridge" else None,
        selector_audit_sha256=file_hash(context["path"]), inference_seed=config["seed"]+fold*1000+inner)
    path = root/f"{'inner' if inner else 'outer'}_{fold:02d}_{learner}_s{seed}_i{inner}_k{k}_a{candidate}.json"
    model_path = path.with_suffix(".joblib" if learner == "ridge" else ".tabpfn_fit")
    p_path = path.with_suffix(".npz")
    nclasses = len(data["source"]["classes"])
    if verify or path.exists():
        audit = json.loads(path.read_text())
        if (any(audit.get(a) != b for a, b in expected.items()) or audit["model_sha256"] != file_hash(model_path)
                or audit["probabilities_sha256"] != file_hash(p_path)):
            raise ValueError("Selected classifier context/model changed")
        with np.load(p_path) as saved:
            if not np.array_equal(saved["rows"], query): raise ValueError("Selected classifier query membership changed")
            p = saved["probabilities"]
        if learner == "tabpfn": pfn_checks.check_backend(audit["backend"], context["selector"], nclasses, data["pfn"], expected["inference_seed"])
        if verify:
            if learner == "ridge":
                model = joblib.load(model_path)
                scaler = StandardScaler().fit(context["train"])
                if (not np.array_equal(model[0].mean_, scaler.mean_) or not np.array_equal(model[0].scale_, scaler.scale_)
                        or model[-1].alpha != expected["alpha"] or model[-1].class_weight != "balanced" or model[-1].solver != "cholesky"):
                    raise ValueError("Selected ridge training scaler/settings changed")
                restored = source_study.predict(model, context["query"], nclasses)
            else:
                restored, actual = backend.run_predict(data["pfn"], model_path, context["query"])
                pfn_checks.check_backend(actual, context["selector"], nclasses, data["pfn"], expected["inference_seed"])
            if not np.array_equal(p.argmax(1), restored.argmax(1)) or not np.allclose(p, restored, rtol=1e-5, atol=1e-6):
                raise ValueError("Restored selected classifier probabilities disagree")
    else:
        _progress(root, state="fitting_classifier", learner=learner, transform_seed=seed, outer_fold=fold,
                  inner_fold=inner, feature_budget=k, candidate_id=candidate)
        extra = {}
        if learner == "ridge":
            model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=expected["alpha"], class_weight="balanced", solver="cholesky"))
            with threadpool_limits(limits=1): model.fit(context["train"], data["bank"].labels[train])
            p = source_study.predict(model, context["query"], nclasses)
            source_study.dump_model(model_path, model)
        else:
            p, actual = backend.run_fit(data["pfn"], expected["inference_seed"], context["train"], data["bank"].labels[train], context["query"], model_path)
            pfn_checks.check_backend(actual, context["selector"], nclasses, data["pfn"], expected["inference_seed"])
            extra["backend"] = actual
        check_p(p, len(query), nclasses)
        temporary = p_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, rows=query, probabilities=p)
        temporary.replace(p_path)
        audit = dict(expected, **extra, model_sha256=file_hash(model_path), probabilities_sha256=file_hash(p_path))
        if inner: audit["validation_macro_f1"] = score(data, query, p)
        _save_json(path, audit)
    check_p(p, len(query), nclasses)
    if inner and audit["validation_macro_f1"] != score(data, query, p): raise ValueError("Selected classifier inner score changed")
    _progress(root, state="verifying" if verify else "classifier_complete", learner=learner, transform_seed=seed,
              outer_fold=fold, inner_fold=inner, feature_budget=k, candidate_id=candidate)
    return p, audit, path


def read_reference(data, seed, fold, inner, candidate, train, query, x, context_path, verify=False):
    """Read original combined-ridge fits without writing to the pinned source."""
    expected = source_study.expected_membership(data["bank"], data["original_scope"], train, query)
    expected.update(transform_seed=seed, outer_fold=fold, inner_fold=inner, arm="combined", candidate_id=candidate,
        alpha=data["source"]["ridge_alphas"][candidate], context_sha256=file_hash(context_path),
        training_features_sha256=array_hash(x[:len(train)]), query_features_sha256=array_hash(x[len(train):]), n_features=x.shape[1])
    suffix = f"_i{inner}_a{candidate}" if inner else ""
    path = data["root"]/f"{'inner' if inner else 'outer'}_{fold:02d}_combined_s{seed}{suffix}.json"
    audit = json.loads(path.read_text())
    if (any(audit.get(k) != v for k, v in expected.items()) or audit["model_sha256"] != file_hash(path.with_suffix(".joblib"))
            or audit["probabilities_sha256"] != file_hash(path.with_suffix(".npz"))):
        raise ValueError("Frozen full-ridge source context changed")
    with np.load(path.with_suffix(".npz")) as saved:
        if not np.array_equal(saved["rows"], query): raise ValueError("Frozen ridge query membership changed")
        p = saved["probabilities"]
    if inner and audit["validation_macro_f1"] != score(data, query, p): raise ValueError("Frozen ridge inner score changed")
    if verify:
        model = joblib.load(path.with_suffix(".joblib"))
        scaler = StandardScaler().fit(x[:len(train)])
        if (not np.array_equal(model[0].mean_, scaler.mean_) or not np.array_equal(model[0].scale_, scaler.scale_)
                or model[-1].alpha != expected["alpha"] or model[-1].class_weight != "balanced" or model[-1].solver != "cholesky"):
            raise ValueError("Frozen ridge scaler/settings changed")
        restored = source_study.predict(model, x[len(train):], len(data["source"]["classes"]))
        if not np.allclose(restored, p, rtol=1e-10, atol=1e-12): raise ValueError("Frozen ridge restoration disagrees")
    check_p(p, len(query), len(data["source"]["classes"]))
    return p, audit, path


def decide(rows):
    return max(range(len(rows)), key=lambda i: rows[i]["mean_inner_macro_f1"])


def evaluate_fold(data, config, scope, root, seed, fold, verify=False):
    bank, source = data["bank"], data["source"]
    train, test = np.flatnonzero(data["assigned"] != fold), np.flatnonzero(data["assigned"] == fold)
    splits = [(train[t], train[v]) for t, v in inner_patient_splits(bank.labels[train], bank.ids[train], source["inner_splits"], config["split_seed"]+fold*1000)]
    ridge_rows = [dict(k=k, candidate=a, scores=[]) for k in config["feature_counts"] for a in range(len(config["ridge_alphas"]))]
    pfn_rows = [dict(k=k, candidate=0, scores=[]) for k in config["feature_counts"]]
    reference_scores, reference_paths, memo = [[] for _ in config["ridge_alphas"]], [], {}
    for inner, (ti, vi) in enumerate(splits, 1):
        _progress(root, state="preparing_selectors", transform_seed=seed, outer_fold=fold, inner_fold=inner)
        x, context = get_features(data, seed, fold, inner, ti, vi, verify)
        prepared = prepare(data, config, scope, root, seed, fold, inner, ti, vi, x, context, memo, verify)
        for a in range(len(config["ridge_alphas"])):
            _, audit, path = read_reference(data, seed, fold, inner, a, ti, vi, x, context, verify)
            reference_scores[a].append(audit["validation_macro_f1"])
            reference_paths.append(path)
        for learner, rows in [("ridge", ridge_rows), ("tabpfn", pfn_rows)]:
            for row in rows:
                _, audit, _ = new_fit(data, config, scope, root, seed, fold, inner, ti, vi, learner, row["k"], row["candidate"], prepared, verify)
                row["scores"].append(audit["validation_macro_f1"])
    for row in ridge_rows+pfn_rows: row["mean_inner_macro_f1"] = float(np.mean(row["scores"]))
    best_ridge, best_pfn = decide(ridge_rows), decide(pfn_rows)
    per_budget = {k: decide([r for r in ridge_rows if r["k"] == k]) for k in config["feature_counts"]}
    decision = dict(study_fingerprint=scope["study_fingerprint"], transform_seed=seed, outer_fold=fold,
        ridge_candidates=ridge_rows, tabpfn_candidates=pfn_rows, selected_ridge=best_ridge, selected_tabpfn=best_pfn,
        ridge_alpha_per_budget={str(k): a for k, a in per_budget.items()}, tie_rule="configured budget then alpha order")
    selection_path = root/f"selection_{fold:02d}_s{seed}.json"
    if verify or selection_path.exists():
        if json.loads(selection_path.read_text()) != decision: raise ValueError("Nested classifier/budget decision changed")
    else: _save_json(selection_path, decision)
    x, context = get_features(data, seed, fold, 0, train, test, verify)
    prepared = prepare(data, config, scope, root, seed, fold, 0, train, test, x, context, memo, verify)
    result = {}
    for k in config["feature_counts"]:
        result[f"ridge_k{k}"], _, _ = new_fit(data, config, scope, root, seed, fold, 0, train, test, "ridge", k, per_budget[k], prepared, verify)
        result[f"tabpfn_k{k}"], _, _ = new_fit(data, config, scope, root, seed, fold, 0, train, test, "tabpfn", k, 0, prepared, verify)
    result["selected_ridge"] = result[f"ridge_k{ridge_rows[best_ridge]['k']}"]
    result["selected_tabpfn"] = result[f"tabpfn_k{pfn_rows[best_pfn]['k']}"]
    means = [float(np.mean(v)) for v in reference_scores]
    selected = max(range(len(means)), key=means.__getitem__)
    original = dict(study_fingerprint=data["original_scope"]["study_fingerprint"], arm="combined", transform_seed=seed, outer_fold=fold,
        inner_scores=reference_scores, mean_inner_macro_f1=means, selected_candidate=selected,
        selected_alpha=source["ridge_alphas"][selected], tie_rule="first configured alpha")
    old_decision = data["root"]/f"selection_{fold:02d}_combined_s{seed}.json"
    if json.loads(old_decision.read_text()) != original: raise ValueError("Frozen full-ridge inner selection changed")
    result["full_ridge"], _, path = read_reference(data, seed, fold, 0, selected, train, test, x, context, verify)
    reference_paths.extend([path, old_decision])
    reference_audit = dict(study_fingerprint=scope["study_fingerprint"], transform_seed=seed, outer_fold=fold,
                           source_audits={str(p): file_hash(p) for p in reference_paths})
    reference_path = root/f"reference_{fold:02d}_s{seed}.json"
    if verify or reference_path.exists():
        if json.loads(reference_path.read_text()) != reference_audit: raise ValueError("Frozen reference provenance changed")
    else: _save_json(reference_path, reference_audit)
    return test, result


def run(config):
    data = load_inputs(config)
    scope = make_scope(config, data)
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True, exist_ok=True)
    path = root/"study_manifest.json"
    if path.exists() and json.loads(path.read_text()) != scope: raise ValueError("Selected-convolution settings/source changed; use a new namespace")
    _save_json(path, scope)
    for fold in range(1, data["source"]["outer_splits"]+1):
        for seed in data["source"]["transform_seeds"]:
            evaluate_fold(data, config, scope, root, seed, fold)
            print(f"Selected-convolution seed {seed}, fold {fold}/{data['source']['outer_splits']}: fits ready; not final performance", flush=True)
    unchanged_source(data)
    _progress(root, state="awaiting_independent_verification", **budget(config, data["source"]))


def smoke(config):
    data = load_inputs(config)
    scope = make_scope(config, data)
    root = Path(config["output_dir"])/"smoke"
    root.mkdir(parents=True, exist_ok=True)
    seed, fold = data["source"]["transform_seeds"][0], 1
    train, query = np.flatnonzero(data["assigned"] != fold), np.flatnonzero(data["assigned"] == fold)
    x, context = get_features(data, seed, fold, 0, train, query, verify=True)
    prepared = prepare(data, config, scope, root, seed, fold, 0, train, query, x, context, {}, verify=False)
    prepare(data, config, scope, root, seed, fold, 0, train, query, x, context, {}, verify=True)
    for k in config["feature_counts"]:
        for learner in ["ridge", "tabpfn"]:
            new_fit(data, config, scope, root, seed, fold, 0, train, query, learner, k, 0, prepared)
            new_fit(data, config, scope, root, seed, fold, 0, train, query, learner, k, 0, prepared, verify=True)
    # Query rows must not fit preprocessing or influence each other's predictions.
    k = config["feature_counts"][-1]
    state = root/f"outer_{fold:02d}_tabpfn_s{seed}_i0_k{k}_a0.tabpfn_fit"
    original, _ = backend.run_predict(data["pfn"], state, prepared[k]["query"])
    single, _ = backend.run_predict(data["pfn"], state, prepared[k]["query"][:1])
    altered = prepared[k]["query"].copy()
    altered[1:] += 10000
    other, _ = backend.run_predict(data["pfn"], state, altered)
    if not np.allclose(single[0], original[0], rtol=1e-5, atol=1e-6) or not np.allclose(other[0], original[0], rtol=1e-5, atol=1e-6):
        raise ValueError("PFN query predictions depend on unrelated query rows")
    unchanged_source(data)
    result = dict(complete=True, study_fingerprint=scope["study_fingerprint"], selected_counts={str(k):len(p["selector"].selected) for k,p in prepared.items()},
                  classifier_restorations=4, bootstrap_cache_replayed=True, query_batch_invariance=True, source_unchanged=True,
                  claim="Real-data isolated PFN/selector/ridge restoration smoke; no classification result")
    _save_json(root/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rocket_tabpfn_v1.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=1):
            if args.smoke: print(json.dumps(smoke(config)))
            else: run(config)
    except Exception as error:
        root = Path(config["output_dir"])/("smoke" if args.smoke else "complete")
        root.mkdir(parents=True, exist_ok=True)
        _progress(root, state="failed", error=str(error))
        raise


if __name__ == "__main__": main()
