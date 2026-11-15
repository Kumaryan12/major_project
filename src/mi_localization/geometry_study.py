"""Small nested patient-level screen before training another neural tensor branch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import geometry_bank
from .benchmark import inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights, save_confusion
from .contrastive import _progress
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .preprocessing_study import directory
from .relational_report import paired_bootstrap
from .signal_audit import file_hash

VIEWS = ("statistics", "fixed_geometry", "adaptive_geometry", "statistics_geometry")
LEARNERS = ("linear", "tree")
PROCEDURES = tuple(f"{learner}_{view}" for learner in LEARNERS for view in VIEWS)
CLAIM_SCOPE = "VCG-only feature utility screen on repeatedly inspected PTB development patients; no independent confirmation, clinical boundary validation or SOTA."


def array_hash(value):
    value = np.asarray(value)
    return _hash([str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()])


def load_features(config):
    root = geometry_bank.prepare_bank(config)
    manifest = json.loads((root/"manifest.json").read_text())
    with np.load(root/"patient_features.npz") as saved:
        ids, labels = saved["patient_id"], saved["labels"]
        views = {view: saved[view] for view in VIEWS}
    if len(ids) != len(np.unique(ids)) or not np.array_equal(ids, np.sort(ids)) or any(x.ndim != 2 or len(x) != len(ids) or not np.isfinite(x).all() for x in views.values()):
        raise ValueError("Invalid geometry patient feature rows")
    mapping = {label: i for i, label in enumerate(config["classes"])}
    y = np.array([mapping[str(label)] for label in labels])
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require existing frozen patient folds")
    folds = load_or_create_folds(y, ids, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"]+1)):
        raise ValueError("Frozen geometry patient folds changed")
    provenance = {"bank_fingerprint": manifest["bank_fingerprint"], "manifest_sha256": file_hash(root/"manifest.json"),
        "patient_features_sha256": file_hash(root/"patient_features.npz"), "summary": manifest["summary"]}
    return views, y, ids, folds, provenance


def candidates(config, learner):
    if learner == "linear":
        return [{"C": float(value)} for value in config["linear_c"]]
    if learner == "tree":
        return [{"max_depth": int(value)} for value in config["tree_depths"]]
    raise ValueError("Unknown feature learner")


def validate_config(config):
    geometry_bank.validate_config(config)
    if len(config["linear_c"]) != 2 or len(set(config["linear_c"])) != 2 or not np.isfinite(config["linear_c"]).all() or min(config["linear_c"]) <= 0:
        raise ValueError("Require two distinct positive linear regularization choices")
    if len(config["tree_depths"]) != 2 or len(set(config["tree_depths"])) != 2 or any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in config["tree_depths"]):
        raise ValueError("Require two distinct positive tree depths")
    if config["tree_estimators"] < 1 or not np.isfinite(config["tree_learning_rate"]) or config["tree_learning_rate"] <= 0 or config["outer_splits"] < 2 or config["inner_splits"] < 2:
        raise ValueError("Invalid geometry screen budget")


def train_audit(X, y, ids, train, evaluate, learner, params, config, seed, role):
    if set(ids[train]) & set(ids[evaluate]):
        raise ValueError("Feature training/evaluation patient overlap")
    weights = patient_balanced_weights(y[train], ids[train])
    scale = StandardScaler().fit(X[train], sample_weight=weights)
    return {"learner": learner, "params": params, "seed": seed, "evaluation_role": role, "patient_overlap": 0,
        "train_patients": len(train), "evaluation_patients": len(evaluate), "n_features": X.shape[1],
        "train_patient_ids_sha256": _hash(ids[train].tolist()), "evaluation_patient_ids_sha256": _hash(ids[evaluate].tolist()),
        "training_features_sha256": array_hash(X[train]), "training_labels_sha256": array_hash(y[train]),
        "training_weights_sha256": array_hash(weights), "scaler_fit_patients": len(train),
        "scaler_mean_sha256": array_hash(scale.mean_), "scaler_scale_sha256": array_hash(scale.scale_)}


def fit_classifier(X, y, ids, train, evaluate, learner, params, config, seed, role):
    audit = train_audit(X, y, ids, train, evaluate, learner, params, config, seed, role)
    weights = patient_balanced_weights(y[train], ids[train])
    if learner == "linear":
        estimator = LogisticRegression(C=params["C"], solver="lbfgs", max_iter=5000, tol=1e-7, random_state=seed)
    else:
        estimator = XGBClassifier(max_depth=params["max_depth"], n_estimators=config["tree_estimators"], learning_rate=config["tree_learning_rate"],
            reg_lambda=1., subsample=1., colsample_bytree=1., objective="multi:softprob", num_class=len(config["classes"]),
            tree_method="hist", eval_metric="mlogloss", n_jobs=1, random_state=seed)
    pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    # One row is one patient. Apply class balancing once, without class_weight.
    # XGBoost metadata allocation can otherwise spawn a second OpenMP pool
    # beside the torch/sklearn runtimes on macOS. This small patient table
    # needs one CPU thread; constrain the global allocation context as well.
    with xgboost.config_context(nthread=1):
        pipeline.fit(X[train], y[train], scale__sample_weight=weights, model__sample_weight=weights)
    if not np.array_equal(pipeline.classes_, np.arange(len(config["classes"]))):
        raise ValueError("Missing fitted feature-class probability columns")
    p = predict_pipeline(pipeline, X[evaluate])
    validate_probabilities(p, len(evaluate), len(config["classes"]))
    if array_hash(pipeline.named_steps["scale"].mean_) != audit["scaler_mean_sha256"] or array_hash(pipeline.named_steps["scale"].scale_) != audit["scaler_scale_sha256"]:
        raise ValueError("Pipeline scaling differs from audited training-only fit")
    return pipeline, p, audit


def predict_pipeline(model, X):
    with xgboost.config_context(nthread=1):
        return model.predict_proba(X)


def validate_probabilities(p, n, classes):
    if p.shape != (n, classes) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid feature-screen probabilities")


def choose_candidate(rows, config, learner):
    grid = candidates(config, learner)
    if len(rows) != len(grid):
        raise ValueError("Missing feature candidates")
    means = []
    for row, params in zip(rows, grid):
        fits = row["fits"]
        values = [fit["validation_macro_f1"] for fit in fits]
        if row["params"] != params or len(fits) != config["inner_splits"] or not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
            raise ValueError("Invalid feature candidate validation")
        means.append(float(np.mean(values)))
    selected = max(range(len(means)), key=lambda index: means[index])
    return {"candidate_index": selected, "params": grid[selected], "mean_inner_macro_f1": means[selected], "candidate_means": means,
        "rule": "maximum mean inner patient macro F1; exact ties favor configured order; no outer selection"}


def run_study(config, outer_folds=None):
    validate_config(config)
    views, y, ids, folds, bank = load_features(config)
    root, chosen, partial = directory(config, outer_folds)
    scope = {"config": config, "bank": bank, "folds_sha256": file_hash(config["folds_path"]), "evaluated_folds": chosen,
        "code_sha256": file_hash(__file__), "numpy": np.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "claim_scope": CLAIM_SCOPE}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Geometry study data/settings/runtime changed; choose a new output_dir")
    _save_json(guard, scope)
    lookup = folds.set_index("patient_id").fold.to_dict()
    patient_folds = np.array([lookup[str(patient)] for patient in ids])
    summary_rows = []
    for learner in LEARNERS:
        for view in VIEWS:
            procedure = f"{learner}_{view}"
            X = views[view]
            probabilities = np.full((len(ids), len(config["classes"])), np.nan)
            audits = []
            for fold in chosen:
                train, test = np.flatnonzero(patient_folds != fold), np.flatnonzero(patient_folds == fold)
                checkpoint, audit_path = root/f"fold_{fold:02d}_{procedure}.npz", root/f"fold_{fold:02d}_{procedure}.json"
                if checkpoint.exists() and audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    if audit["study_fingerprint"] != fingerprint or audit["prediction_checkpoint_sha256"] != file_hash(checkpoint):
                        raise ValueError("Completed geometry checkpoint changed")
                    with np.load(checkpoint) as saved:
                        if not np.array_equal(saved["indices"], test):
                            raise ValueError("Geometry checkpoint patients changed")
                        probabilities[test] = saved["probabilities"]
                    validate_probabilities(probabilities[test], len(test), len(config["classes"]))
                    audits.append(audit)
                    continue
                splits = [(train[ti], train[vi]) for ti, vi in inner_patient_splits(y[train], ids[train], config["inner_splits"], config["split_seed"]+fold*1000)]
                selections = []
                for candidate, params in enumerate(candidates(config, learner)):
                    fits = []
                    for inner, (ti, vi) in enumerate(splits, 1):
                        stem = f"inner_{fold:02d}_{procedure}_{candidate}_{inner}"
                        cache, audit_cache = root/f"{stem}.npz", root/f"{stem}.json"
                        expected = train_audit(X, y, ids, ti, vi, learner, params, config, config["seed"]+fold*1000+inner, "inner_validation")
                        if cache.exists() and audit_cache.exists():
                            fit = json.loads(audit_cache.read_text())
                            if fit["study_fingerprint"] != fingerprint or any(fit.get(key) != value for key, value in expected.items()) or fit["prediction_checkpoint_sha256"] != file_hash(cache):
                                raise ValueError("Geometry inner checkpoint changed")
                            with np.load(cache) as saved:
                                if not np.array_equal(saved["indices"], vi):
                                    raise ValueError("Geometry inner validation membership changed")
                                p = saved["probabilities"]
                        else:
                            model, p, fit = fit_classifier(X, y, ids, ti, vi, learner, params, config, config["seed"]+fold*1000+inner, "inner_validation")
                            del model
                            temporary = cache.with_suffix(".partial.npz")
                            np.savez_compressed(temporary, indices=vi, probabilities=p)
                            temporary.replace(cache)
                            score = float(f1_score(y[vi], p.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
                            fit.update(study_fingerprint=fingerprint, inner_fold=inner, candidate_index=candidate, validation_macro_f1=score, prediction_checkpoint_sha256=file_hash(cache))
                            _save_json(audit_cache, fit)
                        validate_probabilities(p, len(vi), len(config["classes"]))
                        fits.append(fit)
                        _progress(root, state="inner_fit_complete", procedure=procedure, outer_fold=fold, inner_fold=inner, candidate=candidate)
                    selections.append({"params": params, "fits": fits})
                selection = choose_candidate(selections, config, learner)
                model, p, final = fit_classifier(X, y, ids, train, test, learner, selection["params"], config, config["seed"]+fold, "outer_test")
                probabilities[test] = p
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez_compressed(temporary, indices=test, probabilities=p)
                temporary.replace(checkpoint)
                model_path = root/f"model_{fold:02d}_{procedure}.joblib"
                joblib.dump(model, model_path)
                audit = {"study_fingerprint": fingerprint, "procedure": procedure, "fold": fold, "inner_candidates": selections, "selection": selection,
                    "final_fit": final, "patient_metrics": metric_bundle(y[test], p, config["classes"]), "prediction_checkpoint_sha256": file_hash(checkpoint), "model_sha256": file_hash(model_path)}
                _save_json(audit_path, audit)
                audits.append(audit)
                _progress(root, state="outer_fit_complete", procedure=procedure, outer_fold=fold)
                print(f"{procedure} outer {fold}/{config['outer_splits']}: patient macro F1={audit['patient_metrics']['macro_f1']:.4f}", flush=True)
            mask = np.isfinite(probabilities).all(1)
            if not partial and not mask.all():
                raise ValueError("Incomplete geometry patient predictions")
            saved = {"study_fingerprint": fingerprint, "procedure": procedure, "classes": config["classes"], "complete": not partial, "evaluated_folds": chosen,
                "n_patients": int(mask.sum()), "n_beats": bank["summary"]["n_beats"], "patient_metrics": metric_bundle(y[mask], probabilities[mask], config["classes"]), "folds": audits,
                "claim_scope": CLAIM_SCOPE, "input_features": X.shape[1]}
            _save_json(root/f"metrics_{procedure}.json", saved)
            frame = pd.DataFrame({"patient_id": ids[mask], "truth": [config["classes"][i] for i in y[mask]], "prediction": [config["classes"][i] for i in probabilities[mask].argmax(1)], "fold": patient_folds[mask]})
            for column, name in enumerate(config["classes"]):
                frame[f"probability_{name}"] = probabilities[mask, column]
            frame.to_csv(root/f"patient_predictions_{procedure}.csv", index=False)
            save_confusion(y[mask], probabilities[mask].argmax(1), config["classes"], procedure, root)
            summary_rows.append({"procedure": procedure, "n_features": X.shape[1], "patient_macro_f1": saved["patient_metrics"]["macro_f1"], "patient_accuracy": saved["patient_metrics"]["accuracy"], "complete": not partial})
            pd.DataFrame(summary_rows).to_csv(root/"comparison.csv", index=False)
    _progress(root, state="partial_complete" if partial else "complete")
    return pd.DataFrame(summary_rows)


def summarize_study(config, draws=5000):
    validate_config(config)
    views, y, ids, folds, bank = load_features(config)
    root, chosen, partial = directory(config)
    scope = json.loads((root/"study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if partial or _hash(scope) != fingerprint or scope["config"] != config or scope["bank"] != bank or scope["folds_sha256"] != file_hash(config["folds_path"]) or scope["evaluated_folds"] != chosen or scope["code_sha256"] != file_hash(__file__):
        raise ValueError("Geometry report data/settings/code changed or incomplete")
    frozen = folds.set_index("patient_id").sort_index()
    patient_folds = np.array([frozen.fold[str(patient)] for patient in ids])
    predictions, metrics, rows = {}, {}, []
    fit_count = 0
    for learner in LEARNERS:
        for view in VIEWS:
            procedure, X = f"{learner}_{view}", views[view]
            saved = json.loads((root/f"metrics_{procedure}.json").read_text())
            if saved["study_fingerprint"] != fingerprint or saved["procedure"] != procedure or saved["evaluated_folds"] != chosen or saved["claim_scope"] != CLAIM_SCOPE or saved["n_beats"] != bank["summary"]["n_beats"]:
                raise ValueError("Incomplete/mismatched feature-screen metrics")
            truth, pred = _read_predictions(root/f"patient_predictions_{procedure}.csv", saved, frozen, config["classes"])
            predictions[procedure], metrics[procedure] = pred, saved["patient_metrics"]
            pooled, audited = np.full((len(ids), len(config["classes"])), np.nan), []
            for fold in chosen:
                audit = json.loads((root/f"fold_{fold:02d}_{procedure}.json").read_text())
                train, test = np.flatnonzero(patient_folds != fold), np.flatnonzero(patient_folds == fold)
                if audit["study_fingerprint"] != fingerprint or audit["procedure"] != procedure or audit["fold"] != fold:
                    raise ValueError("Geometry fold provenance changed")
                splits = [(train[ti], train[vi]) for ti, vi in inner_patient_splits(y[train], ids[train], config["inner_splits"], config["split_seed"]+fold*1000)]
                if len(audit["inner_candidates"]) != len(candidates(config, learner)):
                    raise ValueError("Missing feature candidate audits")
                for candidate, (selected, params) in enumerate(zip(audit["inner_candidates"], candidates(config, learner))):
                    if selected["params"] != params or len(selected["fits"]) != len(splits):
                        raise ValueError("Invalid feature candidate scope")
                    for inner, (fit, (ti, vi)) in enumerate(zip(selected["fits"], splits), 1):
                        expected = train_audit(X, y, ids, ti, vi, learner, params, config, config["seed"]+fold*1000+inner, "inner_validation")
                        if any(fit.get(key) != value for key, value in expected.items()) or fit["study_fingerprint"] != fingerprint or fit["inner_fold"] != inner or fit["candidate_index"] != candidate:
                            raise ValueError("Geometry inner fit membership/scaling mismatch")
                        path = root/f"inner_{fold:02d}_{procedure}_{candidate}_{inner}.npz"
                        if fit["prediction_checkpoint_sha256"] != file_hash(path):
                            raise ValueError("Geometry inner predictions changed")
                        with np.load(path) as values:
                            if not np.array_equal(values["indices"], vi):
                                raise ValueError("Inner prediction patients changed")
                            p = values["probabilities"]
                        validate_probabilities(p, len(vi), len(config["classes"]))
                        actual = float(f1_score(y[vi], p.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
                        if actual != fit["validation_macro_f1"]:
                            raise ValueError("Geometry selection scores disagree with predictions")
                selection = choose_candidate(audit["inner_candidates"], config, learner)
                expected = train_audit(X, y, ids, train, test, learner, selection["params"], config, config["seed"]+fold, "outer_test")
                if audit["selection"] != selection or audit["final_fit"] != expected:
                    raise ValueError("Geometry outer refit selection/scaling mismatch")
                path = root/f"fold_{fold:02d}_{procedure}.npz"
                if audit["prediction_checkpoint_sha256"] != file_hash(path):
                    raise ValueError("Geometry outer predictions changed")
                with np.load(path) as values:
                    if not np.array_equal(values["indices"], test):
                        raise ValueError("Outer prediction patients changed")
                    p = values["probabilities"]
                validate_probabilities(p, len(test), len(config["classes"]))
                model_path = root/f"model_{fold:02d}_{procedure}.joblib"
                if audit["model_sha256"] != file_hash(model_path):
                    raise ValueError("Fitted geometry model changed")
                model = joblib.load(model_path)
                if not np.array_equal(predict_pipeline(model, X[test]), p) or audit["patient_metrics"] != metric_bundle(y[test], p, config["classes"]):
                    raise ValueError("Saved geometry models/metrics disagree with test predictions")
                pooled[test] = p
                audited.append(audit)
                fit_count += len(candidates(config, learner))*config["inner_splits"] + 1
            frame = pd.read_csv(root/f"patient_predictions_{procedure}.csv").set_index("patient_id").sort_index()
            if audited != saved["folds"] or not np.allclose(pooled, frame[[f"probability_{label}" for label in config["classes"]]].to_numpy(), rtol=1e-7, atol=1e-12) or metric_bundle(y, pooled, config["classes"]) != saved["patient_metrics"]:
                raise ValueError("Geometry pooled patient metrics disagree with saved outer predictions")
            rows.append({"procedure": procedure, "patient_macro_f1": saved["patient_metrics"]["macro_f1"], "patient_accuracy": saved["patient_metrics"]["accuracy"], "n_features": X.shape[1]})
    references = ("linear_statistics", "tree_statistics", "linear_fixed_geometry", "tree_fixed_geometry")
    uncertainty = paired_bootstrap(truth, predictions, len(config["classes"]), draws=draws, seed=config["seed"], references=references)
    _save_json(root/"paired_bootstrap.json", dict(uncertainty, study_fingerprint=fingerprint))
    summary = {"complete": True, "study_fingerprint": fingerprint, "n_patients": len(ids), "n_beats": bank["summary"]["n_beats"],
        "unique_fit_count": fit_count, "procedures": rows, "claim_scope": CLAIM_SCOPE,
        "limits": "Fixed deterministic classifier procedures; no repeated-seed uncertainty. Different aggregation/model/input context from the historical ResNet; compare each geometry view with its own classifier control."}
    _save_json(root/"summary.json", summary)
    pd.DataFrame(rows).to_csv(root/"validated_comparison.csv", index=False)
    lines = ["# VCG phase geometry: patient-level feature screen", "", CLAIM_SCOPE, "",
        f"Complete: {len(ids)} patients, {bank['summary']['n_records']} recordings, {bank['summary']['n_beats']:,} unchanged beats and {len(config['classes'])} source labels. All {config['outer_splits']} frozen patient folds completed; {fit_count} validated fits.", "",
        "| Procedure | Patient macro F1 | Accuracy | Features | Δ vs same-learner statistics | Descriptive interval |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        name = row["procedure"]
        reference = name.split("_", 1)[0]+"_statistics"
        delta = row["patient_macro_f1"] - metrics[reference]["macro_f1"]
        interval = "—"
        if name != reference:
            lo, hi = uncertainty["arms"][name]["paired_comparisons"][reference]["macro_f1_difference_95_interval"]
            interval = f"[{lo:+.3f}, {hi:+.3f}]"
        lines.append(f"| {name} | {row['patient_macro_f1']:.3f} | {row['patient_accuracy']:.1%} | {row['n_features']} | {delta:+.3f} | {interval} |")
    lines += ["", f"{draws:,} class-stratified paired patient-bootstrap draws, conditional on fitted predictions. No refits, correction for correlated CV folds, repeated development or multiple comparisons. Every predeclared model is reported; there is no selected best test model or ensemble.", "",
        "## Exact screen", "",
        "Each patient contributes one row: medians and interquartile ranges over all of their original beats. The statistics control aggregates the existing 192 short-wavelet statistics into 384 columns. Fixed/adaptive geometry each aggregate 78 descriptors into 156 columns; the combined view has 540. This changes aggregation and classifier from the historical beat-trained ResNet: its 0.550 mean is context, not a matched comparator for these procedures.", "",
        f"Regularized multinomial logistic regression selects C from {config['linear_c']}; shallow XGBoost selects depth from {config['tree_depths']} with {config['tree_estimators']} trees and rate {config['tree_learning_rate']}. Each learner/view receives the same two-candidate, {config['inner_splits']}-inner-patient-fold budget. Mean inner macro F1 selects the candidate; ties follow configuration order. A fresh outer-training refit never uses test performance for selection. Class weighting is applied once; scaling uses weighted training-patient moments only. Seed {config['seed']} is fixed; repeating deterministic learners under arbitrary seeds is not treated as new evidence.", "",
        "Fixed geometry is the same dimensional descriptor family with R-relative phases. Adaptive geometry substitutes plausible multi-axis DWT phase estimates and falls back when missing/inconsistent. Both have the same allowed end ceiling; adaptive phases select different intervals. Boundary-success flags are not classifier inputs, but fallback-dependent durations can still encode detector behavior. The [boundary audit](../boundary_audit/REPORT.md) reports class-wise coverage and explicitly lacks expert boundary ground truth.", "",
        "All feature extraction is per-record/per-patient without cohort-fitted templates or normalization. No ECG channels, external data, patient exclusions, relabeling, R-peak changes, neural geometry branch, learned tensor contraction or shared-territory auxiliary loss are included in this first utility screen. Future neural work requires a separately controlled experiment; these results cannot stand in for its evaluation.", "",
        "## Adaptive versus fixed boundaries", ""]
    for learner in LEARNERS:
        name, reference = learner+"_adaptive_geometry", learner+"_fixed_geometry"
        delta = metrics[name]["macro_f1"]-metrics[reference]["macro_f1"]
        lo, hi = uncertainty["arms"][name]["paired_comparisons"][reference]["macro_f1_difference_95_interval"]
        lines.append(f"- {learner}: adaptive − fixed **{delta:+.3f}**, descriptive interval [{lo:+.3f}, {hi:+.3f}].")
    lines += ["", "## Class F1", "", "| Class | " + " | ".join(PROCEDURES) + " |", "|---|"+"---:|"*len(PROCEDURES)]
    for label in config["classes"]:
        lines.append(f"| {label} | "+" | ".join(f"{metrics[name]['classification_report'][label]['f1-score']:.3f}" for name in PROCEDURES)+" |")
    lines += ["", "The 0.80–0.90 target remains aspirational. An informative representation is not established by a plausible phase diagram; automatic boundaries are not clinically adjudicated, and negative results for these two learners do not prove geometry cannot help. Positive development results still need independent validation. Full-record processing is offline/noncausal.", "",
        "Reproduce: `.venv/bin/python -m mi_localization.geometry_bank`, then `.venv/bin/python -m mi_localization.geometry_study`, then `.venv/bin/python -m mi_localization.geometry_study --report`. Patient feature arrays, individual predictions, tensors, boundary examples and fitted models stay local. Only aggregate reports and hashed audits are public.", ""]
    destination = root/"REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


def status(config):
    root, _, _ = directory(config)
    lines = [f"Patient geometry screen: {root}"]
    for procedure in PROCEDURES:
        lines.append(f"{procedure}: {len(list(root.glob(f'fold_*_{procedure}.json')))}/{config['outer_splits']} completed outer fits")
    if (root/"progress.json").exists():
        lines.append("Last saved progress (not a live process check): "+(root/"progress.json").read_text())
    lines.append(f"Validated report: {(root/'REPORT.md').exists()}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geometry_vcg.yaml")
    parser.add_argument("--outer-folds", nargs="+", type=int)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        print(status(config))
    elif args.report:
        print(summarize_study(config, args.draws))
    else:
        print(run_study(config, args.outer_folds).to_string(index=False))
