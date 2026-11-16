"""Replay all source features, bootstrap selectors and ridge/PFN saved models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import yaml

from . import rocket_tabpfn_study as study, residual_context_report as uncertainty
from .benchmark import metric_bundle, save_confusion
from .contrastive import _progress
from .neural import _save_json


def procedures(config):
    return ["full_ridge", "selected_ridge", "selected_tabpfn"]+[
        f"{learner}_k{k}" for k in config["feature_counts"] for learner in ["ridge", "tabpfn"]]


def summarize(config):
    data = study.load_inputs(config)
    scope = study.make_scope(config, data)
    bank, source = data["bank"], data["source"]
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text()) != scope: raise ValueError("Selected-convolution report scope changed")
    arms = procedures(config)
    members = {(arm, seed): np.full((len(bank.ids), len(source["classes"])), np.nan)
               for arm in arms for seed in source["transform_seeds"]}
    selections = {a: {str(k): 0 for k in config["feature_counts"]} for a in ["ridge", "tabpfn"]}
    for fold in range(1, source["outer_splits"]+1):
        for seed in source["transform_seeds"]:
            rows, values = study.evaluate_fold(data, config, scope, root, seed, fold, verify=True)
            for arm, p in values.items():
                if np.isfinite(members[arm, seed][rows]).any(): raise ValueError("Duplicate selected-model outer patients")
                members[arm, seed][rows] = p
            decision = json.loads((root/f"selection_{fold:02d}_s{seed}.json").read_text())
            for learner in selections:
                k = decision[f"{learner}_candidates"][decision[f"selected_{learner}"]]["k"]
                selections[learner][str(k)] += 1
    expected = study.budget(config, source)
    paths = list(root.glob("inner_*.json"))+list(root.glob("outer_*.json"))
    counts = {learner: sum(json.loads(p.read_text())["learner"] == learner for p in paths) for learner in ["ridge", "tabpfn"]}
    if (len(paths) != expected["fits"] or counts["ridge"] != expected["ridge_fits"] or counts["tabpfn"] != expected["tabpfn_fits"]
            or len(list(root.glob("outer_*.json"))) != expected["outer_models"]
            or len(list(root.glob("selector_*.json"))) != expected["selector_contexts"]
            or len(list((root/"bootstrap_cache").glob("*.npz"))) != expected["source_feature_contexts"]):
        raise ValueError("Unexpected selector/classifier context counts")
    member_metrics, ensemble_metrics, ensembles = {}, {}, {}
    for arm in arms:
        member_metrics[arm] = {}
        for seed in source["transform_seeds"]:
            p = members[arm, seed]
            study.check_p(p, len(bank.ids), len(source["classes"]))
            member_metrics[arm][str(seed)] = metric_bundle(bank.labels, p, source["classes"])
        ensembles[arm] = np.mean([members[arm, seed] for seed in source["transform_seeds"]], axis=0)
        ensemble_metrics[arm] = metric_bundle(bank.labels, ensembles[arm], source["classes"])
        for suffix, p in [(f"s{s}", members[arm, s]) for s in source["transform_seeds"]]+[("ensemble", ensembles[arm])]:
            frame = pd.DataFrame(dict(patient_id=bank.ids, truth=np.asarray(source["classes"])[bank.labels],
                prediction=np.asarray(source["classes"])[p.argmax(1)], fold=data["assigned"]))
            for i, c in enumerate(source["classes"]): frame[f"probability_{c}"] = p[:, i]
            frame.to_csv(root/f"patient_predictions_{arm}_{suffix}.csv", index=False)
        save_confusion(bank.labels, ensembles[arm].argmax(1), source["classes"], arm, root)
    original = json.loads((data["root"]/"summary.json").read_text())
    if abs(ensemble_metrics["full_ridge"]["macro_f1"]-original["ensemble_metrics"]["combined"]["macro_f1"]) > 1e-12:
        raise ValueError("Frozen full-feature ensemble does not reproduce source result")
    for seed in source["transform_seeds"]:
        if abs(member_metrics["full_ridge"][str(seed)]["macro_f1"]-original["member_metrics"]["combined"][str(seed)]["macro_f1"]) > 1e-12:
            raise ValueError("Frozen full-feature member result changed")
    weights = uncertainty.patient_weights(bank.labels, np.arange(len(source["classes"])), config["bootstrap_draws"], config["seed"])
    draws = {a: uncertainty.weighted_f1(bank.labels, p.argmax(1), np.arange(len(source["classes"])), weights) for a, p in ensembles.items()}
    pairs = [("selected_tabpfn", "selected_ridge"), ("selected_tabpfn", "full_ridge"), ("selected_ridge", "full_ridge")]
    pairs += [(f"tabpfn_k{k}", f"ridge_k{k}") for k in config["feature_counts"]]
    contrasts = {f"{a}_minus_{b}": dict(ensemble_delta_macro_f1=ensemble_metrics[a]["macro_f1"]-ensemble_metrics[b]["macro_f1"],
        descriptive_patient_interval=np.quantile(draws[a]-draws[b], [.025, .975]).tolist()) for a, b in pairs}
    study.unchanged_source(data)
    summary = dict(complete=True, study_fingerprint=scope["study_fingerprint"], claim_scope=study.CLAIM, **expected,
        all_new_models_restored=expected["fits"], tabpfn_models_restored=expected["tabpfn_fits"], ridge_models_restored=expected["ridge_fits"],
        outer_models_restored=expected["outer_models"], source_features_reconstructed=expected["source_feature_contexts"],
        source_ridge_models_restored=expected["source_ridge_contexts"], bootstrap_caches_replayed=expected["source_feature_contexts"],
        selector_masks_verified=expected["selector_contexts"], source_artifacts_unchanged=True, frozen_reference_reproduced=True,
        all_outer_patients_covered=True, n_patients=len(bank.ids), transform_seeds=source["transform_seeds"],
        member_metrics=member_metrics, ensemble_metrics=ensemble_metrics, contrasts=contrasts, selection_counts=selections)
    lines = ["# Feature-selected convolutional ridge versus local TabPFN", "", study.CLAIM, "",
        f"All {len(bank.ids)} original patients. {expected['fits']} new classifiers restored "
        f"({expected['ridge_fits']} ridge / {expected['tabpfn_fits']} PFN), including {expected['outer_models']} outer models. "
        f"{expected['source_feature_contexts']} source feature contexts reconstructed, {expected['source_ridge_contexts']} frozen ridge contexts restored, "
        f"{expected['source_feature_contexts']} bootstrap caches replayed and {expected['selector_contexts']} masks verified. Pinned source artifacts unchanged.", "",
        "| Procedure | Mean member macro F1 ± pipeline-seed SD | Prespecified seed-ensemble F1 |", "|---|---:|---:|"]
    for arm in arms:
        values = [v["macro_f1"] for v in member_metrics[arm].values()]
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        lines.append(f"| {arm} | {np.mean(values):.3f} ± {sd:.3f} | {ensemble_metrics[arm]['macro_f1']:.3f} |")
    lines += ["", "The three primary procedures are full_ridge, selected_ridge and selected_tabpfn. Fixed-budget rows are declared controls, not additional independently fitted selected models. Every seed is retained; ensemble F1 is not mean member F1. No best seed/budget is chosen using this table.", "",
        "| Ensemble comparison | Δ macro F1 [descriptive interval] |", "|---|---:|"]
    for name, row in contrasts.items():
        lo, hi = row["descriptive_patient_interval"]
        lines.append(f"| {name} | {row['ensemble_delta_macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "## Primary ensemble class F1", "", "| Class | full_ridge | selected_ridge | selected_tabpfn |", "|---|---:|---:|---:|"]
    for c in source["classes"]:
        lines.append(f"| {c} | "+" | ".join(f"{ensemble_metrics[a]['classification_report'][c]['f1-score']:.3f}" for a in arms[:3])+" |")
    lines += ["", "## Protocol and interpretation", "",
        "Source inputs are the completed combined MultiRocket-style/HYDRA recording-template features, with their exact original train/query memberships and transform seeds. No new representation, waveform/label change or full-cohort transform was fitted. Feature contexts were independently reproduced from their own training templates during verification. The selected representation was chosen after earlier development results; this is adaptive development on the same repeatedly inspected patients, not a fresh test.", "",
        "Each partition fits class-stratified bootstrap consensus using 64 training resamples and an independent 32-resample diagnostic panel, ANOVA ranking and absolute Pearson pruning at 0.95. Budgets are at most 32/64 columns, with actual selected counts recorded. Diagnostic-panel support does not choose masks/budgets. Both classifiers see exactly the same selected raw columns within a budget/partition. No validation or outer labels enter feature selection; outer-trained masks never initialize inner contexts.", "",
        "Ridge uses training-only StandardScaler, class-balanced weights and alpha [1,100]. TabPFN uses the existing pinned v2 checkpoint, eight ensemble members, temperature 0.9, unbalanced probabilities and its own training-context preprocessing; it runs in isolated local workers. These are classifier procedures with different preprocessing/prior/weighting, not a pure causal isolation of nonlinearity. No PFN checkpoint/member-count/temperature/prior correction search or hosted prediction service is used.", "",
        "Selected ridge chooses among budget/alpha candidates by highest arithmetic mean inner patient F1; selected PFN chooses its budget the same way. Ties follow configured budget/alpha order. Selected procedures can choose different masks/counts. Fixed-budget ridge-versus-PFN controls isolate the shared mask/count more directly; ridge alpha remains training-only selected. Both budgets' outer predictions were declared for reporting and never pick a global winner. Primary selected predictions reuse these outer models without extra fitting.", "",
        "All source seeds 42/123/2026 are retained, with selector-seed offset 100000019. The PFN inference seed is 42+1000*outer_fold+inner_fold, common across source-transform seeds; member variation therefore includes source transforms and bootstrap selection, not three independent foundation-model pretrainings. Equal probability averaging is prespecified for every procedure. Ridge softmax scores are not calibrated clinical probabilities.", "",
        "Frozen full_ridge reproduces the earlier feature ensemble 0.584266; historical neural 0.584570 is a different separate development reference. All original 164 patients/391 records/55,863 beats and frozen folds remain. Templates compress beat variation; this study does not assess an added variability branch, ECG input or external pretraining.", "",
        f"Intervals use {config['bootstrap_draws']:,} class-stratified paired patient draws conditional on fixed predictions, without model refits or correction for CV dependence, adaptive development or multiple comparisons. Pipeline-seed SD is not generalization uncertainty. No result establishes independent confirmation, clinical validity, SOTA or guaranteed 0.80–0.90 macro F1.", ""]
    (root/"REPORT.md").write_text("\n".join(lines))
    _save_json(root/"summary.json", summary)
    _progress(root, state="complete_validated", **expected)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rocket_tabpfn_v1.yaml")
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=1): result = summarize(config)
        print(json.dumps(dict(complete=result["complete"], fits=result["fits"])))
    except Exception as error:
        root = Path(config["output_dir"])/"complete"
        root.mkdir(parents=True, exist_ok=True)
        _progress(root, state="failed", error=str(error))
        raise
