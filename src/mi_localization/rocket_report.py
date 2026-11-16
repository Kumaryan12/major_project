"""Independent fold replay, transform reconstruction and ridge restoration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import yaml

from . import rocket_study as study
from . import residual_context_report as uncertainty
from .benchmark import metric_bundle, save_confusion
from .contrastive import _progress
from .neural import _save_json


def summarize(config):
    bank, assigned = study.load_inputs(config)
    scope = study.make_scope(config, bank)
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text()) != scope:
        raise ValueError("Random-convolution report source/config/runtime changed")
    member_p = {(a, s): np.full((len(bank.ids), len(config["classes"])), np.nan)
                for a in config["arms"] for s in config["transform_seeds"]}
    for fold in range(1, config["outer_splits"]+1):
        for seed in config["transform_seeds"]:
            rows, values = study.evaluate_fold(bank, assigned, config, scope, root, seed, fold, verify=True)
            for arm, p in values.items():
                if np.isfinite(member_p[arm, seed][rows]).any(): raise ValueError("Duplicate outer patient rows")
                member_p[arm, seed][rows] = p
    expected = study.budget(config)
    fit_paths = list(root.glob("inner_*.json"))+list(root.glob("outer_*.json"))
    if len(fit_paths) != expected["fits"] or len(list(root.glob("context_*.json"))) != expected["feature_contexts"]:
        raise ValueError("Unexpected convolutional fit/context count")
    metrics, ensemble_metrics, ensemble_p, selections = {}, {}, {}, {}
    for arm in config["arms"]:
        metrics[arm] = {}
        selections[arm] = {str(alpha): 0 for alpha in config["ridge_alphas"]}
        for seed in config["transform_seeds"]:
            p = member_p[arm, seed]
            if not np.isfinite(p).all(): raise ValueError("Missing outer predictions")
            metrics[arm][str(seed)] = metric_bundle(bank.labels, p, config["classes"])
            for fold in range(1, config["outer_splits"]+1):
                decision = json.loads((root/f"selection_{fold:02d}_{arm}_s{seed}.json").read_text())
                selections[arm][str(decision["selected_alpha"])] += 1
        ensemble_p[arm] = np.mean([member_p[arm, s] for s in config["transform_seeds"]], axis=0)
        ensemble_metrics[arm] = metric_bundle(bank.labels, ensemble_p[arm], config["classes"])
        for suffix, p in [(f"s{s}", member_p[arm, s]) for s in config["transform_seeds"]]+[("ensemble", ensemble_p[arm])]:
            frame = pd.DataFrame(dict(patient_id=bank.ids, truth=np.asarray(config["classes"])[bank.labels],
                                      prediction=np.asarray(config["classes"])[p.argmax(1)], fold=assigned))
            for i, c in enumerate(config["classes"]): frame[f"probability_{c}"] = p[:, i]
            frame.to_csv(root/f"patient_predictions_{arm}_{suffix}.csv", index=False)
        save_confusion(bank.labels, ensemble_p[arm].argmax(1), config["classes"], arm, root)
    baseline = member_p["dct", config["transform_seeds"][0]]
    if any(not np.array_equal(baseline, member_p["dct", s]) for s in config["transform_seeds"]):
        raise ValueError("Seed-independent DCT control changed across transform seeds")
    labels = np.arange(len(config["classes"]))
    weights = uncertainty.patient_weights(bank.labels, labels, config["bootstrap_draws"], config["seed"])
    draws = {a: uncertainty.weighted_f1(bank.labels, p.argmax(1), labels, weights) for a, p in ensemble_p.items()}
    pairs = [(a, "dct") for a in config["arms"] if a != "dct"]+[("combined", "multirocket"), ("combined", "hydra")]
    contrasts = {f"{a}_minus_{b}": dict(ensemble_delta_macro_f1=ensemble_metrics[a]["macro_f1"]-ensemble_metrics[b]["macro_f1"],
                 descriptive_patient_interval=np.quantile(draws[a]-draws[b], [.025, .975]).tolist()) for a, b in pairs}
    summary = dict(complete=True, study_fingerprint=scope["study_fingerprint"], claim_scope=study.CLAIM,
                   **expected, all_models_restored=len(fit_paths), feature_contexts_reconstructed=expected["feature_contexts"],
                   outer_models_restored=expected["outer_models"], all_original_templates_covered=True,
                   n_patients=len(bank.ids), n_records=len(bank.templates), transform_seeds=config["transform_seeds"],
                   deterministic_dct_control_verified=True, member_metrics=metrics, ensemble_metrics=ensemble_metrics,
                   contrasts=contrasts, selection_counts=selections)
    lines = ["# Bounded random-convolution VCG feature results", "", study.CLAIM, "",
             f"All {len(bank.ids)} patients and {len(bank.templates)} recording templates. {expected['fits']} classifier contexts restored; "
             f"{expected['feature_contexts']} feature contexts reconstructed from their training patients, including {expected['outer_models']} outer models.", "",
             "| Features + matched ridge | Mean member macro F1 ± transform-seed SD | Prespecified three-seed ensemble F1 |", "|---|---:|---:|"]
    for arm in config["arms"]:
        values = [m["macro_f1"] for m in metrics[arm].values()]
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        lines.append(f"| {arm} | {np.mean(values):.3f} ± {sd:.3f} | {ensemble_metrics[arm]['macro_f1']:.3f} |")
    lines += ["", "DCT is deterministic and identical across transform seeds; its repeated entries are one control predictor, not independent replications. Ensemble F1 differs from mean member F1. Softmax of ridge scores at fixed temperature one is a declared aggregation interface, not calibrated clinical probability.", "",
              "| Ensemble comparison | Δ macro F1 [descriptive interval] |", "|---|---:|"]
    for name, value in contrasts.items():
        lo, hi = value["descriptive_patient_interval"]
        lines.append(f"| {name} | {value['ensemble_delta_macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "## Class F1", "", "| Class | "+" | ".join(config["arms"])+" |", "|---|"+"---:|"*len(config["arms"])]
    for c in config["classes"]:
        lines.append(f"| {c} | "+" | ".join(f"{ensemble_metrics[a]['classification_report'][c]['f1-score']:.3f}" for a in config["arms"])+" |")
    lines += ["", "## Protocol and limitations", "",
              "Each recording template is the coordinate median of all its original denoised measured-XYZ beats. This compresses within-record variation, is not a real selected beat, and tests a template-based recipe rather than every possible beat-level random-convolution method. No patient, recording, beat, label, R position or frozen fold was changed. Every query recording contributes equally to its patient's feature row.", "",
              "Each exact inner/outer training partition fits one positive scalar RMS with equal patient/equal-record weighting, then MultiRocket-style quantile thresholds using only training templates. Bias examples are drawn patient-uniformly, then record-uniformly. HYDRA kernels are random and label-free. Inner/outer contexts have fresh partition-specific seeds and are never initialized from an outer-trained transform. Scaling of the final patient features is training-only. All four arms share inputs, aggregation, class-balanced ridge classifier and the same two alpha candidates; highest mean inner patient macro F1 wins, with configured-order ties. Feature dimensions/capacity differ.", "",
              "This bounded implementation uses MultiRocket's 84 length-nine patterns, four log-spaced dilations per input representation, original/differenced input, four paper-style positive-response poolers and consistent multivariate channel subsets. It uses true consecutive positive run lengths, including equality boundaries, and mean positive thresholded response; these are explicit conventions rather than exact historical source-code parity. HYDRA uses the vendored upstream experimental multivariate implementation with eight groups/eight kernels, including its with-replacement channel combinations. Both use the same StandardScaler/ridge interface, rather than HYDRA's separate sparse scaling. This is not the default full-size upstream recipe or MASHT.", "",
              "Transform seeds 42/123/2026 are all retained with equal probability averaging; no best-seed, subset, ensemble-weight or threshold search. The previous neural ensemble 0.584570 is a different frozen development procedure, not the matched DCT control. No TabPFN, neural training, external patients or new tensor-fusion search is part of this bounded phase.", "",
              f"Intervals use {config['bootstrap_draws']:,} class-stratified paired patient draws conditional on fixed predictions. They do not include refits or correction for correlated CV training folds, repeated development or multiple comparisons. Seed SD is random-transform variation, not patient-level generalization uncertainty. These repeatedly inspected patients remain development data; no established novelty, independent confirmation, clinical validity or guaranteed 0.80–0.90 score.", ""]
    (root/"REPORT.md").write_text("\n".join(lines))
    _save_json(root/"summary.json", summary)
    _progress(root, state="complete_validated", **expected)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rocket_vcg_v1.yaml")
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config["torch_threads"])
    try:
        with threadpool_limits(limits=1): result = summarize(config)
        print(json.dumps({"complete": result["complete"], "fits": result["fits"]}))
    except Exception as error:
        _progress(Path(config["output_dir"])/"complete", state="failed", error=str(error))
        raise
