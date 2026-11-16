"""Independent beat-feature reconstruction and every-model distribution replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import distribution_study as study
from . import residual_context_report as uncertainty
from .benchmark import metric_bundle, save_confusion
from .contrastive import _progress
from .neural import _save_json
from .signal_audit import file_hash


def summarize(config):
    bank, assigned = study.load_inputs(config, verify=True)
    scope = study.make_scope(config, bank)
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text()) != scope:
        raise ValueError("Distribution report source/config/runtime changed")
    members = {(a, s): np.full((len(bank.ids), len(config["classes"])), np.nan)
               for a in config["arms"] for s in config["projection_seeds"]}
    matched = 0
    for fold in range(1, config["outer_splits"]+1):
        for seed in config["projection_seeds"]:
            rows, values = study.evaluate_fold(bank, assigned, config, scope, root, seed, fold, verify=True)
            for arm, p in values.items():
                if np.isfinite(members[arm, seed][rows]).any(): raise ValueError("Duplicate outer patients")
                members[arm, seed][rows] = p
            for inner in range(config["inner_splits"]+1):
                ctx = root/"review/contexts"/f"context_{fold:02d}_s{seed}_i{inner}.json"
                audit = json.loads(ctx.read_text())
                if len(set(audit["dimensions"].values())) != 1 or set(audit["dimensions"]) != set(config["arms"]):
                    raise ValueError("Unmatched arm dimensions")
                for arm in config["arms"]:
                    suffixes = [f"_i{inner}_a{i}" for i in range(len(config["ridge_alphas"]))] if inner else [""]
                    for suffix in suffixes:
                        name = f"{'inner' if inner else 'outer'}_{fold:02d}_{arm}_s{seed}{suffix}.json"
                        fit = json.loads((root/"review/fits"/name).read_text())
                        if fit["context_sha256"] != file_hash(ctx): raise ValueError("Arms used unmatched feature state")
                matched += 1
    expected = study.budget(config)
    fits = sorted((root/"review/fits").glob("*.json")); contexts = sorted((root/"review/contexts").glob("*.json"))
    if len(fits) != expected["fits"] or len(contexts) != expected["feature_contexts"] or matched != expected["feature_contexts"]:
        raise ValueError("Incomplete/unexpected distribution context count")
    metrics, means, ensemble_metrics, ensemble, counts = {}, {}, {}, {}, {}
    for arm in config["arms"]:
        metrics[arm] = {}; counts[arm] = {str(x): 0 for x in config["ridge_alphas"]}
        for seed in config["projection_seeds"]:
            p = members[arm, seed]
            if not np.isfinite(p).all(): raise ValueError("Missing outer predictions")
            metrics[arm][str(seed)] = metric_bundle(bank.labels, p, config["classes"])
            for fold in range(1, config["outer_splits"]+1):
                decision = json.loads((root/"review"/f"selection_{fold:02d}_{arm}_s{seed}.json").read_text())
                counts[arm][str(decision["selected_alpha"])] += 1
        values = [x["macro_f1"] for x in metrics[arm].values()]
        means[arm] = dict(macro_f1_mean=float(np.mean(values)), macro_f1_sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.)
        ensemble[arm] = np.mean([members[arm, seed] for seed in config["projection_seeds"]], axis=0)
        ensemble_metrics[arm] = metric_bundle(bank.labels, ensemble[arm], config["classes"])
        save_confusion(bank.labels, ensemble[arm].argmax(1), config["classes"], arm, root)
        for suffix, p in [(f"s{s}", members[arm, s]) for s in config["projection_seeds"]]+[("ensemble", ensemble[arm])]:
            frame = pd.DataFrame(dict(patient_id=bank.ids, truth=np.asarray(config["classes"])[bank.labels],
                                      prediction=np.asarray(config["classes"])[p.argmax(1)], fold=assigned))
            for i, c in enumerate(config["classes"]):frame[f"probability_{c}"] = p[:, i]
            frame.to_csv(root/"review"/f"patient_predictions_{arm}_{suffix}.csv", index=False)
    labels = np.arange(len(config["classes"]))
    weights = uncertainty.patient_weights(bank.labels, labels, config["bootstrap_draws"], config["seed"])
    draws = {a: uncertainty.weighted_f1(bank.labels, p.argmax(1), labels, weights) for a, p in ensemble.items()}
    mean_draws = {a: np.mean([uncertainty.weighted_f1(bank.labels, members[a, s].argmax(1), labels, weights)
                             for s in config["projection_seeds"]], axis=0) for a in config["arms"]}
    pairs = [("quantiles", "mean_spread"), ("quantiles", "mean_embedding"), ("quantiles", "median_waveform"),
             ("mean_spread", "mean_embedding"), ("mean_embedding", "median_waveform")]
    contrasts = {f"{a}_minus_{b}": dict(primary=(a, b) == pairs[0],
                  ensemble_delta_macro_f1=ensemble_metrics[a]["macro_f1"]-ensemble_metrics[b]["macro_f1"],
                  descriptive_patient_interval=np.quantile(draws[a]-draws[b], [.025, .975]).tolist(),
                  mean_member_delta_macro_f1=means[a]["macro_f1_mean"]-means[b]["macro_f1_mean"],
                  mean_member_descriptive_interval=np.quantile(mean_draws[a]-mean_draws[b], [.025, .975]).tolist()) for a, b in pairs}
    audit = {str(p.relative_to(root)): file_hash(p) for p in fits+contexts}
    _save_json(root/"verification_manifest.json", dict(study_fingerprint=scope["study_fingerprint"], restored_audit_hashes=audit))
    lines = ["# Measured-VCG beat-distribution pooling", "", study.CLAIM, "",
             f"All {len(bank.ids)} patients, {len(bank.record_patients)} recordings and {len(bank.features):,} original beats retained. "
             f"Reconstructed every beat and record-median feature, all {len(contexts)} training transforms, and restored all {len(fits)} classifiers "
             f"including {expected['outer_models']} outer models. {matched} matched four-arm feature-context checks passed.", "",
             "| Procedure | Mean member macro F1 ± projection-seed SD | Uniform three-seed ensemble macro F1 |", "|---|---:|---:|"]
    for arm in config["arms"]:
        m = means[arm];lines.append(f"| {arm} | {m['macro_f1_mean']:.6f} ± {m['macro_f1_sample_sd']:.6f} | {ensemble_metrics[arm]['macro_f1']:.6f} |")
    lines += ["", "## Prespecified comparisons", "", "Primary: quantiles minus mean_spread. All contrasts retained. Intervals are descriptive and conditional.", "",
              "| Ensemble comparison | Difference | Paired patient 95% interval |", "|---|---:|---:|"]
    for name, result in contrasts.items():
        lo, hi = result["descriptive_patient_interval"]
        lines.append(f"| {name} | {result['ensemble_delta_macro_f1']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "## Ensemble class F1", "", "| Class | "+" | ".join(config["arms"])+" |", "|---|"+"---:|"*len(config["arms"])]
    for c in config["classes"]:
        lines.append(f"| {c} | "+" | ".join(f"{ensemble_metrics[a]['classification_report'][c]['f1-score']:.6f}" for a in config["arms"])+" |")
    lines += ["", "## Method and limits", "",
              "The fixed feature extractor keeps signed XYZ DCT coefficients and RMS, mean absolute amplitude, minimum and maximum in approximate R-relative windows. Those windows are not clinical delineations. The bank has no learned cohort state. Only exact training patients fit feature normalization, with equal patient, record and within-record beat weights. Seeded unit-norm random projections operate on feature space, without rotating or unit-normalizing physical XYZ.", "",
              "All four arms use the same window/projection/quantile dimensions. Median_waveform encodes each recording's coordinate-median waveform. Mean_embedding repeats its mean projected beat feature on the grid. Mean_spread encodes only mean and population SD as mu + sigma times fixed standard-normal quantiles. This is a moment-control basis, not a claim that physiological beat distributions are Gaussian. Quantiles uses empirical linear-interpolation quantiles of all original beats in that record. Every grid is divided by sqrt(Q); no per-arm StandardScaler undoes that normalization. Recording tensors pool equally per patient. Effective feature ranks differ despite identical stored dimensions.", "",
              "Class-balanced ridge with intercept and Cholesky solver receives the same two declared alpha candidates in every arm. Three inner patient folds choose alpha using mean inner F1; configured-order ties. Every outer refit excludes its query patients from learned preprocessing and selection. All three projection seeds remain in equal probability ensembles. Softmax of ridge scores at fixed temperature one is an aggregation interface, not calibrated clinical risk. Seed SD measures feature-projection variation, not independent patients or a generalization confidence interval. Per-seed scores and mean-member differences remain in summary.json.", "",
              "The primary contrast tests quantile shape beyond mean/spread. Other contrasts distinguish encoding-before-pooling and simpler variability effects. No outer-score-based arm, quantile, window, budget, projection, source encoder, threshold or seed-subset selection occurs. The repeatedly inspected PTB cohort remains development data. Patient bootstrap shares class-stratified draws across arms and seeds, with no refits or corrections for repeated development, overlapping training folds or multiple comparisons. More beats are not more independent patients. Neither a high point score nor a positive conditional interval establishes novelty, SOTA, clinical validity or guaranteed 0.80–0.90 performance.", "",
              "This is a compact projected-quantile adaptation inspired by set/distribution pooling, not an exact implementation of PSWE or FSW and not a claim to their injectivity guarantees. It uses no new patients, ECG input, source pretraining, learned neural encoder or Tucker compression. Historical neural0.584570, random-convolution0.584266 and supervised-transfer results remain separate development references, not the matched controls here.", "",
              "Protocol: docs/distribution_pooling_plan.md. Research basis: https://papers.nips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html and https://proceedings.iclr.cc/paper_files/paper/2025/hash/3dbb8b6b5576b85afb3037e9630812dc-Abstract-Conference.html.", ""]
    (root/"REPORT.md").write_text("\n".join(lines))
    result = dict(complete=True, study_fingerprint=scope["study_fingerprint"], claim_scope=study.CLAIM, **expected,
                  all_models_restored=len(fits), outer_models_restored=expected["outer_models"], feature_contexts_reconstructed=len(contexts),
                  matched_context_checks=matched, all_original_beats_reconstructed=True, original_cohort_unchanged=True,
                  n_patients=len(bank.ids), n_records=len(bank.record_patients), n_beats=len(bank.features), projection_seeds=config["projection_seeds"],
                  tensor_shape_per_record=[len(config["windows"]), config["projections"], len(config["quantiles"])],
                  mean_member_metrics=means, member_metrics=metrics, ensemble_metrics=ensemble_metrics, contrasts=contrasts,
                  selection_counts=counts, report_sha256=file_hash(root/"REPORT.md"),
                  verification_manifest_sha256=file_hash(root/"verification_manifest.json"))
    _save_json(root/"summary.json", result)
    _progress(root, state="complete_independently_validated", **expected)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", default="configs/distribution_pooling_v1.yaml")
    args = p.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); torch.set_num_threads(config["torch_threads"])
    try:
        with threadpool_limits(limits=1): result = summarize(config)
        print(json.dumps({k: result[k] for k in ("complete", "fits", "all_models_restored")}))
    except Exception as error:
        _progress(Path(config["output_dir"])/"complete", state="failed", error=str(error)); raise
