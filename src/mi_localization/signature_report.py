"""Independent replay and complete pooled evaluation of signature feature arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import signature_study as study
from .benchmark import save_confusion
from .contrastive import _progress
from .deviation_report import paired_differences, scores
from .neural import _save_json


def summarize_study(config):
    data = study.load_inputs(config)
    scope = study.make_scope(config, data)
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text()) != scope:
        raise ValueError("Signature report source/config/runtime changed")
    y, ids, classes = data["y"], data["ids"], data["source"]["classes"]
    probabilities, metrics = {}, {arm: {} for arm in config["arms"]}
    total_contexts, outer_contexts = 0, 0
    for arm in config["arms"]:
        for seed in config["sampling_seeds"]:
            pooled = np.full((len(ids), len(classes)), np.nan)
            for fold in range(1, data["source"]["outer_splits"]+1):
                rows, p = study.evaluate_arm(data, config, scope, root, arm, seed, fold, verify=True)
                if np.isfinite(pooled[rows]).any():
                    raise ValueError("Duplicate signature out-of-fold membership")
                pooled[rows] = p
                total_contexts += len(config["feature_counts"])*data["source"]["inner_splits"]+1
                outer_contexts += 1
                _progress(root, state="verifying", arm=arm, sampling_seed=seed, outer_fold=fold)
            study.check_p(pooled, len(ids), len(classes))
            probabilities[arm, seed] = pooled
            metrics[arm][str(seed)] = scores(y, pooled, classes)
            frame = pd.DataFrame({"patient_id": ids, "truth": np.asarray(classes)[y],
                                  "prediction": np.asarray(classes)[pooled.argmax(1)], "fold": data["assigned"]})
            for index, label in enumerate(classes):
                frame[f"probability_{label}"] = pooled[:, index]
            frame.to_csv(root/f"patient_predictions_{arm}_s{seed}.csv", index=False)
            save_confusion(y, pooled.argmax(1), classes, f"{arm}_s{seed}", root)
    comparisons = [(arm, "morphology") for arm in config["arms"] if arm != "morphology"]
    comparisons += [("signature3", control) for control in ("signed_area", "endpoint_polynomial", "signature2")]
    contrasts = {}
    for arm, control in comparisons:
        first = [probabilities[arm, seed] for seed in config["sampling_seeds"]]
        second = [probabilities[control, seed] for seed in config["sampling_seeds"]]
        contrasts[f"{arm}_minus_{control}"] = {
            "six_class": paired_differences(y, first, second, config["bootstrap_draws"], config["seed"]),
            "mi_only": paired_differences(y, first, second, config["bootstrap_draws"], config["seed"], True, classes.index("HC"))}
        _progress(root, state="paired_descriptive_intervals", contrast=f"{arm}_minus_{control}")
    summary = {"complete": True, "study_fingerprint": scope["study_fingerprint"], "claim_scope": study.CLAIM,
               "n_patients": len(ids), "classes": classes, "sampling_seeds": config["sampling_seeds"],
               "total_contexts": total_contexts, "outer_contexts": outer_contexts,
               "all_outer_contexts_restored": True, "all_bootstrap_caches_replayed": True,
               "metrics": metrics, "contrasts": contrasts}
    _save_json(root/"summary.json", summary)
    lines = ["# Ordered VCG path-signature feature comparison", "", study.CLAIM, "",
             f"All {len(ids)} patients, frozen outer/inner patient splits. {total_contexts} fitted contexts; all {outer_contexts} outer contexts restored and relevant selector-bootstrap caches independently replayed.", "",
             "| Procedure | Sampling seed | Macro F1 | MI-only macro F1 |", "|---|---:|---:|---:|"]
    for arm, by_seed in metrics.items():
        for seed, result in by_seed.items():
            lines.append(f"| {arm} | {seed} | {result['macro_f1']:.3f} | {result['mi_only_macro_f1_six_way_predictions']:.3f} |")
        values = [x["macro_f1"] for x in by_seed.values()]
        lines.append(f"| {arm}, mean ± sample SD | all | {np.mean(values):.3f} ± {np.std(values, ddof=1):.3f} | — |")
    lines += ["", "The seeds change classifier-training bootstrap feature selection while the PFN inference-seed policy is held fixed. These are not three independent patient cohorts or three gradient-trained PFN foundation models. MI-only evaluation keeps original six-way predictions, including HC mistakes on MI patients.", "",
              "| Mean-seed comparison | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |", "|---|---:|---:|"]
    for name, result in contrasts.items():
        cells = []
        for endpoint in ("six_class", "mi_only"):
            item = result[endpoint]; low, high = item["descriptive_patient_interval"]
            cells.append(f"{item['mean_seed_delta_macro_f1']:+.3f} [{low:+.3f}, {high:+.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    baseline_columns = data["features"]["morphology"].shape[1]
    added = {arm: X.shape[1]-baseline_columns for arm, X in data["features"].items()}
    lines += ["", f"{config['bootstrap_draws']:,} paired class-stratified patient bootstrap draws, with each draw shared across sampling seeds. Conditional on fixed predictions, no model refits or correction for overlapping training folds, repeated development or multiple comparisons. These are descriptive intervals, not independent confirmation.", "",
              "## Exact comparison", "",
              f"Each arm retains the original {baseline_columns} morphology columns. Added columns: signed-area {added['signed_area']}, endpoint-polynomial {added['endpoint_polynomial']}, signature2 {added['signature2']}, signature3 {added['signature3']}. Descriptor counts include the configured fixed windows, within-record beat median/IQR, and equal-record patient aggregation. Physical XYZ means retain offsets alongside increment-based signatures. Every original beat contributes, with no temporal/spatial warping, per-beat normalization, record filtering or new clinical phase labels.", "",
              "The raw feature bank retains complete tensor words for auditing. Before classification, pure clock-only words and their IQR columns are excluded by their schema, independently of patient values/labels: every beat has the same clock per fixed window. Time–XYZ interactions and physical XYZ means are retained. This avoids tiny negative weighted-variance roundoff on known constants; no full-cohort variable-column screen is fitted.", "",
              "The same-size endpoint-polynomial control uses straight endpoint increments and window means, omitting interior ordered path terms. Signed areas provide a simpler geometric control. Differences in feature-pool structure remain part of each procedure: equal selected budgets do not make representation/selection identical or establish causal physiology.", "",
              "Each outer-training set is fully available to every classifier arm. Training-only 64-bootstrap consensus selection and 32 separate audit resamples retain at most 24/64 columns, with budget selected by mean inner patient macro F1, stable configured-order ties. Scaling, ranking and correlation pruning never use query patients. Raw selected values go to frozen pinned local TabPFN-v2 with eight fixed inference members; no prior correction, fine-tuning, external prediction API or changed checkpoint. Every outer context is freshly fitted after inner selection. A new matched control is included for every sampling seed.", "",
              "Mathematical identity checks and raw/denoised/±4 ms sensitivity audits are in the separate bank audit. They neither choose features nor establish diagnostic value. Signatures preserve selected ordered interactions of piecewise-linear trajectories, not a new sensor measurement or a complete lossless encoding at finite degree. Signature methods and their ECG application are prior art; this is a measured-VCG MI development experiment, not an unqualified originality or SOTA claim.", "",
              "No patient, label, class, original beat or R position changed. Existing offline denoising/record aggregation remains. Historical ResNet 0.550 ± 0.040 and earlier tabular results use different procedures and are context, not matched controls. These PTB folds have been repeatedly inspected; independent cohort confirmation remains necessary.", ""]
    (root/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _progress(root, state="complete_validated")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/signature_vcg.yaml")
    args = parser.parse_args()
    summary = summarize_study(yaml.safe_load(Path(args.config).read_text()))
    print(json.dumps({"complete": summary["complete"], "total_contexts": summary["total_contexts"]}))
