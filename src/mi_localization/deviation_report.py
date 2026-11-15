"""Rebuild reference transforms, replay selectors and restore outer PFN contexts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import deviation_study as study
from .benchmark import metric_bundle, save_confusion
from .contrastive import _progress
from .neural import _save_json


def scores(y, p, classes):
    result = metric_bundle(y, p, classes)
    hc = classes.index("HC")
    mi = y != hc
    result["mi_only_macro_f1_six_way_predictions"] = float(f1_score(
        y[mi], p[mi].argmax(1), labels=[i for i in range(len(classes)) if i != hc], average="macro", zero_division=0))
    return result


def paired_differences(y, first, second, draws, seed, mi_only=False, hc=None):
    """Same patient draw in every seed; never pretend seeds are extra patients."""
    labels = np.unique(y)
    if mi_only:
        labels = labels[labels != hc]
    pools = [np.flatnonzero(y == label) for label in labels]
    rng = np.random.default_rng(seed)
    def difference(rows):
        return float(np.mean([f1_score(y[rows], a[rows].argmax(1), labels=labels, average="macro", zero_division=0)
                             - f1_score(y[rows], b[rows].argmax(1), labels=labels, average="macro", zero_division=0)
                             for a, b in zip(first, second, strict=True)]))
    point = difference(np.concatenate(pools))
    sampled = [difference(np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in pools])) for _ in range(draws)]
    return {"mean_seed_delta_macro_f1": point, "descriptive_patient_interval": np.quantile(sampled, [.025, .975]).tolist()}


def summarize_study(config):
    data = study.load_inputs(config)
    scope = study.make_scope(config, data)
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text()) != scope:
        raise ValueError("Deviation report data/code/config scope changed")
    classes, y, ids = data["source"]["classes"], data["y"], data["ids"]
    predicted = {}
    counts = {"inner_member_contexts": 0, "outer_member_contexts": 0}
    for arm in config["arms"]:
        seeds = [config["seed"]] if arm == "full_morphology" else config["reference_seeds"]
        for seed in seeds:
            p = np.full((len(ids), len(classes)), np.nan)
            for fold in range(1, data["source"]["outer_splits"]+1):
                rows, values = study.evaluate_arm(data, config, scope, root, arm, seed, fold, {}, verify=True)
                if np.isfinite(p[rows]).any():
                    raise ValueError("Duplicate out-of-fold patients")
                p[rows] = values
                member_count = 1 if arm == "full_morphology" else config["reference_parts"]
                counts["inner_member_contexts"] += len(config["feature_counts"])*data["source"]["inner_splits"]*member_count
                counts["outer_member_contexts"] += member_count
                _progress(root, state="verifying", arm=arm, reference_seed=seed, outer_fold=fold)
            study.check_p(p, len(ids), len(classes))
            predicted[arm, seed] = p
    metrics = {arm: {} for arm in config["arms"]}
    for (arm, seed), p in predicted.items():
        metrics[arm][str(seed)] = scores(y, p, classes)
        frame = pd.DataFrame({"patient_id": ids, "truth": np.asarray(classes)[y], "prediction": np.asarray(classes)[p.argmax(1)], "fold": data["assigned"]})
        for index, label in enumerate(classes):
            frame[f"probability_{label}"] = p[:, index]
        frame.to_csv(root/f"patient_predictions_{arm}_s{seed}.csv", index=False)
        save_confusion(y, p.argmax(1), classes, f"{arm}_s{seed}", root)
    contrasts = {}
    for arm, control in (("healthy", "ensemble_morphology"), ("mixed", "ensemble_morphology"),
                         ("healthy", "mixed"), ("ensemble_morphology", "full_morphology")):
        a = [predicted[arm, seed] for seed in config["reference_seeds"]]
        b = [predicted[control, config["seed"] if control == "full_morphology" else seed] for seed in config["reference_seeds"]]
        contrasts[f"{arm}_minus_{control}"] = {
            "six_class": paired_differences(y, a, b, config["bootstrap_draws"], config["seed"]),
            "mi_only": paired_differences(y, a, b, config["bootstrap_draws"], config["seed"], True, classes.index("HC"))}
    summary = {"complete": True, "study_fingerprint": scope["study_fingerprint"], "claim_scope": study.CLAIM,
               "n_patients": len(ids), "classes": classes, "reference_seeds": config["reference_seeds"],
               **counts, "total_member_contexts": sum(counts.values()), "all_outer_contexts_restored": True,
               "all_training_bootstrap_caches_replayed": True, "metrics": metrics, "contrasts": contrasts}
    _save_json(root/"summary.json", summary)
    lines = ["# Healthy-reference VCG deviation study", "", study.CLAIM, "",
             f"All {len(ids)} original patients; patient-disjoint nested selection. {sum(counts.values())} fitted member contexts; every outer context restored and every relevant training-bootstrap cache replayed.", "",
             "| Procedure | Reference seed | Patient macro F1 | MI-only macro F1 |", "|---|---:|---:|---:|"]
    for arm, by_seed in metrics.items():
        for seed, result in by_seed.items():
            lines.append(f"| {arm} | {seed} | {result['macro_f1']:.3f} | {result['mi_only_macro_f1_six_way_predictions']:.3f} |")
        if len(by_seed) > 1:
            values = [m["macro_f1"] for m in by_seed.values()]
            lines.append(f"| {arm}, mean ± sample SD | all | {np.mean(values):.3f} ± {np.std(values, ddof=1):.3f} | — |")
    lines += ["", "MI-only scores use the five original MI classes and unchanged six-way predictions: an MI patient predicted HC remains an error. Seed rows reuse the same patients and are not independent cohort replications. The full-morphology reference is fitted once per outer fold and is not three independently trained baselines.", "",
              "| Mean-seed difference | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |", "|---|---:|---:|"]
    for name, contrast in contrasts.items():
        cells = []
        for key in ("six_class", "mi_only"):
            item = contrast[key]; low, high = item["descriptive_patient_interval"]
            cells.append(f"{item['mean_seed_delta_macro_f1']:+.3f} [{low:+.3f}, {high:+.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", f"{config['bootstrap_draws']:,} paired class-stratified patient bootstrap draws, with the same patient indices applied across reference seeds. Intervals condition on fitted predictions; no refits or correction for correlated folds, repeated development or multiple comparisons. They are not independent confirmation.", "",
              "## Exact representation and comparison", "",
              "Each of three members learns two exact medoids from a separate reference part, and fits its classifier on the other two parts. Reference, classifier and query patients never overlap within a member. Every training patient is a classifier patient in two members. The healthy library uses training HC labels; this is supervised reference construction. Mixed libraries match its number of reference patients but sample identities without using disease labels or scores. Atlas sizes and medoid identities differ by member/partition; seed repeats change reference partitions, with the classifier seed policy held fixed.", "",
              "The deviation tensor subtracts physical XYZ reference waveforms from recording medians, then applies the established five additive wavelet bands and four fixed R-relative windows. Signed mean/RMS and DCT-distance features yield 243 extra columns. These windows are not clinical boundaries. Equal-record coordinate medians give one row per patient. Original 480-column morphology is retained in all arms; only deviation arms append columns. Two budgets (24/64) are selected by mean inner patient macro F1, separately for each procedure/seed, with configured-order ties. Bootstrap ranking/scaling uses classifier-training patients only. Queries never fit an atlas or selector. These are lossy deterministic features, not additional sensor data or clinically verified pathological residuals.", "",
              "Raw pinned local TabPFN-v2 probabilities, eight fixed inference members, no inverse-prior correction, fine-tuning, hosted prediction service or sample-weight approximation. The selector uses class-balanced training scaling, but the PFN sees raw selected columns and its own training-context preprocessing. Reference-disjoint ensembles have smaller classifier contexts and average three probability vectors; their morphology-only arm controls this. The full-training morphology procedure is a practical reference, not a capacity-matched control. Each procedure chooses its own feature budget; comparison includes that selection policy.", "",
              "No patient, class, label, original beat or R position was removed or changed. Recording medians use all beats of that recording, so this is offline inference. Repeatedly inspected PTB folds remain development data. No SOTA, clinical validity, mechanistic proof or 0.80–0.90 claim follows from architecture names or a favorable point estimate.", ""]
    (root/"REPORT.md").write_text("\n".join(lines))
    _progress(root, state="complete_validated")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deviation_vcg.yaml")
    args = parser.parse_args()
    result = summarize_study(yaml.safe_load(Path(args.config).read_text()))
    print(json.dumps({"complete": result["complete"], "total_member_contexts": result["total_member_contexts"]}))
