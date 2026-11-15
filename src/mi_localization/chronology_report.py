"""Verify chronology windows/normalizers/stopping and restore all outer models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import chronology_study as study
from .benchmark import save_confusion
from .contrastive import _progress
from .deviation_report import paired_differences, scores
from .neural import _save_json


def matched_checks(root, config):
    checked = 0
    recurrent = ("ordered_no_rr", "ordered_rr", "shuffled_rr")
    for fold in range(1, config["outer_splits"]+1):
        for seed in config["training_seeds"]:
            for inner in range(config["inner_splits"]+1):
                audits = [json.loads((root/(f"inner_{fold:02d}_{arm}_s{seed}_i{inner}.json" if inner else f"outer_{fold:02d}_{arm}_s{seed}.json")).read_text())["fit"] for arm in recurrent]
                if any(row["initial_state_sha256"] != audits[0]["initial_state_sha256"] or row["normalization"] != audits[0]["normalization"] or row["parameter_count"] != audits[0]["parameter_count"] for row in audits):
                    raise ValueError("Chronology GRU controls differ in initialization/normalization/capacity")
                for epoch in range(min(len(row["history"]) for row in audits)):
                    rows = [row["history"][epoch] for row in audits]
                    if any(row["window_plan_sha256"] != rows[0]["window_plan_sha256"] or row["valid_beat_presentations"] != rows[0]["valid_beat_presentations"] for row in rows):
                        raise ValueError("Chronology controls did not see matched windows")
                checked += 1
    return checked


def summarize_study(config):
    bank, assigned = study.load_inputs(config);scope = study.make_scope(config, bank)
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text()) != scope:
        raise ValueError("Chronology report source/config/runtime changed")
    predicted, metrics = {}, {arm: {} for arm in config["arms"]}
    for arm in config["arms"]:
        for seed in config["training_seeds"]:
            p = np.full((len(bank.ids), len(config["classes"])), np.nan)
            for fold in range(1, config["outer_splits"]+1):
                rows, values = study.evaluate_arm(bank, assigned, config, scope, root, arm, seed, fold, verify=True)
                if np.isfinite(p[rows]).any(): raise ValueError("Duplicate chronological test patients")
                p[rows] = values
                _progress(root, state="verifying", arm=arm, training_seed=seed, outer_fold=fold)
            study.check_p(p, len(bank.ids), len(config["classes"]))
            predicted[arm, seed] = p;metrics[arm][str(seed)] = scores(bank.labels, p, config["classes"])
            frame = pd.DataFrame({"patient_id": bank.ids, "truth": np.asarray(config["classes"])[bank.labels],
                                  "prediction": np.asarray(config["classes"])[p.argmax(1)], "fold": assigned})
            for i, label in enumerate(config["classes"]): frame[f"probability_{label}"] = p[:, i]
            frame.to_csv(root/f"patient_predictions_{arm}_s{seed}.csv", index=False)
            save_confusion(bank.labels, p.argmax(1), config["classes"], f"{arm}_s{seed}", root)
    checked = matched_checks(root, config)
    contrasts = {}
    for control in ("shuffled_rr", "ordered_no_rr", "mean_rr"):
        first = [predicted["ordered_rr", seed] for seed in config["training_seeds"]]
        second = [predicted[control, seed] for seed in config["training_seeds"]]
        contrasts[f"ordered_rr_minus_{control}"] = {
            "six_class": paired_differences(bank.labels, first, second, config["bootstrap_draws"], config["seed"]),
            "mi_only": paired_differences(bank.labels, first, second, config["bootstrap_draws"], config["seed"], True, config["classes"].index("HC"))}
    outer = len(config["arms"])*len(config["training_seeds"])*config["outer_splits"]
    summary = {"complete": True, "study_fingerprint": scope["study_fingerprint"], "claim_scope": study.CLAIM,
               "n_patients": len(bank.ids), "classes": config["classes"], "training_seeds": config["training_seeds"],
               "fits": outer*(config["inner_splits"]+1), "outer_models_restored": outer,
               "matched_gru_partition_checks": checked, "all_query_beats_covered": True, "metrics": metrics, "contrasts": contrasts}
    _save_json(root/"summary.json", summary)
    lines = ["# Consecutive VCG beats and RR timing", "", study.CLAIM, "",
             f"All {len(bank.ids)} original patients. {summary['fits']} fits verified; {outer} outer models restored; {checked} matched-GRU partition checks passed.", "",
             "| Procedure | Training seed | Patient macro F1 | MI-only macro F1 |", "|---|---:|---:|---:|"]
    for arm, values in metrics.items():
        for seed, row in values.items(): lines.append(f"| {arm} | {seed} | {row['macro_f1']:.3f} | {row['mi_only_macro_f1_six_way_predictions']:.3f} |")
        result = [x["macro_f1"] for x in values.values()]
        spread = float(np.std(result, ddof=1)) if len(result)>1 else 0.
        lines.append(f"| {arm}, mean ± sample SD | all | {np.mean(result):.3f} ± {spread:.3f} | — |")
    lines += ["", "| Paired mean-seed difference | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |", "|---|---:|---:|"]
    for name, contrast in contrasts.items():
        cells = []
        for key in ("six_class", "mi_only"):
            row = contrast[key];lo, hi = row["descriptive_patient_interval"]
            cells.append(f"{row['mean_seed_delta_macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "All seeds reuse the same patients/folds; SD is training-seed variation, not a generalization confidence interval. Paired class-stratified patient resampling shares draws across seeds, conditions on fitted predictions, and makes no refits or correction for repeated development/overlapping training folds/multiple comparisons.", "",
              "## Interpretation and exact controls", "",
              "Windows contain consecutive retained beats from one original recording, never mixed recordings/patients. Cached R positions define preceding RR in seconds; these are detector-based intervals, not independently annotated physiological ground truth. Only the first retained beat of each recording has missing RR, represented by a mask. No interval threshold or quality rule removes a beat. Absolute R timestamps, beat indices and recording IDs never enter the network.", "",
              "Each beat has the established compact XYZ ResNet encoder. Preceding RR and its availability flag pass through a small MLP; a compact GRU models ordered tokens. The implementation uses explicit standard GRU equations with masks. The ordered_no_rr arm supplies zero RR inputs but retains the GRU architecture; ordered_rr and shuffled_rr have exactly the same architecture and initialization. Shuffling permutes waveform/RR pairs jointly, preserving token content and local preceding-RR information; it does not invent intervals from shuffled timestamps. This tests added value of order conditional on that local RR input. Evaluation permutations are fixed by seed/record/window and independent of unrelated query rows.", "",
              "All arms use the same sampled patients/recordings/windows for shared epochs. Sampling is balanced by class/patient, then uniform within a patient's recordings and possible contiguous starts. Training uses window cross-entropy. The mean_rr arm is a secondary window-pooling baseline with different active capacity; only the GRU controls isolate order/RR within the matched architecture. Historical unordered bags and the beat-wise ResNet have different aggregation/objective procedures.", "",
              "At inference, nonoverlapping windows cover every query beat once, including padded tail windows. Masks prevent padding updates. Window probabilities are weighted by valid-beat count within a recording, then recordings are averaged equally for each patient. No best-window selection occurs. This is offline aggregation using the examination's own beats.", "",
              "Waveform/RR scaling is fitted only on training patients. Inner patient validation selects the earliest best epoch with declared patience; rounded median inner epochs defines a fresh outer fit, without outer monitoring or transferred target-fold weights. All three declared training seeds are reported. Model/source/runtime hashes guard resumes; MPS is not promised bitwise deterministic across machines/versions. Every saved outer model reproduces probabilities within the declared verification tolerance and identical predicted classes.", "",
              "MI-only scoring retains original six-way predictions, including MI→HC errors. No cohort, label, patient fold, R position or original waveform was changed. These repeatedly inspected PTB folds remain development data; no SOTA or 0.80–0.90 claim follows without independent confirmation.", ""]
    (root/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _progress(root, state="complete_validated")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--config", default="configs/chronological_vcg.yaml")
    args = parser.parse_args();value = summarize_study(yaml.safe_load(Path(args.config).read_text()))
    print(json.dumps({"complete": value["complete"], "fits": value["fits"]}))
