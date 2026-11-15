"""Verify the complete patient-nested beat-length selection procedure."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import chronology_lengths_study as study
from .benchmark import save_confusion
from .contrastive import _progress
from .deviation_report import paired_differences, scores
from .neural import _save_json


def matched_checks(root, base, grid, seed, fold, selected_length):
    checked = 0
    for length in grid["lengths"]:
        phases = list(range(1, base["inner_splits"]+1))+([0] if length == selected_length else [])
        for inner in phases:
            audits = []
            for arm in ("ordered_no_rr", "ordered_rr", "shuffled_rr"):
                name = f"inner_{fold:02d}_{arm}_s{seed}_l{length}_i{inner}.json" if inner else f"outer_{fold:02d}_{arm}_s{seed}_l{length}.json"
                audits.append(json.loads((root/name).read_text())["fit"])
            if any(a["initial_state_sha256"] != audits[0]["initial_state_sha256"] or a["normalization"] != audits[0]["normalization"] or a["parameter_count"] != audits[0]["parameter_count"] for a in audits):
                raise ValueError("Length-study recurrent controls are not matched")
            for epoch in range(min(len(a["history"]) for a in audits)):
                rows = [a["history"][epoch] for a in audits]
                if any(a["window_plan_sha256"] != rows[0]["window_plan_sha256"] or a["valid_beat_presentations"] != rows[0]["valid_beat_presentations"] for a in rows):
                    raise ValueError("Length-study controls saw different windows")
            checked += 1
    return checked


def summarize_study(grid):
    base, bank, assigned = study.load_inputs(grid);scope = study.make_scope(grid, base, bank)
    root = Path(grid["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text()) != scope:
        raise ValueError("Length report source/config/runtime changed")
    predicted = {(arm, seed): np.full((len(bank.ids), len(base["classes"])), np.nan) for arm in base["arms"] for seed in base["training_seeds"]}
    lengths, decisions, checked = Counter(), [], 0
    for fold in range(1, base["outer_splits"]+1):
        for seed in base["training_seeds"]:
            rows, values, decision = study.evaluate_fold(grid, base, bank, assigned, scope, root, seed, fold, verify=True)
            for arm, p in values.items():
                if np.isfinite(predicted[arm, seed][rows]).any(): raise ValueError("Duplicate length-study patient predictions")
                predicted[arm, seed][rows] = p
            lengths[decision["selected_length"]] += 1
            decisions.append({"outer_fold": fold, "training_seed": seed, "selected_length": decision["selected_length"]})
            checked += matched_checks(root, base, grid, seed, fold, decision["selected_length"])
            _progress(root, state="verifying", training_seed=seed, outer_fold=fold, sequence_length=decision["selected_length"])
    metrics = {arm: {} for arm in base["arms"]}
    for (arm, seed), p in predicted.items():
        study.legacy.check_p(p, len(bank.ids), len(base["classes"]))
        metrics[arm][str(seed)] = scores(bank.labels, p, base["classes"])
        frame = pd.DataFrame({"patient_id": bank.ids, "truth": np.asarray(base["classes"])[bank.labels],
                              "prediction": np.asarray(base["classes"])[p.argmax(1)], "fold": assigned})
        for i, label in enumerate(base["classes"]): frame[f"probability_{label}"] = p[:, i]
        frame.to_csv(root/f"patient_predictions_{arm}_s{seed}.csv", index=False)
        save_confusion(bank.labels, p.argmax(1), base["classes"], f"{arm}_s{seed}", root)
    contrasts = {}
    for control in ("shuffled_rr", "ordered_no_rr", "mean_rr"):
        first = [predicted["ordered_rr", seed] for seed in base["training_seeds"]]
        second = [predicted[control, seed] for seed in base["training_seeds"]]
        contrasts[f"ordered_rr_minus_{control}"] = {
            "six_class": paired_differences(bank.labels, first, second, base["bootstrap_draws"], base["seed"]),
            "mi_only": paired_differences(bank.labels, first, second, base["bootstrap_draws"], base["seed"], True, base["classes"].index("HC"))}
    partitions = base["outer_splits"]*len(base["training_seeds"])
    outer = partitions*len(base["arms"])
    total = partitions*len(base["arms"])*(len(grid["lengths"])*base["inner_splits"]+1)
    summary = {"complete": True, "study_fingerprint": scope["study_fingerprint"], "claim_scope": study.CLAIM,
               "n_patients": len(bank.ids), "classes": base["classes"], "training_seeds": base["training_seeds"],
               "candidate_lengths": grid["lengths"], "shared_selected_length_counts": dict(sorted(lengths.items())),
               "length_selection": grid["length_selection"], "length_decisions": decisions,
               "fits": total, "outer_models_restored": outer, "matched_gru_partition_checks": checked,
               "all_query_beats_covered": True, "metrics": metrics, "contrasts": contrasts}
    _save_json(root/"summary.json", summary)
    lines = ["# Patient-nested consecutive-beat length comparison", "", study.CLAIM, "",
             f"All {len(bank.ids)} original patients. {total} fitted contexts verified; {outer} outer models restored; {checked} matched recurrent-partition checks.", "",
             "| Candidate beats | Training windows/epoch | Windows/batch | Inference windows/batch | Shared length selections |", "|---:|---:|---:|---:|---:|"]
    for length in grid["lengths"]:
        cfg = study.context_config(grid, base, length)
        lines.append(f"| {length} | {cfg['windows_per_epoch']} | {cfg['batch_size']} | {cfg['inference_batch_size']} | {lengths[length]} |")
    lines += ["", "Selection counts are training-only decisions over overlapping outer-training sets/seeds, not independent proof of a universally optimal length. No per-length outer-score winner is selected.", "",
              "| Procedure with shared nested length selection | Training seed | Macro F1 | MI-only macro F1 |", "|---|---:|---:|---:|"]
    for arm, values in metrics.items():
        for seed, row in values.items(): lines.append(f"| {arm} | {seed} | {row['macro_f1']:.3f} | {row['mi_only_macro_f1_six_way_predictions']:.3f} |")
        result = [row["macro_f1"] for row in values.values()]
        sd = float(np.std(result, ddof=1)) if len(result)>1 else 0.
        lines.append(f"| {arm}, mean ± sample SD | all | {np.mean(result):.3f} ± {sd:.3f} | — |")
    lines += ["", "| Paired mean-seed difference | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |", "|---|---:|---:|"]
    for name, result in contrasts.items():
        cells = []
        for key in ("six_class", "mi_only"):
            row = result[key];lo, hi = row["descriptive_patient_interval"]
            cells.append(f"{row['mean_seed_delta_macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "## Exact selection and comparison", "",
              "For each outer fold/training seed, every candidate length receives three inner patient fits for each of four arms. The shared length maximizes the mean of ordered_rr and shuffled_rr mean-inner macro F1, with configured-order ties. This is symmetric in the primary order-control pair. All four arms then use that same length for their outer refits; each arm's median inner epochs at that length sets its own fresh outer training duration. Test labels/probabilities never choose the length. RR and mean-pooling comparisons are conditional on this primary-pair selection policy, not independently length-optimized competitors.", "",
              f"Every length uses {grid['beat_slots_per_epoch']} nominal beat slots/epoch, {grid['beat_slots_per_batch']} beat slots/batch and {grid['beat_slots_per_epoch']//grid['beat_slots_per_batch']} optimizer steps. Shorter windows produce more window labels; length also changes recurrent context and aggregation. Equal beat budgets do not make these optimization problems identical. Within a length, recurrent controls share weights at initialization, capacity, normalization and sampled windows at common epochs. There is no claim of identical windows across different lengths.", "",
              "Original measured XYZ, cached R positions, patients, labels and folds are unchanged. Windows are consecutive retained beats from one recording. Preceding detector-RR and availability travel with each beat under shuffling; no fictitious shuffled-timestamp intervals or absolute time inputs. No-RR zeroes those channels but retains the architecture. Mean pooling has different active capacity. Right masks protect tails; every query beat is covered once, window probabilities are weighted by valid-beat count within a record, then records are averaged equally per patient.", "",
              "Report verification rebuilds training-only moments, windows/permutations, stopping and symmetric length choices, and restores every outer model with identical predicted classes and declared probability tolerance. MPS is not guaranteed bitwise deterministic across machines/versions. Partial inner scores are optimized diagnostics, not final project performance.", "",
              "Seed SD describes optimization variation, not independent patients or a generalization confidence interval. Paired class-stratified patient intervals share draws across seeds and condition on fitted predictions, without refits/correction for repeated development, overlapping training folds or multiple comparisons. MI-only scores keep six-way predictions, including MI→HC errors.", "",
              "These PTB folds remain development data. A gain over shuffled control does not alone beat the historical ResNet 0.550. No score target, SOTA claim or independent confirmation is established by length selection. No full fixed-four or fixed-sixteen sweep was used to choose this grid.", ""]
    (root/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _progress(root, state="complete_validated")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--config", default="configs/chronological_lengths_v1.yaml")
    args = parser.parse_args();result = summarize_study(yaml.safe_load(Path(args.config).read_text()))
    print(json.dumps({"complete": result["complete"], "fits": result["fits"]}))
