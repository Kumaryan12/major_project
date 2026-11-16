"""Reconstruct nested residual-context fits and restore every outer model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import residual_context_study as study
from .benchmark import metric_bundle, save_confusion
from .contrastive import _progress
from .neural import _save_json


CONTRASTS = (("ordered_context_aux", "beat_only"), ("ordered_context_aux", "beat_aux"),
             ("ordered_context_aux", "shuffled_context_aux"), ("ordered_context_aux", "ordered_context"),
             ("beat_aux", "beat_only"), ("ordered_context", "beat_only"),
             ("shuffled_context_aux", "shuffled_context"), ("ordered_context", "shuffled_context"))


def patient_weights(y, labels, draws, seed):
    """Class-stratified draws, shared across every model and optimization seed."""
    weights = np.zeros((draws, len(y)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for label in labels:
        pool = np.flatnonzero(y == label)
        if not len(pool): raise ValueError("Missing bootstrap class")
        chosen = rng.choice(pool, (draws, len(pool)), replace=True)
        np.add.at(weights, (np.arange(draws)[:, None], chosen), 1)
    return weights


def weighted_f1(y, prediction, labels, weights):
    actual = np.equal.outer(y, labels).astype(float)
    predicted = np.equal.outer(prediction, labels).astype(float)
    tp = weights@(actual*predicted)
    denominator = weights@(actual+predicted)
    return np.divide(2*tp, denominator, out=np.zeros_like(tp), where=denominator != 0).mean(1)


def paired_contrasts(y, predicted, config):
    results = {f"{a}_minus_{b}": {} for a, b in CONTRASTS}
    classes = config["classes"]
    for name, labels in (("six_class", np.arange(len(classes))),
                         ("mi_only", np.array([i for i, c in enumerate(classes) if c != "HC"]))):
        weights = patient_weights(y, labels, config["bootstrap_draws"], config["seed"])
        select = np.isin(y, labels)
        samples, points = {}, {}
        for arm in config["arms"]:
            values = [predicted[arm, seed].argmax(1) for seed in config["training_seeds"]]
            samples[arm] = np.mean([weighted_f1(y, pred, labels, weights) for pred in values], axis=0)
            points[arm] = float(np.mean([f1_score(y[select], pred[select], labels=labels, average="macro", zero_division=0) for pred in values]))
        for a, b in CONTRASTS:
            results[f"{a}_minus_{b}"][name] = {"mean_seed_delta_macro_f1": points[a]-points[b],
                "descriptive_patient_interval": np.quantile(samples[a]-samples[b], [.025, .975]).tolist()}
    return results


def matched_checks(root, config, seed, fold):
    checked = 0
    for inner in range(config["inner_splits"]+1):
        audits = []
        for arm in config["arms"]:
            name = f"inner_{fold:02d}_{arm}_s{seed}_i{inner}.json" if inner else f"outer_{fold:02d}_{arm}_s{seed}.json"
            audits.append(json.loads((root/name).read_text())["fit"])
        for audit in audits:
            if any(audit[k] != audits[0][k] for k in ("initial_state_sha256", "normalization", "parameter_count")):
                raise ValueError("Residual-context controls are not matched")
        for epoch in range(min(len(a["history"]) for a in audits)):
            rows = [a["history"][epoch] for a in audits]
            for row in rows:
                if any(row[k] != rows[0][k] for k in ("window_plan_sha256", "direct_token_order_sha256", "valid_beat_presentations", "nominal_beat_slots")):
                    raise ValueError("Residual-context controls saw different direct beats")
            if (rows[2]["token_order_sha256"] != rows[3]["token_order_sha256"]
                    or rows[4]["token_order_sha256"] != rows[5]["token_order_sha256"]):
                raise ValueError("Auxiliary-loss controls saw different context order")
        checked += 1
    return checked


def summarize_study(config):
    bank, assigned = study.load_inputs(config);scope = study.make_scope(config, bank)
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text()) != scope:
        raise ValueError("Residual-context report source/config/runtime changed")
    predicted = {(arm, seed): np.full((len(bank.ids), len(config["classes"])), np.nan)
                 for arm in config["arms"] for seed in config["training_seeds"]}
    checked = 0
    for fold in range(1, config["outer_splits"]+1):
        norm_memo = {}
        for seed in config["training_seeds"]:
            for arm in config["arms"]:
                rows, p = study.evaluate_arm(bank, assigned, config, scope, root, arm, seed, fold, verify=True, norm_memo=norm_memo)
                if np.isfinite(predicted[arm, seed][rows]).any(): raise ValueError("Duplicate residual-context patient predictions")
                predicted[arm, seed][rows] = p
                _progress(root, state="verifying", arm=arm, training_seed=seed, outer_fold=fold)
            checked += matched_checks(root, config, seed, fold)
    metrics = {arm: {} for arm in config["arms"]}
    hc = config["classes"].index("HC");mi = bank.labels != hc
    for (arm, seed), p in predicted.items():
        study.cache.check_p(p, len(bank.ids), len(config["classes"]))
        scores = metric_bundle(bank.labels, p, config["classes"])
        scores["mi_only_macro_f1_six_way_predictions"] = float(f1_score(bank.labels[mi], p[mi].argmax(1), labels=[i for i in range(len(config["classes"])) if i != hc], average="macro", zero_division=0))
        metrics[arm][str(seed)] = scores
        frame = pd.DataFrame({"patient_id": bank.ids, "truth": np.asarray(config["classes"])[bank.labels],
                              "prediction": np.asarray(config["classes"])[p.argmax(1)], "fold": assigned})
        for i, label in enumerate(config["classes"]): frame[f"probability_{label}"] = p[:, i]
        frame.to_csv(root/f"patient_predictions_{arm}_s{seed}.csv", index=False)
        save_confusion(bank.labels, p.argmax(1), config["classes"], f"{arm}_s{seed}", root)
    _progress(root, state="paired_patient_intervals")
    contrasts = paired_contrasts(bank.labels, predicted, config)
    outer = config["outer_splits"]*len(config["training_seeds"])*len(config["arms"])
    summary = {"complete": True, "study_fingerprint": scope["study_fingerprint"], "claim_scope": study.CLAIM,
               "n_patients": len(bank.ids), "classes": config["classes"], "training_seeds": config["training_seeds"],
               "sequence_length": config["sequence_length"], "fits": outer*(config["inner_splits"]+1),
               "outer_models_restored": outer, "matched_partition_checks": checked, "all_query_beats_covered": True,
               "metrics": metrics, "contrasts": contrasts}
    lines = ["# Direct beat morphology plus gated context", "", study.CLAIM, "",
             f"All {len(bank.ids)} original patients. {summary['fits']} fits verified, {outer} outer models restored, {checked} matched six-arm partition checks.", "",
             "| Procedure | Seed | Patient macro F1 | MI-only macro F1 |", "|---|---:|---:|---:|"]
    for arm, values in metrics.items():
        for seed, row in values.items(): lines.append(f"| {arm} | {seed} | {row['macro_f1']:.3f} | {row['mi_only_macro_f1_six_way_predictions']:.3f} |")
        result = [v["macro_f1"] for v in values.values()]
        lines.append(f"| {arm}, mean ± sample SD | all | {np.mean(result):.3f} ± {np.std(result, ddof=1) if len(result)>1 else 0.:.3f} | — |")
    lines += ["", "| Paired mean-seed difference | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |", "|---|---:|---:|"]
    for name, result in contrasts.items():
        cells = []
        for key in ("six_class", "mi_only"):
            row = result[key];lo, hi = row["descriptive_patient_interval"]
            cells.append(f"{row['mean_seed_delta_macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "## Protocol and limits", "",
              f"Fixed {config['sequence_length']}-beat consecutive windows. The fixed length was informed by prior inner-fold preferences, not a universally optimal beat count. Original patients, labels, measured XYZ, R positions and frozen ten outer/three inner patient folds are retained. Every fit starts fresh; no saved outer encoder is reused across inner validation patients.", "",
              "Every arm retains the same 64-dimensional encoder and direct per-beat class head. A masked GRU summarizes waveform/RR pairs into a 32-dimensional state; a learned scalar sigmoid gate multiplies a contextual class-logit correction. The correction is added to each direct beat logit, softmax is applied per beat, and real-beat probabilities are averaged within the window. In the beat-only controls the gate is forced to zero. Registered parameter counts and initial states match; the inactive branch means effective capacity is not identical.", "",
              f"The window objective is negative log probability of the true window class. Auxiliary arms add {config['auxiliary_weight']} times direct beat cross entropy, averaged within each real window then across windows. The coefficient and initial gate bias {config['gate_initial_bias']} were fixed before this sweep. There is no outer-score tuning of gate, coefficient, seed, ensemble or stopping. Direct beats/dropout assignments stay in their original order; only context embedding/RR pairs are permuted in shuffled controls, with timing attached and missingness retained.", "",
              f"All arms draw the same class/patient-balanced, uniform-record contiguous windows: {config['windows_per_epoch']} windows, {config['windows_per_epoch']*config['sequence_length']} nominal beat slots and {config['windows_per_epoch']//config['batch_size']} optimizer steps per epoch. Inner patient macro F1 selects the earliest best epoch with patience {config['patience']}, at most {config['max_epochs']} epochs. Its rounded median sets a fresh outer refit duration. Normalization is training-only. Query inference covers every beat once with masked tails, beat-count weighting within each recording, then equal-record patient averaging.", "",
              "Combined-recipe versus beat_only includes both context and auxiliary supervision. ordered_context_aux versus beat_aux tests adding contextual capacity/RR with the auxiliary coefficient matched; its shuffled counterpart controls chronological order. ordered_context_aux versus ordered_context tests the auxiliary term. Preserve all declared comparisons rather than selecting the best seed or arm as confirmation.", "",
              f"Intervals use {config['bootstrap_draws']:,} paired class-stratified patient bootstrap draws shared across optimization seeds and arms, conditional on saved predictions. No model refits, correction for overlapping training folds, repeated development or multiple comparisons. Seed SD is optimization variability, not a generalization confidence interval. MI-only scores retain original six-way predictions, including MI→HC errors.", "",
              "Historical ResNet 0.550 and previous GRU 0.524 use different procedures, not fresh matched controls for this recipe. These repeatedly inspected PTB patients remain development data. No SOTA, independent confirmation, causal mechanism or 0.80–0.90 target achievement follows from an architecture name. Raw inputs, individual predictions and fitted weights stay local.", ""]
    (root/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _save_json(root/"summary.json", summary)
    _progress(root, state="complete_validated")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--config", default="configs/residual_context_v1.yaml")
    args = parser.parse_args();summary = summarize_study(yaml.safe_load(Path(args.config).read_text()))
    print(json.dumps({"complete": summary["complete"], "fits": summary["fits"]}))
