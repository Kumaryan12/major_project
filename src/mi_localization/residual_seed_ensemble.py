"""Post-hoc equal-weight inference over every declared residual-model seed.

Replays source verification and independently restores inference members. No
training, weight fitting, arm/seed subset search, calibration or label routing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

from . import residual_context_study as study, residual_context_report as source_report
from .benchmark import metric_bundle, save_confusion
from .chronology_diagnostics import validate_predictions
from .contrastive import _progress
from .neural import _device, _hash, _save_json
from .signal_audit import file_hash

CLAIM = ("Post-hoc uniform seed-ensemble development analysis after inspecting source results; "
         "all declared seeds and six arms retained, no weight/subset tuning, no new training or independent confirmation.")


def uniform_probability_mean(members):
    values = np.asarray(members, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1 or values.shape[2] < 2
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(2), 1, atol=1e-5, rtol=0)):
        raise ValueError("Require at least two aligned, finite probability matrices")
    return values.mean(0)


def source_inputs(config):
    if set(config) != {"source_config", "output_dir", "operation", "bootstrap_draws", "bootstrap_seed"}:
        raise ValueError("No seed, arm, temperature or weight-selection options are allowed")
    if config["operation"] != "uniform_probability_mean" or config["bootstrap_draws"] < 100:
        raise ValueError("Require uniform probabilities and declared patient draws")
    base = yaml.safe_load(Path(config["source_config"]).read_text())
    if len(base["training_seeds"]) < 2 or len(set(base["training_seeds"])) != len(base["training_seeds"]):
        raise ValueError("Need all unique declared source seeds")
    bank, assigned = study.load_inputs(base)
    source = Path(base["output_dir"])/"complete"
    scope = study.make_scope(base, bank)
    if json.loads((source/"study_manifest.json").read_text()) != scope:
        raise ValueError("Source model/data/config/runtime changed")
    summary = json.loads((source/"summary.json").read_text())
    outer = base["outer_splits"]*len(base["training_seeds"])*len(base["arms"])
    partitions = base["outer_splits"]*len(base["training_seeds"])*(base["inner_splits"]+1)
    if (not (source/"REPORT.md").is_file() or summary.get("complete") is not True
            or summary.get("study_fingerprint") != scope["study_fingerprint"]
            or summary.get("outer_models_restored") != outer or summary.get("fits") != outer*(base["inner_splits"]+1)
            or summary.get("matched_partition_checks") != partitions or summary.get("all_query_beats_covered") is not True
            or summary.get("classes") != base["classes"] or summary.get("training_seeds") != base["training_seeds"]
            or summary.get("n_patients") != len(bank.ids)):
        raise ValueError("Require a complete verified source study with every declared seed/arm")
    return base, bank, assigned, source, scope, summary


def predict_heldout_fold(bank, assigned, config, source, scope, arm, fold, norm_memo=None):
    """Restore every seed for one arm; infer only the frozen held-out patients.

    Caller obtains bank/scope from source_inputs. There is deliberately no seed
    list, probability weight, patient routing or query-label argument.
    """
    if config != scope["config"] or arm not in config["arms"] or fold not in range(1, config["outer_splits"]+1):
        raise ValueError("Invalid ensemble source arm/fold/config")
    if norm_memo is None: norm_memo = {}
    train, query = bank.ids[assigned != fold], bank.ids[assigned == fold]
    if not len(query) or set(train) & set(query): raise ValueError("Invalid held-out query membership")
    key = _hash(train.tolist())
    if key not in norm_memo: norm_memo[key] = study.chronology_model.SequenceNormalizer().fit(bank, train)
    norm = norm_memo[key]
    probabilities, audits = [], []
    for seed in config["training_seeds"]:
        path = source/f"outer_{fold:02d}_{arm}_s{seed}.json"
        audit = json.loads(path.read_text())
        expected = dict(study.expected_fit(bank, scope, arm, seed, fold, 0, train, query),
                        final_epochs=audit["final_epochs"], inner_choices=audit["inner_choices"])
        _, cached = study.cache.read_result(path, expected, query, len(config["classes"]), outer=True)
        state = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        if (state["expected"] != expected or state["normalization"] != norm.audit()
                or audit["fit"]["normalization"] != norm.audit() or audit["fit"]["seed"] != seed+fold):
            raise ValueError("Ensemble member has incompatible membership/normalization")
        model = study.models.initial_model(config, seed+fold, arm)
        model.load_state_dict(state["state_dict"]);model.to(_device(config))
        p, coverage = study.models.predict(model, norm, bank, query, config, seed+fold)
        if (coverage != audit["coverage"] or not np.array_equal(p.argmax(1), cached.argmax(1))
                or not np.allclose(p, cached, rtol=1e-5, atol=1e-6)):
            raise ValueError("Ensemble restored member differs from verified source")
        probabilities.append(p)
        audits.append({"training_seed": seed, "source_audit_sha256": file_hash(path), "model_sha256": file_hash(path.with_suffix(".pt")),
                       "query_membership_sha256": expected["query_membership_sha256"], "patient_overlap": 0,
                       "normalization_sha256": _hash(norm.audit()), "coverage": coverage, "restore_verified": True})
        del model
    return np.flatnonzero(assigned == fold), uniform_probability_mean(probabilities), audits


def ensemble_contrasts(y, ensembles, members, config, base):
    """Use the same class-stratified patients across ensembles and members."""
    results, gains = {}, {}
    for metric, labels in (("six_class", np.arange(len(base["classes"]))),
                           ("mi_only", np.array([i for i,c in enumerate(base["classes"]) if c != "HC"]))):
        weights = source_report.patient_weights(y, labels, config["bootstrap_draws"], config["bootstrap_seed"])
        select = np.isin(y, labels)
        samples, points = {}, {}
        for arm, p in ensembles.items():
            prediction = p.argmax(1)
            samples[arm] = source_report.weighted_f1(y, prediction, labels, weights)
            points[arm] = float(f1_score(y[select], prediction[select], labels=labels, average="macro", zero_division=0))
            member_samples = np.mean([source_report.weighted_f1(y, members[arm, seed].argmax(1), labels, weights)
                                      for seed in base["training_seeds"]], axis=0)
            member_point = float(np.mean([f1_score(y[select], members[arm, seed][select].argmax(1), labels=labels, average="macro", zero_division=0)
                                          for seed in base["training_seeds"]]))
            gains.setdefault(arm, {})[metric] = {"ensemble_minus_mean_member_f1": points[arm]-member_point,
                "descriptive_patient_interval": np.quantile(samples[arm]-member_samples, [.025,.975]).tolist()}
        for a,b in source_report.CONTRASTS:
            results.setdefault(f"{a}_minus_{b}", {})[metric] = {"ensemble_delta_macro_f1": points[a]-points[b],
                "descriptive_patient_interval": np.quantile(samples[a]-samples[b], [.025,.975]).tolist()}
    return results, gains


def run(config):
    base, bank, assigned, source, scope, completed = source_inputs(config)
    root = Path(config["output_dir"])/"complete";root.mkdir(parents=True, exist_ok=True)
    paths = [source/"study_manifest.json", source/"summary.json", source/"REPORT.md", Path(config["source_config"])]
    for pattern in ("inner_*.json", "inner_*.npz", "outer_*.json", "outer_*.npz", "outer_*.pt", "patient_predictions_*.csv"):
        paths.extend(sorted(source.glob(pattern)))
    manifest = {"config": config, "claim_scope": CLAIM, "source_fingerprint": scope["study_fingerprint"],
                "code_sha256": file_hash(__file__), "prediction_reader_sha256": file_hash(Path(__file__).with_name("chronology_diagnostics.py")),
                "source_file_sha256": {str(p): file_hash(p) for p in paths}}
    guard = root/"analysis_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != manifest:
        raise ValueError("Ensemble source/code/settings changed; use a new namespace")
    _save_json(guard, manifest)
    frozen = pd.read_csv(base["folds_path"]).set_index("patient_id").sort_index()
    members = {}
    for arm in base["arms"]:
        for seed in base["training_seeds"]:
            p = validate_predictions(pd.read_csv(source/f"patient_predictions_{arm}_s{seed}.csv"), frozen, base["classes"],
                                     completed["metrics"][arm][str(seed)]["macro_f1"])
            if not np.array_equal(frozen.index, bank.ids): raise ValueError("Source patient order changed")
            members[arm, seed] = p
    ensembles = {a: np.full((len(bank.ids), len(base["classes"])), np.nan) for a in base["arms"]}
    checked, restored, gates = 0, 0, {a: [] for a in base["arms"]}
    for fold in range(1, base["outer_splits"]+1):
        norm_memo = {}
        # Rebuild all source inner choices/stopping and restore source outer
        # models, without calling its report writer or changing its artifacts.
        for seed in base["training_seeds"]:
            for arm in base["arms"]:
                rows, cached = study.evaluate_arm(bank, assigned, base, scope, source, arm, seed, fold,
                                                  verify=True, norm_memo=norm_memo)
                if not np.allclose(cached, members[arm, seed][rows], atol=1e-12, rtol=0):
                    raise ValueError("Source CSV differs from model prediction checkpoint")
                audit = json.loads((source/f"outer_{fold:02d}_{arm}_s{seed}.json").read_text())
                gates[arm].append(audit["fit"]["history"][-1]["gate_mean"])
            checked += source_report.matched_checks(source, base, seed, fold)
        for arm in base["arms"]:
            rows, p, audit = predict_heldout_fold(bank, assigned, base, source, scope, arm, fold, norm_memo)
            if np.isfinite(ensembles[arm][rows]).any(): raise ValueError("Duplicate ensemble query predictions")
            expected = uniform_probability_mean([members[arm, seed][rows] for seed in base["training_seeds"]])
            if not np.array_equal(p.argmax(1), expected.argmax(1)) or not np.allclose(p, expected, atol=1e-6, rtol=1e-5):
                raise ValueError("Restored ensemble differs from source probability average")
            ensembles[arm][rows] = p;restored += len(audit)
            _save_json(root/f"members_{fold:02d}_{arm}.json", {"source_fingerprint": scope["study_fingerprint"],
                "arm": arm, "outer_fold": fold, "weights": [1/len(audit)]*len(audit), "members": audit})
            _progress(root, state="verifying_ensemble", arm=arm, outer_fold=fold, restored_members=restored)
        print(f"Uniform seed ensemble: outer fold {fold}/{base['outer_splits']} verified; no new fits.", flush=True)
    metrics = {};hc = base["classes"].index("HC");mi = bank.labels != hc
    for arm,p in ensembles.items():
        study.cache.check_p(p, len(bank.ids), len(base["classes"]))
        result = metric_bundle(bank.labels, p, base["classes"])
        result["mi_only_macro_f1_six_way_predictions"] = float(f1_score(bank.labels[mi], p[mi].argmax(1),
            labels=[i for i in range(len(base["classes"])) if i != hc], average="macro", zero_division=0))
        metrics[arm] = result
        frame = pd.DataFrame({"patient_id":bank.ids,"truth":np.asarray(base["classes"])[bank.labels],
                              "prediction":np.asarray(base["classes"])[p.argmax(1)],"fold":assigned})
        for i,c in enumerate(base["classes"]): frame[f"probability_{c}"] = p[:,i]
        frame.to_csv(root/f"patient_predictions_{arm}.csv", index=False)
        save_confusion(bank.labels,p.argmax(1),base["classes"],arm,root)
    contrasts, gains = ensemble_contrasts(bank.labels, ensembles, members, config, base)
    summary = {"complete": True, "claim_scope": CLAIM, "source_fingerprint": scope["study_fingerprint"],
               "analysis_fingerprint": _hash(manifest), "n_patients": len(bank.ids), "classes": base["classes"],
               "training_seeds": base["training_seeds"], "new_training_fits": 0, "outer_models_restored_for_ensemble": restored,
               "source_contexts_reverified": completed["fits"], "source_matched_partition_checks": checked,
               "all_query_beats_covered": True, "metrics": metrics,
               "mean_member_f1": {a:float(np.mean([completed['metrics'][a][str(s)]['macro_f1'] for s in base['training_seeds']])) for a in base["arms"]},
               "contrasts": contrasts, "ensemble_minus_mean_member": gains,
               "median_outer_final_training_gate": {a:float(np.median(v)) for a,v in gates.items()}}
    write_report(root, summary, config)
    _save_json(root/"summary.json", summary)
    _progress(root, state="complete_validated")
    return summary


def write_report(root, summary, config):
    lines = ["# Uniform seed ensembles of direct-beat/context models", "", summary["claim_scope"], "",
             f"All {summary['n_patients']} original patients; every source seed {summary['training_seeds']} and all six source arms. {summary['source_contexts_reverified']} source contexts reverified; {summary['outer_models_restored_for_ensemble']} unique outer members independently restored again for ensemble inference. No new training fits. Source-study artifacts were read only.", "",
             "| Procedure | Mean member macro F1 | Uniform ensemble macro F1 | Ensemble MI-only macro F1 |", "|---|---:|---:|---:|"]
    for arm,m in summary["metrics"].items():
        lines.append(f"| {arm} | {summary['mean_member_f1'][arm]:.3f} | {m['macro_f1']:.3f} | {m['mi_only_macro_f1_six_way_predictions']:.3f} |")
    lines += ["", "Mean member F1 averages three independently optimized model scores. Ensemble F1 scores one predictor formed by averaging all three probability vectors; these are different estimands. The ensemble has three times the member inference cost and uses three fitted models per fold. No best seed, arm subset, mixing coefficient, class threshold or temperature was selected or fitted.", "",
              "## Matched ensemble comparisons", "", "| Comparison | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |", "|---|---:|---:|"]
    for name,v in summary["contrasts"].items():
        cells=[]
        for k in ["six_class","mi_only"]:
            r=v[k];lo,hi=r["descriptive_patient_interval"];cells.append(f"{r['ensemble_delta_macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "## Class F1", "", "| Class | Beat-only ensemble | Ordered+aux ensemble | Shuffled+aux ensemble |", "|---|---:|---:|---:|"]
    for c in summary["classes"]:
        cells=[f"{summary['metrics'][a]['classification_report'][c]['f1-score']:.3f}" for a in ["beat_only","ordered_context_aux","shuffled_context_aux"]]
        lines.append(f"| {c} | {' | '.join(cells)} |")
    lines += ["", "## Interpretation", "",
              "The source single-model procedure did not demonstrate a gain: ordered+aux mean 0.539 versus matched beat-only 0.551. This follow-up changes the predictor to a uniform seed ensemble; it does not retroactively change that conclusion or establish chronological superiority. Averaging was examined after viewing source results, so this is explicitly post-hoc development work. All six arms are disclosed, including ensembles that do not improve their member mean. A promising arm's point score is not independent confirmation or SOTA.", "",
              "Predictors are averaged only within the same arm and outer fold; each member excludes that patient and all of their records during training. The inference helper has no patient-label, seed-subset or weight argument. Test labels are used only for aggregate evaluation. Per-member normalization, original context permutations, complete beat coverage and probability restoration are checked. Individual patient predictions and model files remain local.", "",
              f"Intervals use {config['bootstrap_draws']:,} class-stratified paired patient draws shared across all ensembles and their members. No refits or correction for overlapping folds, repeated development, post-hoc analysis or multiple comparisons. A difference interval containing zero does not demonstrate improvement. MI-only metrics retain six-way predictions and MI→HC mistakes. Ensembles have one F1 score, not a seed SD.", "",
              "The initial small gate was not a lasting constraint: contextual outer final training gates had high medians (stored in summary.json). Gate scale and context-head scale interact; that diagnostic is not physiological importance or proof of why a model failed. No gate threshold was tuned in this analysis.", "",
              "Keep the ensemble as a reproducible development candidate. Before a strong generalization claim, freeze the recipe and evaluate a genuinely independent, label-compatible patient cohort. Further tuning on these repeatedly inspected patients remains development. No guaranteed 0.80–0.90 claim is justified.", "",
              "## Reproduce", "", "```sh", ".venv/bin/python -m mi_localization.residual_seed_ensemble", "```", "",
              "`predict_heldout_fold` provides restored-model ensemble inference for the frozen held-out fold; it deliberately cannot score training patients as held-out cases. The reproduction command replays source verification without invoking a trainer or rewriting completed source reports.", ""]
    (root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default="configs/residual_seed_ensemble_v1.yaml")
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    try:
        result=run(config)
        print(json.dumps({"complete":result["complete"],"new_training_fits":0,"output_dir":config["output_dir"]}))
    except Exception as error:
        _progress(Path(config["output_dir"])/"complete",state="failed",error_type=type(error).__name__,error=str(error));raise
