"""Replay bootstrap selectors and independently validate nested predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import stability_study as study
from .benchmark import inner_patient_splits, metric_bundle
from .neural import _save_json
from .neural_report import _read_predictions
from .relational_report import paired_bootstrap
from .signal_audit import file_hash


def summarize_study(config):
    source,ref_config,features,y,ids,names,folds,inherited = study.load_inputs(config)
    scope = study.make_scope(config,source,ref_config,inherited)
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text())!=scope:
        raise ValueError("Stability report source/settings/code changed")
    frozen = folds.set_index("patient_id").sort_index()
    assignment = frozen.loc[ids,"fold"].to_numpy()
    metrics,predictions,counts = {},{},{}
    memo,support = {},[]
    fresh_inner,reused_inner,outer_refits,reference_matches = 0,0,0,0
    for learner in ref_config["learners"]:
        grid = study.candidates(config,source,ref_config,learner)
        for method in config["selectors"]:
            arm = f"{learner}_{method}"
            pooled = np.full((len(ids),len(source["classes"])),np.nan)
            outer_audits = []
            for fold in range(1,source["outer_splits"]+1):
                train,test = np.flatnonzero(assignment!=fold),np.flatnonzero(assignment==fold)
                splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],ids[train],source["inner_splits"],config["split_seed"]+fold*1000)]
                rows = []
                for candidate_id,candidate in enumerate(grid):
                    X = features[candidate["family"]]
                    scores,digests = [],[]
                    for inner,(ti,vi) in enumerate(splits,1):
                        selector,_,expected = study.prepare_selector(X,y,ids,ti,vi,names[candidate["family"]],candidate,method,config,source,config["seed"]+fold*1000+inner,root,memo,verify=True)
                        expected.update(study_fingerprint=scope["study_fingerprint"],outer_fold=fold,inner_fold=inner,candidate_id=candidate_id,candidate=candidate,method=method,learner=learner)
                        path = root/f"inner_{fold:02d}_{learner}_{method}_{candidate_id:02d}_{inner}.json"
                        audit = json.loads(path.read_text())
                        if any(audit.get(key)!=value for key,value in expected.items()) or audit["new_fit"]!=(method!="original"):
                            raise ValueError("Stability inner training selector or fit attribution changed")
                        p = study.reference.read_prediction(path.with_suffix(".npz"),audit,vi,len(source["classes"]))
                        if method=="original":
                            old_path = study.original_checkpoint(scope,source,ref_config,learner,candidate,fold,inner)
                            if audit["reference_audit_sha256"]!=file_hash(old_path):
                                raise ValueError("Stability reused original audit changed")
                            original = json.loads(old_path.read_text())
                            _,_,old_expected = study.reference.prepare_fit(X,y,ids,ti,vi,candidate,source)
                            if original["study_fingerprint"]!=scope["reference"]["reference_scope"]["study_fingerprint"] or any(original.get(key)!=value for key,value in old_expected.items()) or not np.array_equal(p,study.reference.read_prediction(old_path.with_suffix(".npz"),original,vi,len(source["classes"]))):
                                raise ValueError("Stability original predictions/scope mismatch")
                            reused_inner += 1
                        else:
                            fresh_inner += 1
                        score = float(f1_score(y[vi],p.argmax(1),labels=np.arange(len(source["classes"])),average="macro",zero_division=0))
                        if audit["validation_macro_f1"]!=score:
                            raise ValueError("Stability inner score disagrees with predictions")
                        scores.append(score); digests.append(file_hash(path))
                    rows.append({"candidate_id":candidate_id,"candidate":candidate,"mean_inner_macro_f1":float(np.mean(scores)),"inner_audits_sha256":digests})
                selection_path = root/f"selection_{fold:02d}_{arm}.json"
                if json.loads(selection_path.read_text())!={"study_fingerprint":scope["study_fingerprint"],"candidates":rows}:
                    raise ValueError("Stability candidate selection disagrees with inner fits")
                chosen = max(range(len(rows)),key=lambda i:rows[i]["mean_inner_macro_f1"])
                candidate = rows[chosen]["candidate"]
                X = features[candidate["family"]]
                selector,_,expected = study.prepare_selector(X,y,ids,train,test,names[candidate["family"]],candidate,method,config,source,config["seed"]+fold,root,memo,verify=True)
                expected.update(study_fingerprint=scope["study_fingerprint"],fold=fold,candidate_id=chosen,candidate=candidate,method=method,learner=learner,selection_sha256=file_hash(selection_path))
                path = root/f"fold_{fold:02d}_{arm}.json"
                audit = json.loads(path.read_text())
                if any(audit.get(key)!=value for key,value in expected.items()) or audit["selected_feature_names"]!=[names[candidate["family"]][i] for i in selector.selected]:
                    raise ValueError("Stability outer selector/feature choice changed")
                p = study.reference.read_prediction(path.with_suffix(".npz"),audit,test,len(source["classes"]))
                model_path = root/f"model_{fold:02d}_{arm}.joblib"
                if audit["model_sha256"]!=file_hash(model_path):
                    raise ValueError("Stability saved model changed")
                model = joblib.load(model_path)
                if any(model.get(key)!=value for key,value in {"study_fingerprint":scope["study_fingerprint"],"fold":fold,"method":method,"learner":learner,"candidate":candidate}.items()) or model["selector"].audit()!=selector.audit() or not np.array_equal(model["estimator"].classes_,np.arange(len(source["classes"]))):
                    raise ValueError("Stability saved model provenance/selector changed")
                if not np.allclose(study.reference.predict(model,X[test]),p,rtol=1e-12,atol=1e-12) or audit["patient_metrics"]!=metric_bundle(y[test],p,source["classes"]):
                    raise ValueError("Stability fitted model does not reproduce outer predictions")
                if method=="original":
                    previous_path = Path(inherited["reference_root"])/f"fold_{fold:02d}_{learner}_legacy.json"
                    previous = json.loads(previous_path.read_text())
                    if previous["candidate"]==candidate:
                        old_p = study.reference.read_prediction(previous_path.with_suffix(".npz"),previous,test,len(source["classes"]))
                        if not np.allclose(p,old_p,rtol=1e-12,atol=1e-12):
                            raise ValueError("Fresh original control differs from matching prior fit")
                        reference_matches += 1
                pooled[test] = p
                outer_audits.append(audit)
                outer_refits += 1
            saved = json.loads((root/f"metrics_{arm}.json").read_text())
            if not saved["complete"] or saved["study_fingerprint"]!=scope["study_fingerprint"] or saved["classes"]!=source["classes"] or saved["n_patients"]!=len(ids) or saved["folds"]!=outer_audits or saved["claim_scope"]!=study.CLAIM or saved["patient_metrics"]!=metric_bundle(y,pooled,source["classes"]):
                raise ValueError("Stability pooled metrics mismatch")
            truth,prediction = _read_predictions(root/f"patient_predictions_{arm}.csv",saved,frozen,source["classes"])
            frame = pd.read_csv(root/f"patient_predictions_{arm}.csv").set_index("patient_id").loc[ids]
            if not np.array_equal(truth,y) or not np.array_equal(prediction,pooled.argmax(1)) or not np.allclose(frame[[f"probability_{label}" for label in source["classes"]]].to_numpy(),pooled,rtol=1e-7,atol=1e-12):
                raise ValueError("Stability CSV differs from outer predictions")
            metrics[arm],predictions[arm] = saved["patient_metrics"],prediction
            counts[arm] = dict(Counter(audit["candidate"]["family"] for audit in outer_audits))
    # Compare support at the same family and budget, not only the differently
    # tuned winning packages. This is support under a new base-selector panel,
    # not robustness of an entire retrained consensus procedure.
    for fold in range(1,source["outer_splits"]+1):
        train,test = np.flatnonzero(assignment!=fold),np.flatnonzero(assignment==fold)
        for family in config["families"]:
            for budget in source["feature_counts"]:
                candidate = {"family":family,"k":budget,"params":{"C":source["linear_c"][0]}}
                for method in config["selectors"]:
                    selector,_,_ = study.prepare_selector(features[family],y,ids,train,test,names[family],candidate,method,config,source,config["seed"]+fold,root,memo,verify=True)
                    a = selector.audit()
                    support.append({"outer_training_context":fold,"family":family,"budget":budget,"method":method,
                        "n_selected":a["n_selected"],"n_selected_groups":a["n_selected_groups"],
                        "training_feature_support":float(np.mean(a["training_selection_frequencies"])),
                        "audit_feature_support":float(np.mean(a["audit_selection_frequencies"])),
                        "audit_group_support":float(np.mean(a["audit_group_frequencies"]))})
    pd.DataFrame(support).to_csv(root/"resampling_support.csv",index=False)
    support_means = pd.DataFrame(support).groupby(["family","budget","method"],sort=False)[["n_selected","n_selected_groups","training_feature_support","audit_feature_support","audit_group_support"]].mean().reset_index().to_dict("records")
    intervals,contrasts = {},{}
    for learner in ref_config["learners"]:
        arms = {arm:prediction for arm,prediction in predictions.items() if arm.startswith(learner+"_")}
        intervals[learner] = paired_bootstrap(y,arms,len(source["classes"]),draws=config["bootstrap_draws"],seed=config["seed"],references=(f"{learner}_original",f"{learner}_bootstrap"))
        for second,first in (("bootstrap","original"),("group_bootstrap","original"),("group_bootstrap","bootstrap")):
            a,b = f"{learner}_{second}",f"{learner}_{first}"
            contrasts[f"{a}_minus_{b}"] = {"macro_f1_difference":metrics[a]["macro_f1"]-metrics[b]["macro_f1"],
                "descriptive_95_interval":intervals[learner]["arms"][a]["paired_comparisons"][b]["macro_f1_difference_95_interval"]}
    summary = {"complete":True,"study_fingerprint":scope["study_fingerprint"],"n_patients":len(ids),
        "new_inner_fits":fresh_inner,"reused_original_inner_fits":reused_inner,"outer_refits":outer_refits,
        "total_fit_slots":fresh_inner+reused_inner+outer_refits,"bootstrap_caches_fully_replayed":len(memo),
        "resampled_base_selector_fits":len(memo)*(config["bootstrap_replicates"]+config["audit_replicates"]),
        "matching_reference_outer_predictions":reference_matches,"metrics":metrics,"contrasts":contrasts,
        "selected_families":counts,"matched_support_means":support_means,"claim_scope":study.CLAIM}
    _save_json(root/"summary.json",summary)
    _save_json(root/"paired_bootstrap.json",{"study_fingerprint":scope["study_fingerprint"],"learners":intervals})
    lines = ["# Feature consensus: completed patient evaluation","",study.CLAIM,"",
        f"All {len(ids)} original patients, ten frozen outer folds and three inner folds. Validated {summary['total_fit_slots']} classifier fit slots: {fresh_inner} new inner fits, {reused_inner} verified original inner fits and {outer_refits} fresh outer refits. Fully replayed {len(memo)} bootstrap caches ({summary['resampled_base_selector_fits']} base-selector fits).", "",
        "| Selector | Linear patient macro F1 | Shallow-tree patient macro F1 |","|---|---:|---:|"]
    lines += [f"| {method} | {metrics['linear_'+method]['macro_f1']:.3f} | {metrics['tree_'+method]['macro_f1']:.3f} |" for method in config["selectors"]]
    lines += ["","Pooled out-of-fold patient scores; one fixed model/resampling seed, not a neural three-seed ensemble. All arms and patients retained.","",
        "## Paired comparisons","","| Difference | Δ macro F1 | Descriptive patient interval |","|---|---:|---:|"]
    for key,value in contrasts.items():
        lo,hi = value["descriptive_95_interval"]
        lines.append(f"| {key} | {value['macro_f1_difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["",f"{config['bootstrap_draws']:,} class-stratified paired patient draws, conditional on fitted predictions; no model refits or adjustment for correlated folds, repeated development or multiple comparisons. Descriptive intervals, not independent confirmation.","",
        "## Matched resampling support","","| Family | Budget | Selector | Actual features | Audit feature support | Audit group support |","|---|---:|---|---:|---:|---:|"]
    for row in support_means:
        lines.append(f"| {row['family']} | {row['budget']} | {row['method']} | {row['n_selected']:.1f} | {row['audit_feature_support']:.3f} | {row['audit_group_support']:.3f} |")
    lines += ["","Support is the fraction of 32 separate training-patient bootstrap base-selector fits that include a chosen feature/group, averaged over chosen columns and matched outer-training contexts. The audit panel never ranks final features or tunes candidates. This measures resampling support against the base selector, not stability of retraining the complete consensus algorithm, independent biological importance or false-discovery control. Training contexts overlap; no independent stability probability is claimed.","",
        "## Exact method and boundaries","",
        "Fixed pool: original signed morphology and combined summaries only; no new tensor, feature family, beat, patient, label, time alignment or preprocessing. Original uses training ANOVA ranking and Pearson pruning. Bootstrap ranks by selection frequency over 64 class-stratified patient bootstrap fits, then original F score and index. Group bootstrap first ranks fixed semantic groups by any-member selection frequency, then chooses representative columns; at most one column per group. Groups preserve source family, median/IQR, XYZ, wavelet band/window/statistic or coefficient order and merge only corresponding physical-view slots. These groups are a hypothesis, not proof of clinically equivalent features.","",
        "All methods fit class-balanced scaling on training patients, keep at most 24/64 features, and remove full-training correlations ≥0.95. Bootstrap draws retain patient multiplicity and original class counts. Training and audit random streams are separate and shared across classifiers/candidates in a partition. No minimum-frequency threshold is tuned; consensus can retain low-support columns to honor the same maximum budget. The original selector's probabilities are not altered by its descriptive bootstrap audit.","",
        "Each method/learner has the same eight family/budget/model candidates. Inner patient macro F1 selects the candidate; exact ties favor configured order. Outer selectors/models are fresh training-only refits. Linear C 0.1/1.0; tree depth 2/3, 100 trees, rate 0.05, lambda 1, class-balanced weights applied once, one CPU thread. Source templates use held-out recordings' own unlabeled beats for offline within-exam context. Every saved model must reproduce its outer predictions.","",
        f"Original-control outer predictions matched {reference_matches} compatible prior fits. The pool was chosen using previous development findings; the same repeatedly inspected patients are not independent confirmation. No classical stability-selection theorem, clinical biomarker, SOTA or 0.80–0.90 score is guaranteed.","",
        "## Per-class F1","","| Class | "+" | ".join(metrics)+" |","|---|"+"---:|"*len(metrics)]
    for label in source["classes"]:
        lines.append(f"| {label} | "+" | ".join(f"{value['classification_report'][label]['f1-score']:.3f}" for value in metrics.values())+" |")
    lines += ["","Reproduce: `.venv/bin/python -m mi_localization.stability_study`, then `.venv/bin/python -m mi_localization.stability_report`. Feature matrices, bootstrap selections/draws, fitted models and patient predictions remain local; aggregate metrics/audits are public.",""]
    (root/"REPORT.md").write_text("\n".join(lines))
    return summary


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/feature_stability.yaml")
    args = parser.parse_args()
    print(json.dumps(summarize_study(yaml.safe_load(Path(args.config).read_text())),indent=2))
