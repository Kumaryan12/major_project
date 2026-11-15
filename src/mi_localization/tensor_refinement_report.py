"""Independently validate all tensor-refinement fits, models and outer scores."""
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

from . import tensor_refinement_study as study
from .benchmark import inner_patient_splits, metric_bundle
from .neural import _save_json
from .neural_report import _read_predictions
from .relational_report import paired_bootstrap
from .signal_audit import file_hash


def summarize_study(config):
    source,features,y,ids,names,folds,provenance = study.load_inputs(config)
    root = Path(config["output_dir"])/"complete"
    scope = study.make_scope(config,source,provenance)
    if json.loads((root/"study_manifest.json").read_text())!=scope:
        raise ValueError("Refinement report source/config/runtime changed")
    frozen = folds.set_index("patient_id").sort_index()
    assigned = frozen.loc[ids,"fold"].to_numpy()
    fit_slots,new_inner,reused,outer_count = 0,0,0,0
    metrics,predictions,selection_counts = {},{},{}
    for learner in config["learners"]:
        grid = study.candidates(config,source,learner)
        probabilities = {procedure:np.full((len(ids),len(source["classes"])),np.nan) for procedure in study.PROCEDURES}
        audits = {procedure:[] for procedure in study.PROCEDURES}
        for fold in range(1,source["outer_splits"]+1):
            train,test = np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
            splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],ids[train],source["inner_splits"],config["split_seed"]+fold*1000)]
            selection_path = root/f"selection_{fold:02d}_{learner}.json"
            saved = json.loads(selection_path.read_text())
            if saved["study_fingerprint"]!=scope["study_fingerprint"] or len(saved["candidates"])!=len(grid):
                raise ValueError("Incomplete refinement inner selection")
            candidate_rows = []
            for candidate_id,candidate in enumerate(grid):
                scores,digests = [],[]
                X = features[candidate["family"]]
                for inner,(ti,vi) in enumerate(splits,1):
                    path = root/f"inner_{fold:02d}_{learner}_{candidate_id:02d}_{inner}.json"
                    audit = json.loads(path.read_text())
                    _,_,expected = study.prepare_fit(X,y,ids,ti,vi,candidate,source)
                    expected.update(study_fingerprint=scope["study_fingerprint"],outer_fold=fold,inner_fold=inner,candidate_id=candidate_id,candidate=candidate,learner=learner)
                    if any(audit.get(key)!=value for key,value in expected.items()) or type(audit["new_fit"])!=bool:
                        raise ValueError("Refinement report training selector/membership changed")
                    if not audit["new_fit"]:
                        legacy = scope["bank"]["legacy"]
                        if learner!="linear" or candidate_id>=len(study.old.candidates(source)) or legacy is None:
                            raise ValueError("Invalid cached legacy fit attribution")
                        old_path = Path(legacy["path"])/f"inner_{fold:02d}_{candidate_id:02d}_{inner}.json"
                        if audit["reused_legacy_audit_sha256"]!=file_hash(old_path):
                            raise ValueError("Reused legacy fit audit changed")
                        original = json.loads(old_path.read_text())
                        old_p = study.read_prediction(old_path.with_suffix(".npz"),original,vi,len(source["classes"]))
                        reused += 1
                    else:
                        new_inner += 1
                    p = study.read_prediction(path.with_suffix(".npz"),audit,vi,len(source["classes"]))
                    if not audit["new_fit"] and not np.array_equal(p,old_p):
                        raise ValueError("Reused predictions differ from original probe")
                    score = float(f1_score(y[vi],p.argmax(1),labels=np.arange(len(source["classes"])),average="macro",zero_division=0))
                    if audit["validation_macro_f1"]!=score:
                        raise ValueError("Refinement inner macro F1 changed")
                    scores.append(score)
                    digests.append(file_hash(path))
                    fit_slots += 1
                candidate_rows.append({"candidate_id":candidate_id,"candidate":candidate,"mean_inner_macro_f1":float(np.mean(scores)),"inner_audits_sha256":digests})
            if saved["candidates"]!=candidate_rows:
                raise ValueError("Refinement inner selections disagree with complete predictions")
            for procedure in study.PROCEDURES:
                candidate_id,candidate = study.choose(candidate_rows,procedure,source)
                arm = f"{learner}_{procedure}"
                X = features[candidate["family"]]
                path = root/f"fold_{fold:02d}_{arm}.json"
                audit = json.loads(path.read_text())
                selector,_,expected = study.prepare_fit(X,y,ids,train,test,candidate,source)
                expected.update(study_fingerprint=scope["study_fingerprint"],fold=fold,learner=learner,procedure=procedure,candidate_id=candidate_id,candidate=candidate,selection_sha256=file_hash(selection_path))
                if any(audit.get(key)!=value for key,value in expected.items()) or audit["selected_feature_names"]!=[names[candidate["family"]][i] for i in selector.selected]:
                    raise ValueError("Refinement outer selector/selection changed")
                p = study.read_prediction(path.with_suffix(".npz"),audit,test,len(source["classes"]))
                model_path = root/f"model_{fold:02d}_{arm}.joblib"
                if audit["model_sha256"]!=file_hash(model_path):
                    raise ValueError("Refinement fitted model changed")
                model = joblib.load(model_path)
                if any(model.get(key)!=value for key,value in {"study_fingerprint":scope["study_fingerprint"],"fold":fold,"procedure":procedure,"learner":learner,"candidate":candidate}.items()) or model["selector"].audit()!=selector.audit():
                    raise ValueError("Saved refinement model provenance/selector changed")
                if not np.array_equal(model["estimator"].classes_,np.arange(len(source["classes"]))) or not np.allclose(study.predict(model,X[test]),p,rtol=1e-12,atol=1e-12):
                    raise ValueError("Saved refinement model does not reproduce test predictions")
                if audit["patient_metrics"]!=metric_bundle(y[test],p,source["classes"]):
                    raise ValueError("Refinement outer metrics disagree with predictions")
                probabilities[procedure][test] = p
                audits[procedure].append(audit)
                fit_slots += 1
                outer_count += 1
        for procedure,p in probabilities.items():
            arm = f"{learner}_{procedure}"
            study.check_probabilities(p,len(ids),len(source["classes"]))
            saved = json.loads((root/f"metrics_{arm}.json").read_text())
            if not saved["complete"] or saved["study_fingerprint"]!=scope["study_fingerprint"] or saved["classes"]!=source["classes"] or saved["n_patients"]!=len(ids) or saved["claim_scope"]!=study.CLAIM or saved["folds"]!=audits[procedure] or saved["patient_metrics"]!=metric_bundle(y,p,source["classes"]):
                raise ValueError("Refinement pooled metrics mismatch")
            truth,prediction = _read_predictions(root/f"patient_predictions_{arm}.csv",saved,frozen,source["classes"])
            frame = pd.read_csv(root/f"patient_predictions_{arm}.csv").set_index("patient_id").loc[ids]
            if not np.array_equal(truth,y) or not np.array_equal(prediction,p.argmax(1)) or not np.allclose(frame[[f"probability_{label}" for label in source["classes"]]].to_numpy(),p,rtol=1e-7,atol=1e-12):
                raise ValueError("Refinement CSV differs from pooled test predictions")
            metrics[arm] = saved["patient_metrics"]
            predictions[arm] = prediction
            selection_counts[arm] = dict(Counter(audit["candidate"]["family"] for audit in audits[procedure]))
    intervals,contrasts = {},{}
    for learner in config["learners"]:
        arms = {arm:predictions[arm] for arm in predictions if arm.startswith(learner+"_")}
        intervals[learner] = paired_bootstrap(y,arms,len(source["classes"]),draws=config["bootstrap_draws"],seed=config["seed"],references=(f"{learner}_statistics",f"{learner}_legacy"))
        for second,first in (("legacy","statistics"),("refined","statistics"),("refined","legacy")):
            a,b = f"{learner}_{second}",f"{learner}_{first}"
            contrasts[f"{a}_minus_{b}"] = {"macro_f1_difference":metrics[a]["macro_f1"]-metrics[b]["macro_f1"],
                "descriptive_95_interval":intervals[learner]["arms"][a]["paired_comparisons"][b]["macro_f1_difference_95_interval"]}
    summary = {"complete":True,"study_fingerprint":scope["study_fingerprint"],"n_patients":len(ids),"n_records":provenance["refined_bank"]["n_records"],
        "n_beats":provenance["refined_bank"]["n_beats"],"total_fit_slots":fit_slots,"reused_legacy_inner_fits":reused,"new_inner_fits":new_inner,"outer_refits":outer_count,
        "metrics":metrics,"contrasts":contrasts,"selected_families":selection_counts,"claim_scope":study.CLAIM}
    _save_json(root/"summary.json",summary)
    _save_json(root/"paired_bootstrap.json",{"study_fingerprint":scope["study_fingerprint"],"learners":intervals})
    lines = ["# Tensor refinement: nested final classifier results","",study.CLAIM,"",
        f"All {len(ids)} patients, {summary['n_records']} records and {summary['n_beats']:,} original beats; {source['outer_splits']} frozen outer folds, {source['inner_splits']} inner folds. Validated {fit_slots} fit slots: {new_inner} new inner fits, {reused} verified reused legacy probes, {outer_count} fresh outer refits.","",
        "| Learner / procedure | Patient macro F1 | Accuracy |","|---|---:|---:|"]
    lines += [f"| {arm} | {value['macro_f1']:.3f} | {value['accuracy']:.1%} |" for arm,value in metrics.items()]
    lines += ["","Pooled out-of-fold patient scores, not mean fold F1. One row represents one patient; equal-record robust summaries differ from prior beat-probability aggregation. Linear and tree classifiers use a fixed seed, not a three-seed neural ensemble. All predefined arms are retained.","",
        "## Paired comparisons","","| Difference | Δ macro F1 | Descriptive patient interval |","|---|---:|---:|"]
    for key,value in contrasts.items():
        lo,hi = value["descriptive_95_interval"]
        lines.append(f"| {key} | {value['macro_f1_difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["",f"{config['bootstrap_draws']:,} class-stratified paired patient draws, conditional on fixed fitted predictions; no refits or adjustment for correlated folds, repeated development or multiple contrasts. These are descriptive intervals, not independent confirmation.","",
        "## Frozen selection and compression","",
        "statistics tunes only model regularization/depth using the original statistics pool without supervised selection. legacy selects among the original feature families and feature budgets. refined adds higher-rank global signed DCT, denoised-only controls and native-wavelet signed cores. All ranking/scaling/redundancy pruning is refitted inside each inner-training partition; family, budget and model parameter are selected by inner macro F1. Exact ties favor configured order. A fresh selector and classifier are then fitted on outer-training patients without test monitoring. No global union of selected feature names is used.","",
        "The complete input tensor is lossless within numerical tolerance; all compressed summaries are lossy. Native wavelet cores use a fixed DCT basis within each band's differently sampled coefficient-time grid, not learned Tucker/HOOI or clinically annotated phases. The [fidelity audit](../bank_audit/REPORT.md) measures reconstruction, not diagnosis. Raw/template/residual views do not create independent patients. Templates use the held-out recording's own unlabeled beats for offline within-exam context.","",
        f"Linear model: L2 logistic regression, C choices {source['linear_c']}. Tree model: depth choices {config['tree_depths']}, {config['tree_estimators']} XGBoost trees, rate {config['tree_learning_rate']}, lambda 1, full row/column sampling, one CPU thread. Feature counts {source['feature_counts']}, correlation threshold {source['correlation_threshold']}. Class-balanced weights enter scaling and fitting once; ANOVA ranking itself is unweighted. All source patients/beats/labels/R positions are unchanged.","",
        "## Selected families","","| Procedure | Outer-fold family counts |","|---|---|"]
    lines += [f"| {arm} | {counts} |" for arm,counts in selection_counts.items()]
    lines += ["","## Per-class F1","","| Class | "+" | ".join(metrics)+" |","|---|"+"---:|"*len(metrics)]
    for label in source["classes"]:
        lines.append(f"| {label} | "+" | ".join(f"{value['classification_report'][label]['f1-score']:.3f}" for value in metrics.values())+" |")
    maximum = max(value["macro_f1"] for value in metrics.values())
    lines += ["","## Interpretation boundaries","",
        "No arm reached 0.80 macro F1." if maximum<.8 else "An arm reached 0.80 on these development folds; this requires independent confirmation.",
        "The historical neural 0.550 ± 0.040 uses different models, aggregation and three training seeds; it is context, not a matched tensor-refinement control. Use the paired statistics/legacy controls here to judge the representation. Repeatedly studied PTB folds are development data. No favorable seed/patient removal, new clinical labels, automatic relabeling or external pretraining. No clinical validity, novelty or SOTA is established. A favorable result still needs new-cohort evaluation.","",
        "Reporting independently recomputes training-only selectors, validates every inner selection and verifies that all saved models reproduce their outer predictions. Individual feature arrays, models and patient predictions remain local; only aggregate results/audits are published.",""]
    (root/"REPORT.md").write_text("\n".join(lines))
    return summary


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/tensor_refinement.yaml")
    args = parser.parse_args()
    print(json.dumps(summarize_study(yaml.safe_load(Path(args.config).read_text())),indent=2))
