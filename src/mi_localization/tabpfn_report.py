"""Validate context-only TabPFN fits and paired outer patient predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import tabpfn_study as study
from .benchmark import inner_patient_splits, metric_bundle
from .neural import _save_json
from .neural_report import _read_predictions
from .relational_report import paired_bootstrap
from .signal_audit import file_hash


def summarize_study(config):
    data = study.load_inputs(config)
    stability_config,source,ref_config,features,y,ids,names,folds,provenance = data
    scope = study.make_scope(config,provenance)
    root = Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text())!=scope:
        raise ValueError("TabPFN report source/runtime/checkpoint changed")
    frozen = folds.set_index("patient_id").sort_index()
    assigned = frozen.loc[ids,"fold"].to_numpy()
    pooled = {arm:np.full((len(ids),len(source["classes"])),np.nan) for arm in config["arms"]}
    outer_audits = {arm:[] for arm in config["arms"]}
    memo = {}
    inner_pfn,reused_xgb,outer_pfn,outer_xgb,matches = 0,0,0,0,0
    for fold in range(1,source["outer_splits"]+1):
        train,test = np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
        splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],ids[train],source["inner_splits"],config["split_seed"]+fold*1000)]
        rows = {arm:[] for arm in config["arms"]}
        for kind,grid in (("xgboost",study.prior.candidates(stability_config,source,ref_config,"tree")),("tabpfn",study.pfn_candidates(stability_config,source))):
            for candidate_id,candidate in enumerate(grid):
                scores = {arm:[] for arm in (["xgboost"] if kind=="xgboost" else ["tabpfn","tabpfn_balanced"])}
                digests = []
                for inner,(ti,vi) in enumerate(splits,1):
                    seed = config["seed"]+fold*1000+inner
                    selector,_,expected = study.prepare(data,config,candidate,ti,vi,seed,memo,verify=True)
                    expected.update(study_fingerprint=scope["study_fingerprint"],outer_fold=fold,inner_fold=inner,candidate_id=candidate_id,candidate=candidate)
                    expected.update(arm="xgboost") if kind=="xgboost" else expected.update(seed=seed)
                    path = root/f"inner_{fold:02d}_{kind}_{candidate_id}_{inner}.json"
                    audit = json.loads(path.read_text())
                    if any(audit.get(key)!=value for key,value in expected.items()):
                        raise ValueError("TabPFN inner selector/context membership changed")
                    values = study.read_probabilities(path.with_suffix(".npz"),audit,vi,len(source["classes"]))
                    if kind=="xgboost":
                        old_path = Path(provenance["prior_root"])/f"inner_{fold:02d}_tree_bootstrap_{candidate_id:02d}_{inner}.json"
                        original = json.loads(old_path.read_text())
                        if audit["reference_audit_sha256"]!=file_hash(old_path) or original["study_fingerprint"]!=provenance["prior_scope"]["study_fingerprint"] or original["candidate"]!=candidate or original["selector"]!=expected["selector"] or not np.array_equal(values["probabilities"],study.prior.reference.read_prediction(old_path.with_suffix(".npz"),original,vi,len(source["classes"]))):
                            raise ValueError("TabPFN matched-control inner predictions changed")
                        values = {"xgboost":values["probabilities"]}
                        reused_xgb += 1
                    else:
                        study.check_backend(audit["backend"],selector,len(source["classes"]),config,seed)
                        if set(values)!={"tabpfn","tabpfn_balanced"} or not np.allclose(values["tabpfn_balanced"],study.backend.balance(values["tabpfn"],y[ti],len(source["classes"])),rtol=1e-12,atol=1e-12):
                            raise ValueError("TabPFN inner probability balancing changed")
                        inner_pfn += 1
                    observed = {arm:float(f1_score(y[vi],p.argmax(1),labels=np.arange(len(source["classes"])),average="macro",zero_division=0)) for arm,p in values.items()}
                    if audit["validation_macro_f1"]!=(observed["xgboost"] if kind=="xgboost" else observed):
                        raise ValueError("TabPFN inner macro F1 disagrees with probabilities")
                    for arm,value in observed.items():
                        scores[arm].append(value)
                    digests.append(file_hash(path))
                for arm,values in scores.items():
                    rows[arm].append({"candidate_id":candidate_id,"candidate":candidate,"mean_inner_macro_f1":float(np.mean(values)),"inner_audits_sha256":digests})
        for arm in config["arms"]:
            selection_path = root/f"selection_{fold:02d}_{arm}.json"
            if json.loads(selection_path.read_text())!={"study_fingerprint":scope["study_fingerprint"],"candidates":rows[arm]}:
                raise ValueError("TabPFN selected candidate disagrees with inner scores")
            chosen = max(range(len(rows[arm])),key=lambda i:rows[arm][i]["mean_inner_macro_f1"])
            candidate = rows[arm][chosen]["candidate"]
            seed = config["seed"]+fold
            selector,_,expected = study.prepare(data,config,candidate,train,test,seed,memo,verify=True)
            expected.update(study_fingerprint=scope["study_fingerprint"],fold=fold,arm=arm,candidate_id=chosen,candidate=candidate,seed=seed,selection_sha256=file_hash(selection_path))
            path = root/f"fold_{fold:02d}_{arm}.json"
            audit = json.loads(path.read_text())
            if any(audit.get(key)!=value for key,value in expected.items()) or audit["selected_feature_names"]!=[names[candidate["family"]][i] for i in selector.selected]:
                raise ValueError("TabPFN outer selector/context choice changed")
            p = study.read_probabilities(path.with_suffix(".npz"),audit,test,len(source["classes"]))["probabilities"]
            model_path = root/f"model_{fold:02d}_{arm}.joblib"
            if audit["model_sha256"]!=file_hash(model_path):
                raise ValueError("TabPFN benchmark saved model changed")
            model = joblib.load(model_path)
            if any(model.get(key)!=value for key,value in {"study_fingerprint":scope["study_fingerprint"],"fold":fold,"arm":arm,"candidate":candidate}.items()) or model["selector"].audit()!=selector.audit():
                raise ValueError("TabPFN saved selector/model provenance changed")
            X = features[candidate["family"]]
            if arm=="xgboost":
                reproduced = study.prior.reference.predict(model,X[test])
                previous_path = Path(provenance["prior_root"])/f"fold_{fold:02d}_tree_bootstrap.json"
                previous = json.loads(previous_path.read_text())
                if previous["candidate"]!=candidate or not np.allclose(p,study.prior.reference.read_prediction(previous_path.with_suffix(".npz"),previous,test,len(source["classes"])),rtol=1e-12,atol=1e-12):
                    raise ValueError("Fresh matched XGBoost differs from prior benchmark")
                matches += 1;outer_xgb += 1
            else:
                state = root/f"model_{fold:02d}_{arm}.tabpfn_fit"
                if audit["fit_state_sha256"]!=file_hash(state):
                    raise ValueError("TabPFN fitted context state changed")
                estimator = study.backend.load(state,config)
                actual = study.backend.settings_audit(estimator,config)
                study.check_backend(actual,selector,len(source["classes"]),config,seed)
                if actual!=audit["backend"] or actual!=model["backend"]:
                    raise ValueError("Restored TabPFN inference settings changed")
                raw = study.backend.predict(estimator,X[test][:,selector.selected],config)
                reproduced = study.backend.balance(raw,y[train],len(source["classes"])) if arm=="tabpfn_balanced" else raw
                del estimator
                outer_pfn += 1
            if not np.allclose(reproduced,p,rtol=1e-6,atol=1e-7) or not np.array_equal(reproduced.argmax(1),p.argmax(1)) or audit["patient_metrics"]!=metric_bundle(y[test],p,source["classes"]):
                raise ValueError("Saved TabPFN benchmark model does not reproduce outer predictions")
            pooled[arm][test] = p;outer_audits[arm].append(audit)
    metrics,predictions = {},{}
    for arm,p in pooled.items():
        study.prior.reference.check_probabilities(p,len(ids),len(source["classes"]))
        saved = json.loads((root/f"metrics_{arm}.json").read_text())
        if not saved["complete"] or saved["study_fingerprint"]!=scope["study_fingerprint"] or saved["classes"]!=source["classes"] or saved["n_patients"]!=len(ids) or saved["folds"]!=outer_audits[arm] or saved["claim_scope"]!=study.CLAIM or saved["patient_metrics"]!=metric_bundle(y,p,source["classes"]):
            raise ValueError("TabPFN pooled patient metrics mismatch")
        truth,prediction = _read_predictions(root/f"patient_predictions_{arm}.csv",saved,frozen,source["classes"])
        frame = pd.read_csv(root/f"patient_predictions_{arm}.csv").set_index("patient_id").loc[ids]
        if not np.array_equal(truth,y) or not np.array_equal(prediction,p.argmax(1)) or not np.allclose(frame[[f"probability_{label}" for label in source["classes"]]].to_numpy(),p,rtol=1e-7,atol=1e-12):
            raise ValueError("TabPFN CSV does not match complete outer predictions")
        metrics[arm],predictions[arm] = saved["patient_metrics"],prediction
    uncertainty = paired_bootstrap(y,predictions,len(source["classes"]),draws=config["bootstrap_draws"],seed=config["seed"],references=("xgboost","tabpfn"))
    contrasts = {}
    for second,first in (("tabpfn","xgboost"),("tabpfn_balanced","xgboost"),("tabpfn_balanced","tabpfn")):
        contrasts[f"{second}_minus_{first}"] = {"macro_f1_difference":metrics[second]["macro_f1"]-metrics[first]["macro_f1"],
            "descriptive_95_interval":uncertainty["arms"][second]["paired_comparisons"][first]["macro_f1_difference_95_interval"]}
    summary = {"complete":True,"study_fingerprint":scope["study_fingerprint"],"n_patients":len(ids),"metrics":metrics,"contrasts":contrasts,
        "reused_xgboost_inner_fits":reused_xgb,"new_tabpfn_inner_context_fits":inner_pfn,"new_tabpfn_outer_context_fits":outer_pfn,
        "new_xgboost_outer_fits":outer_xgb,"total_fit_context_slots":reused_xgb+inner_pfn+outer_pfn+outer_xgb,
        "bootstrap_caches_replayed":len(memo),"matching_reference_outer_predictions":matches,"checkpoint":provenance["checkpoint"],"claim_scope":study.CLAIM}
    _save_json(root/"summary.json",summary)
    _save_json(root/"paired_bootstrap.json",dict(uncertainty,study_fingerprint=scope["study_fingerprint"]))
    lines = ["# TabPFN-v2 on tensor-derived VCG features","",study.CLAIM,"",
        f"All {len(ids)} original patients; {source['outer_splits']} frozen outer folds, {source['inner_splits']} inner folds. Validated {reused_xgb} reused XGBoost inner fits, {inner_pfn} new PFN inner contexts, {outer_pfn} new PFN outer contexts and {outer_xgb} fresh XGBoost outer fits. Raw/balanced inner scores share each PFN context, not independent fits.","",
        "| Classifier | Patient macro F1 | Accuracy |","|---|---:|---:|"]
    lines += [f"| {arm} | {value['macro_f1']:.3f} | {value['accuracy']:.1%} |" for arm,value in metrics.items()]
    lines += ["","Pooled out-of-fold patient results. All models use the unchanged morphology/combined pools, bootstrap selection and budgets from the source study. One fixed fold/inference-seed policy; eight PFN inference members are not eight independent training seeds.","",
        "| Paired difference | Δ macro F1 | Descriptive patient interval |","|---|---:|---:|"]
    for key,value in contrasts.items():
        lo,hi = value["descriptive_95_interval"]
        lines.append(f"| {key} | {value['macro_f1_difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["",f"{config['bootstrap_draws']:,} paired class-stratified patient draws, conditional on fitted predictions. No model refits or correction for correlated folds, repeated development or multiple comparisons. These are descriptive intervals, not independent confirmation.","",
        "## Exact inference and fair-comparison boundaries","",
        f"Pinned public Nature-era TabPFN-v2 checkpoint, revision {config['checkpoint_revision']}, SHA256 {config['checkpoint_sha256']}; tabpfn=={config['package_version']}. This is not the latest TabPFN-3.5. Local {config['device']} inference, float32, {config['torch_threads']} threads, {config['n_estimators']} fixed permutation/preprocessing members, temperature {config['softmax_temperature']}, no gradient fine-tuning or hosted prediction service. Model-specific preprocessing is fitted to the training context. The foundation weights remain frozen.","",
        "TabPFN sees raw selected feature columns; the bootstrap selector's class-balanced training scaler is used for ranking/pruning, not applied again to PFN input. XGBoost retains its established selected-column scaling and class-balanced fitting weights. TabPFN's API has no sample-weight argument: raw probabilities are reported separately from a fixed inverse-training-class-count renormalization. This correction uses no validation/test prevalence, creates no duplicate patients and is not clinical calibration. It is not equivalent to weighted tree training. Both PFN policies are predeclared and retained.","",
        "For each outer fold, family/budget is selected separately for each PFN probability policy by mean inner macro F1, with configured-order ties. Four PFN candidates versus eight XGBoost candidates (which also tune depth 2/3). No PFN model/checkpoint, member-count, temperature, threshold or prior-correction strength search. Data/splits/feature-mask rules are matched; parameter counts, algorithms and pretraining compute are not matched. Every outer model is freshly fitted on outer-training patients without test monitoring.","",
        f"Every relevant bootstrap selector cache was replayed ({len(memo)} caches); all {matches} fresh XGBoost control predictions reproduced the previous study. Every saved PFN context was restored and reproduced its outer probabilities/classes. A six-class real-backend smoke test checks finite predictions, save/load, correction equivalence and invariance to unrelated query rows. Learned normalization/SVD and labels never use outer-test training context.","",
        "## Per-class F1","","| Class | "+" | ".join(metrics)+" |","|---|"+"---:|"*len(metrics)]
    for label in source["classes"]:
        lines.append(f"| {label} | "+" | ".join(f"{value['classification_report'][label]['f1-score']:.3f}" for value in metrics.values())+" |")
    lines += ["","These repeatedly inspected PTB patients remain development data. The historical neural 0.550 ± 0.040 is a different model/aggregation protocol and not this paired control. A pretrained tabular architecture does not establish MI SOTA, clinical validity or the 0.80–0.90 target. Independent cohort validation is still required. No class, patient, beat, original waveform or R position changed.","",
        "Reproduce: `.venv/bin/python -m mi_localization.tabpfn_study`, then `.venv/bin/python -m mi_localization.tabpfn_report`. Checkpoint, contexts, bootstrap draws, models and individual patient predictions remain local; aggregate metrics/audits are public.",""]
    (root/"REPORT.md").write_text("\n".join(lines))
    return summary


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/tabpfn_v2.yaml")
    args = parser.parse_args()
    print(json.dumps(summarize_study(yaml.safe_load(Path(args.config).read_text())),indent=2))
