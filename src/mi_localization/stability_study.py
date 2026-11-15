"""Matched nested patient evaluation of original and bootstrap selectors."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import xgboost
from xgboost import XGBClassifier

from . import feature_stability as selection, tensor_refinement_study as reference
from .benchmark import inner_patient_splits, metric_bundle, patient_balanced_weights, save_confusion
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

CLAIM = "Bootstrap feature-consensus development on repeatedly inspected PTB patient folds; no independent confirmation, clinical biomarker or formal stability-selection error guarantees."


def load_inputs(config):
    ref_config = yaml.safe_load(Path(config["reference_config"]).read_text())
    source,all_features,y,ids,all_names,folds,provenance = reference.load_inputs(ref_config)
    if config["families"]!=["morphology","combined"] or config["selectors"]!=["original","bootstrap","group_bootstrap"] or config["seed"]!=source["seed"] or config["split_seed"]!=source["split_seed"]:
        raise ValueError("Require the fixed stability pool, controls and seeds")
    if any(type(config[key])!=int or config[key]<(100 if key=="bootstrap_draws" else 2) for key in ("bootstrap_replicates","audit_replicates","bootstrap_draws")):
        raise ValueError("Invalid stability resampling budget")
    ref_root = Path(ref_config["output_dir"])/"complete"
    ref_scope = reference.make_scope(ref_config,source,provenance)
    if not (ref_root/"study_manifest.json").exists() or json.loads((ref_root/"study_manifest.json").read_text())!=ref_scope:
        raise ValueError("Require matching completed reference study")
    summary = json.loads((ref_root/"summary.json").read_text())
    if not summary["complete"] or summary["study_fingerprint"]!=ref_scope["study_fingerprint"]:
        raise ValueError("Reference study must be complete and validated")
    features = {family:all_features[family] for family in config["families"]}
    names = {family:all_names[family] for family in config["families"]}
    inherited = {"reference_scope":ref_scope,"reference_summary_sha256":file_hash(ref_root/"summary.json"),"reference_root":str(ref_root)}
    return source,ref_config,features,y,ids,names,folds,inherited


def make_scope(config,source,ref_config,inherited):
    scope = {"config":config,"source_config":source,"reference_config":ref_config,"reference":inherited,
        "code_sha256":file_hash(__file__),"selector_code_sha256":file_hash(selection.__file__),"claim_scope":CLAIM}
    return dict(scope,study_fingerprint=_hash(scope))


def candidates(config,source,ref_config,learner):
    return [candidate for candidate in reference.candidates(ref_config,source,learner) if candidate["family"] in config["families"]]


def prepare_selector(X,y,ids,train,evaluate,names,candidate,method,config,source,seed,root,memo,verify=False):
    if set(ids[train])&set(ids[evaluate]):
        raise ValueError("Stability training/evaluation patient overlap")
    runtime = dict(config,_selector_code_sha256=file_hash(selection.__file__))
    payload = selection.bootstrap_cache(X,y,ids,train,names,candidate["family"],runtime,source,seed,root,memo,verify)
    weights = patient_balanced_weights(y[train],ids[train])
    selector = selection.ConsensusSelector().fit(X[train],y[train],weights,candidate["k"],source["correlation_threshold"],names,candidate["family"],method,payload)
    audit = {"patient_overlap":0,"training_patients":len(train),"validation_patients":len(evaluate),
        "train_membership_sha256":_hash(ids[train].tolist()),"validation_membership_sha256":_hash(ids[evaluate].tolist()),
        "training_features_sha256":array_hash(X[train]),"training_labels_sha256":array_hash(y[train]),
        "training_weights_sha256":array_hash(weights),"selector":selector.audit()}
    return selector,weights,audit


def fit_selected(selector,X,y,ids,train,evaluate,weights,candidate,learner,config,source,ref_config,seed):
    if set(ids[train])&set(ids[evaluate]):
        raise ValueError("Stability classifier patient overlap")
    if learner=="linear":
        estimator = LogisticRegression(C=candidate["params"]["C"],solver="lbfgs",max_iter=3000,tol=1e-6,random_state=config["seed"])
    else:
        estimator = XGBClassifier(max_depth=candidate["params"]["max_depth"],n_estimators=ref_config["tree_estimators"],
            learning_rate=ref_config["tree_learning_rate"],reg_lambda=1.,subsample=1.,colsample_bytree=1.,objective="multi:softprob",
            num_class=len(source["classes"]),tree_method="hist",eval_metric="mlogloss",n_jobs=1,random_state=seed)
    with xgboost.config_context(nthread=1),threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        estimator.fit(selector.transform(X[train]),y[train],sample_weight=weights)
    if not np.array_equal(estimator.classes_,np.arange(len(source["classes"]))):
        raise ValueError("Stability classifier lost probability classes")
    model = {"selector":selector,"estimator":estimator,"candidate":candidate,"learner":learner}
    p = reference.predict(model,X[evaluate])
    reference.check_probabilities(p,len(evaluate),len(source["classes"]))
    return model,p


def original_checkpoint(scope,source,ref_config,learner,candidate,fold,inner):
    candidate_id = reference.candidates(ref_config,source,learner).index(candidate)
    return Path(scope["reference"]["reference_root"])/f"inner_{fold:02d}_{learner}_{candidate_id:02d}_{inner}.json"


def inner_fit(root,fold,inner,candidate_id,candidate,method,learner,train,valid,config,source,ref_config,features,y,ids,names,scope,memo):
    X = features[candidate["family"]]
    selector,weights,audit = prepare_selector(X,y,ids,train,valid,names[candidate["family"]],candidate,method,config,source,config["seed"]+fold*1000+inner,root,memo)
    expected = dict(audit,study_fingerprint=scope["study_fingerprint"],outer_fold=fold,inner_fold=inner,candidate_id=candidate_id,candidate=candidate,method=method,learner=learner)
    path = root/f"inner_{fold:02d}_{learner}_{method}_{candidate_id:02d}_{inner}.json"
    checkpoint = path.with_suffix(".npz")
    if path.exists() and checkpoint.exists():
        audit = json.loads(path.read_text())
        if any(audit.get(key)!=value for key,value in expected.items()):
            raise ValueError("Stability cached inner selector/scope changed")
        p = reference.read_prediction(checkpoint,audit,valid,len(source["classes"]))
    else:
        if method=="original":
            original_path = original_checkpoint(scope,source,ref_config,learner,candidate,fold,inner)
            original_audit = json.loads(original_path.read_text())
            _,_,old_expected = reference.prepare_fit(X,y,ids,train,valid,candidate,source)
            if original_audit["study_fingerprint"]!=scope["reference"]["reference_scope"]["study_fingerprint"] or original_audit["candidate"]!=candidate or original_audit["learner"]!=learner or original_audit["outer_fold"]!=fold or original_audit["inner_fold"]!=inner or any(original_audit.get(key)!=value for key,value in old_expected.items()) or any(selector.base_audit[key]!=value for key,value in old_expected.items() if key in selector.base_audit):
                raise ValueError("Original control training selector/checkpoint changed")
            p = reference.read_prediction(original_path.with_suffix(".npz"),original_audit,valid,len(source["classes"]))
            audit = dict(expected,new_fit=False,reference_audit_sha256=file_hash(original_path))
        else:
            model,p = fit_selected(selector,X,y,ids,train,valid,weights,candidate,learner,config,source,ref_config,config["seed"]+fold*1000+inner)
            del model
            audit = dict(expected,new_fit=True)
        audit["checkpoint_sha256"] = reference.save_prediction(checkpoint,valid,p)
    score = float(f1_score(y[valid],p.argmax(1),labels=np.arange(len(source["classes"])),average="macro",zero_division=0))
    if "validation_macro_f1" in audit and audit["validation_macro_f1"]!=score:
        raise ValueError("Stability inner score disagrees with predictions")
    audit["validation_macro_f1"] = score
    _save_json(path,audit)
    return audit


def run_study(config):
    source,ref_config,features,y,ids,names,folds,inherited = load_inputs(config)
    scope = make_scope(config,source,ref_config,inherited)
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True,exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text())!=scope:
        raise ValueError("Stability data/config/code changed; use a new output_dir")
    _save_json(guard,scope)
    lookup = folds.set_index("patient_id").fold.to_dict()
    assignment = np.array([lookup[str(patient)] for patient in ids])
    arms = [f"{learner}_{method}" for learner in ref_config["learners"] for method in config["selectors"]]
    pooled = {arm:np.full((len(ids),len(source["classes"])),np.nan) for arm in arms}
    audits = {arm:[] for arm in arms}
    memo = {}
    for fold in range(1,source["outer_splits"]+1):
        train,test = np.flatnonzero(assignment!=fold),np.flatnonzero(assignment==fold)
        splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],ids[train],source["inner_splits"],config["split_seed"]+fold*1000)]
        for learner in ref_config["learners"]:
            for method in config["selectors"]:
                arm = f"{learner}_{method}"
                rows = []
                for candidate_id,candidate in enumerate(candidates(config,source,ref_config,learner)):
                    inner_audits = [inner_fit(root,fold,inner,candidate_id,candidate,method,learner,ti,vi,config,source,ref_config,features,y,ids,names,scope,memo) for inner,(ti,vi) in enumerate(splits,1)]
                    rows.append({"candidate_id":candidate_id,"candidate":candidate,"mean_inner_macro_f1":float(np.mean([audit["validation_macro_f1"] for audit in inner_audits])),
                        "inner_audits_sha256":[file_hash(root/f"inner_{fold:02d}_{learner}_{method}_{candidate_id:02d}_{inner}.json") for inner in range(1,len(splits)+1)]})
                    _progress(root,state="inner_candidate_complete",outer_fold=fold,arm=arm,candidate_id=candidate_id)
                selection_path = root/f"selection_{fold:02d}_{arm}.json"
                _save_json(selection_path,{"study_fingerprint":scope["study_fingerprint"],"candidates":rows})
                chosen = max(range(len(rows)),key=lambda i:rows[i]["mean_inner_macro_f1"])
                candidate = rows[chosen]["candidate"]
                X = features[candidate["family"]]
                selector,weights,expected = prepare_selector(X,y,ids,train,test,names[candidate["family"]],candidate,method,config,source,config["seed"]+fold,root,memo)
                expected.update(study_fingerprint=scope["study_fingerprint"],fold=fold,candidate_id=chosen,candidate=candidate,method=method,learner=learner,selection_sha256=file_hash(selection_path))
                path,checkpoint = root/f"fold_{fold:02d}_{arm}.json",root/f"fold_{fold:02d}_{arm}.npz"
                model_path = root/f"model_{fold:02d}_{arm}.joblib"
                if path.exists() and checkpoint.exists():
                    audit = json.loads(path.read_text())
                    if any(audit.get(key)!=value for key,value in expected.items()) or audit["model_sha256"]!=file_hash(model_path):
                        raise ValueError("Stability cached outer model/selector changed")
                    p = reference.read_prediction(checkpoint,audit,test,len(source["classes"]))
                else:
                    model,p = fit_selected(selector,X,y,ids,train,test,weights,candidate,learner,config,source,ref_config,config["seed"]+fold)
                    model.update(study_fingerprint=scope["study_fingerprint"],fold=fold,method=method)
                    temporary = model_path.with_suffix(".partial.joblib")
                    joblib.dump(model,temporary)
                    temporary.replace(model_path)
                    audit = dict(expected,model_sha256=file_hash(model_path),checkpoint_sha256=reference.save_prediction(checkpoint,test,p))
                audit["patient_metrics"] = metric_bundle(y[test],p,source["classes"])
                audit["selected_feature_names"] = [names[candidate["family"]][i] for i in audit["selector"]["selected_indices"]]
                _save_json(path,audit)
                pooled[arm][test] = p
                audits[arm].append(audit)
                print(f"{arm} fold {fold}/{source['outer_splits']}: {candidate['family']}, {audit['selector']['n_selected']} features, outer F1 {audit['patient_metrics']['macro_f1']:.4f}",flush=True)
    for arm,p in pooled.items():
        reference.check_probabilities(p,len(ids),len(source["classes"]))
        saved = {"complete":True,"study_fingerprint":scope["study_fingerprint"],"classes":source["classes"],"n_patients":len(ids),
            "patient_metrics":metric_bundle(y,p,source["classes"]),"folds":audits[arm],"claim_scope":CLAIM}
        _save_json(root/f"metrics_{arm}.json",saved)
        frame = pd.DataFrame({"patient_id":ids,"truth":[source["classes"][i] for i in y],"prediction":[source["classes"][i] for i in p.argmax(1)],"fold":assignment})
        for i,label in enumerate(source["classes"]):
            frame[f"probability_{label}"] = p[:,i]
        frame.to_csv(root/f"patient_predictions_{arm}.csv",index=False)
        save_confusion(y,p.argmax(1),source["classes"],arm,root)
    _progress(root,state="complete")
    return root


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/feature_stability.yaml")
    args = parser.parse_args()
    print(run_study(yaml.safe_load(Path(args.config).read_text())))
