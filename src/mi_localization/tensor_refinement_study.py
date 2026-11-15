"""Nested final evaluation after signed tensor compression refinement."""
from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from . import tensor_feature_screen as old, tensor_refinement_bank as bank
from .benchmark import inner_patient_splits, metric_bundle, patient_balanced_weights, save_confusion
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

PROCEDURES = ("statistics","legacy","refined")
CLAIM = "Measured-VCG patient-level development on repeatedly inspected frozen PTB folds. Nested feature/parameter selection; no independent confirmation, clinical validation or SOTA."


def candidates(config,source,learner):
    grid = old.candidates(source)
    grid += [{"family":family,"k":k,"C":c} for family in bank.families(config)
        for k in source["feature_counts"] for c in source["linear_c"]]
    if learner=="linear":
        return [{"family":row["family"],"k":row["k"],"params":{"C":float(row["C"])}} for row in grid]
    mapping = dict(zip(source["linear_c"],config["tree_depths"],strict=True))
    return [{"family":row["family"],"k":row["k"],"params":{"max_depth":int(mapping[row["C"]])}} for row in grid]


def validate_config(config,source):
    old.validate_config(source)
    if config["learners"]!=["linear","tree"] or len(config["tree_depths"])!=len(source["linear_c"]) or len(set(config["tree_depths"]))!=len(config["tree_depths"]):
        raise ValueError("Require matched linear/tree choices")
    if any(type(d)!=int or d<1 for d in config["tree_depths"]) or type(config["tree_estimators"])!=int or config["tree_estimators"]<1 or not np.isfinite(config["tree_learning_rate"]) or config["tree_learning_rate"]<=0 or config["bootstrap_draws"]<100:
        raise ValueError("Invalid refinement classifier budget")
    if config["seed"]!=source["seed"] or config["split_seed"]!=source["split_seed"]:
        raise ValueError("Refinement must preserve source seeds/inner splits")


def load_inputs(config):
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    validate_config(config,source)
    features,y,ids,names,folds,old_manifest = old.load_features(source)
    root = bank.prepare_bank(config)
    manifest = json.loads((root/"manifest.json").read_text())
    with np.load(root/"patient_features.npz") as saved:
        if not np.array_equal(ids,saved["patient_id"]) or not np.array_equal(y,saved["y"]):
            raise ValueError("Refined patient rows/labels differ from source")
        for name in bank.families(config):
            features[name] = saved[name]
    names.update(json.loads((root/"feature_names.json").read_text()))
    if any(x.shape!=(len(ids),len(names[key])) or not np.isfinite(x).all() for key,x in features.items()):
        raise ValueError("Invalid refined patient features")
    legacy_path = Path(source["output_dir"])/"screen/study_manifest.json"
    legacy = None
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())
        fingerprint = legacy.pop("study_fingerprint")
        if _hash(legacy)!=fingerprint or legacy["config"]!=source or legacy["code_sha256"]!=file_hash(old.__file__) or legacy["bank_manifest_sha256"]!=file_hash(Path(source["feature_bank_dir"])/"manifest.json") or legacy["folds_sha256"]!=file_hash(source["folds_path"]) or legacy["sklearn"]!=sklearn.__version__ or legacy["numpy"]!=np.__version__:
            raise ValueError("Legacy probe source/runtime changed")
        legacy = {"path":str(legacy_path.parent),"study_fingerprint":fingerprint,"manifest_sha256":file_hash(legacy_path)}
    provenance = {"original_bank":old_manifest,"refined_bank":manifest,"legacy":legacy}
    return source,features,y,ids,names,folds,provenance


def prepare_fit(X,y,ids,train,evaluate,candidate,source):
    if set(ids[train])&set(ids[evaluate]):
        raise ValueError("Refinement training/evaluation patient overlap")
    weights = patient_balanced_weights(y[train],ids[train])
    selector = old.TrainingSelector().fit(X[train],y[train],weights,candidate["k"],source["correlation_threshold"])
    audit = {"patient_overlap":0,"training_patients":len(train),"validation_patients":len(evaluate),
        "train_membership_sha256":_hash(ids[train].tolist()),"validation_membership_sha256":_hash(ids[evaluate].tolist()),
        "training_features_sha256":array_hash(X[train]),"training_labels_sha256":array_hash(y[train]),
        "training_weights_sha256":array_hash(weights),**selector.audit()}
    return selector,weights,audit


def predict(model,X):
    with xgboost.config_context(nthread=1),threadpool_limits(limits=1):
        return model["estimator"].predict_proba(model["selector"].transform(X))


def check_probabilities(p,n,classes):
    if p.shape!=(n,classes) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError("Invalid refinement probabilities")


def fit_classifier(X,y,ids,train,evaluate,candidate,config,source,learner,seed):
    selector,weights,audit = prepare_fit(X,y,ids,train,evaluate,candidate,source)
    if learner=="linear":
        estimator = LogisticRegression(C=candidate["params"]["C"],solver="lbfgs",max_iter=3000,tol=1e-6,random_state=config["seed"])
    else:
        estimator = XGBClassifier(max_depth=candidate["params"]["max_depth"],n_estimators=config["tree_estimators"],
            learning_rate=config["tree_learning_rate"],reg_lambda=1.,subsample=1.,colsample_bytree=1.,
            objective="multi:softprob",num_class=len(source["classes"]),tree_method="hist",eval_metric="mlogloss",n_jobs=1,random_state=seed)
    with xgboost.config_context(nthread=1),threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        estimator.fit(selector.transform(X[train]),y[train],sample_weight=weights)
    if not np.array_equal(estimator.classes_,np.arange(len(source["classes"]))):
        raise ValueError("Refinement fit lost probability classes")
    model = {"selector":selector,"estimator":estimator,"candidate":candidate,"learner":learner}
    p = predict(model,X[evaluate])
    check_probabilities(p,len(evaluate),len(source["classes"]))
    return model,p,audit


def make_scope(config,source,provenance):
    scope = {"config":config,"source_config":source,"bank":provenance,"folds_sha256":file_hash(source["folds_path"]),
        "code_sha256":file_hash(__file__),"selector_code_sha256":file_hash(old.__file__),"claim_scope":CLAIM,
        "runtime":{"numpy":np.__version__,"sklearn":sklearn.__version__,"xgboost":xgboost.__version__,"platform":platform.platform()}}
    return dict(scope,study_fingerprint=_hash(scope))


def read_prediction(path,audit,evaluate,classes):
    if audit["checkpoint_sha256"]!=file_hash(path):
        raise ValueError("Refinement prediction checkpoint changed")
    with np.load(path) as saved:
        if not np.array_equal(saved["indices"],evaluate):
            raise ValueError("Refinement prediction membership changed")
        p = saved["probabilities"]
    check_probabilities(p,len(evaluate),classes)
    return p


def save_prediction(path,indices,p):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary,indices=indices,probabilities=p)
    temporary.replace(path)
    return file_hash(path)


def inner_fit(root,fold,candidate_id,inner,candidate,learner,train,valid,config,source,features,y,ids,scope):
    stem = root/f"inner_{fold:02d}_{learner}_{candidate_id:02d}_{inner}"
    path,checkpoint = stem.with_suffix(".json"),stem.with_suffix(".npz")
    X = features[candidate["family"]]
    _,_,expected = prepare_fit(X,y,ids,train,valid,candidate,source)
    expected.update(study_fingerprint=scope["study_fingerprint"],outer_fold=fold,inner_fold=inner,candidate_id=candidate_id,candidate=candidate,learner=learner)
    if path.exists() and checkpoint.exists():
        audit = json.loads(path.read_text())
        if any(audit.get(key)!=value for key,value in expected.items()):
            raise ValueError("Refinement inner training selector/settings changed")
        p = read_prediction(checkpoint,audit,valid,len(source["classes"]))
    else:
        legacy = scope["bank"]["legacy"]
        old_path = None if legacy is None else Path(legacy["path"])/f"inner_{fold:02d}_{candidate_id:02d}_{inner}.json"
        reuse = learner=="linear" and candidate_id<len(old.candidates(source)) and old_path is not None and old_path.exists() and old_path.with_suffix(".npz").exists()
        if reuse:
            saved = json.loads(old_path.read_text())
            old_candidate = old.candidates(source)[candidate_id]
            if saved["study_fingerprint"]!=legacy["study_fingerprint"] or saved["candidate"]!=old_candidate or saved["outer_fold"]!=fold or saved["inner_fold"]!=inner or saved["candidate_id"]!=candidate_id or any(saved.get(key)!=value for key,value in expected.items() if key not in {"study_fingerprint","candidate","learner"}):
                raise ValueError("Legacy cached probe training scope changed")
            p = read_prediction(old_path.with_suffix(".npz"),dict(checkpoint_sha256=saved["checkpoint_sha256"]),valid,len(source["classes"]))
            audit = dict(expected,reused_legacy_audit_sha256=file_hash(old_path),new_fit=False)
        else:
            model,p,_ = fit_classifier(X,y,ids,train,valid,candidate,config,source,learner,config["seed"]+fold*1000+inner)
            del model
            audit = dict(expected,new_fit=True)
        audit["checkpoint_sha256"] = save_prediction(checkpoint,valid,p)
    score = float(f1_score(y[valid],p.argmax(1),labels=np.arange(len(source["classes"])),average="macro",zero_division=0))
    if "validation_macro_f1" in audit and audit["validation_macro_f1"]!=score:
        raise ValueError("Refinement inner score changed")
    audit["validation_macro_f1"] = score
    _save_json(path,audit)
    return audit


def choose(rows,procedure,source):
    limit = len(source["linear_c"]) if procedure=="statistics" else len(old.candidates(source)) if procedure=="legacy" else len(rows)
    chosen = max(range(limit),key=lambda i:rows[i]["mean_inner_macro_f1"])
    return chosen,rows[chosen]["candidate"]


def run_study(config):
    source,features,y,ids,names,folds,provenance = load_inputs(config)
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True,exist_ok=True)
    scope = make_scope(config,source,provenance)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text())!=scope:
        raise ValueError("Refinement settings/data/runtime changed; use a new output_dir")
    _save_json(guard,scope)
    lookup = folds.set_index("patient_id").fold.to_dict()
    assigned = np.array([lookup[str(patient)] for patient in ids])
    pooled = {f"{learner}_{procedure}":np.full((len(ids),len(source["classes"])),np.nan) for learner in config["learners"] for procedure in PROCEDURES}
    fold_audits = {arm:[] for arm in pooled}
    for fold in range(1,source["outer_splits"]+1):
        train,test = np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
        splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],ids[train],source["inner_splits"],config["split_seed"]+fold*1000)]
        for learner in config["learners"]:
            candidate_rows = []
            for candidate_id,candidate in enumerate(candidates(config,source,learner)):
                audits = [inner_fit(root,fold,candidate_id,inner,candidate,learner,ti,vi,config,source,features,y,ids,scope) for inner,(ti,vi) in enumerate(splits,1)]
                candidate_rows.append({"candidate_id":candidate_id,"candidate":candidate,"mean_inner_macro_f1":float(np.mean([audit["validation_macro_f1"] for audit in audits])),
                    "inner_audits_sha256":[file_hash(root/f"inner_{fold:02d}_{learner}_{candidate_id:02d}_{inner}.json") for inner in range(1,len(splits)+1)]})
                _progress(root,state="inner_candidate_complete",outer_fold=fold,learner=learner,candidate_id=candidate_id,total_candidates=len(candidates(config,source,learner)))
            _save_json(root/f"selection_{fold:02d}_{learner}.json",{"study_fingerprint":scope["study_fingerprint"],"candidates":candidate_rows})
            for procedure in PROCEDURES:
                arm = f"{learner}_{procedure}"
                candidate_id,candidate = choose(candidate_rows,procedure,source)
                X = features[candidate["family"]]
                _,_,expected = prepare_fit(X,y,ids,train,test,candidate,source)
                expected.update(study_fingerprint=scope["study_fingerprint"],fold=fold,learner=learner,procedure=procedure,candidate_id=candidate_id,candidate=candidate,
                    selection_sha256=file_hash(root/f"selection_{fold:02d}_{learner}.json"))
                path,checkpoint = root/f"fold_{fold:02d}_{arm}.json",root/f"fold_{fold:02d}_{arm}.npz"
                model_path = root/f"model_{fold:02d}_{arm}.joblib"
                if path.exists() and checkpoint.exists():
                    audit = json.loads(path.read_text())
                    if any(audit.get(key)!=value for key,value in expected.items()) or audit["model_sha256"]!=file_hash(model_path):
                        raise ValueError("Refinement outer model/selector changed")
                    p = read_prediction(checkpoint,audit,test,len(source["classes"]))
                else:
                    model,p,_ = fit_classifier(X,y,ids,train,test,candidate,config,source,learner,config["seed"]+fold)
                    model.update(study_fingerprint=scope["study_fingerprint"],fold=fold,procedure=procedure)
                    temporary = model_path.with_suffix(".partial.joblib")
                    joblib.dump(model,temporary)
                    temporary.replace(model_path)
                    audit = dict(expected,model_sha256=file_hash(model_path),checkpoint_sha256=save_prediction(checkpoint,test,p))
                audit["patient_metrics"] = metric_bundle(y[test],p,source["classes"])
                audit["selected_feature_names"] = [names[candidate["family"]][i] for i in audit["selected_indices"]]
                _save_json(path,audit)
                pooled[arm][test] = p
                fold_audits[arm].append(audit)
                print(f"{arm} fold {fold}/{source['outer_splits']}: {candidate['family']}, {audit['n_selected']} features, outer F1 {audit['patient_metrics']['macro_f1']:.4f}",flush=True)
                _progress(root,state="outer_fit_complete",outer_fold=fold,arm=arm)
    for arm,p in pooled.items():
        check_probabilities(p,len(ids),len(source["classes"]))
        metrics = {"study_fingerprint":scope["study_fingerprint"],"complete":True,"classes":source["classes"],"n_patients":len(ids),
            "patient_metrics":metric_bundle(y,p,source["classes"]),"folds":fold_audits[arm],"claim_scope":CLAIM}
        _save_json(root/f"metrics_{arm}.json",metrics)
        frame = pd.DataFrame({"patient_id":ids,"truth":[source["classes"][i] for i in y],"prediction":[source["classes"][i] for i in p.argmax(1)],"fold":assigned})
        for i,label in enumerate(source["classes"]):
            frame[f"probability_{label}"] = p[:,i]
        frame.to_csv(root/f"patient_predictions_{arm}.csv",index=False)
        save_confusion(y,p.argmax(1),source["classes"],arm,root)
    _progress(root,state="complete")
    return root


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/tensor_refinement.yaml")
    args = parser.parse_args()
    print(run_study(yaml.safe_load(Path(args.config).read_text())))
