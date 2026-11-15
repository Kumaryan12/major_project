"""Nested TabPFN-v2 benchmark on unchanged bootstrap-selected VCG features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import stability_study as prior, tabpfn_backend as backend
from .benchmark import inner_patient_splits, metric_bundle, save_confusion
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

CLAIM = "Pinned TabPFN-v2 feature-classifier development on repeatedly inspected PTB patient folds; local frozen synthetic-prior foundation model, not demonstrated MI SOTA or independent confirmation."


def load_inputs(config):
    stability_config = yaml.safe_load(Path(config["stability_config"]).read_text())
    source,ref_config,features,y,ids,names,folds,inherited = prior.load_inputs(stability_config)
    if config["seed"]!=source["seed"] or config["split_seed"]!=source["split_seed"] or config["arms"]!=["xgboost","tabpfn","tabpfn_balanced"]:
        raise ValueError("Require fixed TabPFN arms and source patient seeds")
    if config["device"]!="cpu" or any(type(config[key])!=int or config[key]<(100 if key=="bootstrap_draws" else 1) for key in ("torch_threads","n_estimators","bootstrap_draws")) or not np.isfinite(config["softmax_temperature"]) or config["softmax_temperature"]<=0:
        raise ValueError("Invalid pinned TabPFN inference budget")
    prior_root = Path(stability_config["output_dir"])/"complete"
    prior_scope = prior.make_scope(stability_config,source,ref_config,inherited)
    if json.loads((prior_root/"study_manifest.json").read_text())!=prior_scope:
        raise ValueError("TabPFN source stability study changed")
    summary = json.loads((prior_root/"summary.json").read_text())
    if not summary["complete"] or summary["study_fingerprint"]!=prior_scope["study_fingerprint"]:
        raise ValueError("Require complete validated bootstrap reference")
    provenance = {"prior_scope":prior_scope,"prior_root":str(prior_root),"prior_summary_sha256":file_hash(prior_root/"summary.json"),
        "checkpoint":backend.prepare_checkpoint(config),"backend_runtime":backend.runtime_info()}
    return stability_config,source,ref_config,features,y,ids,names,folds,provenance


def make_scope(config,provenance):
    scope = {"config":config,"source":provenance,"code_sha256":file_hash(__file__),"backend_code_sha256":file_hash(backend.__file__),"claim_scope":CLAIM}
    return dict(scope,study_fingerprint=_hash(scope))


def pfn_candidates(stability_config,source):
    return [{"family":family,"k":k,"params":{}} for family in stability_config["families"] for k in source["feature_counts"]]


def prepare(data,config,candidate,train,evaluate,seed,memo,verify=False):
    stability_config,source,ref_config,features,y,ids,names,folds,provenance = data
    X = features[candidate["family"]]
    selector,weights,audit = prior.prepare_selector(X,y,ids,train,evaluate,names[candidate["family"]],candidate,"bootstrap",stability_config,source,seed,Path(provenance["prior_root"]),memo,verify)
    # This hash is the actual unscaled context seen by the PFN; XGBoost instead
    # uses the same mask with its established training-scale transform.
    audit.update(raw_training_context_sha256=array_hash(X[train][:,selector.selected]),training_class_counts=np.bincount(y[train],minlength=len(source["classes"])).tolist())
    return selector,weights,audit


def check_backend(audit,selector,n_classes,config,seed):
    if audit["architecture"]!="TabPFN-v2" or audit["package_version"]!=config["package_version"] or audit["n_estimators"]!=config["n_estimators"] or audit["fitted_ensemble_members"]!=config["n_estimators"] or audit["classes"]!=list(range(n_classes)) or audit["n_features_in"]!=len(selector.selected) or audit["backend_device"]!=config["device"] or audit["random_state"]!=seed or audit["balance_probabilities"] or audit["softmax_temperature"]!=config["softmax_temperature"] or audit["fit_mode"]!="fit_preprocessors" or audit["n_preprocessing_jobs"]!=1:
        raise ValueError("TabPFN fitted backend settings/classes changed")


def read_probabilities(path,audit,indices,n_classes):
    if file_hash(path)!=audit["checkpoint_sha256"]:
        raise ValueError("TabPFN probability checkpoint changed")
    with np.load(path) as saved:
        if not np.array_equal(saved["indices"],indices):
            raise ValueError("TabPFN prediction membership changed")
        values = {key:saved[key] for key in saved.files if key!="indices"}
    for p in values.values():
        prior.reference.check_probabilities(p,len(indices),n_classes)
    return values


def save_probabilities(path,indices,**values):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary,indices=indices,**values)
    temporary.replace(path)
    return file_hash(path)


def xgb_inner(data,config,scope,root,fold,inner,candidate_id,candidate,train,valid,memo):
    stability_config,source,ref_config,features,y,ids,names,folds,provenance = data
    selector,weights,expected = prepare(data,config,candidate,train,valid,config["seed"]+fold*1000+inner,memo)
    expected.update(study_fingerprint=scope["study_fingerprint"],outer_fold=fold,inner_fold=inner,candidate_id=candidate_id,candidate=candidate,arm="xgboost")
    path = root/f"inner_{fold:02d}_xgboost_{candidate_id}_{inner}.json"
    checkpoint = path.with_suffix(".npz")
    old_path = Path(provenance["prior_root"])/f"inner_{fold:02d}_tree_bootstrap_{candidate_id:02d}_{inner}.json"
    original = json.loads(old_path.read_text())
    if original["study_fingerprint"]!=provenance["prior_scope"]["study_fingerprint"] or original["candidate"]!=candidate or original["outer_fold"]!=fold or original["inner_fold"]!=inner or original["method"]!="bootstrap" or original["learner"]!="tree" or any(original.get(key)!=value for key,value in expected.items() if key not in {"study_fingerprint","arm","raw_training_context_sha256","training_class_counts"}):
        raise ValueError("Matched XGBoost source selector/membership changed")
    original_p = prior.reference.read_prediction(old_path.with_suffix(".npz"),original,valid,len(source["classes"]))
    if path.exists() and checkpoint.exists():
        audit = json.loads(path.read_text())
        if any(audit.get(key)!=value for key,value in expected.items()) or audit["reference_audit_sha256"]!=file_hash(old_path):
            raise ValueError("Matched XGBoost cached scope changed")
        p = read_probabilities(checkpoint,audit,valid,len(source["classes"]))["probabilities"]
        if not np.array_equal(p,original_p):
            raise ValueError("Matched XGBoost predictions differ from source")
    else:
        p = original_p
        audit = dict(expected,reference_audit_sha256=file_hash(old_path),checkpoint_sha256=save_probabilities(checkpoint,valid,probabilities=p))
    audit["validation_macro_f1"] = float(f1_score(y[valid],p.argmax(1),labels=np.arange(len(source["classes"])),average="macro",zero_division=0))
    _save_json(path,audit)
    return audit


def pfn_inner(data,config,scope,root,fold,inner,candidate_id,candidate,train,valid,memo):
    stability_config,source,ref_config,features,y,ids,names,folds,provenance = data
    seed = config["seed"]+fold*1000+inner
    selector,weights,expected = prepare(data,config,candidate,train,valid,seed,memo)
    expected.update(study_fingerprint=scope["study_fingerprint"],outer_fold=fold,inner_fold=inner,candidate_id=candidate_id,candidate=candidate,seed=seed)
    path = root/f"inner_{fold:02d}_tabpfn_{candidate_id}_{inner}.json"
    checkpoint = path.with_suffix(".npz")
    if path.exists() and checkpoint.exists():
        audit = json.loads(path.read_text())
        if any(audit.get(key)!=value for key,value in expected.items()):
            raise ValueError("TabPFN inner context/selector changed")
        values = read_probabilities(checkpoint,audit,valid,len(source["classes"]))
    else:
        X = features[candidate["family"]]
        model = backend.fit_context(config,seed,X[train][:,selector.selected],y[train])
        values = {"tabpfn":backend.predict(model,X[valid][:,selector.selected],config)}
        values["tabpfn_balanced"] = backend.balance(values["tabpfn"],y[train],len(source["classes"]))
        audit = dict(expected,backend=backend.settings_audit(model,config),checkpoint_sha256=save_probabilities(checkpoint,valid,**values))
        del model
    check_backend(audit["backend"],selector,len(source["classes"]),config,seed)
    if set(values)!={"tabpfn","tabpfn_balanced"} or not np.allclose(values["tabpfn_balanced"],backend.balance(values["tabpfn"],y[train],len(source["classes"])),rtol=1e-12,atol=1e-12):
        raise ValueError("TabPFN class-prior correction changed")
    scores = {arm:float(f1_score(y[valid],p.argmax(1),labels=np.arange(len(source["classes"])),average="macro",zero_division=0)) for arm,p in values.items()}
    if "validation_macro_f1" in audit and audit["validation_macro_f1"]!=scores:
        raise ValueError("TabPFN cached inner scores changed")
    audit["validation_macro_f1"] = scores
    _save_json(path,audit)
    return audit


def run_study(config):
    data = load_inputs(config)
    stability_config,source,ref_config,features,y,ids,names,folds,provenance = data
    scope = make_scope(config,provenance)
    root = Path(config["output_dir"])/"complete"
    root.mkdir(parents=True,exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text())!=scope:
        raise ValueError("TabPFN settings/checkpoint/runtime changed; use a new output_dir")
    _save_json(guard,scope)
    assigned = folds.set_index("patient_id").loc[ids,"fold"].to_numpy()
    pooled = {arm:np.full((len(ids),len(source["classes"])),np.nan) for arm in config["arms"]}
    audits = {arm:[] for arm in config["arms"]}
    memo = {}
    for fold in range(1,source["outer_splits"]+1):
        train,test = np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
        splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],ids[train],source["inner_splits"],config["split_seed"]+fold*1000)]
        rows = {arm:[] for arm in config["arms"]}
        for candidate_id,candidate in enumerate(prior.candidates(stability_config,source,ref_config,"tree")):
            fits = [xgb_inner(data,config,scope,root,fold,inner,candidate_id,candidate,ti,vi,memo) for inner,(ti,vi) in enumerate(splits,1)]
            rows["xgboost"].append({"candidate_id":candidate_id,"candidate":candidate,"mean_inner_macro_f1":float(np.mean([fit["validation_macro_f1"] for fit in fits])),
                "inner_audits_sha256":[file_hash(root/f"inner_{fold:02d}_xgboost_{candidate_id}_{inner}.json") for inner in range(1,len(splits)+1)]})
        for candidate_id,candidate in enumerate(pfn_candidates(stability_config,source)):
            fits = [pfn_inner(data,config,scope,root,fold,inner,candidate_id,candidate,ti,vi,memo) for inner,(ti,vi) in enumerate(splits,1)]
            digests = [file_hash(root/f"inner_{fold:02d}_tabpfn_{candidate_id}_{inner}.json") for inner in range(1,len(splits)+1)]
            for arm in ("tabpfn","tabpfn_balanced"):
                rows[arm].append({"candidate_id":candidate_id,"candidate":candidate,"mean_inner_macro_f1":float(np.mean([fit["validation_macro_f1"][arm] for fit in fits])),"inner_audits_sha256":digests})
            print(f"TabPFN outer {fold} candidate {candidate_id+1}/{len(pfn_candidates(stability_config,source))}: inner contexts ready",flush=True)
            _progress(root,state="inner_contexts_complete",outer_fold=fold,candidate_id=candidate_id)
        for arm in config["arms"]:
            selection_path = root/f"selection_{fold:02d}_{arm}.json"
            _save_json(selection_path,{"study_fingerprint":scope["study_fingerprint"],"candidates":rows[arm]})
            chosen = max(range(len(rows[arm])),key=lambda i:rows[arm][i]["mean_inner_macro_f1"])
            candidate = rows[arm][chosen]["candidate"]
            seed = config["seed"]+fold
            selector,weights,expected = prepare(data,config,candidate,train,test,seed,memo)
            expected.update(study_fingerprint=scope["study_fingerprint"],fold=fold,arm=arm,candidate_id=chosen,candidate=candidate,seed=seed,selection_sha256=file_hash(selection_path))
            path,checkpoint = root/f"fold_{fold:02d}_{arm}.json",root/f"fold_{fold:02d}_{arm}.npz"
            model_path = root/f"model_{fold:02d}_{arm}.joblib"
            state_path = root/f"model_{fold:02d}_{arm}.tabpfn_fit"
            X = features[candidate["family"]]
            if path.exists() and checkpoint.exists():
                audit = json.loads(path.read_text())
                if any(audit.get(key)!=value for key,value in expected.items()) or audit["model_sha256"]!=file_hash(model_path) or (arm!="xgboost" and audit["fit_state_sha256"]!=file_hash(state_path)):
                    raise ValueError("TabPFN outer model/context changed")
                p = read_probabilities(checkpoint,audit,test,len(source["classes"]))["probabilities"]
            else:
                if arm=="xgboost":
                    model,p = prior.fit_selected(selector,X,y,ids,train,test,weights,candidate,"tree",stability_config,source,ref_config,seed)
                    model.update(arm=arm,study_fingerprint=scope["study_fingerprint"],fold=fold)
                    extra = {}
                else:
                    estimator = backend.fit_context(config,seed,X[train][:,selector.selected],y[train])
                    raw_p = backend.predict(estimator,X[test][:,selector.selected],config)
                    p = backend.balance(raw_p,y[train],len(source["classes"])) if arm=="tabpfn_balanced" else raw_p
                    actual = backend.settings_audit(estimator,config)
                    check_backend(actual,selector,len(source["classes"]),config,seed)
                    backend.save(estimator,state_path)
                    model = {"selector":selector,"candidate":candidate,"arm":arm,"study_fingerprint":scope["study_fingerprint"],"fold":fold,"backend":actual}
                    extra = {"backend":actual,"fit_state_sha256":file_hash(state_path)}
                    del estimator
                temporary = model_path.with_suffix(".partial.joblib")
                joblib.dump(model,temporary);temporary.replace(model_path)
                audit = dict(expected,**extra,model_sha256=file_hash(model_path),checkpoint_sha256=save_probabilities(checkpoint,test,probabilities=p))
            prior.reference.check_probabilities(p,len(test),len(source["classes"]))
            audit["patient_metrics"] = metric_bundle(y[test],p,source["classes"])
            audit["selected_feature_names"] = [names[candidate["family"]][i] for i in selector.selected]
            _save_json(path,audit)
            pooled[arm][test] = p;audits[arm].append(audit)
            print(f"{arm} outer {fold}/{source['outer_splits']}: {candidate['family']}, {len(selector.selected)} features, F1 {audit['patient_metrics']['macro_f1']:.4f}",flush=True)
            _progress(root,state="outer_fit_complete",outer_fold=fold,arm=arm)
    for arm,p in pooled.items():
        prior.reference.check_probabilities(p,len(ids),len(source["classes"]))
        saved = {"complete":True,"study_fingerprint":scope["study_fingerprint"],"classes":source["classes"],"n_patients":len(ids),"patient_metrics":metric_bundle(y,p,source["classes"]),"folds":audits[arm],"claim_scope":CLAIM}
        _save_json(root/f"metrics_{arm}.json",saved)
        frame = pd.DataFrame({"patient_id":ids,"truth":[source["classes"][i] for i in y],"prediction":[source["classes"][i] for i in p.argmax(1)],"fold":assigned})
        for i,label in enumerate(source["classes"]):
            frame[f"probability_{label}"] = p[:,i]
        frame.to_csv(root/f"patient_predictions_{arm}.csv",index=False)
        save_confusion(y,p.argmax(1),source["classes"],arm,root)
    _progress(root,state="complete")
    return root


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/tabpfn_v2.yaml")
    parser.add_argument("--status",action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        root = Path(config["output_dir"])/"complete"
        for arm in config["arms"]:
            print(f"{arm}: {len(list(root.glob(f'fold_*_{arm}.json')))}/10 outer folds")
        if (root/"progress.json").exists():
            print("Saved progress (not a live process check):",(root/"progress.json").read_text())
        print("Validated final report:",(root/"REPORT.md").exists())
    else:
        print(run_study(config))
