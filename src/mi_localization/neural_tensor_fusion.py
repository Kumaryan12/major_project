"""Nested late fusion of a frozen VCG neural ensemble and tensor TabPFN.

Only fold-local inner predictions select weights. No new neural/PFN fits,
global-OOF meta-training, seed selection, calibration or patient routing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import residual_seed_ensemble as neural, residual_context_study as nstudy
from . import tabpfn_study as tensor
from .benchmark import inner_patient_splits, metric_bundle
from .chronology_diagnostics import validate_predictions
from .contrastive import _progress
from .neural import _hash, _save_json
from .signal_audit import file_hash

PROCEDURES = ("neural_only", "tensor_only", "equal_blend", "selected_blend")
CLAIM = ("Neural–tensor late-fusion development on repeatedly inspected PTB patients; "
         "scalar blending selected inside outer-training patients, frozen component recipes, no independent confirmation or established SOTA.")


def blend(neural_p, tensor_p, alpha):
    a, b = np.asarray(neural_p), np.asarray(tensor_p)
    if (not np.isfinite(alpha) or not 0 <= alpha <= 1 or a.shape != b.shape or a.ndim != 2
            or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min()) < 0
            or not np.allclose(a.sum(1),1,atol=1e-5,rtol=0) or not np.allclose(b.sum(1),1,atol=1e-5,rtol=0)):
        raise ValueError("Invalid aligned fusion probabilities/weight")
    return alpha*a+(1-alpha)*b


def choose_weight(inner_neural, inner_tensor, inner_truth, classes, grid):
    """No outer prediction/label arguments exist at this selection boundary."""
    if not len(inner_truth) or not len(inner_neural)==len(inner_tensor)==len(inner_truth):
        raise ValueError("Incomplete inner-fold predictions")
    rows=[]
    for alpha in grid:
        scores=[]
        for a,b,y in zip(inner_neural,inner_tensor,inner_truth,strict=True):
            p=blend(a,b,alpha)
            if len(y)!=len(p): raise ValueError("Inner label/probability rows differ")
            scores.append(float(f1_score(y,p.argmax(1),labels=np.arange(len(classes)),average="macro",zero_division=0)))
        rows.append({"neural_weight":alpha,"inner_macro_f1":scores,"mean_inner_macro_f1":float(np.mean(scores))})
    chosen=max(range(len(rows)),key=lambda i:rows[i]["mean_inner_macro_f1"])
    return grid[chosen],rows


def check_partition(train,valid,outer_query,ids):
    a,b,c=[set(ids[x]) for x in (train,valid,outer_query)]
    if not a or not b or a&b or a&c or b&c or len(a)!=len(train) or len(b)!=len(valid):
        raise ValueError("Fusion inner patients overlap each other or the outer query")


def load_inputs(config):
    if (config["procedures"]!=list(PROCEDURES) or config["neural_arm"]!="ordered_context_aux"
            or config["tensor_arm"]!="tabpfn" or config["neural_weight_grid"]!=[1.,.75,.5,.25,0.]
            or type(config["bootstrap_draws"])!=int or config["bootstrap_draws"]<100):
        raise ValueError("Require frozen branches, four controls and fixed scalar weight grid")
    ensemble_config=yaml.safe_load(Path(config["neural_ensemble_config"]).read_text())
    nc,bank,assigned,nroot,nscope,ncompleted=neural.source_inputs(ensemble_config)
    eroot=Path(ensemble_config["output_dir"])/"complete"
    esummary=json.loads((eroot/"summary.json").read_text());emanifest=json.loads((eroot/"analysis_manifest.json").read_text())
    if (not esummary.get("complete") or esummary.get("analysis_fingerprint")!=_hash(emanifest)
            or emanifest.get("config")!=ensemble_config or esummary.get("training_seeds")!=nc["training_seeds"]
            or esummary.get("classes")!=nc["classes"] or esummary.get("n_patients")!=len(bank.ids)
            or esummary.get("new_training_fits")!=0
            or esummary.get("source_fingerprint")!=nscope["study_fingerprint"]):
        raise ValueError("Require verified frozen neural ensemble reference")
    tc=yaml.safe_load(Path(config["tensor_config"]).read_text());td=tensor.load_inputs(tc)
    sc,source,ref,features,y,ids,names,folds,provenance=td
    tscope=tensor.make_scope(tc,provenance);troot=Path(tc["output_dir"])/"complete"
    tsummary=json.loads((troot/"summary.json").read_text())
    if (json.loads((troot/"study_manifest.json").read_text())!=tscope or not tsummary.get("complete")
            or tsummary.get("study_fingerprint")!=tscope["study_fingerprint"]
            or not (troot/"REPORT.md").is_file() or tsummary.get("n_patients")!=len(ids)):
        raise ValueError("Require matching verified tensor/PFN source")
    ta=folds.set_index("patient_id").loc[ids,"fold"].to_numpy()
    if (not np.array_equal(ids,bank.ids) or not np.array_equal(y,bank.labels) or not np.array_equal(assigned,ta)
            or source["classes"]!=nc["classes"] or source["outer_splits"]!=nc["outer_splits"]
            or source["inner_splits"]!=nc["inner_splits"] or tc["split_seed"]!=nc["split_seed"]):
        raise ValueError("Neural/tensor classes, patients, labels or folds differ")
    frozen=folds.set_index("patient_id").sort_index()
    nr=validate_predictions(pd.read_csv(eroot/f"patient_predictions_{config['neural_arm']}.csv"),frozen,nc["classes"],esummary["metrics"][config["neural_arm"]]["macro_f1"])
    tm=json.loads((troot/"metrics_tabpfn.json").read_text())
    if not tm.get("complete") or tm.get("study_fingerprint")!=tscope["study_fingerprint"] or tm.get("classes")!=nc["classes"] or tm.get("n_patients")!=len(ids):
        raise ValueError("Tensor reference metrics provenance changed")
    tr=validate_predictions(pd.read_csv(troot/"patient_predictions_tabpfn.csv"),frozen,nc["classes"],tm["patient_metrics"]["macro_f1"])
    return {"nc":nc,"bank":bank,"assigned":assigned,"nroot":nroot,"nscope":nscope,"ncompleted":ncompleted,
            "eroot":eroot,"esummary":esummary,"tc":tc,"td":td,"tscope":tscope,"troot":troot,"tsummary":tsummary,
            "neural_reference":nr,"tensor_reference":tr,"classes":nc["classes"],"ids":ids,"y":y}


def source_paths(config,data):
    nc=data["nc"];nroot=data["nroot"];troot=data["troot"];eroot=data["eroot"];arm=config["neural_arm"]
    paths=[Path(config["neural_ensemble_config"]),Path(config["tensor_config"]),eroot/"summary.json",eroot/"analysis_manifest.json",
           eroot/f"patient_predictions_{arm}.csv",nroot/"study_manifest.json",nroot/"summary.json",
           troot/"study_manifest.json",troot/"summary.json",troot/"metrics_tabpfn.json",troot/"patient_predictions_tabpfn.csv"]
    for seed in nc["training_seeds"]:
        for pattern in (f"inner_*_{arm}_s{seed}_i*.json",f"inner_*_{arm}_s{seed}_i*.npz",f"outer_*_{arm}_s{seed}.json",f"outer_*_{arm}_s{seed}.npz",f"outer_*_{arm}_s{seed}.pt"):
            paths.extend(sorted(nroot.glob(pattern)))
    for pattern in ("inner_*_tabpfn_*.json","inner_*_tabpfn_*.npz","selection_*_tabpfn.json","fold_*_tabpfn.json","fold_*_tabpfn.npz","model_*_tabpfn.joblib","model_*_tabpfn.tabpfn_fit"):
        paths.extend(sorted(troot.glob(pattern)))
    return sorted(set(paths))


def make_scope(config,data):
    scope={"config":config,"claim_scope":CLAIM,"neural_fingerprint":data["nscope"]["study_fingerprint"],
           "tensor_fingerprint":data["tscope"]["study_fingerprint"],"code_sha256":file_hash(__file__),
           "report_code_sha256":file_hash(Path(__file__).with_name("neural_tensor_fusion_report.py")),
           "ensemble_helper_sha256":file_hash(neural.__file__),
           "source_file_sha256":{str(p):file_hash(p) for p in source_paths(config,data)}}
    return dict(scope,study_fingerprint=_hash(scope))


def neural_inner(data,config,fold,inner,train,valid,norm_memo):
    bank=data["bank"];nc=data["nc"];ids=data["ids"];arm=config["neural_arm"]
    values=[];choices={};audits=[]
    for seed in nc["training_seeds"]:
        fit_seed=seed+fold*1000+inner
        expected=nstudy.expected_fit(bank,data["nscope"],arm,seed,fold,inner,ids[train],ids[valid])
        path=data["nroot"]/f"inner_{fold:02d}_{arm}_s{seed}_i{inner}.json"
        audit,p=nstudy.cache.read_result(path,expected,ids[valid],len(data["classes"]))
        norm=nstudy.verify_fit(audit["fit"],bank,ids[train],nc,arm,fit_seed,ids[valid],norm_memo)
        plan,coverage=nstudy.chronology_model.inference_schedule(bank,ids[valid],nc["sequence_length"])
        ds=nstudy.models.ResidualDataset(bank,norm,plan,nc["sequence_length"],arm,fit_seed,0)
        coverage.update({k:v for k,v in ds.audit().items() if "order_sha256" in k})
        score=metric_bundle(data["y"][valid],p,data["classes"])["macro_f1"]
        if coverage!=audit["coverage"] or abs(score-audit["fit"]["best_validation_macro_f1"])>1e-12:
            raise ValueError("Neural inner coverage/stopping score changed")
        choices[seed]={"inner_fold":inner,"selected_epoch":audit["fit"]["selected_epoch"],"inner_macro_f1":score,"audit_sha256":file_hash(path)}
        audits.append({"seed":seed,"audit_sha256":file_hash(path),"training_membership_sha256":expected["train_membership_sha256"],"validation_membership_sha256":expected["query_membership_sha256"]})
        values.append(p)
    return neural.uniform_probability_mean(values),choices,audits


def tensor_inner(data,fold,inner,candidate_id,candidate,train,valid,memo):
    tc=data["tc"];path=data["troot"]/f"inner_{fold:02d}_tabpfn_{candidate_id}_{inner}.json"
    audit=json.loads(path.read_text());cache_key=audit["selector"]["bootstrap_cache_key"]
    prior_cache=Path(data["td"][-1]["prior_root"])/"bootstrap_cache"/f"{cache_key}.npz"
    if not prior_cache.is_file(): raise ValueError("Missing original selector cache; rebuild separately before fusion")
    seed=tc["seed"]+fold*1000+inner
    selector,_,expected=tensor.prepare(data["td"],tc,candidate,train,valid,seed,memo,verify=True)
    expected.update(study_fingerprint=data["tscope"]["study_fingerprint"],outer_fold=fold,inner_fold=inner,
                    candidate_id=candidate_id,candidate=candidate,seed=seed)
    if any(audit.get(k)!=v for k,v in expected.items()): raise ValueError("Tensor inner context/selector membership changed")
    values=tensor.read_probabilities(path.with_suffix(".npz"),audit,valid,len(data["classes"]))
    tensor.check_backend(audit["backend"],selector,len(data["classes"]),tc,seed)
    if set(values)!={"tabpfn","tabpfn_balanced"} or not np.allclose(values["tabpfn_balanced"],tensor.backend.balance(values["tabpfn"],data["y"][train],len(data["classes"])),rtol=1e-12,atol=1e-12):
        raise ValueError("Tensor cached probability policy changed")
    score=float(f1_score(data["y"][valid],values["tabpfn"].argmax(1),labels=np.arange(len(data["classes"])),average="macro",zero_division=0))
    if audit["validation_macro_f1"]["tabpfn"]!=score: raise ValueError("Tensor inner score changed")
    return values["tabpfn"],score,file_hash(path)


def neural_outer(data,config,fold,train,query,choices,norm_memo):
    ids=data["ids"];bank=data["bank"];nc=data["nc"];arm=config["neural_arm"]
    for seed in nc["training_seeds"]:
        rows=choices[seed];epochs=int(np.floor(np.median([r["selected_epoch"] for r in rows])+.5))
        expected=dict(nstudy.expected_fit(bank,data["nscope"],arm,seed,fold,0,ids[train],ids[query]),final_epochs=epochs,inner_choices=rows)
        path=data["nroot"]/f"outer_{fold:02d}_{arm}_s{seed}.json"
        audit,p=nstudy.cache.read_result(path,expected,ids[query],len(data["classes"]),outer=True)
        nstudy.verify_fit(audit["fit"],bank,ids[train],nc,arm,seed+fold,None,norm_memo)
        if len(audit["fit"]["history"])!=epochs or audit["patient_metrics"]!=metric_bundle(data["y"][query],p,data["classes"]):
            raise ValueError("Neural outer duration/source score changed")
    rows,p,audits=neural.predict_heldout_fold(bank,data["assigned"],nc,data["nroot"],data["nscope"],arm,fold,norm_memo)
    if not np.array_equal(rows,query) or not np.allclose(p,data["neural_reference"][query],rtol=1e-5,atol=1e-6):
        raise ValueError("Frozen neural ensemble reference changed")
    return p,audits


def tensor_outer(data,fold,train,query,candidate_id,candidate,memo):
    tc=data["tc"];seed=tc["seed"]+fold;root=data["troot"]
    path=root/f"fold_{fold:02d}_tabpfn.json";audit=json.loads(path.read_text())
    cache_key=audit["selector"]["bootstrap_cache_key"]
    if not (Path(data["td"][-1]["prior_root"])/"bootstrap_cache"/f"{cache_key}.npz").is_file():
        raise ValueError("Missing original outer selector cache")
    selector,_,expected=tensor.prepare(data["td"],tc,candidate,train,query,seed,memo,verify=True)
    expected.update(study_fingerprint=data["tscope"]["study_fingerprint"],fold=fold,arm="tabpfn",candidate_id=candidate_id,
                    candidate=candidate,seed=seed,selection_sha256=file_hash(root/f"selection_{fold:02d}_tabpfn.json"))
    if any(audit.get(k)!=v for k,v in expected.items()): raise ValueError("Tensor outer context/selection changed")
    cached=tensor.read_probabilities(path.with_suffix(".npz"),audit,query,len(data["classes"]))["probabilities"]
    model_path=root/f"model_{fold:02d}_tabpfn.joblib";state=root/f"model_{fold:02d}_tabpfn.tabpfn_fit"
    if audit["model_sha256"]!=file_hash(model_path) or audit["fit_state_sha256"]!=file_hash(state):
        raise ValueError("Tensor outer fitted model changed")
    model=joblib.load(model_path)
    if (model["selector"].audit()!=selector.audit() or any(model.get(k)!=v for k,v in
            {"study_fingerprint":data["tscope"]["study_fingerprint"],"fold":fold,"arm":"tabpfn","candidate":candidate}.items())):
        raise ValueError("Tensor outer saved selector/context provenance changed")
    X=data["td"][3][candidate["family"]]
    p,actual=tensor.backend.run_predict(tc,state,X[query][:,selector.selected])
    tensor.check_backend(actual,selector,len(data["classes"]),tc,seed)
    if (actual!=audit["backend"] or actual!=model["backend"] or not np.array_equal(p.argmax(1),cached.argmax(1))
            or not np.allclose(p,cached,rtol=1e-6,atol=1e-7) or not np.allclose(p,data["tensor_reference"][query],rtol=1e-6,atol=1e-7)
            or audit["patient_metrics"]!=metric_bundle(data["y"][query],cached,data["classes"])):
        raise ValueError("Restored tensor predictions differ from reference")
    return p,{"audit_sha256":file_hash(path),"model_sha256":file_hash(model_path),"fit_state_sha256":file_hash(state),"restore_verified":True}


def evaluate_fold(data,config,scope,root,fold,verify=False):
    assigned=data["assigned"];ids=data["ids"];y=data["y"];nc=data["nc"]
    train,query=np.flatnonzero(assigned!=fold),np.flatnonzero(assigned==fold)
    splits=[(train[a],train[b]) for a,b in inner_patient_splits(y[train],ids[train],nc["inner_splits"],nc["split_seed"]+fold*1000)]
    ni=[];truth=[];inner_audits=[];nchoices={s:[] for s in nc["training_seeds"]};nmemo={};tmemo={}
    for inner,(fit,valid) in enumerate(splits,1):
        check_partition(fit,valid,query,ids)
        p,choices,audits=neural_inner(data,config,fold,inner,fit,valid,nmemo)
        ni.append(p);truth.append(y[valid]);inner_audits.append(audits)
        for seed,row in choices.items():nchoices[seed].append(row)
        _progress(root,state="verifying_inner_neural",outer_fold=fold,inner_fold=inner)
    candidate_rows=[];candidate_predictions=[]
    candidates=tensor.pfn_candidates(data["td"][0],data["td"][1])
    for candidate_id,candidate in enumerate(candidates):
        values=[tensor_inner(data,fold,inner,candidate_id,candidate,fit,valid,tmemo) for inner,(fit,valid) in enumerate(splits,1)]
        candidate_predictions.append([v[0] for v in values])
        candidate_rows.append({"candidate_id":candidate_id,"candidate":candidate,"mean_inner_macro_f1":float(np.mean([v[1] for v in values])),"inner_audits_sha256":[v[2] for v in values]})
        _progress(root,state="verifying_inner_tensor",outer_fold=fold,candidate_id=candidate_id)
    original_selection={"study_fingerprint":data["tscope"]["study_fingerprint"],"candidates":candidate_rows}
    if json.loads((data["troot"]/f"selection_{fold:02d}_tabpfn.json").read_text())!=original_selection:
        raise ValueError("Original training-only tensor candidate selection changed")
    chosen=max(range(len(candidate_rows)),key=lambda i:candidate_rows[i]["mean_inner_macro_f1"])
    alpha,weights=choose_weight(ni,candidate_predictions[chosen],truth,data["classes"],config["neural_weight_grid"])
    decision={"study_fingerprint":scope["study_fingerprint"],"outer_fold":fold,"train_membership_sha256":_hash(ids[train].tolist()),
              "query_membership_sha256":_hash(ids[query].tolist()),"neural_inner_audits":inner_audits,"tensor_candidates":candidate_rows,
              "tensor_candidate_id":chosen,"tensor_candidate":candidates[chosen],"weight_choices":weights,"selected_neural_weight":alpha,
              "selection_scope":"Only outer-training inner validation labels; original tensor selection then scalar blending"}
    selection_path=root/f"selection_{fold:02d}.json"
    if verify:
        if json.loads(selection_path.read_text())!=decision:raise ValueError("Fusion training-only weight decision changed")
    else:_save_json(selection_path,decision)
    # Only after the scalar decision is fixed do we restore/read outer outcomes.
    _progress(root,state="restoring_outer_components",outer_fold=fold,selected_neural_weight=alpha)
    pn,naudit=neural_outer(data,config,fold,train,query,nchoices,nmemo)
    pt,taudit=tensor_outer(data,fold,train,query,chosen,candidates[chosen],tmemo)
    values={"neural_only":pn,"tensor_only":pt,"equal_blend":blend(pn,pt,.5),"selected_blend":blend(pn,pt,alpha)}
    audit={"study_fingerprint":scope["study_fingerprint"],"outer_fold":fold,"selection_sha256":file_hash(selection_path),
           "selected_neural_weight":alpha,"neural_members":naudit,"tensor_member":taudit,"patient_overlap":0,
           "query_patients":len(query),"verified_neural_inner_contexts":len(splits)*len(nc["training_seeds"]),
           "verified_tensor_inner_contexts":len(splits)*len(candidates),"tensor_bootstrap_caches_replayed":len(tmemo)}
    path=root/f"fold_{fold:02d}.json";checkpoint=path.with_suffix(".npz")
    if verify:
        saved=json.loads(path.read_text())
        if any(saved.get(k)!=v for k,v in audit.items()) or saved["probabilities_sha256"]!=file_hash(checkpoint):
            raise ValueError("Fusion outer source/audit changed")
        with np.load(checkpoint) as z:
            if not np.array_equal(z["indices"],query):raise ValueError("Fusion outer patient membership changed")
            for name,p in values.items():
                if not np.array_equal(z[name].argmax(1),p.argmax(1)) or not np.allclose(z[name],p,atol=1e-6,rtol=1e-5):
                    raise ValueError("Fusion saved predictions do not reproduce")
    else:
        temporary=checkpoint.with_suffix(".partial.npz");np.savez_compressed(temporary,indices=query,**values);temporary.replace(checkpoint)
        _save_json(path,dict(audit,probabilities_sha256=file_hash(checkpoint)))
    _progress(root,state="verifying" if verify else "outer_complete",outer_fold=fold,selected_neural_weight=alpha,
              completed_folds=fold,source_contexts_verified=fold*(len(nc["training_seeds"])*(nc["inner_splits"]+1)+len(candidates)*nc["inner_splits"]+1))
    return query,values,audit,decision


def run_study(config):
    data=load_inputs(config);scope=make_scope(config,data)
    root=Path(config["output_dir"])/"complete";root.mkdir(parents=True,exist_ok=True)
    path=root/"study_manifest.json"
    if path.exists() and json.loads(path.read_text())!=scope:raise ValueError("Fusion source/code/config changed; use a new namespace")
    _save_json(path,scope)
    for fold in range(1,data["nc"]["outer_splits"]+1):
        evaluate_fold(data,config,scope,root,fold,verify=(root/f"fold_{fold:02d}.json").is_file())
    _progress(root,state="components_and_blends_complete_report_pending")
    return root


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default="configs/neural_tensor_fusion_v1.yaml")
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    try:print(run_study(config))
    except Exception as error:
        _progress(Path(config["output_dir"])/"complete",state="failed",error_type=type(error).__name__,error=str(error));raise
