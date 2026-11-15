"""Matched VCG raw/template/residual view experiment; unchanged patient task."""
from __future__ import annotations

import argparse
import json
import hashlib
import platform
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import neural, multiview_bank
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights, save_confusion
from .contrastive import initial_state_hash, _progress
from .neural import VCGResNet, _device, _hash, _save_json, _seed
from .preprocessing_study import WaveRows, predict_waves, directory
from .signal_audit import file_hash
from .multiview_bank import ARMS, ViewBank, array_hash, select_normalization, fit_normalization

CLAIM_SCOPE = "Measured-VCG development on repeatedly inspected patient folds; information-preserving views are not new patients, verified clean physiology or SOTA."
SLOTS = {"vcg":[1], "capacity_control":[1]*5, "raw_denoised":[0,1,1,1,1], "template_only":[2]*5,
    "template_residual":[1,1,2,3,1], "full_tensor":[0,1,2,3,4]}
CONTRASTS = (("vcg","full_tensor"), ("capacity_control","full_tensor"), ("raw_denoised","full_tensor"),
    ("capacity_control","template_residual"), ("template_residual","full_tensor"), ("capacity_control","raw_denoised"),
    ("capacity_control","template_only"), ("template_only","full_tensor"))


def initial_model(config, seed, arm):
    if arm not in ARMS:
        raise ValueError("Unknown multiview model")
    _seed(seed, config)
    model = VCGResNet(len(config["classes"]), tuple(config["widths"]), config["pool_bins"], config["dropout"])
    if arm != "vcg":
        original = model.encoder[0]
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Conv1d(15, config["widths"][0], 15, stride=2, padding=7, bias=False)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:,3:6].copy_(original.weight)
        model.encoder[0] = expanded
    return model


def normalizer(full, y, groups, train, weights, root, fingerprint, arm):
    scope = {"study_fingerprint":fingerprint,"train_indices_sha256":array_hash(train),"weights_sha256":array_hash(weights)}
    key = _hash(scope)
    path = Path(root)/"normalization"/f"{key}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            mean, scale = saved["mean"], saved["scale"]
            if json.loads(str(saved["scope"].item())) != scope or str(saved["content_sha256"].item()) != _hash([array_hash(mean),array_hash(scale)]):
                raise ValueError("Multiview training-normalization cache changed")
    else:
        mean, scale = fit_normalization(full,y,groups,train,weights)
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary,mean=mean,scale=scale,scope=json.dumps(scope),content_sha256=_hash([array_hash(mean),array_hash(scale)]))
        temporary.replace(path)
    if mean.shape!=(5,3) or scale.shape!=(5,3) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError("Invalid multiview training normalizer")
    return select_normalization(mean,scale,arm)


def fit(waves, full, y, groups, train, valid, epochs, config, seed, arm, root, fingerprint, progress=None):
    if valid is not None and set(groups[train])&set(groups[valid]):
        raise ValueError("Multiview training/validation patient overlap")
    weights = patient_balanced_weights(y[train],groups[train])
    norm = normalizer(full,y,groups,train,weights,root,fingerprint,arm)
    model = initial_model(config,seed,arm).to(_device(config))
    initial, head = initial_state_hash(model), initial_state_hash(model.classifier)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    sampler = WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),config["samples_per_epoch"],replacement=True,generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(WaveRows(waves,y,train,norm),batch_size=config["batch_size"],sampler=sampler,num_workers=0)
    best,best_epoch,best_state,history = -np.inf,0,None,[]
    for epoch in range(1,epochs+1):
        model.train()
        losses,count,digest = [],0,hashlib.sha256()
        for x,labels,rows in loader:
            digest.update(rows.numpy().astype("<i8").tobytes())
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(_device(config))),labels.to(_device(config)))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite multiview loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            count += len(labels)
        row = {"epoch":epoch,"training_loss":float(np.mean(losses)),"sampled_rows_sha256":digest.hexdigest(),"beat_presentations":count}
        if valid is not None:
            p = predict_waves(model,norm,waves,y,valid,config)
            _,truth,scores = aggregate_patients(p,y[valid],groups[valid])
            score = float(f1_score(truth,scores.argmax(1),labels=np.arange(len(config["classes"])),average="macro",zero_division=0))
            row["validation_patient_macro_f1"] = score
            if score>best+1e-8:
                best,best_epoch = score,epoch
                best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        history.append(row)
        if progress:
            progress(row)
        if valid is not None and epoch-best_epoch>=config["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model,norm,{"seed":seed,"arm":arm,"initial_state_sha256":initial,"initial_classifier_sha256":head,
        "parameter_count":sum(p.numel() for p in model.parameters()),"selected_epoch":best_epoch if valid is not None else epochs,
        "best_validation_macro_f1":best if valid is not None else None,"history":history,"patient_overlap":0,
        "train_patients":len(np.unique(groups[train])),"validation_patients":0 if valid is None else len(np.unique(groups[valid])),
        "normalizer_fit_beats":len(train),"normalization":norm.audit()}


def validate_config(config):
    if config["arms"]!=list(ARMS) or not config["training_seeds"] or len(set(config["training_seeds"]))!=len(config["training_seeds"]):
        raise ValueError("Require all six multiview arms and unique training seeds")
    for key in ["batch_size","samples_per_epoch","max_epochs","patience","torch_threads","inner_splits","outer_splits"]:
        if not isinstance(config[key],int) or isinstance(config[key],bool) or config[key]<(2 if key in {"inner_splits","outer_splits"} else 1):
            raise ValueError("Invalid multiview training budget")
    if not np.isfinite([config["learning_rate"],config["weight_decay"],config["dropout"]]).all() or config["learning_rate"]<=0 or config["weight_decay"]<0 or not 0<=config["dropout"]<1:
        raise ValueError("Invalid multiview optimizer settings")


def validate_fit(audit, config, seed, arm, train, valid, groups, expected_norm, final_epochs=None):
    model = initial_model(config,seed,arm)
    if audit["seed"]!=seed or audit["arm"]!=arm or audit["patient_overlap"] or audit["initial_state_sha256"]!=initial_state_hash(model) or audit["initial_classifier_sha256"]!=initial_state_hash(model.classifier) or audit["parameter_count"]!=sum(p.numel() for p in model.parameters()) or audit["normalizer_fit_beats"]!=len(train) or audit["train_patients"]!=len(np.unique(groups[train])) or audit["validation_patients"]!=(0 if valid is None else len(np.unique(groups[valid]))) or audit["normalization"]!=expected_norm.audit():
        raise ValueError("Multiview fit scope/initialization/normalization mismatch")
    history = audit["history"]
    if not history or len(history)>config["max_epochs"]:
        raise ValueError("Invalid multiview training history")
    for epoch,row in enumerate(history,1):
        if row["epoch"]!=epoch or row["beat_presentations"]!=config["samples_per_epoch"] or not np.isfinite(row["training_loss"]) or len(row["sampled_rows_sha256"])!=64:
            raise ValueError("Invalid multiview sampling budget")
    if final_epochs is not None:
        if len(history)!=final_epochs or audit["selected_epoch"]!=final_epochs or audit["best_validation_macro_f1"] is not None or valid is not None or any("validation_patient_macro_f1" in row for row in history):
            raise ValueError("Multiview refit must not monitor held-out patients")
    else:
        values = [row.get("validation_patient_macro_f1",np.nan) for row in history]
        if not np.isfinite(values).all() or min(values)<0 or max(values)>1:
            raise ValueError("Invalid inner patient scores")
        best,selected = -np.inf,0
        for epoch,value in enumerate(values,1):
            if value>best+1e-8:
                best,selected = value,epoch
        if audit["selected_epoch"]!=selected or audit["best_validation_macro_f1"]!=best or (len(history)<config["max_epochs"] and len(history)-selected!=config["patience"]):
            raise ValueError("Multiview inner stopping mismatch")


def check_pair(first,second):
    for key in ["seed","initial_classifier_sha256","normalizer_fit_beats","train_patients","validation_patients"]:
        if first[key]!=second[key]:
            raise ValueError("Matched multiview fits differ: "+key)
    if first["arm"]!="vcg" and second["arm"]!="vcg":
        for key in ["initial_state_sha256","parameter_count"]:
            if first[key]!=second[key]:
                raise ValueError("Multiview capacity/initialization mismatch")
    for slot in set(SLOTS[first["arm"]])&set(SLOTS[second["arm"]]):
        ia,ib = SLOTS[first["arm"]].index(slot),SLOTS[second["arm"]].index(slot)
        for key in ["wave_mean_","wave_scale_"]:
            if first["normalization"][key][ia*3:ia*3+3]!=second["normalization"][key][ib*3:ib*3+3]:
                raise ValueError("Shared-view training normalization mismatch")
    for a,b in zip(first["history"],second["history"]):
        if a["sampled_rows_sha256"]!=b["sampled_rows_sha256"]:
            raise ValueError("Multiview arms did not receive matched beat draws")


def load_inputs(config):
    d,raw,templates,indices,y,groups,bank = multiview_bank.prepare_bank(config)
    views = {arm:ViewBank(d,raw,templates,indices,arm) for arm in ARMS}
    return views,y,groups,bank


def run_study(config,outer_folds=None):
    validate_config(config)
    views,y,groups,bank = load_inputs(config)
    full = views["full_tensor"]
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require existing frozen patient folds")
    folds = load_or_create_folds(y,groups,config)
    if sorted(folds.fold.unique())!=list(range(1,config["outer_splits"]+1)):
        raise ValueError("Frozen multiview fold IDs changed")
    root,chosen,partial = directory(config,outer_folds)
    scope = {"config":config,"bank":bank,"folds_sha256":file_hash(config["folds_path"]),"evaluated_folds":chosen,"claim_scope":CLAIM_SCOPE,
        "stem_initialization":"original XYZ weights on second view only; other input-view weights zero; CPU initialization stream preserved",
        "runtime":{"torch":str(torch.__version__),"numpy":np.__version__,"platform":platform.platform(),"device":str(_device(config)),
            "deterministic_kernels":_device(config).type!="mps","code_sha256":file_hash(__file__),"neural_code_sha256":file_hash(neural.__file__)}}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True,exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text())!=scope:
        raise ValueError("Multiview settings/data/runtime changed; use a new output_dir")
    _save_json(guard,scope)
    lookup = dict(zip(folds.patient_id.astype(str),folds.fold,strict=True))
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    summary_rows = []
    for seed in config["training_seeds"]:
        output = root/f"seed_{seed}"
        output.mkdir(exist_ok=True)
        probabilities = {arm:np.full((len(y),len(config["classes"])),np.nan) for arm in ARMS}
        audits = {arm:[] for arm in ARMS}
        for fold in chosen:
            train,test = np.flatnonzero(beat_folds!=fold),np.flatnonzero(beat_folds==fold)
            if set(groups[train])&set(groups[test]):
                raise ValueError("Multiview outer patient overlap")
            splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],groups[train],config["inner_splits"],config["split_seed"]+fold*1000)]
            for arm in ARMS:
                waves = views[arm]
                checkpoint,audit_path = output/f"fold_{fold:02d}_{arm}.npz",output/f"fold_{fold:02d}_{arm}.json"
                if checkpoint.exists() and audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    if audit["study_fingerprint"]!=fingerprint or audit["prediction_checkpoint_sha256"]!=file_hash(checkpoint):
                        raise ValueError("Multiview completed checkpoint changed")
                    with np.load(checkpoint) as saved:
                        if not np.array_equal(saved["indices"],test):
                            raise ValueError("Multiview test membership changed")
                        probabilities[arm][test] = saved["probabilities"]
                    audits[arm].append(audit)
                    continue
                selections = []
                for inner,(ti,vi) in enumerate(splits,1):
                    path = output/f"inner_{fold:02d}_{arm}_{inner}.json"
                    if path.exists():
                        selected = json.loads(path.read_text())
                        if selected["study_fingerprint"]!=fingerprint or selected["inner_fold"]!=inner:
                            raise ValueError("Multiview inner checkpoint changed")
                    else:
                        callback = lambda row,inner=inner:_progress(root,state="training",training_seed=seed,arm=arm,outer_fold=fold,inner_fold=inner,**row)
                        model,norm,selected = fit(waves,full,y,groups,ti,vi,config["max_epochs"],config,seed+fold*1000+inner,arm,root,fingerprint,callback)
                        selected.update(study_fingerprint=fingerprint,inner_fold=inner)
                        del model,norm
                    expected = normalizer(full,y,groups,ti,patient_balanced_weights(y[ti],groups[ti]),root,fingerprint,arm)
                    validate_fit(selected,config,seed+fold*1000+inner,arm,ti,vi,groups,expected)
                    _save_json(path,selected)
                    selections.append(selected)
                    print(f"seed {seed} {arm} outer {fold} inner {inner}: epoch {selected['selected_epoch']}, F1 {selected['best_validation_macro_f1']:.4f}",flush=True)
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in selections])+.5))
                callback = lambda row:_progress(root,state="training",training_seed=seed,arm=arm,outer_fold=fold,inner_fold=None,**row)
                model,norm,final = fit(waves,full,y,groups,train,None,epochs,config,seed+fold,arm,root,fingerprint,callback)
                validate_fit(final,config,seed+fold,arm,train,None,groups,norm,epochs)
                p = predict_waves(model,norm,waves,y,test,config)
                probabilities[arm][test] = p
                _,truth,scores = aggregate_patients(p,y[test],groups[test])
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez_compressed(temporary,indices=test,probabilities=p)
                temporary.replace(checkpoint)
                audit = {"study_fingerprint":fingerprint,"training_seed":seed,"arm":arm,"fold":fold,"patient_overlap":0,
                    "train_beats":len(train),"test_beats":len(test),"train_patients":len(np.unique(groups[train])),"test_patients":len(np.unique(groups[test])),
                    "inner_selection":selections,"final_epochs":epochs,"final_fit":final,"prediction_checkpoint_sha256":file_hash(checkpoint),"patient_metrics":metric_bundle(truth,scores,config["classes"])}
                model_path = output/f"model_{fold:02d}_{arm}.pt"
                temporary = model_path.with_suffix(".pt.partial")
                torch.save({"state_dict":{key:value.detach().cpu() for key,value in model.state_dict().items()},"normalization":norm.audit(),"arm":arm,"study_fingerprint":fingerprint},temporary)
                temporary.replace(model_path)
                audit["model_sha256"] = file_hash(model_path)
                _save_json(audit_path,audit)
                audits[arm].append(audit)
                del model,norm
                _progress(root,state="outer_fold_complete",training_seed=seed,arm=arm,outer_fold=fold)
                print(f"seed {seed} {arm} outer {fold}/{config['outer_splits']}: F1 {audit['patient_metrics']['macro_f1']:.4f}",flush=True)
            for arm in ARMS[1:]:
                for a,b in zip(audits["vcg"][-1]["inner_selection"],audits[arm][-1]["inner_selection"]):
                    check_pair(a,b)
                check_pair(audits["vcg"][-1]["final_fit"],audits[arm][-1]["final_fit"])
            for arm in ARMS[2:]:
                for a,b in zip(audits["capacity_control"][-1]["inner_selection"],audits[arm][-1]["inner_selection"]):
                    check_pair(a,b)
                check_pair(audits["capacity_control"][-1]["final_fit"],audits[arm][-1]["final_fit"])
        for arm in ARMS:
            mask = np.isfinite(probabilities[arm]).all(1)
            if not partial and not mask.all():
                raise ValueError("Missing multiview predictions")
            p = probabilities[arm][mask]
            if np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-5):
                raise ValueError("Invalid multiview probabilities")
            ids,truth,scores = aggregate_patients(p,y[mask],groups[mask])
            saved = {"study_fingerprint":fingerprint,"training_seed":seed,"arm":arm,"classes":config["classes"],"complete":not partial,
                "evaluated_folds":chosen,"n_patients":len(ids),"n_beats":int(mask.sum()),"patient_metrics":metric_bundle(truth,scores,config["classes"]),"folds":audits[arm],"claim_scope":CLAIM_SCOPE}
            _save_json(output/f"metrics_{arm}.json",saved)
            frame = pd.DataFrame({"patient_id":ids,"truth":[config["classes"][i] for i in truth],"prediction":[config["classes"][i] for i in scores.argmax(1)],"fold":[lookup[str(patient)] for patient in ids]})
            for i,label in enumerate(config["classes"]):
                frame[f"probability_{label}"] = scores[:,i]
            frame.to_csv(output/f"patient_predictions_{arm}.csv",index=False)
            save_confusion(truth,scores.argmax(1),config["classes"],arm,output)
            summary_rows.append({"training_seed":seed,"arm":arm,"complete":not partial,"patient_macro_f1":saved["patient_metrics"]["macro_f1"]})
            pd.DataFrame(summary_rows).to_csv(root/"comparison.csv",index=False)
    _progress(root,state="partial_complete" if partial else "complete")
    return pd.DataFrame(summary_rows)


def status(config):
    root,_,_ = directory(config)
    lines = [f"Information-preserving VCG view study: {root}"]
    for seed in config["training_seeds"]:
        for arm in ARMS:
            lines.append(f"seed {seed} {arm}: {len(list((root/f'seed_{seed}').glob(f'fold_*_{arm}.json')))}/{config['outer_splits']} completed outer folds")
    for name in ["progress.json","last_error.json"]:
        if (root/name).exists():
            lines.append(f"Last saved {name} (not a live process check): "+(root/name).read_text())
    lines.append(f"Validated final report: {(root/'REPORT.md').exists()}")
    return "\n".join(lines)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/multiview_vcg.yaml")
    parser.add_argument("--outer-folds",nargs="+",type=int)
    parser.add_argument("--status",action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.status:
        print(status(config))
    else:
        try:
            print(run_study(config,args.outer_folds).to_string(index=False))
        except BaseException as error:
            root,_,_ = directory(config,args.outer_folds)
            root.mkdir(parents=True,exist_ok=True)
            _save_json(root/"last_error.json",{"type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
            raise
