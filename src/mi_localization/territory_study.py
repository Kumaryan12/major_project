"""Six-class VCG localization with label-derived shared-descriptor supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
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

from . import neural
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights, save_confusion
from .contrastive import initial_state_hash, _progress
from .neural import FoldNormalizer, VCGResNet, _device, _hash, _save_json, _seed
from .preprocessing_study import WaveRows, predict_waves, directory
from .regularization import load_bank
from .signal_audit import file_hash

ARMS = ("ce", "territory")
DESCRIPTORS = ("anterior", "inferior", "lateral", "septal")
MENTIONS = {"AMI": [1,0,0,0], "ALMI": [1,0,1,0], "ASMI": [1,0,0,1],
    "IMI": [0,1,0,0], "ILMI": [0,1,1,0], "HC": [0,0,0,0]}
CLAIM_SCOPE = "VCG-only development on repeatedly inspected patient folds; auxiliary targets are source-label mentions, not independently verified anatomical diagnoses."


def codebook(config):
    try:
        return np.asarray([MENTIONS[label] for label in config["classes"]], dtype=np.float32)
    except KeyError as error:
        raise ValueError("Unknown original label for shared-descriptor supervision") from error


class TerritoryNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        base = VCGResNet(len(config["classes"]), tuple(config["widths"]), config["pool_bins"], config["dropout"])
        self.encoder, self.classifier = base.encoder, base.classifier
        # Keep the original CPU/dropout random stream intact. The extra head
        # exists in BOTH arms; its initialization is not an experimental change.
        with torch.random.fork_rng(devices=[]):
            self.auxiliary = nn.Linear(64, len(DESCRIPTORS))

    def forward(self, waves, return_auxiliary=False):
        z = self.encoder(waves)
        logits = self.classifier(z)
        return (logits, self.auxiliary(z)) if return_auxiliary else logits


def initial_model(config, seed):
    _seed(seed, config)
    return TerritoryNet(config)


def fit(waves, y, groups, train, valid, epochs, config, seed, arm, progress=None):
    if arm not in ARMS or (valid is not None and set(groups[train]) & set(groups[valid])):
        raise ValueError("Unknown arm or overlapping training/validation patients")
    weights = patient_balanced_weights(y[train], groups[train])
    norm = FoldNormalizer().fit(waves, np.empty((len(waves), 0), np.float32), train, weights)
    model = initial_model(config, seed).to(_device(config))
    initial = initial_state_hash(model)
    coefficient = 0. if arm == "ce" else config["territory_weight"]
    targets = torch.as_tensor(codebook(config), device=_device(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), config["samples_per_epoch"], replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(WaveRows(waves, y, train, norm), batch_size=config["batch_size"], sampler=sampler, num_workers=0)
    best, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, epochs+1):
        model.train()
        losses, ces, auxiliaries, count, digest = [], [], [], 0, hashlib.sha256()
        for x, labels, rows in loader:
            digest.update(rows.numpy().astype("<i8").tobytes())
            labels = labels.to(_device(config))
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(x.to(_device(config)), return_auxiliary=True)
            ce = F.cross_entropy(logits, labels)
            descriptor_loss = F.binary_cross_entropy_with_logits(auxiliary, targets[labels])
            loss = ce + coefficient*descriptor_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite shared-descriptor loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ces.append(float(ce.detach().cpu()))
            auxiliaries.append(float(descriptor_loss.detach().cpu()))
            count += len(labels)
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)), "cross_entropy": float(np.mean(ces)),
            "descriptor_bce": float(np.mean(auxiliaries)), "sampled_rows_sha256": digest.hexdigest(), "beat_presentations": count}
        if valid is not None:
            p = predict_waves(model, norm, waves, y, valid, config)
            _, truth, scores = aggregate_patients(p, y[valid], groups[valid])
            score = float(f1_score(truth, scores.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
            row["validation_patient_macro_f1"] = score
            if score > best + 1e-8:
                best, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
        if progress:
            progress(row)
        if valid is not None and epoch-best_epoch >= config["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm, {"seed": seed, "arm": arm, "auxiliary_weight": coefficient, "initial_state_sha256": initial,
        "parameter_count": sum(p.numel() for p in model.parameters()), "selected_epoch": best_epoch if valid is not None else epochs,
        "best_validation_macro_f1": best if valid is not None else None, "history": history, "patient_overlap": 0,
        "train_patients": len(np.unique(groups[train])), "validation_patients": 0 if valid is None else len(np.unique(groups[valid])),
        "normalizer_fit_beats": len(train), "normalization": norm.audit()}


def validate_config(config):
    codebook(config)
    if config["arms"] != list(ARMS) or len(config["training_seeds"]) != len(set(config["training_seeds"])) or not config["training_seeds"]:
        raise ValueError("Require both shared-supervision arms and unique target seeds")
    for key in ["batch_size", "samples_per_epoch", "max_epochs", "patience", "torch_threads", "outer_splits", "inner_splits"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < (2 if key in {"outer_splits", "inner_splits"} else 1):
            raise ValueError("Invalid shared-supervision budget")
    if not np.isfinite([config["territory_weight"], config["learning_rate"], config["weight_decay"], config["dropout"]]).all() or config["territory_weight"] <= 0 or config["learning_rate"] <= 0 or config["weight_decay"] < 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("Invalid objective or optimizer settings")


def validate_fit(audit, config, seed, arm, train, valid, groups, final_epochs=None):
    expected = initial_model(config, seed)
    weight = 0 if arm == "ce" else config["territory_weight"]
    if audit["seed"] != seed or audit["arm"] != arm or audit["auxiliary_weight"] != weight or audit["patient_overlap"] or audit["initial_state_sha256"] != initial_state_hash(expected) or audit["parameter_count"] != sum(p.numel() for p in expected.parameters()) or audit["normalizer_fit_beats"] != len(train) or audit["train_patients"] != len(np.unique(groups[train])) or audit["validation_patients"] != (0 if valid is None else len(np.unique(groups[valid]))):
        raise ValueError("Shared-supervision fit scope/initialization mismatch")
    norm = audit["normalization"]
    if norm["stat_mean_"] or norm["stat_scale_"] or len(norm["wave_mean_"]) != 3 or len(norm["wave_scale_"]) != 3 or not np.isfinite(norm["wave_mean_"]+norm["wave_scale_"]).all() or min(norm["wave_scale_"]) <= 0:
        raise ValueError("Invalid training-only XYZ normalization")
    history = audit["history"]
    if not history or len(history) > config["max_epochs"]:
        raise ValueError("Invalid shared-supervision history")
    for epoch, row in enumerate(history, 1):
        if row["epoch"] != epoch or row["beat_presentations"] != config["samples_per_epoch"] or not np.isfinite([row["training_loss"],row["cross_entropy"],row["descriptor_bce"]]).all() or not np.isclose(row["training_loss"], row["cross_entropy"]+weight*row["descriptor_bce"], atol=1e-6) or len(row["sampled_rows_sha256"]) != 64:
            raise ValueError("Shared-supervision objective/budget mismatch")
    if final_epochs is not None:
        if len(history) != final_epochs or audit["selected_epoch"] != final_epochs or audit["best_validation_macro_f1"] is not None or valid is not None or any("validation_patient_macro_f1" in row for row in history):
            raise ValueError("Outer refit must not monitor held-out patients")
    else:
        values = [row.get("validation_patient_macro_f1", np.nan) for row in history]
        if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
            raise ValueError("Invalid inner validation history")
        best, selected = -np.inf, 0
        for epoch, value in enumerate(values, 1):
            if value > best+1e-8:
                best, selected = value, epoch
        if audit["selected_epoch"] != selected or audit["best_validation_macro_f1"] != best or (len(history) < config["max_epochs"] and len(history)-selected != config["patience"]):
            raise ValueError("Inner epoch selection mismatch")


def check_pair(first, second):
    for key in ["seed", "initial_state_sha256", "parameter_count", "normalization", "normalizer_fit_beats", "train_patients", "validation_patients"]:
        if first[key] != second[key]:
            raise ValueError("Shared-supervision matched fit mismatch: "+key)
    for a, b in zip(first["history"], second["history"]):
        if a["sampled_rows_sha256"] != b["sampled_rows_sha256"]:
            raise ValueError("Shared-supervision sampled rows differ")


def run_study(config, outer_folds=None):
    validate_config(config)
    waves, y, groups, bank = load_bank(config)
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require frozen patient folds")
    folds = load_or_create_folds(y, groups, config)
    if sorted(folds.fold.unique()) != list(range(1, config["outer_splits"]+1)):
        raise ValueError("Frozen fold IDs changed")
    root, chosen, partial = directory(config, outer_folds)
    scope = {"config": config, "bank": bank, "folds_sha256": file_hash(config["folds_path"]), "evaluated_folds": chosen,
        "descriptors": list(DESCRIPTORS), "codebook": codebook(config).tolist(), "claim_scope": CLAIM_SCOPE,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "device": str(_device(config)),
            "deterministic_kernels": _device(config).type != "mps", "code_sha256": file_hash(__file__), "neural_code_sha256": file_hash(neural.__file__)}}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    root.mkdir(parents=True, exist_ok=True)
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError("Shared-supervision study settings/data/runtime changed")
    _save_json(guard, scope)
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(patient)] for patient in groups])
    summary_rows = []
    for seed in config["training_seeds"]:
        output = root/f"seed_{seed}"
        output.mkdir(exist_ok=True)
        probabilities = {arm: np.full((len(y), len(config["classes"])), np.nan) for arm in ARMS}
        audits = {arm: [] for arm in ARMS}
        for fold in chosen:
            train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            if set(groups[train]) & set(groups[test]):
                raise ValueError("Outer patient overlap")
            splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],groups[train],config["inner_splits"],config["split_seed"]+fold*1000)]
            for arm in ARMS:
                checkpoint, audit_path = output/f"fold_{fold:02d}_{arm}.npz", output/f"fold_{fold:02d}_{arm}.json"
                if checkpoint.exists() and audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    if audit["study_fingerprint"] != fingerprint or audit["prediction_checkpoint_sha256"] != file_hash(checkpoint):
                        raise ValueError("Shared-supervision completed checkpoint changed")
                    with np.load(checkpoint) as saved:
                        if not np.array_equal(saved["indices"],test):
                            raise ValueError("Checkpoint test membership changed")
                        probabilities[arm][test] = saved["probabilities"]
                    audits[arm].append(audit)
                    continue
                selections = []
                for inner,(ti,vi) in enumerate(splits,1):
                    path = output/f"inner_{fold:02d}_{arm}_{inner}.json"
                    if path.exists():
                        inner_audit = json.loads(path.read_text())
                        if inner_audit["study_fingerprint"] != fingerprint or inner_audit["inner_fold"] != inner:
                            raise ValueError("Shared-supervision inner checkpoint changed")
                    else:
                        callback = lambda row,inner=inner: _progress(root,state="training",training_seed=seed,arm=arm,outer_fold=fold,inner_fold=inner,**row)
                        model,norm,inner_audit = fit(waves,y,groups,ti,vi,config["max_epochs"],config,seed+fold*1000+inner,arm,callback)
                        inner_audit.update(study_fingerprint=fingerprint,inner_fold=inner)
                        del model,norm
                    validate_fit(inner_audit,config,seed+fold*1000+inner,arm,ti,vi,groups)
                    _save_json(path,inner_audit)
                    selections.append(inner_audit)
                    print(f"seed {seed} {arm} outer {fold} inner {inner}: epoch {inner_audit['selected_epoch']}, F1 {inner_audit['best_validation_macro_f1']:.4f}",flush=True)
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in selections])+.5))
                callback = lambda row: _progress(root,state="training",training_seed=seed,arm=arm,outer_fold=fold,inner_fold=None,**row)
                model,norm,final = fit(waves,y,groups,train,None,epochs,config,seed+fold,arm,callback)
                validate_fit(final,config,seed+fold,arm,train,None,groups,epochs)
                p = predict_waves(model,norm,waves,y,test,config)
                probabilities[arm][test] = p
                _,truth,scores = aggregate_patients(p,y[test],groups[test])
                temporary = checkpoint.with_suffix(".partial.npz")
                np.savez_compressed(temporary,indices=test,probabilities=p)
                temporary.replace(checkpoint)
                audit = {"study_fingerprint":fingerprint,"training_seed":seed,"arm":arm,"fold":fold,"patient_overlap":0,
                    "train_beats":len(train),"test_beats":len(test),"train_patients":len(np.unique(groups[train])),"test_patients":len(np.unique(groups[test])),
                    "inner_selection":selections,"final_epochs":epochs,"final_fit":final,"prediction_checkpoint_sha256":file_hash(checkpoint),
                    "patient_metrics":metric_bundle(truth,scores,config["classes"])}
                _save_json(audit_path,audit)
                torch.save({"state_dict":{key:value.detach().cpu() for key,value in model.state_dict().items()},"normalization":norm.audit(),"study_fingerprint":fingerprint},output/f"model_{fold:02d}_{arm}.pt")
                audits[arm].append(audit)
                del model,norm
                _progress(root,state="outer_fold_complete",training_seed=seed,arm=arm,outer_fold=fold)
                print(f"seed {seed} {arm} outer {fold}/{config['outer_splits']}: F1 {audit['patient_metrics']['macro_f1']:.4f}",flush=True)
            for a,b in zip(audits["ce"][-1]["inner_selection"],audits["territory"][-1]["inner_selection"]):
                check_pair(a,b)
            check_pair(audits["ce"][-1]["final_fit"],audits["territory"][-1]["final_fit"])
        for arm in ARMS:
            mask = np.isfinite(probabilities[arm]).all(1)
            if not partial and not mask.all():
                raise ValueError("Missing shared-supervision predictions")
            p = probabilities[arm][mask]
            if np.any(p < 0) or not np.allclose(p.sum(1),1,atol=1e-5):
                raise ValueError("Invalid shared-supervision probabilities")
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
    lines = [f"Shared-label VCG study: {root}"]
    for seed in config["training_seeds"]:
        for arm in ARMS:
            lines.append(f"seed {seed} {arm}: {len(list((root/f'seed_{seed}').glob(f'fold_*_{arm}.json')))}/{config['outer_splits']} completed outer folds")
    if (root/"progress.json").exists():
        lines.append("Last saved progress (not a live process check): "+(root/"progress.json").read_text())
    lines.append(f"Validated final report: {(root/'REPORT.md').exists()}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/territory_vcg.yaml")
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
