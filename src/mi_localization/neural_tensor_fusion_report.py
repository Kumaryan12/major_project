"""Independently replay nested blend decisions and restored branch inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from . import neural_tensor_fusion as study, residual_context_report as bootstrap
from .benchmark import metric_bundle,save_confusion
from .contrastive import _progress
from .neural import _save_json

CONTRASTS=(("selected_blend","neural_only"),("equal_blend","neural_only"),
           ("selected_blend","tensor_only"),("selected_blend","equal_blend"),("equal_blend","tensor_only"))


def intervals(y,predicted,classes,draws,seed):
    results={f"{a}_minus_{b}":{} for a,b in CONTRASTS}
    for name,labels in (("six_class",np.arange(len(classes))),
                        ("mi_only",np.array([i for i,c in enumerate(classes) if c!="HC"]))):
        weights=bootstrap.patient_weights(y,labels,draws,seed);select=np.isin(y,labels)
        samples={a:bootstrap.weighted_f1(y,p.argmax(1),labels,weights) for a,p in predicted.items()}
        points={a:float(f1_score(y[select],p[select].argmax(1),labels=labels,average="macro",zero_division=0)) for a,p in predicted.items()}
        for a,b in CONTRASTS:
            results[f"{a}_minus_{b}"][name]={"delta_macro_f1":points[a]-points[b],
                "descriptive_patient_interval":np.quantile(samples[a]-samples[b],[.025,.975]).tolist()}
    return results


def summarize(config):
    data=study.load_inputs(config);scope=study.make_scope(config,data)
    root=Path(config["output_dir"])/"complete"
    if json.loads((root/"study_manifest.json").read_text())!=scope:
        raise ValueError("Fusion report source/config/runtime changed")
    predicted={a:np.full((len(data["ids"]),len(data["classes"])),np.nan) for a in study.PROCEDURES}
    decisions=[];neural_inner=tensor_inner=neural_outer=tensor_outer=selector_checks=0
    for fold in range(1,data["nc"]["outer_splits"]+1):
        rows,values,audit,decision=study.evaluate_fold(data,config,scope,root,fold,verify=True)
        for name,p in values.items():
            if np.isfinite(predicted[name][rows]).any():raise ValueError("Duplicate fusion query patients")
            predicted[name][rows]=p
        neural_inner+=audit["verified_neural_inner_contexts"];tensor_inner+=audit["verified_tensor_inner_contexts"]
        neural_outer+=len(audit["neural_members"]);tensor_outer+=1;selector_checks+=audit["tensor_bootstrap_caches_replayed"]
        decisions.append({"outer_fold":fold,"neural_weight":decision["selected_neural_weight"],"tensor_candidate":decision["tensor_candidate"]})
    metrics={};y=data["y"];classes=data["classes"];mi=y!=classes.index("HC")
    for name,p in predicted.items():
        study.nstudy.cache.check_p(p,len(y),len(classes))
        m=metric_bundle(y,p,classes)
        m["mi_only_macro_f1_six_way_predictions"]=float(f1_score(y[mi],p[mi].argmax(1),labels=[i for i,c in enumerate(classes) if c!="HC"],average="macro",zero_division=0))
        metrics[name]=m
        frame=pd.DataFrame({"patient_id":data["ids"],"truth":np.asarray(classes)[y],"prediction":np.asarray(classes)[p.argmax(1)],"fold":data["assigned"]})
        for i,c in enumerate(classes):frame[f"probability_{c}"]=p[:,i]
        frame.to_csv(root/f"patient_predictions_{name}.csv",index=False)
        save_confusion(y,p.argmax(1),classes,name,root)
    for name,reference in [("neural_only",data["neural_reference"]),("tensor_only",data["tensor_reference"])]:
        if not np.array_equal(predicted[name].argmax(1),reference.argmax(1)) or not np.allclose(predicted[name],reference,rtol=1e-5,atol=1e-6):
            raise ValueError("Frozen component reference changed in final report")
    contrasts=intervals(y,predicted,classes,config["bootstrap_draws"],config["bootstrap_seed"])
    summary={"complete":True,"study_fingerprint":scope["study_fingerprint"],"claim_scope":study.CLAIM,
             "n_patients":len(y),"classes":classes,"new_model_fits":0,"completed_outer_folds":data["nc"]["outer_splits"],
             "neural_inner_contexts_verified":neural_inner,"tensor_inner_contexts_verified":tensor_inner,
             "neural_outer_members_restored":neural_outer,"tensor_outer_contexts_restored":tensor_outer,
             "total_source_contexts_verified":neural_inner+tensor_inner+neural_outer+tensor_outer,
             "tensor_bootstrap_caches_replayed":selector_checks,"decisions":decisions,
             "selected_neural_weight_counts":dict(sorted(Counter(r["neural_weight"] for r in decisions).items())),
             "frozen_references_reproduced":True,"metrics":metrics,"contrasts":contrasts}
    lines=["# Patient-nested neural–tensor probability fusion","",study.CLAIM,"",
           f"All {len(y)} original patients. No new model fits. Verified {neural_inner} inner neural members and {tensor_inner} inner tensor/PFN contexts; restored {neural_outer} outer neural members and {tensor_outer} outer PFN contexts. Source artifacts read only. Both frozen component references reproduced.","",
           "| Procedure | Patient macro F1 | MI-only macro F1 | Accuracy |","|---|---:|---:|---:|"]
    for name,m in metrics.items():lines.append(f"| {name} | {m['macro_f1']:.3f} | {m['mi_only_macro_f1_six_way_predictions']:.3f} | {m['accuracy']:.1%} |")
    lines += ["","| Paired comparison | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |","|---|---:|---:|"]
    for name,result in contrasts.items():
        cells=[]
        for key in ["six_class","mi_only"]:
            r=result[key];lo,hi=r["descriptive_patient_interval"];cells.append(f"{r['delta_macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["","## Training-only choices","","| Outer fold | Neural weight | Tensor family | Feature budget |","|---:|---:|---|---:|"]
    for row in decisions:lines.append(f"| {row['outer_fold']} | {row['neural_weight']:.2f} | {row['tensor_candidate']['family']} | {row['tensor_candidate']['k']} |")
    lines += ["","## Per-class F1","","| Class | "+" | ".join(metrics)+" |","|---|"+"---:|"*len(metrics)]
    for c in classes:lines.append(f"| {c} | "+" | ".join(f"{m['classification_report'][c]['f1-score']:.3f}" for m in metrics.values())+" |")
    lines += ["","## Exact protocol and limits","",
              "The neural recipe is frozen ordered_context_aux, averaged equally across every declared optimization seed. The tensor recipe is pinned local raw-probability TabPFN-v2 on bootstrap-selected morphology/combined features. Its original family/budget choice is reconstructed from three inner scores, with original stable ties. The fusion then selects one scalar neural weight from [1, .75, .5, .25, 0] by mean inner patient macro F1; ties favor the first weight. This sequential inner selection reuses inner labels for component and blend selection, but never outer-query labels. Inner scores are optimized diagnostics, not final performance estimates.","",
              "Every neural seed and tensor candidate uses the exact same inner train/validation patient identities within each outer-training set. All original patients, six labels, recordings, cached R positions and frozen folds are unchanged. The globally pooled outer prediction CSVs are used only to verify frozen reference reproduction; they never train/select the combiner. Both base models and feature selectors excluded the outer test patients. Missing or incompatible source caches stop this run; no hidden refits or disabled integrity checks.","",
              "A weight of one reproduces the neural reference; zero reproduces the tensor reference. Equal blending is a fixed control. The four reported procedures share component predictions; selecting the best outer score as the winning method would be post-hoc selection, not confirmation. No class-specific routing, learned uncertainty gate, temperature fit, seed subset search or diagnosis-based filtering is performed.","",
              f"The independent report reconstructs all blend decisions and source selector/stopping audits and repeats restored branch inference. {selector_checks} tensor bootstrap caches were replayed. TabPFN restore/predict runs in isolated workers using the unchanged pinned checkpoint/settings; no TabPFN/LightGBM import is introduced into the feature-selection process. Private source/model/patient files remain ignored.","",
              f"Intervals use {config['bootstrap_draws']:,} paired class-stratified patient draws conditional on fixed predictions, without refits or adjustment for overlapping cross-validation training sets, repeated development or multiple comparisons. Ensembles produce a single F1, not a seed standard deviation. MI-only scores retain all six-way predictions and MI→HC errors.","",
              "The neural architecture/ensemble and the fusion hypothesis were developed on repeatedly inspected PTB patients. Correct nesting prevents the specific outer-label weight-fitting error; it does not turn these patients into an untouched confirmation cohort. Report modest/negative results fully. Independent label-compatible patients are needed for generalization or SOTA claims; no guaranteed 0.80–0.90 score is established.","",
              "Reproduce: `.venv/bin/python -m mi_localization.neural_tensor_fusion`, then `.venv/bin/python -m mi_localization.neural_tensor_fusion_report`. These commands require original local banks, probability checkpoints and fitted contexts; a Git clone lacks the ignored fitted artifacts.",""]
    (root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    _save_json(root/"summary.json",summary);_progress(root,state="complete_validated")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default="configs/neural_tensor_fusion_v1.yaml")
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    try:
        s=summarize(config);print(json.dumps({"complete":s["complete"],"source_contexts_verified":s["total_source_contexts_verified"]}))
    except Exception as error:
        _progress(Path(config["output_dir"])/"complete",state="failed",error_type=type(error).__name__,error=str(error));raise
