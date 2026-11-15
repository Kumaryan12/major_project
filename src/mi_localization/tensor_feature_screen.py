"""Training-only feature design screen; no outer-test predictions or scores."""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import tensor_feature_bank as bank
from .benchmark import inner_patient_splits, load_or_create_folds, patient_balanced_weights
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

SCOPE = "Feature development inside outer-training patients only. Inner scores are selection diagnostics, not held-out project macro F1. No outer-test inference or final neural training."


def candidates(config):
    result = [{"family":"statistics","k":"all","C":float(c)} for c in config["linear_c"]]
    result += [{"family":family,"k":int(k),"C":float(c)} for family in bank.FAMILIES
        for k in config["feature_counts"] for c in config["linear_c"]]
    return result


def validate_config(config):
    if not config["feature_counts"] or any(type(k)!=int or k<1 for k in config["feature_counts"]) or len(set(config["feature_counts"]))!=len(config["feature_counts"]):
        raise ValueError("Require distinct positive feature budgets")
    if not config["linear_c"] or not np.isfinite(config["linear_c"]).all() or min(config["linear_c"])<=0 or len(set(config["linear_c"]))!=len(config["linear_c"]):
        raise ValueError("Require distinct positive probe regularization choices")
    if not 0<config["correlation_threshold"]<=1 or min(config["inner_splits"],config["outer_splits"])<2:
        raise ValueError("Invalid feature-screen protocol")


class TrainingSelector:
    """ANOVA ranking then greedy train-only Pearson redundancy pruning."""
    def fit(self,X,y,weights,k,threshold):
        if X.ndim!=2 or len(X)!=len(y) or not np.isfinite(X).all():
            raise ValueError("Invalid feature selection training rows")
        self.scale = StandardScaler().fit(X,sample_weight=weights)
        scaled = self.scale.transform(X)
        variable = np.flatnonzero(np.ptp(scaled,axis=0)>1e-12)
        if not len(variable):
            raise ValueError("No varying features in training partition")
        self.scores = np.zeros(X.shape[1])
        self.dropped_redundant = 0
        if k=="all":
            self.selected = variable
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",RuntimeWarning)
                scores,_ = f_classif(scaled[:,variable],y)
            scores = np.nan_to_num(scores,nan=0,posinf=np.finfo(float).max,neginf=0)
            self.scores[variable] = scores
            order = variable[np.argsort(-scores,kind="stable")]
            centered = scaled-scaled.mean(0)
            length = np.linalg.norm(centered,axis=0)
            unit = np.divide(centered,length,out=np.zeros_like(centered),where=length>1e-12)
            selected = []
            for column in order:
                if selected and np.max(np.abs(unit[:,selected].T@unit[:,column]))>=threshold:
                    self.dropped_redundant += 1
                    continue
                selected.append(int(column))
                if len(selected)>=k:
                    break
            self.selected = np.asarray(selected,dtype=np.int64)
        return self

    def transform(self,X):
        return self.scale.transform(X)[:,self.selected]

    def audit(self):
        return {"scaler_mean_sha256":array_hash(self.scale.mean_),"scaler_scale_sha256":array_hash(self.scale.scale_),
            "selected_indices":self.selected.tolist(),"n_selected":len(self.selected),"dropped_redundant_before_budget":self.dropped_redundant,
            "training_ranking_sha256":array_hash(self.scores)}


def load_features(config):
    root = bank.prepare_bank(config)
    manifest = json.loads((root/"manifest.json").read_text())
    with np.load(root/"patient_features.npz") as saved:
        ids,y = saved["patient_id"],saved["y"]
        features = {key:saved[key] for key in bank.FAMILIES}
    names = json.loads((root/"feature_names.json").read_text())
    if len(np.unique(ids))!=len(ids) or not np.array_equal(ids,np.sort(ids)) or any(x.shape!=(len(ids),len(names[key])) or not np.isfinite(x).all() for key,x in features.items()):
        raise ValueError("Invalid tensor feature patient table/schema")
    if not Path(config["folds_path"]).exists():
        raise ValueError("Require frozen patient folds")
    folds = load_or_create_folds(y,ids,config)
    if sorted(folds.fold.unique())!=list(range(1,config["outer_splits"]+1)):
        raise ValueError("Frozen feature-screen folds changed")
    return features,y,ids,names,folds,manifest


def fit_probe(X,y,ids,train,valid,candidate,config):
    if set(ids[train])&set(ids[valid]):
        raise ValueError("Feature-screen patient overlap")
    weights = patient_balanced_weights(y[train],ids[train])
    selector = TrainingSelector().fit(X[train],y[train],weights,candidate["k"],config["correlation_threshold"])
    model = LogisticRegression(C=candidate["C"],solver="lbfgs",max_iter=3000,tol=1e-6,random_state=config["seed"])
    with warnings.catch_warnings(),threadpool_limits(limits=1):
        warnings.simplefilter("error",ConvergenceWarning)
        model.fit(selector.transform(X[train]),y[train],sample_weight=weights)
    if not np.array_equal(model.classes_,np.arange(len(config["classes"]))):
        raise ValueError("Inner feature training partition is missing a class")
    p = model.predict_proba(selector.transform(X[valid]))
    audit = {"patient_overlap":0,"training_patients":len(train),"validation_patients":len(valid),
        "train_membership_sha256":_hash(ids[train].tolist()),"validation_membership_sha256":_hash(ids[valid].tolist()),
        "training_features_sha256":array_hash(X[train]),"training_labels_sha256":array_hash(y[train]),
        "training_weights_sha256":array_hash(weights),**selector.audit()}
    return p,audit


def run_screen(config):
    validate_config(config)
    features,y,ids,names,folds,manifest = load_features(config)
    root = Path(config["output_dir"])/"screen"
    root.mkdir(parents=True,exist_ok=True)
    scope = {"config":config,"bank_manifest_sha256":file_hash(Path(config["feature_bank_dir"])/"manifest.json"),
        "folds_sha256":file_hash(config["folds_path"]),"code_sha256":file_hash(__file__),"sklearn":sklearn.__version__,
        "numpy":np.__version__,"scope":SCOPE}
    fingerprint = _hash(scope)
    scope["study_fingerprint"] = fingerprint
    guard = root/"study_manifest.json"
    if guard.exists() and json.loads(guard.read_text())!=scope:
        raise ValueError("Feature screen source/settings changed; use a new output_dir")
    _save_json(guard,scope)
    lookup = folds.set_index("patient_id").fold.to_dict()
    assignment = np.array([lookup[str(patient)] for patient in ids])
    grid = candidates(config)
    selection_rows,all_rows = [],[]
    selected_features = Counter()
    for fold in range(1,config["outer_splits"]+1):
        train = np.flatnonzero(assignment!=fold)
        outer_ids = set(ids[assignment==fold])
        splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],ids[train],config["inner_splits"],config["split_seed"]+fold*1000)]
        scores = []
        for candidate_id,candidate in enumerate(grid):
            X = features[candidate["family"]]
            inner_scores = []
            for inner,(ti,vi) in enumerate(splits,1):
                if (set(ids[ti])|set(ids[vi]))&outer_ids:
                    raise ValueError("Outer-test patient entered feature design")
                path = root/f"inner_{fold:02d}_{candidate_id:02d}_{inner}.json"
                checkpoint = path.with_suffix(".npz")
                expected = {"study_fingerprint":fingerprint,"outer_fold":fold,"inner_fold":inner,"candidate_id":candidate_id,
                    "candidate":candidate,"train_membership_sha256":_hash(ids[ti].tolist()),"validation_membership_sha256":_hash(ids[vi].tolist()),
                    "training_features_sha256":array_hash(X[ti]),"training_labels_sha256":array_hash(y[ti])}
                if path.exists() and checkpoint.exists():
                    audit = json.loads(path.read_text())
                    if any(audit.get(key)!=value for key,value in expected.items()) or audit["checkpoint_sha256"]!=file_hash(checkpoint):
                        raise ValueError("Feature inner checkpoint changed")
                    weights = patient_balanced_weights(y[ti],ids[ti])
                    expected_selector = TrainingSelector().fit(X[ti],y[ti],weights,candidate["k"],config["correlation_threshold"])
                    expected_audit = dict(expected_selector.audit(),patient_overlap=0,training_patients=len(ti),validation_patients=len(vi),training_weights_sha256=array_hash(weights))
                    if any(audit.get(key)!=value for key,value in expected_audit.items()):
                        raise ValueError("Feature cached training selector changed")
                    with np.load(checkpoint) as saved:
                        if not np.array_equal(saved["indices"],vi):
                            raise ValueError("Feature validation membership changed")
                        p = saved["probabilities"]
                else:
                    p,audit = fit_probe(X,y,ids,ti,vi,candidate,config)
                    audit.update(expected)
                    temporary = checkpoint.with_suffix(".partial.npz")
                    np.savez_compressed(temporary,indices=vi,probabilities=p)
                    temporary.replace(checkpoint)
                    audit["checkpoint_sha256"] = file_hash(checkpoint)
                if p.shape!=(len(vi),len(config["classes"])) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6):
                    raise ValueError("Invalid inner feature probabilities")
                score = float(f1_score(y[vi],p.argmax(1),labels=np.arange(len(config["classes"])),average="macro",zero_division=0))
                if "validation_macro_f1" in audit and audit["validation_macro_f1"]!=score:
                    raise ValueError("Feature checkpoint score disagrees with predictions")
                audit["validation_macro_f1"] = score
                _save_json(path,audit)
                inner_scores.append(score)
                _progress(root,state="feature_probe",outer_fold=fold,candidate_id=candidate_id,inner_fold=inner,total_candidates=len(grid))
            mean = float(np.mean(inner_scores))
            scores.append(mean)
            all_rows.append({"outer_training_context":fold,"candidate_id":candidate_id,**candidate,"mean_inner_macro_f1":mean})
        chosen = max(range(len(grid)),key=lambda i:scores[i])
        candidate = grid[chosen]
        X = features[candidate["family"]]
        selector = TrainingSelector().fit(X[train],y[train],patient_balanced_weights(y[train],ids[train]),candidate["k"],config["correlation_threshold"])
        selected_names = [names[candidate["family"]][i] for i in selector.selected]
        selected_features.update(f"{candidate['family']}/{name}" for name in selected_names)
        package = root/f"features_outer_{fold:02d}.npz"
        temporary = package.with_suffix(".partial.npz")
        np.savez_compressed(temporary,study_fingerprint=fingerprint,train_patient_ids=ids[train],selected_indices=selector.selected,
            scaler_mean=selector.scale.mean_,scaler_scale=selector.scale.scale_,candidate=json.dumps(candidate))
        temporary.replace(package)
        selection = {"outer_training_context":fold,"candidate_id":chosen,**candidate,"mean_inner_macro_f1":scores[chosen],
            "train_patients":len(train),"outer_test_patients_used":0,"selection_rule":"maximum mean inner macro F1; exact ties configured order",
            "package_sha256":file_hash(package),"selector":selector.audit(),"feature_names":selected_names}
        _save_json(root/f"selection_outer_{fold:02d}.json",selection)
        selection_rows.append(selection)
        print(f"feature context {fold}/{config['outer_splits']}: {candidate}, selected {len(selected_names)}; inner-only {scores[chosen]:.4f}",flush=True)
    pd.DataFrame(all_rows).to_csv(root/"candidate_inner_scores.csv",index=False)
    _save_json(root/"feature_frequency.json",dict(selected_features.most_common()))
    summary = {"complete":True,"study_fingerprint":fingerprint,"scope":SCOPE,"n_patients":len(ids),
        "probe_fit_count":config["outer_splits"]*len(grid)*config["inner_splits"],"outer_test_predictions":0,
        "selection_counts":dict(Counter(row["family"] for row in selection_rows)),
        "selected_features_per_context":[row["selector"]["n_selected"] for row in selection_rows],
        "bank_feature_dimensions":manifest["feature_dimensions"],"next_step":"Freeze the per-fold selection procedure, then evaluate final classifiers on held-out outer patients. No outer-test score exists for this screen."}
    _save_json(root/"summary.json",summary)
    lines = ["# Tensor feature design screen","",SCOPE,"",f"Completed {summary['probe_fit_count']} compact linear probe fits. **Zero outer-test predictions**. Selected training-only packages for {config['outer_splits']} frozen outer contexts.","",
        "Each candidate fits class-balanced training-only scaling, ANOVA ranking and greedy absolute Pearson redundancy pruning. The unselected-statistics control removes only constant training columns. Both use the same class-balanced logistic probe. Candidate family, feature budget and C are selected by mean inner patient macro F1. No cohort-wide selector is applied before CV.","",
        "| Outer training context | Family | Budget | Actual features | C | Inner selection diagnostic |","|---|---|---:|---:|---:|---:|"]
    lines += [f"| {row['outer_training_context']} | {row['family']} | {row['k']} | {row['selector']['n_selected']} | {row['C']} | {row['mean_inner_macro_f1']:.3f} |" for row in selection_rows]
    lines += ["","These inner scores were optimized during selection and are optimistic. They must not be reported as project performance, improvement over 0.550, or evidence of 0.80–0.90. The folds overlap strongly in training patients; feature frequencies are descriptive reproducibility diagnostics, not independent stability probabilities or clinical biomarkers.","",
        "Features keep signed morphology, amplitude-aware multiscale moments and XYZ covariance plus normalized correlation. Covariance and correlation are complementary candidate summaries, not a reproduction of the sepsis paper. Fixed windows are not independently annotated cardiac phases. Multiscale reconstruction checks the complete tensor, while selected summaries remain lossy. A single probe cannot identify a universally best representation for every eventual classifier.","",
        "Next: inspect selected feature names/frequencies and source fidelity, freeze the selection procedure, then compare final classifiers within the same outer-fold protocol. Refit selectors inside each fold; do not pool these selected lists into a global mask and reuse it on these patients. Any later learned PCA/Tucker basis also belongs inside training folds. Repeatedly inspected PTB remains development data and needs new-cohort confirmation.",""]
    (root/"REPORT.md").write_text("\n".join(lines))
    _progress(root,state="complete",probe_fits=summary["probe_fit_count"],outer_test_predictions=0)
    return summary


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/tensor_features.yaml")
    args = parser.parse_args()
    print(json.dumps(run_screen(yaml.safe_load(Path(args.config).read_text())),indent=2))
