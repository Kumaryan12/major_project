"""Validate every raw/template/residual fit and all prespecified contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import multiview_study as study
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle, patient_balanced_weights
from .contrastive_report import paired_seed_bootstrap
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .signal_audit import file_hash


def summarize_study(config,draws=5000):
    study.validate_config(config)
    root,chosen,partial = study.directory(config)
    if partial or len(config["training_seeds"])<3:
        raise ValueError("Final multiview report requires all folds and three seeds")
    views,y,groups,bank = study.load_inputs(config)
    full = views["full_tensor"]
    scope = json.loads((root/"study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope)!=fingerprint or scope["config"]!=config or scope["bank"]!=bank or scope["folds_sha256"]!=file_hash(config["folds_path"]) or scope["evaluated_folds"]!=chosen or scope["claim_scope"]!=study.CLAIM_SCOPE or scope["runtime"]["code_sha256"]!=file_hash(study.__file__) or scope["runtime"]["neural_code_sha256"]!=file_hash(study.neural.__file__):
        raise ValueError("Multiview report source/settings/code changed")
    frozen = load_or_create_folds(y,groups,config).set_index("patient_id").sort_index()
    beat_folds = np.array([frozen.fold[str(patient)] for patient in groups])
    # Independently reconstruct every distinct training normalization rather
    # than merely trusting the cached values shared by all arms/seeds.
    moments = {}
    def expected_norm(train,arm):
        key = study.array_hash(train)
        if key not in moments:
            weights = patient_balanced_weights(y[train],groups[train])
            moments[key] = study.fit_normalization(full,y,groups,train,weights)
        return study.select_normalization(*moments[key],arm)
    predictions,metrics,rows,fit_count = {},{},[],0
    for seed in config["training_seeds"]:
        output = root/f"seed_{seed}"
        predictions[seed],metrics[seed],audits = {},{},{}
        for arm in study.ARMS:
            saved = json.loads((output/f"metrics_{arm}.json").read_text())
            if saved["study_fingerprint"]!=fingerprint or saved["training_seed"]!=seed or saved["arm"]!=arm or saved["n_beats"]!=len(y) or saved["evaluated_folds"]!=chosen or saved["claim_scope"]!=study.CLAIM_SCOPE:
                raise ValueError("Incomplete/mismatched multiview metrics")
            truth,pred = _read_predictions(output/f"patient_predictions_{arm}.csv",saved,frozen,config["classes"])
            predictions[seed][arm],metrics[seed][arm],audits[arm] = pred,saved["patient_metrics"],[]
            pooled = np.full((len(y),len(config["classes"])),np.nan)
            for fold in chosen:
                train,test = np.flatnonzero(beat_folds!=fold),np.flatnonzero(beat_folds==fold)
                audit = json.loads((output/f"fold_{fold:02d}_{arm}.json").read_text())
                if audit["study_fingerprint"]!=fingerprint or audit["training_seed"]!=seed or audit["arm"]!=arm or audit["fold"]!=fold or audit["patient_overlap"] or audit["train_beats"]!=len(train) or audit["test_beats"]!=len(test) or audit["train_patients"]!=len(np.unique(groups[train])) or audit["test_patients"]!=len(np.unique(groups[test])):
                    raise ValueError("Multiview outer membership mismatch")
                splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],groups[train],config["inner_splits"],config["split_seed"]+fold*1000)]
                if len(audit["inner_selection"])!=len(splits):
                    raise ValueError("Missing multiview inner fits")
                for inner,(fit,(ti,vi)) in enumerate(zip(audit["inner_selection"],splits),1):
                    if fit["study_fingerprint"]!=fingerprint or fit["inner_fold"]!=inner:
                        raise ValueError("Multiview inner provenance mismatch")
                    study.validate_fit(fit,config,seed+fold*1000+inner,arm,ti,vi,groups,expected_norm(ti,arm))
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in audit["inner_selection"]])+.5))
                if audit["final_epochs"]!=epochs:
                    raise ValueError("Multiview refit epochs must be inner-selected")
                study.validate_fit(audit["final_fit"],config,seed+fold,arm,train,None,groups,expected_norm(train,arm),epochs)
                path = output/f"fold_{fold:02d}_{arm}.npz"
                if audit["prediction_checkpoint_sha256"]!=file_hash(path):
                    raise ValueError("Multiview prediction checkpoint changed")
                with np.load(path) as values:
                    if not np.array_equal(values["indices"],test):
                        raise ValueError("Multiview prediction membership changed")
                    p = values["probabilities"]
                if p.shape!=(len(test),len(config["classes"])) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-5):
                    raise ValueError("Invalid multiview checkpoint probabilities")
                _,ft,fp = aggregate_patients(p,y[test],groups[test])
                if audit["patient_metrics"]!=metric_bundle(ft,fp,config["classes"]):
                    raise ValueError("Multiview fold metrics disagree with predictions")
                model_path = output/f"model_{fold:02d}_{arm}.pt"
                if audit["model_sha256"]!=file_hash(model_path):
                    raise ValueError("Multiview fitted model changed")
                model_state = torch.load(model_path,map_location="cpu",weights_only=True)
                if model_state["study_fingerprint"]!=fingerprint or model_state["arm"]!=arm or model_state["normalization"]!=audit["final_fit"]["normalization"] or any(not torch.isfinite(value).all() for value in model_state["state_dict"].values()):
                    raise ValueError("Invalid saved multiview model provenance/weights")
                check_model = study.initial_model(config,seed+fold,arm)
                check_model.load_state_dict(model_state["state_dict"],strict=True)
                pooled[test] = p
                audits[arm].append(audit)
                fit_count += config["inner_splits"]+1
            frame = pd.read_csv(output/f"patient_predictions_{arm}.csv").set_index("patient_id").sort_index()
            ids,pt,pp = aggregate_patients(pooled,y,groups)
            if saved["folds"]!=audits[arm] or not np.array_equal(ids,frame.index.to_numpy()) or not np.array_equal(pt,truth) or not np.allclose(pp,frame[[f"probability_{label}" for label in config["classes"]]].to_numpy(),rtol=1e-7,atol=1e-12) or saved["patient_metrics"]!=metric_bundle(pt,pp,config["classes"]):
                raise ValueError("Multiview pooled results disagree with held-out beats")
        for index,fold in enumerate(chosen):
            for reference in ["vcg","capacity_control"]:
                for arm in study.ARMS[1:]:
                    first,second = audits[reference][index],audits[arm][index]
                    for a,b in zip(first["inner_selection"],second["inner_selection"]):
                        study.check_pair(a,b)
                    study.check_pair(first["final_fit"],second["final_fit"])
        for arm in study.ARMS:
            rows.append({"training_seed":seed,"arm":arm,"patient_macro_f1":metrics[seed][arm]["macro_f1"],"patient_accuracy":metrics[seed][arm]["accuracy"]})
    table = pd.DataFrame(rows)
    statistics = {arm:{"macro_f1_mean":float(table.loc[table.arm==arm,"patient_macro_f1"].mean()),
        "macro_f1_sample_std":float(table.loc[table.arm==arm,"patient_macro_f1"].std(ddof=1)),
        "parameter_count":sum(p.numel() for p in study.initial_model(config,config["seed"],arm).parameters())} for arm in study.ARMS}
    contrasts,bootstrap = {},{}
    for first,second in study.CONTRASTS:
        key = f"{second}_minus_{first}"
        bootstrap[key] = paired_seed_bootstrap(truth,predictions,len(config["classes"]),draws,config["seed"],arms=(first,second))
        contrasts[key] = {"mean_macro_f1_difference":statistics[second]["macro_f1_mean"]-statistics[first]["macro_f1_mean"],
            "descriptive_95_interval":bootstrap[key]["mean_paired_macro_f1_difference_95_interval"]}
    summary = {"complete":True,"study_fingerprint":fingerprint,"n_patients":len(frozen),"n_records":bank["source"]["denoised"]["n_records"],
        "n_beats":len(y),"training_seeds":config["training_seeds"],"arms":statistics,"prespecified_contrasts":contrasts,"unique_fit_count":fit_count,"claim_scope":study.CLAIM_SCOPE}
    _save_json(root/"summary.json",summary)
    _save_json(root/"paired_bootstrap.json",{"study_fingerprint":fingerprint,"contrasts":bootstrap})
    table.to_csv(root/"paired_comparison.csv",index=False)
    lines = ["# Information-preserving measured-VCG views","",study.CLAIM_SCOPE,"",
        f"All {len(frozen)} original patients, {summary['n_records']} records and {len(y):,} beats; six unchanged classes, {config['outer_splits']} frozen patient folds, seeds {config['training_seeds']}. Validated {fit_count} fits.","",
        "| Model | Patient macro F1, mean ± SD | Parameters |","|---|---:|---:|"]
    for arm,value in statistics.items():
        lines.append(f"| {arm} | {value['macro_f1_mean']:.3f} ± {value['macro_f1_sample_std']:.3f} | {value['parameter_count']:,} |")
    lines += ["","SD is across training seeds, not a generalization confidence interval. Every prespecified model/seed is retained, without best-seed selection or probability ensembling.","",
        "## Prespecified contrasts","","| Difference | Mean Δ F1 | Descriptive paired patient interval |","|---|---:|---:|"]
    for name,value in contrasts.items():
        lo,hi = value["descriptive_95_interval"]
        lines.append(f"| {name} | {value['mean_macro_f1_difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["",f"{draws:,} class-stratified patient draws shared across models and fixed seeds. Conditional on fitted predictions; no refits, seed resampling or adjustment for correlated training folds, repeated development or multiple contrasts. These are descriptive intervals, not independent confirmation.","",
        "## What the views mean","",
        "The five physical views are raw R, existing denoised D, recording median M, beat residual D−M and filter residual R−D. Both residuals are retained as potentially meaningful variation, not certified noise. The [bank audit](../bank_audit/REPORT.md) verifies matching sources and residual reconstruction. Templates use only the same recording's unlabeled beats, including all held-out recording beats for its own context. This is offline within-exam processing and creates no new patients. No cohort/label-conditioned template dictionary is learned.","",
        "Models: vcg=D; capacity_control=[D,D,D,D,D]; raw_denoised=[R,D,D,D,D]; template_only=[M,M,M,M,M]; template_residual=[D,D,M,D−M,D]; full_tensor=[R,D,M,D−M,R−D]. Five-view inputs have 81,078 parameters versus the original 78,198. All five-view models have exact matched initial weights and parameter counts. Their initial stem uses the original kernel on view two, with other view weights zero; the classifier initialization and CPU random stream match the baseline. Except template_only, their initial active input is D.","",
        "Per-view/per-lead moments use weighted training beats only. Shared training-partition moments are cached across arms/seeds, with integrity guards; reporting independently recomputes them. The original D moments exactly follow the frozen baseline normalizer. Separate view standardization preserves invertible training-scale amplitude information but can change optimization of low-amplitude residuals; it is not physiological amplitude normalization. Shared normalizers, sampled-row histories, weights, stopping and complete predictions are checked.","",
        f"Same compact residual body, CE, AdamW rate {config['learning_rate']}, decay {config['weight_decay']}, dropout {config['dropout']}, batch {config['batch_size']}, {config['samples_per_epoch']:,} class/patient-balanced draws per epoch, maximum {config['max_epochs']} epochs and patience {config['patience']}. Fixed inner patient folds select epochs using six-class patient macro F1; rounded median determines a fresh outer-training refit without test monitoring. All held-out beats contribute to mean patient probabilities. More input channels add computation despite equal presentations/epochs.","",
        "Tensor views are assembled lazily and flattened into the input convolution. This tests waveform views and record context, not a learned Tucker contraction, phase delineation or a chronological beat-sequence encoder. Residuals are deterministic linear combinations of raw/D/M; any benefit is an inductive-bias/context result, not newly created sensor information. Duplicate-input control changes parameterization/optimization too, so comparison to the fresh original baseline remains essential.","",
        "No temporal realignment, spatial rotation, quality weighting, beat/patient removal, new R detection, resampling, augmentation, auxiliary labels, external pretraining or ECG input enters this stage. Original cached signals, labels and patient folds are unchanged.","",
        "## All training seeds","","| Seed | "+" | ".join(study.ARMS)+" |","|---|"+"---:|"*len(study.ARMS)]
    for seed in config["training_seeds"]:
        lines.append(f"| {seed} | "+" | ".join(f"{metrics[seed][arm]['macro_f1']:.3f}" for arm in study.ARMS)+" |")
    lines += ["","## Mean class F1 across seeds","","| Class | "+" | ".join(study.ARMS)+" |","|---|"+"---:|"*len(study.ARMS)]
    for label in config["classes"]:
        values = [np.mean([metrics[seed][arm]["classification_report"][label]["f1-score"] for seed in config["training_seeds"]]) for arm in study.ARMS]
        lines.append(f"| {label} | "+" | ".join(f"{value:.3f}" for value in values)+" |")
    lines += ["","## Limits","",
        "Repeatedly inspected PTB folds are development data. Denoised signals and templates are not independently known clean physiology; morphology similarity and reconstruction are not localization accuracy. Full-record denoising/templates are noncausal. MPS is seeded but not guaranteed bitwise deterministic. No 0.80–0.90 score, clinical validity, novelty or SOTA is promised. Any favorable result still needs a fixed new-cohort validation. No hard patients or unfavorable seeds are removed.","",
        "Reproduce: `.venv/bin/python -m mi_localization.multiview_bank`, `.venv/bin/python -m mi_localization.multiview_study`, then `.venv/bin/python -m mi_localization.multiview_report`. Saved progress: `.venv/bin/python -m mi_localization.multiview_study --status`. Raw/template arrays, weights, normalizer caches and individual predictions remain local; aggregate reports/audits are public.",""]
    path = root/"REPORT.md"
    path.write_text("\n".join(lines))
    return path


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/multiview_vcg.yaml")
    parser.add_argument("--draws",type=int,default=5000)
    args = parser.parse_args()
    print(summarize_study(yaml.safe_load(Path(args.config).read_text()),args.draws))
