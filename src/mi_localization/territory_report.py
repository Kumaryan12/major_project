"""Validate paired shared-label supervision across all prespecified seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import territory_study as study
from .benchmark import aggregate_patients, inner_patient_splits, load_or_create_folds, metric_bundle
from .contrastive_report import paired_seed_bootstrap
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .signal_audit import file_hash


def summarize_study(config, draws=5000):
    study.validate_config(config)
    root, chosen, partial = study.directory(config)
    if partial or len(config["training_seeds"]) < 3:
        raise ValueError("Final shared-label report requires all folds and three seeds")
    _, y, groups, bank = study.load_bank(config)
    scope = json.loads((root/"study_manifest.json").read_text())
    fingerprint = scope.pop("study_fingerprint")
    if _hash(scope) != fingerprint or scope["config"] != config or scope["bank"] != bank or scope["folds_sha256"] != file_hash(config["folds_path"]) or scope["evaluated_folds"] != chosen or scope["descriptors"] != list(study.DESCRIPTORS) or scope["codebook"] != study.codebook(config).tolist() or scope["claim_scope"] != study.CLAIM_SCOPE or scope["runtime"]["code_sha256"] != file_hash(study.__file__) or scope["runtime"]["neural_code_sha256"] != file_hash(study.neural.__file__):
        raise ValueError("Shared-label report data/code/settings changed")
    frozen = load_or_create_folds(y, groups, config).set_index("patient_id").sort_index()
    beat_folds = np.array([frozen.fold[str(patient)] for patient in groups])
    predictions, metrics, rows = {}, {}, []
    fit_count = 0
    for seed in config["training_seeds"]:
        output = root/f"seed_{seed}"
        predictions[seed], metrics[seed], audits = {}, {}, {}
        for arm in study.ARMS:
            saved = json.loads((output/f"metrics_{arm}.json").read_text())
            if saved["study_fingerprint"] != fingerprint or saved["arm"] != arm or saved["training_seed"] != seed or saved["n_beats"] != bank["n_beats"] or saved["evaluated_folds"] != chosen or saved["claim_scope"] != study.CLAIM_SCOPE:
                raise ValueError("Incomplete/mismatched shared-label metrics")
            truth, pred = _read_predictions(output/f"patient_predictions_{arm}.csv", saved, frozen, config["classes"])
            predictions[seed][arm], metrics[seed][arm], audits[arm] = pred, saved["patient_metrics"], []
            pooled = np.full((len(y), len(config["classes"])), np.nan)
            for fold in chosen:
                train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
                audit = json.loads((output/f"fold_{fold:02d}_{arm}.json").read_text())
                if audit["study_fingerprint"] != fingerprint or audit["training_seed"] != seed or audit["arm"] != arm or audit["fold"] != fold or audit["patient_overlap"] or audit["train_beats"] != len(train) or audit["test_beats"] != len(test) or audit["train_patients"] != len(np.unique(groups[train])) or audit["test_patients"] != len(np.unique(groups[test])):
                    raise ValueError("Shared-label fold membership mismatch")
                splits = [(train[ti],train[vi]) for ti,vi in inner_patient_splits(y[train],groups[train],config["inner_splits"],config["split_seed"]+fold*1000)]
                if len(audit["inner_selection"]) != len(splits):
                    raise ValueError("Missing shared-label inner fits")
                for inner,(fit,(ti,vi)) in enumerate(zip(audit["inner_selection"],splits),1):
                    if fit["study_fingerprint"] != fingerprint or fit["inner_fold"] != inner:
                        raise ValueError("Shared-label inner provenance mismatch")
                    study.validate_fit(fit,config,seed+fold*1000+inner,arm,ti,vi,groups)
                epochs = int(np.floor(np.median([row["selected_epoch"] for row in audit["inner_selection"]])+.5))
                if audit["final_epochs"] != epochs:
                    raise ValueError("Shared-label outer epochs must use inner selection")
                study.validate_fit(audit["final_fit"],config,seed+fold,arm,train,None,groups,epochs)
                checkpoint = output/f"fold_{fold:02d}_{arm}.npz"
                if audit["prediction_checkpoint_sha256"] != file_hash(checkpoint):
                    raise ValueError("Shared-label saved predictions changed")
                with np.load(checkpoint) as saved_p:
                    if not np.array_equal(saved_p["indices"],test):
                        raise ValueError("Shared-label prediction membership changed")
                    p = saved_p["probabilities"]
                if p.shape != (len(test),len(config["classes"])) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-5):
                    raise ValueError("Invalid shared-label probabilities")
                _,fold_truth,fold_scores = aggregate_patients(p,y[test],groups[test])
                if audit["patient_metrics"] != metric_bundle(fold_truth,fold_scores,config["classes"]):
                    raise ValueError("Shared-label fold metrics disagree with predictions")
                pooled[test] = p
                audits[arm].append(audit)
                fit_count += config["inner_splits"]+1
            frame = pd.read_csv(output/f"patient_predictions_{arm}.csv").set_index("patient_id").sort_index()
            ids,pooled_truth,pooled_scores = aggregate_patients(pooled,y,groups)
            if saved["folds"] != audits[arm] or not np.array_equal(ids,frame.index.to_numpy()) or not np.array_equal(pooled_truth,truth) or not np.allclose(pooled_scores,frame[[f"probability_{label}" for label in config["classes"]]].to_numpy(),rtol=1e-7,atol=1e-12) or saved["patient_metrics"] != metric_bundle(pooled_truth,pooled_scores,config["classes"]):
                raise ValueError("Shared-label pooled patient results disagree with held-out beats")
        for first,second in zip(audits["ce"],audits["territory"]):
            for a,b in zip(first["inner_selection"],second["inner_selection"]):
                study.check_pair(a,b)
            study.check_pair(first["final_fit"],second["final_fit"])
        a,b = [metrics[seed][arm]["macro_f1"] for arm in study.ARMS]
        rows.append({"training_seed":seed,"ce_macro_f1":a,"territory_macro_f1":b,"territory_minus_ce":b-a})
    table = pd.DataFrame(rows)
    statistics = {arm:{"macro_f1_mean":float(table[f"{arm}_macro_f1"].mean()),"macro_f1_sample_std":float(table[f"{arm}_macro_f1"].std(ddof=1))} for arm in study.ARMS}
    uncertainty = paired_seed_bootstrap(truth,predictions,len(config["classes"]),draws,config["seed"],arms=study.ARMS)
    _save_json(root/"paired_bootstrap.json",dict(uncertainty,study_fingerprint=fingerprint))
    summary = {"complete":True,"study_fingerprint":fingerprint,"n_patients":len(frozen),"n_records":bank["n_records"],"n_beats":len(y),
        "training_seeds":config["training_seeds"],"arms":statistics,"per_seed":rows,"unique_fit_count":fit_count,
        "mean_territory_minus_ce":float(table.territory_minus_ce.mean()),"claim_scope":study.CLAIM_SCOPE}
    _save_json(root/"summary.json",summary)
    table.to_csv(root/"paired_comparison.csv",index=False)
    lines = ["# Shared-label auxiliary supervision: VCG-only", "",study.CLAIM_SCOPE,"",
        f"Complete: {len(frozen)} unchanged patients, {bank['n_records']} recordings, {len(y):,} original beats, six original classes, {config['outer_splits']} frozen patient folds, seeds {config['training_seeds']}; {fit_count} validated fits.","",
        "| Training seed | Matched CE | CE + shared descriptors | Δ macro F1 |","|---|---:|---:|---:|"]
    lines += [f"| {r['training_seed']} | {r['ce_macro_f1']:.3f} | {r['territory_macro_f1']:.3f} | {r['territory_minus_ce']:+.3f} |" for r in rows]
    for arm,value in statistics.items():
        lines += ["",f"{arm}: **{value['macro_f1_mean']:.3f} ± {value['macro_f1_sample_std']:.3f}** (mean ± sample SD across training seeds)."]
    lo,hi = uncertainty["mean_paired_macro_f1_difference_95_interval"]
    lines += ["",f"Mean paired difference **{summary['mean_territory_minus_ce']:+.3f}**, descriptive paired patient interval [{lo:+.3f}, {hi:+.3f}] using {draws:,} class-stratified patient draws shared across all arms and fixed seeds.","",
        "## Objective and exact control","",
        f"Both arms have the original XYZ residual encoder, unchanged six-class classifier and the same four-output auxiliary head (78,458 parameters at production widths). Control loss is CE + 0 × BCE; experiment is CE + {config['territory_weight']} × mean BCE across the four descriptors. Both auxiliary heads start with the same weights and the original model/dropout initialization stream is preserved. All shared-epoch sampled-row hashes and training-only normalizers are matched. The extra head's outputs are never used to replace the six-class prediction or construct a test-optimized hierarchy.","",
        "Auxiliary targets encode mentions in existing label names: AMI→anterior, ALMI→anterior+lateral, ASMI→anterior+septal, IMI→inferior, ILMI→inferior+lateral, HC→none. A zero means the component is absent from this source-label encoding, not independently confirmed anatomical absence. These are no new clinical annotations and do not repair label ambiguity. The six-class endpoint is unchanged.","",
        f"Ordinary CE, AdamW rate {config['learning_rate']}, decay {config['weight_decay']}, dropout {config['dropout']}, 8,192 class/patient-balanced beat draws per epoch, max {config['max_epochs']} epochs and patience {config['patience']}. There is no auxiliary-weight search, augmentation, geometry input, delineation, ECG input, pretraining or external data. Fixed inner patient partitions select the stopping epochs using six-class patient macro F1 only. Rounded median best epoch determines a fresh outer-training refit without test monitoring. All held-out beats contribute to mean patient probabilities.","",
        "## Mean class F1 across seeds","","| Class | Matched CE | Shared descriptors |","|---|---:|---:|"]
    for label in config["classes"]:
        values = [np.mean([metrics[seed][arm]["classification_report"][label]["f1-score"] for seed in config["training_seeds"]]) for arm in study.ARMS]
        lines.append(f"| {label} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ["","## Limits","",
        "This tests one fixed derived-label objective, not verified anatomical diagnosis, all possible multitask approaches, clinical validity, novelty or SOTA. Repeatedly inspected PTB folds remain development data. Bootstrap intervals condition on fixed fitted predictions, without model/seed refits or correction for correlated folds, repeated development or multiple experiments. Training seeds are not new patients. Offline full-record denoising remains noncausal; seeded MPS is not guaranteed bitwise deterministic. A 0.80–0.90 macro F1 is not promised. No label, patient, beat or R position was changed.","",
        "Patient predictions, waveform arrays and fitted weights stay local; aggregate reports/audits are public. Reproduce: `.venv/bin/python -m mi_localization.territory_study`, then `.venv/bin/python -m mi_localization.territory_report`. Read saved progress: `.venv/bin/python -m mi_localization.territory_study --status`.",""]
    path = root/"REPORT.md"
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/territory_vcg.yaml")
    parser.add_argument("--draws",type=int,default=5000)
    args = parser.parse_args()
    print(summarize_study(yaml.safe_load(Path(args.config).read_text()),args.draws))
