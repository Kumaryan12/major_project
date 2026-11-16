"""Read-only post-hoc diagnosis of completed ResNet and chronology studies.

No network training, probability blending, label-based routing or model selection.
Individual patient comparisons are written only to the ignored review directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, f1_score


CLAIM = ("Post-hoc descriptive comparison on repeatedly inspected PTB development patients. "
         "Different training procedures; no causal attribution, new fitted model, selected ensemble or independent confirmation.")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self, path):
        path = Path(path)
        self.hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def json(self, path):
        return json.loads(self.track(path).read_text())

    def csv(self, path):
        return pd.read_csv(self.track(path))

    def yaml(self, path):
        return yaml.safe_load(self.track(path).read_text())

    def manifest(self, path):
        saved = self.json(path)
        body = {k: v for k, v in saved.items() if k != "study_fingerprint"}
        if saved.get("study_fingerprint") != digest(body):
            raise ValueError("Invalid source study manifest")
        return saved


def validate_predictions(frame, frozen, classes, expected_f1):
    frame = frame.set_index("patient_id").sort_index()
    if (not frame.index.is_unique or not frame.index.equals(frozen.index)
            or not frame.truth.equals(frozen.label.rename("truth"))
            or not frame.fold.equals(frozen.fold)):
        raise ValueError("Prediction patients, labels or folds differ from frozen cohort")
    p = frame[[f"probability_{c}" for c in classes]].to_numpy(float)
    if (p.shape != (len(frozen), len(classes)) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid probabilities")
    pred = np.asarray(classes)[p.argmax(1)]
    if not np.array_equal(pred, frame.prediction):
        raise ValueError("Predicted labels disagree with probabilities")
    score = f1_score(frozen.label, pred, labels=classes, average="macro", zero_division=0)
    if not np.isclose(score, expected_f1, atol=1e-12, rtol=0):
        raise ValueError("Predictions disagree with verified source score")
    return p


def correctness_overlap(truth, first, second):
    a, b = np.asarray(first) == truth, np.asarray(second) == truth
    return {"both_correct": int((a & b).sum()), "reference_only_correct": int((a & ~b).sum()),
            "sequence_only_correct": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum()),
            "different_predictions": int(np.sum(np.asarray(first) != second))}


def regroup_probabilities(beats, probabilities, ids, equal_records=False):
    if (probabilities.ndim != 2 or len(beats) != len(probabilities) or not len(beats)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Incomplete beat prediction coverage")
    cols = list(range(probabilities.shape[1]))
    frame = pd.DataFrame(probabilities, columns=cols)
    frame["patient_id"], frame["record"] = beats.patient_id.to_numpy(), beats.record.to_numpy()
    if frame.groupby("record").patient_id.nunique().max() != 1:
        raise ValueError("Recording crosses patient boundaries")
    values = (frame.groupby(["patient_id", "record"])[cols].mean().groupby("patient_id").mean()
              if equal_records else frame.groupby("patient_id")[cols].mean())
    if set(values.index) != set(ids):
        raise ValueError("Aggregation patient coverage changed")
    return values.loc[ids].to_numpy()


def check_history(fit, max_epochs, patience, inner):
    history = fit["history"]
    if (not history or [r["epoch"] for r in history] != list(range(1, len(history)+1))
            or len(history) > max_epochs or not np.isfinite([r["training_loss"] for r in history]).all()
            or fit.get("patient_overlap", 0)):
        raise ValueError("Invalid fit history")
    if inner:
        best, epoch = -np.inf, 0
        for row in history:
            score = row["validation_patient_macro_f1"]
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid validation score")
            if score > best+1e-8:
                best, epoch = score, row["epoch"]
        if (fit["selected_epoch"] != epoch or abs(fit["best_validation_macro_f1"]-best) > 1e-12
                or (len(history) < max_epochs and len(history)-epoch != patience)):
            raise ValueError("Inner stopping does not match saved history")
    elif (fit["selected_epoch"] != len(history) or fit["best_validation_macro_f1"] is not None
          or any("validation_patient_macro_f1" in r for r in history)):
        raise ValueError("Outer history used validation or changed refit duration")


def summarize_histories(fits, max_epochs):
    return {"n_fits": len(fits),
            "median_selected_epoch": float(np.median([f["selected_epoch"] for f in fits])),
            "median_epochs_run": float(np.median([len(f["history"]) for f in fits])),
            "median_first_training_loss": float(np.median([f["history"][0]["training_loss"] for f in fits])),
            "median_last_training_loss": float(np.median([f["history"][-1]["training_loss"] for f in fits])),
            "median_training_loss_at_selected_epoch": float(np.median([f["history"][f["selected_epoch"]-1]["training_loss"] for f in fits])),
            "histories_reaching_epoch_cap": sum(len(f["history"]) == max_epochs for f in fits),
            "best_at_epoch_cap": sum(f["selected_epoch"] == max_epochs for f in fits),
            "median_best_validation_f1": (float(np.median([f["best_validation_macro_f1"] for f in fits]))
                                           if fits[0]["best_validation_macro_f1"] is not None else None)}


def _wave_norm(fit, sequence=False):
    norm = fit["normalization"]["wave"] if sequence else fit["normalization"]
    return {key: norm[key] for key in ("wave_mean_", "wave_scale_")}


def analyze(config):
    source = Inputs()
    repeat = source.yaml(config["repeat_config"])
    base = source.yaml(repeat["base_config"])
    grid = source.yaml(config["length_config"])
    seq_config = source.yaml(grid["base_config"])
    classes, seeds = base["classes"], repeat["training_seeds"]
    if classes != seq_config["classes"] or seeds != seq_config["training_seeds"]:
        raise ValueError("Source classes or declared seeds differ")
    if base["folds_path"] != seq_config["folds_path"] or base["bank_dir"] != seq_config["bank_dir"]:
        raise ValueError("Sources use different banks or frozen folds")
    frozen = source.csv(base["folds_path"]).set_index("patient_id").sort_index()
    if not frozen.index.is_unique or set(frozen.label) != set(classes):
        raise ValueError("Invalid frozen cohort")
    fold_hash = source.hashes[base["folds_path"]]
    root = Path(grid["output_dir"])/"complete"
    seq_scope = source.manifest(root/"study_manifest.json")
    complete = source.json(root/"summary.json")
    source.track(root/"REPORT.md")
    partitions = base["outer_splits"]*len(seeds)
    if (not complete["complete"] or complete["study_fingerprint"] != seq_scope["study_fingerprint"]
            or complete["outer_models_restored"] != partitions*len(seq_config["arms"])
            or not complete["all_query_beats_covered"] or seq_scope["config"] != grid
            or seq_scope["base_protocol"]["config"] != seq_config
            or seq_scope["base_protocol"]["folds_sha256"] != fold_hash):
        raise ValueError("Require matching, complete, verified chronology source")
    rep_summary = source.json(Path(repeat["output_dir"])/"summary.json")
    if not rep_summary["complete"] or rep_summary["training_seeds"] != seeds:
        raise ValueError("Incomplete reference repeats")
    beats = source.csv(base["beats_path"])
    beats = beats.loc[beats.label.isin(classes)].reset_index(drop=True)
    provenance = seq_scope["base_protocol"]["source"]["source"]
    if source.hashes[base["beats_path"]] != provenance["beats_sha256"]:
        raise ValueError("Original beat metadata changed")
    for filename, column in [("groups.npy", "patient_id"), ("y.npy", "label")]:
        path = source.track(Path(base["bank_dir"])/filename)
        if source.hashes[str(path)] != provenance["files"][filename]:
            raise ValueError("Source waveform row metadata changed")
        actual = np.load(path, allow_pickle=False)
        expected = beats[column].astype(str).to_numpy()
        if not np.array_equal(actual, expected):
            raise ValueError("Beat rows do not match source bank")
    if (not set(beats.patient_id) == set(frozen.index)
            or not np.array_equal(beats.label, beats.patient_id.map(frozen.label))):
        raise ValueError("Beat metadata does not cover frozen patients")
    predictions, reference_audits, history_rows = {}, {}, []
    histories = {a: {"inner_selected_length": [], "outer": []} for a in ["resnet", *seq_config["arms"]]}
    aggregation_rows, private_rows, normalizations = [], [], 0
    by_length = {a: {str(k): [] for k in grid["lengths"]} for a in seq_config["arms"]}
    for seed in seeds:
        ref = Path(repeat["reference_dir"]) if seed == repeat["reference_seed"] else Path(repeat["output_dir"])/f"seed_{seed}"/"complete"
        manifest = source.manifest(ref/"study_manifest.json")
        saved = source.json(ref/"metrics_resnet.json")
        if (manifest["folds_sha256"] != fold_hash or manifest["bank_fingerprint"] != provenance["fingerprint"]
                or saved["study_fingerprint"] != manifest["study_fingerprint"] or not saved["complete"]
                or saved["classes"] != classes or saved["n_patients"] != len(frozen)
                or saved["evaluated_folds"] != list(range(1, base["outer_splits"]+1))):
            raise ValueError("Reference source/cohort is incomplete or mismatched")
        p = validate_predictions(source.csv(ref/"patient_predictions_resnet.csv"), frozen, classes, saved["patient_metrics"]["macro_f1"])
        predictions["resnet", seed] = p
        raw = np.full((len(beats), len(classes)), np.nan)
        for fold_audit in saved["folds"]:
            fold = fold_audit["fold"]
            if fold_audit["patient_overlap"] or fold_audit["study_fingerprint"] != manifest["study_fingerprint"]:
                raise ValueError("Reference fold audit changed")
            audit_path = ref/f"fold_{fold:02d}_resnet.json"
            if source.json(audit_path) != fold_audit:
                raise ValueError("Reference metric/fold audit disagreement")
            with np.load(source.track(ref/f"fold_{fold:02d}_resnet.npz"), allow_pickle=False) as z:
                indices = np.flatnonzero(beats.patient_id.map(frozen.fold).to_numpy() == fold)
                if not np.array_equal(z["indices"], indices):
                    raise ValueError("Raw reference predictions do not follow frozen folds")
                raw[indices] = z["probabilities"]
            final = fold_audit["final_fit"]
            check_history(final, base["max_epochs"], base["patience"], False)
            if final["seed"] != seed+fold or final["selected_epoch"] != fold_audit["final_epochs"]:
                raise ValueError("Reference refit seed/duration changed")
            reference_audits[seed, fold, 0] = final
            histories["resnet"]["outer"].append(final)
            for i, inner in enumerate(fold_audit["inner_selection"], 1):
                check_history(inner, base["max_epochs"], base["patience"], True)
                if inner["inner_fold"] != i or inner["seed"] != seed+fold*1000+i:
                    raise ValueError("Reference inner seed/fold changed")
                reference_audits[seed, fold, i] = inner
                histories["resnet"]["inner_selected_length"].append(inner)
            if fold_audit["final_epochs"] != int(np.floor(np.median([f["selected_epoch"] for f in fold_audit["inner_selection"]])+0.5)):
                raise ValueError("Reference duration not chosen inside training folds")
        reconstructed = regroup_probabilities(beats, raw, frozen.index)
        if not np.allclose(reconstructed, p, atol=1e-12, rtol=0):
            raise ValueError("Raw reference predictions do not reproduce saved patient probabilities")
        equal_record = regroup_probabilities(beats, raw, frozen.index, equal_records=True)
        for mode, values in [("original_beat_weighted", p), ("equal_record_posthoc", equal_record)]:
            aggregation_rows.append({"training_seed": seed, "aggregation": mode,
                                     "macro_f1": float(f1_score(frozen.label, np.asarray(classes)[values.argmax(1)], labels=classes, average="macro", zero_division=0)),
                                     "changed_patient_predictions": int(np.sum(values.argmax(1) != p.argmax(1)))})
        for arm in seq_config["arms"]:
            predictions[arm, seed] = validate_predictions(source.csv(root/f"patient_predictions_{arm}_s{seed}.csv"), frozen, classes, complete["metrics"][arm][str(seed)]["macro_f1"])
        for fold in range(1, base["outer_splits"]+1):
            decision = source.json(root/f"selection_{fold:02d}_s{seed}.json")
            length = decision["selected_length"]
            if (decision["study_fingerprint"] != seq_scope["study_fingerprint"] or length not in grid["lengths"]
                    or decision["outer_fold"] != fold or decision["training_seed"] != seed):
                raise ValueError("Sequence length decision source changed")
            for arm in seq_config["arms"]:
                for k in grid["lengths"]:
                    for i in range(1, base["inner_splits"]+1):
                        audit = source.json(root/f"inner_{fold:02d}_{arm}_s{seed}_l{k}_i{i}.json")
                        if (audit["study_fingerprint"] != seq_scope["study_fingerprint"] or audit["patient_overlap"]
                                or audit["fit"]["seed"] != seed+fold*1000+i):
                            raise ValueError("Sequence inner source/fold/seed changed")
                        fit = audit["fit"]
                        check_history(fit, seq_config["max_epochs"], seq_config["patience"], True)
                        if _wave_norm(fit, True) != _wave_norm(reference_audits[seed, fold, i]):
                            raise ValueError("Reference and sequence waveform scaling differs")
                        normalizations += 1
                        by_length[arm][str(k)].append(fit)
                        if k == length:
                            histories[arm]["inner_selected_length"].append(fit)
                audit_path = root/f"outer_{fold:02d}_{arm}_s{seed}_l{length}.json"
                audit = source.json(audit_path)
                if (audit["study_fingerprint"] != seq_scope["study_fingerprint"] or audit["patient_overlap"]
                        or audit["selection_sha256"] != source.hashes[str(root/f"selection_{fold:02d}_s{seed}.json")]
                        or audit["fit"]["seed"] != seed+fold):
                    raise ValueError("Sequence outer source/selection/seed changed")
                with np.load(source.track(audit_path.with_suffix(".npz")), allow_pickle=False) as z:
                    query = frozen.index[frozen.fold == fold]
                    if (source.hashes[str(audit_path.with_suffix(".npz"))] != audit["probabilities_sha256"]
                            or not np.array_equal(z["patient_ids"], query)
                            or not np.allclose(z["probabilities"], predictions[arm, seed][frozen.fold.to_numpy() == fold], atol=1e-12, rtol=0)):
                        raise ValueError("Sequence outer checkpoint does not reproduce saved predictions")
                fit = audit["fit"]
                check_history(fit, seq_config["max_epochs"], seq_config["patience"], False)
                if _wave_norm(fit, True) != _wave_norm(reference_audits[seed, fold, 0]):
                    raise ValueError("Outer waveform scaling differs")
                normalizations += 1
                histories[arm]["outer"].append(fit)
        for index, patient in enumerate(frozen.index):
            private_rows.append({"patient_id": patient, "training_seed": seed, "truth": frozen.label.iloc[index],
                                 "fold": int(frozen.fold.iloc[index]),
                                 **{a: classes[predictions[a, seed][index].argmax()] for a in histories}})
    truth = frozen.label.to_numpy()
    class_rows, metric_rows, overlap_rows, confusions = [], [], [], {}
    correctness = {}
    for arm in histories:
        values = []
        for seed in seeds:
            p = predictions[arm, seed];pred = np.asarray(classes)[p.argmax(1)]
            values.append(pred == truth)
            metric_rows.append({"arm": arm, "training_seed": seed, "macro_f1": float(f1_score(truth, pred, labels=classes, average="macro", zero_division=0)),
                                "accuracy": float(np.mean(pred == truth)), "mean_max_probability": float(p.max(1).mean()),
                                "mean_entropy_nats": float(-(p*np.log(np.clip(p, 1e-15, 1))).sum(1).mean())})
            if arm != "resnet":
                ref = np.asarray(classes)[predictions["resnet", seed].argmax(1)]
                overlap_rows.append({"arm": arm, "training_seed": seed, **correctness_overlap(truth, ref, pred)})
        correctness[arm] = np.stack(values)
        confusions[arm] = sum(confusion_matrix(truth, np.asarray(classes)[predictions[arm, seed].argmax(1)], labels=classes) for seed in seeds)
        for i, c in enumerate(classes):
            scores = [f1_score(truth, np.asarray(classes)[predictions[arm, seed].argmax(1)], labels=classes, average=None, zero_division=0)[i] for seed in seeds]
            class_rows.append({"arm": arm, "class": c, "patients": int(np.sum(truth == c)), "mean_seed_f1": float(np.mean(scores))})
    robust = {}
    for arm in seq_config["arms"]:
        a, b = correctness["resnet"], correctness[arm]
        robust[arm] = {"both_wrong_all_seeds": int((~a.any(0) & ~b.any(0)).sum()),
                       "reference_correct_all_sequence_wrong_all": int((a.all(0) & ~b.any(0)).sum()),
                       "sequence_correct_all_reference_wrong_all": int((b.all(0) & ~a.any(0)).sum()),
                       "reference_wrong_all_seeds": int((~a.any(0)).sum()),
                       "sequence_wrong_all_seeds": int((~b.any(0)).sum())}
    training = {a: {phase: summarize_histories(fits, base["max_epochs"]) for phase, fits in phases.items()} for a, phases in histories.items()}
    training_by_length = {a: {k: summarize_histories(fits, base["max_epochs"]) for k, fits in lengths.items()} for a, lengths in by_length.items()}
    for arm, phases in histories.items():
        for phase, fits in phases.items():
            for fit in fits:
                for row in fit["history"]:
                    history_rows.append({"arm": arm, "phase": phase, "epoch": row["epoch"], "training_loss": row["training_loss"],
                                         "validation_patient_macro_f1": row.get("validation_patient_macro_f1", np.nan)})
    epoch_frame = pd.DataFrame(history_rows).groupby(["arm", "phase", "epoch"]).agg(
        n_fits=("training_loss", "size"), median_training_loss=("training_loss", "median"),
        median_validation_macro_f1=("validation_patient_macro_f1", "median")).reset_index()
    summary = {"complete": True, "claim_scope": CLAIM, "n_patients": len(frozen), "classes": classes,
               "training_seeds": seeds, "new_training_fits": 0, "source_prediction_sets_checked": len(predictions),
               "waveform_normalization_matches": normalizations, "metrics": metric_rows,
               "class_f1": class_rows, "paired_correctness": overlap_rows, "robust_errors": robust,
               "training": training, "training_by_length": training_by_length,
               "reference_aggregation_sensitivity": aggregation_rows,
               "confusion_counts_sum_over_seeds": {a: m.tolist() for a, m in confusions.items()}}
    output = Path(config["output_dir"])
    manifest = {"config": config, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "input_sha256": source.hashes}
    output.mkdir(parents=True, exist_ok=True)
    guard = output/"analysis_manifest.json"
    if guard.exists() and json.loads(guard.read_text()) != manifest:
        raise ValueError("Diagnostic inputs/code changed; use a new output namespace")
    (output/"review").mkdir(exist_ok=True)
    pd.DataFrame(private_rows).to_csv(output/"review"/"patient_comparison.csv", index=False)
    for name, rows in [("metrics", metric_rows), ("class_f1", class_rows), ("paired_correctness", overlap_rows), ("aggregation_sensitivity", aggregation_rows)]:
        pd.DataFrame(rows).to_csv(output/f"{name}.csv", index=False)
    epoch_frame.to_csv(output/"epoch_trajectories.csv", index=False)
    guard.write_text(json.dumps(manifest, indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_report(output, summary)
    plot_diagnostics(output, summary, epoch_frame)
    return summary


def write_report(root, summary):
    metrics = pd.DataFrame(summary["metrics"])
    lines = ["# ResNet versus consecutive-beat diagnostics", "", summary["claim_scope"], "",
             f"All {summary['n_patients']} original patients, all three declared seeds. {summary['source_prediction_sets_checked']} saved patient prediction sets checked against frozen folds and source scores. No training fits.", "",
             f"Raw ResNet beat checkpoints reproduce saved patient predictions. Sequence outer probability checkpoints reproduce saved patient predictions. {summary['waveform_normalization_matches']} corresponding waveform-normalization comparisons match exactly. Existing model-restoration verification is inherited from the completed study; this diagnostic does not restore neural weights again.", "",
             "## Scores and class effects", "", "| Procedure | Mean macro F1 ± seed SD |", "|---|---:|"]
    for arm, values in metrics.groupby("arm", sort=False):
        lines.append(f"| {arm} | {values.macro_f1.mean():.3f} ± {values.macro_f1.std():.3f} |")
    classes = pd.DataFrame(summary["class_f1"]).pivot(index="class", columns="arm", values="mean_seed_f1").loc[summary["classes"]]
    lines += ["", "| Class | ResNet | Ordered+RR | Shuffled+RR |", "|---|---:|---:|---:|"]
    for c, row in classes.iterrows():
        lines.append(f"| {c} | {row['resnet']:.3f} | {row['ordered_rr']:.3f} | {row['shuffled_rr']:.3f} |")
    lines += ["", "## Paired errors", "", "| Sequence arm | Seed | Both correct | ResNet only correct | Sequence only correct | Both wrong |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary["paired_correctness"]:
        lines.append(f"| {row['arm']} | {row['training_seed']} | {row['both_correct']} | {row['reference_only_correct']} | {row['sequence_only_correct']} | {row['both_wrong']} |")
    lines += ["", "Each row contains the same patients; rows/seeds are not independent cohorts. Both-wrong cases can have different incorrect labels. Correctness complementarity is descriptive and is not a realizable oracle routing or ensemble score.", "",
              "| Arm | Wrong in every seed: ResNet / sequence / both | ResNet always correct, sequence always wrong | Sequence always correct, ResNet always wrong |", "|---|---:|---:|---:|"]
    for arm, row in summary["robust_errors"].items():
        lines.append(f"| {arm} | {row['reference_wrong_all_seeds']} / {row['sequence_wrong_all_seeds']} / {row['both_wrong_all_seeds']} | {row['reference_correct_all_sequence_wrong_all']} | {row['sequence_correct_all_reference_wrong_all']} |")
    lines += ["", "## Aggregation sensitivity without refitting", "", "| Seed | Original beat-weighted ResNet F1 | Equal-record ResNet F1 | Changed predictions |", "|---|---:|---:|---:|"]
    agg = pd.DataFrame(summary["reference_aggregation_sensitivity"])
    for seed in summary["training_seeds"]:
        rows = agg[agg.training_seed == seed].set_index("aggregation")
        lines.append(f"| {seed} | {rows.loc['original_beat_weighted','macro_f1']:.3f} | {rows.loc['equal_record_posthoc','macro_f1']:.3f} | {int(rows.loc['equal_record_posthoc','changed_patient_predictions'])} |")
    means = agg.groupby("aggregation").macro_f1.mean()
    lines += ["", f"Three-seed means: original **{means['original_beat_weighted']:.6f}**, equal-record **{means['equal_record_posthoc']:.6f}**. This is a post-hoc inference sensitivity, not a newly nested training/evaluation procedure: original early stopping used beat-weighted predictions. Do not promote an aggregation winner using these outer outcomes. Equal-record weighting alone does not explain the average sequence-versus-ResNet gap.", "",
              "## Training behavior", "", "| Arm | Selected-length inner fits | Median best epoch | Best at epoch 25 | Median last inner loss | Median outer epochs | Median last outer loss |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, row in summary["training"].items():
        inner, outer = row["inner_selected_length"], row["outer"]
        lines.append(f"| {arm} | {inner['n_fits']} | {inner['median_selected_epoch']:.1f} | {inner['best_at_epoch_cap']} | {inner['median_last_training_loss']:.3f} | {outer['median_selected_epoch']:.1f} | {outer['median_last_training_loss']:.3f} |")
    lines += ["", "Sequence inner rows here use only the length selected inside each outer-training partition (90 fits per arm), matching the reference's 90 inner fits. The JSON also separates all candidate histories by length. Training losses have different beat/window objectives and stochastic sampling; they are not a matched numerical measure of representation quality. Epoch curves have changing denominators after early stopping; the CSV includes n_fits. Best validation scores are optimized inner diagnostics, not project performance.", "",
              "## Interpretation and next experiment", "",
              "The paired order-control result remains the decisive chronology evidence: ordered+RR 0.524 versus shuffled+RR 0.533, no demonstrated ordering gain. Cross-procedure class/error differences do not isolate chronology's causal effect. Identical source beats, patient folds and waveform scaling rule out those specific mismatches; sampling, window labels, supervision, aggregation, encoder/head initialization and optimization still differ.", "",
              "The sequence model compresses each 64-dimensional beat embedding through a 32-dimensional recurrent state (or projection) and classifies one window. The reference directly supervises every beat embedding. Higher sequence training loss and later selected epochs are compatible with a harder optimization problem, but do not establish destructive compression or justify a larger epoch budget by themselves. Check whether best epochs cluster at the cap before blaming truncation.", "",
              "A focused follow-up hypothesis is to retain a direct morphology path and beat-level supervision, then add a small gated context residual. Evaluate a fresh matched beat-only reference, context without beat auxiliary loss, and context with the fixed auxiliary loss; retain a matched shuffled-context control. All folds, seeds, exposure, inner stopping and length selection must be specified before fitting. A pretrained/frozen encoder must come only from that exact training partition; an outer-trained checkpoint cannot be reused across its inner validation patients. This follow-up is proposed, not run or scored here.", "",
              "Patient-specific review stays in ignored review/patient_comparison.csv. Do not remove difficult patients, relabel without independent adjudication, tune on outer errors, or report the diagnostic as an untouched test. No prediction averaging or outer-label-selected routing was performed.", ""]
    (root/"REPORT.md").write_text("\n".join(lines))


def plot_diagnostics(root, summary, epochs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = summary["classes"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, arm in zip(axes, ["resnet", "ordered_rr", "shuffled_rr"]):
        cm = np.asarray(summary["confusion_counts_sum_over_seeds"][arm])
        norm = cm/cm.sum(1, keepdims=True)
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Greys")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{norm[i,j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i,j] > .55 else "black")
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
               yticklabels=classes, xlabel="Predicted class", ylabel="True class", title=arm)
        ax.tick_params(axis="x", labelrotation=45)
    fig.colorbar(im, ax=axes, shrink=.75, label="Fraction within true class")
    fig.suptitle("Out-of-fold confusion across three seeds (repeated patients)")
    fig.savefig(root/"class_confusions.png", dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm, style in [("resnet", "-"), ("ordered_rr", "--"), ("shuffled_rr", ":")]:
        rows = epochs[(epochs.arm == arm) & (epochs.phase == "inner_selected_length")]
        axes[0].plot(rows.epoch, rows.median_training_loss, style, label=arm)
        axes[1].plot(rows.epoch, rows.median_validation_macro_f1, style, label=arm)
    axes[0].set(ylabel="Median training cross entropy", xlabel="Epoch", title="Different beat/window objectives")
    axes[1].set(ylabel="Median inner patient macro F1", xlabel="Epoch", title="Inner diagnostics, not final performance")
    for ax in axes: ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Selected-length inner histories; surviving fit counts change with epoch")
    fig.savefig(root/"training_histories.png", dpi=170);plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chronological_diagnostics_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    summary = analyze(config)
    print(json.dumps({"complete": summary["complete"], "new_training_fits": 0, "output_dir": config["output_dir"]}))
