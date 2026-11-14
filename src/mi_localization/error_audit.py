"""Post-hoc persistent-error review: descriptive, never automatic relabeling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from .benchmark import load_or_create_folds
from .labels import clinical_fields, label_from_comments
from .neural import _hash, _save_json
from .neural_report import _read_predictions
from .regularization import load_bank
from .signal_audit import build_raw_bank_and_audit, file_hash, signal_metrics


def error_table(frames, frozen, classes, seeds):
    """All identities/truth/folds must already align; no probability ensembling."""
    table = frozen.copy()
    for arm in ["baseline", "selected"]:
        wrong, predictions, probabilities = [], [], []
        for seed in seeds:
            frame = frames[seed][arm]
            if not frame.index.equals(frozen.index) or not frame.truth.equals(frozen.label.rename("truth")) or not frame.fold.equals(frozen.fold):
                raise ValueError("Error-audit patient labels/folds do not align")
            pred = frame.prediction.to_numpy()
            values = frame[[f"probability_{label}" for label in classes]].to_numpy()
            if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(1), 1, atol=1e-5) or not np.array_equal(pred, np.array(classes)[values.argmax(1)]):
                raise ValueError("Invalid error-audit probabilities/predictions")
            wrong.append(pred != frozen.label.to_numpy())
            predictions.append(pred)
            probabilities.append(values)
            table[f"{arm}_prediction_seed_{seed}"] = pred
        table[f"{arm}_error_count"] = np.sum(wrong, axis=0)
        table[f"{arm}_persistent_error"] = table[f"{arm}_error_count"] == len(seeds)
        table[f"{arm}_prediction_agreement"] = [len(set(predictions[index][patient] for index in range(len(seeds)))) == 1 for patient in range(len(frozen))]
        truth_index = np.array([classes.index(label) for label in frozen.label])
        table[f"{arm}_mean_true_probability"] = np.mean([values[np.arange(len(frozen)), truth_index] for values in probabilities], axis=0)
        table[f"{arm}_mean_top_probability"] = np.mean([values.max(1) for values in probabilities], axis=0)
        table[f"{arm}_mean_entropy"] = np.mean([-(values * np.log(np.maximum(values, 1e-15))).sum(1) for values in probabilities], axis=0)
    return table


def _known_location(value):
    return str(value).strip().lower() not in {"", "no", "none", "unknown", "n/a", "nan", "not available"}


def source_records(beats, paper, record_audit):
    if not record_audit.record.is_unique or set(record_audit.record) != set(beats.record.unique()):
        raise ValueError("Recording diagnostics do not match the cohort")
    diagnostics = record_audit.set_index("record")
    rows, sources = [], []
    for record, part in beats.groupby("record", sort=True):
        if part.patient_id.nunique() != 1 or part.label.nunique() != 1:
            raise ValueError("Inconsistent patient/label within a cached recording")
        reference = diagnostics.loc[record]
        if reference.patient_id != part.patient_id.iloc[0] or reference.label != part.label.iloc[0] or reference.n_beats != len(part):
            raise ValueError("Recording audit membership/count changed")
        root = Path(paper["data"]["raw_dir"])
        header_path, waveform_path = root / f"{record}.hea", root / f"{record}.xyz"
        header = header_path.read_text(encoding="latin1")
        fields = clinical_fields(header)
        header_hash, waveform_hash = file_hash(header_path), file_hash(waveform_path)
        sources.append({"record": record, "header_sha256": header_hash, "waveform_sha256": waveform_hash})
        rows.append({"record": record, "patient_id": part.patient_id.iloc[0], "label": part.label.iloc[0],
            "header_label": label_from_comments(header), "header_label_disagreement": label_from_comments(header) != part.label.iloc[0],
            "acute_location_text": fields.get("acute infarction (localization)", ""),
            "former_location_text": fields.get("former infarction (localization)", ""),
            "former_location_reported": _known_location(fields.get("former infarction (localization)", "")),
            "n_beats": len(part), "duration_seconds": float(reference.duration_seconds),
            **{name: int(reference[name]) for name in ["n_intervals", "cached_intervals_below_300ms", "cached_intervals_above_2000ms", "next_cached_peak_inside_crop"]},
            "waveform_sha256": waveform_hash})
    return pd.DataFrame(rows), sources


def patient_diagnostics(beats, records, raw, denoised):
    rows = []
    for patient, part in beats.groupby("patient_id", sort=True):
        metadata = records.loc[records.patient_id == patient]
        metrics = signal_metrics(raw[part.index.to_numpy()], denoised[part.index.to_numpy()])
        denominator = max(1, int(metadata.n_intervals.sum()))
        rows.append({"patient_id": patient, "n_records": len(metadata), "n_beats": len(part),
            "header_label_disagreements": int(metadata.header_label_disagreement.sum()),
            "acute_text_variants": int(metadata.acute_location_text.nunique(dropna=False)),
            "former_location_reported": bool(metadata.former_location_reported.any()),
            "short_interval_fraction": float(metadata.cached_intervals_below_300ms.sum() / denominator),
            "long_interval_fraction": float(metadata.cached_intervals_above_2000ms.sum() / denominator),
            "neighbor_overlap_fraction": float(metadata.next_cached_peak_inside_crop.sum() / denominator),
            "median_relative_denoising_change": float(np.median(metrics["centered_relative_l2_change"])),
            "median_derivative_ratio": float(np.median(metrics["derivative_rms_ratio_denoised_over_raw"])),
            "raw_flat_beat_fraction": float(np.mean(metrics["raw_flat_beat"]))})
    return pd.DataFrame(rows).set_index("patient_id")


def diagnostic_summary(table, classes, seeds):
    numeric = ["n_records", "n_beats", "short_interval_fraction", "long_interval_fraction", "neighbor_overlap_fraction",
               "median_relative_denoising_change", "median_derivative_ratio", "raw_flat_beat_fraction"]
    def describe(frame):
        return {"patients": len(frame), "header_label_disagreements": int(frame.header_label_disagreements.sum()),
            "patients_with_former_location_reported": int(frame.former_location_reported.sum()),
            "patients_with_multiple_acute_texts": int((frame.acute_text_variants > 1).sum()),
            "patient_medians": {name: float(frame[name].median()) if len(frame) else None for name in numeric}}
    result = {"n_patients": len(table), "training_seeds": seeds,
        "selected_persistent_errors": int(table.selected_persistent_error.sum()),
        "baseline_persistent_errors": int(table.baseline_persistent_error.sum()),
        "persistent_in_both": int((table.selected_persistent_error & table.baseline_persistent_error).sum()),
        "selected_error_count_distribution": {str(count): int((table.selected_error_count == count).sum()) for count in range(len(seeds) + 1)},
        "class_groups": {}, "pooled_groups": {}}
    for group, mask in [("persistent", table.selected_persistent_error), ("other", ~table.selected_persistent_error)]:
        result["pooled_groups"][group] = describe(table.loc[mask])
    for label in classes:
        frame = table.loc[table.label == label]
        result["class_groups"][label] = {"persistent": describe(frame.loc[frame.selected_persistent_error]),
                                         "other": describe(frame.loc[~frame.selected_persistent_error])}
    return result


def review_plots(table, beats, raw, denoised, config, source_config, review):
    review.mkdir(parents=True, exist_ok=True)
    paper = yaml.safe_load(Path(source_config["paper_config"]).read_text())
    time = np.arange(-paper["data"]["before_r"], paper["data"]["after_r"] + 1)
    ranking = table.loc[table.selected_persistent_error].sort_values("selected_mean_true_probability", kind="stable")
    index = ["# Persistent-error review packet", "", "Post-hoc model-error selection, not proof of label error or signal corruption. All original labels/data remain unchanged.", ""]
    for patient, row in ranking.iterrows():
        patient_rows = beats.loc[beats.patient_id == patient]
        records = sorted(patient_rows.record.unique())
        positions = np.linspace(0, len(records) - 1, min(config["review_records_per_patient"], len(records)), dtype=int)
        chosen = [records[position] for position in positions]
        figure, axes = plt.subplots(3, len(chosen), figsize=(5 * len(chosen), 7), squeeze=False, sharex=True)
        for column, record in enumerate(chosen):
            part = patient_rows.loc[patient_rows.record == record].sort_values("r_peak")
            position = part.index[len(part) // 2]
            for lead in range(3):
                axis = axes[lead, column]
                axis.plot(time, raw[position, lead], color="0.65", linewidth=0.9, label="Raw")
                axis.plot(time, denoised[position, lead], color="#24578c", linewidth=0.9, label="Denoised")
                axis.axvline(0, color="0.3", linestyle="--", linewidth=0.7)
                axis.set_ylabel(f"{'XYZ'[lead]} (physical units)")
                axis.grid(alpha=0.2)
                if lead == 0:
                    axis.set_title(f"{record}\nCached R={beats.loc[position, 'r_peak']}", fontsize=8)
                if lead == 2:
                    axis.set_xlabel("Time from cached R (ms)")
        axes[0, 0].legend(fontsize=7)
        predictions = ", ".join(f"{seed}: {row[f'selected_prediction_seed_{seed}']}" for seed in source_config["training_seeds"])
        figure.suptitle(f"{patient} | Source label: {row.label}\nSelected predictions {predictions}\nNon-adjudicated review; not a confirmed error cause", fontsize=9)
        figure.tight_layout()
        name = f"{patient}.png"
        figure.savefig(review / name, dpi=130)
        plt.close(figure)
        index.append(f"- [{patient}: source {row.label}, predictions {predictions}]({name})")
    (review / "INDEX.md").write_text("\n".join(index) + "\n")
    form = ranking.reset_index()[["patient_id", "label", "fold", "n_records", "n_beats", "selected_mean_true_probability"]]
    for name in ["reviewer", "signal_quality_assessment", "cached_peak_assessment", "source_label_evidence_notes", "adjudication"]:
        form[name] = ""
    form.to_csv(review / "review_form.csv", index=False)


def aggregate_plots(summary, matrix, classes, output):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = summary["selected_error_count_distribution"]
    ax.bar(list(counts), list(counts.values()), color="0.4")
    ax.set(xlabel="Number of fixed training seeds with an incorrect prediction", ylabel="Unique patients", title="Out-of-fold error persistence: selected procedure")
    fig.tight_layout()
    fig.savefig(output / "error_persistence.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    chart = ax.imshow(matrix, cmap="Greys")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
           xlabel="Predicted source class", ylabel="Source class", title="Mean confusion counts across fixed seeds")
    fig.colorbar(chart, ax=ax, label="Mean patients per seed; not independent repetitions")
    fig.tight_layout()
    fig.savefig(output / "confusion_selected.png", dpi=160)
    plt.close(fig)


def run_audit(config):
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    if len(source["training_seeds"]) < 3 or not isinstance(config["review_records_per_patient"], int) or config["review_records_per_patient"] < 1:
        raise ValueError("Need three seeds and a positive review record count")
    root = Path(source["output_dir"]) / "complete"
    study = json.loads((root / "study_manifest.json").read_text())
    fingerprint = study.pop("study_fingerprint")
    waves, y, groups, bank = load_bank(source)
    if _hash(study) != fingerprint or study["config"] != source or study["bank"] != bank or study["folds_sha256"] != file_hash(source["folds_path"]):
        raise ValueError("Audit source study/cohort changed")
    complete = json.loads((root / "summary.json").read_text())
    if not complete["complete"] or complete["study_fingerprint"] != fingerprint:
        raise ValueError("Audit requires the completed validated regularization study")
    frozen = load_or_create_folds(y, groups, source).set_index("patient_id").sort_index()
    frames, input_files = {}, {}
    for seed in source["training_seeds"]:
        frames[seed] = {}
        for arm in ["baseline", "selected"]:
            csv, metadata = root / f"seed_{seed}/patient_predictions_{arm}.csv", root / f"seed_{seed}/metrics_{arm}.json"
            saved = json.loads(metadata.read_text())
            if saved["study_fingerprint"] != fingerprint or saved["arm"] != arm or saved["training_seed"] != seed or saved["n_beats"] != bank["n_beats"] or saved["evaluated_folds"] != list(range(1, source["outer_splits"] + 1)):
                raise ValueError("Incomplete/mismatched audit source result")
            _read_predictions(csv, saved, frozen, source["classes"])
            frames[seed][arm] = pd.read_csv(csv).set_index("patient_id").sort_index()
            input_files[str(csv)], input_files[str(metadata)] = file_hash(csv), file_hash(metadata)
    signal_config = yaml.safe_load(Path(config["signal_audit_config"]).read_text())
    if signal_config["classes"] != source["classes"] or signal_config["beats_path"] != source["beats_path"] or signal_config["denoised_bank_dir"] != source["bank_dir"]:
        raise ValueError("Error and signal audit cohorts do not align")
    raw_bank = build_raw_bank_and_audit(signal_config)
    paper = yaml.safe_load(Path(source["paper_config"]).read_text())
    beats = pd.read_csv(source["beats_path"])
    beats = beats.loc[beats.label.isin(source["classes"])].reset_index(drop=True)
    record_path = Path(signal_config["audit_dir"]) / "review/record_audit.csv"
    records, sources = source_records(beats, paper, pd.read_csv(record_path))
    scope = {"config": config, "source_fingerprint": fingerprint, "source_predictions_and_metrics": input_files,
        "record_diagnostics_sha256": file_hash(record_path), "sources": sources, "raw_manifest_sha256": file_hash(raw_bank / "manifest.json"),
        "code_sha256": file_hash(__file__)}
    audit_fingerprint = _hash(scope)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    guard = output / "audit_manifest.json"
    # Raw paths/header details remain private; public provenance keeps hashes.
    public_scope = {"audit_fingerprint": audit_fingerprint, "source_study_fingerprint": fingerprint,
                    "source_inputs_sha256": _hash(scope), "code_sha256": scope["code_sha256"]}
    if guard.exists() and json.loads(guard.read_text()) != public_scope:
        raise ValueError("Error audit inputs/settings changed; choose a new output_dir")
    _save_json(guard, public_scope)
    raw = np.load(raw_bank / "waveforms.npy", mmap_mode="r")
    table = error_table(frames, frozen, source["classes"], source["training_seeds"])
    table = table.join(patient_diagnostics(beats, records, raw, waves), validate="one_to_one")
    if table.n_beats.isna().any():
        raise ValueError("Missing patient signal diagnostics")
    review = output / "review"
    review.mkdir(exist_ok=True)
    table.to_csv(review / "patient_error_audit.csv")
    records.to_csv(review / "record_source_audit.csv", index=False)
    _save_json(review / "input_provenance.json", scope)
    review_plots(table, beats, raw, waves, config, source, review)
    summary = diagnostic_summary(table, source["classes"], source["training_seeds"])
    duplicates = records.groupby("waveform_sha256").patient_id.nunique()
    summary.update(audit_fingerprint=audit_fingerprint, n_records=len(records), n_beats=len(beats),
        cross_patient_exact_waveform_duplicate_groups=int((duplicates > 1).sum()),
        review_case_count=int(table.selected_persistent_error.sum()), independent_expert_adjudication=False,
        disclosure="Post-hoc descriptive error screening. No relabeling, exclusions or detector changes; flags do not diagnose annotation mistakes, artifact causes or physiology. Learning-curve subsets must not use error flags.")
    matrices = [confusion_matrix(frames[seed]["selected"].truth, frames[seed]["selected"].prediction, labels=source["classes"]) for seed in source["training_seeds"]]
    matrix = np.mean(matrices, axis=0)
    pd.DataFrame(matrix, index=source["classes"], columns=source["classes"]).to_csv(output / "mean_confusion_counts.csv")
    aggregate_plots(summary, matrix, source["classes"], output)
    _save_json(output / "summary.json", summary)
    lines = ["# Persistent patient-error audit", "", f"Completed out-of-fold predictions: {len(table)} unchanged patients, {len(records)} records, {len(beats):,} original beats, seeds {source['training_seeds']}.", "",
        f"The selected procedure misclassifies **{summary['selected_persistent_errors']} patients in every seed**; the baseline has {summary['baseline_persistent_errors']}, with {summary['persistent_in_both']} persistent in both. These are model failures against source labels, not confirmed annotation mistakes.", "",
        "![Error persistence](error_persistence.png)", "", "![Mean confusion counts](confusion_selected.png)", "",
        "## Class-stratified review flags", "", "| Source class | Persistent | Other | Header disagreements (persistent/other) | Former location reported (persistent/other) | Median short-interval fraction (persistent/other) |", "|---|---:|---:|---|---|---|"]
    for label, values in summary["class_groups"].items():
        a, b = values["persistent"], values["other"]
        def number(value):
            return "—" if value is None else f"{value:.3f}"
        lines.append(f"| {label} | {a['patients']} | {b['patients']} | {a['header_label_disagreements']}/{b['header_label_disagreements']} | {a['patients_with_former_location_reported']}/{b['patients_with_former_location_reported']} | {number(a['patient_medians']['short_interval_fraction'])}/{number(b['patient_medians']['short_interval_fraction'])} |")
    lines += ["", "Fractions and denoising diagnostics are summarized at patient level. Class mix differs between persistent/other groups; pooled differences cannot identify causes. Small strata are descriptive, without significance tests or causal claims. Full class/group diagnostic medians are in `summary.json`.", "",
        f"The exact-byte raw XYZ screen found {summary['cross_patient_exact_waveform_duplicate_groups']} cross-patient duplicate groups. This is not a near-duplicate or hidden-identity audit.", "",
        "## Local review packet", "", f"`review/INDEX.md` links all {summary['review_case_count']} persistent cases. Plots show median cached beats from up to {config['review_records_per_patient']} recordings per patient, selected by record order, not waveform appearance. Cached R is a pipeline position, not independent ground truth. Per-patient predictions, source annotation text and diagnostics are local in `review/`. An unfilled review form is supplied for a qualified reviewer.", "",
        "Header disagreement means disagreement with the parser/cache, not an adjudicated clinical error. Multiple acute text variants can be spelling variants. A reported former location is context, not permission to replace the acute label. Short/long cached intervals and neighboring-peak overlap are screening flags, not confirmed missed/false peaks. Low source-class probability is an uncalibrated model score and only orders review; no probability ensemble is evaluated.", "",
        "No independent expert adjudication was supplied, and no patient, beat, label or cached R position was changed. The model-error-selected review cases are not a random cohort sample. This audit cannot establish label noise, diagnostic ambiguity, signal-quality causes or a dataset ceiling. Training-size subsets are generated independently from class labels/patient IDs and must not use these flags.", ""]
    destination = output / "REPORT.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/diagnostics_vcg.yaml")
    args = parser.parse_args()
    print(run_audit(yaml.safe_load(Path(args.config).read_text())))
