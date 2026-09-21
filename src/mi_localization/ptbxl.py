"""Strict PTB-XL label feasibility audit for external patient testing."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from .data import download_file


TARGET_MI = {"AMI", "ALMI", "ASMI", "IMI", "ILMI"}


def parse_scp_codes(value: str) -> dict[str, float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict) or not all(isinstance(k, str) for k in parsed):
        raise ValueError("Expected SCP-code dictionary")
    return {key: float(score) for key, score in parsed.items()}


def classify_record(codes: dict[str, float], mi_codes: set[str], abnormal_diagnostic_codes: set[str], threshold: float = 0) -> tuple[str, str]:
    # SCP likelihood 0 means unknown, not absent; include all listed codes.
    present = {code for code, value in codes.items() if value >= threshold}
    mi = present & mi_codes
    if len(mi) > 1:
        return "EXCLUDE", "multiple_mi_codes"
    if len(mi) == 1:
        location = next(iter(mi))
        if location in TARGET_MI:
            return location, "single_target_mi"
        return "EXCLUDE", "non_target_mi"
    if "NORM" in present and not present & abnormal_diagnostic_codes:
        return "HC", "norm_without_other_diagnostic_code"
    return "EXCLUDE", "no_unambiguous_target"


def download_metadata(config: dict) -> tuple[Path, Path]:
    directory = Path(config["metadata_dir"])
    database = directory / "ptbxl_database.csv"
    statements = directory / "scp_statements.csv"
    base = config["base_url"].rstrip("/")
    download_file(f"{base}/ptbxl_database.csv", database)
    download_file(f"{base}/scp_statements.csv", statements)
    return database, statements


def audit_labels(config: dict) -> dict:
    database_path, statements_path = download_metadata(config)
    database = pd.read_csv(database_path, dtype={"patient_id": "string"})
    statements = pd.read_csv(statements_path).set_index("Unnamed: 0")
    mi_codes = set(statements.index[statements.diagnostic_class == "MI"])
    abnormal = set(statements.index[(statements.diagnostic == 1) & (statements.diagnostic_class != "NORM")])
    classified = [
        classify_record(parse_scp_codes(value), mi_codes, abnormal, float(config.get("minimum_code_likelihood", 0)))
        for value in database.scp_codes
    ]
    database["candidate_label"] = [row[0] for row in classified]
    database["label_reason"] = [row[1] for row in classified]
    # A subject whose other ECG has a different/ambiguous diagnostic status is
    # excluded altogether. This makes patient-level truth unambiguous.
    patient_status_count = database.groupby("patient_id").candidate_label.nunique()
    conflicted_patients = set(patient_status_count.index[patient_status_count > 1])
    strict = database[
        (~database.patient_id.isin(conflicted_patients))
        & (database.candidate_label != "EXCLUDE")
        & database.patient_id.notna()
    ].copy()
    strict = strict.sort_values(["patient_id", "ecg_id"])
    strict["patient_id"] = strict.patient_id.astype(str)
    output = Path(config["audit_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = strict[[
        "ecg_id", "patient_id", "candidate_label", "strat_fold", "filename_hr", "label_reason",
    ]].rename(columns={"candidate_label": "label"})
    manifest.to_csv(output / "strict_manifest.csv", index=False)
    test = manifest[manifest.strat_fold == int(config["test_fold"])]
    test.to_csv(output / "locked_test_manifest.csv", index=False)
    counts = (
        manifest.groupby("label").agg(records=("ecg_id", "count"), patients=("patient_id", "nunique"))
        .reindex(config["classes"], fill_value=0)
    )
    test_counts = (
        test.groupby("label").agg(records=("ecg_id", "count"), patients=("patient_id", "nunique"))
        .reindex(config["classes"], fill_value=0)
    )
    counts.to_csv(output / "cohort_counts.csv")
    test_counts.to_csv(output / "locked_test_counts.csv")
    summary = {
        "version": config["version"],
        "total_records": int(len(database)),
        "total_patients": int(database.patient_id.nunique()),
        "excluded_by_reason_records": {str(k): int(v) for k, v in database.label_reason.value_counts().items() if k != "single_target_mi" and k != "norm_without_other_diagnostic_code"},
        "patients_with_conflicting_record_status": len(conflicted_patients),
        "strict_records": int(len(manifest)),
        "strict_patients": int(manifest.patient_id.nunique()),
        "locked_test_fold": int(config["test_fold"]),
        "locked_test_records": int(len(test)),
        "locked_test_patients": int(test.patient_id.nunique()),
        "strict_class_counts": counts.to_dict(orient="index"),
        "locked_test_class_counts": test_counts.to_dict(orient="index"),
        "label_rule": "one target MI code or isolated NORM; exclude other MI codes, multiple MI codes, and patients with conflicting record status",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

