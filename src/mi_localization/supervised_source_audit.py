"""Read-only PTB-XL native-label/data feasibility; no downloads, model fits or signal decoding."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd
import wfdb
import yaml

from . import ptbxl

STATES = {"unlisted": 0, "listed_unknown_0": 1, "listed_intermediate": 2, "listed_100": 3}
CLAIM = "Source-only native-label/provenance feasibility on PTB-XL; no clinical label adjudication, patient-identity clearance, training result or MI SOTA claim."


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024*1024): digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def validate_config(config):
    if (config["version"] != "1.0.3" or config["train_folds"] != list(range(1, 9))
            or config["validation_fold"] != 9 or config["excluded_fold"] != 10
            or config["review_codes"] != ["AMI", "ALMI", "ASMI", "IMI", "ILMI", "NORM"]):
        raise ValueError("Require native PTB-XL source folds 1–8/9 and excluded fold 10")
    for key in ["minimum_training_positive_patients", "validation_support_caution_below"]:
        if type(config[key]) != int or config[key] < 1: raise ValueError("Invalid support-audit rule")


def source_frame(database, config):
    """Validate all identity/fold/path metadata before parsing only source-role labels."""
    required = ["ecg_id", "patient_id", "strat_fold", "filename_hr", "scp_codes"]
    if not set(required) <= set(database): raise ValueError("Missing source metadata")
    frame = database.copy()
    for key in ["ecg_id", "patient_id", "strat_fold"]:
        x = pd.to_numeric(frame[key], errors="raise")
        if x.isna().any() or not np.isfinite(x).all() or (x <= 0).any() or (x != np.floor(x)).any():
            raise ValueError("Source identifiers/folds must be finite positive integers")
        frame[key] = x.astype(np.int64)
    if not frame.ecg_id.is_unique or not frame.filename_hr.is_unique or set(frame.strat_fold) != set(range(1, 11)):
        raise ValueError("Duplicate recordings or unexpected source folds")
    if (frame.groupby("patient_id").strat_fold.nunique() != 1).any(): raise ValueError("Source patient crosses folds")
    for row in frame.itertuples():
        if (not isinstance(row.filename_hr, str) or not re.fullmatch(r"records500/\d{5}/\d{5}_hr", row.filename_hr)
                or int(row.filename_hr.rsplit("/", 1)[1].split("_")[0]) != row.ecg_id):
            raise ValueError("Unsafe/mismatched source waveform path")
    selected = frame.loc[frame.strat_fold != 10].sort_values(["strat_fold", "patient_id", "ecg_id"]).reset_index(drop=True)
    selected["role"] = np.where(selected.strat_fold.isin(config["train_folds"]), "train", "validation")
    if set(selected.strat_fold) - set(range(1, 10)): raise ValueError("Excluded source fold selected")
    return selected


def parse_codes(value, known):
    """Literal parsing with duplicate-key and probability-range checks; 0 is unknown."""
    if not isinstance(value, str): raise ValueError("Missing native annotation dictionary")
    try:
        node = ast.parse(value, mode="eval").body
        if not isinstance(node, ast.Dict): raise ValueError("Expected native annotation dictionary")
        keys = [ast.literal_eval(k) for k in node.keys]
        if len(set(keys)) != len(keys): raise ValueError("Duplicate native annotation code")
        codes = ast.literal_eval(node)
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError("Malformed native annotation dictionary") from error
    if any(not isinstance(k, str) or k not in known for k in codes): raise ValueError("Unknown native annotation code")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 100 for v in codes.values()):
        raise ValueError("Invalid native annotation likelihood")
    return {k: float(v) for k, v in codes.items()}


def label_states(frame, statements, config):
    if not statements.index.is_unique or not {"description", "diagnostic", "diagnostic_class", "diagnostic_subclass"} <= set(statements):
        raise ValueError("Invalid native statement schema")
    labels = statements.index[statements.diagnostic.eq(1)].astype(str).tolist()
    if not set(config["review_codes"]) <= set(labels): raise ValueError("Required review labels missing")
    known = set(statements.index.astype(str))
    codes = [parse_codes(v, known) for v in frame.scp_codes]
    state = np.zeros((len(frame), len(labels)), np.uint8)
    likelihood = np.full((len(frame), len(labels)), np.nan, np.float32)
    for i, row in enumerate(codes):
        for j, label in enumerate(labels):
            if label in row:
                value = row[label]
                likelihood[i, j] = value
                state[i, j] = 1 if value == 0 else 3 if value == 100 else 2
    return labels, codes, state, likelihood


def role_counts(frame, mask):
    part = frame.loc[mask]
    return dict(records=len(part), patients=int(part.patient_id.nunique()))


def label_summary(frame, labels, states, config):
    result = {}
    for j, label in enumerate(labels):
        result[label] = {}
        for role in ["train", "validation"]:
            selected = frame.role.eq(role).to_numpy()
            result[label][role] = {name: role_counts(frame, selected & (states[:, j] == value)) for name, value in STATES.items()}
            result[label][role]["any_listed"] = role_counts(frame, selected & (states[:, j] > 0))
        # This proposed head-activity rule is training-count-only, not selected by validation support.
        result[label]["training_support_eligible"] = result[label]["train"]["listed_100"]["patients"] >= config["minimum_training_positive_patients"]
        result[label]["validation_support_caution"] = result[label]["validation"]["listed_100"]["patients"] < config["validation_support_caution_below"]
    return result


def cached_label_summary(frame, labels, states, inventory):
    """Usable-file coverage of maximum-likelihood labels, without selecting a task."""
    cached_ids = inventory.loc[inventory.header_and_size_compatible, "ecg_id"]
    cached = frame.ecg_id.isin(cached_ids).to_numpy()
    return {label: {role: role_counts(frame, cached & frame.role.eq(role).to_numpy() & (states[:, j] == 3))
                    for role in ["train", "validation"]} for j, label in enumerate(labels)}


def combination_summary(frame, codes, states, labels, statements):
    mi = set(statements.index[statements.diagnostic_class.eq("MI")])
    abnormal = set(labels)-{"NORM"}
    result = {}
    for role in ["train", "validation"]:
        selected = frame.role.eq(role).to_numpy()
        listed = [set(c) & set(labels) for c in codes]
        maximum = [{k for k, v in c.items() if v == 100} & set(labels) for c in codes]
        old = [ptbxl.classify_record(c, mi, abnormal)[0] for c in codes]
        result[role] = dict(
            multiple_listed_mi_codes=role_counts(frame, selected & np.array([len(set(c) & mi) > 1 for c in codes])),
            multiple_100_mi_codes=role_counts(frame, selected & np.array([len(c & mi) > 1 for c in maximum])),
            no_diagnostic_code_listed=role_counts(frame, selected & np.array([not c for c in listed])),
            no_diagnostic_code_at_100=role_counts(frame, selected & ~np.any(states == 3, axis=1)),
            norm_and_abnormal_listed=role_counts(frame, selected & np.array(["NORM" in c and bool(c & abnormal) for c in listed])),
            norm_and_abnormal_at_100=role_counts(frame, selected & np.array(["NORM" in c and bool(c & abnormal) for c in maximum])),
            patients_with_changing_listed_diagnostic_sets=sum(len(set(tuple(sorted(listed[i])) for i in group.index)) > 1
                for _, group in frame.loc[selected].groupby("patient_id")),
            old_single_record_rule_counts={label: role_counts(frame, selected & np.array([v == label for v in old])) for label in ["AMI", "ALMI", "ASMI", "IMI", "ILMI", "HC", "EXCLUDE"]})
    # Reproduce the old strict patient-consistency rule inside source roles only.
    old = np.asarray([ptbxl.classify_record(c, mi, abnormal)[0] for c in codes])
    strict = frame[["patient_id", "role"]].copy();strict["label"] = old
    consistent = strict.groupby("patient_id").label.nunique().eq(1)
    keep = strict.patient_id.isin(consistent.index[consistent]) & strict.label.ne("EXCLUDE")
    for role in result:
        result[role]["old_strict_patient_rule_counts"] = {label: role_counts(frame, keep.to_numpy() & frame.role.eq(role).to_numpy() & (old == label))
            for label in ["AMI", "ALMI", "ASMI", "IMI", "ILMI", "HC"]}
    return result


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts: raise ValueError("Unsafe waveform path")
    candidate = root/relative
    path = candidate.resolve()
    if not path.is_relative_to(root): raise ValueError("Waveform path escapes its dataset root")
    if path != candidate: raise ValueError("Refusing symlinked waveform access")
    return path


def inspect_header(path, expected_data_name):
    """Read only a local .hea, never decode samples or follow an excluded record."""
    try:
        h = wfdb.rdheader(str(path.with_suffix("")))
        expected_leads = {"i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"}
        if (h.fs != 500 or h.sig_len != 5000 or h.n_sig != 12 or len(set(n.lower() for n in h.sig_name)) != 12
                or set(n.lower() for n in h.sig_name) != expected_leads): return "shape_rate_or_leads"
        if any(str(u).lower() != "mv" for u in h.units): return "non_mv_units"
        if (any(str(v) != "16" for v in h.fmt) or set(h.file_name) != {expected_data_name}
                or any(v not in (None, 0) for v in h.byte_offset)): return "unsupported_encoding_or_data_path"
        if not np.isfinite(h.adc_gain).all() or np.any(np.asarray(h.adc_gain) <= 0): return "invalid_gain"
        return "compatible_header"
    except (OSError, ValueError, TypeError, AttributeError, IndexError): return "unreadable_header"


def source_inventory(frame, config):
    rows = []
    for row in frame.itertuples():
        if row.strat_fold == 10 or (row.role == "train" and row.strat_fold not in config["train_folds"]) or (row.role == "validation" and row.strat_fold != 9):
            raise ValueError("Refusing excluded or wrong-role waveform access")
        item = dict(ecg_id=int(row.ecg_id), patient_id=int(row.patient_id), role=row.role, filename_hr=row.filename_hr)
        paths = {ext: safe_path(config["waveform_dir"], f"{row.filename_hr}.{ext}") for ext in ["hea", "dat"]}
        for ext, path in paths.items():
            present = path.is_file()
            item[f"{ext}_present"] = present
            item[f"{ext}_bytes"] = path.stat().st_size if present else 0
            item[f"{ext}_sha256"] = file_hash(path) if present else ""
        item["header_status"] = inspect_header(paths["hea"], paths["dat"].name) if item["hea_present"] else "missing_header"
        item["complete_pair"] = item["hea_present"] and item["dat_present"]
        item["header_and_size_compatible"] = item["complete_pair"] and item["header_status"] == "compatible_header" and item["dat_bytes"] == 5000*12*2
        rows.append(item)
    inventory = pd.DataFrame(rows)
    summaries = {}
    for role, part in inventory.groupby("role", sort=True):
        summaries[role] = dict(records=len(part), cached_complete_pairs=int(part.complete_pair.sum()),
            compatible_header_and_size_records=int(part.header_and_size_compatible.sum()),
            compatible_header_and_size_patients=int(part.loc[part.header_and_size_compatible, "patient_id"].nunique()),
            missing_complete_pairs=int((~part.complete_pair).sum()), header_status_counts={k:int(v) for k,v in part.header_status.value_counts().items()},
            existing_data_bytes=int(part.dat_bytes.sum()))
    return inventory, summaries


def target_inventory(config):
    """Protect the original target cohort and hash its data files; no target model reads."""
    if (file_hash(config["target_beats_path"]) != config["target_beats_sha256"]
            or file_hash(config["target_folds_path"]) != config["target_folds_sha256"]):
        raise ValueError("Protected target beat/fold checksum changed")
    beats = pd.read_csv(config["target_beats_path"], dtype={"patient_id": str})
    beats = beats.loc[beats.label.isin(config["target_classes"])].copy()
    counts = dict(patients=int(beats.patient_id.nunique()), records=int(beats.record.nunique()), beats=len(beats))
    if counts != config["expected_target"] or (beats.groupby("patient_id").label.nunique() != 1).any():
        raise ValueError("Original target cohort changed")
    folds = pd.read_csv(config["target_folds_path"], dtype={"patient_id": str})
    patients = beats[["patient_id", "label"]].drop_duplicates().sort_values("patient_id")
    if (len(folds) != len(patients) or not folds.patient_id.is_unique or set(folds.patient_id) != set(patients.patient_id)
            or not folds.set_index("patient_id").label.sort_index().equals(patients.set_index("patient_id").label.sort_index())):
        raise ValueError("Frozen target patient labels/folds changed")
    rows = []
    for record in sorted(beats.record.unique()):
        path = safe_path(config["target_waveform_dir"], str(record)+".dat")
        rows.append(dict(record=record, present=path.is_file(), data_sha256=file_hash(path) if path.is_file() else ""))
    return rows, dict(counts, data_files_present=sum(r["present"] for r in rows),
                      beats_sha256=file_hash(config["target_beats_path"]), folds_sha256=file_hash(config["target_folds_path"]))


def duplicate_summary(inventory, target):
    groups = []
    data = inventory.loc[inventory.dat_present]
    for digest, part in data.groupby("dat_sha256"):
        if len(part) > 1:
            groups.append(dict(data_sha256=digest, records=part.ecg_id.astype(int).tolist(),
                               cross_patient=part.patient_id.nunique() > 1, cross_role=part.role.nunique() > 1))
    cross = [g for g in groups if g["cross_patient"]]
    target_hashes = {r["data_sha256"] for r in target if r["present"]}
    matches = inventory.loc[inventory.dat_present & inventory.dat_sha256.isin(target_hashes)]
    result = dict(cached_source_data_files=int(inventory.dat_present.sum()), exact_duplicate_groups=len(groups),
        cross_patient_duplicate_groups=len(cross), cross_role_duplicate_groups=sum(g["cross_role"] for g in groups),
        cross_patient_duplicate_records=sum(len(g["records"]) for g in cross),
        exact_encoded_source_target_match_records=len(matches),
        target_data_files_checked=sum(r["present"] for r in target), target_unique_data_hashes=len(target_hashes), waveform_near_duplicate_screen_performed=False,
        patient_identity_independence_cleared=False,
        limits="Exact encoded-file hashes on available data only; unequal formats/rates, partial copies, different visits and missing source files are not resolved")
    return result, groups, matches.ecg_id.astype(int).tolist()


def prior_evidence(frame, config):
    files = {}
    for key in ["prior_cohort_path", "prior_cohort_manifest", "prior_screen_manifest", "prior_screen_summary"]:
        path = Path(config[key]);files[key] = file_hash(path) if path.is_file() else None
    result = dict(files=files, historical_only=True, expanded_cohort_screened=False,
                  patient_identity_independence_cleared=False)
    if all(files.values()):
        manifest = json.loads(Path(config["prior_cohort_manifest"]).read_text())
        if files["prior_cohort_path"] != manifest["cohort_sha256"]: raise ValueError("Prior source cohort was modified")
        summary = json.loads(Path(config["prior_screen_summary"]).read_text())
        screen = json.loads(Path(config["prior_screen_manifest"]).read_text())
        if summary["screen_fingerprint"] != screen["screen_fingerprint"]: raise ValueError("Prior recording-screen scope mismatch")
        old = pd.read_csv(config["prior_cohort_path"])
        result.update(historical_screen_source_records=int(summary["source_records"]),
            candidate_records_in_historical_cohort=int(frame.ecg_id.isin(old.ecg_id).sum()),
            historical_screen_candidate_pairs=int(summary["continuous_pairs_checked"]),
            historical_screen_known_copy_controls=int(summary["positive_controls"]),
            historical_screen_passed=bool(summary["recording_screen_passed"]),
            note="Historical report/cohort linkage only; old waveform hashes and the full historical execution were not replayed here. Coverage must not be extended to the expanded source.")
    return result


def analyze(config):
    validate_config(config)
    for key, sha in [("metadata_path", "metadata_sha256"), ("statements_path", "statements_sha256")]:
        if file_hash(config[key]) != config[sha]: raise ValueError("Pinned official metadata checksum changed")
    database = pd.read_csv(config["metadata_path"])
    statements = pd.read_csv(config["statements_path"], index_col=0)
    frame = source_frame(database, config)
    labels, codes, states, likelihood = label_states(frame, statements, config)
    counts = label_summary(frame, labels, states, config)
    combinations = combination_summary(frame, codes, states, labels, statements)
    inventory, availability = source_inventory(frame, config)
    target, target_counts = target_inventory(config)
    duplicates, private_duplicates, target_matches = duplicate_summary(inventory, target)
    prior = prior_evidence(frame, config)
    active = [label for label in labels if counts[label]["training_support_eligible"]]
    low_validation = [label for label in active if counts[label]["validation_support_caution"]]
    confidence = {}
    metadata_strata = {}
    for role in ["train", "validation"]:
        mask = frame.role.eq(role).to_numpy()
        histogram = Counter(v for i,c in enumerate(codes) if mask[i] for k,v in c.items() if k in labels)
        confidence[role] = {str(k):int(v) for k,v in sorted(histogram.items())}
        metadata_strata[role] = {col:{str(k):int(v) for k,v in frame.loc[mask,col].fillna("<missing>").astype(str).value_counts().items()}
            for col in ["validated_by_human", "initial_autogenerated_report", "second_opinion", "infarction_stadium1", "infarction_stadium2"] if col in frame}
    inventory_bytes = inventory.to_csv(index=False).encode()
    scope = dict(config=config, metadata_sha256=config["metadata_sha256"], statements_sha256=config["statements_sha256"],
        source_inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(), target=target_counts,
        target_inventory_sha256=canonical(target), historical=prior,
        code_sha256=file_hash(__file__), report_code_sha256=file_hash(Path(__file__).with_name("supervised_source_report.py")),
        old_label_rule_code_sha256=file_hash(ptbxl.__file__),
        runtime=dict(numpy=np.__version__,pandas=pd.__version__,wfdb=wfdb.__version__,platform=platform.platform()), claim_scope=CLAIM)
    fingerprint = canonical(scope)
    summary = dict(complete=True, audit_fingerprint=fingerprint, claim_scope=CLAIM,
        all_metadata_records=len(database), all_metadata_patients=int(database.patient_id.nunique()),
        source_roles={role:role_counts(frame,frame.role.eq(role)) for role in ["train","validation"]},
        excluded_fold_10=dict(records=int(database.strat_fold.eq(10).sum()),patients=int(database.loc[database.strat_fold.eq(10),"patient_id"].nunique()),
                             labels_parsed=0,waveform_files_accessed=0),
        diagnostic_codes=len(labels), labels=counts, combinations=combinations, confidence_histogram=confidence,
        metadata_strata=metadata_strata, availability=availability,
        cached_maximum_likelihood_label_support=cached_label_summary(frame,labels,states,inventory),
        duplicates=duplicates, historical_screen=prior,
        target_cohort=target_counts, training_count_eligible_codes=active, low_validation_support_codes=low_validation,
        normal_ecg_is_not_clinically_healthy_control=True, native_code_target_equivalence_clinically_reviewed=False,
        unlisted_is_adjudicated_negative=False, patient_identity_independence_cleared=False,
        source_only_metadata_feasible=bool(active), expanded_source_waveform_qc_complete=False,
        clean_target_transfer_ready=False, training_launched=False, classification_result_available=False,
        next_step="Define native multilabel supervision and a bounded source-only waveform/QC pilot; keep any later PTB transfer exploratory until provenance/label limits are addressed")
    private = dict(frame=frame,labels=labels,states=states,likelihood=likelihood,inventory=inventory,
                   duplicates=private_duplicates,target_matches=target_matches,statements=statements)
    return summary,dict(scope,audit_fingerprint=fingerprint),private


def run(config, verify=False):
    from .supervised_source_report import render_report
    summary, scope, private = analyze(config)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root/"manifest.json"
    if verify or path.exists():
        saved = json.loads(path.read_text())
        if saved["scope"] != scope: raise ValueError("Supervised source audit inputs/code/cache changed; use a new namespace")
        if any(file_hash(root/name) != digest for name,digest in saved["files"].items()): raise ValueError("Supervised source audit output changed")
        if json.loads((root/"summary.json").read_text()) != summary: raise ValueError("Supervised source counts do not reproduce")
        with np.load(root/"review/native_label_states.npz") as a:
            if (not np.array_equal(a["states"],private["states"]) or not np.array_equal(a["labels"],private["labels"])
                    or not np.array_equal(a["ecg_id"],private["frame"].ecg_id.to_numpy())
                    or not np.array_equal(a["patient_id"],private["frame"].patient_id.to_numpy())
                    or not np.array_equal(a["role"],private["frame"].role.to_numpy(dtype=str))
                    or not np.array_equal(a["likelihood"],private["likelihood"],equal_nan=True)):
                raise ValueError("Native supervision states do not reproduce")
        return summary
    review = root/"review";review.mkdir(exist_ok=True)
    frame = private["frame"]
    # No report strings, names or free clinical text are copied into review artifacts.
    frame[["ecg_id","patient_id","strat_fold","role","filename_hr","scp_codes"]].to_csv(review/"source_records.csv",index=False)
    private["inventory"].to_csv(review/"waveform_inventory.csv",index=False)
    np.savez_compressed(review/"native_label_states.npz",states=private["states"],likelihood=private["likelihood"],
                       labels=private["labels"],ecg_id=frame.ecg_id.to_numpy(),patient_id=frame.patient_id.to_numpy(),role=frame.role.to_numpy(dtype=str))
    save_json(review/"duplicate_groups.json",dict(source_groups=private["duplicates"],target_match_source_records=private["target_matches"]))
    save_json(root/"summary.json",summary)
    render_report(root,config,summary,private["statements"])
    files = ["summary.json","REPORT.md","label_counts.csv","compatibility_review.csv","review/source_records.csv",
             "review/waveform_inventory.csv","review/native_label_states.npz","review/duplicate_groups.json"]
    save_json(path,dict(scope=scope,files={name:file_hash(root/name) for name in files}))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/supervised_source_audit_v1.yaml")
    p.add_argument("--verify", action="store_true")
    args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    result=run(config,args.verify)
    print(json.dumps({k:result[k] for k in ["complete","source_roles","source_only_metadata_feasible","clean_target_transfer_ready","training_launched"]}))
