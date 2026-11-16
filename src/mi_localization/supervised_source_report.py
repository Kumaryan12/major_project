"""Public aggregate report; native record identities/annotations stay in review/."""
from __future__ import annotations

import pandas as pd


def render_report(root, config, summary, statements):
    rows = []
    for label, value in summary["labels"].items():
        for role in ["train", "validation"]:
            for state, counts in value[role].items():
                rows.append(dict(code=label, role=role, annotation_state=state, **counts,
                    training_support_eligible=value["training_support_eligible"], validation_support_caution=value["validation_support_caution"]))
    pd.DataFrame(rows).to_csv(root/"label_counts.csv", index=False)
    compatibility = []
    for code in config["review_codes"]:
        row = statements.loc[code]
        compatibility.append(dict(source_code=code, source_description=str(row.description),
            source_subclass=str(row.diagnostic_subclass), target_name_for_review=code if code != "NORM" else "HC is not automatically equivalent",
            clinical_equivalence_approved=False,
            review_reason="Native normal-ECG annotation does not establish the target clinical healthy-control definition" if code == "NORM" else
            "Same/similar code name does not prove equivalent clinical territory, infarct stage or annotation ground truth"))
    pd.DataFrame(compatibility).to_csv(root/"compatibility_review.csv", index=False)
    roles = summary["source_roles"]
    lines = ["# PTB-XL supervised-source feasibility audit", "", summary["claim_scope"], "",
        f"Source metadata: {summary['all_metadata_records']:,} ECGs / {summary['all_metadata_patients']:,} patients. "
        f"Candidate roles: {roles['train']['records']:,} records / {roles['train']['patients']:,} training patients in folds 1–8, "
        f"{roles['validation']['records']:,} records / {roles['validation']['patients']:,} validation patients in fold 9. "
        "All their records and overlapping native annotations are retained; no one-record-per-patient or target six-way label filtering is imposed.", "",
        "Fold 10 is excluded before label parsing and waveform-file access. Its metadata identity/fold/path fields are validated to catch patient leakage, but its labels/signals do not set the source objective. It was already evaluated in earlier project development and is not a fresh test.", "",
        "## MI and normal-ECG code support", "",
        "Counts are unique patients with at least one matching recording, not independent positive records or adjudicated target-class counts. The same patient can contribute to multiple source labels. Listed likelihood 0 means unknown; 100 is the recorded maximum likelihood, not independent clinical confirmation.", "",
        "| Native code | Train: any listed | Train: likelihood 100 | Validation: any listed | Validation: likelihood 100 | Train: old strict rule |", "|---|---:|---:|---:|---:|---:|"]
    for code in config["review_codes"]:
        r = summary["labels"][code]
        old_name = code if code != "NORM" else "HC"
        old = summary["combinations"]["train"]["old_strict_patient_rule_counts"][old_name]["patients"]
        lines.append(f"| {code} | {r['train']['any_listed']['patients']} | {r['train']['listed_100']['patients']} | "
                     f"{r['validation']['any_listed']['patients']} | {r['validation']['listed_100']['patients']} | {old} |")
    lines += ["", "NORM remains a native normal-ECG statement. No NORM→clinical HC ground-truth mapping is approved. The old strict-rule column is historical comparison only, not a new eligibility rule. `diagnostic_subclass=AMI` is a broad source hierarchy containing ASMI/ALMI and other statements, not the exclusive target AMI class. No injury code or broad parent is silently relabeled as one of the target territories.", "",
        "## Native diagnostic supervision", "",
        f"The schema contains {summary['diagnostic_codes']} native diagnostic codes. The private states matrix uses 0=unlisted, 1=listed/unknown likelihood 0, 2=listed/intermediate likelihood, 3=listed/likelihood 100. Original likelihoods, per-record labels, patient grouping and roles are preserved. No masked-loss targets, calibrated soft labels or clinically verified negative labels are declared here.", "",
        "Unlisted codes are absence of annotation, not adjudicated absence of disease. A future BCE/negative-label convention must be declared and justified before training. Missing/uncertain positives must not silently become negatives; intermediate likelihoods must not be treated as calibrated probabilities. Records without diagnostic statements or maximum-likelihood statements are counted, not automatically called normal.", "",
        f"A proposed activity screen based solely on training counts (at least {config['minimum_training_positive_patients']} patients with code likelihood 100) retains "
        f"{len(summary['training_count_eligible_codes'])} codes: {', '.join(summary['training_count_eligible_codes'])}. "
        "This is a training-only support flag, not a trained classifier/head or clinical approval. All codes remain in the audit and private states matrix.", "",
        f"Among those codes, the validation support caution (<{config['validation_support_caution_below']} patients at likelihood100) applies to: "
        f"{', '.join(summary['low_validation_support_codes']) or 'none'}. Validation support does not choose the activity screen or remove source labels.", "",
        "| Role | Patients with multiple listed MI codes | Patients with multiple likelihood100 MI codes | Records with no diagnostic statement | Records with no likelihood100 diagnostic statement |", "|---|---:|---:|---:|---:|"]
    for role in ["train", "validation"]:
        c = summary["combinations"][role]
        lines.append(f"| {role} | {c['multiple_listed_mi_codes']['patients']} | {c['multiple_100_mi_codes']['patients']} | "
                     f"{c['no_diagnostic_code_listed']['records']} | {c['no_diagnostic_code_at_100']['records']} |")
    lines += ["", "Changing diagnostic sets across a patient's visits and NORM/abnormal co-listings are reported in summary.json. They are not automatically designated label noise or a reason to discard the patient. Record-specific labels must not be replaced by a patient's union of labels. Human-validation, automatic-report and infarct-stage strata are also reported without target-error-driven filtering.", "",
        "## Local waveform availability and encoded-file checks", "",
        "This audit does not download or decode any waveform. For eligible source roles only, it checks file presence/hashes, local headers, standard 12-lead identity, 500Hz/5000 samples, mV units, positive gains, supported format16 and expected 120,000-byte data size. Header/size compatibility is not physiological signal-quality certification. Previously converted pilot beats are not reused as a full-source waveform bank.", "",
        "| Role | Candidate records | Cached pairs | Compatible header/size records | Compatible header/size patients | Missing pairs |", "|---|---:|---:|---:|---:|---:|"]
    for role in ["train", "validation"]:
        a = summary["availability"][role]
        lines.append(f"| {role} | {a['records']} | {a['cached_complete_pairs']} | {a['compatible_header_and_size_records']} | "
                     f"{a['compatible_header_and_size_patients']} | {a['missing_complete_pairs']} |")
    lines += ["", "The cached subset was selected for the earlier label-free pilot, not balanced native diagnosis coverage. Its maximum-likelihood support is:", "",
              "| Native code | Cached compatible training patients, likelihood 100 | Cached compatible validation patients, likelihood 100 |", "|---|---:|---:|"]
    for code in config["review_codes"]:
        c = summary["cached_maximum_likelihood_label_support"][code]
        lines.append(f"| {code} | {c['train']['patients']} | {c['validation']['patients']} |")
    d = summary["duplicates"]
    lines += ["", f"Exact encoded-data duplicate groups in cached source files: {d['exact_duplicate_groups']}; "
        f"across different patients: {d['cross_patient_duplicate_groups']}; across roles: {d['cross_role_duplicate_groups']}. "
        f"Exact source-target encoded-file matches: {d['exact_encoded_source_target_match_records']} records. "
        f"Target data files checked: {d['target_data_files_checked']}. These counts cover available cached files only.", "",
        "Source duplicate identities and any matches stay private. Cross-patient/role duplicate groups would require source-only quarantine/review before splitting/training, without deleting PTB target patients. Zero encoded-byte matches does not establish distinct physical signals, absence of partial copies, different-visit patient disjointness or absence of duplicates among missing files. No new near-recording or identity-linkage screen was performed.", "",
        "## Provenance and transfer limits", ""]
    prior = summary["historical_screen"]
    if "historical_screen_source_records" in prior:
        lines += [f"The earlier bounded near-recording screen covered {prior['historical_screen_source_records']:,} source recordings; "
            f"{prior['candidate_records_in_historical_cohort']:,} of the current candidate recordings appear in that historical cohort. "
            f"It checked {prior['historical_screen_candidate_pairs']:,} candidate pairs with {prior['historical_screen_known_copy_controls']} known-copy controls. "
            "This audit verifies report/cohort linkage, not a replay of the old waveform hashes/execution. Its historical result cannot be extended to the expanded candidate cohort.", ""]
    else: lines += ["Historical source cohort/screen files are incomplete on this checkout; no old screening coverage is claimed.", ""]
    lines += ["PTB-XL's standard ECG acquisition differs from measured Frank VCG. Derived XYZ would be a labeled source-transfer domain, not additional measured-Frank target patients. Different anonymized identifiers or acquisition hardware do not prove patient disjointness. Cross-dataset patient identity remains unresolved. The original target cohort/folds/labels and every completed model study remain unchanged.", "",
        "## Decision and next concrete work", "",
        "The metadata support investigating source-only native multi-label supervised learning. The audit does not approve clinical source-target class equivalence or clear clean patient-independent PTB transfer. No model fits, source waveform downloads/conversion, target evaluation, new macro F1 or automatic downstream experiment were launched.", "",
        "Next: define and review the native annotation/negative-mask convention, build a bounded source-only waveform/QC pilot with grouped patients and explicit duplicate handling, and evaluate whether supervised source features learn useful diagnostic information. Then declare a matched scratch-versus-pretrained measured-VCG comparison. An expanded waveform/provenance screen and explicit unresolved-identity limits are required for any transfer claim. A stronger source score or larger file count alone does not establish better PTB localization.", "",
        "The previous 1,024-patient masked-reconstruction transfer scored 0.520 versus scratch 0.553. This motivates changing the source objective rather than treating reconstruction error as evidence of diagnostic transfer. Keep the full-feature ridge 0.584266 and historical neural ensemble 0.584570 frozen as separate development references; no 0.80–0.90 forecast is made.", "",
        "Private review files: source_records.csv, waveform_inventory.csv, native_label_states.npz and duplicate_groups.json. The compatibility CSV contains schema/definitions only, with all clinical-equivalence approvals false; a clinician or domain reviewer must assess those semantics. Aggregate counts are research metadata, not individual diagnoses.", "",
        "Primary documentation: [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), "
        "[PTB Diagnostic](https://physionet.org/content/ptbdb/1.0.0/). The two local metadata files were byte-hash verified against the official versioned PTB-XL mirror before configuration was frozen. The run itself is local/read-only except for its new audit outputs.", ""]
    (root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
