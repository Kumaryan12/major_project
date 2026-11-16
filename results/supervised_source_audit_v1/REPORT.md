# PTB-XL supervised-source feasibility audit

Source-only native-label/provenance feasibility on PTB-XL; no clinical label adjudication, patient-identity clearance, training result or MI SOTA claim.

Source metadata: 21,799 ECGs / 18,869 patients. Candidate roles: 17,418 records / 15,023 training patients in folds 1–8, 2,183 records / 1,942 validation patients in fold 9. All their records and overlapping native annotations are retained; no one-record-per-patient or target six-way label filtering is imposed.

Fold 10 is excluded before label parsing and waveform-file access. Its metadata identity/fold/path fields are validated to catch patient leakage, but its labels/signals do not set the source objective. It was already evaluated in earlier project development and is not a fresh test.

## MI and normal-ECG code support

Counts are unique patients with at least one matching recording, not independent positive records or adjudicated target-class counts. The same patient can contribute to multiple source labels. Listed likelihood 0 means unknown; 100 is the recorded maximum likelihood, not independent clinical confirmation.

| Native code | Train: any listed | Train: likelihood 100 | Validation: any listed | Validation: likelihood 100 | Train: old strict rule |
|---|---:|---:|---:|---:|---:|
| AMI | 278 | 87 | 36 | 15 | 149 |
| ALMI | 196 | 138 | 26 | 20 | 54 |
| ASMI | 1618 | 1044 | 216 | 145 | 809 |
| IMI | 1873 | 747 | 242 | 80 | 1151 |
| ILMI | 328 | 226 | 45 | 29 | 150 |
| NORM | 7109 | 5440 | 905 | 681 | 6489 |

NORM remains a native normal-ECG statement. No NORM→clinical HC ground-truth mapping is approved. The old strict-rule column is historical comparison only, not a new eligibility rule. `diagnostic_subclass=AMI` is a broad source hierarchy containing ASMI/ALMI and other statements, not the exclusive target AMI class. No injury code or broad parent is silently relabeled as one of the target territories.

## Native diagnostic supervision

The schema contains 44 native diagnostic codes. The private states matrix uses 0=unlisted, 1=listed/unknown likelihood 0, 2=listed/intermediate likelihood, 3=listed/likelihood 100. Original likelihoods, per-record labels, patient grouping and roles are preserved. No masked-loss targets, calibrated soft labels or clinically verified negative labels are declared here.

Unlisted codes are absence of annotation, not adjudicated absence of disease. A future BCE/negative-label convention must be declared and justified before training. Missing/uncertain positives must not silently become negatives; intermediate likelihoods must not be treated as calibrated probabilities. Records without diagnostic statements or maximum-likelihood statements are counted, not automatically called normal.

A proposed activity screen based solely on training counts (at least 20 patients with code likelihood 100) retains 34 codes: NDT, NST_, DIG, LNGQT, NORM, IMI, ASMI, LVH, LAFB, ISC_, IRBBB, 1AVB, IVCD, ISCAL, CRBBB, CLBBB, ILMI, LAO/LAE, AMI, ALMI, ISCIN, INJAS, LMI, ISCIL, LPFB, ISCAS, INJAL, ISCLA, RVH, RAO/RAE, WPW, ILBBB, IPLMI, ISCAN. This is a training-only support flag, not a trained classifier/head or clinical approval. All codes remain in the audit and private states matrix.

Among those codes, the validation support caution (<5 patients at likelihood100) applies to: RVH, IPLMI, ISCAN. Validation support does not choose the activity screen or remove source labels.

| Role | Patients with multiple listed MI codes | Patients with multiple likelihood100 MI codes | Records with no diagnostic statement | Records with no likelihood100 diagnostic statement |
|---|---:|---:|---:|---:|
| train | 916 | 467 | 334 | 3617 |
| validation | 127 | 68 | 37 | 474 |

Changing diagnostic sets across a patient's visits and NORM/abnormal co-listings are reported in summary.json. They are not automatically designated label noise or a reason to discard the patient. Record-specific labels must not be replaced by a patient's union of labels. Human-validation, automatic-report and infarct-stage strata are also reported without target-error-driven filtering.

## Local waveform availability and encoded-file checks

This audit does not download or decode any waveform. For eligible source roles only, it checks file presence/hashes, local headers, standard 12-lead identity, 500Hz/5000 samples, mV units, positive gains, supported format16 and expected 120,000-byte data size. Header/size compatibility is not physiological signal-quality certification. Previously converted pilot beats are not reused as a full-source waveform bank.

| Role | Candidate records | Cached pairs | Compatible header/size records | Compatible header/size patients | Missing pairs |
|---|---:|---:|---:|---:|---:|
| train | 17418 | 1024 | 1024 | 1024 | 16394 |
| validation | 2183 | 128 | 128 | 128 | 2055 |

The cached subset was selected for the earlier label-free pilot, not balanced native diagnosis coverage. Its maximum-likelihood support is:

| Native code | Cached compatible training patients, likelihood 100 | Cached compatible validation patients, likelihood 100 |
|---|---:|---:|
| AMI | 9 | 1 |
| ALMI | 9 | 3 |
| ASMI | 66 | 3 |
| IMI | 48 | 2 |
| ILMI | 18 | 1 |
| NORM | 355 | 52 |

Exact encoded-data duplicate groups in cached source files: 0; across different patients: 0; across roles: 0. Exact source-target encoded-file matches: 0 records. Target data files checked: 391. These counts cover available cached files only.

Source duplicate identities and any matches stay private. Cross-patient/role duplicate groups would require source-only quarantine/review before splitting/training, without deleting PTB target patients. Zero encoded-byte matches does not establish distinct physical signals, absence of partial copies, different-visit patient disjointness or absence of duplicates among missing files. No new near-recording or identity-linkage screen was performed.

## Provenance and transfer limits

The earlier bounded near-recording screen covered 1,152 source recordings; 1,152 of the current candidate recordings appear in that historical cohort. It checked 9,216 candidate pairs with 20 known-copy controls. This audit verifies report/cohort linkage, not a replay of the old waveform hashes/execution. Its historical result cannot be extended to the expanded candidate cohort.

PTB-XL's standard ECG acquisition differs from measured Frank VCG. Derived XYZ would be a labeled source-transfer domain, not additional measured-Frank target patients. Different anonymized identifiers or acquisition hardware do not prove patient disjointness. Cross-dataset patient identity remains unresolved. The original target cohort/folds/labels and every completed model study remain unchanged.

## Decision and next concrete work

The metadata support investigating source-only native multi-label supervised learning. The audit does not approve clinical source-target class equivalence or clear clean patient-independent PTB transfer. No model fits, source waveform downloads/conversion, target evaluation, new macro F1 or automatic downstream experiment were launched.

Next: define and review the native annotation/negative-mask convention, build a bounded source-only waveform/QC pilot with grouped patients and explicit duplicate handling, and evaluate whether supervised source features learn useful diagnostic information. Then declare a matched scratch-versus-pretrained measured-VCG comparison. An expanded waveform/provenance screen and explicit unresolved-identity limits are required for any transfer claim. A stronger source score or larger file count alone does not establish better PTB localization.

The previous 1,024-patient masked-reconstruction transfer scored 0.520 versus scratch 0.553. This motivates changing the source objective rather than treating reconstruction error as evidence of diagnostic transfer. Keep the full-feature ridge 0.584266 and historical neural ensemble 0.584570 frozen as separate development references; no 0.80–0.90 forecast is made.

Private review files: source_records.csv, waveform_inventory.csv, native_label_states.npz and duplicate_groups.json. The compatibility CSV contains schema/definitions only, with all clinical-equivalence approvals false; a clinician or domain reviewer must assess those semantics. Aggregate counts are research metadata, not individual diagnoses.

Primary documentation: [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), [PTB Diagnostic](https://physionet.org/content/ptbdb/1.0.0/). The two local metadata files were byte-hash verified against the official versioned PTB-XL mirror before configuration was frozen. The run itself is local/read-only except for its new audit outputs.
