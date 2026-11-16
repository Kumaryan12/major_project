# PTB-XL native supervised-source feasibility

Implemented and audited on 16 November 2026. This is a **data/annotation audit**, not a model result or a transfer experiment. [Aggregate report](../results/supervised_source_audit_v1/REPORT.md), [machine-readable summary](../results/supervised_source_audit_v1/summary.json), [configuration](../configs/supervised_source_audit_v1.yaml).

## Why this direction

The latest selected-convolution experiment did not improve the frozen full-feature ridge ensemble (0.584266). The separate historical neural ensemble remains 0.584570. There are still only 164 measured-VCG target patients, including 17 AMI and 16 ALMI patients. More beats from those patients do not increase the number of independent patients.

The earlier source pilot used 1,024 PTB-XL training patients and masked reconstruction rather than diagnostic labels. Its target transfer result was 0.520 versus matched scratch 0.553. A source model trained on native diagnoses is a different, still untested hypothesis. This audit establishes whether suitable source annotations and files exist before choosing that objective.

## What was implemented

- Pin the two official PTB-XL 1.0.3 metadata files by SHA-256. Their local bytes were verified against the official versioned mirror. Validate recording paths, unique recording IDs and one fold per patient.
- Retain all recordings in native training folds 1–8 and validation fold 9. Exclude fold 10 before parsing labels or opening waveform files. Its identity/fold/path fields are validated; it is not a fresh test after earlier project use.
- Preserve all 44 native diagnostic labels per recording, including overlapping MI codes and changing annotations across visits. Preserve unlisted, listed/unknown likelihood 0, intermediate likelihood and likelihood 100 as separate states. No patient-level union replaces recording-specific labels.
- Count support by unique patients, confidence state and source role. A proposed activity flag requires at least 20 training patients with likelihood 100. Validation support below five patients is a caution only; it does not select labels.
- Check eligible cached waveform file hashes, local headers, lead identity, units, gains, rate, shape, encoding and byte size. Do not download or decode signals. Count maximum-likelihood label coverage in the compatible cached subset.
- Check exact encoded-file duplicates within the available source and against all 391 original target recordings. Link the earlier bounded overlap report to its historical cohort without claiming a new near-copy screen or patient-identity clearance.
- Verify the original target beat/fold checksums and 164-patient / 391-record / 55,863-beat cohort. Store native record identities, annotations and matrices under ignored `review/`; publish aggregate counts and hashed manifests only.
- Recompute the scope/counts and verify every output hash on resume or `--verify`, including saved native states, likelihoods, recording IDs, patient IDs and roles. Refuse changed code, data, runtime or cached inventory in an existing namespace.

## Findings and decision

PTB-XL metadata contains 21,799 recordings from 18,869 patients. Eligible roles contain **17,418 recordings / 15,023 training patients** and **2,183 recordings / 1,942 validation patients**. These are source patients, not added measured-VCG target patients.

| Native code | Training patients with likelihood 100 | Validation patients with likelihood 100 |
|---|---:|---:|
| AMI | 87 | 15 |
| ALMI | 138 | 20 |
| ASMI | 1,044 | 145 |
| IMI | 747 | 80 |
| ILMI | 226 | 29 |
| NORM | 5,440 | 681 |

Patients can appear in several rows. Likelihood 100 means the recorded maximum likelihood, not independently confirmed disease. Likelihood 0 means unknown. Unlisted codes are unannotated, not adjudicated negatives. Intermediate likelihoods are not assumed calibrated probabilities. NORM is a normal-ECG annotation and does not automatically mean the target clinical HC label. Broad source `diagnostic_subclass=AMI` includes ASMI/ALMI and must not be mapped to the exclusive target AMI class.

There are 34 diagnostic codes meeting the proposed training-only support threshold. Three have fewer than five maximum-likelihood validation patients: RVH, IPLMI and ISCAN. All 44 codes remain in the audit; no classifier head has been trained or clinically approved.

Only **1,024 training / 128 validation recordings** are cached and header/size-compatible. The expanded metadata cohort would require **18,449 additional recording pairs** to make every eligible source recording locally available. This audit starts none of those downloads. The cached subset came from label-free sampling; consult its separate diagnosis-support table before choosing a supervised pilot.

In the cached compatible subset, maximum-likelihood training/validation patient counts are AMI **9/1**, ALMI **9/3**, ASMI **66/3**, IMI **48/2**, ILMI **18/1**, NORM **355/52**. This is too little minority-label coverage to equate the existing masked-pilot cache with the larger supervised-source opportunity. Use a declared expansion protocol rather than silently dropping difficult labels.

No exact encoded-data duplicate group or source-target match was found among those 1,152 cached source files and 391 target files. This does not screen missing recordings, physically equivalent encodings, partial copies or different visits from the same person. The earlier 1,152-source / 9,216-candidate-pair / 20-copy-control screen is historical subset evidence only. Cross-dataset patient identity remains unresolved.

**Decision:** metadata support a source-only native diagnostic pilot. Clean patient-independent transfer is **not cleared**. No new fits, source downloads, target evaluation or macro F1 have been produced. Retain the original target cohort and frozen reference procedures.

## Next implementation boundary

1. Declare the source annotation objective explicitly. If unlisted labels serve as benchmark negatives, state that convention and its limitations; do not call them clinically verified negatives. Decide how unknown/intermediate labels enter masks or sensitivity controls without using target outer errors.
2. Declare a bounded source-only waveform cohort and training budget using native training support, preserve native validation fold 9, group every patient's visits, and handle source duplicates before training. Download/conversion and physiological QC are separate work, with explicit failures and exclusion reasons. ECG-derived XYZ remains a source domain, distinct from measured Frank XYZ.
3. Train and assess source diagnostic prediction first. Compare prespecified objectives/controls on source validation, with per-code support and patient-aware reporting. A large source score alone is not evidence of target localization benefit.
4. Only then specify a matched scratch-versus-supervised-pretrained PTB comparison inside the existing nested patient partitions, including the full source provenance/copy-screen scope and unresolved identity limits. Keep source supervision separate from any later ECG+VCG extension.

No automated follow-on experiment is queued by this audit. The 0.80–0.90 macro-F1 target remains an aspiration without a forecast or SOTA claim.

## Commands and portability

Validation: all **262 tests passed** on the originating Mac, including nine new native-state, support-isolation, fold/path, header/cache, duplicate, privacy, tamper and historical-linkage tests. The finalized real-data CLI audit and independent `--verify` passed. Implementation was pushed in `0af362b`; the public report/summary/CSVs/manifest are published separately from ignored review matrices and record-level annotations.

From the repository root with metadata and the original target beat/fold files available:

```bash
.venv/bin/python -m mi_localization.supervised_source_audit
.venv/bin/python -m mi_localization.supervised_source_audit --verify
.venv/bin/python -m pytest tests/test_supervised_source_audit.py -q
python scripts/project_status.py
```

On Windows use `.venv\Scripts\python.exe`. This audit uses no GPU. The committed manifest describes the originating Mac and cached files; another machine or an expanded waveform cache requires a copied config with a **new output directory** and `--config path/to/config.yaml`. Do not edit historical manifests or overwrite this audit to force a match. Optional historical review/cohort files may be absent in a fresh clone; that reduces claimed historical coverage rather than clearing it. Raw signals, private review packets and the virtual environment are not in Git.

Primary metadata semantics, dataset composition and native folds: [PhysioNet PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/). Target acquisition reference: [PhysioNet PTB Diagnostic](https://physionet.org/content/ptbdb/1.0.0/).
