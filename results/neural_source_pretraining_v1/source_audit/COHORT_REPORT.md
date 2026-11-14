# Frozen source pretraining cohort

PTB-XL 1.0.3: 21,799 records, 18,869 unique source patients.

Pilot: 1,024 training patients from folds 1–8 and 128 validation patients from fold 9. One recording per patient, fixed seeded patient selection, lowest ECG ID per selected patient. No diagnostic labels, six-class mapping, model errors or metadata quality flags enter selection.

All source patient IDs stay within one metadata fold. Fold 10 is excluded from the new waveform pipeline, even though old external-test files already exist locally. Source access functions independently reject excluded/mismatched roles before downloads or waveform reads.

## Provenance and identity limits

PTB-XL uses Schiller ECG data acquired in 1989–1996; PTB Diagnostic uses a PTB prototype recorder and includes measured Frank XYZ. These are different acquisition domains. A source-derived XYZ signal is not a measured Frank signal.

PTB and PTB-XL patient identifiers occupy unrelated de-identified namespaces. Different identifier strings, institutions/devices or a lack of exact file duplicates do not establish that the underlying people are disjoint. Cross-dataset identity linkage and comprehensive aligned waveform-overlap screening are unresolved. This audit does not clear downstream PTB patient independence.

Current permitted experiment: source-only masked reconstruction with source patient-disjoint validation. No PTB fine-tuning/evaluation is launched here. This is a pretraining feasibility pilot, not evidence of MI localization improvement. Subsequent transfer must explicitly address the unresolved provenance limitation before a clean patient-independent claim.

Next stage checks selected waveform shapes, units, finite values and exact duplicates, converts 12-lead ECG to Kors XYZ, resamples to 1000 Hz, applies the existing denoising/detector, and keeps one median eligible cached crop per retained source patient. Any source QC exclusions are counted; target PTB patients/beats/labels remain unchanged.

Sources: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), [PTB Diagnostic](https://physionet.org/content/ptbdb/1.0.0/), [Kors coefficient implementation](https://github.com/Tereshchenkolab/Pacing_spike_removal_v2/blob/main/kors.m). Selected patient/record identities remain local in `review/selected_cohort.csv`.
