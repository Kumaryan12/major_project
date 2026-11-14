# External pretraining, implemented step by step

## Scope of the current implementation

The learning curve showed higher held-out F1 with more PTB training patients, while training resubstitution scores stayed high. This motivates testing broader source-patient representations. It does not establish a required dataset size or a promised 0.80–0.90 F1.

This phase implements **a source-cohort audit, a derived-XYZ pilot bank, and source-only masked reconstruction**. It does not launch PTB fine-tuning/evaluation. Cross-dataset identity overlap is unresolved, so the encoder export carries `ptb_patient_independence_cleared: false`. This is a scientific provenance limitation, not an assertion that overlap has been found.

## Step 1: freeze source patients and provenance

PTB-XL v1.0.3 supplies 21,799 ECG recordings from 18,869 patients. Its metadata patient assignments are checked across all ten folds. The pilot selects 1,024 unique patients from folds 1–8 and 128 from fold 9 using seed 42, one lowest-ID ECG per patient. Patient selection ignores diagnostic labels, six-class mappings, metadata quality flags, target errors and target labels. The old strict six-class external cohort is not reused as the pretraining pool.

Fold 10 is excluded. Existing cached fold-10 files remain untouched. Both downloading and conversion independently reject rows outside their frozen role/fold before waveform access. Source validation patients never enter source normalization, template construction or optimizer updates. Patient/record identifiers stay local in `source_audit/review/selected_cohort.csv`; aggregate counts and fingerprints are public.

The [official PTB-XL documentation](https://physionet.org/content/ptb-xl/1.0.3/) describes Schiller ECG acquisition in 1989–1996. [PTB Diagnostic](https://physionet.org/content/ptbdb/1.0.0/) uses a PTB prototype recorder, includes measured Frank XYZ, and was collected at University Clinic Benjamin Franklin in Berlin. These acquisition descriptions and unrelated de-identified ID namespaces do **not** prove that the underlying people are disjoint. Cross-dataset identity linkage and comprehensive aligned recording-overlap screening remain unresolved. An exact-file duplicate screen cannot resolve those questions.

## Step 2: build a small three-channel source bank

Only the frozen 1,152 source recordings are requested. Each must have 5,000 samples at 500 Hz, 12 distinct leads and mV units. The existing Kors coefficients convert the eight independent ECG leads to XYZ. Polyphase resampling produces 1,000 Hz signals, then the existing full-record bior6.8 denoising and detector are applied. One median eligible −250/+400 ms crop is retained per accepted source patient, giving 3 × 651 input samples without time resizing.

Nonfinite/flat signals and no complete detected crop are counted as source exclusions. Malformed shape/rate/lead/unit inputs fail loudly. Exact source `.dat` duplicates across selected patient IDs quarantine every selected member of that duplicate group, including across source training/validation roles. This is a conservative exact data-file screen, not proof of physical-signal equality under different headers, near-duplicate detection or cross-dataset identity disjointness. No replacement patients are silently drawn after QC. Actual retained role counts are reported.

The bank hashes metadata, source selection, source files, conversion/denoising code, coefficients, preprocessing settings and runtime versions. Per-record conversion checkpoints allow compatible resumes. Completed cache contents and selected source files are verified before reuse. Source review plots show fixed record-order examples per role; they are layout/consistency aids, not independent detector or Frank-vector validation. Source XYZ is derived, not measured Frank VCG. Original PTB patients, beats, labels and folds remain unchanged.

## Step 3: source-only masked reconstruction pilot

The pilot reuses the existing ResNet encoder: widths 16/32/64, four temporal pooling bins and a 64-dimensional embedding. Its 77,808 encoder parameters match the encoder portion of the 78,198-parameter target classifier. A temporary linear decoder reconstructs 3 × 651 values from the embedding. The classifier head is unused and the decoder is discarded in the exported encoder. Strict loading into the existing target encoder is tested, without running a target classification experiment.

Six of 21 contiguous 31-sample time patches are masked jointly across XYZ: 28.57% of time positions. Training uses fresh independently seeded mask draws; source validation uses one fixed mask realization. Masking replaces standardized coordinates with zero (training lead means in physical units). Loss is mean squared error only over masked coordinates, so visible-coordinate copying does not directly reduce the objective.

Normalization is fit to unmasked source training patients only, weighting patients equally. No diagnostic labels are read. One selected crop per patient gives patient-uniform training rows. Each epoch samples 4,096 rows with replacement in batches of 128. AdamW uses learning rate 0.001 and weight decay 0.01, up to 20 epochs with patience 5 on fixed-mask source validation MSE. One seed (42) is used for this feasibility pilot.

Report the best model against a zero predictor, the training mean-beat template, and the trained model with all-zero input, all scored on the same held-out source patients/masks. The latter controls help identify reconstruction driven by an aligned mean waveform instead of useful input context. Even a lower reconstruction error does not establish transferable MI localization features. A source-only MSE is never reported as target macro F1.

Checkpoints preserve current and best weights, AdamW state, sampler/mask/loader generators and torch CPU/MPS RNG states. A CPU interruption/resume test verifies exact agreement with uninterrupted training. Full-size MPS masking/backward/checkpoint/export is also tested. MPS itself is not guaranteed bitwise deterministic. The exported encoder remains local and carries the unresolved identity limitation; no automatic target comparison is chained to this run.

The first pilot completed all 20 epochs, retaining all 1,024 training and 128 validation patients without QC exclusions or exact source-file duplicate groups. Best fixed-mask source validation MSE was **0.24910 at epoch 17**, compared with **0.62411** for the training mean-beat template, **0.96967** for the zero predictor, and **1.43111** for the trained model with all-zero input. This is about 60% lower error than the mean-template control on this validation realization. It demonstrates reconstruction feasibility, not target localization improvement. The exported encoder was checked against the best checkpoint and all exported weights were finite. There is no new PTB macro F1 from this phase.

## Commands and artifacts

```bash
# Freeze metadata roles/patient selection and write the provenance report.
.venv/bin/python -m mi_localization.pretraining_source
# Download selected source recordings, verify/convert, and freeze the bank.
.venv/bin/python -m mi_localization.pretraining_source --prepare
# Train/resume the bounded source-only masked pilot.
.venv/bin/python -m mi_localization.masked_pretraining
# Read-only status; no training, source download or target evaluation.
.venv/bin/python -m mi_localization.masked_pretraining --status
```

Configuration: [`configs/source_pretraining_vcg.yaml`](../configs/source_pretraining_vcg.yaml). Source reports: `results/neural_source_pretraining_v1/source_audit/COHORT_REPORT.md` and `BANK_REPORT.md`. Pilot report: `results/neural_source_pretraining_v1/pilot/REPORT.md`. Source recordings and converted arrays stay in ignored `data/` directories; patient/record manifests, example plots, fitted weights, transient logs and RNG checkpoints remain local. Only aggregate reports, curves and provenance are published.

## Subsequent transfer stage

Before a clean patient-independent PTB transfer claim, address the unresolved source-target overlap issue through stronger provenance/recording checks or an alternative source with suitable provenance. If identity cannot be resolved, a target transfer experiment must remain explicitly exploratory; different anonymized IDs cannot be treated as a clearance certificate. No such target experiment is part of this pilot.

Then declare a matched scratch-versus-pretrained comparison on measured PTB VCG, using the same three target seeds, fixed patient folds, all target patients/beats and equal inner-fold fine-tuning search budgets. Source pretraining is additional compute and must be reported. Target classifier heads should have matched fresh initialization; backbone initialization is the intended experimental difference. Target normalization and stopping remain training-patient-only. The previously evaluated PTB-XL fold 10 is not a fresh test and is not rescored.

Only after a transfer gain is established should a new tensor branch or paired measured/derived domain-alignment objective be tested as a separate declared ablation. This pilot does not establish that those changes help, that the project is SOTA, or that any particular target F1 is reachable.

Sources for existing conversion and broader context: [Kors coefficient implementation](https://github.com/Tereshchenkolab/Pacing_spike_removal_v2/blob/main/kors.m), [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), [PTB Diagnostic](https://physionet.org/content/ptbdb/1.0.0/), and [ECG-FM](https://github.com/bowang-lab/ECG-FM/). This small reconstruction pilot does not reproduce ECG-FM or borrow its checkpoints/results.
