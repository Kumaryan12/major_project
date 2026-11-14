# Making better use of the existing PTB cohort

## Why this experiment

The plain measured-VCG ResNet currently averages 0.550 patient macro F1 over three training seeds (0.586, 0.557, 0.507). Learned patient tensors and matched contrastive experiments have not established a reproducible improvement. These results do not establish a dataset ceiling. Before adding complexity, test whether the retained preprocessing changes useful signal information.

The first experiment is deliberately limited to **raw versus retained wavelet-denoised measured XYZ**, with a fresh ResNet control. It tests preprocessing within our current model/protocol, not every filtering choice. The historical 0.550 mean is context, not the matched control.

## Completed consistency audit

The [aggregate audit](../results/neural_preprocessing_v1/audit/REPORT.md) covers all 164 patients, 391 recordings and 55,863 cached beats in the six-class cohort. Header-derived labels match cached labels; crops are complete; peaks are unique within records; and recomputing the original full-record denoising reproduces every retained float32 beat exactly. This verifies consistency, not independent correctness of clinical labels or detector positions.

Median centered relative signal changes range from 2.0% (HC) to 4.5% (ILMI). Among 55,472 consecutive cached intervals, 713 are below 300 ms and 176 exceed 2,000 ms; 1,157 place the next cached peak at or before the +400 ms endpoint. These are screening counts, not diagnosed false/missed peaks or T-wave truncation rates. Rhythm variation and detector behavior cannot be separated without independent annotations.

Twelve class-balanced, fixed-seed examples have raw/denoised XYZ plots with wider context, cached R positions and original crop boundaries. Their layout has been checked. The local adjudication form remains unfilled; no expert review or independent R/phase annotations have been supplied. No patients, beats or labels were removed, redetected or changed. Review plots and recording-level diagnostics stay local.

## Locked matched comparison

| Choice | Specification |
|---|---|
| Input arms | Existing denoised XYZ; new unfiltered measured XYZ |
| Cohort | Same 164 patients, 391 records, 55,863 beats and six classes |
| Crop | Same cached R positions, −250 through +400 ms, 651 samples at 1 kHz |
| Outer evaluation | Existing frozen 10 patient folds |
| Inner validation | Three patient folds, identical memberships in every arm/seed |
| Training seeds | 42, 123, 2026 |
| Model | Same plain 78,198-parameter ResNet, no feature fusion |
| Normalization | Separate class/patient-weighted moments fit only within each training partition |
| Sampling | Same class/patient-balanced weighted-random rows, 8,192 presentations per epoch |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.01, batch size 128 |
| Stopping | Up to 25 epochs, patience 6 on inner patient macro F1 |
| Outer refit | Fresh initialization, rounded-median inner selected epoch budget; no test monitoring |
| Endpoint | All held-out beats, mean beat probabilities per patient, six-class patient macro F1 |

Both arms start from matching initialization hashes; sampled-row hashes must match for shared epochs. Actual epoch counts may differ because each arm selects stopping from its own training/validation signals. Raw and denoised normalizers must differ when their training moments differ; forcing equal moments would introduce a different comparison. There is no per-beat amplitude normalization, new feature tensor, longer crop, ECG input, pretraining or augmentation in this experiment.

The raw cache fingerprints raw source files, clinical/header inputs, crop table, denoised cache content and audit source. A changed cache/source refuses reuse. Training additionally fingerprints configuration, folds, model/study source and runtime. Partial folds have separate outputs and cannot become a full report. Completed fold and inner-selection checkpoints support compatible resuming. Seeded MPS execution is not guaranteed bitwise deterministic.

The full design requires 240 fits: 3 seeds × 2 arms × 10 outer folds × (3 inner fits + 1 refit). Individual checkpoints do not constitute final improvement evidence.

## Reporting and interpretation

The report requires every fold and seed, recomputes metrics from aligned patient predictions, validates initialization/sampling and stopping audits, and presents all per-seed scores plus mean/sample SD. A class-stratified paired patient bootstrap shares resamples across arms and seeds. The difference is raw minus denoised; seed/patient repetitions are never treated as independent patients or combined into an unplanned ensemble.

These repeatedly inspected PTB folds remain development data. Patient-bootstrap intervals condition on fitted predictions and fixed seeds; they do not correct correlated training folds, repeated development or multiple comparisons. No independent validation, 0.80–0.90 performance, novelty or SOTA result is promised.

## Run and monitor

```bash
# Build/verify the raw bank and audit; completed compatible banks are reused.
.venv/bin/python -m mi_localization.signal_audit
# Train both arms over all frozen folds and seeds.
.venv/bin/python -m mi_localization.preprocessing_study
# Generate the paired report only after the full experiment completes.
.venv/bin/python -m mi_localization.preprocessing_report
# Read-only status: never starts or resumes training.
.venv/bin/python -m mi_localization.preprocessing_study --status
```

Configuration: [`configs/preprocessing_vcg.yaml`](../configs/preprocessing_vcg.yaml). Audit: `results/neural_preprocessing_v1/audit/`. Training and final report: `results/neural_preprocessing_v1/complete/`. The status command shows completed outer folds and the last saved epoch/time/PID; saved progress is not proof the process is alive. A caught failure is recorded locally, and failures do not automatically restart training.

## Follow-up decisions

1. Assess the full matched result across all three seeds and class-level errors. Prefer a repeatable gain; avoid selecting the best seed or changing the cohort to raise the score.
2. Review cached timing and crop coverage with the professor or an appropriately qualified reviewer. If timing problems are confirmed, evaluate any revised detection/window rule as a separate, documented experiment. Do not quietly drop difficult patients or tune annotations from test errors.
3. Test a small, declared set of longer-window or signal-preserving preprocessing alternatives if the result/review supports them. Refit every learned transform within inner training patients and retain explicit matched controls.
4. Then consider bounded inner-fold model/regularization tuning and physiologically justified augmentation, each as a distinct ablation. Do not launch a large untracked search across the outer folds.
5. If progress remains limited, investigate larger-data pretraining/transfer with patient and duplicate separation. Measured VCG remains primary; ECG + VCG is a separate extension. The already evaluated PTB-XL test is not fresh validation and is not rescored here.

This step implements and evaluates the first comparison. The later options remain conditional work, not completed experiments. Existing results and frozen folds remain intact.
