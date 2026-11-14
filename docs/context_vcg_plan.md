# Longer-context measured VCG experiment

## Question and scope

Does adding measured signal context around the same cached R positions improve six-class patient-level localization with the existing ResNet? The prior [raw-versus-denoised comparison](../results/neural_preprocessing_v1/complete/REPORT.md) did not establish a benefit from removing denoising: mean macro F1 was 0.550 for denoised input and 0.521 for raw input. This experiment retains the existing denoising and changes window length only.

No phase-coverage improvement is assumed. The signal audit has no independently adjudicated R or P/QRS/ST/T annotations; interval screening flags are not detector errors or evidence that T waves were truncated. The proposed longer window is a fixed exploratory choice, not a clinically established optimum or a value chosen through an outer-fold window sweep.

## Fixed design

| Item | Both arms |
|---|---|
| Cohort | Same 164 patients, 391 records, 55,863 original beat positions |
| Classes | AMI, ALMI, ASMI, IMI, ILMI, HC |
| Signal | Measured Frank XYZ at 1 kHz; original full-record bior6.8 denoising |
| Short window | −250 through +400 ms, 651 samples |
| Longer window | −350 through +650 ms, 1,001 samples |
| Model | Original 78,198-parameter ResNet, widths 16/32/64, four temporal pooling bins |
| Training seeds | 42, 123, 2026 |
| Evaluation | Same frozen 10 outer patient folds and 3 inner patient partitions |
| Sampling | Class/patient-balanced weighted random sampling, 8,192 beat presentations/epoch |
| Optimization | AdamW, learning rate 0.001, weight decay 0.01, batch 128, dropout 0.2 |
| Stopping | Up to 25 epochs, patience 6 on inner patient macro F1 |
| Outer refit | Fresh initialization, rounded-median inner selected epoch budget |
| Patient prediction | Mean probabilities across every held-out beat |

Both arms are trained afresh within this experiment, using the existing waveform-only fitting function unchanged. Matching initialization and shared-epoch sampled-row hashes are checked during reporting. Each arm fits its own class/patient-weighted normalization using only its training partition and its full input window. There is no per-beat amplitude normalization or pooled raw/short/long normalization. Outer-test patients never enter scaling or stopping.

Adaptive temporal pooling accepts both lengths without resampling time or changing model parameters. The longer arm uses more samples, changes the relative location of the central beat in the pooling bins, and can include neighboring beats. Equal sampled beats, parameters and maximum epochs do not imply equal FLOPs or exactly equal stopping epochs. Any measured gain concerns this complete window change; it cannot be attributed to a specific cardiac phase.

## Boundaries and cache verification

The recording-header check identified 150 original beat positions whose longer crop crosses a recording boundary (about 0.27%). To preserve the full cohort, the declared policy is reflection padding only for missing samples, applied **after** original full-record denoising. Actual observed samples are unchanged. Reflected values are synthetic context, not measured physiology. Normalization includes those values; no padding mask or extra channel is introduced.

Every original short crop is regenerated and compared bitwise to the retained float32 cache. Every longer crop must contain exactly those same original samples at the appropriate offset. All original R positions, row order, patient IDs and labels are preserved; no beats are excluded or redetected. The [aggregate cache audit](../results/neural_context_v1/audit/REPORT.md) records actual padding counts by class and boundary side, affected patients, and maximum missing samples. Per-beat padding arrays and raw recordings remain local.

Source recording/header files, original beat table, denoising source, builder source, windows, boundary policy and control cache contents are fingerprinted. A changed source/cache refuses reuse. Training additionally records study/fitting/model source hashes, runtime, frozen folds, bank fingerprints and file-content hashes. Existing experiments are neither overwritten nor silently reused as the new control. Partial runs have separate outputs; compatible inner-selection and outer-fold checkpoints permit resuming.

## Commands

```bash
# Build and verify the longer-window bank; reuse a compatible finished cache.
.venv/bin/python -m mi_localization.context_bank
# Run the complete three-seed, two-window comparison.
.venv/bin/python -m mi_localization.context_study
# Validate completed predictions/audits and produce the paired report.
.venv/bin/python -m mi_localization.context_report
# Read-only progress; does not start or resume training.
.venv/bin/python -m mi_localization.context_study --status
```

Configuration: [`configs/context_vcg.yaml`](../configs/context_vcg.yaml). New data cache: `data/processed/neural_context_v1/`. Aggregate cache audit: `results/neural_context_v1/audit/`. Training and final report: `results/neural_context_v1/complete/`. The complete design has 240 fits: 3 seeds × 2 arms × 10 outer folds × (3 inner fits + 1 final refit). The status command shows completed folds and the latest saved time/PID/epoch; a saved timestamp alone does not establish that a process is alive. Failures do not automatically restart training. MPS is not guaranteed bitwise deterministic.

## Reporting and decision

Final reporting requires all folds and seeds. It verifies patient identities/truth/folds, recomputes macro F1 from saved predictions, checks initialization and sampled-row matching, and checks that stopping follows inner validation histories with no outer-test monitoring. Report all three seed pairs, mean/sample SD, class-level F1 and the mean long-minus-short difference.

A 5,000-draw class-stratified paired patient bootstrap uses the same patient draws across both arms and all fixed seeds. Do not flatten patient/seed repetitions into independent cases, select a best seed, or create an unplanned ensemble. Intervals condition on fitted predictions and fixed seeds; they do not correct shared training folds, repeated development or multiple comparisons.

These repeatedly inspected PTB patients remain development data. A benefit must be judged across the full comparison, not an individual outer-fold score. This step does not establish independent validation, a dataset ceiling, novelty, clinical validity or a promised 0.80–0.90 macro F1. ECG + VCG remains a separate extension; the previously used PTB-XL test is not rescored. Subsequent tuning or augmentation should remain separate ablations informed by this result.
