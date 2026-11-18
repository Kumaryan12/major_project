# Frozen learned morphology distributions

Frozen partition-local learned morphology distributions on repeatedly inspected PTB development patients; no independent confirmation, physiological identification, clinical validity or established SOTA.

All 164 patients, 391 recordings and 55,863 original beats retained. Re-encoded every original beat in all 120 exact teacher/feature contexts, restored 630 new and 210 source classifiers including 120 outer models. All 120 matched checks passed; 120 frozen teacher query predictions reproduced. No encoder training.

| Procedure | Mean member macro F1 ± seed SD | Uniform seed ensemble macro F1 |
|---|---:|---:|
| baseline | 0.572860 ± 0.019471 | 0.584266 |
| plus_mean | 0.532907 ± 0.017784 | 0.554973 |
| plus_spread | 0.539921 ± 0.042561 | 0.547867 |
| plus_quantiles | 0.543027 ± 0.042541 | 0.580895 |

## Prespecified contrasts

Primary: learned quantiles minus learned mean/spread. All comparisons retained; descriptive conditional patient intervals.

| Comparison | Ensemble difference | Paired patient 95% interval |
|---|---:|---:|
| plus_quantiles_minus_plus_spread | +0.033028 | [-0.011283, +0.079062] |
| plus_quantiles_minus_plus_mean | +0.025922 | [-0.020418, +0.074694] |
| plus_quantiles_minus_baseline | -0.003371 | [-0.060716, +0.052488] |
| plus_spread_minus_plus_mean | -0.007107 | [-0.032281, +0.017379] |
| plus_mean_minus_baseline | -0.029293 | [-0.081132, +0.019903] |

## Class F1

| Class | baseline | plus_mean | plus_spread | plus_quantiles |
|---|---:|---:|---:|---:|
| AMI | 0.344828 | 0.258065 | 0.275862 | 0.275862 |
| ALMI | 0.482759 | 0.400000 | 0.363636 | 0.451613 |
| ASMI | 0.633333 | 0.561404 | 0.535714 | 0.596491 |
| IMI | 0.542373 | 0.575758 | 0.580645 | 0.645161 |
| ILMI | 0.577778 | 0.650000 | 0.634146 | 0.619048 |
| HC | 0.924528 | 0.884615 | 0.897196 | 0.897196 |

## Method and limits

The complete frozen recording-median combined-convolution ridge baseline is retained, including its entire standardized 3,584-column vector. Three learned-summary arms add144columns; the base prefix remains bit-identical. Plain morphology teachers are fixed in advance from the completed measured-VCG staged study, using each exact inner/outer training partition and seed. Inner teachers were early-stopped using their original inner validation; this reused selection is disclosed, not an assertion of validation-independent encoders. Outer teachers used original inner-selected epoch counts and no outer-label monitoring. No source/seed winner selection. Every stored teacher state, scope and freshly reconstructed training normalizer is checked; CPU forward inference only, no retraining.

For every recording, encode all original beats and its own coordinate-median waveform into64learned features. Subtract the median embedding, divide by embedding-column scales fitted only from training-patient equal-record median embeddings, and project into16fixed feature-space directions. One training-only residual RMS uses equal patient, recording and within-record beat weights. Pool mean, mean/population-SD normal-quantile grids or empirical quantiles at q=.1,...,.9. Every grid is divided by sqrt(9) and receives the same fixed sqrt(base_dimension/16) factor; no second per-arm scaler. The moment basis does not assert Gaussian physiology. Equal-record patient pooling remains; feature ranks/capacities differ.

Class-balanced ridge with intercept/Cholesky, alpha[1,100], three inner patient folds and configured-order ties. All three declared feature/teacher seeds remain in uniform probability ensembles; member means/SD and paired differences remain separate. Teacher predictions from the new all-beat embeddings reproduce the saved query probabilities/labels within fixed CPU-versus-source MPS tolerances. Independent reporting re-encodes every waveform, checks every record, rebuilds training scales/projections/summaries, restores all classifiers and repeats selections. Source raw metadata/cache/checkpoint hashes are checked read-only; no old completed scope or report is edited.

This tests frozen learned residual summaries while preserving the full convolution base. It differs from earlier jointly trained16-beat attention bags and from the current fixed-convolution pooling; it is not a claim of a first-ever method, exact PSWE/FSW, physiological identification or guaranteed neural superiority. Supervised feature training adds historical compute; reusing its weights does not erase that cost. These64learned coordinates are not new sensors or patients, and feature-space directions do not rotate physical XYZ. Input normalization is training-only; signed/amplitude-bearing signal information and offline recording medians retain their declared interpretation.

Repeatedly inspected PTB patients are development data. Conditional class-stratified shared patient-bootstrap intervals use fixed predictions without refits or adjustment for repeated development/overlapping training folds/multiple comparisons. No best seed, arm, class routing, threshold or outer-feature-budget selection. More beats or latent features do not add independent target patients. No independent clinical confirmation, established SOTA, demonstrated ceiling or guaranteed0.80–0.90 score. Historical .603374 selected-linear, .594098 moment ensemble and .584570 neural ensemble remain separately labeled references, not interchangeable matched controls.

Protocol: docs/learned_distribution_plan.md. Distribution pooling motivation: https://proceedings.neurips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html. Source HYDRA attribution/license remain in src/mi_localization/_vendor/hydra. Runner publishes verified aggregates and stops; no next automatic experiment.
