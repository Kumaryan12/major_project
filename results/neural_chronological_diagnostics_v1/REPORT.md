# ResNet versus consecutive-beat diagnostics

Post-hoc descriptive comparison on repeatedly inspected PTB development patients. Different training procedures; no causal attribution, new fitted model, selected ensemble or independent confirmation.

All 164 original patients, all three declared seeds. 15 saved patient prediction sets checked against frozen folds and source scores. No training fits.

Raw ResNet beat checkpoints reproduce saved patient predictions. Sequence outer probability checkpoints reproduce saved patient predictions. 1200 corresponding waveform-normalization comparisons match exactly. Existing model-restoration verification is inherited from the completed study; this diagnostic does not restore neural weights again.

## Scores and class effects

| Procedure | Mean macro F1 ± seed SD |
|---|---:|
| resnet | 0.550 ± 0.040 |
| mean_rr | 0.485 ± 0.055 |
| ordered_no_rr | 0.509 ± 0.039 |
| ordered_rr | 0.524 ± 0.018 |
| shuffled_rr | 0.533 ± 0.013 |

| Class | ResNet | Ordered+RR | Shuffled+RR |
|---|---:|---:|---:|
| AMI | 0.373 | 0.355 | 0.393 |
| ALMI | 0.436 | 0.419 | 0.427 |
| ASMI | 0.530 | 0.528 | 0.536 |
| IMI | 0.548 | 0.471 | 0.458 |
| ILMI | 0.576 | 0.525 | 0.583 |
| HC | 0.837 | 0.847 | 0.799 |

## Paired errors

| Sequence arm | Seed | Both correct | ResNet only correct | Sequence only correct | Both wrong |
|---|---:|---:|---:|---:|---:|
| mean_rr | 42 | 62 | 42 | 21 | 39 |
| mean_rr | 123 | 82 | 20 | 20 | 42 |
| mean_rr | 2026 | 67 | 28 | 23 | 46 |
| ordered_no_rr | 42 | 74 | 30 | 16 | 44 |
| ordered_no_rr | 123 | 78 | 24 | 21 | 41 |
| ordered_no_rr | 2026 | 77 | 18 | 18 | 51 |
| ordered_rr | 42 | 79 | 25 | 22 | 38 |
| ordered_rr | 123 | 78 | 24 | 17 | 45 |
| ordered_rr | 2026 | 79 | 16 | 16 | 53 |
| shuffled_rr | 42 | 76 | 28 | 19 | 41 |
| shuffled_rr | 123 | 78 | 24 | 20 | 42 |
| shuffled_rr | 2026 | 72 | 23 | 24 | 45 |

Each row contains the same patients; rows/seeds are not independent cohorts. Both-wrong cases can have different incorrect labels. Correctness complementarity is descriptive and is not a realizable oracle routing or ensemble score.

| Arm | Wrong in every seed: ResNet / sequence / both | ResNet always correct, sequence always wrong | Sequence always correct, ResNet always wrong |
|---|---:|---:|---:|
| mean_rr | 36 / 32 / 16 | 2 | 2 |
| ordered_no_rr | 36 / 35 / 19 | 2 | 3 |
| ordered_rr | 36 / 40 / 25 | 1 | 2 |
| shuffled_rr | 36 / 33 / 22 | 0 | 4 |

## Aggregation sensitivity without refitting

| Seed | Original beat-weighted ResNet F1 | Equal-record ResNet F1 | Changed predictions |
|---|---:|---:|---:|
| 42 | 0.586 | 0.605 | 5 |
| 123 | 0.557 | 0.530 | 5 |
| 2026 | 0.507 | 0.515 | 1 |

Three-seed means: original **0.550108**, equal-record **0.549727**. This is a post-hoc inference sensitivity, not a newly nested training/evaluation procedure: original early stopping used beat-weighted predictions. Do not promote an aggregation winner using these outer outcomes. Equal-record weighting alone does not explain the average sequence-versus-ResNet gap.

## Training behavior

| Arm | Selected-length inner fits | Median best epoch | Best at epoch 25 | Median last inner loss | Median outer epochs | Median last outer loss |
|---|---:|---:|---:|---:|---:|---:|
| resnet | 90 | 4.0 | 0 | 0.039 | 3.5 | 0.336 |
| mean_rr | 90 | 8.0 | 0 | 0.146 | 8.0 | 0.647 |
| ordered_no_rr | 90 | 7.0 | 0 | 0.176 | 6.5 | 0.675 |
| ordered_rr | 90 | 8.0 | 0 | 0.170 | 7.0 | 0.618 |
| shuffled_rr | 90 | 7.5 | 0 | 0.171 | 7.0 | 0.676 |

Sequence inner rows here use only the length selected inside each outer-training partition (90 fits per arm), matching the reference's 90 inner fits. The JSON also separates all candidate histories by length. Training losses have different beat/window objectives and stochastic sampling; they are not a matched numerical measure of representation quality. Epoch curves have changing denominators after early stopping; the CSV includes n_fits. Best validation scores are optimized inner diagnostics, not project performance.

## Interpretation and next experiment

The paired order-control result remains the decisive chronology evidence: ordered+RR 0.524 versus shuffled+RR 0.533, no demonstrated ordering gain. Cross-procedure class/error differences do not isolate chronology's causal effect. Identical source beats, patient folds and waveform scaling rule out those specific mismatches; sampling, window labels, supervision, aggregation, encoder/head initialization and optimization still differ.

The sequence model compresses each 64-dimensional beat embedding through a 32-dimensional recurrent state (or projection) and classifies one window. The reference directly supervises every beat embedding. Higher sequence training loss and later selected epochs are compatible with a harder optimization problem, but do not establish destructive compression or justify a larger epoch budget by themselves. Check whether best epochs cluster at the cap before blaming truncation.

A focused follow-up hypothesis is to retain a direct morphology path and beat-level supervision, then add a small gated context residual. Evaluate a fresh matched beat-only reference, context without beat auxiliary loss, and context with the fixed auxiliary loss; retain a matched shuffled-context control. All folds, seeds, exposure, inner stopping and length selection must be specified before fitting. A pretrained/frozen encoder must come only from that exact training partition; an outer-trained checkpoint cannot be reused across its inner validation patients. This follow-up is proposed, not run or scored here.

Patient-specific review stays in ignored review/patient_comparison.csv. Do not remove difficult patients, relabel without independent adjudication, tune on outer errors, or report the diagnostic as an untouched test. No prediction averaging or outer-label-selected routing was performed.
