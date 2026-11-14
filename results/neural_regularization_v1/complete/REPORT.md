# Bounded regularization and augmentation: nested patient selection

164 unchanged patients, 391 records, 55,863 original denoised beats, 6 classes, 10 frozen patient folds and seeds [42, 123, 2026].

| Training seed | Fixed baseline macro F1 | Inner-selected macro F1 | Selected − baseline |
|---|---:|---:|---:|
| 42 | 0.586 | 0.555 | -0.031 |
| 123 | 0.557 | 0.579 | +0.022 |
| 2026 | 0.507 | 0.517 | +0.010 |

## Across all prespecified seeds

- baseline: **0.550 ± 0.040** (mean ± sample SD).
- selected: **0.550 ± 0.031** (mean ± sample SD).
- Mean selected − baseline difference: **+0.000**, descriptive patient-bootstrap interval [-0.040, +0.041].

5,000 paired class-stratified patient draws, shared across arms and fixed seeds. Repeated seed/patient predictions are not independent patients. No refits, seed resampling or correction for correlated training folds, repeated development or multiple comparisons.

## Four bounded presets

| Preset | Dropout | AdamW decay | Label smoothing | Common XYZ gain | Noise SD / training lead SD |
|---|---:|---:|---:|---|---:|
| baseline | 0.2 | 0.01 | 0.0 | 1.0–1.0 | 0.0 |
| regularized | 0.4 | 0.05 | 0.05 | 1.0–1.0 | 0.0 |
| augmented | 0.2 | 0.01 | 0.0 | 0.95–1.05 | 0.005 |
| combined | 0.4 | 0.05 | 0.05 | 0.95–1.05 | 0.005 |

## Nested selection and matching

Each outer training set evaluates exactly four presets on the same 3 inner patient folds. Highest arithmetic mean of inner best patient macro F1 selects the preset; exact ties follow configured order, starting with baseline. Its rounded-median inner selected epoch budget controls a fresh outer-training refit. All selected settings and scores are in `selection_summary.csv`.

The fixed baseline is freshly trained using the original trainer. Its inner fits also serve as the baseline candidate in the bounded search. If baseline wins selection, the same final model/predictions are reused exactly; this is not an additional independent fit. Otherwise both final models start from matching weights. Outer-test patients never enter normalization, stopping, candidate selection or augmentation.

All models use the same compact waveform-only ResNet, fixed optimizer learning rate, 8,192 weighted-random beat presentations/epoch, batch size 128, at most 25 epochs and patience 6. Shared-epoch row hashes, initialization hashes and training normalization match. The selected procedure uses extra tuning compute: 415 unique inner/final fits in total. This compares a fixed model with a bounded selection procedure; it does not isolate a single regularizer or augmentation effect.

Normalization is fitted to unaugmented training beats only, balancing classes and patients. Augmentation draws use a separate seeded CPU stream, shared between the augmented/combined presets. A single positive physical-unit gain per beat is shared by XYZ; low-level Gaussian noise is scaled by training-only lead SD. No per-beat renormalization is imposed. Original arrays are unchanged; each sampled training occurrence receives a fresh draw. Validation/test prediction never calls augmentation or test-time averaging over augmentations.

The gain bounds and noise strength are exploratory choices, not clinically validated label invariances. Shared gain preserves vector direction before noise, but amplitude can contain diagnostic information. These operations do not create new patients or confirm physiological equivalence. No lead flipping/rotation, time warping, segment masking, longer windows, ECG input, pretraining or new feature tensor is included.

## Selection frequencies

| Preset | Outer seed/fold selections |
|---|---:|
| baseline | 5 |
| regularized | 6 |
| augmented | 11 |
| combined | 8 |

These are repeated selections on the same patients, not independent clinical cases or proof that an individual preset is superior.

## Mean class-level F1 across seeds

| Class | Patients | Baseline | Selected |
|---|---:|---:|---:|
| AMI | 17 | 0.373 | 0.362 |
| ALMI | 16 | 0.436 | 0.451 |
| ASMI | 27 | 0.530 | 0.509 |
| IMI | 29 | 0.548 | 0.527 |
| ILMI | 23 | 0.576 | 0.599 |
| HC | 52 | 0.837 | 0.855 |

## Interpretation boundaries

Repeatedly inspected PTB patients remain development data. Neither candidate selection nor patient bootstrap makes these untouched confirmation. No best-seed selection or unplanned ensemble is used. This experiment does not establish novelty, SOTA, clinical validity, a dataset ceiling or a promised 0.80–0.90 macro F1. MPS is not guaranteed bitwise deterministic.

Existing experiments and PTB-XL tests are unchanged. Patient predictions, weights and waveform arrays stay local; only aggregate reports/metrics/audits are published. The full method and primary sources are in `docs/regularization_vcg_plan.md`.
