# R-relative convolution pooling on measured XYZ

Reconstructed 120 matched feature contexts, restored 210 frozen baseline models and replayed 630 augmented ridge fits. All 391 record templates from the original 55,863 beats contribute across 164 patients.

| Procedure | Mean seed macro F1 | Three-seed ensemble macro F1 |
|---|---:|---:|
| baseline | 0.572860 | 0.584266 |
| whole_beat | 0.552843 | 0.567233 |
| shuffled | 0.568117 | 0.577459 |
| aligned | 0.545994 | 0.577238 |
| historical_selector | 0.562847 | 0.603374 |

The primary comparison is aligned minus the equal-width whole-beat control.

| Comparison | Macro F1 difference | Descriptive paired patient 95% interval |
|---|---:|---|
| aligned_minus_whole_beat | +0.010004 | [-0.025707, +0.046669] |
| aligned_minus_shuffled | -0.000221 | [-0.041515, +0.039815] |
| aligned_minus_baseline | -0.007029 | [-0.041219, +0.022107] |
| aligned_minus_historical_selector | -0.026136 | [-0.060826, +0.002983] |
| whole_beat_minus_baseline | -0.017033 | [-0.050911, +0.012136] |

| Class | Baseline | Whole beat | Shuffled | Aligned | Historical selector |
|---|---:|---:|---:|---:|---:|
| AMI | 0.344828 | 0.296296 | 0.275862 | 0.322581 | 0.344828 |
| ALMI | 0.482759 | 0.451613 | 0.451613 | 0.444444 | 0.500000 |
| ASMI | 0.633333 | 0.655172 | 0.655172 | 0.612903 | 0.633333 |
| IMI | 0.542373 | 0.517241 | 0.566667 | 0.551724 | 0.612903 |
| ILMI | 0.577778 | 0.560000 | 0.590909 | 0.608696 | 0.604651 |
| HC | 0.924528 | 0.923077 | 0.924528 | 0.923077 | 0.924528 |

All augmented arms retain the exact 3,584-column convolution base and add 512 columns. The whole-beat control uses additional distinct short kernels and the same signed response summaries. The shuffled control permutes the same four window slots within each record. The original record templates and training-fitted convolution thresholds are reused. Each procedure standardizes its patient-level columns on training patients and selects ridge alpha in the inner folds.

Fixed windows are relative to the original R location. The cohort is repeatedly used PTB development data. Descriptive intervals condition on predictions and omit refits and repeated-development effects. The historical selector uses a different, previously declared recipe.

[Protocol](../../../docs/r_relative_convolution_plan.md)
