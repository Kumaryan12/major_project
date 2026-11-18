# Native measured XYZ with partial region mentions

Replayed 120 models and 30 matched fold/seed groups for all 164 target patients. Audited 14 outside-target patients and used 13 with at least one shared region mention.

| Procedure | Mean member macro F1 ± seed SD | Three-seed ensemble macro F1 |
|---|---:|---:|
| scratch | 0.541465 ± 0.052231 | 0.601801 |
| target_descriptors | 0.544872 ± 0.066458 | 0.592327 |
| shuffled_source | 0.550108 ± 0.056718 | 0.568947 |
| true_source | 0.545536 ± 0.038519 | 0.575887 |

The primary contrast is true versus shuffled source mentions.

| Contrast | Ensemble difference | Descriptive paired patient 95% interval |
|---|---:|---:|
| true_source_minus_shuffled_source | +0.006940 | [-0.042964, +0.058767] |
| true_source_minus_target_descriptors | -0.016440 | [-0.094532, +0.060940] |
| true_source_minus_scratch | -0.025914 | [-0.085288, +0.035006] |
| target_descriptors_minus_scratch | -0.009473 | [-0.074339, +0.057341] |
| shuffled_source_minus_target_descriptors | -0.023380 | [-0.088837, +0.040684] |

| Class | scratch | target_descriptors | shuffled_source | true_source |
|---|---:|---:|---:|---:|
| AMI | 0.529412 | 0.424242 | 0.457143 | 0.514286 |
| ALMI | 0.428571 | 0.500000 | 0.500000 | 0.466667 |
| ASMI | 0.642857 | 0.538462 | 0.555556 | 0.538462 |
| IMI | 0.535714 | 0.581818 | 0.481481 | 0.526316 |
| ILMI | 0.595745 | 0.612245 | 0.530612 | 0.541667 |
| HC | 0.878505 | 0.897196 | 0.888889 | 0.867925 |

The shuffle reassigned all patient code vectors and changed the observed tag set for 10–10 of the 13 patients per fold/seed group. It preserves exact per-update tag frequencies and excludes self-pairs. The repeated tag patterns limit how many vectors can differ after shuffling. Source loss only uses recorded positive mentions.

All procedures use the same fixed target recipe. Source and target auxiliary heads do not participate in held-out six-class inference. The source review confirmed the headers and measured waveform membership. The reported regions are mentions and have no new independent clinical adjudication.

The comparison uses repeatedly inspected PTB development patients. Paired conditionally stratified patient intervals describe fixed predictions, with no adjustments for refits, shared training folds, repeated development or multiple comparisons.
