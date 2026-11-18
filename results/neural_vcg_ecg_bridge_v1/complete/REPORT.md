# Paired ECG-assisted training with XYZ-only inference

All 164 patients, 391 recordings and 55,863 original beats. Replayed all 90 models and 30 matched fold/seed groups.

| Arm | Mean member macro F1 ± seed SD | Equal-seed ensemble macro F1 |
|---|---:|---:|
| scratch | 0.541465 ± 0.052231 | 0.601801 |
| shuffled_ecg | 0.481652 ± 0.057172 | 0.529345 |
| paired_ecg | 0.544613 ± 0.020266 | 0.564293 |

The primary contrast is paired_ecg versus shuffled_ecg.

| Comparison | Ensemble difference | Paired patient 95% interval |
|---|---:|---:|
| paired_ecg_minus_shuffled_ecg | +0.034948 | [-0.051853, +0.121214] |
| paired_ecg_minus_scratch | -0.037508 | [-0.125846, +0.049101] |
| shuffled_ecg_minus_scratch | -0.072456 | [-0.142850, -0.000690] |

| Class | scratch | shuffled_ecg | paired_ecg |
|---|---:|---:|---:|
| AMI | 0.529412 | 0.352941 | 0.285714 |
| ALMI | 0.428571 | 0.400000 | 0.444444 |
| ASMI | 0.642857 | 0.481481 | 0.607143 |
| IMI | 0.535714 | 0.545455 | 0.566038 |
| ILMI | 0.595745 | 0.520000 | 0.583333 |
| HC | 0.878505 | 0.876190 | 0.899083 |

Target fitting used a fixed 8-epoch budget, with no validation monitoring or target hyperparameter selection. All three declared seeds and ten patient folds contribute to each score. The scratch control uses the same classification schedule. Pretrained arms add the same number of pretraining steps. Exact-partition scaling preserves signed physical XYZ orientation. Independent replay restores the encoder and final model checkpoints, checks sample plans and normalizers, and reproduces held-out probabilities.

These results concern the inspected PTB development cohort. Conditional bootstrap intervals use class-stratified patient resampling of fixed predictions. They do not include refitting or adjust for shared training folds and repeated development.

Only training patients contribute measured ECG targets. Within each matched batch, the shuffled arm uses the identical set of ECG descriptors with cross-patient pairings. Both arms use measured XYZ alone for classification and held-out inference.
