# Native measured XYZ pretraining

All 164 patients, 391 recordings and 55,863 original beats. Replayed all 90 models and 30 matched fold/seed groups.

| Arm | Mean member macro F1 ± seed SD | Equal-seed ensemble macro F1 |
|---|---:|---:|
| scratch | 0.541465 ± 0.052231 | 0.601801 |
| target_only | 0.472359 ± 0.031956 | 0.454298 |
| target_plus_native | 0.495450 ± 0.042546 | 0.518364 |

The primary contrast is target_plus_native versus target_only.

| Comparison | Ensemble difference | Paired patient 95% interval |
|---|---:|---:|
| target_plus_native_minus_target_only | +0.064066 | [-0.003292, +0.133664] |
| target_plus_native_minus_scratch | -0.083436 | [-0.160972, -0.007201] |
| target_only_minus_scratch | -0.147503 | [-0.232981, -0.062013] |

| Class | scratch | target_only | target_plus_native |
|---|---:|---:|---:|
| AMI | 0.529412 | 0.333333 | 0.444444 |
| ALMI | 0.428571 | 0.421053 | 0.375000 |
| ASMI | 0.642857 | 0.318182 | 0.431373 |
| IMI | 0.535714 | 0.375000 | 0.436364 |
| ILMI | 0.595745 | 0.431373 | 0.558140 |
| HC | 0.878505 | 0.846847 | 0.864865 |

Target fitting used a fixed 8-epoch budget, with no validation monitoring or target hyperparameter selection. All three declared seeds and ten patient folds contribute to each score. The scratch control uses the same classification schedule. Pretrained arms add the same number of pretraining steps. Exact-partition scaling preserves signed physical XYZ orientation. Independent replay restores the encoder and final model checkpoints, checks sample plans and normalizers, and reproduces held-out probabilities.

These results concern the inspected PTB development cohort. Conditional bootstrap intervals use class-stratified patient resampling of fixed predictions. They do not include refitting or adjust for shared training folds and repeated development.

The additional patients supply measured XYZ waveforms without using their diagnostic labels. The native source audit documents recording checks and their scope.
