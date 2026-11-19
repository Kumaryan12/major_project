# Frozen patient readouts before and after ResNet compression

Restored 120 epoch-eight encoders and 420 ridge heads. All feature summaries and source query predictions replayed. Encoder updates were zero.

| Cohort | Readout | Ensemble patient macro F1 | Member F1 in seed order |
|---|---|---:|---|
| dated (145) | original | 0.521335 | 0.499865, 0.522067, 0.500015 |
| dated (145) | mean_logits | 0.517351 | 0.494209, 0.534066, 0.472201 |
| dated (145) | ridge64 | 0.535691 | 0.501370, 0.568980, 0.500507 |
| dated (145) | ridge256 | 0.501020 | 0.482017, 0.543771, 0.500044 |
| all_patients (164) | original | 0.532437 | 0.511795, 0.543314, 0.514097 |
| all_patients (164) | mean_logits | 0.529068 | 0.494368, 0.556513, 0.487379 |
| all_patients (164) | ridge64 | 0.546530 | 0.507333, 0.588467, 0.513662 |
| all_patients (164) | ridge256 | 0.525410 | 0.495912, 0.565292, 0.505005 |

| Cohort | Contrast | Macro F1 difference | Descriptive 95% interval |
|---|---|---:|---|
| dated | ridge256_minus_ridge64 | -0.034671 | [-0.094647, +0.021065] |
| dated | ridge256_minus_original | -0.020315 | [-0.081297, +0.037948] |
| dated | mean_logits_minus_original | -0.003984 | [-0.034376, +0.020518] |
| dated | ridge64_minus_original | +0.014356 | [-0.046053, +0.080129] |
| all_patients | ridge256_minus_ridge64 | -0.021119 | [-0.081172, +0.035292] |
| all_patients | ridge256_minus_original | -0.007027 | [-0.065580, +0.052820] |
| all_patients | mean_logits_minus_original | -0.003369 | [-0.029773, +0.018609] |
| all_patients | ridge64_minus_original | +0.014092 | [-0.041880, +0.076187] |

| Class in 145-patient primary cohort | Patients | Original | Mean logits | Ridge 64 | Ridge 256 |
|---|---:|---:|---:|---:|---:|
| AMI | 15 | 0.3704 | 0.3704 | 0.4286 | 0.2308 |
| ALMI | 14 | 0.4615 | 0.4167 | 0.4444 | 0.3704 |
| ASMI | 25 | 0.4727 | 0.5000 | 0.4815 | 0.4727 |
| IMI | 29 | 0.5263 | 0.5085 | 0.5263 | 0.5902 |
| ILMI | 22 | 0.4500 | 0.4615 | 0.5000 | 0.5128 |
| HC | 40 | 0.8471 | 0.8471 | 0.8333 | 0.8293 |

The two main contrasts test 256 versus 64 ridge features and 256 ridge versus the original probability average. The original head on the mean feature controls aggregation. Every head uses the encoder for its own fitting partition, fixed at eight epochs. Three transform-seed members receive equal probability weights. The intervals condition on fixed predictions for repeatedly studied PTB patients. The primary cohort uses earliest dated records.

[Protocol](../../../docs/frozen_patient_readout_plan.md)
