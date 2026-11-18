# Later-recording distillation for earliest-record VCG

Verified 120 saved fits across 10 folds. All 30 student fold/seed groups matched.

| Cohort | Arm | Ensemble macro F1 | Member macro F1 in seed order |
|---|---|---:|---|
| dated (145) | supervised | 0.521335 | 0.499865, 0.522067, 0.500015 |
| dated (145) | paired | 0.523769 | 0.464800, 0.481742, 0.457547 |
| dated (145) | shuffled | 0.502072 | 0.454249, 0.467226, 0.477444 |
| repeat_record (79) | supervised | 0.496483 | 0.445642, 0.501574, 0.511748 |
| repeat_record (79) | paired | 0.517929 | 0.484120, 0.462843, 0.426630 |
| repeat_record (79) | shuffled | 0.507087 | 0.452017, 0.459047, 0.510982 |
| all_patients (164) | supervised | 0.532437 | 0.511795, 0.543314, 0.514097 |
| all_patients (164) | paired | 0.543398 | 0.460554, 0.507151, 0.458118 |
| all_patients (164) | shuffled | 0.517571 | 0.459764, 0.492464, 0.494089 |

| Cohort | Paired contrast | Macro F1 difference | Descriptive 95% interval |
|---|---|---:|---|
| dated | paired_minus_supervised | +0.002434 | [-0.079058, +0.085427] |
| dated | paired_minus_shuffled | +0.021697 | [-0.062848, +0.102840] |
| dated | shuffled_minus_supervised | -0.019263 | [-0.103917, +0.065264] |
| repeat_record | paired_minus_supervised | +0.021446 | [-0.075751, +0.116256] |
| repeat_record | paired_minus_shuffled | +0.010843 | [-0.094034, +0.111392] |
| repeat_record | shuffled_minus_supervised | +0.010603 | [-0.092534, +0.116160] |
| all_patients | paired_minus_supervised | +0.010961 | [-0.066913, +0.090713] |
| all_patients | paired_minus_shuffled | +0.025827 | [-0.056830, +0.109180] |
| all_patients | shuffled_minus_supervised | -0.014866 | [-0.099986, +0.072243] |

| Class on primary dated cohort | Patients | Supervised F1 | Paired F1 | Shuffled F1 |
|---|---:|---:|---:|---:|
| AMI | 15 | 0.3704 | 0.3448 | 0.4444 |
| ALMI | 14 | 0.4615 | 0.4615 | 0.3704 |
| ASMI | 25 | 0.4727 | 0.4444 | 0.3922 |
| IMI | 29 | 0.5263 | 0.4918 | 0.4762 |
| ILMI | 22 | 0.4500 | 0.5000 | 0.5000 |
| HC | 40 | 0.8471 | 0.9000 | 0.8293 |

Teachers used later recordings of outer-training patients and generated patient targets by cross-fitting. Students used the same waveforms, sampled rows and training budget. The dated cohort and repeat-record cohort use the unique earliest archived recording. The all-patient sensitivity includes deterministic fallbacks.

The primary comparisons are paired versus supervised and paired versus shuffled in the 145-patient dated cohort. Intervals are conditional on the saved held-out predictions, and PTB remains development data.

[Protocol](../../../docs/later_recording_distillation_plan.md)
