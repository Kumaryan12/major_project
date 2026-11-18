# Earliest-record measured-XYZ training duration

Replayed 120 trajectories and 360 model checkpoints.

Matched all 30 original eight-epoch probability blocks.

| Cohort | Arm | Ensemble macro F1 | Member macro F1 in seed order |
|---|---|---:|---|
| dated (145) | fixed_8 | 0.521335 | 0.499865, 0.522067, 0.500015 |
| dated (145) | fixed_16 | 0.552757 | 0.474031, 0.487289, 0.508832 |
| dated (145) | fixed_32 | 0.516732 | 0.465472, 0.463697, 0.489778 |
| dated (145) | inner_selected | 0.509127 | 0.453643, 0.531428, 0.451243 |
| all_patients (164) | fixed_8 | 0.532437 | 0.511795, 0.543314, 0.514097 |
| all_patients (164) | fixed_16 | 0.572113 | 0.489620, 0.501762, 0.517480 |
| all_patients (164) | fixed_32 | 0.537490 | 0.477514, 0.479314, 0.497391 |
| all_patients (164) | inner_selected | 0.530107 | 0.470615, 0.540880, 0.467785 |

| Cohort | Comparison against fixed 8 | Macro F1 difference | Descriptive 95% interval |
|---|---|---:|---|
| dated | fixed_16 | +0.031422 | [-0.039516, +0.100476] |
| dated | fixed_32 | -0.004603 | [-0.081105, +0.070988] |
| dated | inner_selected | -0.012208 | [-0.082185, +0.061027] |
| all_patients | fixed_16 | +0.039675 | [-0.027751, +0.109844] |
| all_patients | fixed_32 | +0.005053 | [-0.068179, +0.078670] |
| all_patients | inner_selected | -0.002330 | [-0.071684, +0.067974] |

| Primary cohort class | Patients | Fixed 8 F1 | Inner-selected F1 |
|---|---:|---:|---:|
| AMI | 15 | 0.3704 | 0.2500 |
| ALMI | 14 | 0.4615 | 0.4000 |
| ASMI | 25 | 0.4727 | 0.5357 |
| IMI | 29 | 0.5263 | 0.5357 |
| ILMI | 22 | 0.4500 | 0.5000 |
| HC | 40 | 0.8471 | 0.8333 |

| Epoch | Inner selections across 30 decisions | Mean outer training loss at checkpoint |
|---|---:|---:|
| 8 | 8 | 0.358157 |
| 16 | 8 | 0.104954 |
| 32 | 14 | 0.044369 |

Primary comparison uses the 145 dated patients. The fixed 16- and 32-epoch checkpoints are descriptive. Every displayed ensemble gives each of the three seeds equal probability weight. The paired intervals condition on saved PTB development predictions. Inner selections share overlapping training patients.

[Protocol](../../../docs/earliest_training_duration_plan.md)
