# Measured-XYZ localization with matched beat counts

Replayed 30 original held-out blocks. The dated paired-record cohort contains 79 patients. Median spacing is 4 days. Matched budgets are {50: 1, 64: 78} beats per patient with 256 draws. Zero new fits.

| Arm | Macro F1 of expected counts | Accuracy | Mean draw macro F1 ± SD |
|---|---:|---:|---:|
| earliest | 0.461602 | 0.472953 | 0.461468 ± 0.015483 |
| later | 0.557485 | 0.567296 | 0.557489 ± 0.012421 |
| half_from_each | 0.546457 | 0.555627 | 0.546262 ± 0.014309 |
| full_earliest | 0.458477 | 0.468354 | 0.458477 ± 0.000000 |
| full_pair | 0.532089 | 0.544304 | 0.532089 ± 0.000000 |

| Paired contrast | Macro F1 difference | Descriptive 95% interval |
|---|---:|---:|
| half_from_each_minus_earliest | +0.084854 | [-0.017018, +0.192643] |
| later_minus_earliest | +0.095882 | [-0.037926, +0.230258] |
| half_from_each_minus_later | -0.011028 | [-0.086030, +0.062867] |
| full_pair_minus_half_from_each | -0.014368 | [-0.039352, +0.005435] |
| full_earliest_minus_earliest | -0.003126 | [-0.022507, +0.014704] |

| Class | Patients | Earliest F1 | Later F1 | Half from each F1 |
|---|---:|---:|---:|---:|
| AMI | 11 | 0.206 | 0.533 | 0.479 |
| ALMI | 9 | 0.367 | 0.388 | 0.236 |
| ASMI | 19 | 0.450 | 0.526 | 0.504 |
| IMI | 22 | 0.601 | 0.591 | 0.652 |
| ILMI | 13 | 0.515 | 0.677 | 0.707 |
| HC | 5 | 0.630 | 0.629 | 0.700 |

| Arm | Member F1 in declared seed order |
|---|---|
| earliest | 0.460801, 0.477687, 0.459888 |
| later | 0.601364, 0.530247, 0.519548 |
| half_from_each | 0.552481, 0.505540, 0.528033 |
| full_earliest | 0.470474, 0.477534, 0.447743 |
| full_pair | 0.563675, 0.489688, 0.484989 |

The primary contrast is half from each minus earliest only. The full-pair and full-earliest references use their naturally available beat counts. Intervals condition on the frozen predictions and date-complete repeat-record subset. Differences can reflect changes in injury, treatment, recording quality or other factors across archived visits.

[Registered protocol](../../docs/matched_recording_beats_plan.md)
