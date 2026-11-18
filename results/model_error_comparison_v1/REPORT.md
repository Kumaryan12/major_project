# Saved convolution and waveform prediction comparison

Rebuilt both source ensembles across 164 patients, 3 seeds and all ten original held-out folds from verified prediction files.

| Model | Patient macro F1 | Accuracy |
|---|---:|---:|
| convolution | 0.603374 | 68.3% |
| waveform | 0.601801 | 66.5% |
| equal_blend | 0.582209 | 65.9% |

| Contrast | Macro F1 difference | Descriptive paired patient 95% interval |
|---|---:|---:|
| waveform_minus_convolution | -0.001573 | [-0.085967, +0.080816] |
| equal_blend_minus_convolution | -0.021165 | [-0.087770, +0.043554] |
| equal_blend_minus_waveform | -0.019592 | [-0.074656, +0.034880] |

| True class | Patients | Both correct | Convolution only | Waveform only | Both wrong |
|---|---:|---:|---:|---:|---:|
| AMI | 17 | 4 | 1 | 5 | 7 |
| ALMI | 16 | 5 | 2 | 1 | 8 |
| ASMI | 27 | 16 | 3 | 2 | 6 |
| IMI | 29 | 12 | 7 | 3 | 7 |
| ILMI | 23 | 10 | 3 | 4 | 6 |
| HC | 52 | 46 | 3 | 1 | 2 |
| total | 164 | 93 | 19 | 16 | 36 |

Among the 36 shared errors, both predicted the same wrong label for 21 patients.

The equal blend gives each of the two seed-averaged probability vectors a fixed weight of one half. The two component procedures were chosen retrospectively from inspected PTB development results. Bootstrap intervals describe fixed predictions on these patients. This diagnostic fits no model, selects no mixture and uses no new waveform inputs. The per-class breakdown is descriptive.

[Comparison protocol](../../docs/model_error_comparison_plan.md)
