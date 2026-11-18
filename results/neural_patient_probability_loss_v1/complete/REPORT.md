# Beat and patient probability loss on measured XYZ

Restored all 90 models and audited 30 matched fold and seed groups. Test inference averages every original beat probability per patient for all 164 patients.

| Objective | Mean member macro F1 ± seed SD | Three-seed ensemble macro F1 |
|---|---:|---:|
| beat_ce | 0.444659 ± 0.006854 | 0.510134 |
| patient_ce | 0.442100 ± 0.030900 | 0.469596 |
| hybrid_ce | 0.448025 ± 0.016065 | 0.460082 |

Primary comparison is patient-level cross-entropy versus per-beat cross-entropy.

| Comparison | Ensemble difference | Descriptive paired patient 95% interval |
|---|---:|---|
| patient_ce_minus_beat_ce | -0.040538 | [-0.103712, +0.022126] |
| hybrid_ce_minus_beat_ce | -0.050052 | [-0.095853, -0.007721] |

| Class | beat_ce | patient_ce | hybrid_ce |
|---|---:|---:|---:|
| AMI | 0.421053 | 0.421053 | 0.315789 |
| ALMI | 0.500000 | 0.444444 | 0.500000 |
| ASMI | 0.400000 | 0.360000 | 0.355556 |
| IMI | 0.333333 | 0.339623 | 0.320000 |
| ILMI | 0.588235 | 0.425532 | 0.458333 |
| HC | 0.818182 | 0.826923 | 0.810811 |

Historical development references are 0.601801 from the waveform pilot and 0.603374 from selected linear convolution. The matched comparison uses grouped batches for every arm. The historical waveform pilot used individual beat draws.

All arms used the same signed XYZ waveforms, class- and patient-balanced eight-beat bags, training normalization, model initialization, optimizer, eight epochs, and all-beat inference. No epoch or objective weights were selected from these outer folds.

These PTB patients have been used repeatedly for development. The paired bootstrap describes uncertainty conditional on the saved predictions and does not account for refitting or repeated development.

[Protocol](../../../docs/patient_probability_loss_plan.md)
