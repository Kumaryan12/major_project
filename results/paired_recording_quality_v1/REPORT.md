# Paired measured-XYZ signal quality and timing audit

Replayed all 30 saved prediction blocks, 79 patients and 256 matched beat draws. Raw decoding reproduced cached denoised beats in all 158 paired records. Zero new fits.

| Measure | Earliest median | Later median | Mean later − earliest [95% interval] | Later ↑ / ↓ / equal |
|---|---:|---:|---:|---:|
| drift_mv | 0.05786 | 0.04123 | -0.03120 [-0.06724, -0.00041] | 38 / 41 / 0 |
| adc_rail_fraction | 0.00000 | 0.00000 | +0.00000 [+0.00000, +0.00000] | 0 / 0 / 79 |
| r_disagreement_fraction | 0.00000 | 0.00000 | +0.00100 [-0.00117, +0.00331] | 19 / 15 / 45 |
| relative_denoising_change | 0.03911 | 0.03732 | -0.00120 [-0.00384, +0.00150] | 38 / 41 / 0 |
| unusual_rr_fraction | 0.00000 | 0.00000 | +0.00835 [+0.00020, +0.01870] | 25 / 15 / 39 |

Records with at least one usable ADC rail sample. Earliest 0 of 79, later 0 of 79.

| Association with later-minus-earliest outcome | Patients | Spearman rho | 95% interval |
|---|---:|---:|---:|
| decrease_drift_mv / correctness | 79 | -0.022 | [-0.235, +0.185] |
| decrease_adc_rail_fraction / correctness | 79 | Unavailable | Unavailable |
| decrease_r_disagreement_fraction / correctness | 79 | +0.059 | [-0.175, +0.290] |
| decrease_relative_denoising_change / correctness | 79 | -0.197 | [-0.407, +0.024] |
| decrease_unusual_rr_fraction / correctness | 79 | -0.075 | [-0.277, +0.135] |
| days_between_recordings / correctness | 79 | -0.033 | [-0.229, +0.165] |
| days_from_documented_infarct_to_earliest / correctness | 74 | +0.071 | [-0.140, +0.290] |
| decrease_drift_mv / correct-class probability | 79 | +0.061 | [-0.177, +0.272] |
| decrease_r_disagreement_fraction / correct-class probability | 79 | +0.062 | [-0.169, +0.289] |

| Class | Patients | Later − earliest accuracy |
|---|---:|---:|
| AMI | 11 | +0.46946 |
| ALMI | 9 | -0.21441 |
| ASMI | 19 | +0.04667 |
| IMI | 22 | -0.06587 |
| ILMI | 13 | +0.29087 |
| HC | 5 | +0.20000 |

Overall later-minus-earliest accuracy was +0.09434. The fixed paired macro F1 comparison reproduced 0.461602 earliest and 0.557485 later.

The two primary associations are decreases in baseline drift and raw-detector disagreement with the change in held-out correctness. ADC rail hits detect exact digital extrema. Cached R disagreements and RR outliers are descriptive proxies. PTB headers give archived date fields, and marked infarct-date coverage is reported in the association table. Class-stratified patient intervals condition on the previously inspected cohort and frozen predictions.

Clinical review remains pending. [Protocol](../../docs/paired_recording_quality_plan.md)
