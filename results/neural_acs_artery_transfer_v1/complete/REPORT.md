# ACS-ECG artery-supervised transfer to measured Frank XYZ

All 90 models across the ten frozen folds and three seeds restored. 164 patients, 391 recordings and 55863 beats in all arms.

| Arm | Mean over seeds | Three-seed ensemble macro F1 |
|---|---:|---:|
| scratch | 0.541465 | 0.601801 |
| omi | 0.542764 | 0.570389 |
| omi_artery | 0.540056 | 0.580730 |

Primary contrast is OMI plus artery versus OMI-only pretraining.

| Contrast | Ensemble difference | Descriptive paired patient 95% interval |
|---|---:|---|
| omi_artery_minus_omi | +0.010341 | [-0.066726, +0.085396] |
| omi_artery_minus_scratch | -0.021071 | [-0.095589, +0.052806] |
| omi_minus_scratch | -0.031412 | [-0.114314, +0.050475] |

| Class | Scratch F1 | OMI F1 | OMI + artery F1 |
|---|---:|---:|---:|
| AMI | 0.529412 | 0.529412 | 0.516129 |
| ALMI | 0.428571 | 0.540541 | 0.514286 |
| ASMI | 0.642857 | 0.391304 | 0.461538 |
| IMI | 0.535714 | 0.490566 | 0.500000 |
| ILMI | 0.595745 | 0.571429 | 0.583333 |
| HC | 0.878505 | 0.899083 | 0.909091 |

Historical PTB development references are 0.601801 for the original waveform and 0.603374 for selected linear convolution.

Both source arms use identical ACS-ECG inputs, initialization and training exposures. Only the encoder weights transfer. Each target arm uses the same new six-class head, exact target-train normalization and fixed eight-epoch budget. Source artery labels stay in ACS-ECG, and the target receives measured XYZ alone at inference.

PTB is repeatedly studied development data. These conditional intervals describe predictions from two fixed source fits and three target seeds. The recording-copy screen describes waveform overlap evidence. Cross-dataset patient identity remains unresolved.

[Protocol](../../../docs/acs_artery_transfer_plan.md)
