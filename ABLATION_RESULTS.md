# Patient-independent representation study

Run date: 2026-09-21  
Frozen patients and folds: `configs/patient_folds.csv`  
Primary endpoint: patient-level macro F1 across six MI/healthy classes

## Exploratory ablations

All 12 representations are generated from the same 60,507 detected beats and evaluated on the same 55,863 beats from 164 selected patients, 10 outer patient folds, patient-balanced training weights, and a fixed depth-6 XGBoost classifier. This table changes only the input representation.

| Representation | Features | Patient accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Wavelet statistics, no Tucker | 192 | **53.66%** | **0.464** | **0.808** |
| Tucker rank 5 | 120 | 48.17% | 0.423 | 0.773 |
| Tucker rank 8 | 192 | 49.39% | 0.416 | 0.786 |
| Paper Tucker rank 3 | 72 | 48.17% | 0.416 | 0.768 |
| Raw-signal Tucker rank 3 | 72 | 47.56% | 0.411 | 0.768 |
| Tucker rank 2 | 48 | 45.12% | 0.377 | 0.770 |
| Vector-RMS-normalized Tucker rank 3 | 72 | 43.29% | 0.369 | 0.770 |
| Z lead only | 24 | 42.68% | 0.363 | 0.756 |
| Tucker rank 1 | 24 | 44.51% | 0.353 | 0.770 |
| X lead only | 24 | 37.80% | 0.320 | 0.706 |
| Y lead only | 24 | 35.37% | 0.301 | 0.695 |
| No-wavelet Tucker rank 3 | 9 | 35.37% | 0.278 | 0.696 |

Wavelet statistics summarize each lead and each of the eight signal/subband slices using mean, standard deviation, RMS, peak-to-peak amplitude, mean absolute value, log energy, skewness, and kurtosis. Their 192 values retain explicit subband and lead morphology but avoid compressing time into a low-rank core.

The ablations suggest that (1) wavelet subbands are important, (2) information from all three orthogonal leads is useful, (3) rank 3 is better than ranks 1 and 2, but extra Tucker rank alone adds little, and (4) removing global beat amplitude did not help. Denoising had only a small effect in this experiment.

## Nested confirmation of the strongest candidate

The wavelet-statistics representation was then run through the **same nested 10-by-3 patient-level model-selection protocol** used for the established Tucker/XGBoost benchmark, using the same two XGBoost candidates. Both results are out-of-fold patient predictions for the same 164 patients.

| Representation | Patient accuracy | Balanced accuracy | Macro F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Tucker rank 3 + XGBoost | 46.95% | 40.00% | 0.403 | 0.770 | 0.399 |
| Wavelet statistics + XGBoost | **54.88%** | **48.38%** | **0.487** | **0.817** | **0.479** |
| Difference | **+7.93 points** | +8.38 points | **+0.084** | +0.046 | +0.080 |

On a paired, class-stratified patient bootstrap (10,000 resamples; seed 42), the accuracy difference has a 95% interval of **+1.22 to +14.63 percentage points** and the macro-F1 difference has an interval of **+0.003 to +0.165**. The new representation corrects 25 patients missed by the Tucker baseline while losing 12 that Tucker classified correctly.

| Class | Patients | Tucker F1 | Wavelet-statistics F1 |
|---|---:|---:|---:|
| AMI | 17 | 0.188 | **0.357** |
| ALMI | 16 | 0.400 | **0.471** |
| ASMI | 27 | **0.412** | 0.373 |
| IMI | 29 | 0.412 | **0.438** |
| ILMI | 23 | 0.263 | **0.450** |
| HC | 52 | 0.742 | **0.835** |

AMI and ILMI, the classes identified as weak in the previous phase, both improve materially. ASMI is the one class whose F1 falls, so the representation is not uniformly better.

## Interpretation and limitations

This is a meaningful within-PTB improvement under patient-independent splitting, but **not an independent validation**: the winning representation was selected after inspecting all 12 ablations on the same outer folds. The paired bootstrap quantifies sampling variability for these patients; it does not remove representation-selection bias or establish transportability to another dataset. The next confirmatory step is an external patient cohort or a newly locked, untouched test set, ideally with more patients per MI location.

The full ablation metrics and confusion matrices are in `results/ablation/`. Nested-confirmation metrics and the patient confusion matrix are in `results/wavelet_statistics_confirm/`. The machine-readable summary is `results/ablation/comparison.csv`.
