# VCG phase geometry: patient-level feature screen

VCG-only feature utility screen on repeatedly inspected PTB development patients; no independent confirmation, clinical boundary validation or SOTA.

Complete: 164 patients, 391 recordings, 55,863 unchanged beats and 6 source labels. All 10 frozen patient folds completed; 560 validated fits.

| Procedure | Patient macro F1 | Accuracy | Features | Δ vs same-learner statistics | Descriptive interval |
|---|---:|---:|---:|---:|---:|
| linear_statistics | 0.445 | 50.6% | 384 | +0.000 | — |
| linear_fixed_geometry | 0.448 | 53.0% | 156 | +0.003 | [-0.077, +0.083] |
| linear_adaptive_geometry | 0.467 | 54.9% | 156 | +0.021 | [-0.061, +0.107] |
| linear_statistics_geometry | 0.454 | 54.3% | 540 | +0.008 | [-0.051, +0.066] |
| tree_statistics | 0.409 | 50.0% | 384 | +0.000 | — |
| tree_fixed_geometry | 0.434 | 51.8% | 156 | +0.025 | [-0.074, +0.123] |
| tree_adaptive_geometry | 0.408 | 48.8% | 156 | -0.001 | [-0.095, +0.092] |
| tree_statistics_geometry | 0.461 | 53.0% | 540 | +0.052 | [-0.024, +0.131] |

5,000 class-stratified paired patient-bootstrap draws, conditional on fitted predictions. No refits, correction for correlated CV folds, repeated development or multiple comparisons. Every predeclared model is reported; there is no selected best test model or ensemble.

## Exact screen

Each patient contributes one row: medians and interquartile ranges over all of their original beats. The statistics control aggregates the existing 192 short-wavelet statistics into 384 columns. Fixed/adaptive geometry each aggregate 78 descriptors into 156 columns; the combined view has 540. This changes aggregation and classifier from the historical beat-trained ResNet: its 0.550 mean is context, not a matched comparator for these procedures.

Regularized multinomial logistic regression selects C from [0.01, 1.0]; shallow XGBoost selects depth from [2, 3] with 100 trees and rate 0.05. Each learner/view receives the same two-candidate, 3-inner-patient-fold budget. Mean inner macro F1 selects the candidate; ties follow configuration order. A fresh outer-training refit never uses test performance for selection. Class weighting is applied once; scaling uses weighted training-patient moments only. Seed 42 is fixed; repeating deterministic learners under arbitrary seeds is not treated as new evidence.

Fixed geometry is the same dimensional descriptor family with R-relative phases. Adaptive geometry substitutes plausible multi-axis DWT phase estimates and falls back when missing/inconsistent. Both have the same allowed end ceiling; adaptive phases select different intervals. Boundary-success flags are not classifier inputs, but fallback-dependent durations can still encode detector behavior. The [boundary audit](../boundary_audit/REPORT.md) reports class-wise coverage and explicitly lacks expert boundary ground truth.

All feature extraction is per-record/per-patient without cohort-fitted templates or normalization. No ECG channels, external data, patient exclusions, relabeling, R-peak changes, neural geometry branch, learned tensor contraction or shared-territory auxiliary loss are included in this first utility screen. Future neural work requires a separately controlled experiment; these results cannot stand in for its evaluation.

## Adaptive versus fixed boundaries

- linear: adaptive − fixed **+0.018**, descriptive interval [-0.045, +0.081].
- tree: adaptive − fixed **-0.026**, descriptive interval [-0.106, +0.057].

## Class F1

| Class | linear_statistics | linear_fixed_geometry | linear_adaptive_geometry | linear_statistics_geometry | tree_statistics | tree_fixed_geometry | tree_adaptive_geometry | tree_statistics_geometry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMI | 0.308 | 0.244 | 0.316 | 0.211 | 0.133 | 0.375 | 0.343 | 0.286 |
| ALMI | 0.357 | 0.323 | 0.323 | 0.323 | 0.357 | 0.250 | 0.333 | 0.414 |
| ASMI | 0.440 | 0.302 | 0.353 | 0.458 | 0.436 | 0.393 | 0.364 | 0.500 |
| IMI | 0.316 | 0.415 | 0.456 | 0.370 | 0.355 | 0.438 | 0.312 | 0.419 |
| ILMI | 0.490 | 0.522 | 0.468 | 0.490 | 0.364 | 0.300 | 0.256 | 0.304 |
| HC | 0.762 | 0.885 | 0.885 | 0.870 | 0.807 | 0.846 | 0.838 | 0.840 |

The 0.80–0.90 target remains aspirational. An informative representation is not established by a plausible phase diagram; automatic boundaries are not clinically adjudicated, and negative results for these two learners do not prove geometry cannot help. Positive development results still need independent validation. Full-record processing is offline/noncausal.

Reproduce: `.venv/bin/python -m mi_localization.geometry_bank`, then `.venv/bin/python -m mi_localization.geometry_study`, then `.venv/bin/python -m mi_localization.geometry_study --report`. Patient feature arrays, individual predictions, tensors, boundary examples and fitted models stay local. Only aggregate reports and hashed audits are public.
