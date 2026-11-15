# Tensor refinement: nested final classifier results

Measured-VCG patient-level development on repeatedly inspected frozen PTB folds. Nested feature/parameter selection; no independent confirmation, clinical validation or SOTA.

All 164 patients, 391 records and 55,863 original beats; 10 frozen outer folds, 3 inner folds. Validated 2820 fit slots: 2100 new inner fits, 660 verified reused legacy probes, 60 fresh outer refits.

| Learner / procedure | Patient macro F1 | Accuracy |
|---|---:|---:|
| linear_statistics | 0.432 | 49.4% |
| linear_legacy | 0.498 | 57.9% |
| linear_refined | 0.425 | 50.0% |
| tree_statistics | 0.440 | 50.6% |
| tree_legacy | 0.420 | 50.6% |
| tree_refined | 0.452 | 53.7% |

Pooled out-of-fold patient scores, not mean fold F1. One row represents one patient; equal-record robust summaries differ from prior beat-probability aggregation. Linear and tree classifiers use a fixed seed, not a three-seed neural ensemble. All predefined arms are retained.

## Paired comparisons

| Difference | Δ macro F1 | Descriptive patient interval |
|---|---:|---:|
| linear_legacy_minus_linear_statistics | +0.066 | [-0.031, +0.165] |
| linear_refined_minus_linear_statistics | -0.007 | [-0.093, +0.083] |
| linear_refined_minus_linear_legacy | -0.073 | [-0.144, -0.005] |
| tree_legacy_minus_tree_statistics | -0.019 | [-0.104, +0.068] |
| tree_refined_minus_tree_statistics | +0.012 | [-0.074, +0.106] |
| tree_refined_minus_tree_legacy | +0.032 | [-0.038, +0.103] |

5,000 class-stratified paired patient draws, conditional on fixed fitted predictions; no refits or adjustment for correlated folds, repeated development or multiple contrasts. These are descriptive intervals, not independent confirmation.

## Frozen selection and compression

statistics tunes only model regularization/depth using the original statistics pool without supervised selection. legacy selects among the original feature families and feature budgets. refined adds higher-rank global signed DCT, denoised-only controls and native-wavelet signed cores. All ranking/scaling/redundancy pruning is refitted inside each inner-training partition; family, budget and model parameter are selected by inner macro F1. Exact ties favor configured order. A fresh selector and classifier are then fitted on outer-training patients without test monitoring. No global union of selected feature names is used.

The complete input tensor is lossless within numerical tolerance; all compressed summaries are lossy. Native wavelet cores use a fixed DCT basis within each band's differently sampled coefficient-time grid, not learned Tucker/HOOI or clinically annotated phases. The [fidelity audit](../bank_audit/REPORT.md) measures reconstruction, not diagnosis. Raw/template/residual views do not create independent patients. Templates use the held-out recording's own unlabeled beats for offline within-exam context.

Linear model: L2 logistic regression, C choices [0.1, 1.0]. Tree model: depth choices [2, 3], 100 XGBoost trees, rate 0.05, lambda 1, full row/column sampling, one CPU thread. Feature counts [24, 64], correlation threshold 0.95. Class-balanced weights enter scaling and fitting once; ANOVA ranking itself is unweighted. All source patients/beats/labels/R positions are unchanged.

## Selected families

| Procedure | Outer-fold family counts |
|---|---|
| linear_statistics | {'statistics': 10} |
| linear_legacy | {'combined': 3, 'morphology': 7} |
| linear_refined | {'wavelet_core16': 2, 'denoised_dct64': 1, 'all_dct32': 3, 'denoised_dct32': 2, 'combined': 1, 'morphology': 1} |
| tree_statistics | {'statistics': 10} |
| tree_legacy | {'combined': 1, 'morphology': 9} |
| tree_refined | {'combined': 1, 'denoised_dct32': 2, 'all_dct32': 2, 'denoised_dct64': 2, 'wavelet_core32': 2, 'morphology': 1} |

## Per-class F1

| Class | linear_statistics | linear_legacy | linear_refined | tree_statistics | tree_legacy | tree_refined |
|---|---:|---:|---:|---:|---:|---:|
| AMI | 0.258 | 0.278 | 0.176 | 0.375 | 0.270 | 0.069 |
| ALMI | 0.312 | 0.343 | 0.258 | 0.438 | 0.320 | 0.412 |
| ASMI | 0.500 | 0.383 | 0.429 | 0.345 | 0.310 | 0.281 |
| IMI | 0.361 | 0.526 | 0.469 | 0.231 | 0.298 | 0.533 |
| ILMI | 0.435 | 0.583 | 0.435 | 0.435 | 0.491 | 0.591 |
| HC | 0.725 | 0.876 | 0.784 | 0.815 | 0.833 | 0.827 |

## Interpretation boundaries

No arm reached 0.80 macro F1.
The historical neural 0.550 ± 0.040 uses different models, aggregation and three training seeds; it is context, not a matched tensor-refinement control. Use the paired statistics/legacy controls here to judge the representation. Repeatedly studied PTB folds are development data. No favorable seed/patient removal, new clinical labels, automatic relabeling or external pretraining. No clinical validity, novelty or SOTA is established. A favorable result still needs new-cohort evaluation.

Reporting independently recomputes training-only selectors, validates every inner selection and verifies that all saved models reproduce their outer predictions. Individual feature arrays, models and patient predictions remain local; only aggregate results/audits are published.
