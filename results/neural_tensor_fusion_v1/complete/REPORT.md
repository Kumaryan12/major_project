# Patient-nested neural–tensor probability fusion

Neural–tensor late-fusion development on repeatedly inspected PTB patients; scalar blending selected inside outer-training patients, frozen component recipes, no independent confirmation or established SOTA.

All 164 original patients. No new model fits. Verified 90 inner neural members and 120 inner tensor/PFN contexts; restored 30 outer neural members and 10 outer PFN contexts. Source artifacts read only. Both frozen component references reproduced.

| Procedure | Patient macro F1 | MI-only macro F1 | Accuracy |
|---|---:|---:|---:|
| neural_only | 0.585 | 0.543 | 65.2% |
| tensor_only | 0.542 | 0.487 | 61.6% |
| equal_blend | 0.543 | 0.485 | 62.8% |
| selected_blend | 0.583 | 0.537 | 65.2% |

| Paired comparison | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |
|---|---:|---:|
| selected_blend_minus_neural_only | -0.002 [-0.051, +0.050] | -0.006 [-0.065, +0.055] |
| equal_blend_minus_neural_only | -0.042 [-0.111, +0.028] | -0.058 [-0.139, +0.025] |
| selected_blend_minus_tensor_only | +0.041 [-0.055, +0.135] | +0.050 [-0.061, +0.162] |
| selected_blend_minus_equal_blend | +0.041 [-0.008, +0.093] | +0.052 [-0.004, +0.114] |
| equal_blend_minus_tensor_only | +0.000 [-0.080, +0.078] | -0.001 [-0.097, +0.092] |

## Training-only choices

| Outer fold | Neural weight | Tensor family | Feature budget |
|---:|---:|---|---:|
| 1 | 0.50 | combined | 64 |
| 2 | 0.75 | morphology | 24 |
| 3 | 1.00 | combined | 64 |
| 4 | 0.75 | morphology | 24 |
| 5 | 0.75 | morphology | 64 |
| 6 | 0.75 | morphology | 64 |
| 7 | 0.50 | combined | 64 |
| 8 | 0.75 | morphology | 64 |
| 9 | 0.75 | morphology | 64 |
| 10 | 1.00 | morphology | 24 |

## Per-class F1

| Class | neural_only | tensor_only | equal_blend | selected_blend |
|---|---:|---:|---:|---:|
| AMI | 0.294 | 0.357 | 0.294 | 0.353 |
| ALMI | 0.471 | 0.452 | 0.357 | 0.467 |
| ASMI | 0.640 | 0.500 | 0.561 | 0.571 |
| IMI | 0.596 | 0.517 | 0.566 | 0.604 |
| ILMI | 0.638 | 0.558 | 0.596 | 0.625 |
| HC | 0.868 | 0.870 | 0.881 | 0.879 |

## Exact protocol and limits

The neural recipe is frozen ordered_context_aux, averaged equally across every declared optimization seed. The tensor recipe is pinned local raw-probability TabPFN-v2 on bootstrap-selected morphology/combined features. Its original family/budget choice is reconstructed from three inner scores, with original stable ties. The fusion then selects one scalar neural weight from [1, .75, .5, .25, 0] by mean inner patient macro F1; ties favor the first weight. This sequential inner selection reuses inner labels for component and blend selection, but never outer-query labels. Inner scores are optimized diagnostics, not final performance estimates.

Every neural seed and tensor candidate uses the exact same inner train/validation patient identities within each outer-training set. All original patients, six labels, recordings, cached R positions and frozen folds are unchanged. The globally pooled outer prediction CSVs are used only to verify frozen reference reproduction; they never train/select the combiner. Both base models and feature selectors excluded the outer test patients. Missing or incompatible source caches stop this run; no hidden refits or disabled integrity checks.

A weight of one reproduces the neural reference; zero reproduces the tensor reference. Equal blending is a fixed control. The four reported procedures share component predictions; selecting the best outer score as the winning method would be post-hoc selection, not confirmation. No class-specific routing, learned uncertainty gate, temperature fit, seed subset search or diagnosis-based filtering is performed.

The independent report reconstructs all blend decisions and source selector/stopping audits and repeats restored branch inference. 70 tensor bootstrap caches were replayed. TabPFN restore/predict runs in isolated workers using the unchanged pinned checkpoint/settings; no TabPFN/LightGBM import is introduced into the feature-selection process. Private source/model/patient files remain ignored.

Intervals use 5,000 paired class-stratified patient draws conditional on fixed predictions, without refits or adjustment for overlapping cross-validation training sets, repeated development or multiple comparisons. Ensembles produce a single F1, not a seed standard deviation. MI-only scores retain all six-way predictions and MI→HC errors.

The neural architecture/ensemble and the fusion hypothesis were developed on repeatedly inspected PTB patients. Correct nesting prevents the specific outer-label weight-fitting error; it does not turn these patients into an untouched confirmation cohort. Report modest/negative results fully. Independent label-compatible patients are needed for generalization or SOTA claims; no guaranteed 0.80–0.90 score is established.

Reproduce: `.venv/bin/python -m mi_localization.neural_tensor_fusion`, then `.venv/bin/python -m mi_localization.neural_tensor_fusion_report`. These commands require original local banks, probability checkpoints and fitted contexts; a Git clone lacks the ignored fitted artifacts.
