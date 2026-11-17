# Intact-base all-beat convolution distributions

Intact-base all-beat convolution-distribution PTB development; repeated patient cohort, no independent confirmation, physiological distribution model or established SOTA.

Retained 164 patients, 391 recordings and 55,863 original beats. Replayed all original beats in 120 exact training contexts, restored 630 new classifiers and 210 frozen baseline classifiers, including 120 outer models. All 120 matched checks passed. The baseline probabilities reproduce the frozen source exactly.

| Procedure | Mean member macro F1 ± seed SD | Uniform seed-ensemble macro F1 |
|---|---:|---:|
| baseline | 0.572860 ± 0.019471 | 0.584266 |
| plus_mean | 0.548247 ± 0.031406 | 0.581913 |
| plus_spread | 0.538610 ± 0.027460 | 0.594098 |
| plus_quantiles | 0.549509 ± 0.015852 | 0.589485 |

## Prespecified contrasts

Primary: plus_quantiles minus plus_spread. All comparisons retained; intervals are descriptive.

| Comparison | Ensemble difference | Paired patient 95% interval |
|---|---:|---:|
| plus_quantiles_minus_plus_spread | -0.004613 | [-0.013297, +0.000000] |
| plus_quantiles_minus_plus_mean | +0.007572 | [-0.008149, +0.026726] |
| plus_quantiles_minus_baseline | +0.005218 | [-0.023246, +0.033727] |
| plus_spread_minus_plus_mean | +0.012185 | [+0.000000, +0.030466] |
| plus_mean_minus_baseline | -0.002353 | [-0.025740, +0.019883] |

## Class F1

| Class | baseline | plus_mean | plus_spread | plus_quantiles |
|---|---:|---:|---:|---:|
| AMI | 0.344828 | 0.322581 | 0.322581 | 0.322581 |
| ALMI | 0.482759 | 0.500000 | 0.500000 | 0.482759 |
| ASMI | 0.633333 | 0.610169 | 0.610169 | 0.610169 |
| IMI | 0.542373 | 0.566667 | 0.612903 | 0.612903 |
| ILMI | 0.577778 | 0.577778 | 0.604651 | 0.604651 |
| HC | 0.924528 | 0.914286 | 0.914286 | 0.903846 |

## Method and limits

The complete frozen recording-median MultiRocket-style + experimental multivariate HYDRA ridge procedure is the matched baseline. Every augmented arm preserves its full standardized vector bit-for-bit. Frozen kernels, feature scales and exact patient partitions are reused read-only; HYDRA attribution and GPL license remain in third_party/hydra. Historical MultiRocket conventions differ from this project implementation. Source feature transforms are independently refitted on training templates; all original source classifiers are restored without refitting.

For every original beat, subtract its own recording-median convolution vector, divide by the source training-column scales, and apply 16 seeded unit-norm feature-space projections. This does not rotate physical XYZ or unit-normalize beat amplitudes. A common residual RMS uses only training patients, with equal patient, recording and within-record beat weights. The added grid represents mean only, mean plus population SD on fixed normal quantiles, or empirical quantiles. The normal basis does not assert Gaussian physiology. Nine quantiles are divided by sqrt(9); all auxiliary grids receive the same fixed sqrt(base_dimension/projection_count) factor. No per-arm scaler undoes it. Full base and auxiliary grids pool equally across recordings per patient; stored dimensions and effective ranks differ.

Class-balanced ridge with intercept/Cholesky receives the same two alpha candidates, selected by mean inner patient macro F1 and configured-order ties. All three declared seeds remain in uniform probability ensembles. Mean member results are separate. Fixed-temperature ridge softmax is an aggregation interface, not calibrated clinical risk. All fitted preprocessing stays in its exact training partition. Full-record medians make this offline inference.

The primary comparison tests quantile shape beyond mean and spread; additions versus the intact baseline test preservation of beat variation. No outer-based arm, budget, source, seed, class routing or threshold selection. The repeated PTB cohort is development data. Conditional class-stratified patient bootstrap intervals share draws, with no refits or correction for repeated development, overlapping training folds or multiple comparisons. More beats do not add independent patients. No independent confirmation, clinical validity, SOTA or guaranteed 0.80–0.90 macro F1 follows. Historical neural 0.584570 remains a separate reference.

This is a compact projected-quantile adaptation, not an exact implementation of FSW or PSWE and not a claim to their theoretical guarantees. No ECG, new patients, source pretraining, learned encoder or tensor decomposition is added. See docs/convolution_distribution_plan.md. After independent publication this runner stops; no further study is queued.
