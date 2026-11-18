# Frozen baseline score correction

All 164 patients, 391 recordings and 55,863 source beats are represented. All 120 partition/seed contexts replayed, with 210 baseline restores and 1440 inner + 30 outer active correction fits reproduced. The original baseline metrics reproduced exactly.

| Procedure | Mean member macro F1 ± seed SD | Three-seed ensemble macro F1 |
|---|---:|---:|
| baseline | 0.572860 ± 0.019471 | 0.584266 |
| convolution_spread | 0.572860 ± 0.019471 | 0.584266 |
| convolution_quantiles | 0.573222 ± 0.019460 | 0.584266 |
| learned_spread | 0.569224 ± 0.014227 | 0.584266 |
| learned_quantiles | 0.569224 ± 0.014227 | 0.584266 |

The primary comparison is learned quantiles against the frozen baseline.

| Paired comparison | Ensemble difference | Descriptive patient 95% interval |
|---|---:|---:|
| learned_quantiles_minus_baseline | +0.000000 | [+0.000000, +0.000000] |
| learned_spread_minus_baseline | +0.000000 | [+0.000000, +0.000000] |
| convolution_quantiles_minus_baseline | +0.000000 | [+0.000000, +0.000000] |
| convolution_spread_minus_baseline | +0.000000 | [+0.000000, +0.000000] |
| learned_quantiles_minus_learned_spread | +0.000000 | [+0.000000, +0.000000] |
| convolution_quantiles_minus_convolution_spread | +0.000000 | [+0.000000, +0.000000] |
| learned_quantiles_minus_convolution_quantiles | +0.000000 | [+0.000000, +0.000000] |

| Procedure | Correction off | Strong penalty | Weaker penalty |
|---|---:|---:|---:|
| convolution_spread | 24 | 1 | 5 |
| convolution_quantiles | 24 | 2 | 4 |
| learned_spread | 21 | 1 | 8 |
| learned_quantiles | 21 | 1 | 8 |

| Class | Baseline | Convolution spread | Convolution quantiles | Learned spread | Learned quantiles |
|---|---:|---:|---:|---:|---:|
| AMI | 0.344828 | 0.344828 | 0.344828 | 0.344828 | 0.344828 |
| ALMI | 0.482759 | 0.482759 | 0.482759 | 0.482759 | 0.482759 |
| ASMI | 0.633333 | 0.633333 | 0.633333 | 0.633333 | 0.633333 |
| IMI | 0.542373 | 0.542373 | 0.542373 | 0.542373 | 0.542373 |
| ILMI | 0.577778 | 0.577778 | 0.577778 | 0.577778 | 0.577778 |
| HC | 0.924528 | 0.924528 | 0.924528 | 0.924528 | 0.924528 |

The frozen baseline alpha and correction penalty are chosen with inner patient macro F1. Penalty ties favor the disabled branch and then the stronger penalty. The baseline uses its original coefficients. Each auxiliary branch learns a centered linear correction from the frozen baseline’s in-sample training residuals. Those residuals reflect the baseline’s degree of fit in the training partition. Softmax converts scores to probabilities for equal-seed averaging.

This is a cache-based follow-up. Numerical replay checked source context membership, cached feature hashes, matching baseline columns, baseline probabilities, saved correction coefficients and inner decisions. The upstream studies provide the waveform replay evidence. Inherited inner-validation early stopping of the frozen learned encoder still applies.

PTB remains an inspected development cohort. The paired intervals use shared, class-stratified patient resampling of fixed predictions. They omit refits and adjustments for overlapping folds, repeated development or multiple comparisons. Selection counts describe overlapping partition decisions.
