# Frozen baseline score correction results

The four corrections reproduced the original ensemble’s **0.584266** six-class patient macro F1 and its predicted class for every one of the 164 patients. All three seeds and ten patient folds contributed. A probability comparison across the saved outer predictions found the following differences. [Protocol](distribution_score_correction_plan.md) · [Validated report](../results/distribution_score_correction_v1/complete/REPORT.md) · [Aggregate results](../results/distribution_score_correction_v1/complete/summary.json)

| Procedure | Ensemble macro F1 | Mean member macro F1 | Correction enabled out of 30 fold/seed decisions | Maximum absolute ensemble probability change |
|---|---:|---:|---:|---:|
| Frozen convolution baseline | 0.584266 | 0.572860 | — | — |
| Convolution mean/spread | 0.584266 | 0.572860 | 6 | 0.003214 |
| Convolution quantiles | 0.584266 | 0.573222 | 6 | 0.003799 |
| Learned mean/spread | 0.584266 | 0.569224 | 9 | 0.009817 |
| Learned quantiles | 0.584266 | 0.569224 | 9 | 0.008669 |

The primary learned quantile contrast was **0.000000**. Identical ensemble labels force the fixed-prediction paired bootstrap difference to zero on every resample. This describes the observed decision tie on PTB development patients. Convolution quantiles changed one member prediction in seed 123 and increased that member’s F1 from 0.572111 to 0.573195. Each learned branch changed one member prediction in seed 2026 and reduced that member’s F1 from 0.592695 to 0.581786. Ensemble averaging kept all five procedures at the same class decisions and per-class F1.

The original model was frozen to isolate the contribution of the added summary. The conservative candidate grid and the use of in-sample baseline residuals limit the correction this study can learn. Inner validation chose the off option in 21 or 24 of 30 decisions per branch. Even when enabled, the largest ensemble probability shift from any branch was under one percentage point. This experiment supports keeping the original convolution baseline as the simpler reference. It tested the declared additive correction procedure on an inspected development cohort.

The independent pass reconstructed 120 contexts, restored 210 baseline models, replayed 1,440 active inner and 30 active outer correction fits, and matched the original baseline metrics exactly. A separate read-only check matched all 120 audit/data hashes and recomputed the label and probability differences above. The real inner-only smoke restored both source baseline candidates and 16 active correction candidates. The full macOS suite passed all 323 tests. The follow-up used the original cached measured Frank XYZ features and inherited the learned encoders’ original inner-validation stopping policy.
