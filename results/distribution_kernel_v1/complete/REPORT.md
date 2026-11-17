# Linear and nonlinear distribution readouts

Nonlinear readout PTB development using already verified fixed feature caches; repeated cohort, no independent confirmation, clinical validity or established SOTA.

All 164 patients retained. Restored 2400 classifiers including 240 outer models; rebuilt 480 representation kernels from 120 independently verified source feature contexts; 480 matching checks and 840 original-ridge equivalence checks passed. No waveform re-extraction or new feature fitting.

| Procedure | Mean member F1 ± seed SD | Uniform seed ensemble F1 |
|---|---:|---:|
| baseline__linear | 0.572860 ± 0.019471 | 0.584266 |
| baseline__nonlinear | 0.584453 ± 0.018903 | 0.583987 |
| plus_mean__linear | 0.548247 ± 0.031406 | 0.581913 |
| plus_mean__nonlinear | 0.562123 ± 0.015353 | 0.599492 |
| plus_spread__linear | 0.538610 ± 0.027460 | 0.594098 |
| plus_spread__nonlinear | 0.563611 ± 0.012004 | 0.595859 |
| plus_quantiles__linear | 0.549509 ± 0.015852 | 0.589485 |
| plus_quantiles__nonlinear | 0.575470 ± 0.011041 | 0.592147 |
| selected_linear | 0.562847 ± 0.017794 | 0.603374 |
| selected_nonlinear | 0.573104 ± 0.013918 | 0.583933 |

## Prespecified paired comparisons

Primary: selected_nonlinear minus selected_linear. Both families use the SAME inner-selected representation in each fold/seed. This is not an outer-score winner.

| Comparison | Ensemble difference | Descriptive paired 95% interval |
|---|---:|---:|
| selected_nonlinear_minus_selected_linear | -0.019441 | [-0.048244, +0.005086] |
| baseline__nonlinear_minus_baseline__linear | -0.000279 | [-0.030905, +0.028433] |
| plus_mean__nonlinear_minus_plus_mean__linear | +0.017578 | [-0.013003, +0.049895] |
| plus_spread__nonlinear_minus_plus_spread__linear | +0.001761 | [-0.029312, +0.033004] |
| plus_quantiles__nonlinear_minus_plus_quantiles__linear | +0.002662 | [-0.026689, +0.032079] |
| plus_quantiles__nonlinear_minus_plus_spread__nonlinear | -0.003712 | [-0.021897, +0.009799] |

## Interpretation and scope

Each exact training partition supplies class-balanced weights, an unpenalized weighted intercept, Gaussian bandwidth and kernel energy scale. Kernel weights and ridge alpha use inner patients only. Feature representations, all beats, labels, R positions, folds and source preprocessing remain unchanged. The linear dual solution reproduces the original primal/dual RidgeClassifier probabilities within declared numerical tolerances and its labels/inner selections/metrics exactly. No new physical rotation, amplitude normalization or raw-signal encoding.

Nonlinear kernel is (1−lambda) XX^T + lambda * training_energy * exp(−squared_distance/(2*training_median_positive_squared_distance)). Each X is the ENTIRE verified feature vector for that representation, including the complete morphology base and any auxiliary summary. It tests nonlinear readout of fixed summaries and their interactions; it is not an exact sliced-Wasserstein distance, separate physiology kernel, full optimal-transport solver or import of theoretical guarantees. Degenerate training scale uses a declared fallback. Every kernel is checked for positive semidefiniteness and centered using training weights only.

Linear family has two alpha candidates; nonlinear has two alphas times two positive mixture weights. The nonlinear search has twice the candidate budget, which is disclosed, not treated as a pure architecture-only causal intervention. For each representation/family, highest mean inner macro F1 selects a candidate, with configured-order ties. The shared representation maximizes the average of the two family best inner scores; both outer readouts use that same chosen representation. All four fixed-representation paired comparisons are retained. All three seeds remain in equal probability ensembles, with mean member scores separate. Softmax is a fixed-temperature aggregation interface, not calibrated clinical risk.

The old cached feature study independently re-encoded all original beats before completion. This follow-up rechecks its pinned hashes, complete chain of source artifacts/record caches, memberships and restored source predictions. It rebuilds every new kernel and restores every new classifier; it does not rerun the expensive waveform feature extraction. Hash failures block rather than trigger a hidden source rebuild. Cache-only chain of custody is not a new independent raw-signal validation.

Source outcomes were already inspected before this development hypothesis. Patient-separated fitting does not make the cohort untouched. Conditional stratified patient intervals use shared draws and fixed predictions, with no correction for repeated development, overlapping training sets, multiple comparisons or fitting variability. Small gains do not establish independent generalization, clinical validity, novel physiology, SOTA or guaranteed 0.80–0.90 macro F1. The source spread ensemble .594098 and earlier neural ensemble .584570 remain labeled references, not freely substituted matched controls.

Research motivation: Meunier et al., Distribution Regression with Sliced Wasserstein Kernels, ICML 2022, https://proceedings.mlr.press/v162/meunier22b.html. Protocol: docs/distribution_kernel_plan.md. Source HYDRA attribution/license remain in src/mi_localization/_vendor/hydra. The runner publishes only verified aggregate outputs and then stops.
