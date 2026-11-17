# Paired linear and nonlinear readouts of verified distribution features

Authorized 17 November 2026 after the user accepted the bounded kernel follow-up. This is a new study: `distribution_kernel_*`, `configs/distribution_kernel_v1.yaml`, `results/distribution_kernel_v1`. Do not edit completed upstream numerical code/config/results. No further experiment is queued afterward.

## Question and controls

Does a nonlinear readout improve patient-level localization from the SAME verified feature representation? The preceding study gave baseline ensemble .584266, mean .581913, mean/spread .594098 and quantiles .589485; the favorable moment ensemble point was uncertain and its three individual members were each below their paired baseline. These observations do not establish the mechanism or a winning representation.

Retain all four source representations: original full morphology, plus mean, plus mean/spread, plus quantiles. Pair each with a linear and nonlinear kernel-ridge readout (eight fixed-representation arms). Every augmented source vector retains the full original 3,584 columns and its 144-column auxiliary grid; do not recompute, normalize per arm, compress or select feature columns.

The **primary comparison** is `selected_nonlinear minus selected_linear`, with one SHARED representation selected inside each outer-training partition/seed. For each representation and family, select its best candidate by mean three-inner-fold macro F1. Average the two families' best inner scores for that representation, and select the highest shared score (configured representation-order ties). Both outer readouts then use the SAME selected representation, with their own training-only selected candidate. All eight fixed arms remain reported; the two selected procedures reuse their already fitted outer models. There is no outer-score representation winner or global OOF combiner.

Secondary paired contrasts: nonlinear−linear separately for every representation, and nonlinear quantiles−nonlinear mean/spread. This new readout question and primary are declared before new full fitting; they do not retroactively replace the source study's quantile−spread primary. Inner selection is symmetric across families, while nonlinear search has twice the candidate count and its extra flexibility is disclosed.

## Data and upstream cache custody

All 164 patients, 391 records, 55,863 original beats, six labels and `configs/patient_folds.csv` remain unchanged. Measured Frank XYZ primary; no new source data, ECG, filtering/relabeling, arbitrary rotation, beat unit-amplitude normalization, waveform encoder or learned tensor basis. The original offline/noncausal recording aggregation remains.

Use only independently completed `results/convolution_distribution_v1/complete` (`0a861b9`, implementation `2dfd577`) with exact source fingerprint/summary hash in the new config. The existing upstream report already re-encoded every original beat in all 120 contexts and restored all 840 source-study classifiers. Reuse its 120 exact-inner/outer cached feature arrays; never reuse an outer-trained transform inside its inner validation.

New read-only checks verify source config, runtime, code, summary/report/verification hashes, all 960 fit/reference/context audits and 1,500 fitted/data artifacts, protected original convolution artifacts, every recording audit/data pair, original labels/groups/chronology/R-position metadata, beats/paper configs, folds and all source selections. The verified file-map digest and count are pinned in the new scope. Initial probe checked 90,427 files in 13.4 seconds; the final protocol additionally checks the original beats and paper config (90,429 files). No upstream progress/report is rewritten, and no mismatch silently triggers a rebuild.

Source query memberships/features and restored source-ridge predictions are rechecked during linear equivalence. Source cache hashes are checked again at new independent reporting. This is transitive cache integrity plus upstream completed numerical replay, not a new independent waveform validation. New verification rebuilds every kernel and re-infers every classifier; it does NOT repeat convolution extraction from raw beats.

## Kernel and weighted-intercept mathematics

Let X be the ENTIRE fixed patient-feature vector for one representation, fitted/scaled previously within this exact training partition. For six classes set training sample weights `w_i = N / (6 * count(y_i))`, normalized weights `r_i = w_i / sum(w)`, and signed one-versus-rest targets `Y_ic = +1` for class c and −1 otherwise.

Subtract the weighted training feature mean for numerical stability. The unnormalized linear kernel is `K_L = X X^T`; no division by feature count changes the original ridge penalty. Training-only energy `tau = sum_i r_i ||X_i||^2`. Gaussian bandwidth squared `s2` is the median strictly positive training-pair squared Euclidean distance times the fixed multiplier 1.0. Values below the declared 1e-12 floor use a scale-one fallback; query signals/labels never determine either scale.

`K_lambda = (1-lambda) K_L + lambda * tau * exp(-||X_i-X_j||^2 / (2*s2))`.

- Linear family: lambda=0, alphas [1,100].
- Nonlinear family: lambda [0.25,0.75] × alphas [1,100], four candidates.

This uses whole-vector similarity, including the full morphology features and any pooled residual grid. It can test nonlinear interactions in that fixed representation. It is NOT an exact sliced-Wasserstein kernel on raw distributions, a full optimal-transport solver, separate proven physiology kernel or an import of published theoretical guarantees. Finite projections/quantiles and record-averaged summaries retain their source limitations. The kernel cannot recover information already removed by the source representation.

Both constituent kernels are positive semidefinite; each constructed training matrix is checked numerically. Class-weighted kernel centering supplies an **unpenalized intercept**, rather than silently changing to no-intercept kernel ridge. With `K_c` the training-weight-centered matrix, `Y_c = Y - r^T Y`, solve:

`dual = sqrt(w) * (sqrt(w) K_c sqrt(w) + alpha I)^(-1) * sqrt(w) Y_c`.

Query logits use the SAME training row/grand means, `K_query_centered * dual + r^T Y`. Fixed-temperature-one softmax is an aggregation interface, not calibrated risk. Fitted artifacts store dual coefficients, intercept and hashes/scales; features are referenced from the immutable cache rather than duplicated into every model.

## Mandatory equivalence and nesting

Before full nonlinear fitting, fit and restore every declared linear context, comparing with its original saved ridge classifier. Probability tolerance is fixed before fitting at rtol1e-8/atol1e-10, reflecting primal-versus-dual floating-point arithmetic; source models themselves restore with original tighter tolerances. Labels, inner F1, alpha selections and all per-seed/ensemble metric bundles must agree EXACTLY with the completed source study for all four representations. No integrity checks or tolerance changes to rescue a failed gate. Small synthetic tests compare against independent weighted normal-equation and sklearn oracles.

The preflight's 840 fitted linear models are reused during full fitting; they are not extra hidden training. Ten outer folds, three inner folds, all three feature seeds [42,123,2026]. Within each family configured-order ties; shared representation order [baseline,plus_mean,plus_spread,plus_quantiles]. All bandwidth/energy/kernel centering/class weights use exact training patients. Outer labels never choose hyperparameters, representation, seeds, stopping, thresholds or ensemble weights.

## Bounded budget and outputs

- 120 upstream feature contexts; 480 representation-specific kernel contexts.
- **840 linear fits** and source-equivalence checks.
- **1,560 nonlinear fits**, including all candidates and selected outer refits.
- **2,400 new classifiers total**, including **240 outer models**.
- 480 matched family/context checks; 30 paired representation choices.
- Eight fixed procedures plus two selected predictors in the report; selected predictors add no fitted outer models.

Linear family searches 8 representation/candidate combinations per inner partition; nonlinear searches 16. This asymmetry is disclosed. Report all fixed and selected procedures, every seed, mean member F1/sample seed SD, equal-three-seed probability ensembles, class F1/confusions, all declared differences and shared selection counts. The selected linear predictor is a new inner-selected PROCEDURE, not necessarily the same predictor as one fixed source arm.

Independent reporting loads the source chain again, reconstructs every kernel from fixed source features and current train-only statistics, restores every saved classifier, repeats all candidate/shared-representation choices and matching, reproduces source linear metrics, then publishes. It does not call classifier fit. Class-stratified paired patient bootstrap uses 5,000 shared draws/seed42, with mean-member differences retained separately. Intervals are conditional on fixed fitted predictions, without adjustment for repeated development, overlapping training folds, multiple comparisons or refits. Retain the PTB-development limitation; no SOTA, clinical validity, independent confirmation, demonstrated ceiling or guaranteed0.80–0.90 score.

## Engineering, timing, lifecycle

Meaningful tests: imbalanced weighted-intercept linear oracle, independent nonlinear weighted KRR oracle and PSD, query-independent bandwidth/centering, batch invariance, constant-scale fallbacks, source fingerprint/membership/cache tampering, outer-label counterfactuals, shared representation selection, complete nested restoration without refitting, publication/hash guards and compact fitted state.

A real engineering smoke uses only the first inner partition of outer fold1 and all its cached rows, excluding every outer-test patient. It fits/restores all24representation/family/candidate combinations and records actual timing. Smoke is not final performance. Use observed timing to assess cost before full launch; no promised runtime or score.

```sh
.venv/bin/python -m pytest tests/test_distribution_kernel.py -q
.venv/bin/python -m mi_localization.distribution_kernel_study --smoke
.venv/bin/python scripts/run_distribution_kernel.py --publish
.venv/bin/python scripts/distribution_kernel_status.py --watch
```

Runner holds one OS lock; performs full linear equivalence, fits remaining kernel models, independently reports, publishes only aggregates, then exits. No automatic follow-on study, raw-signal re-extraction or older paused resume. Monitor checks actual local PID commands, not just saved counters. Ctrl-C exits only the monitor. Private kernels/models/context audits and individual predictions remain ignored; public hashed scope, aggregates and non-patient receipts may be committed. No numerical package changes or OpenMP override hacks. Windows has not been validated; a changed source platform/config requires its own namespace.

Research basis: [Meunier et al., Distribution Regression with Sliced Wasserstein Kernels, ICML2022](https://proceedings.mlr.press/v162/meunier22b.html), motivating distribution kernels rather than establishing MI gains; [kernel ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html). Our signed classification targets, class weights, unpenalized intercept and finite-summary hybrid are explicitly specified above. HYDRA GPL attribution/license remain under `src/mi_localization/_vendor/hydra`.

Engineering receipts at preparation: 11 focused tests passed. Real smoke fitted/restored all24 combinations with no outer-test patients in12.99s including custody checks; kernel fit/restore0.113s. Full840linear equivalence gate reproduced all four original member/ensemble metric bundles and choices. These are gates, not nonlinear final performance.

Final prelaunch verification: **320 tests passed in130.22s**. All840real linear equivalence checks passed with maximum absolute probability error7.69e-14, unchanged labels/alpha choices and full original metrics. Source caches/older numerical code remain unchanged.
