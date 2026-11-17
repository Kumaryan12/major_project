# Completed intact-base all-beat convolution distributions

Reviewed 17 November 2026. Implementation `2dfd577`, independently verified aggregate results `0a861b9`. [Frozen protocol](convolution_distribution_plan.md), [validated report](../results/convolution_distribution_v1/complete/REPORT.md), [full metrics](../results/convolution_distribution_v1/complete/summary.json).

## Completion and integrity

All **630 new classifiers + 210 frozen source classifiers** were restored, including **90 new + 30 source outer models**. Every original beat was replayed in all **120 feature contexts**, and all **120 matched checks** passed. The frozen baseline probabilities reproduce exactly. The cohort remains 164 patients, 391 recordings and 55,863 beats, with original labels, R positions and folds.

The runner finished independent validation at 14:33:57 UTC (20:03:57 IST), published aggregates to GitHub and exited. No new training study is running or automatically queued. The separately proposed nonlinear kernel follow-up has not been implemented or launched. Older multiview work remains paused.

At completion review, all **960 new fit/reference/context audit files**, their **1,500 fitted/data artifacts**, all **1,023 protected source artifacts**, the report/verification hashes and pinned numerical code matched their saved hashes. These are integrity observations, not extra independent patients or performance evidence. The last full implementation suite passed 309 tests; no numerical code changed in this review.

## All declared results

| Procedure | Mean member macro F1 ± seed SD | Uniform three-seed ensemble macro F1 |
|---|---:|---:|
| Full original convolution baseline | 0.572860 ± 0.019471 | 0.584266 |
| Baseline + mean residual summary | 0.548247 ± 0.031406 | 0.581913 |
| Baseline + mean/spread summary | 0.538610 ± 0.027460 | 0.594098 |
| Baseline + empirical quantiles | 0.549509 ± 0.015852 | 0.589485 |

The highest ensemble point in this study is the mean/spread arm, **0.594098**. It exceeds its matched baseline by **0.009832 macro F1**, about **0.98 points on a 0–100 F1 scale**. This is an ensemble result, not accuracy or the average single-model result. Do not select this arm as a newly confirmed champion after inspecting outer outcomes.

The historical neural ensemble 0.584570 remains a separate development reference. The new mean/spread point is numerically higher by 0.009528, but that is not this study's matched comparison or evidence of independent superiority. The aspiration of 0.80–0.90 remains unachieved.

## Prespecified primary conclusion

**Quantiles minus mean/spread: −0.004613**, descriptive paired patient interval **[−0.013297, 0.000000]**. The primary quantile-beyond-moments hypothesis is not supported by a positive difference. The zero endpoint is retained exactly; it is not converted into a claim of superiority, equivalence or formal significance. Quantile and spread ensembles differ on only two patient predictions, explaining the discrete, narrow conditional difference.

Quantiles minus the original baseline is **+0.005218 [−0.023246,+0.033727]**, also uncertain. All five original contrasts and the separate mean-member contrasts remain in the pinned summary/report. None is replaced by the favorable arm after inspecting outcomes.

## Explicitly post-hoc spread-versus-baseline supplement

The original report did not declare a direct spread−baseline interval among its five contrasts. To interpret the highest ensemble point, this completion review separately computed that comparison from the saved complete outer predictions. It used the same 5,000 class-stratified patient draws, seed 42 and fixed-prediction method as the report, first reproducing every published ensemble interval exactly to numerical tolerance. No model was fitted, no subset selected and no original report changed.

- Ensemble spread−baseline: **+0.009832 [−0.016967,+0.037176]**.
- Mean-member spread−baseline: **−0.034250**.
- Spread ensemble differs from baseline on **12 patients**: five become correct, four become wrong, and three change between incorrect classes.
- All three individual spread seeds score below their paired original baseline seeds.

[Supplemental aggregate receipt](../results/convolution_distribution_v1/posthoc_spread_baseline_v1.json) pins the source summary/prediction hashes and identifies the analysis as post-hoc. Its interval is conditional on the already fitted models and repeatedly inspected development cohort. It does not account for new training, repeated method development, source uncertainty or multiple comparisons. It does not establish a reliable overall gain.

## What changed by class

For mean/spread versus baseline, ensemble F1 increases for IMI **0.542373→0.612903**, ILMI **0.577778→0.604651** and ALMI **0.482759→0.500000**. AMI declines **0.344828→0.322581**, ASMI **0.633333→0.610169**, and HC **0.924528→0.914286**. These are descriptive outcomes; they do not authorize class-specific routing, patient deletion, relabeling or class-aware feature tuning on outer results.

## Interpretation and boundary

The original representation was retained in full, and all original beats were used. Adding variation summaries produced a small favorable ensemble point, while individual-run mean performance declined. That combination does not demonstrate that a typical individual model improved. A possible role for complementary ensemble errors is a hypothesis; these results do not establish a physiological mechanism.

More elaborate quantiles did not beat the simpler moment control. This does not prove that all distribution representations fail, that every quantile budget is ineffective or that the cohort has a demonstrated ceiling. It also does not justify an outer-score feature-budget sweep or a retroactive change to the primary hypothesis.

The next previously proposed bounded question is whether a nonlinear classifier can exploit the already verified morphology/distribution features differently. That proposal remains unlaunched and would require its own fixed controls, training-only kernel scaling/selection, exact partition reuse and independent restoration. No new source data, ECG extension, seed selection, model combiner or study is authorized by this completion review.

The cohort is development data, with no independent clinical confirmation or SOTA claim. Preserve `convolution_distribution_*`, its config and completed outputs; do not edit their pinned code to change a hypothesis or accelerate a future experiment. Source HYDRA attribution/license are at `src/mi_localization/_vendor/hydra/README.md` and `src/mi_localization/_vendor/hydra/LICENSE` (the generated report's `third_party/hydra` wording refers to this vendored component).
