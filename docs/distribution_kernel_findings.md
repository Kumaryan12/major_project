# Completed paired linear/nonlinear kernel readouts

Reviewed 17 November 2026. Implementation `143a016`; independently validated aggregate results `30b8c2e`. [Prespecified protocol](distribution_kernel_plan.md), [validated report](../results/distribution_kernel_v1/complete/REPORT.md), [machine-readable metrics](../results/distribution_kernel_v1/complete/summary.json).

## What was implemented and verified

Class-balanced kernel ridge with an unpenalized weighted intercept, using the four complete, previously verified patient-feature representations. Each representation has a linear and linear+Gaussian kernel readout; all eight fixed arms and all three feature seeds are retained. A single shared representation is selected symmetrically from the two families' inner scores per outer fold/seed; the primary paired comparison is selected nonlinear minus selected linear. No outer-score representation selection or global OOF combiner.

All **2,400 classifiers** restored (840 linear, 1,560 nonlinear), including **240 outer models**. All **480 representation-specific kernel contexts/matching checks** and **840 source-ridge equivalence checks** passed; original source labels, selections and member/ensemble metric bundles reproduced. All **120 verified upstream feature contexts** reused read-only, with **zero waveform re-extraction**. Original 164 patients, 391 recordings, 55,863 beats, labels, R positions and frozen folds remain unchanged.

The real inner-only smoke fitted/restored 24 classifiers without outer-test patients. All **320 tests passed**. The full runner launched 17:15:41 UTC and finished publication 17:17:03 UTC, about **81 seconds**, after preparation/tests/initial equivalence. This is not a timing forecast for a different machine or fresh waveform bank. The runner has stopped and no following experiment is queued. Older multiview work remains paused.

At completion review, all **3,990 new audit files**, **4,800 model/prediction artifacts**, **90,429 upstream custody files** and pinned code/report hashes matched. Shared inner representation choices were baseline16, mean3, mean/spread4 and quantiles7 across30fold/seed contexts. These are overlapping training decisions, not independent evidence that one representation is universally best.

## All results

| Fixed representation or selected procedure | Linear mean member F1 ± seed SD | Nonlinear mean member F1 ± seed SD | Linear ensemble F1 | Nonlinear ensemble F1 |
|---|---:|---:|---:|---:|
| Baseline morphology | .572860 ± .019471 | .584453 ± .018903 | .584266 | .583987 |
| Plus mean | .548247 ± .031406 | .562123 ± .015353 | .581913 | .599492 |
| Plus mean/spread | .538610 ± .027460 | .563611 ± .012004 | .594098 | .595859 |
| Plus quantiles | .549509 ± .015852 | .575470 ± .011041 | .589485 | .592147 |
| Shared inner-selected representation | .562847 ± .017794 | .573104 ± .013918 | .603374 | .583933 |

**Primary selected nonlinear−linear ensemble difference: −.019441 [−.048244,+.005086].** The primary hypothesis of a reliable nonlinear improvement is not supported. The selected nonlinear point is lower, but the interval includes zero; this is not proof that nonlinear kernels are generally inferior.

The selected linear **.603374** is a new inner-selected procedure, not an improvement to the original fixed baseline by kernel nonlinearity. Its representation selection used training evidence, while this completion review must not promote it into an independently confirmed winner from its favorable outer score. It is numerically above the prior fixed spread ensemble .594098; that historical point difference is not the primary paired causal effect or a new-cohort confirmation.

The nonlinear mean-summary ensemble **.599492** is a secondary fixed-arm point, with matched difference **+.017578 [−.013003,+.049895]**. Nonlinear spread−linear spread is **+.001761 [−.029312,+.033004]**; nonlinear quantiles−linear quantiles **+.002662 [−.026689,+.032079]**; nonlinear morphology−linear morphology **−.000279 [−.030905,+.028433]**. Every fixed-representation family interval includes zero.

Nonlinear quantiles−nonlinear mean/spread is **−.003712 [−.021897,+.009799]**, not evidence for richer distribution shape. All per-seed/class results, mean-member contrasts and representation choice counts remain in the pinned summary.

## Interpretation

Nonlinear mean member scores are higher than linear means for all four fixed representations in this study. That observation does not establish an ensemble improvement: the primary selected mean rises .562847→.573104 (difference +.010257 [−.012354,+.033828], uncertain) while its ensemble falls .603374→.583933. Score averaging and probability ensembling remain distinct. Correlated errors, confidence scales and feature variation could affect this behavior; no causal mechanism was established by these tables.

These are deterministic readouts of the same finite summaries. They cannot recover signal discarded by the source representations. A whole-vector Gaussian/linear mixture is not an exact sliced-Wasserstein distance, full transport solver, acquisition bridge or new physiological mechanism. Nonlinear search has four candidates versus two linear candidates, a disclosed tuning-budget difference.

No declared ensemble comparison establishes a reliable overall gain. The highest selected linear point exceeds .60 on this repeatedly inspected development cohort, while .80–.90 remains unachieved and cannot be forecast. There is no independently confirmed new champion, clinical validity, demonstrated dataset ceiling or SOTA. All model/representation choices stay in training partitions; future hypotheses cannot relabel patients, remove errors or select class-specific routing from these outer outcomes.

## Boundary and integrity

Source cache provenance was rechecked without re-decoding raw waveforms. New independent reporting rebuilt kernels from the fixed cache, restored all fitted models, repeated all decisions and checked the paired source equivalences. This preserves the source's prior numerical validation and gives new readout verification; it is not a new independent raw-signal validation.

Preserve `distribution_kernel_model/source/study/report.py`, `configs/distribution_kernel_v1.yaml`, all completed results and all older source numerical scopes. No new encoder, source dataset, feature budget, kernel sweep, seed subset or follow-on run was started at completion. Public artifacts contain aggregate results and hashes; individual predictions/models remain ignored. See the handoff for subsequent user-authorized work, if any.
