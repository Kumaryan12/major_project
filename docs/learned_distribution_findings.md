# Frozen learned morphology distributions: completed review

Reviewed 18 November 2026. Implementation `49e9ea7`; independently validated results `e1a616e`. The run started at 09:38:14 IST and published/stopped at 12:11:52 IST, approximately 2 h 34 min later. No following experiment was launched.

**This procedure did not improve the matched convolution baseline.** Empirical quantiles were the strongest added-feature ensemble, but the primary quantile-versus-spread comparison remains uncertain. These are six-class patient-level macro F1 results on repeatedly inspected PTB development patients.

Evidence: [protocol](learned_distribution_plan.md), [configuration](../configs/learned_distribution_v1.yaml), [validated report](../results/learned_distribution_v1/complete/REPORT.md), [machine-readable results](../results/learned_distribution_v1/complete/summary.json), [verification manifest](../results/learned_distribution_v1/complete/verification_manifest.json).

## What was tested

Retain the entire standardized 3,584-column combined-convolution baseline. Append 144 columns describing the distribution of frozen learned beat residuals, using mean, mean/population-SD grids, or empirical quantiles. Each beat and its own recording median are encoded into 64 features by the plain morphology teacher for the exact patient partition and seed. The median embedding is subtracted; latent column scales and an equal-patient/record/beat RMS are fitted within the training partition. Sixteen fixed feature projections produce nine-entry summary grids. The moment control uses a normal-quantile basis without asserting Gaussian physiology.

All 164 patients, 391 recordings, 55,863 beats, original labels/R positions and patient folds remain unchanged. Class-balanced ridge selects alpha from [1, 100] using three inner patient folds. All ten outer folds and all three seeds (42, 123, 2026) remain in the analysis. Encoder weights stay frozen; this is a test of learned residual summaries, rather than end-to-end distribution learning.

The reused inner teachers originally used their own inner-validation patients for early stopping. This inherited selection is disclosed; the encoders are not validation-independent. Outer teachers did not monitor outer-test labels, and no outer-trained teacher was reused across its inner-validation patients.

## Results

| Procedure | Mean member macro F1 ± seed SD | Uniform three-seed ensemble macro F1 |
|---|---:|---:|
| Original convolution baseline | 0.572860 ± 0.019471 | **0.584266** |
| Baseline + learned mean | 0.532907 ± 0.017784 | 0.554973 |
| Baseline + learned mean/spread | 0.539921 ± 0.042561 | 0.547867 |
| Baseline + learned quantiles | 0.543027 ± 0.042541 | 0.580895 |

The primary quantiles-minus-spread ensemble difference is **+0.033028**, with a descriptive paired-patient 95% interval **[−0.011283, +0.079062]**. It does not establish a quantile benefit. Its mean-member difference is +0.003105 [−0.018206, +0.025170].

Quantiles minus the original baseline is **−0.003371 [−0.060716, +0.052488]**. The strongest added-feature ensemble is therefore slightly below the baseline point score, with uncertainty spanning both directions. Quantiles also have a lower member mean: −0.029834 [−0.075399, +0.015450]. All declared comparisons are retained in the report; none of the ensemble-difference intervals excludes zero.

These conditional, class-stratified patient-bootstrap intervals use fixed predictions. They do not incorporate model refitting, repeated development decisions, overlapping training folds or multiplicity. Seed SD measures variation across training seeds and is not a generalization confidence interval.

## Seed consistency

| Procedure | Seed 42 | Seed 123 | Seed 2026 |
|---|---:|---:|---:|
| Baseline | 0.553775 | 0.572111 | 0.592695 |
| + learned mean | 0.520752 | 0.524649 | 0.553319 |
| + learned mean/spread | 0.494676 | 0.545928 | 0.579160 |
| + learned quantiles | 0.494104 | 0.571312 | 0.563664 |

All nine added-feature member scores are below their paired baseline scores. The quantile ensemble being close to baseline does not imply that its average member improved: averaging predictions and averaging F1 are different operations. No seed was selected or discarded.

## Class tradeoffs

| Class | Baseline ensemble F1 | Learned-quantile ensemble F1 |
|---|---:|---:|
| AMI (anterior MI) | 0.344828 | 0.275862 |
| ALMI | 0.482759 | 0.451613 |
| ASMI | 0.633333 | 0.596491 |
| IMI | 0.542373 | 0.645161 |
| ILMI | 0.577778 | 0.619048 |
| HC | 0.924528 | 0.897196 |

The quantile arm shifts performance toward IMI/ILMI while the other four class scores decline. This is a descriptive diagnostic, not permission to route classes, relabel patients or tune against these outer outcomes.

## Verification and interpretation

Independent reporting re-encoded every original beat and median across all **120 teacher/feature contexts**, reconstructed training transforms, restored **630 new + 210 source classifiers**, including **120 outer models**, repeated selections and passed **120 matched checks**. All **120 teacher query probability/label replays** passed, baseline metrics were exact, the original cohort and protected source files were unchanged, and encoder training steps were zero. The runner is stopped, with no matching local runner/worker at review.

A fresh read-only completion check matched the report/verification hashes, all 960 fitted/context audit hashes, 1,500 fitted/data artifacts and 43,401 record embedding audit/data pairs. Pinned numerical code and the current source provenance matched the study manifest; 90,804 protected source files were inventoried by the source loader. The maximum recorded teacher probability replay error was 6.81e−7, within the fixed CPU-versus-source-MPS tolerance. This check did not retrain or restart the study.

Before launch, nine focused cases, a real full-beat inner-only restoration/reconstruction smoke and **321 main-suite tests** passed. The focused cases run in a child interpreter matching production isolation after native Torch Conv1d crashed in mixed-suite synthetic fixtures; no tests were skipped and no numerical packages or OpenMP overrides were changed.

Retaining the original feature block does not guarantee baseline performance: the joint ridge fit can change the coefficients on those original columns. Possible explanations include redundant/noisy residual summaries, their relative scale, or a frozen encoder optimized for a different objective; none is established as the cause by this experiment. Subtracting each recording's median embedding specifically tests residual variation and may omit absolute learned morphology. This result does not rule out all learned representations or distribution models.

Keep the baseline as the matched reference for this study. Historical selected-linear 0.603374, fixed-convolution spread 0.594098 and neural ensemble 0.584570 remain separately labeled development procedures, not interchangeable controls or independently confirmed winners. This completed experiment establishes neither SOTA nor a dataset ceiling. No further sweep or older paused experiment was started during review.
