# Uniform seed ensembles after residual-context evaluation

Implemented **16 November 2026**, after the source result `07dd9e2`. This is explicitly **post-hoc development analysis**, not a prespecified result of the original six-arm study or independent confirmation.

The completed original study has mean patient macro F1 **0.551 ± 0.027** for beat-only versus **0.539 ± 0.032** for ordered context plus auxiliary beat supervision. Its combined-recipe difference interval includes zero. Those results remain intact.

**Completed verification on 16 November 2026:** all 720 source contexts reverified, all 120 matched partition checks passed, and all **180 outer members** were restored again for ensemble inference. The ordered+aux ensemble scored **0.584570**, versus beat-only ensemble **0.545112**: **+0.039458 [−0.034767,+0.112701]**, descriptive paired patient interval. Shuffled+aux ensemble scored **0.556158**; ordered−shuffled **+0.028412 [−0.043095,+0.098762]**. Both intervals span zero. These are promising development point scores, without a demonstrated improvement or chronological mechanism. The full suite passed **220 tests**. [Complete report](../results/neural_residual_seed_ensemble_v1/complete/REPORT.md), [comparison figure](../results/neural_residual_seed_ensemble_v1/complete/ensemble_comparison.png), [PDF figure](../results/neural_residual_seed_ensemble_v1/complete/ensemble_comparison.pdf).

## Predictor and controls

Average class probabilities with fixed weights **1/3, 1/3, 1/3** over every declared source seed **42/123/2026**, separately for each of the **six original arms**. Do not average across arms, select a seed pair, fit a temperature, tune class thresholds or route patients using labels. All six ensemble results, positive and negative, are reported.

This changes inference from a single model to three members. **F1 of averaged probabilities is different from average member F1**; it is not a fourth optimization seed or a seed standard deviation. Each member's own training-only epoch selection remains unchanged. Inference cost is approximately tripled, with members restored serially to limit memory.

For each original outer fold, all three members excluded that patient and every recording from training. The cohort, labels, cached R positions, windows, normalizers and per-record aggregation stay as in the source study. No new training fits are invoked. Patient-disjoint member training does not erase the fact that the ensemble analysis was chosen after source results were examined.

## Verification

The new module first requires the complete, matching source manifest, report and summary. It then replays **all 720 source contexts**, including inner stopping, windows, normalization and outer duration, along with **120 matched six-arm partition checks**. It independently restores **all 180 unique outer members again** for the actual ensemble inference path, checks their predictions against source checkpoints, and checks the ensemble against the arithmetic probability mean. The completed source reports, prediction files and checkpoints are read only.

Tests cover uniform probability averaging versus hard voting, member/class-order equivariance, invalid probabilities, paired bootstrap estimands, an actual three-seed tiny nested training/reverification example, source immutability, inference independence from query labels, and membership/weight-option/model-tamper rejection. The inference helper `predict_heldout_fold` accepts an arm and frozen outer fold, and deliberately has no query-label, member-subset or fitted-weight argument.

Paired intervals use 5,000 class-stratified patient draws shared across all ensembles and member predictions. They are conditional on fitted predictions, without refits or correction for repeated development, overlapping folds, multiple comparisons or post-hoc examination. A positive point difference with an interval spanning zero does not demonstrate a gain.

## Run

```sh
.venv/bin/python -m mi_localization.residual_seed_ensemble
.venv/bin/python scripts/plot_seed_ensemble.py
```

Native Windows uses `.venv\Scripts\python.exe`, but numerical validation so far is on macOS/MPS. Reproduction requires the original ignored local waveform banks, per-patient prediction checkpoints and fitted model files; a Git clone alone does not contain them. Changed code/source/settings require a new output namespace.

Configuration: [residual_seed_ensemble_v1.yaml](../configs/residual_seed_ensemble_v1.yaml). Output namespace: `results/neural_residual_seed_ensemble_v1`. Private patient predictions remain ignored; public outputs contain aggregate metrics, hashed verification audits, reports and figures. This follow-up does not change or restart the paused multiview study.
