# Uniform seed ensembles of direct-beat/context models

Post-hoc uniform seed-ensemble development analysis after inspecting source results; all declared seeds and six arms retained, no weight/subset tuning, no new training or independent confirmation.

All 164 original patients; every source seed [42, 123, 2026] and all six source arms. 720 source contexts reverified; 180 unique outer members independently restored again for ensemble inference. No new training fits. Source-study artifacts were read only.

| Procedure | Mean member macro F1 | Uniform ensemble macro F1 | Ensemble MI-only macro F1 |
|---|---:|---:|---:|
| beat_only | 0.551 | 0.545 | 0.504 |
| beat_aux | 0.518 | 0.549 | 0.497 |
| ordered_context | 0.520 | 0.554 | 0.503 |
| ordered_context_aux | 0.539 | 0.585 | 0.543 |
| shuffled_context | 0.518 | 0.573 | 0.516 |
| shuffled_context_aux | 0.510 | 0.556 | 0.507 |

Mean member F1 averages three independently optimized model scores. Ensemble F1 scores one predictor formed by averaging all three probability vectors; these are different estimands. The ensemble has three times the member inference cost and uses three fitted models per fold. No best seed, arm subset, mixing coefficient, class threshold or temperature was selected or fitted.

## Matched ensemble comparisons

| Comparison | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |
|---|---:|---:|
| ordered_context_aux_minus_beat_only | +0.039 [-0.035, +0.113] | +0.039 [-0.050, +0.127] |
| ordered_context_aux_minus_beat_aux | +0.035 [-0.035, +0.109] | +0.046 [-0.039, +0.133] |
| ordered_context_aux_minus_shuffled_context_aux | +0.028 [-0.043, +0.099] | +0.036 [-0.049, +0.120] |
| ordered_context_aux_minus_ordered_context | +0.031 [-0.047, +0.109] | +0.040 [-0.051, +0.132] |
| beat_aux_minus_beat_only | +0.004 [-0.060, +0.070] | -0.007 [-0.084, +0.074] |
| ordered_context_minus_beat_only | +0.009 [-0.049, +0.071] | -0.001 [-0.070, +0.071] |
| shuffled_context_aux_minus_shuffled_context | -0.017 [-0.098, +0.065] | -0.010 [-0.107, +0.086] |
| ordered_context_minus_shuffled_context | -0.019 [-0.087, +0.050] | -0.014 [-0.094, +0.068] |

## Class F1

| Class | Beat-only ensemble | Ordered+aux ensemble | Shuffled+aux ensemble |
|---|---:|---:|---:|
| AMI | 0.286 | 0.294 | 0.286 |
| ALMI | 0.467 | 0.471 | 0.452 |
| ASMI | 0.633 | 0.640 | 0.538 |
| IMI | 0.492 | 0.596 | 0.576 |
| ILMI | 0.542 | 0.638 | 0.609 |
| HC | 0.851 | 0.868 | 0.876 |

## Interpretation

The source single-model procedure did not demonstrate a gain: ordered+aux mean 0.539 versus matched beat-only 0.551. This follow-up changes the predictor to a uniform seed ensemble; it does not retroactively change that conclusion or establish chronological superiority. Averaging was examined after viewing source results, so this is explicitly post-hoc development work. All six arms are disclosed, including ensembles that do not improve their member mean. A promising arm's point score is not independent confirmation or SOTA.

Predictors are averaged only within the same arm and outer fold; each member excludes that patient and all of their records during training. The inference helper has no patient-label, seed-subset or weight argument. Test labels are used only for aggregate evaluation. Per-member normalization, original context permutations, complete beat coverage and probability restoration are checked. Individual patient predictions and model files remain local.

Intervals use 5,000 class-stratified paired patient draws shared across all ensembles and their members. No refits or correction for overlapping folds, repeated development, post-hoc analysis or multiple comparisons. A difference interval containing zero does not demonstrate improvement. MI-only metrics retain six-way predictions and MI→HC mistakes. Ensembles have one F1 score, not a seed SD.

The initial small gate was not a lasting constraint: contextual outer final training gates had high medians (stored in summary.json). Gate scale and context-head scale interact; that diagnostic is not physiological importance or proof of why a model failed. No gate threshold was tuned in this analysis.

Keep the ensemble as a reproducible development candidate. Before a strong generalization claim, freeze the recipe and evaluate a genuinely independent, label-compatible patient cohort. Further tuning on these repeatedly inspected patients remains development. No guaranteed 0.80–0.90 claim is justified.

## Reproduce

```sh
.venv/bin/python -m mi_localization.residual_seed_ensemble
```

`predict_heldout_fold` provides restored-model ensemble inference for the frozen held-out fold; it deliberately cannot score training patients as held-out cases. The reproduction command replays source verification without invoking a trainer or rewriting completed source reports.
