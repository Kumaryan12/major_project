# Staged morphology and frozen-context learning

Staged morphology/context development on repeatedly inspected PTB patient folds; fold-local teachers, prespecified seed ensembles, no independent confirmation or established SOTA.

All 164 original patients. 600 phase-fit/checkpoint contexts verified and restored (including 150 outer models); 120 matched adaptation-partition checks. 82 outer adaptation models selected zero updates from inner epochs. These remain phase checkpoints, not gradient-trained fits.

| Procedure | Mean member macro F1 ± seed SD | Prespecified seed-ensemble F1 | Ensemble MI-only F1 |
|---|---:|---:|---:|
| morphology | 0.551 ± 0.027 | 0.545 | 0.504 |
| continued_morphology | 0.553 ± 0.044 | 0.550 | 0.506 |
| joint_context | 0.547 ± 0.039 | 0.559 | 0.512 |
| frozen_context | 0.555 ± 0.013 | 0.564 | 0.522 |
| frozen_shuffled | 0.554 ± 0.014 | 0.565 | 0.521 |

Seed ensembles were declared before this study, and use all declared seeds with equal probabilities. Ensemble F1 is not the mean member F1 or an extra independent cohort. All individual-seed metrics are in summary.json.

| Ensemble comparison | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |
|---|---:|---:|
| frozen_context_minus_morphology | +0.018 [-0.022, +0.061] | +0.018 [-0.031, +0.069] |
| frozen_context_minus_continued_morphology | +0.014 [-0.035, +0.067] | +0.015 [-0.045, +0.079] |
| frozen_context_minus_joint_context | +0.004 [-0.045, +0.054] | +0.009 [-0.052, +0.070] |
| frozen_context_minus_frozen_shuffled | -0.002 [-0.021, +0.018] | +0.000 [-0.024, +0.024] |
| joint_context_minus_continued_morphology | +0.009 [-0.052, +0.074] | +0.006 [-0.070, +0.085] |
| continued_morphology_minus_morphology | +0.004 [-0.063, +0.073] | +0.002 [-0.078, +0.086] |

## Ensemble class F1

| Class | morphology | continued_morphology | joint_context | frozen_context | frozen_shuffled |
|---|---:|---:|---:|---:|---:|
| AMI | 0.286 | 0.308 | 0.353 | 0.400 | 0.387 |
| ALMI | 0.467 | 0.424 | 0.429 | 0.444 | 0.429 |
| ASMI | 0.633 | 0.561 | 0.607 | 0.633 | 0.644 |
| IMI | 0.492 | 0.525 | 0.508 | 0.484 | 0.500 |
| ILMI | 0.542 | 0.625 | 0.583 | 0.565 | 0.565 |
| HC | 0.851 | 0.854 | 0.874 | 0.854 | 0.865 |

## Exact protocol and interpretation

Morphology is learned from scratch separately within each inner-training or outer-training partition. Its inner best epoch sets a fresh outer refit duration. Every adaptation arm receives only that exact partition teacher’s encoder/direct head, plus a freshly initialized contextual branch whose final logit layer is zero. Thus adaptation starts at the unchanged teacher prediction. No previously fitted outer encoder is reused across inner validation patients.

Continued morphology trains only the direct predictor; joint context trains base and context; frozen context and its shuffled control train only context/RR/gate parameters. All adaptation controls disable direct-head dropout and retain context-head dropout during training. Equal registered counts do not imply equal trainable counts. Common positive epochs have identical sampled windows and direct beats; shuffling changes only contextual waveform/RR order. Frozen base weights are hashed every epoch and in selected checkpoints.

Adaptation inner selection includes epoch zero, with earliest-best ties and the same 25-epoch cap / patience six. Each arm’s rounded median best inner epoch (including zero) sets the outer refit, with no query monitoring. Outer zero-update adaptations reproduce the teacher. This is an inner selection fallback, not a guarantee of improved outer F1. Baseline morphology has no second phase; continued morphology controls extra optimization. Total exposure and selected duration can differ by arm.

Both phases use the original class/patient-balanced, uniform-record contiguous four-beat sampling, 8,192 nominal beat slots and 64 updates/epoch at the default configuration. All original patients, six labels, measured XYZ, R positions and frozen folds are retained. Query inference covers every beat once, masks tails, weights windows by real-beat count within records and averages records equally.

Intervals use 5,000 class-stratified paired patient draws, conditional on fixed ensemble predictions. No refits or correction for repeated development, overlapping CV training folds or multiple comparisons. Seed SD is optimization variation, not a generalization interval. MI-only scores retain unchanged six-way predictions, including MI→HC errors.

The frozen historical residual-context ensemble 0.584570 is a different procedure, not the matched teacher or joint control here. Side-tuning and feature-preserving adaptation are prior art; this is a VCG-specific controlled hypothesis, not demonstrated novelty, SOTA or a guaranteed 0.80–0.90 result. These repeatedly inspected PTB patients remain development data, requiring independent label-compatible confirmation.
