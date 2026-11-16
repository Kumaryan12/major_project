# Direct beat morphology plus gated context

Fixed-four-beat direct morphology plus gated context development on repeatedly inspected patient folds; no independent confirmation or established SOTA.

All 164 original patients. 720 fits verified, 180 outer models restored, 120 matched six-arm partition checks.

| Procedure | Seed | Patient macro F1 | MI-only macro F1 |
|---|---:|---:|---:|
| beat_only | 42 | 0.527 | 0.517 |
| beat_only | 123 | 0.580 | 0.547 |
| beat_only | 2026 | 0.545 | 0.502 |
| beat_only, mean ± sample SD | all | 0.551 ± 0.027 | — |
| beat_aux | 42 | 0.523 | 0.483 |
| beat_aux | 123 | 0.539 | 0.497 |
| beat_aux | 2026 | 0.493 | 0.453 |
| beat_aux, mean ± sample SD | all | 0.518 ± 0.023 | — |
| ordered_context | 42 | 0.534 | 0.483 |
| ordered_context | 123 | 0.569 | 0.528 |
| ordered_context | 2026 | 0.458 | 0.410 |
| ordered_context, mean ± sample SD | all | 0.520 ± 0.057 | — |
| ordered_context_aux | 42 | 0.504 | 0.455 |
| ordered_context_aux | 123 | 0.564 | 0.533 |
| ordered_context_aux | 2026 | 0.549 | 0.518 |
| ordered_context_aux, mean ± sample SD | all | 0.539 ± 0.032 | — |
| shuffled_context | 42 | 0.528 | 0.486 |
| shuffled_context | 123 | 0.532 | 0.493 |
| shuffled_context | 2026 | 0.494 | 0.441 |
| shuffled_context, mean ± sample SD | all | 0.518 ± 0.021 | — |
| shuffled_context_aux | 42 | 0.508 | 0.457 |
| shuffled_context_aux | 123 | 0.574 | 0.544 |
| shuffled_context_aux | 2026 | 0.449 | 0.378 |
| shuffled_context_aux, mean ± sample SD | all | 0.510 ± 0.063 | — |

| Paired mean-seed difference | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |
|---|---:|---:|
| ordered_context_aux_minus_beat_only | -0.012 [-0.051, +0.027] | -0.020 [-0.067, +0.025] |
| ordered_context_aux_minus_beat_aux | +0.021 [-0.020, +0.061] | +0.024 [-0.023, +0.072] |
| ordered_context_aux_minus_shuffled_context_aux | +0.029 [-0.009, +0.067] | +0.042 [-0.003, +0.089] |
| ordered_context_aux_minus_ordered_context | +0.019 [-0.026, +0.064] | +0.028 [-0.025, +0.081] |
| beat_aux_minus_beat_only | -0.032 [-0.077, +0.011] | -0.044 [-0.098, +0.007] |
| ordered_context_minus_beat_only | -0.031 [-0.079, +0.015] | -0.048 [-0.105, +0.005] |
| shuffled_context_aux_minus_shuffled_context | -0.008 [-0.051, +0.034] | -0.014 [-0.065, +0.037] |
| ordered_context_minus_shuffled_context | +0.002 [-0.043, +0.045] | +0.000 [-0.054, +0.053] |

## Protocol and limits

Fixed 4-beat consecutive windows. The fixed length was informed by prior inner-fold preferences, not a universally optimal beat count. Original patients, labels, measured XYZ, R positions and frozen ten outer/three inner patient folds are retained. Every fit starts fresh; no saved outer encoder is reused across inner validation patients.

Every arm retains the same 64-dimensional encoder and direct per-beat class head. A masked GRU summarizes waveform/RR pairs into a 32-dimensional state; a learned scalar sigmoid gate multiplies a contextual class-logit correction. The correction is added to each direct beat logit, softmax is applied per beat, and real-beat probabilities are averaged within the window. In the beat-only controls the gate is forced to zero. Registered parameter counts and initial states match; the inactive branch means effective capacity is not identical.

The window objective is negative log probability of the true window class. Auxiliary arms add 0.5 times direct beat cross entropy, averaged within each real window then across windows. The coefficient and initial gate bias -3.0 were fixed before this sweep. There is no outer-score tuning of gate, coefficient, seed, ensemble or stopping. Direct beats/dropout assignments stay in their original order; only context embedding/RR pairs are permuted in shuffled controls, with timing attached and missingness retained.

All arms draw the same class/patient-balanced, uniform-record contiguous windows: 2048 windows, 8192 nominal beat slots and 64 optimizer steps per epoch. Inner patient macro F1 selects the earliest best epoch with patience 6, at most 25 epochs. Its rounded median sets a fresh outer refit duration. Normalization is training-only. Query inference covers every beat once with masked tails, beat-count weighting within each recording, then equal-record patient averaging.

Combined-recipe versus beat_only includes both context and auxiliary supervision. ordered_context_aux versus beat_aux tests adding contextual capacity/RR with the auxiliary coefficient matched; its shuffled counterpart controls chronological order. ordered_context_aux versus ordered_context tests the auxiliary term. Preserve all declared comparisons rather than selecting the best seed or arm as confirmation.

Intervals use 5,000 paired class-stratified patient bootstrap draws shared across optimization seeds and arms, conditional on saved predictions. No model refits, correction for overlapping training folds, repeated development or multiple comparisons. Seed SD is optimization variability, not a generalization confidence interval. MI-only scores retain original six-way predictions, including MI→HC errors.

Historical ResNet 0.550 and previous GRU 0.524 use different procedures, not fresh matched controls for this recipe. These repeatedly inspected PTB patients remain development data. No SOTA, independent confirmation, causal mechanism or 0.80–0.90 target achievement follows from an architecture name. Raw inputs, individual predictions and fitted weights stay local.
