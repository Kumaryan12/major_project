# Patient-nested consecutive-beat length comparison

Nested chronological length selection on repeatedly inspected PTB patient folds; shared length chosen symmetrically from ordered/shuffled inner F1, not outer scores or independent confirmation.

All 164 original patients. 1200 fitted contexts verified; 120 outer models restored; 300 matched recurrent-partition checks.

| Candidate beats | Training windows/epoch | Windows/batch | Inference windows/batch | Shared length selections |
|---:|---:|---:|---:|---:|
| 4 | 2048 | 32 | 64 | 18 |
| 8 | 1024 | 16 | 32 | 11 |
| 16 | 512 | 8 | 16 | 1 |

Selection counts are training-only decisions over overlapping outer-training sets/seeds, not independent proof of a universally optimal length. No per-length outer-score winner is selected.

| Procedure with shared nested length selection | Training seed | Macro F1 | MI-only macro F1 |
|---|---:|---:|---:|
| mean_rr | 42 | 0.437 | 0.393 |
| mean_rr | 123 | 0.544 | 0.491 |
| mean_rr | 2026 | 0.473 | 0.431 |
| mean_rr, mean ± sample SD | all | 0.485 ± 0.055 | — |
| ordered_no_rr | 42 | 0.474 | 0.413 |
| ordered_no_rr | 123 | 0.550 | 0.509 |
| ordered_no_rr | 2026 | 0.502 | 0.460 |
| ordered_no_rr, mean ± sample SD | all | 0.509 ± 0.039 | — |
| ordered_rr | 42 | 0.545 | 0.493 |
| ordered_rr | 123 | 0.511 | 0.459 |
| ordered_rr | 2026 | 0.517 | 0.473 |
| ordered_rr, mean ± sample SD | all | 0.524 ± 0.018 | — |
| shuffled_rr | 42 | 0.528 | 0.500 |
| shuffled_rr | 123 | 0.547 | 0.518 |
| shuffled_rr | 2026 | 0.523 | 0.487 |
| shuffled_rr, mean ± sample SD | all | 0.533 ± 0.013 | — |

| Paired mean-seed difference | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |
|---|---:|---:|
| ordered_rr_minus_shuffled_rr | -0.008 [-0.057, +0.039] | -0.026 [-0.085, +0.033] |
| ordered_rr_minus_ordered_no_rr | +0.016 [-0.029, +0.061] | +0.014 [-0.042, +0.068] |
| ordered_rr_minus_mean_rr | +0.039 [-0.012, +0.091] | +0.037 [-0.025, +0.099] |

## Exact selection and comparison

For each outer fold/training seed, every candidate length receives three inner patient fits for each of four arms. The shared length maximizes the mean of ordered_rr and shuffled_rr mean-inner macro F1, with configured-order ties. This is symmetric in the primary order-control pair. All four arms then use that same length for their outer refits; each arm's median inner epochs at that length sets its own fresh outer training duration. Test labels/probabilities never choose the length. RR and mean-pooling comparisons are conditional on this primary-pair selection policy, not independently length-optimized competitors.

Every length uses 8192 nominal beat slots/epoch, 128 beat slots/batch and 64 optimizer steps. Shorter windows produce more window labels; length also changes recurrent context and aggregation. Equal beat budgets do not make these optimization problems identical. Within a length, recurrent controls share weights at initialization, capacity, normalization and sampled windows at common epochs. There is no claim of identical windows across different lengths.

Original measured XYZ, cached R positions, patients, labels and folds are unchanged. Windows are consecutive retained beats from one recording. Preceding detector-RR and availability travel with each beat under shuffling; no fictitious shuffled-timestamp intervals or absolute time inputs. No-RR zeroes those channels but retains the architecture. Mean pooling has different active capacity. Right masks protect tails; every query beat is covered once, window probabilities are weighted by valid-beat count within a record, then records are averaged equally per patient.

Report verification rebuilds training-only moments, windows/permutations, stopping and symmetric length choices, and restores every outer model with identical predicted classes and declared probability tolerance. MPS is not guaranteed bitwise deterministic across machines/versions. Partial inner scores are optimized diagnostics, not final project performance.

Seed SD describes optimization variation, not independent patients or a generalization confidence interval. Paired class-stratified patient intervals share draws across seeds and condition on fitted predictions, without refits/correction for repeated development, overlapping training folds or multiple comparisons. MI-only scores keep six-way predictions, including MI→HC errors.

These PTB folds remain development data. A gain over shuffled control does not alone beat the historical ResNet 0.550. No score target, SOTA claim or independent confirmation is established by length selection. No full fixed-four or fixed-sixteen sweep was used to choose this grid.
