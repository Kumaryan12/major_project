# Ordered VCG path-signature feature comparison

Ordered-trajectory representation development on repeatedly inspected PTB patient folds; no independent confirmation, clinical validity, novel signature theory or demonstrated MI SOTA.

All 164 patients, frozen outer/inner patient splits. 1050 fitted contexts; all 150 outer contexts restored and relevant selector-bootstrap caches independently replayed.

| Procedure | Sampling seed | Macro F1 | MI-only macro F1 |
|---|---:|---:|---:|
| morphology | 42 | 0.513 | 0.445 |
| morphology | 123 | 0.520 | 0.457 |
| morphology | 2026 | 0.512 | 0.444 |
| morphology, mean ± sample SD | all | 0.515 ± 0.004 | — |
| signed_area | 42 | 0.521 | 0.460 |
| signed_area | 123 | 0.475 | 0.403 |
| signed_area | 2026 | 0.495 | 0.431 |
| signed_area, mean ± sample SD | all | 0.497 ± 0.023 | — |
| endpoint_polynomial | 42 | 0.547 | 0.485 |
| endpoint_polynomial | 123 | 0.507 | 0.437 |
| endpoint_polynomial | 2026 | 0.484 | 0.413 |
| endpoint_polynomial, mean ± sample SD | all | 0.513 ± 0.032 | — |
| signature2 | 42 | 0.490 | 0.431 |
| signature2 | 123 | 0.469 | 0.398 |
| signature2 | 2026 | 0.482 | 0.415 |
| signature2, mean ± sample SD | all | 0.480 ± 0.010 | — |
| signature3 | 42 | 0.506 | 0.444 |
| signature3 | 123 | 0.509 | 0.438 |
| signature3 | 2026 | 0.509 | 0.448 |
| signature3, mean ± sample SD | all | 0.508 ± 0.002 | — |

The seeds change classifier-training bootstrap feature selection while the PFN inference-seed policy is held fixed. These are not three independent patient cohorts or three gradient-trained PFN foundation models. MI-only evaluation keeps original six-way predictions, including HC mistakes on MI patients.

| Mean-seed comparison | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |
|---|---:|---:|
| signed_area_minus_morphology | -0.018 [-0.077, +0.042] | -0.017 [-0.088, +0.057] |
| endpoint_polynomial_minus_morphology | -0.002 [-0.046, +0.043] | -0.004 [-0.058, +0.053] |
| signature2_minus_morphology | -0.034 [-0.101, +0.029] | -0.034 [-0.113, +0.044] |
| signature3_minus_morphology | -0.007 [-0.083, +0.066] | -0.005 [-0.094, +0.081] |
| signature3_minus_signed_area | +0.011 [-0.041, +0.064] | +0.012 [-0.049, +0.073] |
| signature3_minus_endpoint_polynomial | -0.004 [-0.082, +0.069] | -0.002 [-0.088, +0.087] |
| signature3_minus_signature2 | +0.028 [-0.017, +0.076] | +0.029 [-0.025, +0.086] |

5,000 paired class-stratified patient bootstrap draws, with each draw shared across sampling seeds. Conditional on fixed predictions, no model refits or correction for overlapping training folds, repeated development or multiple comparisons. These are descriptive intervals, not independent confirmation.

## Exact comparison

Each arm retains the original 480 morphology columns. Added columns: signed-area 120, endpoint-polynomial 840, signature2 210, signature3 840. Descriptor counts include the configured fixed windows, within-record beat median/IQR, and equal-record patient aggregation. Physical XYZ means retain offsets alongside increment-based signatures. Every original beat contributes, with no temporal/spatial warping, per-beat normalization, record filtering or new clinical phase labels.

The raw feature bank retains complete tensor words for auditing. Before classification, pure clock-only words and their IQR columns are excluded by their schema, independently of patient values/labels: every beat has the same clock per fixed window. Time–XYZ interactions and physical XYZ means are retained. This avoids tiny negative weighted-variance roundoff on known constants; no full-cohort variable-column screen is fitted.

The same-size endpoint-polynomial control uses straight endpoint increments and window means, omitting interior ordered path terms. Signed areas provide a simpler geometric control. Differences in feature-pool structure remain part of each procedure: equal selected budgets do not make representation/selection identical or establish causal physiology.

Each outer-training set is fully available to every classifier arm. Training-only 64-bootstrap consensus selection and 32 separate audit resamples retain at most 24/64 columns, with budget selected by mean inner patient macro F1, stable configured-order ties. Scaling, ranking and correlation pruning never use query patients. Raw selected values go to frozen pinned local TabPFN-v2 with eight fixed inference members; no prior correction, fine-tuning, external prediction API or changed checkpoint. Every outer context is freshly fitted after inner selection. A new matched control is included for every sampling seed.

Mathematical identity checks and raw/denoised/±4 ms sensitivity audits are in the separate bank audit. They neither choose features nor establish diagnostic value. Signatures preserve selected ordered interactions of piecewise-linear trajectories, not a new sensor measurement or a complete lossless encoding at finite degree. Signature methods and their ECG application are prior art; this is a measured-VCG MI development experiment, not an unqualified originality or SOTA claim.

No patient, label, class, original beat or R position changed. Existing offline denoising/record aggregation remains. Historical ResNet 0.550 ± 0.040 and earlier tabular results use different procedures and are context, not matched controls. These PTB folds have been repeatedly inspected; independent cohort confirmation remains necessary.
