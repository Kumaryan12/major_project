# Bounded random-convolution VCG feature results

Bounded record-template random-convolution development on repeatedly inspected PTB patient folds; no independent confirmation, exact upstream reproduction or established MI SOTA.

All 164 patients and 391 recording templates. 840 classifier contexts restored; 120 feature contexts reconstructed from their training patients, including 120 outer models.

| Features + matched ridge | Mean member macro F1 ± transform-seed SD | Prespecified three-seed ensemble F1 |
|---|---:|---:|
| dct | 0.446 ± 0.000 | 0.446 |
| multirocket | 0.551 ± 0.017 | 0.561 |
| hydra | 0.538 ± 0.012 | 0.548 |
| combined | 0.573 ± 0.019 | 0.584 |

DCT is deterministic and identical across transform seeds; its repeated entries are one control predictor, not independent replications. Ensemble F1 differs from mean member F1. Softmax of ridge scores at fixed temperature one is a declared aggregation interface, not calibrated clinical probability.

| Ensemble comparison | Δ macro F1 [descriptive interval] |
|---|---:|
| multirocket_minus_dct | +0.116 [+0.023, +0.205] |
| hydra_minus_dct | +0.102 [+0.009, +0.198] |
| combined_minus_dct | +0.138 [+0.048, +0.228] |
| combined_minus_multirocket | +0.023 [-0.020, +0.069] |
| combined_minus_hydra | +0.036 [-0.046, +0.116] |

## Class F1

| Class | dct | multirocket | hydra | combined |
|---|---:|---:|---:|---:|
| AMI | 0.188 | 0.267 | 0.333 | 0.345 |
| ALMI | 0.200 | 0.485 | 0.424 | 0.483 |
| ASMI | 0.509 | 0.643 | 0.582 | 0.633 |
| IMI | 0.357 | 0.517 | 0.540 | 0.542 |
| ILMI | 0.640 | 0.553 | 0.476 | 0.578 |
| HC | 0.781 | 0.904 | 0.933 | 0.925 |

## Protocol and limitations

Each recording template is the coordinate median of all its original denoised measured-XYZ beats. This compresses within-record variation, is not a real selected beat, and tests a template-based recipe rather than every possible beat-level random-convolution method. No patient, recording, beat, label, R position or frozen fold was changed. Every query recording contributes equally to its patient's feature row.

Each exact inner/outer training partition fits one positive scalar RMS with equal patient/equal-record weighting, then MultiRocket-style quantile thresholds using only training templates. Bias examples are drawn patient-uniformly, then record-uniformly. HYDRA kernels are random and label-free. Inner/outer contexts have fresh partition-specific seeds and are never initialized from an outer-trained transform. Scaling of the final patient features is training-only. All four arms share inputs, aggregation, class-balanced ridge classifier and the same two alpha candidates; highest mean inner patient macro F1 wins, with configured-order ties. Feature dimensions/capacity differ.

This bounded implementation uses MultiRocket's 84 length-nine patterns, four log-spaced dilations per input representation, original/differenced input, four paper-style positive-response poolers and consistent multivariate channel subsets. It uses true consecutive positive run lengths, including equality boundaries, and mean positive thresholded response; these are explicit conventions rather than exact historical source-code parity. HYDRA uses the vendored upstream experimental multivariate implementation with eight groups/eight kernels, including its with-replacement channel combinations. Both use the same StandardScaler/ridge interface, rather than HYDRA's separate sparse scaling. This is not the default full-size upstream recipe or MASHT.

Transform seeds 42/123/2026 are all retained with equal probability averaging; no best-seed, subset, ensemble-weight or threshold search. The previous neural ensemble 0.584570 is a different frozen development procedure, not the matched DCT control. No TabPFN, neural training, external patients or new tensor-fusion search is part of this bounded phase.

Intervals use 5,000 class-stratified paired patient draws conditional on fixed predictions. They do not include refits or correction for correlated CV training folds, repeated development or multiple comparisons. Seed SD is random-transform variation, not patient-level generalization uncertainty. These repeatedly inspected patients remain development data; no established novelty, independent confirmation, clinical validity or guaranteed 0.80–0.90 score.
