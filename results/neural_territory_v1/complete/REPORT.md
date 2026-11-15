# Shared-label auxiliary supervision: VCG-only

VCG-only development on repeatedly inspected patient folds; auxiliary targets are source-label mentions, not independently verified anatomical diagnoses.

Complete: 164 unchanged patients, 391 recordings, 55,863 original beats, six original classes, 10 frozen patient folds, seeds [42, 123, 2026]; 240 validated fits.

| Training seed | Matched CE | CE + shared descriptors | Δ macro F1 |
|---|---:|---:|---:|
| 42 | 0.586 | 0.558 | -0.028 |
| 123 | 0.557 | 0.549 | -0.008 |
| 2026 | 0.507 | 0.509 | +0.001 |

ce: **0.550 ± 0.040** (mean ± sample SD across training seeds).

territory: **0.538 ± 0.026** (mean ± sample SD across training seeds).

Mean paired difference **-0.012**, descriptive paired patient interval [-0.056, +0.032] using 5,000 class-stratified patient draws shared across all arms and fixed seeds.

## Objective and exact control

Both arms have the original XYZ residual encoder, unchanged six-class classifier and the same four-output auxiliary head (78,458 parameters at production widths). Control loss is CE + 0 × BCE; experiment is CE + 0.2 × mean BCE across the four descriptors. Both auxiliary heads start with the same weights and the original model/dropout initialization stream is preserved. All shared-epoch sampled-row hashes and training-only normalizers are matched. The extra head's outputs are never used to replace the six-class prediction or construct a test-optimized hierarchy.

Auxiliary targets encode mentions in existing label names: AMI→anterior, ALMI→anterior+lateral, ASMI→anterior+septal, IMI→inferior, ILMI→inferior+lateral, HC→none. A zero means the component is absent from this source-label encoding, not independently confirmed anatomical absence. These are no new clinical annotations and do not repair label ambiguity. The six-class endpoint is unchanged.

Ordinary CE, AdamW rate 0.001, decay 0.01, dropout 0.2, 8,192 class/patient-balanced beat draws per epoch, max 25 epochs and patience 6. There is no auxiliary-weight search, augmentation, geometry input, delineation, ECG input, pretraining or external data. Fixed inner patient partitions select the stopping epochs using six-class patient macro F1 only. Rounded median best epoch determines a fresh outer-training refit without test monitoring. All held-out beats contribute to mean patient probabilities.

## Mean class F1 across seeds

| Class | Matched CE | Shared descriptors |
|---|---:|---:|
| AMI | 0.373 | 0.299 |
| ALMI | 0.436 | 0.336 |
| ASMI | 0.530 | 0.542 |
| IMI | 0.548 | 0.538 |
| ILMI | 0.576 | 0.655 |
| HC | 0.837 | 0.861 |

## Limits

This tests one fixed derived-label objective, not verified anatomical diagnosis, all possible multitask approaches, clinical validity, novelty or SOTA. Repeatedly inspected PTB folds remain development data. Bootstrap intervals condition on fixed fitted predictions, without model/seed refits or correction for correlated folds, repeated development or multiple experiments. Training seeds are not new patients. Offline full-record denoising remains noncausal; seeded MPS is not guaranteed bitwise deterministic. A 0.80–0.90 macro F1 is not promised. No label, patient, beat or R position was changed.

Patient predictions, waveform arrays and fitted weights stay local; aggregate reports/audits are public. Reproduce: `.venv/bin/python -m mi_localization.territory_study`, then `.venv/bin/python -m mi_localization.territory_report`. Read saved progress: `.venv/bin/python -m mi_localization.territory_study --status`.
