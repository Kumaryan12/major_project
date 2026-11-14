# Cross-patient contrastive VCG: matched experiment

Complete evaluation: 164 unique patients, 55,863 beats, 6 unchanged classes, 10 frozen patient folds, training seeds [42, 123, 2026].

**Primary comparison: contrastive loss versus its matched cross-entropy control.** Both use the same structured batches, initialization, waveform model and maximum training budget. The historical ResNet mean of 0.550 is context, not the control for this changed sampler.

| Training seed | Matched CE macro F1 | Cross-patient SupCon macro F1 | Paired Δ F1 |
|---|---:|---:|---:|
| 42 | 0.540 | 0.565 | +0.026 |
| 123 | 0.512 | 0.478 | -0.034 |
| 2026 | 0.542 | 0.503 | -0.040 |

## Across all prespecified seeds

- matched_ce: mean macro F1 **0.531 ± 0.017** (sample SD across training seeds, not a generalization confidence interval).
- cross_patient_supcon: mean macro F1 **0.515 ± 0.045** (sample SD across training seeds, not a generalization confidence interval).
- Mean paired improvement: **-0.016**; descriptive patient-bootstrap interval [-0.065, +0.033].

Bootstrap uses 5,000 class-stratified patient draws shared across all seeds and both arms. Scores are averaged across fixed seeds; seed/patient repetitions are NOT treated as independent patients and probabilities are not ensembled. Conditional on fitted models: no training/seed resampling, or adjustment for correlated training folds, repeated development or multiple comparisons.

## Exact experiment

- Same compact measured-XYZ ResNet and 64-dimensional encoder output in both arms; no extra projection head, tensor branch, augmentation, pretraining or ECG input.
- Every batch selects 8 distinct training patients per class and 2 beats per selected patient: 96 beats. Training patients are selected uniformly within class. Beats use replacement only if a selected patient has fewer beats than requested.
- 86 batches = 8,256 beat presentations per epoch in BOTH arms. This differs from the historical 8,192-beat epoch. Both have max 25 epochs and patience 6; actual epochs may differ due to independent inner-only stopping.
- Control loss: CE. Experimental loss: CE + 0.1 × cross-patient SupCon, temperature 0.1. Contrastive embeddings use cosine normalization only inside that loss; the classifier retains its original embeddings and training-set-scaled XYZ input.
- Same-class DIFFERENT-patient examples are positives. All same-patient pairs, including self-pairs, are excluded from both positives and denominator. No outer/validation examples enter a training contrastive batch or memory bank.
- Audit checks verified identical initial weights, parameter counts, normalization and sampled-row hashes in every shared epoch of paired fits. The contrastive diagnostic is also recorded for CE, but contributes no gradient to that control.
- Inner patient partitions are fixed across arms AND seeds. They select the epoch budget; a fresh outer-training fit uses the rounded median best epoch without outer test monitoring. All test beats are predicted; mean probabilities yield one patient prediction.
- Existing labels, R positions, recordings and patient folds were not changed. Full-record denoising remains offline/noncausal. MPS is seeded but not guaranteed bitwise deterministic.

## Mean class-level F1 across seeds

| Class | Patients | Matched CE | Cross-patient SupCon |
|---|---:|---:|---:|
| AMI | 17 | 0.288 | 0.265 |
| ALMI | 16 | 0.340 | 0.380 |
| ASMI | 27 | 0.575 | 0.521 |
| IMI | 29 | 0.530 | 0.494 |
| ILMI | 23 | 0.607 | 0.574 |
| HC | 52 | 0.848 | 0.858 |

## Limits

All these PTB patients have been repeatedly inspected. Neither nesting nor three seeds creates independent confirmation. No label was changed to agree with predictions, no favorable seed was selected, and no outer-test-optimized ensemble is reported. A 0.80–0.90 macro F1 target is not guaranteed. A genuinely new validation cohort is required for an improvement claim.

Research basis: [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362). The cross-patient exclusion and structured batch policy are task-specific adaptations, not an exact reproduction or an established novelty/SOTA claim.

Reproduce: `.venv/bin/python -m mi_localization.contrastive` followed by `.venv/bin/python -m mi_localization.contrastive_report`. Read saved progress without training: `.venv/bin/python -m mi_localization.contrastive --status`.

Only aggregate metrics and audits are published. Patient prediction CSVs, waveform arrays and fitted weights stay local. Geometry-aware masked pretraining and ECG + VCG are not implemented in this phase.
