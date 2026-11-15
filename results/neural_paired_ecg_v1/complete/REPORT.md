# Measured ECG + VCG extension: low-rank fusion

Separate measured ECG extension; repeatedly inspected PTB patient folds are development data, not independent confirmation or SOTA.

Complete: 164 unchanged patients, 391 recordings, 55,863 original beats, 6 classes, 10 frozen patient folds, target seeds [42, 123, 2026]; 600 validated fits.

| Model | Patient macro F1, mean ± SD | Parameters |
|---|---:|---:|
| vcg | 0.550 ± 0.040 | 78,198 |
| ecg | 0.509 ± 0.023 | 80,358 |
| vcg_twin | 0.511 ± 0.017 | 160,694 |
| joint_additive | 0.536 ± 0.035 | 160,694 |
| joint_bilinear | 0.534 ± 0.016 | 160,694 |

SD is across the three training seeds, not a generalization confidence interval. Every seed is retained; there is no best-seed selection or probability ensemble.

## Prespecified paired comparisons

| Comparison | Mean Δ macro F1 | Descriptive patient-bootstrap interval |
|---|---:|---:|
| joint_bilinear_minus_vcg | -0.017 | [-0.073, +0.041] |
| joint_bilinear_minus_vcg_twin | +0.023 | [-0.025, +0.075] |
| joint_bilinear_minus_joint_additive | -0.003 | [-0.046, +0.042] |
| joint_bilinear_minus_ecg | +0.025 | [-0.033, +0.085] |

5,000 class-stratified patient draws shared across arms and all fixed seeds. Repeated seed/patient predictions are not independent patients. These are descriptive, unadjusted intervals conditional on fitted predictions, without refitting, seed resampling or correction for correlated CV folds, repeated development or multiple comparisons.

## What was tested

Measured Frank XYZ stays the VCG primary benchmark. This is a separately labeled ECG extension using the actual simultaneous twelve ECG leads, not ECG reconstructed from VCG or transferred source-dataset representations. No PTB-XL records, checkpoints or test folds enter this phase. See [bank audit](../bank_audit/REPORT.md).

The single-stream models use the same compact residual architecture; ECG changes only the input convolution to twelve channels. The three dual models use the same two encoder architectures, exact initial weights, factor matrices, classifier and parameter counts. The VCG-only twin feeds a deterministic fourfold tiling of XYZ to its twelve-channel second branch; it adds capacity but no ECG information. All shared inputs use identical training-only per-lead normalization and all paired fits share exact sampled-row sequences in common epochs.

The dual embeddings a,b each have 64 dimensions. Rank-16 factors U,V project them. Bilinear fusion uses tanh((Ua) ⊙ (Vb)), which compresses cross-embedding outer-product interactions without materializing the full 64 × 64 tensor. The additive control uses tanh((Ua + Vb)/√2) with the same trainable tensors. Both concatenate the interaction with the two original embeddings before classification. Tanh makes this a nonlinear low-rank bilinear adaptation, not an exact reproduction of the cited models. The additive comparison tests the interaction rule; the VCG twin comparison tests replacing the second VCG view with measured ECG at matched capacity. It is not a full factorial isolation of all possible fusion choices.

All models start from scratch with ordinary CE, AdamW rate 0.001, decay 0.01, dropout 0.2, 8,192 class/patient-balanced beat draws per epoch, max 25 epochs and patience 6. There is no external pretraining, auxiliary loss, augmentation, learned calibration or rank/rate search. Three inner patient splits choose epochs; rounded median best epoch sets a fresh outer-training refit with no outer-test monitoring. Every held-out beat contributes to mean patient probabilities.

All patients, labels, cached R positions and crop boundaries are preserved. Denoising uses complete recordings and remains offline/noncausal. There is no clinical waveform delineation or validated attention explanation. Seeded MPS is not guaranteed bitwise deterministic.

## All seed results

| Training seed | vcg | ecg | vcg_twin | joint_additive | joint_bilinear |
|---|---:|---:|---:|---:|---:|
| 42 | 0.586 | 0.531 | 0.523 | 0.561 | 0.530 |
| 123 | 0.557 | 0.486 | 0.518 | 0.496 | 0.520 |
| 2026 | 0.507 | 0.510 | 0.491 | 0.551 | 0.551 |

## Mean class F1 across seeds

| Class | Patients | vcg | ecg | vcg_twin | joint_additive | joint_bilinear |
|---|---:|---:|---:|---:|---:|---:|
| AMI | 17 | 0.373 | 0.442 | 0.269 | 0.379 | 0.349 |
| ALMI | 16 | 0.436 | 0.368 | 0.463 | 0.408 | 0.401 |
| ASMI | 27 | 0.530 | 0.559 | 0.538 | 0.545 | 0.589 |
| IMI | 29 | 0.548 | 0.403 | 0.461 | 0.483 | 0.484 |
| ILMI | 23 | 0.576 | 0.481 | 0.514 | 0.532 | 0.530 |
| HC | 52 | 0.837 | 0.801 | 0.818 | 0.871 | 0.849 |

## Interpretation

These same PTB patients have been repeatedly inspected; added sensors, nesting and multiple seeds do not create independent confirmation. Larger models have additional compute and can overfit this small patient cohort. Similar initialization and bounded training are controls, not a guarantee of equally optimal tuning. No improvement, novelty, SOTA or 0.80–0.90 score follows merely from using a low-rank tensor. Any favorable development result still needs a genuinely new appropriate cohort and fixed protocol. Do not merge labels, remove hard patients or retune against old external test outputs to meet the target.

Research basis: [Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325), [Efficient Low-rank Multimodal Fusion](https://aclanthology.org/P18-1209/), [PTB documentation](https://physionet.org/content/ptbdb/1.0.0/). These motivate the representation and data choice; they do not supply performance evidence for this adaptation.

Reproduce: `.venv/bin/python -m mi_localization.paired_bank`, `.venv/bin/python -m mi_localization.paired_study`, then `.venv/bin/python -m mi_localization.paired_report`. Progress: `.venv/bin/python -m mi_localization.paired_study --status`. Patient predictions, source inventories, waveforms and fitted weights stay local; only aggregate reports/audits are public.
