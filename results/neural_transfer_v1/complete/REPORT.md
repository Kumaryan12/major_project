# Exploratory masked-encoder transfer to measured VCG

Exploratory PTB development; source-target patient identity unresolved; no independent confirmation or SOTA claim.

164 patients, 391 records, 55,863 denoised measured-Frank XYZ beats, 6 classes; 10 frozen patient folds; target training seeds [42, 123, 2026].

| Target training seed | Scratch macro F1 | Pretrained macro F1 | Pretrained − scratch |
|---|---:|---:|---:|
| 42 | 0.589 | 0.515 | -0.075 |
| 123 | 0.577 | 0.528 | -0.049 |
| 2026 | 0.494 | 0.518 | +0.024 |

scratch: **0.553 ± 0.052** (mean ± sample SD across target training seeds).

pretrained: **0.520 ± 0.007** (mean ± sample SD across target training seeds).

Mean paired difference **-0.033**, descriptive paired patient-bootstrap interval [-0.091, +0.023] (5,000 class-stratified draws shared across both arms and all fixed seeds).

## Exact comparison

Source encoder: one fixed masked-reconstruction pilot, seed [42], 1,024 source-training and 128 source-validation PTB-XL patients. PTB-XL fold 10 was excluded. Source validation selects the source encoder; no PTB target examples enter source training or source stopping. Three target seeds are not three independent pretraining replications.

Only encoder weights transfer. The six-class classifier is freshly initialized with the same weights in paired fits; source reconstruction decoder, normalization and optimizer are discarded. Both arms fine-tune the entire same compact ResNet with ordinary CE, dropout and AdamW. Target normalization is fitted only on each target-training partition with the same patient/class weights in both arms.

Both arms receive the SAME two target learning-rate candidates [0.001, 0.0001], up to 25 epochs, patience 6, 8,192 sampled beats per epoch, and 3 fixed inner patient splits per outer fold. The maximum mean best inner-patient macro F1 selects the rate separately per arm; exact ties use configured order. Rounded median best inner epoch sets a fresh outer-training refit. Outer test patients never select rates, epochs or normalization. Historical 0.550 ± 0.040 is context, not the control for this new learning-rate search.

Validated 420 target fits: initialization provenance, fresh matching classifier, equal parameter counts, training normalization, seeds and shared-epoch sampled-row digests. Initial encoders deliberately differ across arms. Actual early-stopped training lengths may differ. Pretraining adds compute beyond the equal target tuning budgets. All held-out beats contribute to mean patient probabilities; there is no best-seed selection or probability ensemble.

## Mean class F1 across target seeds

| Class | Patients | Scratch | Pretrained |
|---|---:|---:|---:|
| AMI | 17 | 0.414 | 0.325 |
| ALMI | 16 | 0.436 | 0.419 |
| ASMI | 27 | 0.516 | 0.531 |
| IMI | 29 | 0.530 | 0.492 |
| ILMI | 23 | 0.583 | 0.509 |
| HC | 52 | 0.841 | 0.846 |

## Interpretation limits

The source–target recording screen passed its bounded top-k continuous-ECG design with synthetic copied-segment controls. This is not proof of disjoint patient identities. See [screen report](../overlap_audit/REPORT.md). Different recordings of the same person, shorter/modified copies and candidates outside retrieval can be missed. No target patient was removed or relabeled.

Kors-derived source XYZ differs from directly measured Frank target XYZ; better source reconstruction does not guarantee target improvement. All PTB folds have been repeatedly inspected during development. These bootstrap intervals are conditional on fixed fitted predictions and one fixed source encoder, without model refits, seed resampling or correction for correlated folds, repeated development or multiple testing. The study cannot establish patient-independent external confirmation, clinical utility or SOTA. It does not guarantee 0.80–0.90 macro F1.

Cached full-record denoising remains offline/noncausal. Seeded MPS is not guaranteed bitwise deterministic. Raw waveforms, recording-pair identities, patient predictions and weights remain local; only aggregate results and audits are published.

Reproduce: `.venv/bin/python -m mi_localization.overlap_screen`, then `.venv/bin/python -m mi_localization.transfer`, then `.venv/bin/python -m mi_localization.transfer_report`. Read progress: `.venv/bin/python -m mi_localization.transfer --status`.
