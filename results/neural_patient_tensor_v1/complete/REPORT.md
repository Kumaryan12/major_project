# Patient-level multi-scale tensor results

Complete patient-disjoint development evaluation: 164 patients, 55,863 beats, 6 classes, 10 outer folds.

These PTB patients have been repeatedly inspected. This is not independent confirmation, clinical validation or an established SOTA claim. No 0.80–0.90 score is promised.

| Model | Patient macro F1 | Accuracy | Δ vs ResNet reference | Paired 95% interval for Δ |
|---|---:|---:|---:|---:|
| Single-beat ResNet reference | 0.586 | 63.4% | +0.000 | — |
| Patient mean embedding | 0.522 | 57.9% | -0.064 | [-0.165, +0.036] |
| Patient gated attention | 0.443 | 52.4% | -0.143 | [-0.231, -0.056] |
| Learned scales + fixed compression + attention | 0.480 | 53.7% | -0.106 | [-0.194, -0.018] |
| Learned scales + supervised Tucker + attention | 0.507 | 57.9% | -0.080 | [-0.167, +0.007] |

Intervals use 5,000 class-stratified paired patient draws, conditional on fitted predictions. No model refits or correction for repeated development, correlated training folds or multiple comparisons. They do not include training-seed uncertainty.

## Tensor and patient learning

A training bag contains up to 16 beats from one patient. It is an unordered set, potentially spanning that patient's recordings—not a claimed continuous time series. All eligible beats are used exactly once at inference, with chunked encoding followed by full-patient embedding pooling.

The primary tensor is beats × XYZ × time × scales. The raw waveform is retained; three shared trainable filters initialize as local averages. They can learn signed responses and are not validated physiological frequency bands. The XYZ axis is never treated as interchangeable leads.

The auxiliary tensor branch pools time to 64 bins. The fixed control uses further average pooling; the learned branch uses supervised mode factors with ranks [3, 12, 3]. The beat axis is never contracted by an order-dependent Tucker factor. The CNN waveform path remains present in all models.

Factors and filters are optimized using patient-label loss, rather than a reconstruction-energy objective. An orthogonality penalty regularizes the Tucker factors. Mean and gated-attention controls distinguish patient pooling from the incremental scale/compression contribution. Parameter counts are recorded; these are not perfectly capacity-matched architectures.

Each epoch samples 512 class/patient-balanced bags, up to 8,192 beat presentations. Normalization uses only training patients. Fixed inner patient splits select early-stopping epochs; the rounded median determines a fresh outer-training refit without test monitoring. No encoder is initialized from a model trained on its validation patients.

The original reference averages beat probabilities; the new models pool embeddings and optimize a patient-bag loss. This intentionally tests a different objective and aggregation, not just a new tensor shape. MPS is seeded but not guaranteed bitwise deterministic.

## Class-level F1

| Class | Patients | Single-beat ResNet reference | Patient mean embedding | Patient gated attention | Learned scales + fixed compression + attention | Learned scales + supervised Tucker + attention |
|---|---:|---:|---:|---:|---:|---:|
| AMI | 17 | 0.438 | 0.529 | 0.372 | 0.378 | 0.276 |
| ALMI | 16 | 0.533 | 0.364 | 0.250 | 0.400 | 0.444 |
| ASMI | 27 | 0.582 | 0.370 | 0.293 | 0.308 | 0.453 |
| IMI | 29 | 0.571 | 0.400 | 0.476 | 0.456 | 0.456 |
| ILMI | 23 | 0.578 | 0.625 | 0.426 | 0.538 | 0.578 |
| HC | 52 | 0.816 | 0.846 | 0.843 | 0.800 | 0.833 |

## Incremental tensor comparison

- Tucker vs Patient gated attention: Δ F1 +0.063, descriptive paired interval [-0.025, +0.149].
- Tucker vs Learned scales + fixed compression + attention: Δ F1 +0.027, descriptive paired interval [-0.066, +0.118].

## Error review

The following are aggregate confusions. Errors should motivate signal/annotation review under consistent rules, not changes to labels to match predictions.

- Single-beat ResNet reference: HC → IMI: 7 patients; ILMI → IMI: 7 patients; AMI → ASMI: 5 patients.
- Patient mean embedding: IMI → ILMI: 9 patients; ALMI → ASMI: 7 patients; ASMI → ALMI: 7 patients.
- Patient gated attention: ILMI → IMI: 11 patients; ALMI → AMI: 9 patients; ASMI → AMI: 8 patients.
- Learned scales + fixed compression + attention: IMI → ILMI: 9 patients; AMI → ASMI: 7 patients; ASMI → AMI: 7 patients.
- Learned scales + supervised Tucker + attention: AMI → ALMI: 7 patients; ILMI → IMI: 7 patients; IMI → HC: 7 patients.

## Limits and next step

A more structured tensor is a testable hypothesis, not guaranteed extra information. Training-seed repeatability is reported separately; do not select a best seed, optimize an ensemble on outer predictions, or tune against PTB-XL's previously evaluated test split. Any improvement still needs a genuinely new validation cohort. No ECG + VCG model or self-supervised pretraining is included here.

Research basis: [attention MIL](https://proceedings.mlr.press/v80/ilse18a.html) and [tensor contraction layers](https://www.jmlr.org/papers/v21/18-503.html). This is a VCG adaptation, not an exact reproduction or an established novel method.

Reproduce with `.venv/bin/python -m mi_localization.patient_tensor` and `.venv/bin/python -m mi_localization.patient_tensor_report`. Patient prediction CSVs, fitted weights and feature arrays stay local. Only aggregate metrics/audits are published.
