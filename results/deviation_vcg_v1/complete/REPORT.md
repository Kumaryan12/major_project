# Healthy-reference VCG deviation study

Healthy-reference deviation development on repeatedly inspected PTB folds; supervised reference construction, not clinical counterfactuals, independent confirmation or demonstrated SOTA.

All 164 original patients; patient-disjoint nested selection. 1960 fitted member contexts; every outer context restored and every relevant training-bootstrap cache replayed.

| Procedure | Reference seed | Patient macro F1 | MI-only macro F1 |
|---|---:|---:|---:|
| full_morphology | 42 | 0.513 | 0.445 |
| ensemble_morphology | 42 | 0.493 | 0.422 |
| ensemble_morphology | 123 | 0.485 | 0.417 |
| ensemble_morphology | 2026 | 0.520 | 0.454 |
| ensemble_morphology, mean ± sample SD | all | 0.499 ± 0.019 | — |
| healthy | 42 | 0.511 | 0.445 |
| healthy | 123 | 0.483 | 0.407 |
| healthy | 2026 | 0.444 | 0.365 |
| healthy, mean ± sample SD | all | 0.479 ± 0.034 | — |
| mixed | 42 | 0.507 | 0.447 |
| mixed | 123 | 0.467 | 0.391 |
| mixed | 2026 | 0.414 | 0.335 |
| mixed, mean ± sample SD | all | 0.463 ± 0.047 | — |

MI-only scores use the five original MI classes and unchanged six-way predictions: an MI patient predicted HC remains an error. Seed rows reuse the same patients and are not independent cohort replications. The full-morphology reference is fitted once per outer fold and is not three independently trained baselines.

| Mean-seed difference | Six-class Δ [descriptive interval] | MI-only Δ [descriptive interval] |
|---|---:|---:|
| healthy_minus_ensemble_morphology | -0.020 [-0.061, +0.017] | -0.025 [-0.072, +0.019] |
| mixed_minus_ensemble_morphology | -0.036 [-0.085, +0.011] | -0.040 [-0.095, +0.016] |
| healthy_minus_mixed | +0.017 [-0.022, +0.056] | +0.014 [-0.030, +0.063] |
| ensemble_morphology_minus_full_morphology | -0.013 [-0.072, +0.040] | -0.014 [-0.083, +0.052] |

5,000 paired class-stratified patient bootstrap draws, with the same patient indices applied across reference seeds. Intervals condition on fitted predictions; no refits or correction for correlated folds, repeated development or multiple comparisons. They are not independent confirmation.

## Exact representation and comparison

Each of three members learns two exact medoids from a separate reference part, and fits its classifier on the other two parts. Reference, classifier and query patients never overlap within a member. Every training patient is a classifier patient in two members. The healthy library uses training HC labels; this is supervised reference construction. Mixed libraries match its number of reference patients but sample identities without using disease labels or scores. Atlas sizes and medoid identities differ by member/partition; seed repeats change reference partitions, with the classifier seed policy held fixed.

The deviation tensor subtracts physical XYZ reference waveforms from recording medians, then applies the established five additive wavelet bands and four fixed R-relative windows. Signed mean/RMS and DCT-distance features yield 243 extra columns. These windows are not clinical boundaries. Equal-record coordinate medians give one row per patient. Original 480-column morphology is retained in all arms; only deviation arms append columns. Two budgets (24/64) are selected by mean inner patient macro F1, separately for each procedure/seed, with configured-order ties. Bootstrap ranking/scaling uses classifier-training patients only. Queries never fit an atlas or selector. These are lossy deterministic features, not additional sensor data or clinically verified pathological residuals.

Raw pinned local TabPFN-v2 probabilities, eight fixed inference members, no inverse-prior correction, fine-tuning, hosted prediction service or sample-weight approximation. The selector uses class-balanced training scaling, but the PFN sees raw selected columns and its own training-context preprocessing. Reference-disjoint ensembles have smaller classifier contexts and average three probability vectors; their morphology-only arm controls this. The full-training morphology procedure is a practical reference, not a capacity-matched control. Each procedure chooses its own feature budget; comparison includes that selection policy.

No patient, class, label, original beat or R position was removed or changed. Recording medians use all beats of that recording, so this is offline inference. Repeatedly inspected PTB folds remain development data. No SOTA, clinical validity, mechanistic proof or 0.80–0.90 claim follows from architecture names or a favorable point estimate.
