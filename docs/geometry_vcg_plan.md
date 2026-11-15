# VCG phase geometry: first utility screen

## What this phase implements

The proposed physiology/geometry direction starts with a bounded feature screen, not another large neural run. It implements VCG-only automatic boundary estimation, explicit fallback accounting, 78 spatial/temporal descriptors per original beat, fixed/adaptive phase tensors, and two small patient-level classifier families. Shared-territory supervision and a neural geometry branch are subsequent hypotheses, not completed components of this first screen.

The same 164 patients, 391 recordings, 55,863 beats, six original labels and ten frozen patient folds remain. No ECG, source pretraining, external cohort, label corrections or patient removals enter feature extraction or classification. Every recomputed denoised Frank crop must exactly reproduce the original 651-sample VCG bank. Source headers and physical units are verified.

## Boundary estimates and their limits

NeuroKit2 **0.3.0** DWT delineation is applied separately to each measured XYZ axis, with the original R positions supplied unchanged. The recording's median R sign/scale orients each axis only for the delineator. Feature amplitudes and coordinates retain their original physical frame; no beat-level amplitude normalization or arbitrary rotation is performed.

An estimate must have an ordered QRS onset/offset and T onset/end, plausible durations, and an end before the allowed context ceiling. At least two axis estimates must pass; their maximum spread is limited to 40 ms for QRS markers and 80 ms for T markers. Accepted boundary locations are their rounded medians.

Missing/inconsistent estimates fall back to R−60 ms, R+80 ms and R+140 ms phase starts. The final end is min(R+650 ms, next **cached** R−80 ms, recording end). No new R positions are detected. A next heartbeat absent from the original cached list is not newly identified by this guard. This is an offline analysis with full-record denoising and per-record morphology estimation, not a streaming system.

The completed extraction accepted **12,977/55,863 beats (23.2%)**, with **42,886 fallback beats**. Acceptance differed by source class: approximately 15–22% for the MI classes and 43% for healthy controls. This is a substantial limitation, not evidence of reliable delineation. The classifier does not receive explicit success/quality flags, but fallback-dependent durations can still encode detector behavior. The fixed-window control is therefore required.

There are no independent expert boundary annotations for this PTB cohort in the pipeline. Timing/plausibility checks, synthetic tests and sample plots do not establish clinical boundary accuracy. The local review packet contains two label-stratified patients per class, selected by ID/label without model errors or classification scores. No patient is dropped when the delineator fails.

## Features and tensors

A beat baseline is the XYZ median over R−200 to R−120 ms. The 11-sample cubic Savitzky–Golay smoother supports stable derivative calculations; exact original-crop verification occurs before smoothing.

Each proposed QRS/ST/T segment contributes 24 descriptors: physical duration, XYZ mean/integral, maximum-magnitude vector, maximum/RMS magnitude, path length, mean/95th-percentile speed, signed projected polygon areas, singular-value shape fractions, mean turning angle and endpoint displacement. Six cross-phase descriptors describe peak-vector cosines, path ratios and QT span, totaling **78 per beat**. Polygon closure is a geometric definition, including for open physiological paths; it is not an anatomical area. Flat/degenerate curves produce explicit finite zero-angle/shape conventions.

The corresponding tensor is **phase × descriptor channel × normalized phase time = 3 × 7 × 32** per beat. Channels are XYZ, physical-time velocity XYZ and vector magnitude. Resampling aligns phase progression but duration remains a separate descriptor. Fixed and adaptive tensors are both cached for a possible later controlled neural experiment; the small classifiers use the descriptors, not learned tensor contraction.

Every patient becomes one row using median and interquartile range across all their beats. More-beat recordings contribute more within that patient; no recording weighting or beat selection is learned from labels.

| Feature view | Patient columns | Role |
|---|---:|---|
| Existing short-wavelet statistics | 384 | Same-classifier control: median/IQR of 192 descriptors |
| Fixed-window geometry | 156 | Geometric family with predefined intervals |
| Adaptive geometry with fallback | 156 | Change interval estimation at identical feature count |
| Statistics + adaptive geometry | 540 | Test complementary information |

## Nested classifier screen

All four views are evaluated with both learners, reporting **all eight procedures**:

- L2 multinomial logistic regression, C in [0.01, 1.0].
- Shallow XGBoost, depth in [2, 3], 100 boosting rounds, learning rate 0.05, lambda 1, full row/column sampling. One CPU thread is used for the small patient table, including XGBoost's metadata-allocation context, to avoid competing macOS OpenMP pools.

Both use patient-class-balanced weights once. The scaler's weighted moments are fitted only to training patients. Three fixed inner patient folds select the candidate by mean macro F1; exact ties favor configured order. The chosen candidate is freshly refit on all outer-training patients, without test-based selection or early stopping. One fixed computational seed is recorded; these deterministic classifier procedures are not repeated under arbitrary seeds as if that created independent evidence.

Total: **560 fits** = 8 procedures × 10 outer folds × (2 candidates × 3 inner fits + 1 refit). Individual fit/prediction checkpoints support compatible resumes. Patient membership, labels, feature matrices, training weights and scaler moments are hashed. Final validation reconstructs inner scores, verifies selected settings, reloads each saved outer model and checks its predictions and all pooled patient metrics.

The primary comparisons use the same learner's statistics control; adaptive geometry is also compared with fixed geometry. Historical ResNet mean F1 0.550 is context, not a matched control: aggregation, classifier, feature count and input context differ. All class-stratified paired patient-bootstrap intervals are descriptive, conditional on fitted predictions, without refits or adjustment for repeated development, correlated folds or multiple comparisons.

## Run and artifacts

```bash
.venv/bin/python -m pip install -e '.[neural,geometry,dev]'
.venv/bin/python -m mi_localization.geometry_bank
.venv/bin/python -m mi_localization.geometry_study
.venv/bin/python -m mi_localization.geometry_study --report
# Read-only saved checkpoint progress:
.venv/bin/python -m mi_localization.geometry_study --status
```

Config: `configs/geometry_vcg.yaml`. Cache: `data/processed/geometry_vcg_v1/`. Aggregate boundary report: `results/neural_geometry_v1/boundary_audit/REPORT.md`. Classifier report: `results/neural_geometry_v1/complete/REPORT.md`. Private review images, waveform tensors, patient features/predictions, raw inventories and fitted models remain local and ignored. Public results contain aggregate metrics and hashed fit audits.

## Interpretation and next gate

All 560 fits completed and the [final report](../results/neural_geometry_v1/complete/REPORT.md) validated. Linear adaptive geometry scored **0.467** versus its statistics control **0.445**, descriptive paired difference interval **[−0.061, +0.107]**. Tree statistics+geometry scored **0.461** versus statistics **0.409**, interval **[−0.024, +0.131]**. No contrast establishes a gain. These are different aggregation/classifier procedures from the historical 0.550 ResNet, so that score is context rather than a matched control.

The low boundary coverage and inconclusive feature gains do not justify adopting a geometry branch yet. The [separate shared-label objective study](territory_vcg_plan.md) tests the other proposed idea independently, without any delineation or geometry inputs. The source-label auxiliary objective described below is not part of this geometry screen.

A favorable feature-screen result would motivate a separate controlled neural test, not establish a breakthrough. Weak adaptive-boundary coverage must be addressed or made explicit before describing a physiological phase model. A negative result applies to these descriptors/aggregation/learners and does not prove all VCG geometry is uninformative.

The separate shared-label hypothesis would add auxiliary predictions derived from the source label names (anterior/inferior/lateral/septal descriptors). These would be label-derived supervision, not independently verified anatomical diagnoses. The original six-class endpoint must remain unchanged. It has not been fitted by this geometry screen.

No score of 0.80–0.90 is promised. The same PTB cohort has been repeatedly inspected; meaningful improvement still needs independent confirmation and potentially more consistently labeled patients. Hard patients and unfavorable results are retained.

Primary sources: [NeuroKit delineation documentation](https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-delineate), [NeuroKit 0.3.0 release](https://github.com/neuropsychology/NeuroKit/releases/tag/v0.3.0), [Martínez et al., wavelet delineation](https://pubmed.ncbi.nlm.nih.gov/15072211/), [Hasan et al., VCG loop variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC3498118/). Their results do not validate this XYZ adaptation or establish six-class localization performance.
