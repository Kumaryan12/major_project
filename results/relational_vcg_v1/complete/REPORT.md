# VCG relational tensor development: completed results

Measured Frank XYZ is the sole input. ECG + VCG remains a separate, **not yet implemented** extension.

All five arms completed all 10 frozen patient folds: 164 patients, 55,863 beats, 6 classes. Scores below use pooled out-of-fold patient predictions, not an average of fold F1 scores.

These previously inspected PTB folds are a **development benchmark**, not untouched confirmation. No arm reached 0.80 macro F1 in this study; 0.80–0.90 remains an aspirational target.

## Results

| Representation | Patient macro F1 | Accuracy | Δ F1 vs short control | Paired 95% interval for Δ |
|---|---:|---:|---:|---:|
| Short-beat statistics (control) | 0.469 | 52.4% | +0.000 | — |
| Full-record wavelet statistics | 0.421 | 50.0% | -0.047 | [-0.115, +0.021] |
| Uncompressed correlations + context statistics | 0.459 | 54.9% | -0.010 | [-0.086, +0.066] |
| Shared correlation cores + context statistics | 0.413 | 50.6% | -0.056 | [-0.118, +0.005] |
| Shared covariance cores + context statistics | 0.470 | 54.9% | +0.001 | [-0.069, +0.071] |

Intervals use 5,000 class-stratified paired patient bootstrap draws. They are conditional on these fitted cross-validation predictions: models are not refitted, and intervals do not adjust for repeated development, correlated training folds or multiple comparisons. Do not treat them as independent evidence or a formal significance test.

## Class-level F1

| Class | Patients | Short control | Context statistics | Flat correlation | Shared correlation | Shared covariance |
|---|---:|---:|---:|---:|---:|---:|
| AMI | 17 | 0.357 | 0.276 | 0.231 | 0.148 | 0.231 |
| ALMI | 16 | 0.471 | 0.323 | 0.424 | 0.364 | 0.471 |
| ASMI | 27 | 0.316 | 0.364 | 0.483 | 0.351 | 0.500 |
| IMI | 29 | 0.418 | 0.361 | 0.413 | 0.373 | 0.406 |
| ILMI | 23 | 0.439 | 0.409 | 0.333 | 0.409 | 0.372 |
| HC | 52 | 0.812 | 0.796 | 0.868 | 0.833 | 0.838 |

## Implementation and fair comparison

- Same R peaks, 651-sample beat windows, six classes and patient folds for every arm.
- Every arm uses fixed depth-3, 250-tree XGBoost, class/patient-balanced training weights and mean beat-probability patient aggregation.
- XYZ × (original + D3–D9) gives 24 channels. Four R-relative regions give 24 × 24 × 4 correlation/covariance tensors. Regions are approximate, not validated QRS/ST/T delineations.
- Bands are calculated over the full denoised recording before cropping beats. This is offline, noncausal processing; it is not a streaming system.
- A single shared channel basis compresses both matrix axes, leaving regions separate. Covariance scaling and bases are fitted on each training split only.
- Three inner patient folds select rank 4 or 8; no outer test patient enters a projection fit or rank selection. Ranks yield 40 or 144 core features, plus 192 context statistics.
- The uncompressed arm has 1,200 relational features + 192 context statistics. Statistics-only controls have 192 features.
- Correlation omits absolute amplitude; covariance retains it relative to training-only channel scales. All relational arms also retain amplitude-bearing statistics.
- Existing paper reproduction, nested depth-3/6 benchmark and PTB-XL external results were not overwritten or rescored. The historical 0.487 nested model result is a different classifier protocol, not this study's control.

## Inner-selected ranks

- Shared correlation cores + context statistics: rank 4 in 1/10 outer folds, rank 8 in 9/10 outer folds.
- Shared covariance cores + context statistics: rank 4 in 2/10 outer folds, rank 8 in 8/10 outer folds.

## Interpretation and next experiments

This is a completed test of a sepsis-paper-inspired representation, not a reproduction of sepsis results or an established novelty/SOTA claim. A representation must improve the paired control before investing in a larger tuning budget. Retain unsuccessful results as ablations.

Next VCG-only hypotheses are morphology-preserving region features and multi-beat aggregation, then a regularized waveform encoder with training-patient-only pretraining. Prespecify each candidate and select it inside patient-grouped validation. Do not optimize against PTB-XL's previously evaluated test split. ECG + VCG must be reported separately with a matched VCG-only control.

An improvement claim ultimately needs a newly reserved independent cohort or external site. A fresh random split of these same repeatedly studied PTB patients is not independent confirmation.

## Reproducibility

```bash
.venv/bin/mi-localization relational-features
.venv/bin/mi-localization relational-study
.venv/bin/python -m mi_localization.relational_report
```

Configuration: `configs/relational_vcg.yaml`. Per-fold aggregate audits, confusion matrices, metrics and bootstrap intervals are in this directory. Patient prediction CSVs, feature banks and beat-probability checkpoints stay local and are not published to GitHub.

Study fingerprint: `5a058ca97c07a749157e9e0e09412a5340f1447859927d016f85597f0d594325`.

## Research basis

Nesaragi, Patidar and Thangaraj (2021), *A correlation matrix-based tensor decomposition method for early prediction of sepsis from clinical data*, [DOI: 10.1016/j.bbe.2021.06.009](https://doi.org/10.1016/j.bbe.2021.06.009). The proposed MI tensor and training-only shared projection are adaptations; the sepsis paper's metrics do not establish an achievable MI macro F1 target.
