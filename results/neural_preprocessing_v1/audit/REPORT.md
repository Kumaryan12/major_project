# PTB waveform preprocessing audit

164 patients, 391 records, 55,863 unchanged cached beats.

All header-derived labels agree with cached labels. Crop boundaries are complete, cached R positions are unique per record, and recomputed denoised crops reproduce the retained cache exactly. This is consistency checking, not independent clinical or detector validation.

| Class | Patients | Beats | Median centered relative L2 change | Median centered RMS ratio (denoised/raw) | Median derivative RMS ratio |
|---|---:|---:|---:|---:|---:|
| AMI | 17 | 6333 | 0.037 | 0.990 | 0.892 |
| ALMI | 16 | 6827 | 0.039 | 0.990 | 0.879 |
| ASMI | 27 | 11489 | 0.038 | 0.990 | 0.876 |
| IMI | 29 | 12410 | 0.041 | 0.989 | 0.896 |
| ILMI | 23 | 8049 | 0.045 | 0.989 | 0.889 |
| HC | 52 | 10755 | 0.020 | 0.995 | 0.964 |

## Screening counts—not error diagnoses

Among 55,472 consecutive cached-peak intervals, 713 are shorter than 300 ms and 176 exceed 2,000 ms. These may reflect rhythm variation or detector behavior; there are no independent annotations to decide which.

1,157 intervals place the next cached peak at or before the +400 ms crop endpoint. This describes neighboring-peak overlap, not a confirmed false R peak or a measured T-wave truncation rate.

Raw/denoised review plots were selected for 12 patients using a fixed class-balanced seed, not model errors. Local-only plots and an UNFILLED adjudication form are in `review/`. No expert adjudication or independent peak/boundary annotations were supplied. The ±wider context and shaded original crop enable review; no sensitivity/PPV or phase-coverage accuracy is claimed.

## Interpretation

Changes in derivative energy and amplitude quantify what denoising does, not whether it helps localization. Centering is used only for these diagnostics, not to impose per-beat normalization in the model. Edge-energy fractions are not phase delineations. Raw versus denoised classification needs a matched patient-separated experiment; longer windows should not be added or declared better from these screening counts alone.

No data were excluded, re-detected or relabeled. Raw XYZ crops retain the original ordering and beat positions. The raw cache is newly created; the denoised cache and all completed experiments are unchanged. Raw signals and review plots remain local; public outputs contain aggregate diagnostics only.

Sources: [PTB database](https://physionet.org/content/ptbdb/1.0.0/) and [PyWavelets thresholding documentation](https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html). The existing pipeline soft-thresholds detail coefficients across all used decomposition levels; these sources do not establish whether that choice is optimal for MI localization.
