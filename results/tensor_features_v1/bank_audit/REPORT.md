# Tensor construction audit

Label-free construction and fidelity diagnostics only; summaries are lossy; neither reconstruction nor retained energy is diagnostic accuracy.

Retained 164 patients, 391 records, 55,863 beats.

Tensor shape: [55863, 5, 3, 5, 651]; wavelet sym4, complete bands ['A4', 'D4', 'D3', 'D2', 'D1'], symmetric extension. Maximum additive reconstruction error 6.4e-12 mV.

| Feature family | Patient features |
|---|---:|
| statistics | 384 |
| morphology | 480 |
| multiscale | 1200 |
| relations | 1800 |
| combined | 3864 |

Within-record beat median/IQR, then equal-record coordinate median within each patient. Existing wavelet statistics use the same aggregation. No learned cohort transform, feature selection or classifier enters this stage. Full time traces remain in the source bank; feature summaries are deliberately lossy. Full-data constant-column counts are audit diagnostics only, never a selection mask.

DCT coefficients share a fixed signed basis across every beat; truncation loses temporal detail and energy retention does not certify disease preservation. Covariance retains amplitude; off-diagonal correlations add normalized XYZ relationships, with undefined zero-variance correlations encoded as zero. Fixed R-relative windows are not clinical boundaries. No per-beat amplitude division, quality deletion, alignment, phase warping or new labels.
