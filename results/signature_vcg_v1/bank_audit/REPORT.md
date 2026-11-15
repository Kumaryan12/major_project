# Ordered VCG trajectory feature audit

Label-free construction and mathematical/sensitivity diagnostics only; no classifier score.

Retained 164 patients, 391 recordings and 55,863 original beats. Every denoised beat contributes. Record summaries are beat median/IQR; patients are equal-record coordinate medians.

| Family | Patient columns before appending morphology |
|---|---:|
| morphology | 480 |
| signed_area | 130 |
| endpoint_polynomial | 870 |
| signature2 | 230 |
| signature3 | 870 |

Paths use physical seconds and signed XYZ mV, exact piecewise-linear signatures through degree three, computed using balanced Chen products. Each fixed window includes physical XYZ means to retain offsets lost by increment signatures. All original morphology columns are copied unchanged. No rotation, resampling, quality deletion, new R positions or learned full-cohort basis is introduced.

The endpoint-polynomial control has the same column count as degree-three signatures, but replaces the path by its straight endpoint increment: it preserves endpoint polynomial terms and window means while removing interior ordered interactions. The signed-area control retains level-one increments, six time/XYZ pair areas and means. Signature2/3 keep all tensor words, including algebraic redundancies; train-only selection handles variable/redundant columns. Fixed windows are not clinical phases.

Sensitivity diagnostics use every recording median, not the classification feature matrix: raw versus denoised medians, and ±4-sample edge-extended shifts of denoised medians. Groups share physical units mV^q seconds^(degree−q). Relative errors use a 1e−12 denominator floor with near-zero counts reported. These are descriptive checks, not clean-signal accuracy, physiological importance or a reason to remove patients. No audit chooses a feature mask or classifier.
