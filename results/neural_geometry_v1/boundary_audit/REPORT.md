# VCG-only automatic boundary and geometry audit

164 unchanged patients, 391 records and 55,863 original R positions. Every recomputed Frank crop exactly reproduced the original VCG cache.

Accepted multi-axis boundary consensus: **12,977/55,863 (23.2%)**. Remaining beats use the declared fixed fallback; no patient or beat is excluded.

NeuroKit2 0.3.0 DWT is run on each record's three oriented/scaled Frank axes using supplied original R positions. At least two complete plausible axis estimates must agree within 40 ms for QRS and 80 ms for T. These are engineering checks, not accuracy against manual annotations. No clinical boundary ground truth is available here.

Fallback windows start at R−60/R+80/R+140 ms, ending at min(R+650 ms, next cached R−80 ms, recording end). Adaptive estimates stay within that same end ceiling. Phase durations, coordinates, integrals, speed, projected polygon area and orientation are retained. Cubic 11 ms smoothing precedes derivatives. Closed polygon area is a geometric summary, including for open physiological trajectories; it is not an anatomical area.

A beat's baseline is the XYZ median from R−200 to R−120 ms. The actual amplitude and axes are retained. Per-phase interpolation creates a 3 × 7 × 32 tensor of XYZ, physical-time velocity XYZ and magnitude; original durations remain separate descriptors. Fixed and adaptive paths have identical feature counts. All beats contribute to patient medians/IQR; longer recordings contribute more beats within a patient. Scaling/model fitting are deferred to training folds.

## Boundary coverage by source class

| Class | Beats | Accepted | Fraction |
|---|---:|---:|---:|
| AMI | 6333 | 1367 | 21.6% |
| ALMI | 6827 | 1021 | 15.0% |
| ASMI | 11489 | 1893 | 16.5% |
| IMI | 12410 | 2607 | 21.0% |
| ILMI | 8049 | 1458 | 18.1% |
| HC | 10755 | 4631 | 43.1% |

Consensus success/fallback flags are audit outputs, not classifier features. Class-dependent success may still affect adaptive descriptors; compare the fixed-window control. Source labels never enter delineation. The adaptive and fixed methods share an outer context ceiling but select different intervals within it; this is not a pure tensor-compression comparison.

The private `review/` packet has label-stratified examples chosen before classification using only IDs/labels, plus per-record/beat audits. Plots are not clinical adjudication. No hard cases are removed. Full-record denoising remains offline/noncausal; VCG is the only signal input.

References: [NeuroKit delineation](https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-delineate), [Martínez et al.](https://pubmed.ncbi.nlm.nih.gov/15072211/), [VCG loop variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC3498118/). Tool performance on ECG does not establish accuracy of this VCG adaptation.
