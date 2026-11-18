# Paired STAFF III source transfer to measured Frank XYZ

Replayed 90 outer fits on 164 PTB patients, ten frozen outer folds and three seeds.

| Method | Mean seed macro F1 | Three-seed ensemble macro F1 |
|---|---:|---:|
| scratch | 0.541465 | 0.601801 |
| mean | 0.516059 | 0.555118 |
| change | 0.519206 | 0.539589 |

The primary contrast is change minus mean.

| Contrast | Macro F1 difference | Descriptive paired 95% interval |
|---|---:|---|
| change_minus_mean | -0.015528 | [-0.083389, +0.051662] |
| change_minus_scratch | -0.062211 | [-0.139503, +0.015059] |
| mean_minus_scratch | -0.046683 | [-0.119144, +0.026807] |

| Class | Scratch | Mean | Change |
|---|---:|---:|---:|
| AMI | 0.529412 | 0.352941 | 0.358974 |
| ALMI | 0.428571 | 0.400000 | 0.387097 |
| ASMI | 0.642857 | 0.440000 | 0.382979 |
| IMI | 0.535714 | 0.592593 | 0.610169 |
| ILMI | 0.595745 | 0.666667 | 0.590909 |
| HC | 0.878505 | 0.878505 | 0.907407 |

STAFF III native labels describe the vessel involved in an annotated balloon occlusion. Both source arms see identical before/during ECG windows. Source ECG uses Mason–Likar limbs and Kors-derived XYZ. The target is measured Frank XYZ with a fresh six-class head and all original beats. This is an ECG-assisted training extension on repeatedly inspected PTB development data. Paired intervals condition on the saved predictions. Recording-level screening leaves cross-dataset patient identity unresolved.

Historical measured-XYZ references were 0.601801 for waveform scratch and 0.603374 for selected linear convolution.

[Protocol](../../../docs/staffiii_pair_transfer_plan.md)
