# Within-beat lead direction tensor results

Re-encoded all 55,863 measured XYZ beats, reconstructed all 120 matched contexts, restored 210 source ridge references and replayed 630 new classifiers across all 164 target patients.

| Arm | Mean member macro F1 ± seed SD | Three-seed ensemble macro F1 |
|---|---:|---:|
| baseline | 0.572860 ± 0.019471 | 0.584266 |
| magnitude | 0.495324 ± 0.014521 | 0.517919 |
| sign_scrambled | 0.522525 ± 0.024152 | 0.546633 |
| signed | 0.517650 ± 0.020514 | 0.531364 |

The primary comparison is signed minus magnitude.

| Paired comparison | Ensemble difference | Descriptive patient 95% interval |
|---|---:|---:|
| signed_minus_magnitude | +0.013444 | [-0.027522, +0.054415] |
| signed_minus_sign_scrambled | -0.015269 | [-0.066919, +0.034030] |
| signed_minus_baseline | -0.052903 | [-0.122082, +0.013902] |
| magnitude_minus_baseline | -0.066347 | [-0.123440, -0.011423] |
| sign_scrambled_minus_baseline | -0.037633 | [-0.088823, +0.011350] |

| Class | baseline | magnitude | sign_scrambled | signed |
|---|---:|---:|---:|---:|
| AMI | 0.344828 | 0.312500 | 0.322581 | 0.312500 |
| ALMI | 0.482759 | 0.275862 | 0.370370 | 0.400000 |
| ASMI | 0.633333 | 0.500000 | 0.576271 | 0.533333 |
| IMI | 0.542373 | 0.535714 | 0.551724 | 0.566667 |
| ILMI | 0.577778 | 0.577778 | 0.553191 | 0.488889 |
| HC | 0.924528 | 0.905660 | 0.905660 | 0.886792 |

The 144 directional tensor columns summarize within-window lag covariance. The three added arms share the symmetric block, the original convolution base, training-only scales and the same alpha budget. The scrambled arm flips the antisymmetric entries per beat while preserving their magnitudes and the symmetric entries. All beats contribute and the patient summaries weight recordings equally. Reversal acts on the time index, and the lead coordinates retain the measured frame.

The post-R windows are fixed relative to the original R indices. The sign statistics are observed lead interactions. The paired patient intervals condition on fixed predictions of this repeatedly inspected PTB development cohort. They omit refits and repeated-development effects.

[Protocol](../../../docs/lag_direction_tensor_plan.md)
