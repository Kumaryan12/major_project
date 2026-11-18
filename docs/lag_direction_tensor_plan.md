# Within-beat lead direction tensor

Test whether signed lead interactions at a delay add information to the strong convolution ridge baseline. This is a measured-XYZ experiment on 164 patients, 391 recordings and 55,863 original beats. The prior chronological studies changed the order of beats. This experiment studies the direction of lead interactions inside each beat. [Earlier dynamics proposal](research_frontiers_20261116.md) · [Related time-reversal research on ECG-derived series](https://arxiv.org/abs/2307.15998)

For each XYZ beat, use four fixed R-relative windows and lags of 4, 12, 24 and 48 ms. Center each lead by its window mean and compute the 3×3 lag covariance matrix from all valid pairs inside the window. Split it into symmetric and antisymmetric parts. Reversing the samples within the window transposes the lag matrix. It preserves the symmetric part and flips the sign of the antisymmetric part. The measurements remain in covariance units and do not rotate, warp or scale each beat. The signed terms describe directional statistics of the recorded leads, with no causal anatomical interpretation.

The per-beat tensor has axes window × lag × component, with 4 × 4 × 9 entries. The six unique symmetric entries stay identical in every added branch. The other three are the signed antisymmetric terms, their magnitudes, or the signed terms multiplied by a common random sign for each beat. Signs are balanced within each recording, with the extra sign chosen randomly for odd counts. The same sign applies to every delay, window and lead pair of that beat. This is a selective tensor control for direction coherence. It gives each beat the same magnitudes and symmetric interactions as the signed arm. Mean-pool all original beats within each recording, then weight a patient's recordings equally.

| Arm | Added to the complete baseline |
|---|---|
| Baseline | Original 3,584 columns |
| Magnitude | 144 columns containing the shared symmetric entries and absolute antisymmetric entries |
| Sign scrambled | Same width and symmetric entries, with per-beat sign reversals |
| Signed | Same width and symmetric entries, with true antisymmetric signs |

The two ridge alpha candidates [1,100] are fixed before fitting. A common training-only scale per auxiliary column is estimated from pooled magnitude features for all three arms. Its floor is the larger of 10% of training RMS or the absolute numerical floor. Ridge handles centering with a weighted intercept. A fixed sqrt(0.25 × 3584 / 144) factor sets the auxiliary block's scale relative to the base. The primary comparison is signed minus magnitude. Signed minus sign scrambled tests whether record-level sign coherence helps, and signed minus baseline tests whether the new tensor adds value over the unmodified strong reference.

Ten original patient folds, three inner folds and all three feature seeds. All preprocessing with fitted statistics and alpha selection occurs in the exact training scope. No new clinical boundary estimates, repeated-beat removal or other labels. Report ensemble and mean member macro F1 separately, with all seed and class results and descriptive paired patient intervals. The planned budget is 630 new ridge fits, 210 baseline restorations and 120 contexts. Independent replay re-encodes all original beats, reconstructs scales and sign controls, restores every saved fit and reproduces baseline metrics. This is an inspected PTB development study.

```bash
.venv/bin/python scripts/run_lag_direction_tensor.py --smoke
caffeinate -i .venv/bin/python scripts/run_lag_direction_tensor.py
```

The smoke evaluates one real inner partition and repeats the saved-fit replay without using outer-test predictions. The full runner verifies the report and exits.
