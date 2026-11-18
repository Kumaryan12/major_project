# R-relative convolution response pooling

Preserve the verified 3,584-column combined convolution baseline on the original signed Frank XYZ recording templates. Each template uses the coordinate median over its own original denoised beats. The cohort includes all 164 patients, 391 recording templates and 55,863 original beats. The target remains six-class patient macro F1 using the original ten frozen patient folds.

## Matched added blocks

Reuse the exact-partition training-fitted MultiRocket-style weights, thresholds and common waveform gain for raw and first-difference input at dilations 1 and 4. Summarize positive and negative thresholded response components with separate RMS values, normalized by the number of response centers. All arms use valid convolution centers and leave the original template unchanged. These RMS summaries add features beyond the baseline's positive count, mean, index and run length.

Use the four equal-width half-open windows from −200 to −64, −64 to +72, +72 to +208 and +208 to +344 ms relative to the original R peak. Pool by response center and use the original continuous convolution across window boundaries. First differences are located between their two samples. These are fixed R-relative windows.

| Procedure | Added features |
|---|---|
| Frozen baseline | None |
| Whole beat | Use 64 distinct kernels per raw/difference and dilation group. Pool the signed RMS over all valid centers. The first 16 kernels match the aligned branch. |
| Shuffled | Use the exact aligned window features, with an independent, seed-fixed slot permutation per recording, shared across all groups and statistics. |
| Aligned | Use 16 kernels per group and pool each kernel separately in all four windows. |

All three added blocks have 512 columns. Their patient features weight recordings equally. Every augmented arm receives the same base, equal auxiliary width, and the same learnable classifier and tuning budget. Aligned and shuffled use identical kernel responses and statistics. The whole-beat control spends the extra slots on more distinct kernels. Apply column standardization to patient features using each exact training partition. Preserve all signed lead coordinates and original R locations.

## Evaluation and verification

Primary contrast is aligned minus the equal-width whole-beat control. Aligned minus shuffled tests whether consistent position labels add value. Also report the matched untouched baseline and a descriptive comparison with the historical, separately selected 0.603374 ensemble. Report each seed, the mean seed macro F1, the three-seed ensemble macro F1, paired class-stratified patient intervals and all class F1 values, with AMI and ALMI emphasized.

Use all seeds 42, 123 and 2026 and all ten outer folds with three inner patient folds. Ridge alpha from {1, 100} uses inner patient macro F1 only. Restore the original baseline and verify its alpha choices and metrics exactly. An independent pass reconstructs all 120 contexts, replays 630 new and 210 baseline fits, checks feature and model evidence, and replays the historical selector's saved predictions. The original source waveform bank and completed experiments are read-only.

Run `.venv/bin/python scripts/run_r_relative_convolution.py --smoke` from the repository root. This uses only patients inside the first outer training partition. After smoke verification, `.venv/bin/python scripts/run_r_relative_convolution.py --publish` runs the declared comparison and publishes its verified aggregate evidence. The detached run logs to `tmp/r_relative_convolution_v1.log`. Read the final report under `results/r_relative_convolution_v1/complete`. Private model and patient files stay local.

[Backbone protocol](rocket_vcg_plan.md) · [Temporal ROCKET prior art](https://doi.org/10.1007/s10618-025-01162-y)
