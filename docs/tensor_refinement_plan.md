# Signed tensor compression and final classifier comparison

## Completed findings and decision

All **2,820 fit slots** completed and passed independent report validation: **2,100 fresh inner fits, 660 verified reused probes and 60 fresh outer refits**. All 171 tests pass. Every saved final model reproduced its held-out predictions. The [full report](../results/tensor_refinement_v1/complete/REPORT.md) retains every predefined model and class result.

| Feature procedure | Linear macro F1 | Shallow-tree macro F1 |
|---|---:|---:|
| Statistics control | 0.432 | 0.440 |
| Original feature procedure | **0.498** | 0.420 |
| Expanded tensor procedure | 0.425 | **0.452** |

The expanded linear procedure fell **0.073** below its original-feature control (descriptive paired interval **[−0.144, −0.005]**). The expanded tree procedure was **0.012** above statistics (**[−0.074, +0.106]**) and **0.032** above its original-feature control (**[−0.038, +0.103]**). This establishes no repeatable accuracy gain from the expansion. The original linear feature procedure's +0.066 over statistics also has an interval including zero (**[−0.031, +0.165]**). These intervals are conditional development diagnostics, not formal independent significance tests.

The fidelity gain was large: global DCT16→64 reduced mean-record denoised relative squared reconstruction error **0.1527→0.0025** and central-window RMS error **0.14035→0.01573 mV**. The [fidelity plot](../results/tensor_refinement_v1/bank_audit/compression_fidelity.png) separates relative loss from absolute amplitude. The filter residual's relative error remains high even though its absolute error is small. Better reconstruction did not imply better localization.

Mean selected inner scores rose **0.476→0.500** for linear and **0.456→0.489** for trees when the pool expanded, while outer performance did not improve consistently. This is consistent with unstable selection in a small development cohort; it does not prove a single causal explanation. Do not choose the expanded tensor based on its higher optimized inner score. Linear original features remain the reference **within this feature protocol**; the historical neural 0.550 ± 0.040 is a different protocol and remains unchanged.

Class errors also matter: the expanded tree achieved IMI/ILMI F1 0.533/0.591 but AMI F1 only 0.069. Its overall 0.452 must not be presented as reliable localization across all MI classes. No method reached the 0.80–0.90 target.

**Decision:** preserve these representations as completed ablations and do not adopt the expanded candidate pool as an improvement. For another feature-design iteration, a bounded next hypothesis is training-patient bootstrap stability and group-aware redundancy selection within a fixed small representation pool, compared with the original selector. This would test selection reliability rather than keep expanding resolution. Such a method is not implemented or validated by this study. A new selection policy must be specified and refitted inside the training folds, and favorable results still require new-cohort confirmation.

## Reason for this refinement

The first training-only feature screen selected signed morphology in 7/10 contexts and the combined pool in 3/10. Across the ten selected packages, 210 of 440 feature occurrences came from the **filter residual R−D**. This can represent removed physiology, acquisition effects or artifacts; occurrence frequency is neither clinical importance nor independent stability. The truncated global DCT16 summary retained about 70% energy under its original aggregate diagnostic. Complete tensors reconstructed their input, but the summaries did not.

The next fixed comparison tests whether more temporal resolution or a band-aware signed representation helps, while retaining the original feature procedure and statistics controls. No outer result from this new study is used to change its candidates. Existing paper reproduction, neural runs, feature screen and all source caches remain unchanged. The paused multiview neural sweep is not restarted.

## New tensor representations

All inputs remain the same measured XYZ raw R, denoised D, recording median M, beat residual D−M and filter residual R−D. Templates use the same recording's unlabeled beats, including held-out exam beats for their own context. This is offline within-exam processing, not causal prediction. All original patients, recordings, labels and R positions are retained.

| Added family | Patient features before selection | Purpose |
|---|---:|---|
| denoised_dct32 | 192 | D-only control at rank 32 |
| denoised_dct64 | 384 | D-only control at rank 64 |
| all_dct32 | 960 | All five views at rank 32 |
| all_dct64 | 1,920 | All five views at rank 64 |
| wavelet_core16 | 2,400 | Signed native-band time core, rank 16 |
| wavelet_core32 | 4,800 | Signed native-band time core, rank 32 |

Global DCT uses a shared fixed orthonormal cosine basis across the original 651 samples, so coefficients remain signed and comparable across patients. Larger ranks expand the candidate temporal bandwidth; they do not automatically preserve useful pathology or improve classification. Denoised-only controls use exactly the same coefficients as the D slot in their matched all-view representation.

Native-wavelet cores first use sym4 DWT at level 4 with symmetric extension to form **A4, D4, D3, D2, D1** coefficient arrays. These have different native coefficient-time resolutions/lengths. Apply the same fixed signed cosine basis along each band's own coefficient-time axis and retain 16 or 32 coefficients. This gives a rectangular core **[view, XYZ, band, compressed coefficient-time]**. It is a fixed partial time projection, not learned Tucker/HOOI, per-beat SVD, frequency-adaptive clinical delineation or a common physiological phase grid. The retained rank is valid for every native coefficient array. Boundary redundancy means coefficient energy is not a simple partition of waveform energy.

Feature aggregation exactly follows the prior screen: beat median/IQR within a record, then coordinate median of record summaries within each patient. The original global DCT16 feature matrix must reproduce **bit for bit** before the new bank can be accepted. Feature names retain view, XYZ, band, coefficient order and median/IQR provenance. No labels enter tensor construction or these fixed transforms.

The fidelity audit measures per-view relative squared waveform reconstruction error and absolute RMS error in the original fixed windows, including the central R region. Native cores reconstruct through inverse DCT in each band's native grid and inverse wavelets. Zero-energy lead-beats are excluded from relative-error denominators, while absolute error remains reported. Diagnostics average records equally. These are fidelity measures, not clinical signal-quality ground truth or localization accuracy. Source waveforms remain intact and all summaries are explicitly lossy.

## Frozen nested classifier protocol

Same six original classes, **164 patients / 391 records / 55,863 beats**, ten frozen outer patient folds and the same three inner patient partitions. Each row now represents a patient, with equal-record robust aggregation. This differs from historical mean beat-probability neural evaluation, so matched controls are essential.

Two learners are tested separately:

- **Linear:** class-balanced L2 logistic regression, C 0.1 or 1.0, matching the first screen exactly.
- **Shallow tree:** class-balanced XGBoost, depth 2 or 3, 100 trees, learning rate 0.05, lambda 1, full row/column sampling, one CPU thread. Inner/final seeds follow the fixed fold seed rule; there is no favorable-seed search.

For each learner, evaluate three complete procedures:

1. **statistics:** existing statistics only, training-constant columns removed, tune C/depth inside inner folds. No supervised feature ranking.
2. **legacy:** select among the original 22 candidates (five feature pools × feature budgets 24/64 × two model settings, plus two unselected-statistics controls).
3. **refined:** retain all original candidates and add the six new families × two feature budgets × two model settings, giving **46 candidates**. The original candidates remain available if extra detail does not help.

Every inner-training fit learns class-balanced feature scaling, removes its own constant columns, ranks by unweighted ANOVA F score and greedily skips features with absolute training Pearson correlation ≥0.95 to a selected column. Selected budgets are at most 24/64. The unselected control omits ranking/pruning. Class weights enter scaling and classifier fitting once. Family/budget/model setting is chosen by mean inner patient macro F1; exact ties favor configured order. Only then does a fresh selector and classifier fit the outer-training patients and predict the held-out outer patients. No global feature mask or outer-optimized selection is used.

There are **2,820 fit slots**: 2 learners × 46 candidates × 3 inner splits × 10 outer folds + 6 procedures × 10 fresh outer refits. **660 linear inner fits are reused** from the completed first screen after independent checks of source/runtime, train/validation membership, fitted scaling/ranking/selection and prediction hashes. A CPU regression confirms identical old/new linear fit probabilities and selection audits. Expected new work is **2,100 inner fits + 60 outer refits = 2,160 fresh fits**. Every legacy tree and all new-feature fits are fresh.

Final results report all six procedures, pooled out-of-fold patient macro F1, accuracy, class F1 and family selections. The six predefined descriptive contrasts are legacy−statistics, refined−statistics and refined−legacy separately for each learner. Paired class-stratified patient bootstrap intervals condition on fixed fitted predictions, without refits or adjustment for correlated folds, multiple contrasts or repeated development. Outer folds have been repeatedly inspected in this project and remain development data. Any favorable result requires a fixed new-cohort confirmation; no 0.80–0.90, novelty, SOTA or clinical-validity promise.

## Validation and reproducibility

```bash
.venv/bin/python -m mi_localization.tensor_refinement_bank
.venv/bin/python -m mi_localization.tensor_refinement_study
.venv/bin/python -m mi_localization.tensor_refinement_report
tail -f tmp/tensor_refinement_run.log
cat results/tensor_refinement_v1/complete/progress.json

# Combined guarded run and publication after independent report validation:
caffeinate -i /bin/zsh scripts/run_tensor_refinement.sh --publish > tmp/tensor_refinement_run.log 2>&1
```

Config: `configs/tensor_refinement.yaml`. New bank: `data/processed/tensor_refinement_v1/`. New aggregate output: `results/tensor_refinement_v1/`. Guarded record caches and inner/outer predictions resume only with matching data/config/code/runtime. Saved progress is not by itself proof of a live process. Final reporting reconstructs all training selectors, verifies every inner choice, checks prediction membership and hashes, and requires every saved final model to reproduce its outer predictions. Waveforms, features, models, patient prediction files and private mappings stay local. Public results contain aggregate metrics, feature-name selections and fit audits.

Sources: [SciPy orthonormal DCT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html), [PyWavelets DWT](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html), [scikit-learn feature selection and leakage](https://scikit-learn.org/stable/common_pitfalls.html). These methods motivate the implementation, not a guaranteed improvement.

Reproduce the saved fidelity figure with `.venv/bin/python scripts/plot_tensor_fidelity.py`.
