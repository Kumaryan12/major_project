# Tensor construction before final model training

The next stage is **construct and audit tensors → extract candidate feature families → select inside training patients → freeze the procedure → evaluate final classifiers**. The long six-model neural comparison was stopped on 15 November 2026 to follow this direction. Its checkpoints remain local and resumable: seed 42 has six completed model/fold checkpoints for outer fold 1 and the original baseline for outer fold 2. Those partial results are not a final comparison. Do not restart `run_multiview_vcg.sh` while this feature-design stage is running.

## Stage 1: explicit, auditable tensors

Use the existing five physical views: raw R, denoised D, recording median M, beat residual D−M and filter residual R−D. M uses only the same recording's unlabeled beats. All held-out recording beats are available for its own template, so this is offline within-exam processing. No cohort/label-conditioned template dictionary is learned. Both residuals may contain genuine pathology; neither is certified noise.

Build **T[beat, view, XYZ, wavelet band, original time]**. Current size is **55,863 × 5 × 3 × 5 × 651**. Each view uses `sym4`, level 4, DWT multiresolution analysis with symmetric extension, producing **A4, D4, D3, D2, D1**. All bands are retained and their sum is checked against the input view in float64. This is a complete additive multiresolution representation, not the paper's truncated Tucker core or a learned tensor model. The level is within the wavelet's supported crop length; the existing high-level bior6.8 statistics remain a clearly identified baseline.

The tensor is constructed in bounded chunks, and compact per-record feature caches allow resumption. Source waveforms, original R positions, all patients and all beats remain unchanged. No quality deletion, time warping, spatial rotation, per-beat amplitude scaling or automatic relabeling is introduced. Full-data descriptive audits never determine the training selector.

## Stage 2: candidate features

| Family | Features per patient | What it tests |
|---|---:|---|
| Existing wavelet statistics | 384 | Matched aggregation control |
| Signed morphology | 480 | First 16 orthonormal DCT coefficients per view/XYZ |
| Multiscale windows | 1,200 | Signed mean and physical RMS per view/XYZ/band/window |
| XYZ relations | 1,800 | Six covariance and three off-diagonal correlation entries per view/band/window |
| Combined | 3,864 | Whether complementary summaries survive compact selection |

For every family, first compute coordinate median and IQR across beats **within each recording**, then take the coordinate median across that patient's record summaries. Each recording has equal influence at the second stage. This differs from previous all-beat pooling, so comparisons must use the newly aggregated statistics control. One row is one independent patient, not one beat. None of these fixed feature operations uses diagnostic labels.

The DCT basis is shared and signed across all beats, avoiding incomparable per-beat SVD axes. Its truncation is lossy and can miss fast QRS detail; other feature families retain band-specific amplitude/relationships. High retained energy alone does not imply retained diagnostic information. Covariance preserves physical amplitude dependence, while correlation normalizes XYZ relationships. Undefined zero-variance correlations are encoded as zero. Covariance entries have squared-voltage units. These are unregularized empirical within-window covariance summaries, not an SPD manifold model or an implementation of the sepsis paper.

Fixed crop intervals are `[50,170)`, `[170,350)`, `[350,470)`, `[470,651)` at 1 kHz, with R at index 250: approximately −200 to −81 ms, −80 to +99 ms, +100 to +219 ms, +220 to +400 ms. They are named pre/central/early-post/late-post and **are not clinically delineated QRS/ST/T boundaries**. Morphology DCT and the complete tensor still include the full original crop. Source waveforms remain available even when selected feature summaries discard detail.

## Stage 3: small training-only probes

Do not select features on all patients and then cross-validate a classifier. For each of the ten frozen outer folds, only that fold's outer-training patients enter the design screen:

1. Make the original three patient-separated inner folds.
2. Fit class-balanced feature scaling using inner-training patients only.
3. Remove training-constant columns. For selection candidates, rank remaining columns by training ANOVA F score; greedily skip a column if its absolute training Pearson correlation with an already selected column is at least 0.95.
4. Keep up to **24 or 64** columns. Ranking is ordinary ANOVA; class balancing applies to scaling and probe fitting, not to ANOVA itself. F scores rank features without significance testing. Strongly redundant data may yield fewer than the requested budget.
5. Fit a compact logistic probe with C **0.1 or 1.0**; score inner-validation patient macro F1.
6. Select family/budget/C by mean inner score; exact ties favor configured order. Refit the selected scaler/selector on the whole outer-training partition and save its indices/scales as the training-ready package.

Also retain two unselected-statistics controls (C 0.1/1.0), removing training-constant columns but applying neither ANOVA selection nor correlation pruning. Five families × two budgets × two C values plus the two controls give **22 candidates**, hence **660 small inner probe fits = 10 outer-training contexts × 22 × 3**. There are **zero outer-test predictions** at this stage. The complete table includes every candidate, selected feature names, audit hashes, requested/actual dimensions and feature occurrence counts across the overlapping outer-training contexts.

The screen answers whether representations contain information accessible to this probe under the declared policy. It cannot identify a universally best tensor for every eventual nonlinear classifier. The selected inner scores are optimized diagnostics and must **not** be reported as final project macro F1, a gain over 0.550 or evidence for 0.80–0.90. The same PTB cohort has already been repeatedly inspected and remains development data.

## Stage 4: final training after inspection

Review the tensor audit, candidate results, retained dimensions and feature-name occurrences. Freeze the selection procedure, then compare final classifiers against matched controls in outer patient folds. Every fold must retain its own scaler, selector and any learned PCA/Tucker basis. Do not combine selected lists into a global feature mask and reuse it on these patients. Feature occurrence across highly overlapping training contexts is descriptive; it does not establish independent stability or clinical biomarkers.

This stage implements construction, audit and training-only selection. It does **not** automatically restart the expensive neural sweep, train a final classifier on outer patients, or claim independent confirmation. Favorable later results require a fixed new-cohort validation. No tensor can guarantee a target F1 by adding axes alone.

## Reproduce and inspect

```bash
.venv/bin/python -m mi_localization.tensor_feature_bank
.venv/bin/python -m mi_localization.tensor_feature_screen
cat results/tensor_features_v1/bank_audit/progress.json
cat results/tensor_features_v1/screen/progress.json
tail -f tmp/tensor_feature_screen.log

# Resumable combined workflow; optional publication after completion:
caffeinate -i /bin/zsh scripts/run_tensor_feature_screen.sh --publish > tmp/tensor_feature_screen.log 2>&1
```

Config: `configs/tensor_features.yaml`. Aggregate outputs: `results/tensor_features_v1/bank_audit/` and `screen/`. Patient features, record caches and fold-specific feature packages stay local under ignored `.npz` files. Source/config/runtime guards reject incompatible resumption. Completed probe prediction caches and fitted-selector audits are checked when resuming. Reports contain no outer-test performance claim. Saved progress needs a matching live process and current timestamp to establish that work is running.

## Method references

- [PyWavelets multiresolution analysis](https://pywavelets.readthedocs.io/en/latest/ref/mra.html): additive temporally aligned projected bands; this decomposition is redundant and does not provide SWT variance partition guarantees.
- [scikit-learn: leakage and feature selection](https://scikit-learn.org/stable/common_pitfalls.html): fit feature selection and other learned transforms inside training partitions.
- [scikit-learn: nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): model selection scores need a separate outer evaluation. This stage intentionally stops before that evaluation.
