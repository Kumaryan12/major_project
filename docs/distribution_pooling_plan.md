# Prespecified all-beat morphology distribution study

Authorized 17 November 2026 after the independently completed supervised-transfer study (`46612ff`). The user accepted the proposed four-arm distribution comparison. This stage uses only the original measured Frank XYZ target cohort. It starts no acquisition-bridge, synthetic-source or dynamical-feature experiment.

## Question and matched controls

Does retaining the distribution of beat morphology improve six-class patient macro F1 beyond mean and spread? The primary comparison is **quantiles minus mean_spread**, using equal-probability ensembles over all three declared feature-projection seeds. Median and mean controls are secondary comparisons, with no outer-score winner substitution.

Every original beat enters a fixed feature extractor. In five half-open sample windows `[0,200)`, `[200,310)`, `[310,450)`, `[450,651)`, and `[0,651)`, each signed XYZ lead contributes eight orthonormal DCT-II coefficients divided by sqrt(window length), RMS, mean absolute amplitude, minimum and maximum. This gives 36 physical/DCT features per window. The final full-beat window overlaps the preceding four. Windows are approximate R-relative coordinates, not clinically adjudicated phases.

| Arm | Per-record representation |
|---|---|
| median_waveform | Extract/project features of the coordinate-median waveform of every original beat, then repeat on the quantile grid |
| mean_embedding | Extract/project every original beat, then repeat its mean on the grid |
| mean_spread | Use only the projected mean and population SD, represented as `mu + sigma * Phi_inverse(q)` |
| quantiles | Empirical linear-interpolation quantiles of all projected beat features |

The fixed grid is q = 0.1, 0.2, ..., 0.9. Each window has 16 unit-norm Gaussian feature projections, drawn from seeds42/123/2026 independently of all labels/data. They mix features, without rotating physical XYZ or imposing per-beat unit-amplitude scaling. Projections for a seed are shared across partitions, as data-independent constants; learned normalization is fresh for each exact training partition.

The moment arm is a linear encoding of mean/SD on the same quantile grid. It does not assert Gaussian cardiac physiology. It controls whether empirical distribution shape helps beyond those two moments. All four tensors have **window5 × projection16 × quantile9 =720** entries per recording, divided by sqrt(9). Repeating a mean then dividing by sqrt(Q) preserves its Euclidean kernel. Do not subsequently apply separate per-arm column standardization, which would undo that control. Effective ranks still differ and equal stored width is not identical model capacity. No Tucker/SVD compression or feature selection is added in this first screen.

## Normalization, aggregation and unchanged target

Training-only morphology means/scales give equal total weight to each training patient, equal weight to their recordings and equal weight to all beats within a recording. Use a stable two-pass variance calculation and fixed1e-8 feature-scale floor. Query patients never contribute to fitted moments or references. Fixed deterministic feature extraction can be cached across the cohort because it fits no population state.

All retained recordings contribute equally to each patient's pooled feature row. Quantiles describe variation within each recording before patient pooling. The method retains neither a learned recording-order axis nor chronological beat order. It does not create new independent patients, and may preserve noise as well as disease morphology.

Keep164patients,391records,55,863beats, original labels/R positions and `configs/patient_folds.csv`. Classes remain AMI/ALMI/ASMI/IMI/ILMI/HC. Keep signed/amplitude-bearing measured XYZ, original651samples at1000Hz. No quality filtering, relabeling, beat confidence selection, source-encoder reuse, ECG channels or external patients.

## Classifier, budget and endpoint

One class-balanced RidgeClassifier, intercept=True, Cholesky solver, **alpha[1,100]** in every arm. Shared preprocessing already occurs in the morphology space, so no separate per-arm scaler. Three fixed inner patient folds select alpha by highest mean inner patient macro F1, with configured-order ties; a fresh refit uses all outer-training patients. Ten frozen outer patient folds and all three projection seeds. Softmax at fixed temperature1 converts ridge scores for probability averaging, without learned calibration or thresholds.

Declared budget: **120 feature contexts,840 classifiers and120 outer models**. The same four-arm features/memberships are checked in every context. Report all seeds and pooled outer patient predictions. Report mean member F1±projection-seed SD separately from equal-three-seed ensemble F1. No subset/seed/weight search. Secondary contrasts: quantiles−mean, quantiles−median, moments−mean, mean−median. Use5,000 shared class-stratified paired patient draws, resampling a patient once rather than treating seed predictions as independent patients. Intervals are conditional/descriptive without refits or correction for overlapping folds, repeated development or multiple comparisons.

## Integrity, smoke and stopping boundary

New modules `distribution_features/bank/study/report.py`, config `configs/distribution_pooling_v1.yaml`, separate bank/output namespaces. Preserve all older modules/manifests/results. Scope pins source/config/runtime/folds, raw feature/median banks and helpers. Save every fit, training-only transform, tensor and query prediction privately under ignored review directories. Compatible resume restores only matching completed contexts; source/config/runtime drift requires a new namespace.

Before full training, meaningful synthetic tests cover a DCT cosine oracle, amplitude/sign behavior, distributions with identical mean/variance, quantile interpolation and kernel normalization, equal-patient/record weights, query isolation, constant features, altered outer labels, source/cache/membership tampering and actual tiny nested fitting/restoration without classifier refitting in the independent report. Real-data smoke uses only inner train/query patients inside outer fold1's training partition, restores all four arms and reconstructs its transform. A smoke is not a performance result.

The separate report independently re-extracts all original beat and median features, refits every training-only normalizer from exactly its training patients, restores transforms/tensors, restores every inner/outer classifier and query prediction, replays every alpha choice and verifies all840models/120contexts. Public artifacts are aggregate metrics/confusions and hashed audits. Signals, fitted states, feature arrays and individual predictions stay ignored. Runner holds an OS lock, can publish only verified aggregates, then stops. No following sweep is queued. Windows requires its own compatible bank/output namespace and remains unvalidated.

Commands:

```sh
.venv/bin/python -m mi_localization.distribution_study --smoke
.venv/bin/python scripts/run_distribution_pooling.py --publish
.venv/bin/python scripts/distribution_pooling_status.py --watch
```

## Interpretation and prior art

If quantiles exceed median/mean but not mean-plus-spread, simple variability may explain the point result. If they reproducibly exceed moments, that supports investigating distribution shape. Neither outcome independently confirms generalization or novelty. Keep historical neural ensemble0.584570 and convolution ridge0.584266 as separate references, rather than matched controls. The completed certain-source transfer ensemble0.573361 is also separate. This repeatedly inspected PTB cohort remains development data, with no SOTA, clinical-validity, ceiling or promised0.80–0.90 claim.

This is a compact projected-quantile adaptation, not exact PSWE/FSW or an inherited injectivity theorem. Research basis: [Pooling by Sliced-Wasserstein Embedding, NeurIPS2021](https://papers.nips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html), [Fourier Sliced-Wasserstein Embedding, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3dbb8b6b5576b85afb3037e9630812dc-Abstract-Conference.html).
