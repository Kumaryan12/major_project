# Bounded random-convolution VCG feature study

Approved and implemented **16 November 2026**, following the completed staged-context comparison. The user approved the bounded current-data study, not a data expansion or unbounded architecture search. This first phase uses matched linear classifiers only; no TabPFN fitting, external data download or neural training is included.

## Question and controls

Do temporal-pattern features from a bounded MultiRocket-style transform, experimental multivariate HYDRA, or their concatenation improve six-class **patient macro F1** over a DCT representation of the same recording templates?

| Arm | Patient features | Dimensions at default settings |
|---|---|---:|
| `dct` | First 32 orthonormal DCT-II coefficients of each signed XYZ template | 96 |
| `multirocket` | 84 kernels × four dilations × two representations × four poolers | 2,688 |
| `hydra` | Seven dilations × original/difference × four groups per representation × eight kernels × two summaries | 896 |
| `combined` | Concatenate MultiRocket-style and HYDRA features | 3,584 |

Every arm uses training-only StandardScaler and the same class-balanced RidgeClassifier (`solver=cholesky`). Two alpha candidates **[1, 100]** are evaluated on the same three inner patient folds; maximum arithmetic mean inner macro F1 wins, with ties favoring the first configured alpha. Refit only the chosen alpha on outer-training patients. Feature dimensions and effective capacity differ, so this is a matched input/classifier/budget comparison, not a parameter-matched architecture test.

The historical **0.584570** neural ensemble remains a different frozen development reference, not the matched DCT control. The new DCT control is not claimed to reproduce the older five-view morphology feature procedure.

## Fixed input and aggregation

Keep the original **164 patients, 391 recordings, 55,863 denoised measured-Frank XYZ beats**, six labels `[AMI, ALMI, ASMI, IMI, ILMI, HC]`, R positions and frozen patient folds. The existing verified waveform/chronology banks are prerequisites; no completed classifier study or multiview sweep must be resumed.

For computational bounds, every original beat enters its recording's **coordinate-median beat template**, with all 651 samples retained. No waveform is chosen using confidence/quality/labels, and no alignment or redetection occurs. The template is a summary, not necessarily a physically observed beat. It discards within-record variability; a negative result would reject this template-based recipe, not all beat-level random-convolution features.

Extract record features, then take the **equal-record arithmetic mean** within each patient. Fit each classifier to one row per training patient. A query template depends only on that patient's own examination, with no across-patient aggregation. Inference remains offline/noncausal because original denoising/templates use an entire examination.

Each exact inner/outer partition fits one common positive RMS gain on its training templates, averaging energy equally over patients and then records. Divide all XYZ samples by this one scalar: no per-beat normalization, per-axis scaling, rotation, channel permutation or coordinate centering. Relative orientation/amplitude is preserved. Final feature-column scaling is fitted on training patient rows only.

## Declared transforms and departures from upstream defaults

**MultiRocket-style:** use all 84 length-nine patterns with three +2 positions and six −1 positions. For each original/differenced representation, use four integer log-spaced dilations **[1,4,18,81]**, one low-discrepancy quantile threshold per kernel/dilation, and randomly chosen nonempty XYZ channel subsets. Bias fitting chooses a training patient uniformly, then one of their recordings uniformly for each kernel; query signals never fit thresholds. Keep that same channel subset during bias fitting and transformation. Alternate padded/valid response supports by kernel/dilation parity.

Compute PPV, mean positive thresholded response, mean zero-based positive index (−1 if empty), and longest consecutive positive run in samples (0 if empty). A response exactly at threshold is nonpositive. These are explicit paper-style conventions. They are **not exact historical source-code parity** with the upstream implementation's `C+bias` positive-value accumulation or gap-based run accounting. The reduced dilation/threshold count, patient-uniform bias sampling, template input and NumPy/PyTorch implementation also differ from the default 49,728-feature recipe. It is labeled MultiRocket-style throughout, not a full reproduction or a new invention.

**HYDRA:** vendor the unchanged authors' **experimental multivariate** implementation at revision `144bb7aa3186654042f00f36b078370c620acb06`, with the upstream GPL-3.0 license/attribution in [_vendor/hydra](../src/mi_localization/_vendor/hydra/README.md). Use eight groups/eight kernels and power-of-two dilations through 64. Half the groups process each representation. Preserve its with-replacement channel sampling, magnitude-weighted maximum-winner sums and minimum-winner counts. A scoped Torch RNG creates weights/channel choices without changing the caller's RNG. Use common StandardScaler/ridge controls rather than the upstream SparseScaler recipe. This is an evaluated local use of an experimental extension, not a claim of upstream multivariate validation.

No new numerical packages, Numba dependency, binary replacements or changes to old pinned code are needed. Extraction is CPU-only, with two Torch threads, one BLAS thread and chunks of 32 records. These choices are fingerprinted in configuration/runtime scope.

## Budget, seeds and reporting

- **Three transform seeds: 42, 123, 2026**, all retained. Partition fit seed is `transform_seed + 1000*outer_fold + inner_fold`, with inner fold zero for the outer refit.
- Ten outer folds × three seeds × (three inner + one outer) = **120 independently fitted feature contexts**.
- Four arms × ten outer folds × three seeds × (three inner folds × two alphas + one outer refit) = **840 classifier contexts**, including **120 outer models**.
- All four arms receive fresh partition-local transforms; inner contexts never borrow an outer transform. The four arms share one fitted feature context per partition/seed, and both alpha candidates share its features.
- Report every seed, mean member F1/seed SD, and the **prespecified equal-probability three-seed ensemble** for every arm. Ridge scores use fixed softmax temperature one; these are not calibrated clinical probabilities. DCT is seed-independent and should produce identical member predictions; repeated DCT entries are one control predictor, not independent replications.
- Prespecified comparisons: each new feature arm minus DCT, combined minus MultiRocket-style, and combined minus HYDRA. Use 5,000 class-stratified paired patient draws conditional on fixed ensemble predictions. No refits, CV-dependence, repeated-development or multiple-comparison correction; report intervals accordingly.
- No best seed, favorable class subset, new thresholds, per-patient routing, extra feature-budget search or fitted ensemble weights. All six class F1 values remain visible.

The separate reporting process rebuilds each transform **from only its training templates**, checks state/threshold/scalar hashes, restores the saved transform and reproduces all patient features. It then restores **all 840 classifiers**, checks train-fitted scalers/settings and cached probabilities, reconstructs every inner alpha choice and covers all outer patients/recordings. Raw arrays, fitted weights/scalers, record/patient identifiers and individual predictions remain ignored. Public outputs are aggregate metrics, confusion matrices and hashed audits.

This PTB cohort has been repeatedly inspected and remains development data. Success requires reproducible gains over the matched controls and ultimately independent confirmation. The study cannot establish a dataset ceiling, clinical validity, originality, MI SOTA or a guaranteed 0.80–0.90 macro F1.

## Commands and implementation

```sh
.venv/bin/python -m mi_localization.rocket_study --smoke
.venv/bin/python scripts/run_rocket_vcg.py --publish
.venv/bin/python scripts/live_status.py --watch
```

Core: `rocket_bank.py`, `rocket_features.py`, `rocket_study.py`, `rocket_report.py`. [Config](../configs/rocket_vcg_v1.yaml). Local output: `results/rocket_vcg_v1`; template bank: `data/processed/rocket_templates_v1`; launcher log: `tmp/rocket_vcg_v1.log`. The launcher holds an OS lock, resumes only matching completed contexts, independently verifies the full study and publishes aggregate results from clean main. Saved PIDs alone do not prove a live run. Ctrl+C closes the monitor only. Windows uses `.venv\Scripts\python.exe` and its own compatible banks/output namespace; numerical Windows execution is unvalidated.

At implementation, eleven targeted tests passed, covering a slow NumPy convolution/pooling oracle, zero/equality/run cases, batching, train-only transforms and preserved RNG, template/aggregation boundaries, actual tiny nested fit/restore/resume, outer-label counterfactuals, tamper/partition rejection and monitor/publication guards. The actual-data smoke passed: 391 templates, feature dimensions 96/2688/896/3584, training-only transform reconstruction and all four classifier restorations. A smoke is not classification performance. The complete macOS regression suite also passed **245 tests** before full launch. Launch status is recorded in the handoff after it occurs.

## Primary research references

- [Tan et al., MultiRocket, Data Mining and Knowledge Discovery 2022](https://arxiv.org/abs/2102.00457), [authors' multivariate implementation](https://github.com/ChangWeiTan/MultiRocket/blob/main/multirocket/multirocket_multivariate.py).
- [Dempster et al., HYDRA, Data Mining and Knowledge Discovery 2023](https://arxiv.org/abs/2203.13652), [pinned multivariate source](https://github.com/angus924/hydra/blob/144bb7aa3186654042f00f36b078370c620acb06/code/hydra_multivariate.py).
- [Cüppers and Vreeken, MASHT preprint, 2026](https://arxiv.org/abs/2607.19234). Its MultiRocket/HYDRA plus tabular foundation-model combination is prior art. This first bounded phase does not reproduce MASHT, use TabPFN or inherit its benchmark scores.

Full study launched **16 November 2026, 16:35 IST** from implementation commit `24071aa`. Launcher and worker were observed alive, with the first seed/fold advancing through its inner feature contexts. This launch record is historical; full verification and publication subsequently completed in `20e7325`. No worker/launcher was alive at completion review.

## Completed outcome — 16 November 2026

[Verified report](../results/rocket_vcg_v1/complete/REPORT.md), [summary](../results/rocket_vcg_v1/complete/summary.json), result commit `20e7325`. All 120 feature contexts were independently reconstructed from their own training patients, all 840 classifiers restored, and all 120 outer models verified. At review, the current scope and all transform/feature/classifier/probability artifact hashes matched. Training and automatic GitHub publication completed; no new training was started during review.

| Features + ridge | Mean member macro F1 ± transform-seed SD | Prespecified ensemble F1 |
|---|---:|---:|
| DCT | 0.446 ± 0.000 | 0.445781 |
| MultiRocket-style | 0.551 ± 0.017 | 0.561442 |
| HYDRA | 0.538 ± 0.012 | 0.548100 |
| Combined | 0.573 ± 0.019 | 0.584266 |

Combined−DCT is **+0.138485 [+0.047546,+0.228367]**, a positive matched-control development comparison. Combined−MultiRocket **+0.022825 [−0.020183,+0.069062]** and combined−HYDRA **+0.036166 [−0.045640,+0.115638]** remain uncertain. These intervals do not correct for repeated development or multiple comparisons. All seeds are retained; the combined member scores are 0.553775/0.572111/0.592695, and their mean is not the ensemble score. DCT has identical member predictions across seeds.

The combined point score is effectively level with the historical neural ensemble **0.584570**, lower by 0.000304. That earlier procedure is not the matched control, and the present result does not establish equivalence or a new highest ensemble score. It supplies a competitive feature-based alternative and evidence against the simple DCT template baseline, not independent confirmation or SOTA. Combined AMI/ALMI F1 remain .345/.483 while HC is .925; weaknesses in MI localization remain. No larger kernel budget, beat-variation extension, TabPFN classifier or neural/feature combination was fitted as part of this bounded study or review.
