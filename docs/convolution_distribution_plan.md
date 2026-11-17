# Intact-base all-beat convolution distributions

Authorized 17 November 2026, after the completed physical/DCT distribution study. The user asked to proceed and improve macro F1. This is one bounded follow-up, not a promise of improvement. No subsequent study is queued. Older paused work stays paused.

## Hypothesis and matched procedures

The first distribution screen's quantile ensemble reached 0.487114 versus its moment control 0.475709, with uncertain difference. The stronger frozen recording-median random-convolution ridge ensemble is 0.584266. Test whether preserving that entire representation and adding nonlinear beat-level variation helps. Do not substitute a weaker representation for the original baseline.

| Arm | Complete frozen base | Added residual grid |
|---|---|---|
| baseline | 3,584 standardized combined convolution features | None; original saved ridge models restored |
| plus_mean | Same full vector, bit-identical | Mean only |
| plus_spread | Same full vector, bit-identical | Mean + population SD on fixed normal-quantile grid |
| plus_quantiles | Same full vector, bit-identical | Empirical quantiles |

The primary contrast is **plus_quantiles minus plus_spread**, using equal three-seed probability ensembles. Secondary contrasts quantiles−mean, quantiles−baseline, spread−mean and mean−baseline are all retained. Mean member results are separately reported; there is no outer-score arm or seed selection.

## Frozen data and source

Keep all 164 patients, 391 recordings, 55,863 original beats, measured signed/amplitude-bearing Frank XYZ, original labels, cached R positions and `configs/patient_folds.csv`. No filtering, arbitrary physical rotation, unit-amplitude beat normalization, new patients, ECG, source-pretrained encoder or feature selection. Offline recording medians use only an examination's own signal.

Read-only source: `configs/rocket_vcg_v1.yaml`, `results/rocket_vcg_v1/complete`, original `rocket_bank/features/study/report.py`. Source MultiRocket-style conventions differ from the historical method; multivariate HYDRA remains experimental. Preserve its GPL attribution and license at `src/mi_localization/_vendor/hydra/README.md` and `src/mi_localization/_vendor/hydra/LICENSE`. No completed study or integrity check is edited.

Source preflight independently reconstructs all 120 training feature transforms, restores 210 original combined-ridge contexts, repeats inner selection and reproduces the complete source metric bundle, including 0.584266 ensemble macro F1. Snapshot and recheck 1,023 source report/context/model/selection files. Read original caches only; redirect nonnumerical bank audits into the new namespace. A cache/dataset/platform/config change requires another namespace.

## Auxiliary tensor

For each source seed and exact inner/outer partition:

1. Use the original convolution transform fitted on exact training patients' recording medians. Use original training-patient column scales (the frozen ridge StandardScaler).
2. Encode **every original beat**. Subtract the convolution vector of that beat's own recording-median waveform, divide by the frozen training column scales, then project to 16 feature-space directions. Unit-norm Gaussian projection columns use the declared source seed plus 8,123. Directions are data-independent; XYZ is not physically rotated.
3. Fit one shared scalar residual RMS from exact training patients only. Each patient has equal weight, each recording within a patient equal weight, and each beat within its recording equal weight; average squared energy across projections. Below a fixed 1e-8 floor use gain one.
4. Normalize projections by that shared gain. Build a `projection × quantile` grid with q=0.1,…,0.9. Mean repeats its mean; spread encodes μ+σΦ⁻¹(q); quantiles uses empirical linear interpolation. The normal basis is a matched moment control, not a claim of Gaussian physiological distributions.
5. Divide every grid by sqrt(9); multiply each auxiliary branch by the same fixed sqrt(3,584/16). This dimensional energy convention is declared before outer evaluation, not tuned. No per-arm scaler changes the branch weighting.
6. Average recording summaries equally within patients. Concatenate all 144 auxiliary columns to the full unchanged 3,584-column base. No additional compression or selection. Augmented stored widths match (3,728), but effective ranks/capacities differ. A positive result alone would not establish a uniquely distributional cause.

Stream by recording; persist only beat-row indices and compact projected residuals, not a 55,863×3,584 convolution bank. Each record checkpoint pins scope, source transform, source scales, directions, median features, rows and data hash. Interrupted extraction resumes valid completed records. A completed context also saves fitted scales/directions/gain and pooled arrays. Incomplete records have no final audit and are regenerated.

## Training, restoration and reporting

Ten outer patient folds, three inner folds, seeds 42/123/2026, class-balanced ridge with intercept, Cholesky solver and alphas [1,100]. Mean inner macro F1 chooses alpha; configured-order ties. All preprocessing belongs inside the exact fitting partition. Baseline reuses its original scaler/model/selection; three added arms fit new ridge classifiers with no second scaler. No outer-trained representation is reused across inner validation patients.

- 120 feature contexts.
- 630 new classifiers, including 90 outer models.
- 210 frozen baseline classifier restorations, including 30 outer models.
- 840 evaluated classifier contexts and 120 total outer models.

After fitting, the independent report reconstructs original source transforms, re-encodes every original beat for every auxiliary context, checks coverage/matching and restores every new/reused classifier. Repeats all selections, source probabilities and protected-file checks. Only then writes complete summary/report. All original beat coverage includes query signals for inference, while query labels never fit preprocessing or choose settings. Smoke/preflight/partial counts are not final performance.

Patient-level ensemble/per-seed/mean-member macro F1, class F1, confusion matrices and all declared contrasts are retained. Stratified paired patient bootstrap (5,000 shared draws) gives descriptive conditional intervals, without refits or adjustment for repeated development, overlapping training folds or multiple comparisons. The repeatedly inspected PTB cohort is development data; this cannot establish independent generalization, clinical validity, SOTA or guaranteed 0.80–0.90 performance. Historical neural ensemble 0.584570 remains a separate reference.

## Engineering checks and monitoring

Meaningful synthetic tests cover residual/projection oracles, equal-weight training gain, query isolation, quantile shape beyond matched moments, intact base, record resume, beat replay/tampering, frozen source memberships, outer-label counterfactuals, complete nested restoration without classifier refits and publication guards. Focused verification: 10 passed. Full macOS suite: 309 passed in 132.87 seconds. Run the full suite after relevant changes.

A real-data engineering smoke uses first two beats per recording from one inner partition; all outer-test patients are excluded. It restores three new classifiers and one baseline classifier, and replays the small auxiliary context. It is deliberately not full beat coverage or a score estimate. The timing benchmark uses all beats from four training-only recordings. An initial probe measured 519 beats in 2.32 seconds (~4.48 ms/beat); a rough upper extrapolation for sequential extraction plus replay is ~16.7 hours. The formal probe measured 2.83 seconds (~5.45 ms/beat), extrapolating ~20.3 hours, while the source replay was also active. Actual timing varies and this is not a guaranteed ETA. Benchmark and smoke receipts are saved separately.

```sh
.venv/bin/python -m pytest tests/test_convolution_distribution.py -q
.venv/bin/python -m mi_localization.convolution_distribution_study --benchmark
.venv/bin/python -m mi_localization.convolution_distribution_study --smoke
.venv/bin/python scripts/run_convolution_distribution.py --publish
.venv/bin/python scripts/convolution_distribution_status.py --watch
```

Runner: lock one local namespace, read-only source preflight, new study, independent report, verified aggregate publication, then exit. Monitor checks local PID commands and separates baseline replay, full extraction/fitting, independent replay and completion. Ctrl-C exits only the monitor. Fitted artifacts, record caches and individual predictions remain ignored. No numerical library changes or OpenMP override hacks. Python entry points are portable conveniences; Windows has not been validated and cannot resume Mac results under the same namespace.

Launch: implementation `2dfd577`, 17 November 2026 02:09:36 IST, sleep inhibition enabled. Startup preflight independently repeated all 120 source transforms/210 baseline restorations before fitting. A dated 08:46 IST local command check confirmed full extraction/fitting, with 483/630 new classifiers and 92/120 contexts saved. These are partial progress counters; independent replay and final reporting remain pending. Use the live monitor rather than this dated snapshot.
