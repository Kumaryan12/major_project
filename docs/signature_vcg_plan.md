# Ordered VCG trajectory signatures: controlled feature study

Implemented 15 November 2026. This is the next representation hypothesis after the healthy-reference recipe's provisional outputs did not show a gain. Training completion and validated scores are separate milestones; implementation alone establishes neither improvement nor SOTA.

**Completed follow-up:** all 1,050 fits, 150 outer restorations and independent report checks passed. Mean macro F1: morphology **0.515 ± 0.004**, signed areas **0.497 ± 0.023**, endpoint polynomials **0.513 ± 0.032**, signature2 **0.480 ± 0.010**, signature3 **0.508 ± 0.002**. Signature3 minus morphology: **−0.007 [−0.083,+0.066]**, descriptive paired patient interval. The tested signature recipes did not demonstrate an improvement. [Validated report](../results/signature_vcg_v1/complete/REPORT.md). The next approved study is [4/8/16 consecutive-beat length selection](chronological_lengths_plan.md), without changing this completed protocol.

## Hypothesis and research basis

Ordered interactions along a signed XYZ trajectory may make MI subtype differences easier for a small-sample classifier to learn than morphology compression or zero-lag covariance alone. Existing waveforms and DCT features already contain temporal information: signatures are a different inductive bias, not newly acquired information.

The mathematical basis is [Chevyrev and Kormilitzin, *A Primer on the Signature Method in Machine Learning*](https://arxiv.org/abs/1603.03788). [Feng et al., *PathFusion-Net*, accepted manuscript](https://discovery.ucl.ac.uk/id/eprint/10216614/) supplies related ECG arrhythmia research. Their task, architecture, splits and metrics are not our measured-VCG six-class patient macro F1. Signatures and ECG signature models are prior art. We are implementing a controlled VCG feature adaptation, not reproducing their network or asserting priority.

## Physical representation

Retain all **164 patients, 391 records, 55,863 beats**, original labels, R positions and frozen folds. For every original denoised XYZ beat, form the path:

```
path(sample) = ((sample - 250)/1000 seconds, X mV, Y mV, Z mV)
```

Use piecewise-linear interpolation between the original samples and exact truncated tensor signatures. Level one has 4 terms, level two 16, and level three 64: 84 coordinates through degree three. These coordinates have algebraic redundancies and are not independent biomarkers. Physical time is an explicit path coordinate; XYZ amplitude and direction are retained. No rotations, per-beat amplitude normalization, dynamic time warping or new sampling grid.

For each straight increment d, its signature is `(d, d⊗d/2, d⊗d⊗d/6)`. Balanced chronological Chen products compose the increments, preserving coordinate-word ordering. This is a NumPy float64 implementation without an additional compiled signature dependency. Tests cover analytic straight paths, concatenation, shuffle identities, signed orientation, translation/amplitude laws and piecewise-linear subdivision invariance.

Signatures of increments lose absolute offsets. Append physical XYZ window means in every representation. Use the whole beat [0,651) and the established fixed windows [50,170), [170,350), [350,470), [470,651). These are R-relative index windows, not verified clinical P/QRS/ST/T boundaries. Full-beat and subwindow descriptors intentionally overlap.

Per recording, summarize each beat descriptor with median and IQR. Per patient, take an equal-record coordinate median. This matches the existing morphology aggregation; no long recording gets extra patient-level weight. Every original beat contributes, and waveform means stay parallel to the increment features.

## Five declared arms

Every arm retains the same original **480 morphology columns**. The table shows added patient columns before training-only selection.

| Arm | Added features | Added columns |
|---|---|---:|
| morphology | None; fresh full-training reference | 0 |
| signed_area | Level-one increments, six antisymmetric second-order pair areas, XYZ means, for every window | 120 |
| endpoint_polynomial | Endpoint tensor powers through degree three and XYZ means; same size as signature3 | 840 |
| signature2 | Signature levels one and two plus XYZ means | 210 |
| signature3 | Signature levels one through three plus XYZ means | 840 |

The raw audit bank retains 130/870/230/870 columns respectively. Classification excludes only schema-defined pure clock words (`time`, `time_time`, `time_time_time`) and their IQR columns, because the clock is identical for every beat in each fixed window. The exclusion is algebraic, independent of all patient values/labels. Time–XYZ terms remain. This was fixed during the finite smoke phase, before any new outer model was fitted: weighted scaling had emitted tiny negative-variance roundoff warnings on these known constants, with no nonfinite scaled output. No full-cohort variable-column screen or warning suppression is used.

The signed-area arm is a simple geometric control, including time–XYZ and XYZ–XYZ pairs. The endpoint-polynomial arm replaces each path with its straight endpoint increment, preserving polynomial endpoint terms and window means but omitting interior ordered interactions. It controls one form of generic nonlinear expansion, not every alternative representation of interior geometry. Signature3 versus signature2 tests whether the extra third-order terms help under the complete selection policy. A higher-dimensional feature pool also changes selection, even with a matched selected-column budget.

## Evaluation fixed before outer scoring

- Ten frozen outer patient folds; three inner patient folds, using the established split-seed policy. Every arm has the full outer/inner training-patient set; there is no atlas/reference holdout here.
- Same training-only bootstrap consensus selector: 64 class-stratified patient resamples, 32 separate audit resamples, class-balanced training scaling, ANOVA base ranking and Pearson redundancy pruning at 0.95. No semantic group constraint.
- Two maximum budgets **24/64**, selected separately for every arm/sampling seed using mean inner patient macro F1 and configured-order ties. No feature-family choice across arms using outer scores.
- Sampling seeds **42, 123, 2026** vary feature-selection bootstrap draws. The PFN inference-seed policy stays fixed across arms and sampling seeds. These are not three independent cohorts or three gradient-trained foundation models.
- Fresh local pinned **TabPFN-v2**, unchanged checkpoint/package/eight fixed inference members, CPU float32, raw selected columns, no inverse-prior correction or fine-tuning. Use the existing isolated worker boundary.
- **1,050 new fit contexts:** 900 inner and 150 outer; 150 outer arm/seed evaluations. A PFN context fit is not gradient training of the foundation weights.
- Report all seeds, six-class macro F1, each class F1, accuracy and MI-only five-class macro F1 using unchanged six-way predictions. MI→HC mistakes remain errors. Paired patient draws are shared across seeds; intervals condition on predictions and do not refit models or correct repeated development/multiple comparisons.

The first sampling-seed morphology policy intentionally matches the prior healthy-reference study's full-morphology selector/inference seed formulas. It is still fitted afresh here. Historical ResNet 0.550 ± 0.040 and the earlier four-candidate TabPFN 0.542 are different procedures, not matched controls for new feature gains.

## Construction and sensitivity audit

Bank construction verifies source identity and copies original morphology columns unchanged. The bank caches per-record descriptors with content hashes, allowing resume. It checks signature endpoint/shuffle identities on each recording median.

Additional audits compare raw versus denoised recording medians and ±4-sample (4 ms) shifts of denoised medians, with edge extension rather than circular wrap. Those diagnostic signals are never training augmentation, new R positions, selection criteria or exclusion rules. Errors are grouped by homogeneous physical units, with near-zero denominators counted. These audits do not measure clean-signal fidelity or establish that a sensitive term is clinically useful. No global constant-column audit fits a classifier mask.

The independent final report reconstructs the training selectors, replays their bootstrap caches, recomputes inner choices and pooled predictions, and restores every saved outer PFN context. Only then is a completed report published.

## Run and monitor

```sh
# Finite actual-data construction, selection and PFN save/load check.
.venv/bin/python -m mi_localization.signature_study --smoke

# Complete/resume the declared comparison, verify, then publish aggregates.
zsh scripts/run_signature_vcg.sh --publish

.venv/bin/python -m mi_localization.signature_study --status
tail -n 12 tmp/signature_vcg_v1.log
pgrep -fl 'mi_localization.signature|run_signature_vcg'
```

For native Windows, use `.venv\Scripts\python.exe` for the module commands. The zsh launcher is optional; run `signature_study --smoke`, `signature_study`, then `signature_report` separately with the appropriate config. Rebuild/migrate source banks correctly when fingerprints differ; do not bypass guards. Unlike the previous deviation loader, this study needs the verified source/feature banks but does not require completed refinement/stability classifier studies. It still is not runnable on a clone with missing data.

Source modules: `signature_tensor.py`, `signature_bank.py`, `signature_study.py`, `signature_report.py`. Config: [signature_vcg.yaml](../configs/signature_vcg.yaml). Ignored cache: `data/processed/signature_vcg_v1`. Aggregate audit/results: `results/signature_vcg_v1`. Individual predictions, caches and model files stay local. The launcher only stages that new result directory after independent verification and a clean tracked working tree.

## Decision rule

Retain a representation as promising only if it improves its fresh morphology control consistently across the declared sampling seeds, with class-level examination showing what changed. A signature3 point gain alone is insufficient to infer that third-order ordering caused it; compare the signed-area, endpoint-polynomial and signature2 controls. If the simpler control performs as well, prefer the simpler explanation. Do not select a favorable seed or add higher orders because an outer result looks attractive.

These repeatedly inspected PTB folds remain development data. No cohort, label or evaluation change is justified to reach a target score. Independent, label-compatible cohort confirmation is still necessary before broad generalization or SOTA claims.
