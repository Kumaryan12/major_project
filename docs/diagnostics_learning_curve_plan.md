# Persistent-error review and patient-training-size sensitivity

## Why this step

The fixed denoised 651-sample ResNet averages 0.550 patient macro F1 across three training seeds. Raw input, longer context, and the bounded regularization/augmentation selection procedure did not establish a consistent improvement. The next step examines persistent failures and training-patient sensitivity before changing the model again.

The audit and learning curve answer different questions. The audit is post-hoc and uses existing held-out errors to organize review. The learning curve samples patients independently of those errors and retains the original labels and outer-test cohort. No error flags change eligibility or labels.

## Completed persistent-error audit

The [aggregate audit](../results/neural_diagnostics_v1/REPORT.md) covers all 164 patients, 391 records and 55,863 original beats. Thirty-five patients are incorrect in every selected-model seed; the fixed baseline has 36 persistent failures, with 27 persistent in both procedures. These are repeated predictions against source labels, not 492 independent patients or proof of annotation mistakes.

Across the existing six-class cohort, source-header labels agree with cached labels. The exact-byte raw XYZ screen finds no cross-patient duplicate groups. This does not validate clinical truth, hidden patient identities, near-duplicates, R positions or signal coverage. Per-record source text, former-location fields and existing cached-interval diagnostics are available locally. Prior MI text is context, not permission to replace acute-location labels.

The review packet contains all 35 persistent selected-model cases. Each has median cached beats from up to three records selected by record order, raw/denoised XYZ plots, source annotations, existing predictions and signal-change/interval summaries. Low mean source-class probability only orders review; it is an uncalibrated model score, and no probability ensemble is evaluated. The adjudication form is unfilled. Example plots and aggregate figures were checked for readable layout; there is no independent expert adjudication.

Local packet: `results/neural_diagnostics_v1/review/INDEX.md`. Patient/record tables and plots remain local. Public outputs contain only aggregate counts, class-stratified diagnostic summaries and provenance hashes. Differences between persistent and other cases are descriptive and confounded by class mix; no cause, statistical significance or label-noise rate is inferred. No patients, beats, labels or cached peaks were altered.

```bash
.venv/bin/python -m mi_localization.error_audit
```

Configuration: [`configs/diagnostics_vcg.yaml`](../configs/diagnostics_vcg.yaml). The audit validates source study fingerprints, complete paired predictions, patient identities/truth/folds, cache/record alignment and source file hashes. Changed inputs require a new output directory rather than silently replacing a different audit.

## Fixed learning-curve design

| Choice | Specification |
|---|---|
| Model/input | Original waveform-only 78,198-parameter ResNet; denoised measured XYZ, 651 samples at original R positions |
| Training sizes | 25%, 50%, 75%, 100% of each outer training pool, rounding up within each source class |
| Actual patients per final fit | 38–40, 74–76, 112–114, and 147–148, respectively |
| Patient subset rule | Fixed seeded per-class permutations; nested prefixes; all beats of chosen patients |
| Subset seed | 42 + 1,000 × outer fold; independent of training seed and model errors |
| Training seeds | 42, 123, 2026; same subsets for all three |
| Outer evaluation | Original ten patient folds; every original outer-test patient/beat at every size |
| Inner validation | Three stratified patient folds rebuilt within each selected subset |
| Scaling/stopping | Training-only class/patient-weighted normalization; inner patient macro F1 chooses epochs |
| Optimization | AdamW, learning rate 0.001, weight decay 0.01, dropout 0.2, batch 128 |
| Budget | 8,192 class/patient-balanced sampled beat presentations/epoch, up to 25 epochs, patience 6 |
| Final refit | Fresh initialization; rounded-median best inner epoch count; no outer-test monitoring |
| Patient prediction | Mean probabilities over every held-out beat |

All four points are trained afresh, including 100%; historical scores/checkpoints are not substituted. Every point has the same architecture and model-seed rules, so initial weights match across sizes. Sampling sequences, fitted normalizers and stopping epochs can differ. At 100%, original inner partition memberships and training protocol are retained.

Rebuilding inner folds within each subset is intentional: patients excluded from a smaller training pool must not supply validation labels. Each selected class must have at least three patients; the full-cohort plan passes this check, with at least four per class at the smallest size. Training membership and every inner train/validation membership are fingerprinted. The subset rule consumes only class labels, IDs and fixed seeds, never audit flags or predictions.

The complete design has **480 fresh fits**: 4 sizes × 3 seeds × 10 outer folds × (3 inner fits + 1 final refit). No new datasets, augmentation, feature tensors or architecture search are involved. Existing experiments and PTB-XL tests remain unchanged.

## Training scores and held-out scores

After each final fit, predict every selected training beat once and compute a training-patient resubstitution score. This score is descriptive and does not affect epoch selection. Store those training-patient probabilities locally, allowing the report to recompute the score and verify membership. It is not held-out validation.

At each size, held-out macro F1 is computed from pooled out-of-fold predictions on the same 164 patients, separately for each seed. The plotted endpoint is the mean across all three seeds, with sample SD across seeds. Training scores are averaged across outer fits and then seeds. The two lines have different populations/averaging rules; their gap is not a formal bias/variance decomposition. Training patients recur across outer fits and are not independent samples.

Only one nested training-subset realization is used per outer fold, shared across model seeds. Three-seed SD describes training-seed variation, not confidence intervals or training-subset variability. Fractions round up within classes, so actual patient counts and ranges are reported rather than presenting the nominal percentages as exact sample counts.

The same presentation budget means small cohorts receive more repeats per patient. Inner stopping can also be noisy with few patients/class. Therefore, the curve describes this model and compute/stopping policy. A rising curve does not give a validated required sample size; a flat curve does not establish that more data or transfer learning cannot help. Do not extrapolate to 0.80–0.90 or claim a PTB performance ceiling.

## Run, resume and inspect

```bash
# Validate the full patient plan without fitting or writing results.
.venv/bin/python -m mi_localization.learning_curve --plan
# Run all points, seeds and frozen folds; compatible checkpoints resume.
.venv/bin/python -m mi_localization.learning_curve
# Validate all predictions/fits and write tables, PNG/SVG curve and report.
.venv/bin/python -m mi_localization.learning_curve_report
# Read-only progress; does not launch/resume training.
.venv/bin/python -m mi_localization.learning_curve --status
```

Configuration: [`configs/learning_curve_vcg.yaml`](../configs/learning_curve_vcg.yaml). New outputs: `results/neural_learning_curve_v1/complete/`. Waveforms, weights and individual training/test probabilities remain local. Hashes cover bank files, beat table, paper settings, frozen folds, patient subset plan, relevant sources and runtime. Seed/size/fold identities guard resumable inner and outer checkpoints. Partial runs have separate directories and cannot produce a full report. Failures do not automatically restart. Saved status is not proof that its PID is still alive; MPS is not guaranteed bitwise deterministic.

The final report validates every fraction/fold/seed, checks training/validation scope and epoch selection, reconstructs the subset plan, recomputes resubstitution scores, checks pooled patient probabilities against fold checkpoints, and verifies matched architecture/initializations. It reports per-seed scores, mean/sample SD, actual patient counts, per-class F1 and paired 100%-minus-smaller-size differences. Class-stratified patient bootstrap draws are shared across paired sizes and fixed seeds; intervals are conditional on fitted predictions, without refits or corrections for repeated development, correlated folds or multiple comparisons.

These repeatedly inspected PTB patients remain development data. The curve is a diagnostic, not an opportunity to select the best fraction/seed and claim a new independent score. The outcome should guide a subsequent declared experiment, not prove that persistent cases are wrong or dictate automatic exclusions.

Method background: [scikit-learn learning-curve documentation](https://scikit-learn.org/stable/modules/learning_curve.html). Its general distinction between training and held-out performance motivates the diagnostic; this implementation explicitly samples patients and performs inner patient stopping rather than treating individual beats as independent examples.
