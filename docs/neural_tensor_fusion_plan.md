# Patient-nested neural–tensor late fusion

Protocol fixed **16 November 2026**, after the user approved the proposed comparison. This is a new development experiment on repeatedly inspected PTB patients, not independent confirmation or a claim of novelty/SOTA. The original **0.584570** neural ensemble remains a frozen reference.

Validation before launch: **226 tests passed** on macOS, including six fusion tests. An actual-data inner-cache probe verified all three neural members plus one tensor context, exact patient/class alignment, neural stopping/coverage and bootstrap-selector replay. The probe evaluated no fused outer result. Full model restoration and fusion-score verification remain the job of the study and independent report.

**Completed and independently verified 16 November 2026**, result commit `2fe75a0`: all **250 source contexts**, **30 neural outer member restorations**, **10 PFN outer context restorations** and **70 selector bootstrap-cache replays** passed; zero new base-model fits. Frozen neural **0.584570**, tensor **0.542418**, fixed 50:50 **0.542530**, inner-selected blend **0.583052**. Selected−neural **−0.001517 [−0.050922,+0.050275]**, descriptive paired patient interval; no demonstrated improvement. Inner choices used neural weight .50 twice, .75 six times and 1.00 twice, which are overlapping training decisions, not a universal best weight. AMI F1 rose .294→.353 but ASMI fell .640→.571; this does not justify outer-label-based class routing. Keep the frozen neural ensemble as the reference. The conditional adaptive-gate follow-up is deferred because this comparison did not demonstrate an overall benefit. [Verified report](../results/neural_tensor_fusion_v1/complete/REPORT.md), [figure](../results/neural_tensor_fusion_v1/complete/fusion_comparison.png), [PDF figure](../results/neural_tensor_fusion_v1/complete/fusion_comparison.pdf).

## Inputs and motivation

The measured-XYZ neural ensemble and tensor-derived TabPFN classify different patients correctly. A post-hoc error comparison found 23 tensor-only correct cases, 29 neural-only correct cases, 78 both correct and 34 both wrong. This motivates testing complementary representations; it is not an oracle gain or a forecast of how many cases fusion recovers.

Neural branch: the frozen `ordered_context_aux` four-beat residual-context recipe, averaging all source seeds **42/123/2026** with equal weights. Tensor branch: the pinned raw-probability **TabPFN-v2** classifier, with the original bootstrap feature selection and inner choice between morphology/combined families and budgets 24/64. No new feature bank, waveform normalization, architecture, seed subset, temperature or prior correction is introduced.

All original **164 patients, 391 records, 55,863 beats**, six classes, cached R positions and frozen outer folds remain unchanged. Both inputs come from measured Frank XYZ; no ECG extension is mixed in.

## Four declared procedures

Let `alpha` be the neural weight: `p = alpha * p_neural + (1 - alpha) * p_tensor`.

| Procedure | Neural weight |
|---|---:|
| `neural_only` | 1 |
| `tensor_only` | 0 |
| `equal_blend` | 0.5 |
| `selected_blend` | Select from **[1, 0.75, 0.5, 0.25, 0]** inside each outer-training set |

Choose one global scalar per outer fold by **mean inner patient macro F1**, with stable configured-order ties favoring the frozen neural reference. No patient-specific/class-specific weights, uncertainty gate, learned meta-classifier, threshold tuning or outer-score model selection occurs in this experiment. Endpoints allow an inner choice to retain either branch; this does not guarantee a gain on outer patients.

## Leakage boundaries and cache reuse

Ten outer / three inner patient folds remain fixed. The source cache audit found exact neural/tensor validation-patient matches for all **90** neural seed/inner/outer combinations.

For each outer fold:

1. Reconstruct the exact three inner train/validation patient sets inside the outer-training patients. Confirm that both sets exclude every outer-query patient and are mutually disjoint.
2. Rebuild the three seeds' neural normalization, sampled windows, context permutations and stopping audits. Validate cached inner prediction hashes and identities, then average the three neural inner probability matrices. No outer-trained encoder is reused in an inner fit.
3. Independently replay the tensor bootstrap selectors and verify all four candidates' three inner contexts. Reconstruct the original tensor family/budget selection exactly, using its original mean-inner F1 and ties.
4. Select `alpha` using only these aligned inner validation predictions/labels, conditional on the original tensor candidate. Save the decision before loading/evaluating outer component outcomes. Reusing inner labels for stopping, component selection and scalar selection is a model-selection procedure; its inner scores are not unbiased performance estimates.
5. Verify and restore the three outer neural members and selected outer TabPFN context. Use the unchanged isolated PFN worker boundary. Apply the saved scalar and the three fixed controls to the outer probabilities.

The globally pooled outer prediction CSVs are **reference checks only**. They do not train a combiner or choose a weight. In particular, predictions generated by another outer-fold model may have seen the current fold's query patients during training, so they are not interchangeable with the correct fold-local inner predictions.

The expected reuse is **90 neural inner + 30 neural outer + 120 tensor inner + 10 tensor outer = 250 source fit contexts**, with **40 unique outer model/context restorations**, no new fits. Bootstrap caches are replayed per training partition/family, reusing shared cache computations across feature budgets. Missing/incompatible caches stop the run; they are not silently rebuilt or accepted after disabling a guard. Completed source artifacts remain read only.

## Verification and reporting

The independent report repeats every inner choice, stopping/selector audit, model restoration and outer fusion. It checks saved prediction hashes, membership and reproducibility, and verifies that neural-only/tensor-only reproduce the frozen component references. Source/config/code/runtime fingerprints prevent incompatible resume.

Tests exercise probability endpoints/normalization, stable ties, outer-patient exclusion, actual tiny neural training plus synthetic fitted tensor contexts, source immutability, resume/tamper checks, and a counterfactual showing that altering outer-query labels cannot change the saved scalar decision. The unit-test tensor context is explicitly synthetic; actual-data evaluation restores the real pinned TabPFN in isolated workers.

Report all four procedures, six class F1 values, unchanged-six-way MI-only F1, per-fold selected weights/candidates and descriptive paired patient intervals. Intervals use 5,000 class-stratified draws with the same patient draw across procedures, conditional on fitted predictions. No refits or corrections for repeated development, correlated folds or multiple comparisons. An ensemble yields one F1, not an optimization-seed SD. The reliability gate remains a separate later hypothesis, conditional on this simpler comparison.

## Commands and portability

```sh
.venv/bin/python scripts/run_neural_tensor_fusion.py --publish
.venv/bin/python scripts/live_status.py --watch
```

The launcher runs the study and independent report, holds an OS lock, and publishes only verified aggregate outputs on a clean main branch. Log: `tmp/neural_tensor_fusion_v1.log`; runner metadata: `tmp/neural_tensor_fusion_v1_runner.json`; output: `results/neural_tensor_fusion_v1`. The monitor labels reused source contexts separately from new fits (zero). Ctrl+C stops the monitor only.

Direct entry points are `python -m mi_localization.neural_tensor_fusion` then `python -m mi_localization.neural_tensor_fusion_report`. Native Windows uses `.venv\Scripts\python.exe`; Windows numerical behavior has not been validated. A clone lacks ignored local banks, per-patient prediction files, selector caches and fitted neural/PFN contexts. The source loader follows the older tensor refinement/stability prerequisites as well as the neural banks; changing just the last output directory does not migrate an incompatible source manifest.

[Configuration](../configs/neural_tensor_fusion_v1.yaml). Preserve completed numerical modules/configurations in the source studies; this implementation uses a new module/result namespace.
