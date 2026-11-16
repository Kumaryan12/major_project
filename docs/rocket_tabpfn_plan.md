# Bounded convolution-feature selection: ridge versus local TabPFN

Approved **16 November 2026** after the completed random-convolution study. The user authorized this classifier/selection follow-up. It does not authorize a new waveform representation, variability branch, external dataset, ECG extension or unbounded foundation-model search.

## Question and fixed reference

Does training-only bootstrap selection of the combined MultiRocket-style/HYDRA features improve patient macro F1, and does local TabPFN improve on ridge with the same candidate masks?

The completed [source report](../results/rocket_vcg_v1/complete/REPORT.md), result `20e7325`, provides the **full-feature ridge reference: 0.584266**. The historical neural ensemble **0.584570** is a different separate reference. Neither is new independent confirmation. This follow-up selected the combined representation after development results; subsequent nested evaluation on the same repeatedly inspected patients remains **adaptive development**, not a fresh untouched test.

Primary procedures:

1. **`full_ridge`:** reuse the completed combined-feature ridge predictions/models and reconstruct its original inner alpha choices, without new reference fits.
2. **`selected_ridge`:** bootstrap-select at most 32/64 features, choose budget and ridge alpha [1,100] inside the original three inner patient folds, then refit on outer-training patients.
3. **`selected_tabpfn`:** use the same candidate feature masks and choose 32/64 inside those inner folds; fit the pinned local PFN configuration without inference/checkpoint tuning.

Also report the prespecified **fixed-budget** controls `ridge_k32/tabpfn_k32` and `ridge_k64/tabpfn_k64`. Each ridge control chooses its alpha from training-only inner scores at that budget. Both outer budgets are explicitly declared for reporting; they do not select a favorable global budget from outer results. Selected procedures reuse these outer models without extra fitting.

Selected ridge and selected PFN can choose different budgets/masks. Their difference compares selected procedures; fixed-budget comparisons share exactly the same mask/count. The ridge procedure has four inner candidates (two budgets × two alphas), while PFN has two. Report this difference in tuning compute rather than implying equal candidate counts.

## Source reuse and patient boundaries

Keep the original **164 patients, 391 records, 55,863 measured-XYZ beats**, six classes and `configs/patient_folds.csv`. Input remains each recording's all-beat coordinate-median template, with equal-record averaging of convolutional features within patients. Templates compress within-record variation; this experiment does not test an added variability representation.

Consume the exact completed source feature context for each outer fold, inner fold and transform seed. Its training-only common RMS, kernels and bias thresholds are already scoped to those exact patients. Never substitute an outer-fitted feature context for an inner context. Missing or changed source contexts fail rather than trigger source fitting or silently relax integrity checks.

Pin the source study scope, validated summary, all 120 relevant feature contexts, all 210 combined-ridge contexts and their 30 selection decisions: **1,023 source files** (audits, matrices, transforms/models, probabilities, scope/summary/report). Check the aggregate artifact digest at load and again at completion. Source numerical modules/results are unchanged. Source diagnostic progress files are not used as evidence of training or performance.

## Training-only feature selection

Use the existing immutable `feature_stability.ConsensusSelector` / `bootstrap_cache` implementation:

- **64 class-stratified training-patient resamples**, preserving duplicate draws' multiplicity.
- A separate **32-resample diagnostic support panel**, which does not choose masks or budgets.
- Train-only weighted standardization, ANOVA rankings, bootstrap inclusion frequencies and absolute Pearson redundancy pruning at **0.95**.
- At most **32 or 64 selected columns**; report actual counts and masks. No hard semantic-group constraint.
- Selector seed `100000019 + source_seed + 1000*outer_fold + 10*inner_fold`; inner fold zero denotes the outer refit.

Fit once per training context, sharing its bootstrap panel across budgets and classifiers. Training labels/membership/features, query membership/features, mask/scaler/ranking hashes, selected raw contexts and selector provenance are audited. Query values or labels do not fit the selector. Do not select a global mask using pooled outer predictions.

Each classifier receives the **same raw selected columns** for a given budget and partition. Ridge then fits an unweighted StandardScaler on training patient rows and class-balanced RidgeClassifier (`solver=cholesky`), matching the source classifier convention. The selector's class-weighted scaler determines rankings/pruning; it is not the fitted ridge scaler. TabPFN instead receives raw selected values and fits its own model-specific preprocessing to its training context.

## Fixed PFN and selection rules

Reuse [the pinned PFN configuration](../configs/tabpfn_v2.yaml): TabPFN-v2 checkpoint revision `275f19d2add9c2c0de38f9ebaded3856599fa2e7`, SHA256 `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`, package 9.2.0, CPU, eight inference ensemble members, temperature **0.9**, probability balancing disabled. No new checkpoints, member-count/temperature/prior correction search, numerical dependency changes or hosted prediction API.

Every fit/restore uses `tabpfn_backend.run_fit/run_predict` in an isolated local worker; no direct TabPFN/LightGBM import in the selector process and no OpenMP override hacks. PFN state is saved for **all inner and outer contexts**, not only final refits. Ridge models are also saved for every candidate.

For each procedure, highest arithmetic mean of its three inner patient macro F1 values selects the candidate. Exact ties follow configured budget order [32,64], then ridge alpha [1,100]. No outer labels monitor stopping, choose budgets/alphas, adjust probabilities or route patients/classes. Original full-ridge inner choices are replayed independently and its saved predictions must reproduce the source result.

Retain all source seeds **42/123/2026**. PFN's inference random seed is `42 + 1000*outer_fold + inner_fold`, common across source-transform seeds. Source transforms and bootstrap panels vary across members; these are not independent foundation-model pretrainings. Equal probability averaging over all three members is prespecified for all seven reported procedures. Ridge uses the original fixed softmax of decision scores, not calibrated clinical probabilities.

Ridge class balancing and PFN's pretrained prior/preprocessing differ. This is a comparison of classifier procedures with matched candidate columns, not a pure causal isolation of nonlinearity.

## Exact budget and independent verification

Ten outer folds × three source seeds:

- **420 new ridge fits:** 30 × (three inner folds × two feature budgets × two alphas + two outer budget refits).
- **240 new PFN contexts:** 30 × (three inner folds × two budgets + two outer budget fits).
- **660 new classifiers total**, including **120 new outer models**. Selected procedures reuse the fixed-budget outer predictions.
- **120 reused source feature contexts**, **120 bootstrap panels**, **240 selector masks** and **210 reused full-ridge model contexts**, including 30 reference outer models.

Separate reporting reconstructs all 120 source transforms from only their training templates, reproduces their patient features, replays all 120 bootstrap caches, verifies all 240 saved selectors, restores **all 660 new classifiers** (including all 240 PFN contexts in isolated workers) and **all 210 source ridge models**, reconstructs all candidate choices, checks full outer-patient coverage and requires the frozen reference to reproduce **0.584266**. Reconstructing source transforms is verification, not a new representation study. Fitted states, arrays, raw IDs and patient predictions remain local/ignored; public outputs are aggregate metrics, confusion matrices and hashed audits.

Report all member F1 values and pipeline-seed mean/SD separately from ensemble F1, all six class F1 values and selected-budget counts. Prespecified ensemble contrasts: selected PFN−selected ridge, selected PFN−full ridge, selected ridge−full ridge, and PFN−ridge at each fixed budget. Use **5,000 class-stratified paired patient draws**, conditional on fixed predictions. Intervals do not include refits or correction for correlated CV training sets, repeated adaptive development or multiple comparisons. No best seed, arm-subset/weight search, clinical validity, independent confirmation, established SOTA or guaranteed 0.80–0.90 claim.

## Validation and execution

Eight targeted tests passed: actual tiny nested source/new fits and restoration with an explicitly mocked PFN backend; source immutability, shared masks, training-only selection, outer-label counterfactuals, missing-partition rejection, altered selector/model/source rejection, compatible resume, configured ties and monitor/publication guards. These mocked-backend tests validate orchestration, not PFN model performance.

The actual-data smoke used the real pinned checkpoint in isolated workers. It selected exactly 32/64 columns, replayed the bootstrap cache and saved selectors, restored all four ridge/PFN models, verified query-batch invariance under altered unrelated queries, and left pinned source files unchanged. It reports no classification performance. The complete macOS regression suite passed **253 tests** before full launch. Launch status is recorded in the handoff after it occurs.

```sh
.venv/bin/python -m mi_localization.rocket_tabpfn_study --smoke
.venv/bin/python scripts/run_rocket_tabpfn.py --publish
.venv/bin/python scripts/live_status.py --watch
```

[Configuration](../configs/rocket_tabpfn_v1.yaml). Core: `rocket_tabpfn_study.py`, `rocket_tabpfn_report.py`; output `results/rocket_tabpfn_v1`; log `tmp/rocket_tabpfn_v1.log`; runner metadata `tmp/rocket_tabpfn_v1_runner.json`. The OS-locked launcher repeats the matching real-data smoke, resumes matching completed contexts, verifies the full report and publishes only its aggregate namespace from clean main. Verification/publication failures are distinguished. Ctrl+C closes the monitor only. Native Windows uses `.venv\Scripts\python.exe`, compatible local prerequisite banks and a separate namespace; numerical Windows execution is unvalidated.

## Research basis

[MASHT, Cüppers and Vreeken (2026 preprint)](https://arxiv.org/abs/2607.19234) combines MultiRocket/HYDRA transforms with tabular foundation inference. This motivates the classifier comparison; our template input, bounded transforms, bootstrap selection, pinned older PFN and patient-nested six-class task differ. It is not a full MASHT reproduction or an originality claim.

The underlying [MultiRocket](https://arxiv.org/abs/2102.00457), [HYDRA](https://arxiv.org/abs/2203.13652) and [TabPFN](https://doi.org/10.1038/s41586-024-08328-6) are prior methods. Their benchmark scores do not establish MI localization performance. Preserve the source HYDRA vendor attribution/license and the local PFN checkpoint provenance.

Full fits launched **16 November 2026, 17:08 IST** from `eebe494`, after the repeated matching smoke passed. This launch observation is historical. Fits, independent verification and publication completed in `500f6a1`; no worker/launcher was alive at completion review.

During the run, a [spot numerical diagnostic](../results/rocket_tabpfn_v1/numerical_audit/summary.json) traced sklearn warnings in the first inner partition to weighted constant-feature variance roundoff around −1.46e−43. Across all 96 resampled scalers checked, existing constant-feature handling yielded finite scales/transformed features and unit scales for those entries. This is not an audit of every partition; full verification remains required. No numerical code, installed dependencies or active configuration changed.

## Completed outcome — 16 November 2026

[Verified report](../results/rocket_tabpfn_v1/complete/REPORT.md), [summary](../results/rocket_tabpfn_v1/complete/summary.json), result commit `500f6a1`. All 660 new classifiers restored, all 120 source feature contexts reconstructed, 210 frozen ridge contexts restored, 120 bootstrap caches replayed and 240 masks verified; source artifacts unchanged. Completion review matched the current scope, checked all saved model/probability/selector/cache hashes, and found all 240 final saved selector scales finite ([artifact check](../results/rocket_tabpfn_v1/numerical_audit/completion_check.json)). Training and publication completed normally; no next full run was started during review.

| Procedure | Three-seed ensemble macro F1 |
|---|---:|
| Frozen full ridge | 0.584266 |
| Inner-selected ridge | 0.544036 |
| Inner-selected PFN | 0.530966 |
| Fixed32 ridge | 0.495415 |
| Fixed32 PFN | 0.510066 |
| Fixed64 ridge | 0.576198 |
| Fixed64 PFN | 0.532195 |

Selected PFN−full ridge is **−0.053301 [−0.127591,+0.021462]**; selected PFN−selected ridge **−0.013070 [−0.083655,+0.056175]**; selected ridge−full ridge **−0.040231 [−0.126448,+0.051966]**. All are descriptive paired patient intervals including zero; no classifier/selection gain is demonstrated. Full-feature ridge remains the strongest point score within this study, with the historical neural ensemble0.584570 separate.

AMI class F1 fell **0.344828→0.080000** for both selected procedures: 5/17→1/17 correctly classified AMI patients. Fixed32 controls had no correct AMI predictions. The PFN AMI change contributes **−0.044138** arithmetically to the macro-F1 difference **−0.053301**; class contributions are score arithmetic, not proof of a causal mechanism. PFN ASMI also fell .633→.533; smaller ALMI/IMI/ILMI gains did not offset these losses. This pattern is consistent with useful subtype information being lost by this 98–99% feature reduction, but does not prove all compression/feature-selection/PFN recipes fail.

Inner selection favored64 over32 in23/30 ridge and27/30 PFN contexts. These are overlapping training decisions, not proof of a universal optimal budget. Fixed64 ridge's ensemble .576198 exceeds selected ridge's .544036, while its mean member F1 .535982 is lower than selected ridge's .546154. Selection optimized individual inner predictions, and averaging probabilities changes the predictor; this is not evidence of a selection-code error. Do not retroactively select a fixed budget or adjust ensemble weights using these outer results.

The tested switch to bootstrap-selected PFN is not supported as an improvement. Retain the full-feature ridge and historical neural references, preserve all outcomes, and require a separately declared controlled hypothesis before any further fitting.
