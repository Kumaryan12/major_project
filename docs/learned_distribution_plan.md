# Frozen learned morphology distributions with an intact baseline

Authorized 17 November 2026 after the completed kernel comparison. The user accepted the checkpoint-audit/benchmark/learned-pooling implementation. New namespace: `learned_distribution_source/features/study/report.py`, `configs/learned_distribution_v1.yaml`, `results/learned_distribution_v1`. No older numerical module/config/report is edited. No following experiment is queued.

## Hypothesis and four controls

The kernel study changed the readout of fixed features and did not establish a nonlinear gain. This study instead asks whether frozen supervised morphology features can describe useful all-beat variation while preserving the entire original convolution representation. A learned feature vector is a hypothesis, not evidence of higher information quality or clinical meaning.

| Arm | Full fixed convolution base | Auxiliary learned summary |
|---|---|---|
| baseline | Unchanged, 3,584 standardized columns | None; original fitted baseline restored |
| plus_mean | Same base, bit-identical prefix | Mean learned residual |
| plus_spread | Same base, bit-identical prefix | Mean and population SD on fixed normal-quantile grid |
| plus_quantiles | Same base, bit-identical prefix | Empirical quantiles of learned residuals |

Each added arm has 144 auxiliary entries (16 projections × 9 quantiles), matching the earlier convolution-pooling budget. Main comparison: **plus_quantiles minus plus_spread**, equal-three-seed probability ensembles. Secondary quantiles−mean, quantiles−baseline, spread−mean and mean−baseline all retained. Report member means/SD and differences separately. Do not select a new arm/seed/feature budget from outer results. The 0.603374 selected-linear result is a separate development procedure, not this study's fixed baseline.

## Frozen encoder source and custody

Use ONLY plain `morphology` checkpoints from the completed measured-VCG staged study (`a56fe23`, `configs/staged_context_v1.yaml`). There are 120 required checkpoints: 10 outer folds × 3 seeds × (3 inner + 1 outer). Distinguish exact names from `continued_morphology`; a broad glob can accidentally count both. Do not use the older residual-context beat-only source, which did not retain its inner fitted weights. Do not choose encoders from favorable outer scores or source objectives.

Every teacher must match the exact fitting and query patient IDs, fitting labels, beat counts, seed, phase and selected epoch; `.pt` expected metadata and normalization must match its audit. Reconstruct each original training normalizer from the original signals and confirm exact stored parameters. Audit all original inner stopping decisions and outer epoch selection from source inner choices. Verify initialization hashes and state/model/prediction hashes. Only then load the saved state, set eval mode and disable all gradients. No new encoder training.

Inner teachers were selected by early stopping on THEIR original inner-validation patients. That selection is disclosed; these encoders are not independent of those validation labels. New head alpha selection uses the same declared inner partitions, with outer patients still wholly excluded from fitting/selection. Outer teachers were refitted for original inner-selected durations with no outer query monitoring. Never reuse an outer teacher inside its inner folds.

Read-only source chain: the already independently completed convolution-distribution caches (`0a861b9`) provide the full fixed baseline, exact partitions and frozen ridge models. Their 90,429-file custody check is reused. Add the staged source config/scope/report/checkpoints plus original raw-waveform/template hashes; initial inventory counted 90,804 protected files. `chronology_bank.RecordBank` is constructed from existing read-only arrays, preserving original labels/groups/R-position/RR metadata and record medians. No cache preparation function rewrites old progress or reports. Current code/runtime must reproduce the saved teacher scope; changed data/config/platform requires a new namespace.

## Embedding tensor

All 164 patients, 391 records and 55,863 original beats/folds remain. Input is measured signed/amplitude-bearing Frank XYZ, not ECG or an external source encoder. No filtering, new R detector, arbitrary physical rotation, unit-amplitude beat normalization, new patients or anatomical relabeling. Offline recording medians retain their noncausal interpretation.

For each exact source seed/partition:

1. Restore that teacher's input normalizer and frozen encoder. Encode every original beat into its 64-dimensional morphology embedding. CPU inference, two Torch threads, no gradients/dropout; batch size128 is pinned. These coordinates are not sensors, independently verified tissue territories or new samples.
2. Encode each selected recording's own coordinate-median waveform with the same encoder. Average those median embeddings equally across recordings within each training patient. Fit a 64-column StandardScaler on those training-patient rows only, including sklearn's constant-column handling. Query patients never fit embedding scales.
3. For each beat calculate `(embedding(beat) - embedding(record_median)) / training_embedding_column_scale`. Use 16 data-independent unit-norm Gaussian directions, seed `source_seed + 8123`. Feature-space projection is not rotation of physical XYZ.
4. Fit a single residual RMS from training patients only, with equal patient/equal recording/equal within-record beat weights. Below 1e-8 use gain one. Normalize projections with this gain.
5. Pool mean, mean/population-SD normal-quantile values or empirical linear-interpolation quantiles at q=.1,...,.9. The normal basis does not assert Gaussian physiology. Divide every grid by sqrt(9), multiply by the same fixed sqrt(3584/16), average records equally within patient and append the full grid to the intact baseline. No per-arm second scaler or feature selection; effective ranks still differ.

Persist all 64-dimensional embeddings, source-head beat probabilities and original row IDs by recording in ignored caches. Each record pins scope, frozen teacher, normalization, device, batch and original rows plus data hash. Save only a final record audit after atomic array publication. Completed valid records resume when a context was interrupted; a changed source/context is rejected rather than silently loaded. Fit/context parameters and patient arrays are also hashed.

Using every query beat, reaggregate the frozen original head's probabilities equally by recording and patient. They must reproduce saved source teacher query probabilities within fixed CPU-versus-original-MPS rtol1e-5/atol1e-6 AND identical predicted labels, before the context can supply new ridge fits. This source-head check does not tune the new model. Teacher predicted labels are not supplied to the new head as labels; only original target labels train the head.

## Training and complete independent replay

Class-balanced RidgeClassifier, intercept/Cholesky, alphas [1,100], same configured-order ties. Ten outer/three inner patient folds; all source seeds42/123/2026. Full source baseline alpha choices/predictions remain the original procedure. New head scaling, features and selection use the relevant training partition. No global outer-OOF fitting, thresholds, seed subsets or class routing.

Budget:

- 120 frozen teachers/feature contexts; no encoder optimization.
- 630 new ridge classifiers, including90outer.
- 210 restored fixed-baseline classifier contexts, including30outer.
- 840 evaluated classifier contexts/120total outer models/120matched checks.
- 40 unique training partitions for normalization, shared only among seeds with identical membership.

First run the teacher audit: restore all120checkpoint metadata/normalizers, repeat epoch choices and restore all210baseline contexts; baseline ensemble metrics must reproduce exactly. This gate is not full teacher query inference or new predictive performance. Each full embedding context then performs the teacher's actual all-query-beat inference replay before fitting new heads.

The independent report reloads the read-only source chain, restores every teacher, **re-encodes every original waveform** in every context, checks each record, reconstructs embedding medians/scales/projections/gain/pooling, restores all630new and210baseline classifiers and repeats all inner selections/matching. Fixed CPU embedding replay tolerances are rtol1e-6/atol1e-7; fitted pooled parameters/state and features must reconstruct under the declared strict checks. No classifier or encoder is trained during reporting. Source file snapshot must remain unchanged at the end. Publish only after complete declared coverage passes.

Report ensemble/member macro F1, class F1/confusions, all primary/secondary differences and 5,000 class-stratified shared patient-bootstrap draws/seed42. These intervals condition on fitted predictions, without refits or corrections for overlapping training sets, repeated development, source checkpoint selection or multiple comparisons. More beats/latent coordinates do not add independent target patients. No guarantee of0.80–0.90, clinical validity, independently confirmed superiority, demonstrated ceiling, exact PSWE/FSW guarantees or SOTA. Reused supervised encoder training still has historical compute cost.

## Engineering and timing

Meaningful tests check frozen eval inference, source-head replay, exact teacher partitions and normalization, query isolation, batch agreement, full original row coverage, intact base, partial-context record resume, raw-waveform/cache tampering, outer-label counterfactuals, all nested restores without classifier/encoder refits, config/scope/publication guards.

The real smoke uses all beats from outer1's FIRST INNER fitting/query partition and excludes all outer1 test patients. It checks the full source teacher predictor, all four head procedures and independent full embedding reconstruction. It is not a final metric. Training-only timing probe uses four fitting-patient recordings (519beats): 0.292s, ~0.562ms/beat; conservative two-pass extrapolation ~2.09h. Earlier probe0.311s/~2.23h. These exclude some I/O/normalization/reporting overhead; allow roughly2–3h, subject to actual progress. Source audit and tests are separate. Do not infer progress/completion from a saved directory name or PID.

```sh
.venv/bin/python -m pytest tests/test_learned_distribution.py -q
.venv/bin/python -m mi_localization.learned_distribution_study --benchmark
.venv/bin/python -m mi_localization.learned_distribution_study --audit
.venv/bin/python -m mi_localization.learned_distribution_study --smoke
.venv/bin/python scripts/run_learned_distribution.py --publish
.venv/bin/python scripts/learned_distribution_status.py --watch
```

Runner holds one namespace OS lock, requires matching audit/smoke, extracts and fits, independently replays/reports, publishes aggregate outputs and exits. No following study or older paused sweep. The monitor checks process commands and separates teacher gate, extraction/fitting, replay and validated completion; Ctrl-C exits only the monitor. All embeddings/checkpoints/patient predictions stay local and ignored. Public aggregate/provenance receipts may be committed. No numerical dependency/binary changes or OpenMP hacks; Windows unvalidated and Mac output scopes cannot be mixed with Windows runs.

Related pooling motivation: [Pooling by Sliced-Wasserstein Embedding, NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html). This is a fixed projected-quantile adaptation with a frozen supervised VCG encoder; MI benefit and final-method novelty remain hypotheses. HYDRA attribution/license remain under `src/mi_localization/_vendor/hydra`.

Validation note: all9focused numerical/integrity cases passed in a fresh interpreter. A combined-process full run segfaulted in native Torch Conv1d during the synthetic teacher fixture; the actual CLI audit and full-beat CPU smoke passed. The9cases now run in an isolated child interpreter within the main suite, matching production process isolation and preserving failure propagation. No package/binary change, numerical override or skipped integrity check was used.

Prelaunch checks: 321main-suite tests passed in152.06s, including the isolated9-case learned-feature module. Real audit passed120teachers/40training normalizers/210source classifier restorations and exact baseline metrics. Real inner-only full-beat smoke restored all four heads and independently reconstructed the full learned feature context, matching source teacher query labels/probabilities. No completed new localization result exists at preparation.
