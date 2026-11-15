# Pinned TabPFN-v2 benchmark on selected VCG tensor features

## Completed findings and decision

The [validated report](../results/tabpfn_v2_v1/complete/REPORT.md) completed all ten outer folds and **390 fit/context slots**: 240 verified prior XGBoost inner fits, 120 new PFN inner contexts, 20 new PFN outer contexts and 10 fresh XGBoost outer fits. Every saved PFN context restored in an isolated worker and reproduced its outer probabilities/classes. All ten fresh tree controls matched their prior predictions. Relevant bootstrap-selector replay covered 72 caches. **179 tests passed**, alongside real-checkpoint synthetic and real-project context checks.

| Classifier | Patient macro F1 | Patient accuracy |
|---|---:|---:|
| Matched bootstrap-XGBoost | 0.513 | 57.3% |
| Raw TabPFN-v2 | **0.542** | **61.6%** |
| Prior-corrected TabPFN-v2 | 0.498 | 56.7% |

Raw PFN's mean paired point difference is **+0.030** versus XGBoost, descriptive interval **[−0.055, +0.115]**. This is a higher point score, not a conclusive or independently confirmed gain. Corrected PFN is −0.014 versus XGBoost (**[−0.095, +0.070]**) and −0.044 versus raw PFN (**[−0.097, +0.009]**). Correction-policy variants independently selected their masks using inner scores; their difference is not a pure correction-only causal comparison with masks held fixed.

Compared with the matched tree, raw PFN improved AMI F1 **0.294→0.357**, ASMI **0.414→0.500**, IMI **0.500→0.517** and HC **0.811→0.870**. ALMI stayed **0.452** and ILMI declined **0.605→0.558**. Rare-location weaknesses remain; the result is not uniformly improved localization.

Raw PFN selected combined/64 features in three outer folds, morphology/64 in four, and morphology/24 in three. XGBoost selected combined/64 once, morphology/64 three times, morphology/24 six times. This is consistent with learner-dependent feature preferences, without proving why performance differs. Only the fixed 24/64 selected-feature inputs were tested; all-feature PFN, learned embeddings and newer checkpoints are separate untested hypotheses.

**Decision:** retain raw TabPFN-v2 as a promising feature-model candidate and keep XGBoost as the matched control. Do not adopt the tested prior correction as an improvement. The historical waveform ResNet **0.550 ± 0.040** remains a different, slightly higher point of reference; this experiment has not reached 0.80–0.90 or demonstrated MI SOTA. Repeatability across inference/resampling seeds and independent-cohort confirmation remain outstanding. The proposed chronological/geometric neural architecture has not been trained by this benchmark.

## Fixed question

Does a pretrained tabular transformer improve the existing bootstrap-feature pipeline over its matched shallow-tree classifier? Measured Frank XYZ remains the sole signal source. Keep the two existing patient feature pools, morphology and combined, the same 164 patients / 391 recordings / 55,863 beats, original six classes, ten frozen outer patient folds and three fixed inner folds. No waveform, R position, label, feature family, clinical annotation or patient exclusion changes.

This implements the immediate benchmark in the [modern architecture roadmap](modern_architecture_roadmap.md). It is a **TabPFN-v2** experiment, not the latest TabPFN-3.5, a new ECG encoder or demonstrated MI SOTA. The publicly downloadable Nature-era checkpoint gives a reproducible first foundation-model comparison. Current newer checkpoints have separate access/terms and must be tested as separately pinned experiments.

## Pinned backend and source

- Python package: **tabpfn==9.2.0**.
- Model: `Prior-Labs/TabPFN-v2-clf`, `tabpfn-v2-classifier-v2_default.ckpt`.
- Repository revision: `275f19d2add9c2c0de38f9ebaded3856599fa2e7`.
- Size: **29,009,539 bytes**.
- SHA256: `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`.
- Inference: local CPU, float32, two CPU threads, eight fixed estimator/preprocessing members, temperature 0.9, `fit_preprocessors`, one preprocessing job, no torch compilation or gradient fine-tuning.

The checkpoint downloads from its pinned public revision and must match both size and checksum. `ModelVersion.V2` and an explicit model path prevent the library's current default from silently selecting v3.5. Model weights remain frozen. The published model uses synthetic-task pretraining; this is a different source/provenance route from our earlier small PTB-XL reconstruction transfer pilot. See [Nature paper](https://doi.org/10.1038/s41586-024-08328-6), [official model card](https://huggingface.co/Prior-Labs/TabPFN-v2-clf), and [official library](https://github.com/PriorLabs/TabPFN). Observe the model's license and attribution terms.

The package adds its optional dependencies, but installation constrained existing numerical versions. NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.15.0, XGBoost 3.4.1, pandas 3.0.6, joblib 1.6.0 and PyWavelets 1.10.0 remain unchanged. `pip check` passed. Full backend/runtime/source versions are recorded in the study manifest. Checkpoint, context states and patient feature arrays remain local; no hosted prediction API is used.

### macOS numerical-runtime isolation

The first in-process attempt hit a native OpenMP mutex failure while loading foundation weights after the full feature-selection path. Startup thread limits alone did not resolve it. No PFN model or outer prediction completed in that attempt; its cached tree-inner prefix is preserved under `tmp/tabpfn_failed_inprocess_attempt/`.

Foundation fitting and restored prediction now run in **fresh local Python subprocesses** with serialized training/query arrays in short-lived ignored temporary directories. The parent feature-selection process does not import TabPFN/LightGBM. Worker thread settings are fixed and recorded, and each worker has a timeout. This avoids inheriting initialized numerical-library state without modifying library binaries. Checkpoint, data, masks, seeds and inference settings are unchanged. The real-data path passed a 98-patient training-context fit, separate validation prediction and independent-process state restoration before the full run restarted.

## Three retained arms

1. **xgboost:** the existing bootstrap-feature procedure, depth 2/3, 100 trees, rate 0.05, lambda 1, full row/column sampling, one CPU thread, class-balanced training weights. Its inner predictions are reused only after checking complete source/selector/membership/hash provenance. Outer models are freshly fitted and must reproduce the earlier 0.513 procedure's predictions.
2. **tabpfn:** fixed pretrained v2 model with raw output probabilities.
3. **tabpfn_balanced:** same pretrained architecture with fixed inverse-training-class-count probability renormalization. There is no tuned correction strength, threshold or use of validation/test prevalence.

For raw PFN probability p(c) and number of training-context patients n(c), the corrected score is `[p(c)/n(c)] / sum_j[p(j)/n(j)]`. This matches the library's probability-balancing option, verified on unequal synthetic class counts. The API does not accept sample weights in this version; probability correction is **not equivalent** to weighted tree training and is not a clinical calibration claim. No patient is duplicated or removed to alter the context. Both policies are prespecified, reported and selected separately using inner scores.

## Feature processing and nested selection

All classifiers retain the same training-only bootstrap selector: 64 class-stratified training-patient resamples, separate 32-resample support panel, ANOVA base ranking, Pearson pruning 0.95, budgets at most 24/64. Existing bootstrap caches are reused with strict source/data/code/draw guards. No global feature mask is derived from the full cohort.

The selector's weighted training scaler determines the feature mask. XGBoost retains its existing scaled selected-column inputs. **TabPFN receives the raw selected columns** and fits its own model-specific preprocessing to its training context, including the pinned defaults' quantile/SVD and permutation behavior. Do not assume external scaling is required for a pretrained prior. Store the actual raw training-context matrix hash, training-label hash, selected indices, class counts, source scaler audit, fitted ensemble count and backend inference configuration for each fit.

For each outer fold, keep the original three inner patient partitions. PFN has **four candidates = two feature pools × two budgets**, with all architecture/inference settings fixed. XGBoost has eight candidates because it also selects depth 2/3. Raw and balanced PFN inner scores share each fitted context, then independently choose family/budget by mean inner patient macro F1; ties favor configured order. A fresh outer-training context is fitted for each reported PFN arm. No checkpoint, temperature, member-count, strength, seed or outer-test threshold search occurs.

There are **390 fit/context slots**:

- 240 verified reused XGBoost inner fits;
- 120 new PFN inner context fits, each producing raw and balanced scores;
- 20 new PFN outer context fits;
- 10 fresh XGBoost outer refits.

Thus new work is **140 PFN contexts plus 10 XGBoost refits**. A context fit is not gradient training of the foundation model. Eight inference members are not independent training seeds. Data/splits/feature-mask rules are matched; parameter counts, pretraining compute and algorithm-specific preprocessing/weighting mechanisms are not matched. All held-out patients contribute exactly one probability vector, with pooled patient macro F1 as the endpoint.

## Runtime smoke and independent report checks

The real public checkpoint has a six-class, 24-feature synthetic smoke test with uneven class counts. It checks finite normalized predictions, save/load equality, manual versus official balancing, insensitivity of one query to other query rows, and XGBoost coexistence in the same process. This is runtime evidence, not clinical performance. Aggregate smoke audit: `results/tabpfn_v2_v1/backend_audit/summary.json`.

A separate real-checkpoint 64-feature context/save-load check passed. Worker-boundary tests verify that only training labels enter a fit request and restore requests carry query features without labels. End-to-end context checks also cover the actual project-loader path, beyond the synthetic numerical smoke.

End-to-end tests use a lightweight fake context backend to verify the clinical workflow without network/model downloads: raw training context membership, labels, inner-only selection, probability correction, source-control reuse, complete reporting, guarded resumption and corruption rejection. The real checkpoint smoke supplies the separate numerical/backend test.

Final reporting independently reconstructs all relevant bootstrap selectors, recomputes every inner choice/score, checks complete frozen patient coverage, restores all saved PFN fit states and requires reproduced outer probabilities and classes. It also verifies every fresh XGBoost control against prior outer predictions. Only after this validation is the aggregate report published. No raw features, resampling matrices, model contexts or patient predictions are committed.

Report all three arms and classes, and three predeclared contrasts: raw PFN−XGBoost, balanced PFN−XGBoost, balanced−raw PFN. Use 5,000 class-stratified paired patient draws, conditional on fitted predictions without model refits or corrections for correlated folds, repeated development or multiple comparisons. The historical neural 0.550 ± 0.040 is a different aggregation/model protocol and not this matched control. These repeatedly inspected PTB folds remain development data; any favorable result still needs independent-cohort confirmation. No 0.80–0.90 or SOTA claim follows from using a foundation architecture.

## Reproduce, resume and inspect

```bash
.venv/bin/python -m pip install -e '.[neural,foundation,dev]'
.venv/bin/python -m mi_localization.tabpfn_backend --smoke
.venv/bin/python -m mi_localization.tabpfn_study
.venv/bin/python -m mi_localization.tabpfn_report
.venv/bin/python -m mi_localization.tabpfn_study --status
tail -f tmp/tabpfn_v2_run.log

caffeinate -i /bin/zsh scripts/run_tabpfn_v2.sh --publish > tmp/tabpfn_v2_run.log 2>&1
```

Config: `configs/tabpfn_v2.yaml`. Isolated outputs: `results/tabpfn_v2_v1/`. Atomic context/prediction checkpoints support resumption; changed config, source, checkpoint, library versions or backend source hashes reject incompatible reuse. Saved progress is not proof of a live process. The wrapper validates the final report before publication and refuses automatic commits if tracked work or the main branch changed. Existing studies and the paused multiview neural sweep are unchanged.
