# Patient-bootstrap feature consensus in a fixed tensor pool

## Completed result and decision

The [validated report](../results/feature_stability_v1/complete/REPORT.md) completed all **1,500 classifier fit slots**: 960 new inner fits, 480 verified prior inner fits and 60 fresh outer refits. All **80 caches / 7,680 resampled base-selector fits** were independently replayed. Every fitted final classifier reproduced its held-out predictions; all **20 original-control refits** also matched their compatible prior predictions. Test suite: **175 passed**, with the end-to-end tests rerun after strengthening support-only cache replay.

| Selector | Linear macro F1 | Shallow-tree macro F1 |
|---|---:|---:|
| Original | 0.498 | 0.420 |
| Bootstrap consensus | 0.497 | **0.513** |
| Group bootstrap consensus | 0.463 | 0.357 |

For trees, bootstrap consensus improved the matched original procedure by **+0.092 macro F1**, descriptive paired patient interval **[+0.027, +0.158]**. Linear bootstrap was essentially unchanged (−0.001, **[−0.084, +0.081]**). Group bootstrap hurt trees relative to bootstrap (−0.155, **[−0.234, −0.078]**) and did not improve linear performance. These are conditional development intervals without refits or repeated-search/multiple-comparison adjustment; they are not independent confirmation.

The bootstrap-tree result improved all five MI class F1 values relative to its tree control: AMI **0.270→0.294**, ALMI **0.320→0.452**, ASMI **0.310→0.414**, IMI **0.298→0.500**, ILMI **0.491→0.605**; HC decreased **0.833→0.811**. AMI remains weak. A higher macro score must not be presented as uniformly reliable localization.

Support gains were small: matched audit feature support changed morphology/24 **0.589→0.594**, morphology/64 **0.656→0.658**, combined/24 **0.486→0.494**, combined/64 **0.503→0.512**. Grouping corresponding features across views reduced this support. The one-feature-per-semantic-group restriction is not supported by this test; different views should not be presumed interchangeable.

The gain belongs to the **complete nested bootstrap-tree procedure**. Inner selection also chose different budgets/depths: original trees used 24 features in 4 folds and depth 3 in 3 folds; bootstrap trees used 24 features in 6 folds and depth 3 in 5 folds. The result does not isolate a frequency-only causal effect with classifier settings held fixed.

**Decision:** retain ordinary bootstrap consensus with shallow trees as a promising feature-pipeline candidate, preserve the original linear selector as its reference, and do not adopt the group constraint. Next confirm the fixed bootstrap procedure across independent resampling seeds before expanding the search. That seed-robustness phase has not run. The historical neural **0.550 ± 0.040** uses different aggregation/models and three training seeds; the new 0.513 does not supersede it or reach the 0.80–0.90 target. New-cohort confirmation remains necessary.

## Question and fixed scope

The previous [tensor refinement](tensor_refinement_plan.md) improved signal reconstruction but did not improve localization consistently. Its original feature procedure reached linear macro F1 0.498; expanded representations reached 0.425 with linear and 0.452 with shallow trees. Higher optimized inner scores did not predict better outer performance.

This experiment tests **selection reliability** while keeping the tensor data fixed. Use only the two existing representations favored by the first feature screen: **signed morphology (480 patient features)** and **combined summaries (3,864)**. This pool choice uses prior development findings, not new independent data. No higher-rank tensors, new feature families, preprocessing, label changes, patient/beat exclusions or neural training are introduced. All 164 patients, 391 records and 55,863 beats remain represented with the existing offline within-record templates and equal-record patient summaries.

Compare three selectors with the same feature/model candidates:

1. **original:** full-training ANOVA ranking, then training absolute Pearson pruning at 0.95.
2. **bootstrap:** frequency consensus over 64 class-stratified training-patient bootstrap samples, then full-training Pearson pruning.
3. **group_bootstrap:** consensus over corresponding feature groups across physical views, then selection of representative columns, at most one per group and the same Pearson pruning.

Feature budgets remain at most 24 or 64. There is no tuned frequency cutoff. Low-support columns may remain when filling a budget; actual selected counts and support must be reported. This is a heuristic bootstrap-consensus test, not a reproduction of classical stability selection or a method with established false-discovery guarantees.

## Resampling and groups

Each input row is one independent patient. Draw with replacement **within each original training class**, keeping its original count. Repeated patient draws retain multiplicity. Applying inverse duplicate-patient weights would cancel the bootstrap, so it is deliberately avoided. The class-balanced row weights used for scaling repeat with the draws. No classifier is trained on these resamples; they produce base-selector rankings and ordered selected lists.

For each training partition and family, run the original selector up to the largest feature budget on **64 training resamples**. The 24-feature prefix matches the greedy selector's 24-feature result. Count how often each feature appears within the requested prefix. For groups, count each group at most once per resample if any member appears. All feature/classifier candidates share these panels in a partition. Seeds are fixed by outer/inner fold, and sampling never uses validation/test patients.

Semantic groups are fixed by the feature schema. Preserve source feature family, median/IQR summary, XYZ, band/window/statistic or DCT coefficient order; merge only the physical-view slot (raw, denoised, median, beat residual, filter residual). Statistic-only baseline columns remain distinct. These groups represent a hypothesis about corresponding coordinates, not proof that different views are interchangeable or clinically equivalent. No data-derived grouping is fitted on the cohort.

Bootstrap ranking uses descending feature frequency, then original full-training F score, then index. Group ranking uses any-member frequency, then its strongest feature frequency/F score, then group index. Within each group, rank representatives by individual frequency/F score. Greedy selection skips full-training absolute correlations ≥0.95 and allows at most one selected column per group. Full-training scaling remains identical across methods. Ranking F scores themselves are unweighted, as in the original method; class balancing enters scaling and classifier fitting once.

## Separate support panel

A different random stream produces **32 audit resamples**, never used to rank final features or select family/budget/model settings. Report how often each final feature/group appears in the audit panel's original base-selector fits.

This measures **support against independent resamples of the same training patients**. It does not measure stability of rerunning the entire consensus algorithm under new datasets, establish biological importance, certify clinical quality or create independent patients. Compare support at the same family and budget across all outer-training contexts, rather than compare only differently tuned winning packages. Those contexts overlap heavily; frequencies are descriptive diagnostics, not independent stability probabilities. Higher support alone is not improved macro F1.

## Nested classifier evaluation

Retain the existing ten frozen patient folds and three fixed inner patient folds. For each selector and classifier, choose among the same **eight candidates = two representations × two feature budgets × two model settings**. Every scaler, ANOVA ranking, bootstrap panel, consensus and correlation check uses its relevant training partition only. Inner-validation patient macro F1 selects the candidate; exact ties favor the original configured order. A fresh selector and classifier refit on outer-training patients, then predict the held-out patients. The audit support panel does not enter this selection.

Classifiers match the previous study:

- Linear L2 logistic regression, C 0.1/1.0, max 3,000 iterations, tolerance 1e-6.
- Shallow XGBoost, depth 2/3, 100 trees, rate 0.05, lambda 1, full row/column sampling, one CPU thread.

There are **1,500 classifier fit slots = 2 learners × 3 selectors × 8 candidates × 3 inner folds × 10 outer folds + 60 fresh outer refits**. The 480 original-selector inner fits reuse verified prior predictions from the same families, partitions and model settings. New classifier work is **960 inner fits + 60 outer refits = 1,020 fits**. Original-selector outer models are freshly trained; compatible previous predictions must reproduce exactly.

Bootstrap selection is a separate computation: **80 distinct caches = 40 training partitions × two families**, each containing 64 training and 32 audit base-selector fits, totaling **7,680 base-selector fits**. The same cache serves both budgets, all classifier settings and all selector variants. Final report validation independently replays every bootstrap base-selector fit, in addition to validating all classifier selections and requiring every saved final model to reproduce its held-out predictions.

Report all six procedures and all classes. Predeclared paired contrasts are bootstrap−original, group_bootstrap−original, group_bootstrap−bootstrap separately for each learner. Use 5,000 paired class-stratified patient bootstrap draws for descriptive intervals conditional on fitted predictions. No model refits or adjustments for correlated folds, repeated development or multiple contrasts are included. These repeatedly inspected PTB folds remain development data; no 0.80–0.90, SOTA, clinical biomarker or independent-confirmation claim is warranted by the design alone.

## Reproduce

```bash
.venv/bin/python -m mi_localization.stability_study
.venv/bin/python -m mi_localization.stability_report
tail -f tmp/feature_stability_run.log
cat results/feature_stability_v1/complete/progress.json

caffeinate -i /bin/zsh scripts/run_feature_stability.sh --publish > tmp/feature_stability_run.log 2>&1
```

Config: `configs/feature_stability.yaml`; isolated output: `results/feature_stability_v1/`. Source features, prior results and other experiments remain unchanged. Dataset/config/code/runtime and prior-result fingerprints guard resumption. Bootstrap matrices, feature arrays, patient predictions and models remain local in ignored files. Aggregate metrics, per-fit audits, selected feature names and matched support summaries are public. The publication wrapper runs full report validation first and refuses publication if tracked work or the main branch changed. Saved progress needs a matching live process to establish that training is running.

Method context: [Meinshausen & Bühlmann (2010)](https://doi.org/10.1111/j.1467-9868.2010.00740.x), [Nogueira, Sechidis & Brown (2018)](https://www.jmlr.org/papers/v18/17-514.html). Their methods/assumptions and stability measures differ from this specific ANOVA/Pearson bootstrap heuristic; formal guarantees do not transfer automatically.
