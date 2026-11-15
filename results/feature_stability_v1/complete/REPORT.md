# Feature consensus: completed patient evaluation

Bootstrap feature-consensus development on repeatedly inspected PTB patient folds; no independent confirmation, clinical biomarker or formal stability-selection error guarantees.

All 164 original patients, ten frozen outer folds and three inner folds. Validated 1500 classifier fit slots: 960 new inner fits, 480 verified original inner fits and 60 fresh outer refits. Fully replayed 80 bootstrap caches (7680 base-selector fits).

| Selector | Linear patient macro F1 | Shallow-tree patient macro F1 |
|---|---:|---:|
| original | 0.498 | 0.420 |
| bootstrap | 0.497 | 0.513 |
| group_bootstrap | 0.463 | 0.357 |

Pooled out-of-fold patient scores; one fixed model/resampling seed, not a neural three-seed ensemble. All arms and patients retained.

## Paired comparisons

| Difference | Δ macro F1 | Descriptive patient interval |
|---|---:|---:|
| linear_bootstrap_minus_linear_original | -0.001 | [-0.084, +0.081] |
| linear_group_bootstrap_minus_linear_original | -0.035 | [-0.115, +0.048] |
| linear_group_bootstrap_minus_linear_bootstrap | -0.034 | [-0.115, +0.050] |
| tree_bootstrap_minus_tree_original | +0.092 | [+0.027, +0.158] |
| tree_group_bootstrap_minus_tree_original | -0.063 | [-0.141, +0.016] |
| tree_group_bootstrap_minus_tree_bootstrap | -0.155 | [-0.234, -0.078] |

5,000 class-stratified paired patient draws, conditional on fitted predictions; no model refits or adjustment for correlated folds, repeated development or multiple comparisons. Descriptive intervals, not independent confirmation.

## Matched resampling support

| Family | Budget | Selector | Actual features | Audit feature support | Audit group support |
|---|---:|---|---:|---:|---:|
| morphology | 24 | original | 24.0 | 0.589 | 0.758 |
| morphology | 24 | bootstrap | 24.0 | 0.594 | 0.768 |
| morphology | 24 | group_bootstrap | 24.0 | 0.524 | 0.648 |
| morphology | 64 | original | 64.0 | 0.656 | 0.859 |
| morphology | 64 | bootstrap | 64.0 | 0.658 | 0.855 |
| morphology | 64 | group_bootstrap | 64.0 | 0.551 | 0.678 |
| combined | 24 | original | 24.0 | 0.486 | 0.570 |
| combined | 24 | bootstrap | 24.0 | 0.494 | 0.578 |
| combined | 24 | group_bootstrap | 24.0 | 0.455 | 0.503 |
| combined | 64 | original | 64.0 | 0.503 | 0.604 |
| combined | 64 | bootstrap | 64.0 | 0.512 | 0.605 |
| combined | 64 | group_bootstrap | 64.0 | 0.477 | 0.549 |

Support is the fraction of 32 separate training-patient bootstrap base-selector fits that include a chosen feature/group, averaged over chosen columns and matched outer-training contexts. The audit panel never ranks final features or tunes candidates. This measures resampling support against the base selector, not stability of retraining the complete consensus algorithm, independent biological importance or false-discovery control. Training contexts overlap; no independent stability probability is claimed.

## Exact method and boundaries

Fixed pool: original signed morphology and combined summaries only; no new tensor, feature family, beat, patient, label, time alignment or preprocessing. Original uses training ANOVA ranking and Pearson pruning. Bootstrap ranks by selection frequency over 64 class-stratified patient bootstrap fits, then original F score and index. Group bootstrap first ranks fixed semantic groups by any-member selection frequency, then chooses representative columns; at most one column per group. Groups preserve source family, median/IQR, XYZ, wavelet band/window/statistic or coefficient order and merge only corresponding physical-view slots. These groups are a hypothesis, not proof of clinically equivalent features.

All methods fit class-balanced scaling on training patients, keep at most 24/64 features, and remove full-training correlations ≥0.95. Bootstrap draws retain patient multiplicity and original class counts. Training and audit random streams are separate and shared across classifiers/candidates in a partition. No minimum-frequency threshold is tuned; consensus can retain low-support columns to honor the same maximum budget. The original selector's probabilities are not altered by its descriptive bootstrap audit.

Each method/learner has the same eight family/budget/model candidates. Inner patient macro F1 selects the candidate; exact ties favor configured order. Outer selectors/models are fresh training-only refits. Linear C 0.1/1.0; tree depth 2/3, 100 trees, rate 0.05, lambda 1, class-balanced weights applied once, one CPU thread. Source templates use held-out recordings' own unlabeled beats for offline within-exam context. Every saved model must reproduce its outer predictions.

Original-control outer predictions matched 20 compatible prior fits. The pool was chosen using previous development findings; the same repeatedly inspected patients are not independent confirmation. No classical stability-selection theorem, clinical biomarker, SOTA or 0.80–0.90 score is guaranteed.

## Per-class F1

| Class | linear_original | linear_bootstrap | linear_group_bootstrap | tree_original | tree_bootstrap | tree_group_bootstrap |
|---|---:|---:|---:|---:|---:|---:|
| AMI | 0.278 | 0.359 | 0.308 | 0.270 | 0.294 | 0.057 |
| ALMI | 0.343 | 0.400 | 0.154 | 0.320 | 0.452 | 0.222 |
| ASMI | 0.383 | 0.375 | 0.423 | 0.310 | 0.414 | 0.370 |
| IMI | 0.526 | 0.533 | 0.500 | 0.298 | 0.500 | 0.259 |
| ILMI | 0.583 | 0.444 | 0.511 | 0.491 | 0.605 | 0.400 |
| HC | 0.876 | 0.871 | 0.885 | 0.833 | 0.811 | 0.835 |

Reproduce: `.venv/bin/python -m mi_localization.stability_study`, then `.venv/bin/python -m mi_localization.stability_report`. Feature matrices, bootstrap selections/draws, fitted models and patient predictions remain local; aggregate metrics/audits are public.
