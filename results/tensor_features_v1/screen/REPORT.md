# Tensor feature design screen

Feature development inside outer-training patients only. Inner scores are selection diagnostics, not held-out project macro F1. No outer-test inference or final neural training.

Completed 660 compact linear probe fits. **Zero outer-test predictions**. Selected training-only packages for 10 frozen outer contexts.

Each candidate fits class-balanced training-only scaling, ANOVA ranking and greedy absolute Pearson redundancy pruning. The unselected-statistics control removes only constant training columns. Both use the same class-balanced logistic probe. Candidate family, feature budget and C are selected by mean inner patient macro F1. No cohort-wide selector is applied before CV.

| Outer training context | Family | Budget | Actual features | C | Inner selection diagnostic |
|---|---|---:|---:|---:|---:|
| 1 | combined | 24 | 24 | 0.1 | 0.464 |
| 2 | morphology | 24 | 24 | 0.1 | 0.482 |
| 3 | morphology | 64 | 64 | 0.1 | 0.493 |
| 4 | morphology | 24 | 24 | 0.1 | 0.471 |
| 5 | morphology | 64 | 64 | 0.1 | 0.456 |
| 6 | morphology | 24 | 24 | 0.1 | 0.448 |
| 7 | morphology | 64 | 64 | 0.1 | 0.479 |
| 8 | combined | 24 | 24 | 0.1 | 0.489 |
| 9 | combined | 64 | 64 | 0.1 | 0.458 |
| 10 | morphology | 64 | 64 | 1.0 | 0.517 |

These inner scores were optimized during selection and are optimistic. They must not be reported as project performance, improvement over 0.550, or evidence of 0.80–0.90. The folds overlap strongly in training patients; feature frequencies are descriptive reproducibility diagnostics, not independent stability probabilities or clinical biomarkers.

Features keep signed morphology, amplitude-aware multiscale moments and XYZ covariance plus normalized correlation. Covariance and correlation are complementary candidate summaries, not a reproduction of the sepsis paper. Fixed windows are not independently annotated cardiac phases. Multiscale reconstruction checks the complete tensor, while selected summaries remain lossy. A single probe cannot identify a universally best representation for every eventual classifier.

Next: inspect selected feature names/frequencies and source fidelity, freeze the selection procedure, then compare final classifiers within the same outer-fold protocol. Refit selectors inside each fold; do not pool these selected lists into a global mask and reuse it on these patients. Any later learned PCA/Tucker basis also belongs inside training folds. Repeatedly inspected PTB remains development data and needs new-cohort confirmation.
