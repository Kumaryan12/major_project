# Matched probabilistic MI subtype readout

## Research question

On the frozen measured-XYZ features, test whether estimating MI/HC and then estimating the location conditional on MI improves six-class patient-level macro F1. The historical 0.603374 linear selector is a descriptive reference. The matched control is flat multinomial logistic regression.

## Features, head and patient folds

Reuse the four compatible, independently verified feature caches. In each outer fold and feature seed, freeze the representation to the originally recorded selection from inner scores of the paired kernel study. Both new models receive that same representation. Its transforms always come from the corresponding exact training partition. Historical representation selection reuses the original inner patients. The new head comparisons use the unchanged held-out outer patients.

Fit a flat six-class logistic model and a two-head model with binary MI/HC logistic regression over all training patients and multinomial logistic regression over MI training patients. Fit with weights that give each original class the same mass. Before regularization, this gives training MI and HC mass 5/6 and 1/6 in the detection head. Normalize weights to mean one in each fit. Use the same three C candidates (0.001, 0.01, 0.1), optimizer and convergence tolerance for both procedures. The hierarchical model uses one shared C for its two heads. Select each procedure's C on six-class patient macro F1 in its three inner folds with configured-order ties.

For each query patient set P(HC) = 1 − P(MI) and P(subtype) = P(MI) × P(subtype given MI). Apply a single six-class argmax after averaging probabilities over all three seeds. Report every patient in all ten fixed folds and each seed, including patients who are misclassified as HC or MI. The primary contrast is hierarchical minus flat. Report paired patient macro F1 intervals, per-class F1, and counts of subtype confusions, MI predicted HC and HC predicted MI. Historical results serve as development references. The bootstrap conditions on the saved predictions.

## Verification

Use the original 164 patients, 391 records and 55,863 source beats. Independently check the pinned feature chain and all cached row memberships. The full comparison has 120 matched contexts and 600 procedure fits, comprising 900 logistic head fits. Verify coefficients and score-to-probability calculations and replay all contexts. Check historical prediction hashes and reproduce its metrics. The inner-only smoke uses fold 1's training patients and seed 42. Private fit coefficients, patient predictions and context caches stay local.

Run `.venv/bin/python scripts/run_probabilistic_subtype.py --smoke` followed by `.venv/bin/python scripts/run_probabilistic_subtype.py --publish` from the repository root. The detached Mac run logs to `tmp/probabilistic_subtype_v1.log`. The verified report is `results/probabilistic_subtype_v1/complete/REPORT.md`.

[Project decision and error budget](path_to_080_macro_f1.md)
