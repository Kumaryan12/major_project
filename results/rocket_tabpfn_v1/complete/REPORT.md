# Feature-selected convolutional ridge versus local TabPFN

Feature-selected ridge versus pinned local TabPFN-v2 on repeatedly inspected PTB convolutional features; adaptive development, no independent confirmation or established MI SOTA.

All 164 original patients. 660 new classifiers restored (420 ridge / 240 PFN), including 120 outer models. 120 source feature contexts reconstructed, 210 frozen ridge contexts restored, 120 bootstrap caches replayed and 240 masks verified. Pinned source artifacts unchanged.

| Procedure | Mean member macro F1 ± pipeline-seed SD | Prespecified seed-ensemble F1 |
|---|---:|---:|
| full_ridge | 0.573 ± 0.019 | 0.584 |
| selected_ridge | 0.546 ± 0.006 | 0.544 |
| selected_tabpfn | 0.502 ± 0.028 | 0.531 |
| ridge_k32 | 0.462 ± 0.016 | 0.495 |
| tabpfn_k32 | 0.470 ± 0.043 | 0.510 |
| ridge_k64 | 0.536 ± 0.016 | 0.576 |
| tabpfn_k64 | 0.509 ± 0.022 | 0.532 |

The three primary procedures are full_ridge, selected_ridge and selected_tabpfn. Fixed-budget rows are declared controls, not additional independently fitted selected models. Every seed is retained; ensemble F1 is not mean member F1. No best seed/budget is chosen using this table.

| Ensemble comparison | Δ macro F1 [descriptive interval] |
|---|---:|
| selected_tabpfn_minus_selected_ridge | -0.013 [-0.084, +0.056] |
| selected_tabpfn_minus_full_ridge | -0.053 [-0.128, +0.021] |
| selected_ridge_minus_full_ridge | -0.040 [-0.126, +0.052] |
| tabpfn_k32_minus_ridge_k32 | +0.015 [-0.046, +0.071] |
| tabpfn_k64_minus_ridge_k64 | -0.044 [-0.115, +0.022] |

## Primary ensemble class F1

| Class | full_ridge | selected_ridge | selected_tabpfn |
|---|---:|---:|---:|
| AMI | 0.345 | 0.080 | 0.080 |
| ALMI | 0.483 | 0.537 | 0.516 |
| ASMI | 0.633 | 0.480 | 0.533 |
| IMI | 0.542 | 0.679 | 0.586 |
| ILMI | 0.578 | 0.640 | 0.583 |
| HC | 0.925 | 0.849 | 0.887 |

## Protocol and interpretation

Source inputs are the completed combined MultiRocket-style/HYDRA recording-template features, with their exact original train/query memberships and transform seeds. No new representation, waveform/label change or full-cohort transform was fitted. Feature contexts were independently reproduced from their own training templates during verification. The selected representation was chosen after earlier development results; this is adaptive development on the same repeatedly inspected patients, not a fresh test.

Each partition fits class-stratified bootstrap consensus using 64 training resamples and an independent 32-resample diagnostic panel, ANOVA ranking and absolute Pearson pruning at 0.95. Budgets are at most 32/64 columns, with actual selected counts recorded. Diagnostic-panel support does not choose masks/budgets. Both classifiers see exactly the same selected raw columns within a budget/partition. No validation or outer labels enter feature selection; outer-trained masks never initialize inner contexts.

Ridge uses training-only StandardScaler, class-balanced weights and alpha [1,100]. TabPFN uses the existing pinned v2 checkpoint, eight ensemble members, temperature 0.9, unbalanced probabilities and its own training-context preprocessing; it runs in isolated local workers. These are classifier procedures with different preprocessing/prior/weighting, not a pure causal isolation of nonlinearity. No PFN checkpoint/member-count/temperature/prior correction search or hosted prediction service is used.

Selected ridge chooses among budget/alpha candidates by highest arithmetic mean inner patient F1; selected PFN chooses its budget the same way. Ties follow configured budget/alpha order. Selected procedures can choose different masks/counts. Fixed-budget ridge-versus-PFN controls isolate the shared mask/count more directly; ridge alpha remains training-only selected. Both budgets' outer predictions were declared for reporting and never pick a global winner. Primary selected predictions reuse these outer models without extra fitting.

All source seeds 42/123/2026 are retained, with selector-seed offset 100000019. The PFN inference seed is 42+1000*outer_fold+inner_fold, common across source-transform seeds; member variation therefore includes source transforms and bootstrap selection, not three independent foundation-model pretrainings. Equal probability averaging is prespecified for every procedure. Ridge softmax scores are not calibrated clinical probabilities.

Frozen full_ridge reproduces the earlier feature ensemble 0.584266; historical neural 0.584570 is a different separate development reference. All original 164 patients/391 records/55,863 beats and frozen folds remain. Templates compress beat variation; this study does not assess an added variability branch, ECG input or external pretraining.

Intervals use 5,000 class-stratified paired patient draws conditional on fixed predictions, without model refits or correction for CV dependence, adaptive development or multiple comparisons. Pipeline-seed SD is not generalization uncertainty. No result establishes independent confirmation, clinical validity, SOTA or guaranteed 0.80–0.90 macro F1.
