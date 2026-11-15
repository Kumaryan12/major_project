# TabPFN-v2 on tensor-derived VCG features

Pinned TabPFN-v2 feature-classifier development on repeatedly inspected PTB patient folds; local frozen synthetic-prior foundation model, not demonstrated MI SOTA or independent confirmation.

All 164 original patients; 10 frozen outer folds, 3 inner folds. Validated 240 reused XGBoost inner fits, 120 new PFN inner contexts, 20 new PFN outer contexts and 10 fresh XGBoost outer fits. Raw/balanced inner scores share each PFN context, not independent fits.

| Classifier | Patient macro F1 | Accuracy |
|---|---:|---:|
| xgboost | 0.513 | 57.3% |
| tabpfn | 0.542 | 61.6% |
| tabpfn_balanced | 0.498 | 56.7% |

Pooled out-of-fold patient results. All models use the unchanged morphology/combined pools, bootstrap selection and budgets from the source study. One fixed fold/inference-seed policy; eight PFN inference members are not eight independent training seeds.

| Paired difference | Δ macro F1 | Descriptive patient interval |
|---|---:|---:|
| tabpfn_minus_xgboost | +0.030 | [-0.055, +0.115] |
| tabpfn_balanced_minus_xgboost | -0.014 | [-0.095, +0.070] |
| tabpfn_balanced_minus_tabpfn | -0.044 | [-0.097, +0.009] |

5,000 paired class-stratified patient draws, conditional on fitted predictions. No model refits or correction for correlated folds, repeated development or multiple comparisons. These are descriptive intervals, not independent confirmation.

## Exact inference and fair-comparison boundaries

Pinned public Nature-era TabPFN-v2 checkpoint, revision 275f19d2add9c2c0de38f9ebaded3856599fa2e7, SHA256 cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49; tabpfn==9.2.0. This is not the latest TabPFN-3.5. Local cpu inference, float32, 2 threads, 8 fixed permutation/preprocessing members, temperature 0.9, no gradient fine-tuning or hosted prediction service. Model-specific preprocessing is fitted to the training context. The foundation weights remain frozen.

TabPFN sees raw selected feature columns; the bootstrap selector's class-balanced training scaler is used for ranking/pruning, not applied again to PFN input. XGBoost retains its established selected-column scaling and class-balanced fitting weights. TabPFN's API has no sample-weight argument: raw probabilities are reported separately from a fixed inverse-training-class-count renormalization. This correction uses no validation/test prevalence, creates no duplicate patients and is not clinical calibration. It is not equivalent to weighted tree training. Both PFN policies are predeclared and retained.

For each outer fold, family/budget is selected separately for each PFN probability policy by mean inner macro F1, with configured-order ties. Four PFN candidates versus eight XGBoost candidates (which also tune depth 2/3). No PFN model/checkpoint, member-count, temperature, threshold or prior-correction strength search. Data/splits/feature-mask rules are matched; parameter counts, algorithms and pretraining compute are not matched. Every outer model is freshly fitted on outer-training patients without test monitoring.

Every relevant bootstrap selector cache was replayed (72 caches); all 10 fresh XGBoost control predictions reproduced the previous study. Every saved PFN context was restored and reproduced its outer probabilities/classes. A six-class real-backend smoke test checks finite predictions, save/load, correction equivalence and invariance to unrelated query rows. Learned normalization/SVD and labels never use outer-test training context.

## Per-class F1

| Class | xgboost | tabpfn | tabpfn_balanced |
|---|---:|---:|---:|
| AMI | 0.294 | 0.357 | 0.300 |
| ALMI | 0.452 | 0.452 | 0.400 |
| ASMI | 0.414 | 0.500 | 0.383 |
| IMI | 0.500 | 0.517 | 0.500 |
| ILMI | 0.605 | 0.558 | 0.560 |
| HC | 0.811 | 0.870 | 0.846 |

These repeatedly inspected PTB patients remain development data. The historical neural 0.550 ± 0.040 is a different model/aggregation protocol and not this paired control. A pretrained tabular architecture does not establish MI SOTA, clinical validity or the 0.80–0.90 target. Independent cohort validation is still required. No class, patient, beat, original waveform or R position changed.

Reproduce: `.venv/bin/python -m mi_localization.tabpfn_study`, then `.venv/bin/python -m mi_localization.tabpfn_report`. Checkpoint, contexts, bootstrap draws, models and individual patient predictions remain local; aggregate metrics/audits are public.
