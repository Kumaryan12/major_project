# Leakage-free patient benchmark

Run date: 2026-09-20  
Configuration: `configs/patient_benchmark.yaml`  
Fixed folds: `configs/patient_folds.csv`

## Research question

Can the paper's 72-dimensional wavelet-Tucker features localize myocardial infarction in completely unseen patients when classes with too few independent subjects are removed from the primary endpoint?

The primary benchmark contains 164 patients and six classes with at least 10 subjects: AMI (17), ALMI (16), ASMI (27), IMI (29), ILMI (23), and HC (52). All 10 test folds contain every class. No patient occurs in both training and testing, and hyperparameters are selected using a separate three-fold patient split inside each outer training fold.

Training beats are weighted so every class has equal total weight and every patient has equal weight within a class. Test scores are averaged across all beats from each patient before assigning the patient diagnosis.

## Results

| Model | Patient accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC | Macro AP | Beat accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | **47.56%** | 39.27% | 0.3797 | 0.7642 | **0.4147** | **39.30%** |
| Linear SVM | 43.90% | 37.18% | 0.3661 | 0.7667 | 0.3837 | 35.43% |
| XGBoost | 46.95% | **40.00%** | **0.4027** | **0.7702** | 0.3993 | 38.08% |
| Healthy-only majority baseline | 31.71% | 16.67% | 0.0802 | - | - | - |

XGBoost is the primary winner because model selection was defined in advance using patient-level macro F1, which treats all diagnostic classes equally. Random Forest has slightly higher raw accuracy because the cohort contains more healthy controls.

Stratified patient-bootstrap 95% intervals (5,000 resamples) are:

| Model | Accuracy interval | Macro-F1 interval |
|---|---:|---:|
| Random Forest | 40.85%-54.27% | 0.307-0.451 |
| Linear SVM | 37.20%-51.22% | 0.298-0.432 |
| XGBoost | 39.63%-54.27% | 0.323-0.480 |

The intervals overlap substantially, so the present PTB cohort does not establish that one classifier is definitively superior. The defensible conclusion is that XGBoost leads on the prespecified macro-F1 endpoint, while all three models remain limited on unseen patients.

## XGBoost class results

| Class | Patients | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AMI | 17 | 0.200 | 0.176 | 0.188 |
| ALMI | 16 | 0.556 | 0.313 | 0.400 |
| ASMI | 27 | 0.341 | 0.519 | 0.412 |
| IMI | 29 | 0.359 | 0.483 | 0.412 |
| ILMI | 23 | 0.333 | 0.217 | 0.263 |
| HC | 52 | 0.800 | 0.692 | 0.742 |

Healthy controls are identified most reliably. AMI and ILMI remain especially difficult, while ASMI and IMI are frequently confused with other MI locations. This indicates that the next improvement should target representations that remove patient-specific morphology while preserving location-specific morphology, rather than simply adding more trees.

## Hyperparameter-selection audit

- Random Forest selected `min_samples_leaf=1` in 7/10 outer folds and `5` in 3/10.
- Linear SVM selected `C=0.01` in 6/10 folds and `C=0.1` in 4/10; `C=1.0` was never selected.
- XGBoost selected depth 6 in all 10 folds over depth 3. This shows that a later tuning study should test additional capacity, but expanding the search after seeing test results must be reported as a new experiment rather than folded into this benchmark.

## Conclusion

Restricting the task to adequately represented classes and aggregating beats at patient level improves interpretability but does not recover the paper's beat-level performance. The best patient macro F1 is 0.403, compared with the reproduced beat-level macro F1 of 0.991. The core scientific finding therefore remains: the tensor representation strongly separates beats from already-seen patients but generalizes poorly to new patients.

Machine-readable fold-level tuning decisions, class reports, ROC-AUC/AP values, and confusion matrices are stored in `results/patient_benchmark/`.

