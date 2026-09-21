# Locked external validation: PTB to PTB-XL

Run date: 2026-09-21. This is a **transportability test**, not a replication of the paper's random beat-fold experiment and not a clinical performance claim.

## Protocol

The development set is the six-class PTB cohort of 164 patients and 55,863 beats. The representation is the 192-dimensional wavelet statistics chosen in the earlier [PTB ablation study](ABLATION_RESULTS.md). XGBoost uses the fixed depth-3, 250-tree candidate from that PTB-only study, with patient-balanced training weights. The fitted model is trained on all eligible PTB patients. There is no PTB-XL training, threshold selection, or hyperparameter tuning.

The external set is PTB-XL v1.0.3, recommended patient-disjoint fold 10. The [strict cohort audit](results/ptbxl_audit/summary.json) requires one unambiguous target MI location (`AMI`, `ALMI`, `ASMI`, `IMI`, `ILMI`) or `NORM` with no other diagnostic code. Records with multiple/non-target MI codes and patients with conflicting record statuses are excluded. This yields 1,185 ECGs from 1,092 patients. The same patient never appears in PTB-XL's other folds, although those folds are not used here.

PTB training uses measured 1,000 Hz Frank XYZ. PTB-XL testing uses 500 Hz 12-lead ECG, converted from independent leads I, II, V1–V6 to estimated XYZ by the published [Kors regression matrix](https://github.com/Tereshchenkolab/Pacing_spike_removal_v2/blob/main/kors.m), then resampled to 1,000 Hz. Denoising, R-peak detection, 651-sample beat windows, wavelet tensorization, and statistics are otherwise shared. Patient probabilities are the mean of all their beat probabilities, followed by argmax classification.

## Results

| Endpoint | Accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC | Macro average precision |
|---|---:|---:|---:|---:|---:|
| PTB-XL patients (n=1,092) | 56.23% | 24.21% | 0.242 | 0.685 | 0.253 |
| PTB-XL beats (n=13,564) | 54.00% | 24.53% | 0.239 | 0.665 | 0.244 |

| True class | Patients | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| AMI | 19 | 0.045 | 0.053 | 0.049 |
| ALMI | 5 | 0.000 | 0.000 | 0.000 |
| ASMI | 104 | 0.453 | 0.327 | 0.380 |
| IMI | 154 | 0.194 | 0.318 | 0.241 |
| ILMI | 24 | 0.022 | 0.083 | 0.035 |
| HC | 786 | 0.842 | 0.672 | 0.747 |

The majority-class (always-HC) patient accuracy is 786/1,092 = **71.98%**, above this model's 56.23%. Accuracy alone is therefore misleading here. The model does identify some ASMI and IMI patients, but localization is weak overall; in particular, ALMI has only five test patients and no correct classifications. The patient-level confusion matrix and complete machine-readable metrics are in [`results/ptbxl_external/`](results/ptbxl_external/).

## Interpretation and limits

External macro F1 falls from the PTB-only nested out-of-fold estimate of 0.487 to 0.242. That comparison is descriptive, not a controlled estimate of the effect of one factor: the acquisition system, measured versus derived VCG, recording duration, case mix, diagnostic labeling, and class prevalence all change at once. The strict label mapping favors specificity over cohort size but does not make PTB and PTB-XL clinical labels identical. PTB-XL is highly imbalanced after this filter (72.0% HC); AMI, ALMI, and ILMI have small test samples. The detector yields 13,564 extracted beats across 1,185 ten-second records, and its behavior on derived VCG has not been independently validated against beat annotations. The patient identifiers of PTB and PTB-XL belong to different databases; cross-database identity overlap has not been independently verified.

The result is a useful negative finding: the current PTB-trained localization model does **not** generalize well enough to support deployment or diagnostic use. Any domain adaptation or PTB-XL-specific training should be a new, explicitly labeled experiment with a separate untouched external test set, not a reinterpretation of this locked evaluation.

Source: [PTB-XL dataset and recommended folds](https://physionet.org/content/ptb-xl/1.0.3/).
