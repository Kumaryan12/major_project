# Reproduction results

Run date: 2026-09-20  
Configuration: `configs/paper.yaml`  
Random seed: 42

## Dataset and preprocessing audit

The reproduced cohort contains 178 subjects: 126 with an acute MI localization and 52 healthy controls. It includes 425 VCG recordings and 60,507 detected beats. The paper reports 60,527 beats, a difference of 20 beats (0.033%). Patient 294 is excluded for excessive signal noise, and MI subjects without an acute-location annotation are excluded.

| Class | Reproduced beats | Paper beats | Difference |
|---|---:|---:|---:|
| AMI | 6,333 | 6,402 | -69 |
| ALMI | 6,827 | 6,628 | +199 |
| ASMI | 11,489 | 11,333 | +156 |
| ASLMI | 271 | 272 | -1 |
| IMI | 12,410 | 12,702 | -292 |
| ILMI | 8,049 | 8,158 | -109 |
| IPMI | 49 | 48 | +1 |
| IPLMI | 2,686 | 2,711 | -25 |
| LMI | 444 | 460 | -16 |
| PMI | 429 | 466 | -37 |
| PLMI | 765 | 781 | -16 |
| HC | 10,755 | 10,566 | +189 |
| **Total** | **60,507** | **60,527** | **-20** |

The detector threshold is calibrated against the published beat-count table because the paper does not provide its Pan-Tompkins parameters. The class-level differences show that matching the total is not equivalent to recovering the unpublished R-peak annotations.

## Main comparison

| Protocol | Accuracy | Macro F1 | Macro ROC-AUC | Macro AP | Patient overlap |
|---|---:|---:|---:|---:|---:|
| Random beat-level 10-fold CV | **99.136%** | 0.9908 | 0.9999 | 0.9992 | 177-178 patients/fold |
| Patient-independent 10-fold CV | **41.385%** | 0.1908 | 0.5746 | 0.1960 | **0 patients/fold** |
| Paper's reported beat-level result | 99.80% | 0.9998 | per-class >0.88 | per-class >0.86 | not reported |

The faithful beat-level reimplementation comes within 0.664 percentage points of the paper's headline accuracy. Its fold range is 98.942%-99.322%. However, almost every patient appears in both training and testing in every beat-level fold.

With patient grouping enforced, accuracy drops by **57.75 percentage points** and varies from 28.70% to 51.38% across folds. The model performs best on healthy controls (F1 0.6477), ASMI (0.4727), and IMI (0.4563). The one-patient classes cannot be learned when their sole patient is held out, so ASLMI, IPMI, LMI, and PMI have zero F1. This is a dataset limitation as well as a model limitation.

## Beat-level class metrics

| Class | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| AMI | 0.9973 | 0.9869 | 0.9921 | 0.9998 | 0.9990 |
| ALMI | 0.9922 | 0.9814 | 0.9867 | 0.9998 | 0.9986 |
| ASMI | 0.9826 | 0.9934 | 0.9880 | 0.9998 | 0.9991 |
| ASLMI | 1.0000 | 0.9705 | 0.9850 | 1.0000 | 0.9979 |
| IMI | 0.9853 | 0.9969 | 0.9911 | 0.9999 | 0.9996 |
| ILMI | 0.9956 | 0.9901 | 0.9928 | 0.9998 | 0.9990 |
| IPMI | 1.0000 | 0.9592 | 0.9792 | 1.0000 | 1.0000 |
| IPLMI | 0.9985 | 0.9713 | 0.9847 | 0.9998 | 0.9986 |
| LMI | 1.0000 | 0.9955 | 0.9977 | 1.0000 | 0.9999 |
| PMI | 0.9977 | 1.0000 | 0.9988 | 1.0000 | 1.0000 |
| PLMI | 1.0000 | 0.9908 | 0.9954 | 1.0000 | 0.9985 |
| HC | 0.9975 | 0.9979 | 0.9977 | 0.9999 | 0.9998 |

## Interpretation

The implementation reproduces the paper's near-perfect result under its beat-level evaluation regime. The same 72-feature tensor pipeline does not generalize comparably to unseen patients. The 57.75-point gap is the strongest result for the next project phase and supports treating patient identity leakage as the baseline's central limitation.

All values above come from out-of-fold predictions. Full machine-readable results are in `results/metrics_beat.json`, `results/metrics_patient.json`, and the confusion-matrix CSV files.

