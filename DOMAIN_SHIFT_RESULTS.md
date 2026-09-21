# Paired PTB acquisition-domain investigation

Run date: 2026-09-21. This experiment follows the [locked PTB-XL external evaluation](EXTERNAL_RESULTS.md), but uses **PTB data only**. It does not refit, rescore, or alter the PTB-XL test result.

## Question and design

How much does replacing measured Frank XYZ with ECG-derived Kors XYZ change the current classifier's behavior, when patient identity, clinical label, recording, and beat positions are held fixed?

PTB provides simultaneous 1,000 Hz 12-lead ECG and measured Frank VCG for each recording. For the same 391 records, 164 patients, and 55,863 six-class beats used in the patient-independent PTB study, we read the eight independent ECG leads and apply the [Kors regression matrix](https://github.com/Tereshchenkolab/Pacing_spike_removal_v2/blob/main/kors.m). Both representations use the same 651-sample windows centered on the **original measured-VCG R peaks**. This removes beat alignment and patient composition as explanations for the paired classifier comparison. The 192 wavelet-statistics features, fixed XGBoost depth-3 model, patient-balanced weights, and original frozen ten patient folds are reused. Each test fold consists of patients absent from its training fold.

The three conditions are:

| Condition | Training VCG | Held-out VCG |
|---|---|---|
| Measured→measured | Frank XYZ | Frank XYZ |
| Measured→derived | Frank XYZ | Kors-estimated XYZ |
| Derived→derived | Kors-estimated XYZ | Kors-estimated XYZ |

## Paired signal and feature agreement

Medians across the 391 complete recordings:

| Lead | Centered signal correlation | Derived/measured RMS ratio | Centered RMSE / measured RMS |
|---|---:|---:|---:|
| X | 0.740 | 1.156 | 0.972 |
| Y | 0.821 | 1.193 | 0.809 |
| Z | 0.584 | 1.162 | 1.110 |

Across the 192 wavelet-statistics coordinates, the median measured-versus-derived correlation across matched beats is **0.869**. The median absolute paired feature gap is **0.189 measured-domain standard deviations**. These summaries show a material, nonuniform waveform change—particularly for Z—even though many feature coordinates remain correlated.

As a separate detector check, the R-peak detector run on derived PTB VCG finds a peak within 100 ms of a measured-domain beat for a median of **100%** of beats per record (mean 95.9%). It detects 56,378 complete beats versus 55,863 original measured-domain beats on these records. This is only a PTB check; it does not validate R-peak detection on PTB-XL. Classifier comparisons above deliberately use shared measured-domain peak positions, not the derived detector's peaks.

## Patient-independent results

All conditions use the same 164 held-out patients across the frozen folds.

| Condition | Accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC | Macro average precision |
|---|---:|---:|---:|---:|---:|
| Measured→measured | 52.44% | 46.50% | 0.469 | 0.819 | 0.481 |
| Measured→derived | 47.56% | 42.30% | 0.422 | 0.811 | 0.470 |
| Derived→derived | 51.83% | 46.54% | 0.467 | 0.814 | 0.466 |

Measured→derived loses **0.046 macro F1** relative to measured→measured. Its paired, class-stratified patient bootstrap 95% percentile interval is **−0.108 to +0.017** (5,000 draws; seed 42). Derived→derived recovers **0.044 macro F1** relative to measured→derived, with interval **−0.025 to +0.116**. Both intervals include zero. They describe uncertainty on this PTB cohort; neither establishes a statistically certain population effect.

## What this explains—and what it does not

The directional pattern is consistent with a train/test acquisition mismatch: the Frank-trained classifier loses performance when its held-out patients are represented as derived VCG, while a classifier trained in the derived domain regains most of that loss. Because the intervals include zero, the size of this effect is uncertain. Importantly, derived→derived performance is still near the measured→measured result on PTB, so the Kors conversion does not appear to destroy most of the signal useful to this model **within PTB**.

This paired test does **not** explain away the much lower **0.242 macro F1** observed in the [PTB-XL external evaluation](EXTERNAL_RESULTS.md). PTB-XL also differs in device, era, 500 Hz acquisition, 10-second recording duration, diagnostic labeling, class prevalence, and possibly other unmeasured factors. The PTB-XL cohort is especially sparse for ALMI. This experiment isolates the conversion effect only for PTB's simultaneous recordings; it cannot quantify the separate contributions of those other shifts or prove that domain-matched training would fix PTB-XL performance. The fixed depth-3 model here also differs from the nested model selection used for the earlier 0.487 PTB estimate; comparisons should use this report's internal measured→measured control of 0.469.

Full metrics, patient probabilities, confusion matrices, and per-record signal diagnostics are in [`results/domain_shift/`](results/domain_shift/). Reproduction commands are in the [README](README.md#paired-acquisition-domain-investigation). The large ECG files and paired feature archive are intentionally gitignored.
