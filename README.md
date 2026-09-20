# VCG myocardial infarction localization via tensor decomposition

This repository is an end-to-end, reproducible reimplementation of Zhang et al., *Automated Localization of Myocardial Infarction From Vectorcardiographic via Tensor Decomposition* (IEEE TBME, 2023). It also includes the patient-independent experiment proposed for the follow-on project.

> Research code only. It is not a medical device and must not be used for clinical diagnosis.

## Implemented pipeline

1. Download the open PTB Diagnostic ECG Database headers and Frank `vx`, `vy`, `vz` files (not the unused 12-lead files).
2. Keep healthy controls and MI records that map to the paper's 11 infarct locations.
3. Wavelet-denoise the VCG and detect R peaks with a Pan-Tompkins-style detector.
4. Extract 651-sample beats: 250 samples before through 400 samples after each R peak.
5. Construct each `3 x 651 x 8` tensor from the original beat and reconstructed bior6.8 details D3-D9.
6. Preserve the lead and subband modes while compressing the time mode to rank 3 by truncated SVD, yielding `3 x 3 x 8 = 72` features.
7. Train a 500-tree bootstrap random forest (the scikit-learn equivalent of MATLAB TreeBagger).
8. Evaluate both random beat-level 10-fold CV and patient-independent stratified group 10-fold CV.

## Setup and run

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/mi-localization run --config configs/paper.yaml --mode both
```

The public data download is a few hundred MB because only `.hea` and `.xyz` files are fetched. Intermediate features are cached in `data/processed/paper_features.npz`; evaluation can then be repeated without signal processing:

```bash
.venv/bin/mi-localization evaluate --mode beat
.venv/bin/mi-localization evaluate --mode patient
```

Outputs are written to `results/`: JSON metrics, out-of-fold predictions, confusion matrices, plots, and a final fitted model.

## Reproduction boundaries

The paper specifies bior6.8 wavelet denoising, Pan-Tompkins R-peak detection, D3-D9 plus the original signal, rank `(3, 3, 8)`, 72 features, 500 trees, and random 10-fold beat CV. It does **not** publish its code or fully specify:

- the wavelet threshold rule and boundary mode;
- which VCG lead drives R-peak detection and its thresholds;
- normalization of spelling variants in PTB clinical summaries;
- random seed and exact fold membership;
- TreeBagger options beyond the tree count;
- SVD sign handling and vectorization order.

Those choices are deterministic here and exposed in `configs/paper.yaml`. Consequently, matching the paper's 99.80% exactly is not guaranteed. More importantly, random beat folds contain beats from the same patients on both sides of the split. The `patient` mode removes this leakage and is the scientifically relevant extension.

## Paper target counts

The paper reports 60,527 beats from 178 subjects (126 MI and 52 healthy): AMI 6,402; ALMI 6,628; ASMI 11,333; ASLMI 272; IMI 12,702; ILMI 8,158; IPMI 48; IPLMI 2,711; LMI 460; PMI 466; PLMI 781; HC 10,566. The paper's cohort excludes patient294 for excessive signal noise and excludes 21 MI subjects without acute-location annotations. The generated `manifest.csv` and `beats.csv` make any count differences auditable.

## Tests

```bash
.venv/bin/pytest -q
```
