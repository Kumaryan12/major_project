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

The completed baseline and patient-independent findings are summarized in [`RESULTS.md`](RESULTS.md).

## Leakage-free model benchmark

The follow-on benchmark uses a fixed patient split and the six classes with at least ten independent patients (`AMI`, `ALMI`, `ASMI`, `IMI`, `ILMI`, `HC`). Hyperparameters are selected inside each training fold using three-fold patient-grouped validation and patient-level macro F1. Beats are weighted so classes contribute equally and patients contribute equally within each class. Test predictions are averaged across all beats belonging to a patient before classification.

```bash
.venv/bin/mi-localization benchmark --models rf svm xgboost
```

The fixed outer assignments are stored in `configs/patient_folds.csv`; benchmark metrics and patient confusion matrices are written to `results/patient_benchmark/`.

Completed results and interpretation are in [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).

## Representation ablations

Generate the shared feature bank, then evaluate every representation with a fixed XGBoost model and the frozen patient folds:

```bash
.venv/bin/mi-localization ablation-features
.venv/bin/mi-localization ablation
```

The study compares Tucker ranks 1/2/3/5/8, original versus denoised signals, amplitude-normalized VCG, individual X/Y/Z leads, removal of wavelet subbands, and wavelet statistics without Tucker compression. These are exploratory comparisons on the frozen outer folds; the strongest candidate must subsequently be confirmed with nested tuning.

The wavelet-statistics candidate can be rerun with the benchmark's nested selection protocol:

```bash
.venv/bin/mi-localization benchmark --benchmark-config configs/wavelet_statistics_confirm.yaml --models xgboost
```

Completed comparisons and their limitations are in [`ABLATION_RESULTS.md`](ABLATION_RESULTS.md).

## Locked external validation on PTB-XL

The follow-on external test trains the PTB-selected wavelet-statistics/XGBoost model on all 164 eligible PTB patients, then evaluates it **once** on PTB-XL's patient-disjoint recommended fold 10. PTB-XL has 12-lead ECG rather than measured Frank VCG, so the eight independent ECG leads are converted to estimated XYZ with the published Kors regression matrix before applying the same beat extraction and 192-feature representation. No PTB-XL record is used to select the representation, model, or hyperparameters.

```bash
.venv/bin/mi-localization ptbxl-audit
.venv/bin/mi-localization ptbxl-features
.venv/bin/mi-localization ptbxl-external
```

The audit downloads PTB-XL metadata and freezes a strict six-class test manifest. Feature generation downloads the 500 Hz waveforms for that manifest (about 1,185 records), converts them to 1,000 Hz, and caches the external feature matrix. The final command uses the previously generated PTB feature bank from `ablation-features`. All configuration is in [`configs/ptbxl_external.yaml`](configs/ptbxl_external.yaml); the cohort and results are in [`EXTERNAL_RESULTS.md`](EXTERNAL_RESULTS.md). Raw waveforms, feature archives, and the fitted model are intentionally not committed.

## Paired acquisition-domain investigation

PTB records contain simultaneous 12-lead ECG and measured Frank XYZ. A separate experiment converts each PTB ECG to Kors XYZ and extracts statistics at the **same R-peak positions** as measured XYZ. It uses the frozen PTB patient folds to compare measured→measured, measured→derived, and derived→derived performance without touching the PTB-XL test.

```bash
.venv/bin/mi-localization domain-features
.venv/bin/mi-localization domain-evaluate
```

Configuration is in [`configs/domain_shift.yaml`](configs/domain_shift.yaml); findings and limitations are in [`DOMAIN_SHIFT_RESULTS.md`](DOMAIN_SHIFT_RESULTS.md). The additional PTB ECG downloads and paired feature archive are cached locally and not committed.

## Reproduction boundaries

The paper specifies bior6.8 wavelet denoising, Pan-Tompkins R-peak detection, D3-D9 plus the original signal, rank `(3, 3, 8)`, 72 features, 500 trees, and random 10-fold beat CV. It does **not** publish its code or fully specify:

- the wavelet threshold rule and boundary mode;
- which VCG lead drives R-peak detection and its thresholds;
- normalization of spelling variants in PTB clinical summaries;
- random seed and exact fold membership;
- TreeBagger options beyond the tree count;
- SVD sign handling and vectorization order.

Those choices are deterministic here and exposed in `configs/paper.yaml`. Consequently, matching the paper's 99.80% exactly is not guaranteed. More importantly, random beat folds contain beats from the same patients on both sides of the split. The `patient` mode removes this leakage and is the scientifically relevant extension.

The Pan-Tompkins threshold scale is set to `0.06`, selected against the only detector target published by the paper: its class-wise beat-count table. On the matched cohort this implementation detects 60,507 beats versus the reported 60,527 (0.03% difference). This calibration is preprocessing reproduction, not classifier tuning; the choice is recorded to avoid presenting an undocumented threshold as independently derived.

## Paper target counts

The paper reports 60,527 beats from 178 subjects (126 MI and 52 healthy): AMI 6,402; ALMI 6,628; ASMI 11,333; ASLMI 272; IMI 12,702; ILMI 8,158; IPMI 48; IPLMI 2,711; LMI 460; PMI 466; PLMI 781; HC 10,566. The paper's cohort excludes patient294 for excessive signal noise and excludes 21 MI subjects without acute-location annotations. The generated `manifest.csv` and `beats.csv` make any count differences auditable.

## Tests

```bash
.venv/bin/pytest -q
```
