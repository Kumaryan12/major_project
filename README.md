# VCG myocardial infarction localization via tensor decomposition

This repository is an end-to-end, reproducible reimplementation of Zhang et al., *Automated Localization of Myocardial Infarction From Vectorcardiographic via Tensor Decomposition* (IEEE TBME, 2023). It also includes the patient-independent experiment proposed for the follow-on project.

> Research code only. It is not a medical device and must not be used for clinical diagnosis.

For professor-facing material, see the four-page [project summary](output/pdf/mi_localization_summary_report.pdf) or the more detailed [major-project progress report](output/pdf/mi_localization_progress_report.pdf). Rebuild them with `pip install -e '.[report]'`, followed by `python scripts/build_summary_report.py` or `python scripts/build_professor_progress_report.py`.

For Aryan's implementation section of the presentation, use the [implementation and results deck](output/pptx/mi_implementation_presentation.pptx). Slides 1–14 cover the completed pipeline and experiments, and slides 15–17 provide technical backup. Each slide includes presenter notes. The [speaking guide](docs/implementation_presentation_notes.md) includes transitions, timing and likely questions.

For a formal monochrome version, use the [LaTeX Beamer source](output/latex/mi_implementation.tex), [compiled slide PDF](output/pdf/mi_implementation_beamer.pdf), or [Overleaf-ready ZIP](output/mi_implementation_latex.zip). It retains all 17 slides and hidden speaker notes. Upload the ZIP to Overleaf and compile `mi_implementation.tex` with pdfLaTeX; see the [LaTeX instructions](output/latex/README.md) for local compilation and notes options.

Rebuild the deck on macOS with Keynote installed and the local PTB signal files available:

```bash
.venv/bin/python scripts/build_implementation_presentation.py prepare
osascript tmp/presentations/implementation/build.applescript
.venv/bin/python scripts/build_implementation_presentation.py finalize
```

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

## VCG-first relational tensors

Measured XYZ VCG remains the primary input. The new [relational tensor study](docs/relational_vcg_plan.md) compares paper-style short-beat statistics, longer-context wavelet statistics, and correlation/covariance tensors with shared training-fold-only channel projections. ECG + VCG is a separate planned extension, not an input to this study.

```bash
.venv/bin/mi-localization relational-features
.venv/bin/mi-localization relational-study
.venv/bin/python -m mi_localization.relational_report
```

The same six classes and frozen patient folds are preserved. Rank selection uses inner patient validation; projection/scaling parameters are refitted inside every training split. New aggregate metrics and tuning audits go to `results/relational_vcg_v1/complete/`, with local resumable fold checkpoints. The previously inspected PTB folds remain a development benchmark, not untouched confirmation. See [`configs/relational_vcg.yaml`](configs/relational_vcg.yaml) for the fixed classifier and approximate beat-region definitions.

After all five arms finish, the report command validates paired patient predictions and produces `REPORT.md` and class-stratified paired bootstrap intervals. It refuses incomplete or mismatched studies. Intervals are conditional on fitted predictions and do not correct repeated development, shared training folds or multiple comparisons.

If only correlation/covariance cache arrays are missing, `mi-localization relational-repair` rebuilds them without discarding fold checkpoints. It verifies exact regenerated context statistics across every record, plus cohort order and region/shrinkage settings, before publishing restored matrices. A mismatch requires a new bank/study rather than a silent resume.

## Compact measured-VCG ResNet

The [neural development plan](docs/neural_vcg_plan.md) adds a 78k-parameter XYZ ResNet and a separate ResNet + 192-statistics fusion ablation. Architecture/optimizer settings are fixed in [`configs/neural_vcg.yaml`](configs/neural_vcg.yaml). Inner patient folds select the epoch budget; normalization and early stopping never use outer test patients.

```bash
.venv/bin/python -m pip install -e '.[neural,dev]'
.venv/bin/mi-localization neural-features
.venv/bin/mi-localization neural-study
.venv/bin/python -m mi_localization.neural_report
```

All 164 patients and original beat positions are retained. Class/patient-balanced sampling draws 8,192 training beats per epoch; all validation/test beats are evaluated. Models are compared with the fixed depth-3 statistics control on pooled patient predictions. Partial runs are clearly separated, and inner/outer checkpoints support resuming. The report is produced only after the full run finishes; no improvement or SOTA result is claimed in advance. Existing PTB-XL results are untouched, and ECG + VCG remains a future extension.

## Patient-level learned multi-scale tensors

The [next-phase plan](docs/patient_tensor_plan.md) separates repeatability from architecture changes. The ResNet repeat study uses three training seeds with fixed outer **and inner** patient partitions. The patient-tensor study compares mean embedding pooling, gated attention, learned scales with fixed compression, and learned scales with supervised Tucker mode factors. The raw waveform/CNN path stays present; padding and beat permutations cannot change patient predictions.

```bash
.venv/bin/python -m mi_localization.neural_repeat
.venv/bin/python -m mi_localization.patient_tensor
.venv/bin/python -m mi_localization.patient_tensor_report
```

The modern tensor has beats × XYZ × time × scales axes and is built on demand, preserving the raw view. Filters/factors are learned only within training patients. Inference uses all eligible beats, not a random test subset. New configurations are `configs/neural_repeat.yaml` and `configs/patient_tensor.yaml`; outputs are separate from completed studies. Scores, improvement, novelty and SOTA are not claimed before evaluation, and these repeatedly inspected PTB folds remain development data.

## Matched cross-patient contrastive learning

The [contrastive plan](docs/contrastive_vcg_plan.md) compares a newly trained CE control with CE + cross-patient supervised contrastive loss. Both use the same plain ResNet, initial weights, structured patient-balanced batches and maximum epoch budget. Same-patient pairs are excluded entirely from the contrastive comparisons. Three training seeds share frozen inner/outer patient partitions. There is no pretraining, extra head or ECG input.

```bash
.venv/bin/python -m mi_localization.contrastive
.venv/bin/python -m mi_localization.contrastive_report
# Read-only progress; does not launch training:
.venv/bin/python -m mi_localization.contrastive --status
```

Configuration: `configs/contrastive_vcg.yaml`. New output: `results/neural_contrastive_v1/complete/`. The batch policy differs from the old ResNet study, so benefit is measured against the new matched control, not assumed from the historical 0.550 mean. Final reporting validates matching sample/initialization hashes and all seeds, with patient-paired uncertainty. Saved epochs and checkpoints support resuming; partial runs remain clearly separated. No improvement, novelty or SOTA is claimed before full evaluation.

The first matched study completed at 0.479 mean macro F1 for CE and 0.516 for cross-patient SupCon, with an inconclusive paired interval. The [patient-diversity follow-up](docs/contrastive_diversity_followup.md) changes only batch composition: 8 patients/class × 2 beats instead of 2 × 8, keeping 96 beats and the same maximum budget in both arms. It uses separate outputs and remains exploratory development on the same patients.

```bash
.venv/bin/python -m mi_localization.contrastive --config configs/contrastive_vcg_diverse.yaml
.venv/bin/python -m mi_localization.contrastive_report --config configs/contrastive_vcg_diverse.yaml
.venv/bin/python -m mi_localization.contrastive --config configs/contrastive_vcg_diverse.yaml --status
```

## Raw versus denoised measured VCG

The [preprocessing study](docs/preprocessing_vcg_plan.md) audits the existing 164-patient, 391-record, 55,863-beat cohort, then compares fresh raw/denoised ResNet fits over three seeds. All original beat positions and frozen patient folds are retained. Both arms have matching initial weights and shared-epoch sampled rows, with separate training-only scaling and inner-patient stopping.

```bash
.venv/bin/python -m mi_localization.signal_audit
.venv/bin/python -m mi_localization.preprocessing_study
.venv/bin/python -m mi_localization.preprocessing_report
# Read-only progress:
.venv/bin/python -m mi_localization.preprocessing_study --status
```

The completed [signal audit](results/neural_preprocessing_v1/audit/REPORT.md) verifies header/cache consistency and exact reproduction of all denoised crops. Interval and signal-change diagnostics are screening findings; they do not establish detector accuracy or denoising harm. No data were removed or relabeled. Raw review plots remain local. Configuration is in `configs/preprocessing_vcg.yaml`; new study results go to `results/neural_preprocessing_v1/complete/`. The final report requires all folds/seeds and validates paired predictions and training audits before describing any gain. These are development comparisons on repeatedly inspected PTB patients, with no promised target score or SOTA claim.

## Longer-context denoised VCG

The [longer-context experiment](docs/context_vcg_plan.md) compares the original −250/+400 ms crop (651 samples) with a fixed −350/+650 ms crop (1,001 samples). Both arms retain denoised measured XYZ, all original beat positions and frozen patient folds, and freshly train the same ResNet over three seeds. No time resizing or additional model parameters are introduced. Recording-boundary gaps use declared reflection padding; the original central samples must reproduce the short cache exactly.

```bash
.venv/bin/python -m mi_localization.context_bank
.venv/bin/python -m mi_localization.context_study
.venv/bin/python -m mi_localization.context_report
# Read-only progress:
.venv/bin/python -m mi_localization.context_study --status
```

Configuration: `configs/context_vcg.yaml`. New aggregate cache audits and experiment outputs: `results/neural_context_v1/`. The comparison preserves matched initializations/sampled rows, separate training-only normalization and inner-patient stopping. Longer inputs require more computation and change pooling coordinates; they are not adjudicated cardiac phases. The final report validates every seed/fold before assessing benefit, and no improvement is claimed in advance.

## Bounded regularization and signal augmentation

The [regularization study](docs/regularization_vcg_plan.md) retains the original 651-sample denoised XYZ input and ResNet. It compares a fresh fixed baseline with an inner-patient-selected procedure choosing among exactly four presets: baseline, stronger regularization, mild common-XYZ gain/noise augmentation, and their combination. Selection, normalization and stopping use training/inner-validation patients only. All original patients/beats and frozen folds remain, across three seeds.

```bash
.venv/bin/python -m mi_localization.regularization
.venv/bin/python -m mi_localization.regularization_report
# Read-only progress:
.venv/bin/python -m mi_localization.regularization --status
```

Configuration: `configs/regularization_vcg.yaml`. New outputs: `results/neural_regularization_v1/complete/`. Augmentation uses an independent random stream and is never applied to validation/test inputs. Every candidate has the same maximum epoch/sample budget; the selected procedure uses extra tuning compute. The report validates paired predictions, all candidate histories, selection, sampling and exact baseline fallback before describing any gain. Augmentation strengths are experimental choices; no target score or clinical invariance is assumed.

## Persistent-error review and patient learning curves

The [diagnostic phase](docs/diagnostics_learning_curve_plan.md) audits persistent out-of-fold errors and tests training-patient sensitivity independently. The completed [error audit](results/neural_diagnostics_v1/REPORT.md) supplies aggregate findings plus a local review packet for all 35 persistent selected-model cases. No labels, patients, beats or peaks are changed; source/signal flags do not establish error causes or clinical truth.

```bash
.venv/bin/python -m mi_localization.error_audit
.venv/bin/python -m mi_localization.learning_curve --plan
.venv/bin/python -m mi_localization.learning_curve
.venv/bin/python -m mi_localization.learning_curve_report
# Read-only progress:
.venv/bin/python -m mi_localization.learning_curve --status
```

The learning curve freshly trains the original 651-sample denoised ResNet using 25%, 50%, 75% and 100% class-stratified patient subsets across three seeds and the same ten outer test folds. All beats of selected patients are retained; subsets are nested and selected without model errors. Inner normalization/stopping stays within each subset. The report separates training resubstitution from pooled held-out patient F1 and does not extrapolate a target score or dataset ceiling. Configurations: `configs/diagnostics_vcg.yaml` and `configs/learning_curve_vcg.yaml`; new curve outputs: `results/neural_learning_curve_v1/complete/`.

## Audited source-only pretraining pilot

The [pretraining phase](docs/source_pretraining_plan.md) freezes 1,024 PTB-XL training patients from folds 1–8 and 128 source validation patients from fold 9, one recording per patient without diagnostic-label filtering. Fold 10 is excluded before waveform I/O. The new source bank verifies units/leads/rates, converts ECG to Kors XYZ, records QC exclusions and exact source duplicate quarantine, and retains one original-size crop per source patient.

```bash
.venv/bin/python -m mi_localization.pretraining_source
.venv/bin/python -m mi_localization.pretraining_source --prepare
.venv/bin/python -m mi_localization.masked_pretraining
# Read-only status:
.venv/bin/python -m mi_localization.masked_pretraining --status
```

The bounded masked-reconstruction pilot reuses the compact VCG encoder with a temporary decoder, train-only source normalization, fixed source-validation masks and resumable RNG/optimizer checkpoints. Mean-template and zero-input controls accompany reconstruction MSE. Configuration: `configs/source_pretraining_vcg.yaml`; new reports: `results/neural_source_pretraining_v1/`. PTB-versus-PTB-XL identity overlap is unresolved, and the export explicitly remains uncleared for a clean target patient-independent claim. No PTB localization evaluation or automatic transfer run is included in this source-only phase; measured VCG remains the primary target input.

## Exploratory pretrained-versus-scratch VCG comparison

The [transfer phase](docs/transfer_vcg_plan.md) follows the completed source-only pilot with a bounded recording-copy screen and a matched target comparison. The screen checked 9,216 retrieved source/target pairs, detected 20/20 synthetic copied-segment controls and flagged no matches. This does **not** clear cross-dataset patient identity: transfer remains exploratory on repeatedly inspected PTB folds.

```bash
.venv/bin/python -m mi_localization.overlap_screen
.venv/bin/python -m mi_localization.transfer
.venv/bin/python -m mi_localization.transfer_report
# Read-only saved progress:
.venv/bin/python -m mi_localization.transfer --status
```

Both arms use the same measured XYZ bank, patient folds, architecture, fresh classifier initialization, training-only target normalization, weighted sampling and two inner-selected learning rates. Only the pretrained arm imports the fixed source encoder; its source decoder/normalization/optimizer are discarded. All **420 target fits completed** across three target seeds and ten outer folds. The [validated final comparison](results/neural_transfer_v1/complete/REPORT.md) gives **0.553 ± 0.052** patient macro F1 from scratch versus **0.520 ± 0.007** with pretraining (mean ± sample SD). Pretrained minus scratch is **−0.033**, with a descriptive paired patient-bootstrap interval **[−0.091, +0.023]**. This pilot does not demonstrate a transfer gain; the interval also does not establish a definitive harmful effect.

Configurations: `configs/source_target_overlap.yaml` and `configs/transfer_vcg.yaml`; outputs: `results/neural_transfer_v1/`. See the [recording-screen report](results/neural_transfer_v1/overlap_audit/REPORT.md) for its limited coverage and unresolved identity caveat. Source reconstruction improvement did not translate into higher mean localization F1 in this tested setup. Scratch training remains the reference procedure; these results do not justify adopting this source encoder as an improvement.

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
