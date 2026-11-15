# Start here: project handoff for a new collaborator and Codex

Last reviewed: **15 November 2026**, after implementation commit `e81d26e`. This document reconstructs the useful project context without the original conversation. It is a dated handoff, not an automatically updated training monitor. Later validated reports and Git commits take precedence.

## 1. Objective and current decisions

Aryan Kumar and Kunal Maka are developing their major project under Dr Shivnarayan Patidar, NIT Goa: **Patient-Independent Localization of Myocardial Infarction from Vectorcardiographic Signals Using Tensor Decomposition and Machine Learning**.

We first implemented Zhang et al., *Automated Localization of Myocardial Infarction From Vectorcardiographic via Tensor Decomposition* ([IEEE TBME, DOI](https://doi.org/10.1109/TBME.2022.3202962)), then investigated stronger representations and models with patient-disjoint evaluation. The objective is to improve **macro F1 fairly**, with 0.80–0.90 an aspiration, not an achieved or forecast score.

Accepted user decisions:

- Measured Frank XYZ VCG is the primary input. ECG+VCG or ECG-teacher-assisted training must remain a separately labeled extension.
- Build representations carefully, preserve information, and test matched controls instead of equating larger tensors or better reconstruction with better localization.
- Keep macro F1 primary; micro F1/accuracy is secondary.
- Preserve the six classes, patients, labels, cached R positions and frozen patient folds. No score-driven filtering/relabeling or favorable-seed selection.
- Keep progress and reviewable work in GitHub. Avoid overlapping edits/runs between collaborators by assigning branches and experiment namespaces.

Repository: [Kumaryan12/major_project](https://github.com/Kumaryan12/major_project). Older presentation material is useful history, not the latest scorecard.

## 2. Data and evaluation contract

Current six-class PTB cohort: **164 patients, 391 recordings, 55,863 beats**. Class patient counts: AMI 17, ALMI 16, ASMI 27, IMI 29, ILMI 23, HC 52. Original measured XYZ is at 1,000 Hz. A beat has 651 samples: R−250 through R+400 ms. R is at crop index 250.

Ten fixed outer patient folds are stored in [patient_folds.csv](../configs/patient_folds.csv); most later studies use three inner patient folds. The patient, not the beat, is the independent unit. The original neural procedure averages all test-beat probabilities per patient. Recent feature models aggregate within recordings and then across records. Their scores share an endpoint but their procedures differ.

All fitted preprocessing, feature selection, reference construction, stopping, tuning and stacking must be training-only at the appropriate nesting level. Independent restarts are not independent patients. Cross-validation folds have overlapping training sets. These repeatedly studied PTB patients are development data, even with nested selection.

Full-record denoising and recording medians use an examination's own signal: inference is offline/noncausal. Fixed R-relative windows are not independently verified clinical boundaries. Do not assume arbitrary XYZ rotations, unit-amplitude scaling, delineation or signal-quality deletion preserves MI-localization information.

## 3. What was shown in the presentation, and what changed

The implementation presentation was committed **7 October 2026**. Its strongest nested feature result was macro F1 **0.487**. It proposed neural models as future work. See the [speaking notes](implementation_presentation_notes.md).

Completed neural results were published on **12 November**, and the three-seed check on **13 November**. The plain ResNet produced **0.586, 0.557, 0.507**, with mean **0.550 ± 0.040**. The ± value is sample SD across training seeds, not a generalization confidence interval. The mean is not a probability ensemble. This is actual completed training, not the old twelve-representation XGBoost ablation table or a current-run forecast.

Thus there is an approximate historical increase of **0.063 macro F1** since that saved presentation. It is a comparison between different procedures on a development cohort, not a new-cohort confirmation or a fully isolated causal effect.

## 4. Completed evidence and lessons

Use the links for exact controls, populations, seed policies and limitations. Do not present all table rows as a single interchangeable leaderboard.

| Work | Result / decision | Evidence |
|---|---|---|
| Paper implementation | Random-beat 12-class accuracy about 99.14%; grouped-patient evaluation was much lower. Different endpoint/task from six-class patient macro F1. | [Initial results](../RESULTS.md) |
| Feature benchmark and 12 ablations | Six-class Tucker/XGBoost macro F1 0.403; exploratory wavelet-statistics 0.464; nested statistics rerun 0.487. | [Ablations](../ABLATION_RESULTS.md) |
| External PTB-XL test | External macro F1 about 0.242; do not claim external generalization solved. | [External report](../EXTERNAL_RESULTS.md) |
| Paired acquisition-domain experiment | Fixed measured→measured 0.469; measured→derived 0.422; derived→derived 0.467. Domain mismatch is a hypothesis, not a complete explanation. | [Domain report](../DOMAIN_SHIFT_RESULTS.md) |
| Correlation/covariance tensors | Implemented training-only shared representations; no established improvement over the matched feature control. | [Plan](relational_vcg_plan.md) |
| Waveform ResNet and statistics fusion | Seed 42: plain 0.586, fusion 0.584; matched fixed tree 0.469. | [Neural report](../results/neural_vcg_v1/complete/REPORT.md) |
| Neural seed repeatability | Plain ResNet mean 0.550 ± 0.040; all three seeds retained. | [Repeatability](../results/neural_repeat_v1/REPORT.md) |
| Unordered bags, attention and learned Tucker | Tested; no gain over the matched waveform reference. These are not true chronological beat sequences. | [Patient tensor report](../results/neural_patient_tensor_v1/complete/REPORT.md) |
| Cross-patient contrastive objectives | Original and diversity follow-up tested; no consistent superiority over the established waveform reference. | [Original](../results/neural_contrastive_v1/complete/REPORT.md), [follow-up](../results/neural_contrastive_diverse_v1/complete/REPORT.md) |
| Raw versus denoised, longer crops, regularization | Raw mean about 0.521 and longer crops about 0.538 versus historical denoised 0.550; tested recipes did not establish a consistent gain. | [Preprocessing](../results/neural_preprocessing_v1/complete/REPORT.md), [context](../results/neural_context_v1/complete/REPORT.md), [regularization](../results/neural_regularization_v1/complete/REPORT.md) |
| Error review and learning curve | 35 patients wrong across all three selected-model seeds; no expert adjudication. Training-budget curve rose roughly 0.407→0.550. This is not proof of label noise, a ceiling or a required sample count. | [Error audit](../results/neural_diagnostics_v1/REPORT.md), [learning curve](../results/neural_learning_curve_v1/complete/REPORT.md) |
| Source SSL and transfer | Bounded PTB-XL pilot: 1,024 source training patients. Target scratch 0.553 versus pretrained 0.520, three target seeds. Cross-dataset identity remains unresolved; this is not a large-foundation-model failure proof. | [Source plan](source_pretraining_plan.md), [transfer](../results/neural_transfer_v1/complete/REPORT.md) |
| Paired ECG/VCG neural fusion | ECG-only, additive and bilinear extensions tested; no demonstrated gain over XYZ reference. | [Paired report](../results/neural_paired_ecg_v1/complete/REPORT.md) |
| Geometry and adaptive phases | Physical descriptors and phase features tested; strict phase checks passed only about 23.2% of beats, with class-dependent coverage. No reliable overall gain. | [Geometry report](../results/neural_geometry_v1/complete/REPORT.md) |
| Territory auxiliary labels | Class-name-derived territory task: 0.538 versus CE reference 0.550. Not independent anatomical ground truth. | [Territory report](../results/neural_territory_v1/complete/REPORT.md) |
| Recording templates and five views | Constructed raw R, denoised D, record median M, D−M, R−D, with reconstruction checks. Views are redundant transformations, not new sensors. | [Record plan](record_tensor_quality_plan.md), [multiview plan](multiview_vcg_plan.md) |
| Feature-first tensor construction | Beat × five views × XYZ × five bands × time; morphology, multiscale and relation features. First inner-only screen gave no final outer score. | [Feature design](tensor_feature_design_plan.md) |
| Compression refinement | Higher DCT ranks improved fidelity without consistent score gains. Expanded procedure: linear 0.425/tree 0.452; legacy linear 0.498/tree 0.420. | [Refinement report](../results/tensor_refinement_v1/complete/REPORT.md) |
| Bootstrap feature consensus | Tree 0.420→0.513, descriptive paired interval for gain [+0.027,+0.158]; one fixed resampling seed policy. Hard semantic grouping hurt. Seed-robustness confirmation remains outstanding. | [Stability report](../results/feature_stability_v1/complete/REPORT.md) |
| Local TabPFN-v2 | Raw 0.542 versus matched bootstrap-XGB 0.513; difference interval includes zero. Fixed prior correction 0.498. Not above the historical neural mean conclusively. | [PFN report](../results/tabpfn_v2_v1/complete/REPORT.md) |

The model still struggles with MI subtypes, especially AMI/ALMI; HC classification is stronger. Recurring confusions include IMI↔ILMI and AMI/ALMI/ASMI. Do not mine outer labels for new tuning decisions and then call the same folds untouched tests.

## 5. Active, paused and proposed work

**Active at this handoff:** healthy-reference deviation study, launched on Aryan's Mac on 15 November. Code/config/smoke were pushed in `e81d26e`. It has four arms: full morphology, matched three-member morphology ensemble, healthy-reference features, and size-matched mixed-reference features. Reference seeds: 42/123/2026. Planned: **1,960 member fits**, 100 outer arm/seed evaluations, then independent report verification including restoration of 280 outer member contexts.

Each member builds two medoids from a disjoint reference part and trains its classifier on the other two parts. All query patients are separate. Signed five-band deviations from recording templates yield 240 window summaries plus three distance features, appended to 480 morphology features. Two budgets (24/64) are selected inside the inner folds. Healthy-reference construction uses HC labels and is supervised; residuals are not a patient's counterfactual healthy heart.

- [Exact implementation and monitoring](deviation_vcg_implementation.md)
- [Configuration](../configs/deviation_vcg.yaml)
- Core: `deviation_tensor.py`, `deviation_study.py`, `deviation_report.py`
- Local log on the launching Mac: `tmp/deviation_vcg_v1.log`. That file/process does not appear on a fresh clone.
- Final location: `results/deviation_vcg_v1/complete/REPORT.md`, accompanied by a completed validated summary. Until then there is no complete improvement result. A `smoke/summary.json` or `progress.json` is not a performance result.

**Paused:** the earlier six-arm multiview neural sweep. Local partial checkpoints are preserved. Do not call it complete or resume it automatically.

**Proposed, not trained:** time-augmented signed path-signature features; true contiguous within-record beat context against a shuffled-order control; later combination only if component ablations help. Small confusion-specialist branches and properly nested waveform/tabular combinations were discussed, not implemented by the handoff. See [research plan](deviation_tensor_research_plan.md) and [architecture roadmap](modern_architecture_roadmap.md). A recent ECG paper is inspiration, not evidence for a score or originality claim on measured Frank VCG.

## 6. Repository map and missing local state

- `src/mi_localization/`: pipeline, representations, model training and independent reports.
- `configs/`: fixed protocols and patient folds; paths are generally repository-relative.
- `tests/`: synthetic behavior, leakage/integrity, caching and restoration checks.
- `results/`: public aggregate metrics/reports/audits; some ignored files may exist on the originating machine only.
- `docs/`: detailed experiment contracts; `output/`: presentation/report artifacts.
- `data/raw/`, `data/processed/`, `data/models/`, `.venv/`, `tmp/`: ignored local state.

At this handoff the full macOS suite passed **185 tests**. Windows has not been validated. Do not copy a Mac `.venv`, claim byte-identical cross-platform training, or disable source/runtime checks to reuse a cache.

The newest study is **not a standalone one-command cold-start pipeline**. Its loader follows `deviation_study → tabpfn_study → stability_study → tensor_refinement_study → tensor_feature_screen`, with verified source banks and completed prerequisite manifests. A fresh clone lacks feature arrays, patient-level prediction checkpoints and fitted model files. Some compatible local artifacts may be transferable, while fingerprints using paths, file mtimes, package versions, numerical content and platform can require a controlled rebuild/migration.

Inspect these dependencies before launching anything expensive. Rebuilding predecessor studies under new Windows namespaces requires corresponding config references; changing only the final output directory does not fix an incompatible ancestor manifest. Never edit a historical manifest to pretend it matches.

## 7. First task for the new Codex

Read this handoff, root [AGENTS.md](../AGENTS.md) and [Windows setup](WINDOWS_SETUP.md). Run `python scripts/project_status.py` and inspect `git status`. Summarize what is completed, active remotely, and available locally. Check the environment with small tests. Then implement the collaborator's assigned task on a separate branch and output namespace. Do not start or duplicate the full current sweep during orientation.

Use [this ready-to-paste prompt](CODEX_START_PROMPT.txt). Keep the evidence table and active-work section current after genuinely completed results, rather than copying stale presentation numbers into new claims.
