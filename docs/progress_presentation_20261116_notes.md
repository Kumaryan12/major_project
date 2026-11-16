# Implementation progress presentation

Formal black-and-white update to the 7 October presentation. Slides 1–22 form the main talk (about 15 minutes). Slides 23–33 are technical backup. Speaker notes are also embedded in the editable PowerPoint.

The transfer study was resumed on 16 November 2026. Its counters change during training; slide 20 deliberately gives the fixed study scope and no partial score. Check the live monitor before presenting.

## 1. Implementation progress

My friend has introduced the research paper and the problem. I will explain what we implemented after the previous presentation, what the completed experiments show, and what is running now. The primary task remains six-class localization from measured Frank XYZ VCG.

Evidence:

- [docs/CODEX_HANDOFF.md](../docs/CODEX_HANDOFF.md)
- [docs/implementation_presentation_notes.md](../docs/implementation_presentation_notes.md)

## 2. Progress since the previous presentation

In October the strongest nested feature result was 0.487. The later ResNet was trained with all three declared seeds, giving a mean of 0.550. A separate post-hoc uniform ensemble later reached 0.584570. The mean and ensemble are different predictors and should not be treated as one uninterrupted improvement curve. The approximate 0.063 increase from the old feature result to the ResNet mean is historical development evidence, not fresh-cohort confirmation.

Evidence:

- [docs/CODEX_HANDOFF.md](../docs/CODEX_HANDOFF.md)
- [results/neural_repeat_v1/REPORT.md](../results/neural_repeat_v1/REPORT.md)
- [results/neural_residual_seed_ensemble_v1/complete/REPORT.md](../results/neural_residual_seed_ensemble_v1/complete/REPORT.md)

## 3. Unchanged patient-level evaluation

The unit of independence is a patient. More beats or overlapping windows do not add independent patients. Each original beat has 651 samples at 1000 Hz, from 250 ms before R through 400 ms after R. Aggregation is declared separately for each study. We retain every target patient, original label, cached R position and frozen fold. Micro F1 equals accuracy for this single-label problem and remains secondary.

Evidence:

- [AGENTS.md](../AGENTS.md)
- [docs/CODEX_HANDOFF.md](../docs/CODEX_HANDOFF.md)
- [configs/patient_folds.csv](../configs/patient_folds.csv)

## 4. Implemented pipeline after October

The major implementation change is that the project is now a set of controlled, reproducible studies rather than one tensor recipe. Some methods classify beats and average probabilities. Others first build recording features and aggregate records. We compare each method with its matched control rather than attributing every difference to the tensor alone.

Evidence:

- [docs/CODEX_HANDOFF.md](../docs/CODEX_HANDOFF.md)

## 5. Waveform ResNet and repeatability

We implemented a compact 1D ResNet that learns waveform features directly. The seed-42 plain model scored 0.586 and its statistics-fusion variant 0.584. We did not stop at that favorable seed. Seeds 123 and 2026 scored 0.557 and 0.507, giving 0.550 mean with sample standard deviation 0.040. This slide reports individual models and their mean, not a probability ensemble.

Evidence:

- [results/neural_vcg_v1/complete/REPORT.md](../results/neural_vcg_v1/complete/REPORT.md)
- [results/neural_repeat_v1/REPORT.md](../results/neural_repeat_v1/REPORT.md)

## 6. Carefully constructed multiview tensors

We separated the shared recording shape from beat-to-beat residuals and denoising residuals. These are transformations of the same measured signal, not new sensors or extra patients. We checked their numerical reconstruction identities. We then extracted complementary morphology, scale and relation features. The feature-first route completed. The older full multiview neural sweep remains paused and is not a completed result.

Evidence:

- [docs/record_tensor_quality_plan.md](../docs/record_tensor_quality_plan.md)
- [docs/multiview_vcg_plan.md](../docs/multiview_vcg_plan.md)
- [docs/tensor_feature_design_plan.md](../docs/tensor_feature_design_plan.md)
- [results/tensor_refinement_v1/complete/REPORT.md](../results/tensor_refinement_v1/complete/REPORT.md)

## 7. Feature stability and tabular learning

A larger feature bank was not sufficient. The refinement study gave mixed results: expanded linear 0.425 and tree 0.452, versus legacy linear 0.498 and tree 0.420. Bootstrap consensus then improved the matched tree score to 0.513. A local TabPFN-v2 procedure scored 0.542, but its difference from bootstrap XGBoost was uncertain. These are feature procedures and must not be conflated with the neural seed mean or its ensemble.

Evidence:

- [results/tensor_refinement_v1/complete/REPORT.md](../results/tensor_refinement_v1/complete/REPORT.md)
- [results/feature_stability_v1/complete/REPORT.md](../results/feature_stability_v1/complete/REPORT.md)
- [results/tabpfn_v2_v1/complete/REPORT.md](../results/tabpfn_v2_v1/complete/REPORT.md)

## 8. Relations, geometry and ordered paths

The sepsis-paper idea became correlation and covariance tensors with shared training-only bases. Geometry described signed VCG loops and approximate phase features. Healthy-reference tensors described deviation from training references. Path signatures encoded within-beat ordered paths. Their controls help test whether the added structure is useful, and the results did not establish a consistent gain. The adaptive phase detector accepted only 23.2% of beats and its acceptance differed by class, so it cannot be called validated clinical delineation.

Evidence:

- [results/relational_vcg_v1/complete/REPORT.md](../results/relational_vcg_v1/complete/REPORT.md)
- [results/neural_geometry_v1/complete/REPORT.md](../results/neural_geometry_v1/complete/REPORT.md)
- [results/deviation_vcg_v1/complete/REPORT.md](../results/deviation_vcg_v1/complete/REPORT.md)
- [results/signature_vcg_v1/complete/REPORT.md](../results/signature_vcg_v1/complete/REPORT.md)

## 9. Consecutive beats and RR timing

The hypothesis was that consistency and timing across beats distinguish infarct subtypes. We used a shared beat encoder and a recurrent context model. The matched shuffle destroys ordering while preserving beat/RR pairs. We evaluated 4, 8 and 16 beats with a comparable nominal beat budget and symmetric inner length selection. The ordered-versus-shuffled difference was minus 0.008 with an interval from minus 0.057 to plus 0.039. This did not demonstrate an ordering benefit. All 1200 fitted contexts and 120 outer models were verified.

Evidence:

- [results/neural_chronological_lengths_v1/complete/REPORT.md](../results/neural_chronological_lengths_v1/complete/REPORT.md)
- [docs/chronological_lengths_plan.md](../docs/chronological_lengths_plan.md)

## 10. Direct morphology with residual context

The chronology diagnostics suggested that context training might sacrifice direct beat morphology. We therefore retained a direct beat path and added context as a residual correction. We tested ordered and shuffled context with and without an auxiliary beat loss. The original trained-model result was 0.539 for ordered plus auxiliary compared with 0.551 for beat-only. All six arms, three seeds, 720 fits, 180 outer restorations and 120 matched checks are retained. The full arm table is in backup.

Evidence:

- [results/neural_residual_context_v1/complete/REPORT.md](../results/neural_residual_context_v1/complete/REPORT.md)
- [docs/chronological_diagnostics_findings.md](../docs/chronological_diagnostics_findings.md)

## 11. Uniform seed ensembles

An ensemble is one predictor obtained by averaging model probabilities. Its F1 is not the arithmetic mean of member F1 scores. We retained all six arms and all three seeds, restored every outer member and replayed the source checks. Ordered plus auxiliary scored 0.584570, versus 0.545112 for the matched beat-only ensemble. The paired difference interval includes zero. Because we examined ensembling after seeing the source results, it is explicitly post-hoc development and does not rewrite the earlier mean-model conclusion.

Evidence:

- [results/neural_residual_seed_ensemble_v1/complete/REPORT.md](../results/neural_residual_seed_ensemble_v1/complete/REPORT.md)
- [docs/residual_seed_ensemble_plan.md](../docs/residual_seed_ensemble_plan.md)

## 12. Fusion and staged context learning

We tried to combine complementary neural and tensor predictions with a simple inner-selected scalar. It scored 0.583052 against 0.584570 for neural alone. We also tested continued morphology, joint context, frozen context and shuffled context in a separate staged study. Frozen context scored 0.563539 against matched morphology 0.545112, but its interval included zero and the shuffled control scored 0.565056. Inner validation chose zero adaptation updates in 82 of 120 outer adaptations. These results do not support an ordering benefit or a larger adaptive gate.

Evidence:

- [results/neural_tensor_fusion_v1/complete/REPORT.md](../results/neural_tensor_fusion_v1/complete/REPORT.md)
- [results/neural_staged_context_v1/complete/REPORT.md](../results/neural_staged_context_v1/complete/REPORT.md)

## 13. Random-convolution features with ridge

This route extracts many fixed convolutional responses rather than training a deep encoder. Training-only scale and threshold estimation remain inside each inner or outer partition. The combined three-seed ensemble scored 0.584266. Its difference from DCT was plus 0.138485 with a positive descriptive interval. Its gain over MultiRocket alone remained uncertain. It is essentially level with, and slightly below, the separate historical neural ensemble 0.584570. The bounded implementation differs from upstream defaults and the HYDRA multivariate code remains experimental with GPL attribution.

Evidence:

- [results/rocket_vcg_v1/complete/REPORT.md](../results/rocket_vcg_v1/complete/REPORT.md)
- [docs/rocket_vcg_plan.md](../docs/rocket_vcg_plan.md)

## 14. Selected convolution features and TabPFN

The next bounded study asked whether a compact set of stable convolution features would let PFN help. We retained the full ridge reference and used nested budget selection for both classifiers. Neither selected procedure improved it. AMI correctly classified patients fell from 5 of 17 to 1 of 17. Fixed-budget outcomes are retained in backup, but cannot replace the prespecified selector after viewing outer scores. All 660 new classifiers and the reused source features were restored or reconstructed.

Evidence:

- [results/rocket_tabpfn_v1/complete/REPORT.md](../results/rocket_tabpfn_v1/complete/REPORT.md)
- [docs/rocket_tabpfn_plan.md](../docs/rocket_tabpfn_plan.md)

## 15. What the data diagnostics showed

We audited errors and training behavior before changing labels or adding complexity. None of the persistent-error observations is expert label adjudication. The chronology review replayed all 1200 corresponding waveform normalizations and found agreement. Equal-record aggregation left the historical ResNet mean close to 0.550, and no selected-length model stopped at the 25-epoch cap. Those checks rule out specific easy explanations but do not establish why context failed. The learning curve is descriptive and cannot predict a guaranteed sample size for 0.80.

Evidence:

- [results/neural_diagnostics_v1/REPORT.md](../results/neural_diagnostics_v1/REPORT.md)
- [results/neural_learning_curve_v1/complete/REPORT.md](../results/neural_learning_curve_v1/complete/REPORT.md)
- [docs/chronological_diagnostics_findings.md](../docs/chronological_diagnostics_findings.md)

## 16. Expanded supervised source cohort

The native audit examined official training folds 1 through 8 and validation fold 9. The bounded expansion retained 5063 training visits from 4096 patients, with the entire fold-9 validation set of 2183 records from 1942 patients. The source input is Kors-derived XYZ from twelve-lead ECG, not measured Frank XYZ. Every record uses up to four uniformly spaced complete crops, with masks for padding. Source training is additional information and compute, so it requires explicit provenance and a new matched target comparison.

Evidence:

- [results/supervised_source_audit_v1/REPORT.md](../results/supervised_source_audit_v1/REPORT.md)
- [results/supervised_source_pilot_v1/REPORT.md](../results/supervised_source_pilot_v1/REPORT.md)

## 17. Native source annotation objectives

We preserve overlapping native diagnoses instead of forcing PTB-XL into our exclusive six target classes. An unlisted code is not adjudicated absence of disease. Likelihood values are not calibrated soft probabilities. Source NORM is not automatically clinical HC, and broad source AMI cannot be silently mapped to the target exclusive AMI class. Both objectives share common validation masks and train-only normalization, sampling and positive weights. Validation macro average precision chooses the checkpoint.

Evidence:

- [results/supervised_source_pilot_v1/REPORT.md](../results/supervised_source_pilot_v1/REPORT.md)
- [docs/supervised_source_pilot_plan.md](../docs/supervised_source_pilot_plan.md)

## 18. Completed supervised source training

The primary source model reached macro AP 0.317456 compared with a training-prevalence constant baseline of 0.037470. Its AUROC was 0.908703 and fixed-threshold F1 was 0.290552. The intermediate-positive sensitivity had nearly identical AP. We retain both rather than selecting an objective from this tiny difference. These are patient-weighted record-level multilabel source metrics on model-selection data. They are not target six-class macro F1, and good source learning does not prove transfer.

Evidence:

- [results/supervised_source_pilot_v1/REPORT.md](../results/supervised_source_pilot_v1/REPORT.md)

## 19. Expanded recording-overlap screen

We screened source training and source model-selection validation, not only a convenient subset. Encoded-file matching found zero copies. Template retrieval selected top-eight candidates and the continuous screen checked 57968 pairs out of 2833186 possible. All 20 known-copy controls passed, with zero high-correlation candidates and zero insufficient-active-lead pairs. This checks a declared class of recording copies. Different visits by the same person, short or transformed copies and retrieval misses remain possible, so this cannot be called patient-identity clearance.

Evidence:

- [results/supervised_source_overlap_v1/REPORT.md](../results/supervised_source_overlap_v1/REPORT.md)
- [docs/supervised_source_overlap_plan.md](../docs/supervised_source_overlap_plan.md)

## 20. Current matched supervised-transfer study

The current question is whether supervised source learning improves initialization for measured-VCG localization. Both source objectives enter the comparison regardless of their small source-AP difference. We discard source heads, source normalization and optimizer state, fit target normalization on the exact training partition, and fine-tune the same architecture. All arms share target sampling and head initialization for matched contexts. The primary result will be certain minus scratch for an equal ensemble of all three target seeds. The interrupted MPS run resumed from compatible saved fits; only an unfinished fit repeats. The runner verifies the final study, publishes aggregate results and stops without launching another experiment.

Evidence:

- [docs/supervised_transfer_plan.md](../docs/supervised_transfer_plan.md)
- [configs/supervised_transfer_v1.yaml](../configs/supervised_transfer_v1.yaml)

## 21. Reproducibility and project status

We expanded the codebase with numerical and integrity checks rather than just reporting better scores. The 289-test suite includes transfer provenance, matching, training-only normalization and selection, tamper checks and small-study restoration. Full real-data studies also have their own independent replay counts, listed in backup. We keep only aggregate metrics and hashed audits public. Signals, fitted weights, patient predictions and source contexts stay local and ignored. A clone does not control another machine’s processes. No following training experiment is queued.

Evidence:

- [AGENTS.md](../AGENTS.md)
- [docs/CODEX_HANDOFF.md](../docs/CODEX_HANDOFF.md)

## 22. Current conclusions and next decisions

The project has progressed substantially in implementation and evidence, but the 0.80 to 0.90 target is still unachieved. A practical next hypothesis is to preserve the distribution of all beat shapes rather than only a recording median or a mean probability. Proposed controls are encoding the median waveform, averaging individual-beat features, pooling moments, and pooling projected quantiles with a comparable feature budget. Paired acquisition alignment is a separate ECG-assisted proposal. Synthetic physiology needs a source-label and anatomy audit first. Within-beat dynamics is another bounded proposal. None is implemented, queued or guaranteed to help. The present transfer study finishes first.

Evidence:

- [docs/CODEX_HANDOFF.md](../docs/CODEX_HANDOFF.md)
- [docs/research_frontiers_20261116.md](../docs/research_frontiers_20261116.md)

## 23. Backup: patient bags and contrastive learning

Patient-bag learning and learned Tucker factors were trained end to end with patient labels. The bag can include separate recordings and has no claimed temporal order. Attention, learned scales and compression did not outperform the single-seed waveform reference. The original contrastive comparison improved its altered-sampler CE control by 0.037 on average, but the interval included zero. A more patient-diverse follow-up reversed the point difference. This does not establish robust superiority.

Evidence:

- [results/neural_patient_tensor_v1/complete/REPORT.md](../results/neural_patient_tensor_v1/complete/REPORT.md)
- [results/neural_contrastive_v1/complete/REPORT.md](../results/neural_contrastive_v1/complete/REPORT.md)
- [results/neural_contrastive_diverse_v1/complete/REPORT.md](../results/neural_contrastive_diverse_v1/complete/REPORT.md)

## 24. Backup: waveform follow-ups

The preprocessing study tested raw against the established denoised input. The longer-crop study tested more context around R. Bounded regularization selected among four fixed presets using inner patients and shared baseline fits where appropriate. Territory supervision derived labels from the existing class names and therefore supplied no independent anatomy annotation. The error audit and learning curve are diagnostics rather than new clinical ground truth.

Evidence:

- [results/neural_preprocessing_v1/complete/REPORT.md](../results/neural_preprocessing_v1/complete/REPORT.md)
- [results/neural_context_v1/complete/REPORT.md](../results/neural_context_v1/complete/REPORT.md)
- [results/neural_regularization_v1/complete/REPORT.md](../results/neural_regularization_v1/complete/REPORT.md)
- [results/neural_territory_v1/complete/REPORT.md](../results/neural_territory_v1/complete/REPORT.md)
- [results/neural_diagnostics_v1/REPORT.md](../results/neural_diagnostics_v1/REPORT.md)
- [results/neural_learning_curve_v1/complete/REPORT.md](../results/neural_learning_curve_v1/complete/REPORT.md)

## 25. Backup: correlation and covariance tensors

All arms used the same fixed depth-3 XGBoost and training weights. Statistics retain amplitude information. Covariance also retains amplitude relative to training-only channel scales, whereas correlation omits absolute scale. Shared covariance gave 0.470 compared with 0.469 for short-beat statistics, with a paired interval including zero. The approximate regions are not expert QRS, ST or T delineations.

Evidence:

- [results/relational_vcg_v1/complete/REPORT.md](../results/relational_vcg_v1/complete/REPORT.md)
- https://doi.org/10.1016/j.bbe.2021.06.009

## 26. Backup: geometry and reference limitations

Geometry can provide physically interpretable descriptors without proving a clinical delineator. The adaptive linear result 0.467 versus statistics 0.445 had an interval from minus 0.061 to plus 0.107. The tree statistics-plus-geometry result 0.461 versus 0.409 had an interval from minus 0.024 to plus 0.131. Healthy deviation scored 0.479 versus matched morphology 0.499. Signature3 scored 0.508 versus morphology 0.515. We therefore retain these as completed ablations with no demonstrated reliable overall gain.

Evidence:

- [results/neural_geometry_v1/complete/REPORT.md](../results/neural_geometry_v1/complete/REPORT.md)
- [results/deviation_vcg_v1/complete/REPORT.md](../results/deviation_vcg_v1/complete/REPORT.md)
- [results/signature_vcg_v1/complete/REPORT.md](../results/signature_vcg_v1/complete/REPORT.md)

## 27. Backup: earlier transfer and ECG extension

The old source-transfer experiment averaged three target-seed scores: pretrained 0.520 versus matched scratch 0.553. That is not the current supervised transfer result. The ECG extension used simultaneous measured twelve-lead ECG in PTB. Its VCG twin control matched added capacity without adding ECG information. Bilinear fusion scored 0.534 versus additive 0.536 and VCG 0.550, all seed means. The new primary project input remains measured XYZ.

Evidence:

- [results/neural_transfer_v1/complete/REPORT.md](../results/neural_transfer_v1/complete/REPORT.md)
- [results/neural_paired_ecg_v1/complete/REPORT.md](../results/neural_paired_ecg_v1/complete/REPORT.md)

## 28. Backup: feature refinement and budgets

Increasing DCT ranks improved reconstruction but did not give consistent localization gains. The early feature-stability result and the later convolution-feature selection result concern different feature banks. We cannot generalize that selection always helps. In the later study both selected procedures reduced AMI F1, while fixed 32 and 64 feature controls showed that the classifier and budget interact. These controls are evidence for a future hypothesis, not permission to select a new outer-score winner retrospectively.

Evidence:

- [results/tensor_refinement_v1/complete/REPORT.md](../results/tensor_refinement_v1/complete/REPORT.md)
- [results/feature_stability_v1/complete/REPORT.md](../results/feature_stability_v1/complete/REPORT.md)
- [results/tabpfn_v2_v1/complete/REPORT.md](../results/tabpfn_v2_v1/complete/REPORT.md)
- [results/rocket_tabpfn_v1/complete/REPORT.md](../results/rocket_tabpfn_v1/complete/REPORT.md)

## 29. Backup: all residual-context arms

This complete table prevents a favorable ensemble arm from hiding the original study. Mean member F1 averages three scores, whereas ensemble F1 scores the average probability predictor. Ordered context plus auxiliary has the highest ensemble point score within this table but its matched improvement remains uncertain. The full source study verified 720 fits, 180 outer restorations and 120 matched checks. The follow-up independently restored all 180 members again for inference.

Evidence:

- [results/neural_residual_context_v1/complete/REPORT.md](../results/neural_residual_context_v1/complete/REPORT.md)
- [results/neural_residual_seed_ensemble_v1/complete/REPORT.md](../results/neural_residual_seed_ensemble_v1/complete/REPORT.md)

## 30. Backup: all staged-learning arms

Staged learning protected the morphology encoder while asking whether context could improve it. Continued morphology uses more training without context. Joint and frozen branches separate adaptation from context learning. A shuffled branch tests ordering. Frozen versus morphology had a point gain of 0.018428 with an interval including zero, and shuffled scored slightly higher than ordered frozen. All 600 phase checkpoints, 150 outer checkpoints and 120 matched checks were verified.

Evidence:

- [results/neural_staged_context_v1/complete/REPORT.md](../results/neural_staged_context_v1/complete/REPORT.md)

## 31. Backup: class-level reference results

HC is easier than the small MI subtypes in both references. In the neural ordered-plus-aux ensemble AMI F1 is 0.294, while HC is 0.868. The convolution ensemble has AMI 0.345 and HC 0.925. Patient counts are small, especially AMI and ALMI. Macro F1 gives each class equal weight, so improving healthy-control discrimination alone does not solve subtype localization. These outer class outcomes must not choose class-specific routing or relabeling.

Evidence:

- [results/neural_residual_seed_ensemble_v1/complete/REPORT.md](../results/neural_residual_seed_ensemble_v1/complete/REPORT.md)
- [results/rocket_vcg_v1/complete/REPORT.md](../results/rocket_vcg_v1/complete/REPORT.md)

## 32. Backup: completed independent verification

Each completed study has its own manifest and independent report. Restoration means loading saved parameters and reproducing predictions, not merely observing that a filename exists. Scope and numerical hashes keep completed studies immutable. These checks reduce implementation and provenance errors but do not correct repeated development or establish clinical generalization.

Evidence:

- [results/neural_chronological_lengths_v1/complete/REPORT.md](../results/neural_chronological_lengths_v1/complete/REPORT.md)
- [results/neural_residual_context_v1/complete/REPORT.md](../results/neural_residual_context_v1/complete/REPORT.md)
- [results/neural_tensor_fusion_v1/complete/REPORT.md](../results/neural_tensor_fusion_v1/complete/REPORT.md)
- [results/neural_staged_context_v1/complete/REPORT.md](../results/neural_staged_context_v1/complete/REPORT.md)
- [results/rocket_vcg_v1/complete/REPORT.md](../results/rocket_vcg_v1/complete/REPORT.md)
- [results/rocket_tabpfn_v1/complete/REPORT.md](../results/rocket_tabpfn_v1/complete/REPORT.md)
- [results/supervised_source_pilot_v1/REPORT.md](../results/supervised_source_pilot_v1/REPORT.md)
- [results/supervised_source_overlap_v1/REPORT.md](../results/supervised_source_overlap_v1/REPORT.md)

## 33. Backup: selected research references

These references motivate the implemented methods. Our VCG adaptations and bounded budgets differ from upstream settings where disclosed. The sepsis metric, tabular benchmark metrics and time-series benchmark rankings are not our localization scores. Detailed sources for the proposed next methods are in the research shortlist. The repository contains each exact protocol and aggregate report.

Evidence:

- https://doi.org/10.1109/TBME.2022.3202962
- https://doi.org/10.1016/j.bbe.2021.06.009
- https://proceedings.mlr.press/v80/ilse18a.html
- https://arxiv.org/abs/2004.11362
- [docs/rocket_vcg_plan.md](../docs/rocket_vcg_plan.md)
- [docs/tabpfn_v2_plan.md](../docs/tabpfn_v2_plan.md)
- https://physionet.org/content/ptbdb/1.0.0/
- https://physionet.org/content/ptb-xl/1.0.3/
