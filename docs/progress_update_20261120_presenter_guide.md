# Presentation update, 20 November 2026

34 main slides and 12 appendix slides. Based on the illustrated 17 November deck. All study results are completed development evidence. The original presentation and numerical study artifacts are unchanged.

The deck has no footer notes or slide-number fragments. Evidence and image attribution are in presenter notes. The appendix preserves detailed matched comparisons.

## 1. Implementation progress

Patient-Independent Localization of Myocardial Infarction

from Vectorcardiographic Signals

Aryan Kumar (23ECE1006) and Kunal Maka (23ECE1017)

Guide: Dr. Shivnarayan Patidar

Department of Electronics and Communication Engineering

National Institute of Technology Goa

Update after the previous presentation

20 November 2026


Presenter notes

This presentation begins where our illustrated 17 November deck ended. It covers subsequent completed experiments, recording-level audits and data acquisition preparation. The primary task remains six-class patient localization using measured Frank XYZ. The original paper and problem statement can be introduced separately by Kunal. Main slides form a complete account, with detailed comparisons in the appendix.


Evidence

- docs/CODEX_HANDOFF.md
- docs/progress_presentation_20261117_illustrated_notes.md

## 2. Where we ended last time

We had implemented waveform learning, tensor alternatives and patient-level evaluation.

Supervised source training had finished. Target transfer was still running.

Our next proposal was to preserve variation across all heartbeats.


Presenter notes

The previous illustrated deck is the baseline for this update, rather than the much older October deck. Its neural ensemble was post-hoc and averaged all three seeds. The random-convolution ensemble was a separate procedure. Both results used the same original target cohort, but the controls and aggregation differed. The source AP from PTB-XL was a different multilabel endpoint.


Evidence

- output/pptx/mi_progress_since_october_2026_illustrated.pptx
- results/neural_residual_seed_ensemble_v1/complete/REPORT.md
- results/rocket_vcg_v1/complete/REPORT.md

## 3. Current results in context

The strongest all-record development procedures now score approximately 0.60.

These are historical procedures with different training recipes.

Repeated PTB development still requires independent confirmation.


Presenter notes

The selected linear result comes from a shared inner representation selection procedure in the kernel study. The fixed-eight-epoch waveform scratch result comes from the native pretraining pilots and reproduces across several later controls. Their point differences versus the prior presentation are approximately +0.0191 and +0.0172, respectively, but these are not matched causal improvement estimates. The aspiration of 0.80 to 0.90 remains unachieved. Do not present the 145-patient earliest-record endpoint as a lower version of this same metric.


Evidence

- results/distribution_kernel_v1/complete/REPORT.md
- results/neural_vcg_native_pretraining_v1/complete/REPORT.md
- results/model_error_comparison_v1/REPORT.md

## 4. The target data and evaluation stay fixed

164 patients, 391 recordings and 55,863 measured Frank XYZ beats.

Six classes: AMI, ALMI, ASMI, IMI, ILMI and healthy control.

Patient macro F1 remains the primary endpoint.

Ten frozen outer patient folds. Inner patients choose settings when required.

All records from one patient stay in the same partition.


Presenter notes

Original signed XYZ is sampled at 1,000 Hz. Each crop has 651 samples, from R minus 250 ms to R plus 400 ms. Preserve clinical class names: AMI means anterior MI in this cohort, not a generic acute-MI flag. Counts are 17,16,27,29,23,52 in class order. Training alone fits scaling, reference libraries, selection and stopping. Micro F 1 equals accuracy in this single-label task. Every displayed ensemble retains all three declared seeds unless a source-only metric is explicitly identified.


Evidence

- AGENTS.md
- configs/patient_folds.csv
- docs/CODEX_HANDOFF.md

## 5. Measured XYZ remains the primary input

Recorded and denoised signals retain signed orientation and amplitude.

Detected R peaks anchor the original beat crops.

New representations reuse this audited signal bank.


Presenter notes

This figure is the public PTB patient 001/s 0010_re example already used in the previous formal deck. It shows recorded and denoised measured XYZ, detector peaks and a shaded R-relative crop. It is an illustrative public recording, not a class prototype or evidence of a validated clinical delineator. We retain the original target R positions. Image credit: existing project figure, public PTB database.


Evidence

- https://physionet.org/content/ptbdb/1.0.0/
- output/latex/assets/vcg_processing_grayscale.png

## 6. Preserving variation across heartbeats

A median waveform can hide differences between the individual beats.

We encoded every original beat before pooling.

Matched controls separate average shape, spread and distribution shape.


Presenter notes

The representation is recording by R-relative window by feature projection by quantile. Equal weighting across patients, records and beats defines training statistics. Fixed feature projections operate in feature space and do not rotate physical XYZ. Mean/spread uses a fixed normal-quantile basis as a control, without asserting Gaussian physiology. The first study holds all four stored tensor sizes at 720 columns. Later studies keep the full original convolution base and append 144 columns.


Evidence

- docs/distribution_pooling_plan.md
- docs/convolution_distribution_plan.md

## 7. Distribution pooling on physical features

Empirical quantiles scored above the simpler controls in this study.

Quantiles minus mean/spread: +0.0114, interval [-0.0526, +0.0762].

The comparison did not establish a reliable gain.


Presenter notes

All original beats and record medians were re-encoded. Four arms share signed DCT/physical features, training-only normalization, projections and feature-grid dimensions. Independently reconstructed 120 contexts and restored 840 classifiers, including 120 outer models. These compact physical-feature scores are not matched comparisons against the stronger convolution and waveform references. The primary contrast is quantiles versus moments, not whichever historical reference is lowest.


Evidence

- results/distribution_pooling_v1/complete/REPORT.md
- docs/distribution_pooling_findings.md

## 8. Keeping the full convolution representation

We retained all 3,584 original features and added 144 distribution features.

Quantiles minus spread: -0.0046. No positive quantile benefit.

The spread result is a small, uncertain development observation.


Presenter notes

This follow-up addresses loss of the stronger original representation. All beats were processed using the source transforms for their exact training partition. Independently restored 630 new and 210 source classifiers, with exact baseline probabilities. The separately labeled post-hoc spread minus baseline interval is [-0.016967,+0.037176]. Spread improved the ensemble point but lowered all three member scores, with mean member .538610 versus baseline .572860. It does not establish better individual models or independent superiority.


Evidence

- results/convolution_distribution_v1/complete/REPORT.md
- docs/convolution_distribution_findings.md

## 9. Pooling learned beat embeddings

Frozen ResNet teachers encoded every beat into 64 features.

Mean, spread and quantile summaries augmented the full convolution base.

Learned quantiles did not improve the matched baseline.


Presenter notes

All teachers are plain morphology teachers from their exact training partitions, with no encoder updates. Original inner-validation stopping remains inherited and disclosed. All 630 new and 210 source classifiers were restored, and all 120 teacher feature contexts and 120 teacher query probabilities were replayed. Mean .554973, spread .547867, quantiles .580895 versus baseline .584266. Quantiles minus spread +.033028 [-.011283,+.079062]; quantiles minus baseline -.003371 [-.060716,+.052488]. All nine added-feature member scores fell below their paired baseline.


Evidence

- results/learned_distribution_v1/complete/REPORT.md
- docs/learned_distribution_findings.md

## 10. Linear and nonlinear classifiers

Inner validation chose one shared representation for both classifier families.

A nonlinear kernel did not improve the selected ensemble.

The 0.603374 result comes from the linear selection procedure.


Presenter notes

We compared class-balanced linear kernel ridge and linear-plus-Gaussian kernel ridge on four fixed representations. The representation selection is shared symmetrically, preventing each family from selecting a separate representation for the primary comparison. All eight fixed arms, all three seeds and 2,400 model restorations remain reported. Selected nonlinear minus linear is -.019441 [-.048244,+.005086]. Mean member F 1 rises .562847 to .573104 while ensemble F 1 falls, so those endpoints must remain distinct. Nonlinear search used four candidates versus two linear candidates.


Evidence

- results/distribution_kernel_v1/complete/REPORT.md
- docs/distribution_kernel_findings.md

## 11. Corrections to frozen baseline scores

We kept the original convolution predictions and learned a small correction.

Corrections used convolution summaries or learned spread/quantile features.

Every final ensemble retained the original predictions and 0.584266 score.

Inner validation found no useful correction under the tested settings.


Presenter notes

The score-correction study isolates whether a regularized residual can improve the original decision scores without replacing the base. Four procedures selected correction off or penalty inside training folds. All four ensembles reproduce the original 164 labels. Independently verified 120 contexts, 210 source restorations, 1,440 active inner and 30 active outer correction fits. The result does not prove residual correction can never help. Learned features inherit original inner-validation stopping.


Evidence

- results/distribution_score_correction_v1/complete/REPORT.md
- docs/distribution_score_correction_findings.md

## 12. Direction and timing within a heartbeat

A lag tensor captured symmetric relations and signed directional changes.

Magnitude and scrambled-sign controls tested what direction contributes.

Signed direction did not beat the matched baseline or scrambled control.


Presenter notes

We used four R-relative windows and four positive lags. Symmetric entries are shared. The antisymmetric component receives original signs, absolute magnitude or balanced per-beat scrambled signs. Time reversal preserves symmetric entries and flips directional signs in the integrity tests. All 55,863 beats, 120 contexts, 210 source models and 630 new classifiers were independently replayed. Signed minus magnitude +.013444 [-.027522,+.054415], signed minus scrambled -.015269 [-.066919,+.034030].


Evidence

- results/lag_direction_tensor_v1/complete/REPORT.md
- docs/lag_direction_tensor_findings.md

## 13. Subtype relations and local timing

Subtype graph regularization reached 0.591424 versus ridge 0.584266.

Its gain remained uncertain: +0.0072, interval [-0.0129, +0.0283].

A separate local-timing audit failed its patient-coverage requirement.

We deferred the timing tensor before fitting a classifier.


Presenter notes

The graph uses clinical subtype adjacency with HC isolated, compared with a relabeled graph under the same strength grid. It does not add independent anatomical labels. The local-timing audit compares two disjoint same-record median references per query group. Only 20,8,16 of 164 patients met 10% beat support in the three windows, below the required 33 patients. Shape-only controls produced apparent shifts, so offsets cannot be interpreted as clinical timings. No timing-classifier F 1 was measured.


Evidence

- docs/subtype_graph_findings.md
- docs/local_timing_audit_findings.md

## 14. Supervised PTB-XL transfer completed

Source diagnostic learning initialized the encoder. Target heads stayed fresh.

The primary transfer gain was uncertain: +0.0346, interval [-0.0465, +0.1138].

All 630 fits and 90 outer models passed independent replay.


Presenter notes

This is the study that was active during the previous presentation. Certain-source pretraining is the primary arm; intermediate-positive is sensitivity. Source objectives use one pretraining seed, while target fine-tuning retains seeds 42,123,2026. Input at target inference is measured XYZ only. The historical .584570 and .603374 procedures are not this study's matched scratch control. Recording screening found no retrieved copies, but cross-dataset patient identity remains unresolved. Source AP is not target macro F 1.


Evidence

- results/neural_supervised_transfer_v1/complete/REPORT.md
- docs/supervised_transfer_findings.md

## 15. Native VCG and ECG-assisted learning

124 additional native-XYZ patients supplied unlabeled pretraining signals.

A paired ECG teacher supplied an alternative training signal.

Both pilots retained measured XYZ alone at target inference.

Neither procedure demonstrated an ensemble gain over matched scratch.


Presenter notes

Native pretraining used 124 extra patients,156 records,17573 beats after the stated copy/QC screen. Target-only reconstruction .454298, expanded .518364, scratch .601801. Expanded minus target-only +.064066 [-.003292,+.133664], but expanded minus scratch -.083436 [-.160972,-.007201]. ECG-assisted paired .564293 versus shuffled .529345 and scratch .601801. Pairing advantage +.034948 [-.051853,+.121214] remains uncertain. Each study independently replayed 90 models and 30 matched groups. More source data did not automatically improve the fixed recipe.


Evidence

- docs/vcg_pretraining_pilots_findings.md
- results/neural_vcg_native_pretraining_v1/complete/REPORT.md
- results/neural_vcg_ecg_bridge_v1/complete/REPORT.md

## 16. ACS-ECG artery supervision

We pretrained on native occlusion-MI labels, with or without artery labels.

Artery labels stayed in the source task. They did not relabel PTB subtypes.

Artery supervision added no demonstrated benefit over the matched controls.


Presenter notes

OMI refers to occlusion myocardial infarction in the ACS source. Both source models share inputs, initialization and exposure. Only the encoder transfers. The target uses measured XYZ, fresh six-class heads and fixed eight-epoch training. All 90 target models were restored. Primary OMI+artery minus OMI +.010341 [-.066726,+.085396]. OMI+artery minus scratch -.021071 [-.095589,+.052806]. Cross-dataset patient identity remains unresolved despite recording-copy screening.


Evidence

- results/neural_acs_artery_transfer_v1/complete/REPORT.md
- docs/acs_artery_transfer_plan.md

## 17. Controlled changes from STAFF III

Source ECG pairs captured before and during annotated balloon occlusion.

We compared learning the change with learning the mean of the same pair.

Change minus mean: -0.0155, interval [-0.0834, +0.0517].


Presenter notes

STAFF III source vessel labels describe the vessel under balloon occlusion, rather than the six PTB infarct-location classes. Source ECG uses Mason-Likar limbs and Kors-derived XYZ. The source protocol verifies timing and excludes specified lead-error patients. Change and mean see exactly the same paired windows. All 90 target outer fits replayed. Change -.062211 versus scratch, interval[-.139503,+.015059]. This extension demonstrates a controlled source-learning implementation without an established localization gain.


Evidence

- results/neural_staffiii_pair_transfer_v1/complete/REPORT.md
- docs/staffiii_pair_transfer_plan.md

## 18. Partial regional labels from native PTB

13 outside-target patients contributed positive region mentions.

Unmentioned regions stayed masked. Shuffling preserved tag counts.

True labels did not improve the matched scratch ensemble.


Presenter notes

The source audit reproduced 34 recordings and 4,644 beats from 14 candidate patients. Thirteen patients had a shared observed region tag and supplied 4,215 supervised beats. Header mentions have not received independent anatomical adjudication. True minus shuffled gave +0.006940 with interval [-0.042964,+0.058767]. True minus scratch gave -0.025914 with interval [-0.085288,+0.035006]. The no-self shuffle changed 10 of 13 tag sets while preserving per-update tag counts and source waveforms. All 120 fits and 30 matched groups replayed. All 30 scratch probability blocks matched the earlier pilot exactly.


Evidence

- results/neural_native_partial_labels_v1/complete/REPORT.md
- docs/native_partial_labels_findings.md

## 19. Temporal pooling and subtype classifiers

R-aligned convolution windows tested whether local position helps.

Aligned and shuffled windows scored almost identically.

A hierarchical classifier first estimated MI probability, then its subtype.

Its gain over flat classification remained uncertain.


Presenter notes

The temporal study retained the 3,584-feature base and added 512 columns. Aligned minus whole-beat pooling gave +0.010004 with interval [-0.025707,+0.046669]. Aligned minus shuffled was -0.000221 with interval [-0.041515,+0.039815]. All 630 new fits, 210 source baselines and 120 contexts replayed. In the separate subtype study, hierarchical minus flat classification gave +0.024010 with interval [-0.001207,+0.056060]. The historical selected-linear reference was 0.603374. Both new classifiers used the same representation and matched regularization grids. All 600 procedures replayed. Hierarchy supplied no additional clinical labels.


Evidence

- results/r_relative_convolution_v1/complete/REPORT.md
- results/probabilistic_subtype_v1/complete/REPORT.md

## 20. Patient loss and model averaging

Training on average patient probabilities did not help its grouped-batch control.

Equal averaging across four convolution readouts scored 0.586053.

A fixed convolution-waveform blend scored 0.582209.

Complementary errors did not automatically produce a better average.


Presenter notes

The patient-loss study used the same eight-beat bags in every arm. Beat cross-entropy scored 0.510134, patient cross-entropy 0.469596 and their hybrid 0.460082. This sampler differs from the historical waveform recipe that scored 0.601801. All 90 models and 30 groups replayed. Averaging four convolution readouts gave 0.586053 versus the original selector at 0.603374. In the separate convolution-waveform comparison, both models were correct for 93 patients, convolution alone for 19 and waveform alone for 16. Both missed 36, with the same wrong label for 21. The fixed blend scored 0.582209. No weights or patient routing were learned from these outer predictions.


Evidence

- results/neural_patient_probability_loss_v1/complete/REPORT.md
- results/convolution_readout_average_v1/REPORT.md
- results/model_error_comparison_v1/REPORT.md

## 21. Recording differences became a key question

57 of 93 patients with repeated recordings received different record-level labels.

Most remaining convolution errors exchange one MI subtype for another.

We audited examination choice, beat count, signal quality and recording dates.

These diagnostics identify questions for review, without changing target labels.


Presenter notes

The research audit reproduced convolution macro F 1 of 0.603374 and waveform macro F 1 of 0.601801. It checked 2,400 fit memberships and all 391 original target headers. Of 52 convolution errors, 44 exchange one MI subtype for another. The 57 of 93 statistic describes disagreement between model predictions across recordings, rather than proven inconsistency in clinical diagnoses. Uniform one-record expected-count sensitivity was about 0.5683. The later earliest-date analysis answers a different question. All original labels remain fixed and independent clinical review is pending.


Evidence

- docs/patient_independent_research_audit.md
- docs/path_to_080_macro_f1.md

## 22. One earliest recording changes the endpoint

145 patients had a usable date and a unique earliest recording.

The same frozen model scored 0.586275 using all their beats.

Using only their earliest recording scored 0.515049.

Difference: -0.0712, interval [-0.1470, +0.0036].


Presenter notes

This audit fits no models and replays all 30 held-out prediction blocks. Thirteen patients have an earliest-date tie and six have a missing or invalid date, leaving 145 with a unique earliest recording. The main comparison uses those same 145 patients under both inference policies. The all-record result on all 164 patients remains 0.601801. Earliest archived recording does not necessarily mean first acute clinical presentation. Equal weighting across records left all 164 ensemble class decisions unchanged.


Evidence

- results/index_recording_audit_v1/REPORT.md
- docs/index_recording_audit_plan.md

## 23. Matching the number of beats

79 patients had two usable dated recordings.

Matched sampling compared earliest, later and half from each recording.

Later recordings retained the higher point score at the same beat budget.


Presenter notes

The matched experiment uses 256 draws per patient, with 64 beats for 78 patients and 50 for one patient. Displayed scores are macro F 1 of expected confusion counts. Later minus earliest was +0.095882 with interval [-0.037926,+0.230258]. Half from each minus earliest was +0.084854 with interval [-0.017018,+0.192643]. The wide intervals prevent a firm conclusion. Median archival spacing was four days. Timing, treatment, injury evolution and acquisition could all differ. These 79 repeat-record patients are a separate descriptive subset.


Evidence

- results/matched_recording_beats_v1/REPORT.md
- docs/matched_recording_beats_plan.md

## 24. Signal quality did not explain the gap

Raw decoding reproduced cached beats in all 158 paired recordings.

We checked drift, ADC extremes, R-peak disagreement and RR outliers.

Drift was lower on average in later records, with mixed patient-level changes.

The measured quality proxies showed no clear explanation for improved predictions.

Independent clinical and timing review remains pending.


Presenter notes

The audit uses zero new fits. Lower drift versus improved correctness had Spearman rho -0.022 with interval [-0.235,+0.185]. The R-detector disagreement association was +0.059 with interval [-0.175,+0.290]. No ADC rail hits occurred. Mean later-minus-earliest drift was -0.03120 m V, with 41 patients showing lower and 38 higher drift. These proxies do not cover all acquisition quality or disease evolution. A private blinded packet of 24 matched review pairs is prepared, with its key separate. Independent clinical review has not happened yet.


Evidence

- results/paired_recording_quality_v1/REPORT.md
- docs/paired_recording_quality_plan.md

## 25. Learning earlier signals from later recordings

Cross-fitted teachers supplied later-record targets to earliest-record students.

A same-class shuffled teacher control tested whether correct pairing matters.

The paired gain over supervised training remained negligible and uncertain.


Presenter notes

Teachers hold out both the donor and the recipient in the same-class shuffled comparison. The primary endpoint uses 145 dated patients. Paired minus supervised was +0.002434 with interval [-0.079058,+0.085427]. Paired minus shuffled was +0.021697 with interval [-0.062848,+0.102840]. Every paired student seed scored below its supervised counterpart despite the small increase in ensemble point score. All 30 cross-fitted teachers and 90 students replayed, with 30 matched groups. Earliest-record inference uses no later recording.


Evidence

- results/neural_later_recording_distillation_v1/complete/REPORT.md
- docs/later_recording_distillation_plan.md

## 26. Longer training did not reliably help

We compared 8, 16 and 32 epochs under the earliest-record endpoint.

Inner-selected duration scored 0.509127 versus fixed-eight 0.521335.

The fixed-16 point is descriptive and cannot replace the declared selector.

Training loss kept falling while patient-level gains remained uncertain.


Presenter notes

On the 145-patient endpoint, inner-selected duration minus fixed eight epochs gave -0.012208 with interval [-0.082185,+0.061027]. The fixed-16 result was a secondary descriptive contrast of +0.031422 with interval [-0.039516,+0.100476]. All 120 trajectories, 360 checkpoints and 30 original baseline blocks replayed. Mean training loss fell from 0.358157 at epoch eight to 0.044369 at epoch 32. A separate earliest-record convolution comparison scored 0.478122 for combined convolution, 0.461037 for DCT and 0.521335 for the matched waveform. It replayed 420 ridge fits and 120 contexts.


Evidence

- results/neural_earliest_training_duration_v1/complete/REPORT.md
- results/earliest_record_convolution_v1/complete/REPORT.md

## 27. Testing the frozen ResNet readout

We compared the original head with new patient-level ridge classifiers.

64 features summarize the final embedding. 256 retain temporal bins.

Retaining 256 features did not improve the 64-feature procedure.


Presenter notes

Every readout uses the encoder for its own exact training partition, fixed at epoch eight. Encoder updates are zero. Training patients alone fit scaling and choose ridge alpha from 1 or 100 in the inner folds. On 145 dated patients, original probabilities score 0.521335, mean logits 0.517351, ridge 64 0.535691 and ridge 256 0.501020. The 256 minus 64 contrast is -0.034671 with interval [-0.094647,+0.021065]. The 64-feature gain over the original is secondary and uncertain. All 120 encoders and 420 ridge heads replayed. The 164-patient sensitivity retains one selected record per person.


Evidence

- results/frozen_patient_readout_v1/complete/REPORT.md
- docs/frozen_patient_readout_findings.md

## 28. Ridge, centroid and Mahalanobis distance

All three classifiers used the same frozen 64-dimensional embeddings.

Centroid distance ignores covariance. Mahalanobis uses a regularized covariance.

The covariance-aware head did not improve the matched ridge reference.


Presenter notes

The latest completed study finished on 19 November at 13:15 IST. All 420 new distance heads, 210 source ridge heads, 120 contexts and 104 identity controls replayed. All 1,444 protected source files stayed unchanged and encoder updates were zero. Mahalanobis minus ridge was -0.028510 with interval [-0.088700,+0.031304]. Mahalanobis minus centroid was -0.000589 with interval [-0.041155,+0.038686]. AMI F 1 fell from 0.4286 to 0.2963. Every Mahalanobis member scored below its paired ridge member. The separate 164-patient selected-record sensitivity is in the appendix. No follow-on training is queued.


Evidence

- results/frozen_distance_readout_v1/complete/REPORT.md
- docs/frozen_distance_readout_findings.md

## 29. What the experiments tell us

Richer summaries sometimes raise point scores, with uncertain matched gains.

Changing the classifier alone has not resolved the difficult MI subtypes.

Recording choice changes performance and deserves independent review.

Data diversity, regional label evidence and acquisition differences remain plausible bottlenecks.

We have not established a dataset ceiling or a route guaranteed to reach 0.80.


Presenter notes

These are evidence-informed interpretations, not a causal diagnosis that the problem is definitely data or definitely tensors. Distribution and kernel studies do not exhaust possible architectures. Current robust reference procedures remain separate. The scientific next step combines a bounded source feasibility decision with clinical review and an external measured-XYZ cohort plan, rather than unconstrained outer-score search.


Evidence

- docs/patient_independent_research_audit.md
- docs/next_readout_experiments.md
- docs/open_dataset_review_20261119.md

## 30. Additional data: separate research roles

Regional source labels: Shandong ECG statements.

Patient diversity: MIMIC-IV-ECG.

Manual P/QRS/T boundaries: LUDB.

Controlled simulation: MedalCare-XL.

These sources do not directly supply our six-class measured-Frank cohort.


Presenter notes

This is the verified 19 November release review. Standard ECG can supply estimated XYZ through the existing Kors conversion. That estimate differs from measured Frank XYZ and cannot create absent clinical region labels. MIMIC waveform access and linked clinical-note access have different conditions. Medal Care adds synthetic signals, with no new real patients. LUDB addresses delineation, rather than immediate regional MI class support. Any source experiment needs an explicit ECG-assisted label. ECG image credit: Yitzhak Nat, Normal EKG wave, Wikimedia Commons, CC BY-SA 4.0. The image is unchanged and illustrates an ECG, not a measured project recording.


Evidence

- docs/open_dataset_review_20261119.md
- https://commons.wikimedia.org/wiki/File:Normal_EKG_wave.svg
- https://creativecommons.org/licenses/by-sa/4.0/

## 31. Shandong metadata: what we verified

25,770 ECGs from 24,666 patients in the published metadata.

260 records from 242 patients carry one or more MI-region statements.

The dictionary includes anterior, inferior and anteroseptal statements.

Most describe old MI. ALMI and ILMI equivalents are missing.

Only metadata are downloaded. Waveform QC and compatibility remain pending.


Presenter notes

The downloaded metadata and dictionary total 1,001,058 bytes and match publisher MD 5 and local SHA-256 hashes. Native region supports are 47 anterior, 117 inferior, 82 anteroseptal and seven extensive-anterior patients. Supports overlap and must not be summed. Ten records carry multiple regional statements. The dictionary lacks ALMI and ILMI counterparts. No waveform download, quality audit, identity clearance, target-patient addition or new source fit has happened. The full signal archive is about 2.28 GB. A normal ECG statement does not establish an independently reviewed healthy control.


Evidence

- docs/open_dataset_review_20261119.md
- results/open_dataset_screen_20261119/sph_metadata_summary.json

## 32. Implementation and validation delivered

Versioned protocols, frozen folds and matched controls for each study.

Saved models, resumable runners and local progress monitors.

Independent feature reconstruction and model-prediction replay.

376 tests passed before the latest classifier study.

Partner-cohort intake is prepared. Real data and clinical review are pending.


Presenter notes

A fresh clone lacks the ignored raw data, feature caches, fitted weights and private patient predictions. Windows setup and handoff documents exist, while Windows numerical training remains unvalidated. The latest completed experiment is stopped, and the older multiview sweep remains paused. The partner intake smoke used synthetic signals and verifies software only. The downloaded MI Imaging format fixture contains two public reference cases totaling 6.48 MB. It is not an external validation cohort. Restoration counts do not add independent patients.


Evidence

- docs/CODEX_HANDOFF.md
- docs/frozen_distance_readout_findings.md
- docs/measured_vcg_partner_cohort.md
- docs/WINDOWS_SETUP.md

## 33. Next research decisions

Complete the blinded review of regional evidence and recording timing.

Audit Shandong signal units, lead order, quality and native label meaning.

Register one bounded source-learning comparison before fitting.

Seek independently labeled measured XYZ patients through a clinical partner.

Reserve external patients before selecting the next model.


Presenter notes

These are proposed steps. No new experiment or external message has been started by this presentation. A first 20-to-30-patient intake is a technical and label feasibility pilot, rather than a sufficient confirmatory cohort. Louvain, CSE, Helsinki and local clinical routes remain leads with access, custody, rights or regional-label coverage unresolved. Request templates remain unsent. Source label meaning, identity policy and laptop budget should be defined before training. A confirmatory external evaluation requires a reserved endpoint and frozen model selection.


Evidence

- docs/data_acquisition_action_plan_20261119.md
- docs/measured_vcg_partner_cohort.md
- docs/open_dataset_review_20261119.md

## 34. Progress since the previous presentation

Completed the transfer and distribution studies proposed last time.

Built and tested further tensor, supervision and classifier alternatives.

Reached approximately 0.60 on the existing all-record development task.

Identified recording selection and regional evidence as priorities for review.

Prepared the next data and validation pathway. The 0.80-0.90 target remains open.


Presenter notes

End the main talk here. The project has progressed through implemented methods, controlled comparisons and a clearer definition of the intended examination. The strongest all-record point estimates remain development results, while the earliest-record studies use a different endpoint. The appendix contains detailed comparisons, class results and verification scope. Reaching 0.80 to 0.90 remains an open research ambition. More experiments or a larger tensor alone do not establish state-of-the-art performance.


Evidence

- docs/CODEX_HANDOFF.md
- docs/patient_independent_research_audit.md

## 35. Appendix: distribution comparisons


Presenter notes

Rows belong to separate studies. All scores are three-seed ensemble macro F 1 on the original 164 patients. The physical-feature base means the median waveform, while the other bases retain the original convolution representation. The empirical-quantile gains over spread remain uncertain. The original convolution spread contrast is a separately labeled post-hoc observation. No between-row comparison isolates a single causal change.


Evidence

- results/distribution_pooling_v1/complete/REPORT.md
- results/convolution_distribution_v1/complete/REPORT.md
- results/learned_distribution_v1/complete/REPORT.md
- results/distribution_kernel_v1/complete/REPORT.md

## 36. Appendix: kernel members and ensembles


Presenter notes

Mean member F 1 averages the three F 1 scores. Ensemble F 1 instead scores decisions after averaging the three probability vectors. Both retain seeds 42, 123 and 2026. The primary selected nonlinear-minus-linear contrast is -0.019441 with interval [-0.048244,+0.005086]. All fixed representations and both families remain reported. Inner representation decisions overlap in training patients and are not independent replications.


Evidence

- results/distribution_kernel_v1/complete/REPORT.md

## 37. Appendix: source learning comparisons


Presenter notes

Each row uses its own matched control and source hypothesis. All displayed primary gain intervals include zero. The expanded native-pretraining and paired ECG procedures also failed to beat their matched scratch ensemble at 0.601801. PTB-XL transfer uses a different target schedule and therefore has a different scratch reference. Source AP and AUROC are separate endpoints. Recording-copy screens do not resolve all cross-dataset patient identity.


Evidence

- results/neural_supervised_transfer_v1/complete/REPORT.md
- docs/vcg_pretraining_pilots_findings.md
- results/neural_acs_artery_transfer_v1/complete/REPORT.md
- results/neural_staffiii_pair_transfer_v1/complete/REPORT.md
- results/neural_native_partial_labels_v1/complete/REPORT.md

## 38. Appendix: other matched readouts


Presenter notes

These are separate experiments on the 164-patient all-record task. Intervals are descriptive paired patient bootstraps conditional on saved predictions. They omit refitting uncertainty and repeated-development effects. The table retains unsuccessful comparisons and the original declared controls. Independent confirmation remains necessary.


Evidence

- docs/subtype_graph_findings.md
- docs/lag_direction_tensor_findings.md
- results/r_relative_convolution_v1/complete/REPORT.md
- results/probabilistic_subtype_v1/complete/REPORT.md
- results/neural_patient_probability_loss_v1/complete/REPORT.md
- results/model_error_comparison_v1/REPORT.md

## 39. Appendix: matched prediction errors


Presenter notes

The source ensembles reproduce macro F 1 of 0.603374 and 0.601801. Of 36 shared errors, 21 have the same wrong prediction. This is a diagnostic of complementary errors. It uses no fitted combiner and gives no permission to choose the correct model per patient using known answers. The fixed equal blend scored 0.582209.


Evidence

- results/model_error_comparison_v1/REPORT.md

## 40. Appendix: all-record class results


Presenter notes

Both columns use the same 164 patients, original class order and frozen folds. All three seeds contribute equally to each ensemble. These class results describe development predictions. We did not fit class-specific routing, change labels or choose weights using the outer results. AMI denotes anterior MI. Small class counts create substantial uncertainty.


Evidence

- results/distribution_kernel_v1/complete/REPORT.md
- results/neural_vcg_native_pretraining_v1/complete/REPORT.md

## 41. Appendix: earliest-record endpoints


Presenter notes

Rows use the same 145 dated patients but different training and inference policies. They must not be presented as one continuous improvement curve. The 164-patient all-record score of 0.601801 is separate. Earliest archived date does not establish clinical presentation time. The 164-patient one-selected-record sensitivity uses deterministic fallbacks for missing or tied dates.


Evidence

- results/index_recording_audit_v1/REPORT.md
- results/neural_later_recording_distillation_v1/complete/REPORT.md
- results/neural_earliest_training_duration_v1/complete/REPORT.md
- results/earliest_record_convolution_v1/complete/REPORT.md

## 42. Appendix: frozen readout endpoints


Presenter notes

Both columns use one record per patient. The 164-patient column is not all-record inference. The 145 dated patients remain primary. All heads use encoders from exact training partitions and training-only preprocessing. The 256-versus-64 and Mahalanobis-versus-ridge contrasts remain uncertain and unfavorable in point estimate. No outer threshold tuning or winner selection follows from this table.


Evidence

- results/frozen_patient_readout_v1/complete/REPORT.md
- results/frozen_distance_readout_v1/complete/REPORT.md

## 43. Appendix: latest classifier by class


Presenter notes

The primary cohort has 145 dated earliest-record patients. All three classifiers use the same 64-dimensional feature representation. Mahalanobis uses a shared covariance estimated from training patients with inner-selected shrinkage. Lambda choices across 30 fold-seed contexts overlap in patients. The class table does not justify changing the classifier per class based on outer performance.


Evidence

- results/frozen_distance_readout_v1/complete/REPORT.md

## 44. Appendix: data sources and storage


Presenter notes

This is the release inventory reviewed on 19 November. Download size differs from extracted arrays, feature caches and model output. MIMIC is about 90.4 GB uncompressed. LUDB is about 23.6 MB uncompressed. Medal Care supplies no real people and uses artery/transmurality labels. Shandong lacks the full six-class target mapping. Its local screen covers metadata only. A future ECG-assisted study requires its own compatibility and overlap checks.


Evidence

- docs/open_dataset_review_20261119.md

## 45. Appendix: verification coverage


Presenter notes

These counts belong to separate studies, and some studies reuse source models. They are not one combined new-fit total or independent patient count. Verification reconstructs features or restores saved parameters and repeats predictions and selections, according to the registered scope. The latest mac OS suite passed 376 tests before the final study. Windows training remains unvalidated. The presentation includes aggregate evidence only.


Evidence

- results/neural_supervised_transfer_v1/complete/REPORT.md
- results/distribution_pooling_v1/complete/REPORT.md
- results/convolution_distribution_v1/complete/REPORT.md
- results/distribution_kernel_v1/complete/REPORT.md
- results/frozen_patient_readout_v1/complete/REPORT.md
- results/frozen_distance_readout_v1/complete/REPORT.md

## 46. Appendix: selected references

Zhang et al. Automated Localization of Myocardial Infarction From Vectorcardiographic via Tensor Decomposition. IEEE TBME, 2023.

Nesaragi et al. Correlation matrix-based tensor decomposition for early prediction of sepsis. 2021.

Tan et al. MultiRocket, 2022. Dempster et al. HYDRA, 2023.

Projected distribution pooling: PSWE, NeurIPS 2021, and FSW, ICLR 2025.

PTB, PTB-XL, STAFF III and LUDB: official PhysioNet releases.

Exact experiment protocols, source publications and image credits are in presenter notes.


Presenter notes

The main paper DOI is 10.1109/TBME.2022.3202962. The sepsis paper DOI is 10.1016/j.bbe.2021.06.009. These publications motivate methods. Our bounded adaptations do not inherit their benchmark claims. Project protocols disclose implementation differences and attributions. The PTB signal illustration is an existing project figure from public data. The ECG schematic is by Yitzhak Nat under CC BY-SA 4.0, unchanged. Direct links appear below.


Evidence

- https://doi.org/10.1109/TBME.2022.3202962
- https://doi.org/10.1016/j.bbe.2021.06.009
- https://papers.nips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/3dbb8b6b5576b85afb3037e9630812dc-Abstract-Conference.html
- https://physionet.org/content/ptbdb/1.0.0/
- https://physionet.org/content/ptb-xl/1.0.3/
- https://physionet.org/content/staffiii/1.0.0/
- https://physionet.org/content/ludb/1.0.1/
- https://commons.wikimedia.org/wiki/File:Normal_EKG_wave.svg
