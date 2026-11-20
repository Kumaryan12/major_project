# Presentation update, 20 November 2026

34 main slides and 31 appendix slides. Covers all completed work after the 7 October 2026 presentation through 19 November 2026. The November decks are intermediate artifacts, not the presentation baseline. All study results are development evidence. The original presentation and numerical study artifacts are unchanged.

The deck has no footer notes or slide-number fragments. Evidence and image attribution are in presenter notes. The appendix preserves detailed matched comparisons.

## 1. Implementation progress

Patient-Independent Localization of Myocardial Infarction

from Vectorcardiographic Signals

Aryan Kumar (23ECE1006) and Kunal Maka (23ECE1017)

Guide: Dr. Shivnarayan Patidar

Department of Electronics and Communication Engineering

National Institute of Technology Goa

Progress since 7 October 2026

20 November 2026


Presenter notes

This is the complete implementation update after the presentation dated 7 October 2026. We first recall the tensor reproduction and patient benchmark. We then explain the waveform, tensor, sequence, transfer and diagnostic studies completed since then. Our primary task remains six-class localization from measured Frank XYZ on unseen patients.


Evidence

- docs/implementation_presentation_notes.md
- output/latex/mi_implementation.tex
- docs/CODEX_HANDOFF.md

## 2. What we had completed in October

We reproduced the tensor pipeline and established a patient-disjoint benchmark.

Wavelet statistics gave 0.487 patient macro F1 on PTB.

The external result was weaker. Neural models were still proposed work.


Presenter notes

The original presentation is dated 7 October 2026. Random-beat and patient-grouped rows below are beat accuracy on the same 12-class reproduction task. The six-class patient endpoint is different. The nested wavelet result is 0.4872 and the older external model scored about 0.242 on 1,092 PTB-XL patients. The October presentation reported 15 passing tests. It proposed compact neural models and hybrids, which we subsequently implemented.


Evidence

- docs/implementation_presentation_notes.md
- output/latex/mi_implementation.tex
- RESULTS.md
- ABLATION_RESULTS.md
- EXTERNAL_RESULTS.md

## 3. Progress since the October presentation

These procedures use the same PTB development endpoint with different recipes.

The score is now approximately 0.60. Independent confirmation remains necessary.


Presenter notes

The historical October score was 0.4872. The first Res Net three-seed mean is 0.550 with sample seed SD 0.040. The neural residual-context ensemble is a separate post-hoc procedure at 0.584570. The selected linear convolution ensemble is 0.603374, and the fixed-eight-epoch waveform ensemble is 0.601801. Mean member F 1 and ensemble F 1 are different endpoints. The numerical point increase from October wavelet to selected convolution is about 0.116, but this historical comparison does not isolate one cause or establish performance on a new cohort. Do not call this SOTA or a guarantee of 0.80 to 0.90.


Evidence

- ABLATION_RESULTS.md
- results/neural_repeat_v1/REPORT.md
- results/neural_residual_seed_ensemble_v1/complete/REPORT.md
- results/distribution_kernel_v1/complete/REPORT.md
- results/neural_vcg_native_pretraining_v1/complete/REPORT.md

## 4. The target data and evaluation stay fixed

164 patients, 391 recordings and 55,863 measured Frank XYZ beats.

Six classes: AMI, ALMI, ASMI, IMI, ILMI and healthy control.

Patient macro F1 gives each of the six classes equal weight.

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

## 6. The implemented research pipeline

The measured XYZ bank supports both engineered features and learned waveforms.

Training patients alone determine preprocessing, tuning and stopping.

Held-out patient predictions and saved-model replay support each report.


Presenter notes

This diagram summarizes the implementation since October. Some routes classify beats before pooling probabilities. Other routes build recording summaries and then pool records into patient features. Exact aggregation is declared per experiment. Every fitted transform belongs to its corresponding training partition. Inner validation selects settings where the protocol calls for selection. Outer folds retain patients unseen by that fitted model. Independent replay checks implementation, while a reserved external cohort is still needed for confirmation.


Evidence

- docs/CODEX_HANDOFF.md
- configs/patient_folds.csv

## 7. Waveform ResNet: proposed in October, now trained

1D ResNet: widths 16/32/64, a 64-feature embedding and a six-class head.

All held-out beat probabilities contribute to a patient prediction.

Three-seed mean: 0.550 ± 0.040. The spread measures training-seed variation.


Presenter notes

We implemented a compact 1 D Res Net that learns waveform features directly. The seed-42 plain model scored 0.586 and its statistics-fusion variant 0.584. We did not stop at that favorable seed. Seeds 123 and 2026 scored 0.557 and 0.507, giving 0.550 mean with sample standard deviation 0.040. This slide reports individual models and their mean, not a probability ensemble.


Evidence

- results/neural_vcg_v1/complete/REPORT.md
- results/neural_repeat_v1/REPORT.md

## 8. Carefully constructed multiview tensors

We separated typical recording shape, beat variation and denoising residuals.

Reconstruction checks preserve the original signed signal.

The feature studies completed. The older multiview neural sweep stays paused.


Presenter notes

We separated the shared recording shape from beat-to-beat residuals and denoising residuals. These are transformations of the same measured signal, not new sensors or extra patients. We checked their numerical reconstruction identities. We then extracted complementary morphology, scale and relation features. The feature-first route completed. The older full multiview neural sweep remains paused and is not a completed result.


Evidence

- docs/record_tensor_quality_plan.md
- docs/multiview_vcg_plan.md
- docs/tensor_feature_design_plan.md
- results/tensor_refinement_v1/complete/REPORT.md

## 9. Feature stability and tabular learning

Bootstrap feature consensus helped the matched tree procedure.

TabPFN on the resulting tensor features gave a higher point score, with an uncertain difference.

Hard semantic grouping and fixed prior correction did not improve the tested recipe.


Presenter notes

A larger feature bank was not sufficient. The refinement study gave mixed results: expanded linear 0.425 and tree 0.452, versus legacy linear 0.498 and tree 0.420. Bootstrap consensus then improved the matched tree score to 0.513. A local Tab PFN-v 2 procedure scored 0.542, but its difference from bootstrap XGBoost was uncertain. These are feature procedures and must not be conflated with the neural seed mean or its ensemble.


Evidence

- results/tensor_refinement_v1/complete/REPORT.md
- results/feature_stability_v1/complete/REPORT.md
- results/tabpfn_v2_v1/complete/REPORT.md

## 10. Relations, geometry and ordered paths

We tested physiological structure as a representation hypothesis.

Each study retained its own matched control. None established a reliable overall gain.


Presenter notes

The sepsis-paper idea became correlation and covariance tensors with shared training-only bases. Geometry described signed VCG loops and approximate phase features. Healthy-reference tensors described deviation from training references. Path signatures encoded within-beat ordered paths. Their controls help test whether the added structure is useful, and the results did not establish a consistent gain. The adaptive phase detector accepted only 23.2% of beats and its acceptance differed by class, so it cannot be called validated clinical delineation.


Evidence

- results/relational_vcg_v1/complete/REPORT.md
- results/neural_geometry_v1/complete/REPORT.md
- results/deviation_vcg_v1/complete/REPORT.md
- results/signature_vcg_v1/complete/REPORT.md

## 11. Consecutive heartbeats and RR timing

Real consecutive windows of 4, 8 or 16 beats stay within one recording.

Inner selection chooses one shared length for ordered and shuffled controls.

Shuffling moves each beat with its own RR pair. All original query beats remain covered.


Presenter notes

The hypothesis was that consistency and timing across beats distinguish infarct subtypes. We used a shared beat encoder and a recurrent context model. The matched shuffle destroys ordering while preserving beat/RR pairs. We evaluated 4, 8 and 16 beats with a comparable nominal beat budget and symmetric inner length selection. The ordered-versus-shuffled difference was minus 0.008 with an interval from minus 0.057 to plus 0.039. This did not demonstrate an ordering benefit. All 1200 fitted contexts and 120 outer models were verified.


Evidence

- results/neural_chronological_lengths_v1/complete/REPORT.md
- docs/chronological_lengths_plan.md

## 12. Direct morphology and residual context

A direct beat path preserves morphology while context adds a gated correction.

The original mean-model comparison did not establish a gain.

A separate post-hoc ensemble reached 0.584570, with an uncertain paired gain.


Presenter notes

All six original arms, three seeds and 720 fitted contexts remain reported. Mean member F 1 averages separate F 1 scores. Ensemble F 1 averages probabilities first and scores one predictor. The ordered-plus-auxiliary mean is 0.539 versus beat-only 0.551. Its later equal-seed ensemble scores 0.584570 versus matched beat-only ensemble 0.545112. Ensemble difference +0.039458 has interval [-0.034767,+0.112701]. All 180 outer members were restored. The ensemble does not rewrite the source mean-model conclusion or prove ordering is useful.


Evidence

- results/neural_residual_context_v1/complete/REPORT.md
- results/neural_residual_seed_ensemble_v1/complete/REPORT.md

## 13. Fusion and staged context learning

Neural–tensor late fusion selected a single blend weight using inner patients.

Staged learning first trained morphology, then adapted or froze it while learning context.

Neither follow-up established a gain. The adaptive fusion gate remains deferred.


Presenter notes

We tried to combine complementary neural and tensor predictions with a simple inner-selected scalar. It scored 0.583052 against 0.584570 for neural alone. We also tested continued morphology, joint context, frozen context and shuffled context in a separate staged study. Frozen context scored 0.563539 against matched morphology 0.545112, but its interval included zero and the shuffled control scored 0.565056. Inner validation chose zero adaptation updates in 82 of 120 outer adaptations. These results do not support an ordering benefit or a larger adaptive gate.


Evidence

- results/neural_tensor_fusion_v1/complete/REPORT.md
- results/neural_staged_context_v1/complete/REPORT.md

## 14. Random-convolution features with ridge

Each recording uses the coordinate-median waveform of all its original beats.

MultiRocket-style and experimental multivariate HYDRA features feed a matched ridge classifier.

Combined features improved the matched DCT control, while losing within-record variation.


Presenter notes

This route extracts many fixed convolutional responses rather than training a deep encoder. Training-only scale and threshold estimation remain inside each inner or outer partition. The combined three-seed ensemble scored 0.584266. Its difference from DCT was plus 0.138485 with a positive descriptive interval. Its gain over Multi Rocket alone remained uncertain. It is essentially level with, and slightly below, the separate historical neural ensemble 0.584570. The bounded implementation differs from upstream defaults and the HYDRA multivariate code remains experimental with GPL attribution.


Evidence

- results/rocket_vcg_v1/complete/REPORT.md
- docs/rocket_vcg_plan.md

## 15. Selected convolution features and TabPFN

Training-only bootstrap consensus selected 32 or 64 convolution features.

The selected-feature ridge and PFN procedures both lost score against full-feature ridge.

AMI class F1 fell from 0.345 to 0.080 under both selected procedures.


Presenter notes

The next bounded study asked whether a compact set of stable convolution features would let PFN help. We retained the full ridge reference and used nested budget selection for both classifiers. Neither selected procedure improved it. AMI correctly classified patients fell from 5 of 17 to 1 of 17. Fixed-budget outcomes are retained in backup, but cannot replace the prespecified selector after viewing outer scores. All 660 new classifiers and the reused source features were restored or reconstructed.


Evidence

- results/rocket_tabpfn_v1/complete/REPORT.md
- docs/rocket_tabpfn_plan.md

## 16. Source learning and supervised transfer

A separate PTB-XL task learned native diagnostic representations.

Target inference uses measured XYZ and fresh six-class heads.

Transfer minus scratch: +0.0346, interval [-0.0465, +0.1138].


Presenter notes

This complete source-to-target route was implemented after October. The bounded supervised source used 4,096 training patients and 1,942 validation patients, with source fold 10 excluded before labels and signals were read. Both source objectives trained 15 epochs, selecting epoch 13. Their source macro AP was about 0.317, a different multilabel endpoint. Source and overlap verification preceded target fitting. The matched target study completed all 630 fits and 90 outer models. The certain-source gain is uncertain. Source-target patient identity remains unresolved despite a passed recording-copy screen. Detailed source cohort, objectives, metrics and overlap results are in the appendix.


Evidence

- results/neural_supervised_transfer_v1/complete/REPORT.md
- docs/supervised_transfer_findings.md
- results/supervised_source_pilot_v1/REPORT.md
- results/supervised_source_overlap_v1/REPORT.md

## 17. Preserving variation across heartbeats

A median waveform can hide variation between the individual beats.

We encoded every original beat before pooling.

Mean describes the centre. Spread and quantiles describe variation.


Presenter notes

The representation is recording by R-relative window by feature projection by quantile. Equal weighting across patients, records and beats defines training statistics. Fixed feature projections operate in feature space and do not rotate physical XYZ. Mean/spread uses a fixed normal-quantile basis as a control, without asserting Gaussian physiology. The first study holds all four stored tensor sizes at 720 columns. Later studies keep the full original convolution base and append 144 columns.


Evidence

- docs/distribution_pooling_plan.md
- docs/convolution_distribution_plan.md

## 18. Distribution tensors: what changed

We tested physical features, intact convolution features and learned embeddings.

Quantile gains over matched mean/spread controls remained uncertain.

More distribution detail did not consistently improve localization.


Presenter notes

Each table row is a different matched study on the original 164 patients. The physical study keeps a 720-column tensor across all four arms. The convolution and learned studies keep the intact 3,584-column original base and append 144 columns. Their baseline is 0.584266. Every beat was replayed in each respective feature study. The physical quantile-minus-moments interval is [-0.052573,+0.076160]. Intact-convolution quantiles minus spread is -0.004613 with interval [-0.013297,0]. Learned quantiles minus spread is +0.033028 with interval [-0.011283,+0.079062]. Full arm results remain in the appendix.


Evidence

- results/distribution_pooling_v1/complete/REPORT.md
- results/convolution_distribution_v1/complete/REPORT.md
- results/learned_distribution_v1/complete/REPORT.md

## 19. Linear and nonlinear classifiers

Inner validation chose one shared representation for both classifier families.

A nonlinear kernel did not improve the selected ensemble.

The 0.603374 result comes from the linear selection procedure.


Presenter notes

We compared class-balanced linear kernel ridge and linear-plus-Gaussian kernel ridge on four fixed representations. The representation selection is shared symmetrically, preventing each family from selecting a separate representation for the primary comparison. All eight fixed arms, all three seeds and 2,400 model restorations remain reported. Selected nonlinear minus linear is -.019441 [-.048244,+.005086]. Mean member F 1 rises .562847 to .573104 while ensemble F 1 falls, so those endpoints must remain distinct. Nonlinear search used four candidates versus two linear candidates.


Evidence

- results/distribution_kernel_v1/complete/REPORT.md
- docs/distribution_kernel_findings.md

## 20. Further tensor hypotheses and controls

Each method keeps the same patients, original labels and frozen folds.

These tests did not establish a reliable gain or validated clinical timing.


Presenter notes

The within-beat lag tensor compares signed direction, magnitude and balanced sign scrambling. Signed 0.531364 is below the original base 0.584266 and scrambled 0.546633. Signed minus magnitude remains uncertain. Subtype graph regularization compares the clinical graph with ridge and a relabeled graph, keeping HC isolated. Subtype graph minus ridge is +0.007158 with interval [-0.012932,+0.028282]. The local timing audit replays all 391 records and injected controls. None of the three post-R windows met its required patient coverage, so the full timing tensor was deferred before classifier fitting.


Evidence

- docs/lag_direction_tensor_findings.md
- docs/subtype_graph_findings.md
- docs/local_timing_audit_findings.md

## 21. Additional source-learning pilots

All four extensions use measured XYZ alone at target inference.

None demonstrated a gain over its matched scratch ensemble of 0.601801.

The source questions and their matched controls differ.


Presenter notes

Native masked reconstruction added 124 unlabeled measured-XYZ patients. It scored 0.518364 versus target-only pretraining 0.454298, but remained below scratch. Correct ECG pairing scored 0.564293 versus shuffled 0.529345, with an uncertain pairing difference. ACS OMI-plus-artery scored 0.580730 versus OMI-only 0.570389. STAFF paired-change scored 0.539589 versus pair-mean 0.555118. Each study has separately replayed matched models. ECG-derived source vectors differ from measured Frank target vectors, and source native labels are not substituted for PTB class labels. Individual sources and controls are detailed in the appendix.


Evidence

- docs/vcg_pretraining_pilots_findings.md
- results/neural_acs_artery_transfer_v1/complete/REPORT.md
- results/neural_staffiii_pair_transfer_v1/complete/REPORT.md

## 22. Supervision and patient-level classifiers

Partial region mentions, hierarchical probabilities and patient losses were tested.

Their declared matched comparisons did not establish a reliable gain.

Frozen score corrections and equal model averaging also failed to improve the references.


Presenter notes

The partial-label study retains positive-only native header mentions from 13 outside-target patients, with a tag-count-matched shuffled control. True 0.575887 versus shuffled 0.568947 is uncertain and below scratch 0.601801. Hierarchical classification scored 0.586828 versus flat 0.562819, with interval for the difference [-0.001207,+0.056060]. Patient-level cross-entropy scored 0.469596 versus beat cross-entropy 0.510134 under the same grouped-batch sampler. This sampler differs from the older 0.601801 waveform procedure. Frozen score corrections leave all original ensemble decisions unchanged. Equal convolution-waveform blend 0.582209 fails to beat either component. R-relative aligned pooling likewise did not beat shuffled pooling. Complete results and matched controls are in the appendix.


Evidence

- results/neural_native_partial_labels_v1/complete/REPORT.md
- results/probabilistic_subtype_v1/complete/REPORT.md
- results/neural_patient_probability_loss_v1/complete/REPORT.md
- results/distribution_score_correction_v1/complete/REPORT.md
- results/model_error_comparison_v1/REPORT.md
- results/r_relative_convolution_v1/complete/REPORT.md

## 23. Recording differences became a key question

57 of 93 patients with repeated recordings received different record-level labels.

Most remaining convolution errors exchange one MI subtype for another.

We audited examination choice, beat count, signal quality and recording dates.

These diagnostics identify questions for review, without changing target labels.


Presenter notes

The research audit reproduced convolution macro F 1 of 0.603374 and waveform macro F 1 of 0.601801. It checked 2,400 fit memberships and all 391 original target headers. Of 52 convolution errors, 44 exchange one MI subtype for another. The 57 of 93 statistic describes disagreement between model predictions across recordings, rather than proven inconsistency in clinical diagnoses. Uniform one-record expected-count sensitivity was about 0.5683. The later earliest-date analysis answers a different question. All original labels remain fixed and independent clinical review is pending.


Evidence

- docs/patient_independent_research_audit.md
- docs/path_to_080_macro_f1.md

## 24. One earliest recording changes the endpoint

145 patients had a usable date and a unique earliest recording.

The same frozen model scored 0.586275 using all their beats.

Using only their earliest recording scored 0.515049.

Difference: -0.0712, interval [-0.1470, +0.0036].


Presenter notes

This audit fits no models and replays all 30 held-out prediction blocks. Thirteen patients have an earliest-date tie and six have a missing or invalid date, leaving 145 with a unique earliest recording. The main comparison uses those same 145 patients under both inference policies. The all-record result on all 164 patients remains 0.601801. Earliest archived recording does not necessarily mean first acute clinical presentation. Equal weighting across records left all 164 ensemble class decisions unchanged.


Evidence

- results/index_recording_audit_v1/REPORT.md
- docs/index_recording_audit_plan.md

## 25. Beat count and signal quality checks

Matching beat counts preserved the higher point score for later records.

The intervals remain broad. The quality proxies did not explain the gap.

Independent clinical and recording-timing review remains pending.


Presenter notes

The matched experiment uses 256 draws per patient, with 64 beats for 78 patients and 50 for one patient. Displayed scores are macro F 1 of expected confusion counts. Later minus earliest was +0.095882 with interval [-0.037926,+0.230258]. Half from each minus earliest was +0.084854 with interval [-0.017018,+0.192643]. The wide intervals prevent a firm conclusion. Median archival spacing was four days. Timing, treatment, injury evolution and acquisition could all differ. These 79 repeat-record patients are a separate descriptive subset.

The audit uses zero new fits. Lower drift versus improved correctness had Spearman rho -0.022 with interval [-0.235,+0.185]. The R-detector disagreement association was +0.059 with interval [-0.175,+0.290]. No ADC rail hits occurred. Mean later-minus-earliest drift was -0.03120 m V, with 41 patients showing lower and 38 higher drift. These proxies do not cover all acquisition quality or disease evolution. A private blinded packet of 24 matched review pairs is prepared, with its key separate. Independent clinical review has not happened yet.


Evidence

- results/matched_recording_beats_v1/REPORT.md
- docs/matched_recording_beats_plan.md
- results/paired_recording_quality_v1/REPORT.md
- docs/paired_recording_quality_plan.md

## 26. Training specifically on the earliest record

The primary endpoint here uses the same 145 dated patients.

Distillation, longer training and convolution features did not establish a gain.

These scores answer a different question from all-record macro F1 near 0.60.


Presenter notes

Teachers hold out both the donor and the recipient in the same-class shuffled comparison. The primary endpoint uses 145 dated patients. Paired minus supervised was +0.002434 with interval [-0.079058,+0.085427]. Paired minus shuffled was +0.021697 with interval [-0.062848,+0.102840]. Every paired student seed scored below its supervised counterpart despite the small increase in ensemble point score. All 30 cross-fitted teachers and 90 students replayed, with 30 matched groups. Earliest-record inference uses no later recording.

On the 145-patient endpoint, inner-selected duration minus fixed eight epochs gave -0.012208 with interval [-0.082185,+0.061027]. The fixed-16 result was a secondary descriptive contrast of +0.031422 with interval [-0.039516,+0.100476]. All 120 trajectories, 360 checkpoints and 30 original baseline blocks replayed. Mean training loss fell from 0.358157 at epoch eight to 0.044369 at epoch 32. A separate earliest-record convolution comparison scored 0.478122 for combined convolution, 0.461037 for DCT and 0.521335 for the matched waveform. It replayed 420 ridge fits and 120 contexts.


Evidence

- results/neural_later_recording_distillation_v1/complete/REPORT.md
- docs/later_recording_distillation_plan.md
- results/neural_earliest_training_duration_v1/complete/REPORT.md
- results/earliest_record_convolution_v1/complete/REPORT.md

## 27. Testing the frozen ResNet readout

We froze the ResNet encoder and fitted patient-level classifiers.

Ridge on 64 features scored 0.535691. Retaining 256 features scored 0.501020.

The larger readout did not establish an improvement on the earliest-record task.


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

The package expanded from 15 tested cases in October to 376 before the latest study.

Versioned protocols, frozen folds and matched controls define each experiment.

Independent reports replay saved features, models, selections and predictions.

Partner-cohort intake is prepared. Real data and clinical review are pending.

Completed studies are stopped. The older multiview sweep stays paused.


Presenter notes

A fresh clone lacks the ignored raw data, feature caches, fitted weights and private patient predictions. Windows setup and handoff documents exist, while Windows numerical training remains unvalidated. The latest completed experiment is stopped, and the older multiview sweep remains paused. The partner intake smoke used synthetic signals and verifies software only. The downloaded MI Imaging format fixture contains two public reference cases totaling 6.48 MB. It is not an external validation cohort. Restoration counts do not add independent patients.


Evidence

- docs/CODEX_HANDOFF.md
- docs/frozen_distance_readout_findings.md
- docs/measured_vcg_partner_cohort.md
- docs/WINDOWS_SETUP.md
- docs/implementation_presentation_notes.md

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

## 34. What we have completed since October

Trained the neural models that were proposed in the October presentation.

Implemented richer tensors, feature learning, sequence models and source transfer.

Raised historical all-record point results from 0.487 to approximately 0.60.

Audited recording choice and prepared a pathway to additional data.

The 0.80-0.90 target and independent external confirmation remain open.


Presenter notes

The main presentation ends here. October is the baseline for the entire update. The project progressed through model implementation, controlled comparisons and stronger development point results, while preserving patient separation. The historical 0.487-to-0.60 comparison spans different procedures and repeated development. The appendix covers unsuccessful branches, all important controls, member-versus-ensemble distinctions, class results and source evidence. More experiments alone do not establish SOTA or guarantee a clinical result.


Evidence

- docs/CODEX_HANDOFF.md
- docs/patient_independent_research_audit.md

## 35. Appendix: patient bags and contrastive learning


Presenter notes

Patient-bag learning and learned Tucker factors were trained end to end with patient labels. The bag can include separate recordings and has no claimed temporal order. Attention, learned scales and compression did not outperform the single-seed waveform reference. The original contrastive comparison improved its altered-sampler CE control by 0.037 on average, but the interval included zero. A more patient-diverse follow-up reversed the point difference. This does not establish robust superiority.


Evidence

- results/neural_patient_tensor_v1/complete/REPORT.md
- results/neural_contrastive_v1/complete/REPORT.md
- results/neural_contrastive_diverse_v1/complete/REPORT.md

## 36. Appendix: waveform follow-ups


Presenter notes

The preprocessing study tested raw against the established denoised input. The longer-crop study tested more context around R. Bounded regularization selected among four fixed presets using inner patients and shared baseline fits where appropriate. Territory supervision derived labels from the existing class names and therefore supplied no independent anatomy annotation. The error audit and learning curve are diagnostics rather than new clinical ground truth.


Evidence

- results/neural_preprocessing_v1/complete/REPORT.md
- results/neural_context_v1/complete/REPORT.md
- results/neural_regularization_v1/complete/REPORT.md
- results/neural_territory_v1/complete/REPORT.md
- results/neural_diagnostics_v1/REPORT.md
- results/neural_learning_curve_v1/complete/REPORT.md

## 37. Appendix: correlation and covariance tensors


Presenter notes

All arms used the same fixed depth-3 XGBoost and training weights. Statistics retain amplitude information. Covariance also retains amplitude relative to training-only channel scales, whereas correlation omits absolute scale. Shared covariance gave 0.470 compared with 0.469 for short-beat statistics, with a paired interval including zero. The approximate regions are not expert QRS, ST or T delineations.


Evidence

- results/relational_vcg_v1/complete/REPORT.md
- https://doi.org/10.1016/j.bbe.2021.06.009

## 38. Appendix: geometry and reference limitations

Adaptive boundaries passed strict checks for 12,977 of 55,863 beats, about 23.2%.

Acceptance was class dependent: about 43% for HC and 15–22% for MI classes.

Fixed and adaptive descriptors retained the full target cohort through declared fallback rules.

Healthy references came only from the relevant training patients.

Path signatures summarize within-beat order. They differ from cross-beat chronology.


Presenter notes

Geometry can provide physically interpretable descriptors without proving a clinical delineator. The adaptive linear result 0.467 versus statistics 0.445 had an interval from minus 0.061 to plus 0.107. The tree statistics-plus-geometry result 0.461 versus 0.409 had an interval from minus 0.024 to plus 0.131. Healthy deviation scored 0.479 versus matched morphology 0.499. Signature 3 scored 0.508 versus morphology 0.515. We therefore retain these as completed ablations with no demonstrated reliable overall gain.


Evidence

- results/neural_geometry_v1/complete/REPORT.md
- results/deviation_vcg_v1/complete/REPORT.md
- results/signature_vcg_v1/complete/REPORT.md

## 39. Appendix: earlier transfer and ECG extension


Presenter notes

The old source-transfer experiment averaged three target-seed scores: pretrained 0.520 versus matched scratch 0.553. That is not the current supervised transfer result. The ECG extension used simultaneous measured twelve-lead ECG in PTB. Its VCG twin control matched added capacity without adding ECG information. Bilinear fusion scored 0.534 versus additive 0.536 and VCG 0.550, all seed means. The new primary project input remains measured XYZ.


Evidence

- results/neural_transfer_v1/complete/REPORT.md
- results/neural_paired_ecg_v1/complete/REPORT.md

## 40. Appendix: feature refinement and budgets


Presenter notes

Increasing DCT ranks improved reconstruction but did not give consistent localization gains. The early feature-stability result and the later convolution-feature selection result concern different feature banks. We cannot generalize that selection always helps. In the later study both selected procedures reduced AMI F 1, while fixed 32 and 64 feature controls showed that the classifier and budget interact. These controls are evidence for a future hypothesis, not permission to select a new outer-score winner retrospectively.


Evidence

- results/tensor_refinement_v1/complete/REPORT.md
- results/feature_stability_v1/complete/REPORT.md
- results/tabpfn_v2_v1/complete/REPORT.md
- results/rocket_tabpfn_v1/complete/REPORT.md

## 41. Appendix: all residual-context arms


Presenter notes

This complete table prevents a favorable ensemble arm from hiding the original study. Mean member F 1 averages three scores, whereas ensemble F 1 scores the average probability predictor. Ordered context plus auxiliary has the highest ensemble point score within this table but its matched improvement remains uncertain. The full source study verified 720 fits, 180 outer restorations and 120 matched checks. The follow-up independently restored all 180 members again for inference.


Evidence

- results/neural_residual_context_v1/complete/REPORT.md
- results/neural_residual_seed_ensemble_v1/complete/REPORT.md

## 42. Appendix: all staged-learning arms


Presenter notes

Staged learning protected the morphology encoder while asking whether context could improve it. Continued morphology uses more training without context. Joint and frozen branches separate adaptation from context learning. A shuffled branch tests ordering. Frozen versus morphology had a point gain of 0.018428 with an interval including zero, and shuffled scored slightly higher than ordered frozen. All 600 phase checkpoints, 150 outer checkpoints and 120 matched checks were verified.


Evidence

- results/neural_staged_context_v1/complete/REPORT.md

## 43. Appendix: distribution comparisons


Presenter notes

Rows belong to separate studies. All scores are three-seed ensemble macro F 1 on the original 164 patients. The physical-feature base means the median waveform, while the other bases retain the original convolution representation. The empirical-quantile gains over spread remain uncertain. The original convolution spread contrast is a separately labeled post-hoc observation. No between-row comparison isolates a single causal change.


Evidence

- results/distribution_pooling_v1/complete/REPORT.md
- results/convolution_distribution_v1/complete/REPORT.md
- results/learned_distribution_v1/complete/REPORT.md
- results/distribution_kernel_v1/complete/REPORT.md

## 44. Appendix: kernel members and ensembles


Presenter notes

Mean member F 1 averages the three F 1 scores. Ensemble F 1 instead scores decisions after averaging the three probability vectors. Both retain seeds 42, 123 and 2026. The primary selected nonlinear-minus-linear contrast is -0.019441 with interval [-0.048244,+0.005086]. All fixed representations and both families remain reported. Inner representation decisions overlap in training patients and are not independent replications.


Evidence

- results/distribution_kernel_v1/complete/REPORT.md

## 45. Appendix: source learning comparisons


Presenter notes

Each row uses its own matched control and source hypothesis. All displayed primary gain intervals include zero. The expanded native-pretraining and paired ECG procedures also failed to beat their matched scratch ensemble at 0.601801. PTB-XL transfer uses a different target schedule and therefore has a different scratch reference. Source AP and AUROC are separate endpoints. Recording-copy screens do not resolve all cross-dataset patient identity.


Evidence

- results/neural_supervised_transfer_v1/complete/REPORT.md
- docs/vcg_pretraining_pilots_findings.md
- results/neural_acs_artery_transfer_v1/complete/REPORT.md
- results/neural_staffiii_pair_transfer_v1/complete/REPORT.md
- results/neural_native_partial_labels_v1/complete/REPORT.md

## 46. Appendix: other matched readouts


Presenter notes

These are separate experiments on the 164-patient all-record task. Intervals are descriptive paired patient bootstraps conditional on saved predictions. They omit refitting uncertainty and repeated-development effects. The table retains unsuccessful comparisons and the original declared controls. Independent confirmation remains necessary.


Evidence

- docs/subtype_graph_findings.md
- docs/lag_direction_tensor_findings.md
- results/r_relative_convolution_v1/complete/REPORT.md
- results/probabilistic_subtype_v1/complete/REPORT.md
- results/neural_patient_probability_loss_v1/complete/REPORT.md
- results/model_error_comparison_v1/REPORT.md

## 47. Appendix: matched prediction errors


Presenter notes

The source ensembles reproduce macro F 1 of 0.603374 and 0.601801. Of 36 shared errors, 21 have the same wrong prediction. This is a diagnostic of complementary errors. It uses no fitted combiner and gives no permission to choose the correct model per patient using known answers. The fixed equal blend scored 0.582209.


Evidence

- results/model_error_comparison_v1/REPORT.md

## 48. Appendix: all-record class results


Presenter notes

Both columns use the same 164 patients, original class order and frozen folds. All three seeds contribute equally to each ensemble. These class results describe development predictions. We did not fit class-specific routing, change labels or choose weights using the outer results. AMI denotes anterior MI. Small class counts create substantial uncertainty.


Evidence

- results/distribution_kernel_v1/complete/REPORT.md
- results/neural_vcg_native_pretraining_v1/complete/REPORT.md

## 49. Appendix: earliest-record endpoints


Presenter notes

Rows use the same 145 dated patients but different training and inference policies. They must not be presented as one continuous improvement curve. The 164-patient all-record score of 0.601801 is separate. Earliest archived date does not establish clinical presentation time. The 164-patient one-selected-record sensitivity uses deterministic fallbacks for missing or tied dates.


Evidence

- results/index_recording_audit_v1/REPORT.md
- results/neural_later_recording_distillation_v1/complete/REPORT.md
- results/neural_earliest_training_duration_v1/complete/REPORT.md
- results/earliest_record_convolution_v1/complete/REPORT.md

## 50. Appendix: frozen readout endpoints


Presenter notes

Both columns use one record per patient. The 164-patient column is not all-record inference. The 145 dated patients remain primary. All heads use encoders from exact training partitions and training-only preprocessing. The 256-versus-64 and Mahalanobis-versus-ridge contrasts remain uncertain and unfavorable in point estimate. No outer threshold tuning or winner selection follows from this table.


Evidence

- results/frozen_patient_readout_v1/complete/REPORT.md
- results/frozen_distance_readout_v1/complete/REPORT.md

## 51. Appendix: latest classifier by class


Presenter notes

The primary cohort has 145 dated earliest-record patients. All three classifiers use the same 64-dimensional feature representation. Mahalanobis uses a shared covariance estimated from training patients with inner-selected shrinkage. Lambda choices across 30 fold-seed contexts overlap in patients. The class table does not justify changing the classifier per class based on outer performance.


Evidence

- results/frozen_distance_readout_v1/complete/REPORT.md

## 52. Appendix: data sources and storage


Presenter notes

This is the release inventory reviewed on 19 November. Download size differs from extracted arrays, feature caches and model output. MIMIC is about 90.4 GB uncompressed. LUDB is about 23.6 MB uncompressed. Medal Care supplies no real people and uses artery/transmurality labels. Shandong lacks the full six-class target mapping. Its local screen covers metadata only. A future ECG-assisted study requires its own compatibility and overlap checks.


Evidence

- docs/open_dataset_review_20261119.md

## 53. Appendix: verification coverage


Presenter notes

These counts belong to separate studies, and some studies reuse source models. They are not one combined new-fit total or independent patient count. Verification reconstructs features or restores saved parameters and repeats predictions and selections, according to the registered scope. The latest mac OS suite passed 376 tests before the final study. Windows training remains unvalidated. The presentation includes aggregate evidence only.


Evidence

- results/neural_supervised_transfer_v1/complete/REPORT.md
- results/distribution_pooling_v1/complete/REPORT.md
- results/convolution_distribution_v1/complete/REPORT.md
- results/distribution_kernel_v1/complete/REPORT.md
- results/frozen_patient_readout_v1/complete/REPORT.md
- results/frozen_distance_readout_v1/complete/REPORT.md

## 54. Expanded supervised source cohort

PTB-XL native diagnostic annotations provide a separate source-learning task.

We include all 1,841 likelihood-100 MI training patients, then fill to 4,096 patients.

All selected visits remain. Source fold 10 stays excluded before label and signal access.


Presenter notes

The native audit examined official training folds 1 through 8 and validation fold 9. The bounded expansion retained 5063 training visits from 4096 patients, with the entire fold-9 validation set of 2183 records from 1942 patients. The source input is Kors-derived XYZ from twelve-lead ECG, not measured Frank XYZ. Every record uses up to four uniformly spaced complete crops, with masks for padding. Source training is additional information and compute, so it requires explicit provenance and a new matched target comparison.


Evidence

- results/supervised_source_audit_v1/REPORT.md
- results/supervised_source_pilot_v1/REPORT.md

## 55. Native source annotation objectives

Twenty-eight heads meet training-only support in the retained source cohort.

Unlisted codes are annotation negatives under a disclosed research convention.

Listed likelihood 0 is unknown and masked in both objectives.

Primary “certain” objective uses likelihood-100 positives and masks intermediate annotations.

Sensitivity objective also counts listed intermediate annotations as positive.


Presenter notes

We preserve overlapping native diagnoses instead of forcing PTB-XL into our exclusive six target classes. An unlisted code is not adjudicated absence of disease. Likelihood values are not calibrated soft probabilities. Source NORM is not automatically clinical HC, and broad source AMI cannot be silently mapped to the target exclusive AMI class. Both objectives share common validation masks and train-only normalization, sampling and positive weights. Validation macro average precision chooses the checkpoint.


Evidence

- results/supervised_source_pilot_v1/REPORT.md
- docs/supervised_source_pilot_plan.md

## 56. Completed supervised source training

Both source objectives trained 15 epochs and selected epoch 13 using validation AP.

One source training seed, shared crop-pooling ResNet and all 28 heads retained.

Restored models reproduced full validation predictions.


Presenter notes

The primary source model reached macro AP 0.317456 compared with a training-prevalence constant baseline of 0.037470. Its AUROC was 0.908703 and fixed-threshold F 1 was 0.290552. The intermediate-positive sensitivity had nearly identical AP. We retain both rather than selecting an objective from this tiny difference. These are patient-weighted record-level multilabel source metrics on model-selection data. They are not target six-class macro F 1, and good source learning does not prove transfer.


Evidence

- results/supervised_source_pilot_v1/REPORT.md

## 57. Expanded recording-overlap screen

All 7,246 retained source recordings screened against 391 target recordings.

Independent replay re-decoded signals and repeated every selected continuous comparison.

The recording-copy screen passed. Patient identity across datasets remains unresolved.


Presenter notes

We screened source training and source model-selection validation, not only a convenient subset. Encoded-file matching found zero copies. Template retrieval selected top-eight candidates and the continuous screen checked 57968 pairs out of 2833186 possible. All 20 known-copy controls passed, with zero high-correlation candidates and zero insufficient-active-lead pairs. This checks a declared class of recording copies. Different visits by the same person, short or transformed copies and retrieval misses remain possible, so this cannot be called patient-identity clearance.


Evidence

- results/supervised_source_overlap_v1/REPORT.md
- docs/supervised_source_overlap_plan.md

## 58. Appendix: within-beat direction tensor

A lag tensor captured symmetric relations and signed directional changes.

Magnitude and scrambled-sign controls tested what direction contributes.

Signed direction did not beat the matched baseline or scrambled control.


Presenter notes

We used four R-relative windows and four positive lags. Symmetric entries are shared. The antisymmetric component receives original signs, absolute magnitude or balanced per-beat scrambled signs. Time reversal preserves symmetric entries and flips directional signs in the integrity tests. All 55,863 beats, 120 contexts, 210 source models and 630 new classifiers were independently replayed. Signed minus magnitude +.013444 [-.027522,+.054415], signed minus scrambled -.015269 [-.066919,+.034030].


Evidence

- results/lag_direction_tensor_v1/complete/REPORT.md
- docs/lag_direction_tensor_findings.md

## 59. Appendix: all partial-label arms

13 outside-target patients contributed positive region mentions.

Unmentioned regions stayed masked. Shuffling preserved tag counts.

True labels did not improve the matched scratch ensemble.


Presenter notes

The source audit reproduced 34 recordings and 4,644 beats from 14 candidate patients. Thirteen patients had a shared observed region tag and supplied 4,215 supervised beats. Header mentions have not received independent anatomical adjudication. True minus shuffled gave +0.006940 with interval [-0.042964,+0.058767]. True minus scratch gave -0.025914 with interval [-0.085288,+0.035006]. The no-self shuffle changed 10 of 13 tag sets while preserving per-update tag counts and source waveforms. All 120 fits and 30 matched groups replayed. All 30 scratch probability blocks matched the earlier pilot exactly.


Evidence

- results/neural_native_partial_labels_v1/complete/REPORT.md
- docs/native_partial_labels_findings.md

## 60. Appendix: frozen score corrections

We kept the original convolution predictions and learned a small correction.

Corrections used convolution summaries or learned spread/quantile features.

Every final ensemble retained the original predictions and 0.584266 score.

Inner validation found no useful correction under the tested settings.


Presenter notes

The score-correction study isolates whether a regularized residual can improve the original decision scores without replacing the base. Four procedures selected correction off or penalty inside training folds. All four ensembles reproduce the original 164 labels. Independently verified 120 contexts, 210 source restorations, 1,440 active inner and 30 active outer correction fits. The result does not prove residual correction can never help. Learned features inherit original inner-validation stopping.


Evidence

- results/distribution_score_correction_v1/complete/REPORT.md
- docs/distribution_score_correction_findings.md

## 61. Appendix: R-relative and subtype readouts

R-aligned convolution windows tested whether local position helps.

Aligned and shuffled windows scored almost identically.

A hierarchical classifier first estimated MI probability, then its subtype.

Its gain over flat classification remained uncertain.


Presenter notes

The temporal study retained the 3,584-feature base and added 512 columns. Aligned minus whole-beat pooling gave +0.010004 with interval [-0.025707,+0.046669]. Aligned minus shuffled was -0.000221 with interval [-0.041515,+0.039815]. All 630 new fits, 210 source baselines and 120 contexts replayed. In the separate subtype study, hierarchical minus flat classification gave +0.024010 with interval [-0.001207,+0.056060]. The historical selected-linear reference was 0.603374. Both new classifiers used the same representation and matched regularization grids. All 600 procedures replayed. Hierarchy supplied no additional clinical labels.


Evidence

- results/r_relative_convolution_v1/complete/REPORT.md
- results/probabilistic_subtype_v1/complete/REPORT.md

## 62. Appendix: paired recording quality

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

## 63. Appendix: training-patient learning curve

More training patients improved the historical waveform result.

The curve does not establish a ceiling or predict the sample size needed for 0.80.

Error bars describe variation across the declared training seeds.


Presenter notes

Existing completed project figure. The plotted patient budgets and fixed held-out folds belong to the original learning-curve study. Approximate macro F 1 rose from 0.407 to 0.550 as the training budget rose. This is an observed data-budget experiment, not a forecast for a new dataset. The original figure is unchanged. Source image and experiment metrics are cited below.


Evidence

- results/neural_learning_curve_v1/complete/REPORT.md
- results/neural_learning_curve_v1/complete/learning_curve.png

## 64. Appendix: completed source-learning curves

Both source objectives trained 15 epochs and selected epoch 13.

Source validation AP chooses the source checkpoint.

This figure reports source diagnostics, separate from target localization macro F1.


Presenter notes

Existing unmodified aggregate source-only figure. Both declared annotation objectives and the prevalence-constant comparator remain. The PTB-XL source cohort and 28 diagnostic heads differ from the six-class Frank-XYZ target. Final target transfer results are completed and reported in the main deck. Source macro AP about 0.317 is not a localization score.


Evidence

- results/supervised_source_pilot_v1/REPORT.md
- results/supervised_source_pilot_v1/learning_curves.png

## 65. Appendix: selected references

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
- docs/progress_presentation_20261117_illustrated_notes.md
- docs/implementation_presentation_notes.md
- https://proceedings.mlr.press/v80/ilse18a.html
- https://arxiv.org/abs/2004.11362
