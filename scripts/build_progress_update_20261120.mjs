/** Formal November update. Uses the established native Keynote fallback.
 * Historical decks and numerical studies remain read-only.
 * node scripts/build_progress_update_20261120.mjs prepare|build|finish|pdf
 */
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const root=process.cwd(), build=path.join(root,'tmp/presentations/update_20261120');
fs.mkdirSync(build,{recursive:true});
const slides=[];
const R=n=>`results/${n}/complete/REPORT.md`;
function s(title,lines,notes,sources,extra={}){slides.push({title,lines,notes,sources,...extra});}
function backup(title,table,notes,sources,extra={}){s('Appendix: '+title,[],notes,sources,{backup:true,table,...extra});}
s('Implementation progress',[
 'Patient-Independent Localization of Myocardial Infarction',
 'from Vectorcardiographic Signals',
 'Aryan Kumar (23ECE1006) and Kunal Maka (23ECE1017)',
 'Guide: Dr. Shivnarayan Patidar',
 'Department of Electronics and Communication Engineering',
 'National Institute of Technology Goa',
 'Update after the previous presentation',
 '20 November 2026'
], 'This presentation begins where our illustrated 17 November deck ended. It covers subsequent completed experiments, recording-level audits and data acquisition preparation. The primary task remains six-class patient localization using measured Frank XYZ. The original paper and problem statement can be introduced separately by Kunal. Main slides form a complete account, with detailed comparisons in the appendix.', ['docs/CODEX_HANDOFF.md','docs/progress_presentation_20261117_illustrated_notes.md'],{cover:true});
s('Where we ended last time',[
 'We had implemented waveform learning, tensor alternatives and patient-level evaluation.',
 'Supervised source training had finished. Target transfer was still running.',
 'Our next proposal was to preserve variation across all heartbeats.'
], 'The previous illustrated deck is the baseline for this update, rather than the much older October deck. Its neural ensemble was post-hoc and averaged all three seeds. The random-convolution ensemble was a separate procedure. Both results used the same original target cohort, but the controls and aggregation differed. The source AP from PTB-XL was a different multilabel endpoint.', ['output/pptx/mi_progress_since_october_2026_illustrated.pptx',R('neural_residual_seed_ensemble_v1'),R('rocket_vcg_v1')],{table:[['Previous completed reference','Ensemble macro F1'],['Neural residual-context ensemble','0.584570'],['Combined convolution features + ridge','0.584266']],widths:[630,230],rowH:46});
s('Current results in context',[
 'The strongest all-record development procedures now score approximately 0.60.',
 'These are historical procedures with different training recipes.',
 'Repeated PTB development still requires independent confirmation.'
], 'The selected linear result comes from a shared inner representation selection procedure in the kernel study. The fixed-eight-epoch waveform scratch result comes from the native pretraining pilots and reproduces across several later controls. Their point differences versus the prior presentation are approximately +0.0191 and +0.0172, respectively, but these are not matched causal improvement estimates. The aspiration of 0.80 to 0.90 remains unachieved. Do not present the 145-patient earliest-record endpoint as a lower version of this same metric.', [R('distribution_kernel_v1'),R('neural_vcg_native_pretraining_v1'),'results/model_error_comparison_v1/REPORT.md'],{chart:{labels:['Previous convolution','Selected linear convolution','Fixed-eight waveform'],values:[.584266,.603374,.601801],max:.7},chartH:236});
s('The target data and evaluation stay fixed',[
 '164 patients, 391 recordings and 55,863 measured Frank XYZ beats.',
 'Six classes: AMI, ALMI, ASMI, IMI, ILMI and healthy control.',
 'Patient macro F1 remains the primary endpoint.',
 'Ten frozen outer patient folds. Inner patients choose settings when required.',
 'All records from one patient stay in the same partition.'
], 'Original signed XYZ is sampled at 1,000 Hz. Each crop has 651 samples, from R minus 250 ms to R plus 400 ms. Preserve clinical class names: AMI means anterior MI in this cohort, not a generic acute-MI flag. Counts are 17,16,27,29,23,52 in class order. Training alone fits scaling, reference libraries, selection and stopping. Micro F1 equals accuracy in this single-label task. Every displayed ensemble retains all three declared seeds unless a source-only metric is explicitly identified.', ['AGENTS.md','configs/patient_folds.csv','docs/CODEX_HANDOFF.md']);
s('Measured XYZ remains the primary input',[
 'Recorded and denoised signals retain signed orientation and amplitude.',
 'Detected R peaks anchor the original beat crops.',
 'New representations reuse this audited signal bank.'
], 'This figure is the public PTB patient001/s0010_re example already used in the previous formal deck. It shows recorded and denoised measured XYZ, detector peaks and a shaded R-relative crop. It is an illustrative public recording, not a class prototype or evidence of a validated clinical delineator. We retain the original target R positions. Image credit: existing project figure, public PTB database.', ['https://physionet.org/content/ptbdb/1.0.0/','output/latex/assets/vcg_processing_grayscale.png'],{image:'output/latex/assets/vcg_processing_grayscale.png',imageBox:[44,101,610,338],side:true});
s('Preserving variation across heartbeats',[
 'A median waveform can hide differences between the individual beats.',
 'We encoded every original beat before pooling.',
 'Matched controls separate average shape, spread and distribution shape.'
], 'The representation is recording by R-relative window by feature projection by quantile. Equal weighting across patients, records and beats defines training statistics. Fixed feature projections operate in feature space and do not rotate physical XYZ. Mean/spread uses a fixed normal-quantile basis as a control, without asserting Gaussian physiology. The first study holds all four stored tensor sizes at 720 columns. Later studies keep the full original convolution base and append 144 columns.', ['docs/distribution_pooling_plan.md','docs/convolution_distribution_plan.md'],{diagram:'distribution'});
s('Distribution pooling on physical features',[
 'Empirical quantiles scored above the simpler controls in this study.',
 'Quantiles minus mean/spread: +0.0114, interval [-0.0526, +0.0762].',
 'The comparison did not establish a reliable gain.'
], 'All original beats and record medians were re-encoded. Four arms share signed DCT/physical features, training-only normalization, projections and feature-grid dimensions. Independently reconstructed 120 contexts and restored 840 classifiers, including 120 outer models. These compact physical-feature scores are not matched comparisons against the stronger convolution and waveform references. The primary contrast is quantiles versus moments, not whichever historical reference is lowest.', [R('distribution_pooling_v1'),'docs/distribution_pooling_findings.md'],{table:[['Matched pooling procedure','Ensemble macro F1'],['Median waveform','0.434441'],['Mean beat embedding','0.463013'],['Mean and spread','0.475709'],['Empirical quantiles','0.487114']],widths:[630,230],rowH:39});
s('Keeping the full convolution representation',[
 'We retained all 3,584 original features and added 144 distribution features.',
 'Quantiles minus spread: -0.0046. No positive quantile benefit.',
 'The spread result is a small, uncertain development observation.'
], 'This follow-up addresses loss of the stronger original representation. All beats were processed using the source transforms for their exact training partition. Independently restored 630 new and 210 source classifiers, with exact baseline probabilities. The separately labeled post-hoc spread minus baseline interval is [-0.016967,+0.037176]. Spread improved the ensemble point but lowered all three member scores, with mean member .538610 versus baseline .572860. It does not establish better individual models or independent superiority.', [R('convolution_distribution_v1'),'docs/convolution_distribution_findings.md'],{table:[['Intact convolution base','Ensemble macro F1'],['Original base','0.584266'],['Base + mean','0.581913'],['Base + mean and spread','0.594098'],['Base + quantiles','0.589485']],widths:[630,230],rowH:39});
s('Pooling learned beat embeddings',[
 'Frozen ResNet teachers encoded every beat into 64 features.',
 'Mean, spread and quantile summaries augmented the full convolution base.',
 'Learned quantiles did not improve the matched baseline.'
], 'All teachers are plain morphology teachers from their exact training partitions, with no encoder updates. Original inner-validation stopping remains inherited and disclosed. All 630 new and 210 source classifiers were restored, and all 120 teacher feature contexts and 120 teacher query probabilities were replayed. Mean .554973, spread .547867, quantiles .580895 versus baseline .584266. Quantiles minus spread +.033028 [-.011283,+.079062]; quantiles minus baseline -.003371 [-.060716,+.052488]. All nine added-feature member scores fell below their paired baseline.', [R('learned_distribution_v1'),'docs/learned_distribution_findings.md'],{table:[['Frozen learned summary','Ensemble macro F1'],['Original convolution base','0.584266'],['Base + learned mean','0.554973'],['Base + learned spread','0.547867'],['Base + learned quantiles','0.580895']],widths:[630,230],rowH:39});
s('Linear and nonlinear classifiers',[
 'Inner validation chose one shared representation for both classifier families.',
 'A nonlinear kernel did not improve the selected ensemble.',
 'The 0.603374 result comes from the linear selection procedure.'
], 'We compared class-balanced linear kernel ridge and linear-plus-Gaussian kernel ridge on four fixed representations. The representation selection is shared symmetrically, preventing each family from selecting a separate representation for the primary comparison. All eight fixed arms, all three seeds and 2,400 model restorations remain reported. Selected nonlinear minus linear is -.019441 [-.048244,+.005086]. Mean member F1 rises .562847 to .573104 while ensemble F1 falls, so those endpoints must remain distinct. Nonlinear search used four candidates versus two linear candidates.', [R('distribution_kernel_v1'),'docs/distribution_kernel_findings.md'],{table:[['Shared inner-selected representation','Ensemble macro F1'],['Linear kernel ridge','0.603374'],['Linear + Gaussian kernel ridge','0.583933']],widths:[630,230],rowH:47});
s('Corrections to frozen baseline scores',[
 'We kept the original convolution predictions and learned a small correction.',
 'Corrections used convolution summaries or learned spread/quantile features.',
 'Every final ensemble retained the original predictions and 0.584266 score.',
 'Inner validation found no useful correction under the tested settings.'
], 'The score-correction study isolates whether a regularized residual can improve the original decision scores without replacing the base. Four procedures selected correction off or penalty inside training folds. All four ensembles reproduce the original 164 labels. Independently verified 120 contexts, 210 source restorations, 1,440 active inner and 30 active outer correction fits. The result does not prove residual correction can never help. Learned features inherit original inner-validation stopping.', [R('distribution_score_correction_v1'),'docs/distribution_score_correction_findings.md']);
s('Direction and timing within a heartbeat',[
 'A lag tensor captured symmetric relations and signed directional changes.',
 'Magnitude and scrambled-sign controls tested what direction contributes.',
 'Signed direction did not beat the matched baseline or scrambled control.'
], 'We used four R-relative windows and four positive lags. Symmetric entries are shared. The antisymmetric component receives original signs, absolute magnitude or balanced per-beat scrambled signs. Time reversal preserves symmetric entries and flips directional signs in the integrity tests. All 55,863 beats, 120 contexts, 210 source models and 630 new classifiers were independently replayed. Signed minus magnitude +.013444 [-.027522,+.054415], signed minus scrambled -.015269 [-.066919,+.034030].', [R('lag_direction_tensor_v1'),'docs/lag_direction_tensor_findings.md'],{table:[['Tensor augmentation','Ensemble macro F1'],['Original convolution base','0.584266'],['Direction magnitude','0.517919'],['Scrambled direction signs','0.546633'],['Original signed direction','0.531364']],widths:[630,230],rowH:39});
s('Subtype relations and local timing',[
 'Subtype graph regularization reached 0.591424 versus ridge 0.584266.',
 'Its gain remained uncertain: +0.0072, interval [-0.0129, +0.0283].',
 'A separate local-timing audit failed its patient-coverage requirement.',
 'We deferred the timing tensor before fitting a classifier.'
], 'The graph uses clinical subtype adjacency with HC isolated, compared with a relabeled graph under the same strength grid. It does not add independent anatomical labels. The local-timing audit compares two disjoint same-record median references per query group. Only 20,8,16 of 164 patients met 10% beat support in the three windows, below the required 33 patients. Shape-only controls produced apparent shifts, so offsets cannot be interpreted as clinical timings. No timing-classifier F1 was measured.', ['docs/subtype_graph_findings.md','docs/local_timing_audit_findings.md']);
s('Supervised PTB-XL transfer completed',[
 'Source diagnostic learning initialized the encoder. Target heads stayed fresh.',
 'The primary transfer gain was uncertain: +0.0346, interval [-0.0465, +0.1138].',
 'All 630 fits and 90 outer models passed independent replay.'
], 'This is the study that was active during the previous presentation. Certain-source pretraining is the primary arm; intermediate-positive is sensitivity. Source objectives use one pretraining seed, while target fine-tuning retains seeds42,123,2026. Input at target inference is measured XYZ only. The historical .584570 and .603374 procedures are not this study\'s matched scratch control. Recording screening found no retrieved copies, but cross-dataset patient identity remains unresolved. Source AP is not target macro F1.', [R('neural_supervised_transfer_v1'),'docs/supervised_transfer_findings.md'],{table:[['Matched target procedure','Ensemble macro F1'],['Scratch','0.538736'],['Certain-source initialization','0.573361'],['Intermediate-positive initialization','0.549364']],widths:[630,230],rowH:43});
s('Native VCG and ECG-assisted learning',[
 '124 additional native-XYZ patients supplied unlabeled pretraining signals.',
 'A paired ECG teacher supplied an alternative training signal.',
 'Both pilots retained measured XYZ alone at target inference.',
 'Neither procedure demonstrated an ensemble gain over matched scratch.'
], 'Native pretraining used124 extra patients,156 records,17573beats after the stated copy/QC screen. Target-only reconstruction .454298, expanded .518364, scratch .601801. Expanded minus target-only +.064066 [-.003292,+.133664], but expanded minus scratch -.083436 [-.160972,-.007201]. ECG-assisted paired .564293 versus shuffled .529345 and scratch .601801. Pairing advantage +.034948 [-.051853,+.121214] remains uncertain. Each study independently replayed90 models and30 matched groups. More source data did not automatically improve the fixed recipe.', ['docs/vcg_pretraining_pilots_findings.md',R('neural_vcg_native_pretraining_v1'),R('neural_vcg_ecg_bridge_v1')],{diagram:'bridge'});
s('ACS-ECG artery supervision',[
 'We pretrained on native occlusion-MI labels, with or without artery labels.',
 'Artery labels stayed in the source task. They did not relabel PTB subtypes.',
 'Artery supervision added no demonstrated benefit over the matched controls.'
], 'OMI refers to occlusion myocardial infarction in the ACS source. Both source models share inputs, initialization and exposure. Only the encoder transfers. The target uses measured XYZ, fresh six-class heads and fixed eight-epoch training. All90 target models were restored. Primary OMI+artery minus OMI +.010341 [-.066726,+.085396]. OMI+artery minus scratch -.021071 [-.095589,+.052806]. Cross-dataset patient identity remains unresolved despite recording-copy screening.', [R('neural_acs_artery_transfer_v1'),'docs/acs_artery_transfer_plan.md'],{table:[['Matched target procedure','Ensemble macro F1'],['Scratch','0.601801'],['OMI source pretraining','0.570389'],['OMI + artery source pretraining','0.580730']],widths:[630,230],rowH:44});
s('Controlled changes from STAFF III',[
 'Source ECG pairs captured before and during annotated balloon occlusion.',
 'We compared learning the change with learning the mean of the same pair.',
 'Change minus mean: -0.0155, interval [-0.0834, +0.0517].'
], 'STAFF III source vessel labels describe the vessel under balloon occlusion, rather than the six PTB infarct-location classes. Source ECG uses Mason-Likar limbs and Kors-derived XYZ. The source protocol verifies timing and excludes specified lead-error patients. Change and mean see exactly the same paired windows. All90 target outer fits replayed. Change -.062211 versus scratch, interval[-.139503,+.015059]. This extension demonstrates a controlled source-learning implementation without an established localization gain.', [R('neural_staffiii_pair_transfer_v1'),'docs/staffiii_pair_transfer_plan.md'],{table:[['Matched target procedure','Ensemble macro F1'],['Scratch','0.601801'],['Mean-pair source pretraining','0.555118'],['Change-pair source pretraining','0.539589']],widths:[630,230],rowH:44});
s('Partial regional labels from native PTB',[
 '13 outside-target patients contributed positive region mentions.',
 'Unmentioned regions stayed masked. Shuffling preserved tag counts.',
 'True labels did not improve the matched scratch ensemble.'
], 'Source audit reproduced34records4644beats14candidatepatients, of which13patients4215beats had shared observed tags. Header mentions are not independently adjudicated anatomical evidence. True versus shuffled +.006940 [-.042964,+.058767]; true versus scratch -.025914 [-.085288,+.035006]. The no-self shuffle changed10of13tag sets while matching per-update counts and source waveforms. All120fits30groups replayed, and all30scratch probability blocks matched the earlier pilot exactly.', [R('neural_native_partial_labels_v1'),'docs/native_partial_labels_findings.md'],{table:[['Matched learning procedure','Ensemble macro F1'],['Scratch','0.601801'],['Target descriptors','0.592327'],['Shuffled source mentions','0.568947'],['True source mentions','0.575887']],widths:[630,230],rowH:39});
s('Temporal pooling and subtype classifiers',[
 'R-aligned convolution windows tested whether local position helps.',
 'Aligned and shuffled windows scored almost identically.',
 'A hierarchical classifier first estimated MI probability, then its subtype.',
 'Its gain over flat classification remained uncertain.'
], 'R-relative convolution retains3584base features and adds512columns. Whole-beat .567233, shuffled .577459, aligned .577238. Aligned minus whole +.010004 [-.025707,+.046669]; aligned minus shuffled -.000221 [-.041515,+.039815]. All630newfits210baselines120contexts replayed. Probabilistic subtype flat .562819, hierarchical .586828, historical selector .603374. Hierarchical minus flat +.024010 [-.001207,+.056060]. Both use the same selected representation and matched C grids, with all600procedures replayed. Hierarchy adds no new labels.', [R('r_relative_convolution_v1'),R('probabilistic_subtype_v1')],{table:[['Completed comparison','Ensemble macro F1'],['R-aligned / shuffled pooling','0.577238 / 0.577459'],['Hierarchical / flat classifier','0.586828 / 0.562819']],widths:[590,270],rowH:46});
s('Patient loss and model averaging',[
 'Training on average patient probabilities did not help its grouped-batch control.',
 'Equal averaging across four convolution readouts scored 0.586053.',
 'A fixed convolution-waveform blend scored 0.582209.',
 'Complementary errors did not automatically produce a better average.'
], 'Patient-probability study uses matched eight-beat bags for every arm: beat CE .510134, patient CE .469596, hybrid .460082. The sampler differs from the historical .601801 waveform recipe. All90models30groups replayed. Four-readout average minus historical selector -.017321 [-.042859,+.005133], zero fits. In the waveform/convolution comparison, both correct93, convolution-only19,waveform-only16,bothwrong36.21shared errors carry the same wrong label. Equalblend minus convolution -.021165 [-.087770,+.043554], no fitted weighting or routing.', [R('neural_patient_probability_loss_v1'),'results/convolution_readout_average_v1/REPORT.md','results/model_error_comparison_v1/REPORT.md']);
s('Recording differences became a key question',[
 '57 of 93 patients with repeated recordings received different record-level labels.',
 'Most remaining convolution errors exchange one MI subtype for another.',
 'We audited examination choice, beat count, signal quality and recording dates.',
 'These diagnostics identify questions for review, without changing target labels.'
], 'The research audit reproduced .603374convolution and .601801waveform, checked2400fit memberships and391rawtargetheaders. Of52convolution errors44arewithin-MI confusions. The 57/93 statistic concerns model prediction disagreements across a patient\'s recordings, not demonstrated inconsistent clinical diagnoses. Uniform one-record expected-count sensitivity .5683 is distinct from choosing the earliest dated recording. Subsequent slides report separately registered audits. Original labels remain frozen and clinical review is pending.', ['docs/patient_independent_research_audit.md','docs/path_to_080_macro_f1.md'],{diagram:'visits'});
s('One earliest recording changes the endpoint',[
 '145 patients had a usable date and a unique earliest recording.',
 'The same frozen model scored 0.586275 using all their beats.',
 'Using only their earliest recording scored 0.515049.',
 'Difference: -0.0712, interval [-0.1470, +0.0036].'
], 'This is a zero-fit archival sensitivity, replaying all30heldoutprediction blocks.13patients have an earliest-date tie,6have a missing or invalid date. Date eligibility remains descriptive, not outcome-selected filtering. The comparison uses exactly the same145patients. All164all-record score .601801 remains separate. The earliest archived record does not necessarily coincide with acute clinical presentation or untreated disease. Equal record weighting left all164ensembleclassdecisions unchanged.', ['results/index_recording_audit_v1/REPORT.md','docs/index_recording_audit_plan.md'],{chart:{labels:['All records, same 145 patients','Earliest record, same 145'],values:[.586275,.515049],max:.7},chartH:210});
s('Matching the number of beats',[
 '79 patients had two usable dated recordings.',
 'Matched sampling compared earliest, later and half from each recording.',
 'Later recordings retained the higher point score at the same beat budget.'
], 'There are256draws per patient,64beats for78patients and50beats for1patient. Scores are macroF1 of expected confusion counts, not average per-draw F1, although both are in the report. Later-earliest +.095882 [-.037926,+.230258]; half-earliest +.084854 [-.017018,+.192643]. The intervals remain broad. The experiment does not prove a clinical disease mechanism, and it does not make the79patients representative of all164. Median archival interval4days.', ['results/matched_recording_beats_v1/REPORT.md','docs/matched_recording_beats_plan.md'],{table:[['Same 79 patients and matched beat budget','Macro F1'],['Earliest recording','0.461602'],['Later recording','0.557485'],['Half from each','0.546457']],widths:[630,230],rowH:43});
s('Signal quality did not explain the gap',[
 'Raw decoding reproduced cached beats in all 158 paired recordings.',
 'We checked drift, ADC extremes, R-peak disagreement and RR outliers.',
 'Drift was lower on average in later records, with mixed patient-level changes.',
 'The measured quality proxies showed no clear explanation for improved predictions.',
 'Independent clinical and timing review remains pending.'
], 'Zero new fits. Decreased drift versus improved correctness rho-.022[-.235,+.185]. R-disagreement association rho+.059[-.175,+.290]. No ADC rail hits occurred. Mean later-minus-earliest drift -.03120mV[-.06724,-.00041], with41lower and38higher. These measures do not cover all acquisition quality or disease evolution.24matched blindedreviewpairs are prepared locally, with the label key held separately. Do not publish private review material or imply the review has happened.', ['results/paired_recording_quality_v1/REPORT.md','docs/paired_recording_quality_plan.md']);
s('Learning earlier signals from later recordings',[
 'Cross-fitted teachers supplied later-record targets to earliest-record students.',
 'A same-class shuffled teacher control tested whether correct pairing matters.',
 'The paired gain over supervised training remained negligible and uncertain.'
], 'Teachers hold out both donor and recipient for the same-class shuffle. The145datedpatients remain primary. Supervised .521335, paired .523769, shuffled .502072. Paired-supervised +.002434[-.079058,+.085427]; paired-shuffled+.021697[-.062848,+.102840]. Every paired student seed scored below its corresponding supervised seed even though the ensemble point is slightly higher.30crossfittedteachers90students120fits30matchedgroups all independently replayed. No inference on later records for the earliest-record endpoint.', [R('neural_later_recording_distillation_v1'),'docs/later_recording_distillation_plan.md'],{table:[['Earliest-record training, 145 patients','Ensemble macro F1'],['Supervised reference','0.521335'],['Correct later-record teacher','0.523769'],['Shuffled later-record teacher','0.502072']],widths:[630,230],rowH:44});
s('Longer training did not reliably help',[
 'We compared 8, 16 and 32 epochs under the earliest-record endpoint.',
 'Inner-selected duration scored 0.509127 versus fixed-eight 0.521335.',
 'The fixed-16 point is descriptive and cannot replace the declared selector.',
 'Training loss kept falling while patient-level gains remained uncertain.'
], 'On145datedpatients,fixed16 .552757,fixed32 .516732,innerselected .509127. Primary innerselected-fixed8-.012208[-.082185,+.061027]. Fixed16-fixed8+.031422[-.039516,+.100476]secondary. All120trajectories360checkpoints and30baseline blocks replayed. Innerselectcounts8/8/14over8/16/32epochs areoverlappingdecisions. Mean outertrainingloss .358157 at8,.104954at16,.044369at32. Earliestconvolutionfollowup also completed:combined .478122,DCT .461037,ResNet .521335,with420ridgefits120contextsreplayed.', [R('neural_earliest_training_duration_v1'),R('earliest_record_convolution_v1')],{table:[['Earliest-record duration, 145 patients','Ensemble macro F1'],['Fixed 8 epochs','0.521335'],['Fixed 16 epochs','0.552757'],['Fixed 32 epochs','0.516732'],['Inner-selected duration','0.509127']],widths:[630,230],rowH:39});
s('Testing the frozen ResNet readout',[
 'We compared the original head with new patient-level ridge classifiers.',
 '64 features summarize the final embedding. 256 retain temporal bins.',
 'Retaining 256 features did not improve the 64-feature procedure.'
], 'All encoders come from their exact training partition and fixed epoch8, with zero encoder updates. Patient training only fits the readout scaling and inner alpha[1,100].145datedpatientsoriginal .521335,meanlogits .517351,ridge64 .535691,ridge256 .501020.256minus64-.034671[-.094647,+.021065].64minusoriginal+.014356[-.046053,+.080129]secondaryuncertain. Independently restored120encoders420ridgeheads.164sensitivityretains one selected record perpatient,notall-recordinference.', [R('frozen_patient_readout_v1'),'docs/frozen_patient_readout_findings.md'],{diagram:'readout'});
s('Ridge, centroid and Mahalanobis distance',[
 'All three classifiers used the same frozen 64-dimensional embeddings.',
 'Centroid distance ignores covariance. Mahalanobis uses a regularized covariance.',
 'The covariance-aware head did not improve the matched ridge reference.'
], 'This is the latest completed experiment, finished19November13:15IST. All420newdistanceheads,210sourceridgeheads,120featurecontexts and104identitycontrols replayed. Zeroencoderupdates,all1444sourcefilesunchanged. PrimaryMah-ridge-.028510[-.088700,+.031304],Mah-centroid-.000589[-.041155,+.038686]. AMIF1drops .4286to.2963. All3Mahmemberseeds arebelowpairedridge. The164sensitivity uses one-selected-record andscores .546530/.520508/.519400.376main-suite tests andactualinneronlysmokepassedbeforelaunch. No model run is queued behind this study.', [R('frozen_distance_readout_v1'),'docs/frozen_distance_readout_findings.md'],{table:[['Earliest-record classifier, 145 patients','Ensemble macro F1'],['Ridge','0.535691'],['Nearest class centroid','0.507770'],['Regularized Mahalanobis','0.507181']],widths:[630,230],rowH:44});
s('What the experiments tell us',[
 'Richer summaries sometimes raise point scores, with uncertain matched gains.',
 'Changing the classifier alone has not resolved the difficult MI subtypes.',
 'Recording choice changes performance and deserves independent review.',
 'Data diversity, regional label evidence and acquisition differences remain plausible bottlenecks.',
 'We have not established a dataset ceiling or a route guaranteed to reach 0.80.'
], 'These are evidence-informed interpretations, not a causal diagnosis that the problem is definitely data or definitely tensors. Distribution and kernel studies do not exhaust possible architectures. Current robust reference procedures remain separate. The scientific next step combines a bounded source feasibility decision with clinical review and an external measured-XYZ cohort plan, rather than unconstrained outer-score search.', ['docs/patient_independent_research_audit.md','docs/next_readout_experiments.md','docs/open_dataset_review_20261119.md']);
s('Additional data: separate research roles',[
 'Regional source labels: Shandong ECG statements.',
 'Patient diversity: MIMIC-IV-ECG.',
 'Manual P/QRS/T boundaries: LUDB.',
 'Controlled simulation: MedalCare-XL.',
 'These sources do not directly supply our six-class measured-Frank cohort.'
], 'This summarizes the19Novemberrelease audit, not a new liveavailability search. ECG can be mapped to estimated XYZ using the existing Kors conversion, but estimatedXYZ isnotmeasuredFrank andcannotcreate missingclinicalregionlabels. MIMICwaveformsareopenwhilelinkedclinicalnotes have separateaccess. MedalCare adds syntheticexamples,notrealpatients. LUDB supports a delineation question,not immediateMIclasscounts. Explicitly label ECG-assistedtraining for a sourceextension. IllustrativeECGimage credit:YitzhakNat,NormalEKGwave,WikimediaCommonsCCBY-SA4.0,unchanged.', ['docs/open_dataset_review_20261119.md','https://commons.wikimedia.org/wiki/File:Normal_EKG_wave.svg','https://creativecommons.org/licenses/by-sa/4.0/'],{image:'output/pptx/assets/progress_illustrated_20261117/ecg_cycle_yitzhaknat.png',imageBox:[610,110,300,240],sideLeft:true});
s('Shandong metadata: what we verified',[
 '25,770 ECGs from 24,666 patients in the published metadata.',
 '260 records from 242 patients carry one or more MI-region statements.',
 'The dictionary includes anterior, inferior and anteroseptal statements.',
 'Most describe old MI. ALMI and ILMI equivalents are missing.',
 'Only metadata are downloaded. Waveform QC and compatibility remain pending.'
], 'Downloadedmetadataanddictionarytotal1,001,058byteswithpublisherMD5andlocalSHA256checks. Nativepatientregioncounts47anterior117inferior82anteroseptal7extensiveanterior overlapandmustnotbesummed. Tenrecords carrymultipleMIregionstatements. No sixclassmapping,noincomingpatientadditions,noidentityclearance,nosourcefits. Fullsignalarchive2,281,799,680bytes,about2.28GB. The13,905normalstatementonlyrecordsarenotindependentlyreviewedHC. Preserve these native multi-label statements and age modifiers for any later source task.', ['docs/open_dataset_review_20261119.md','results/open_dataset_screen_20261119/sph_metadata_summary.json']);
s('Implementation and validation delivered',[
 'Versioned protocols, frozen folds and matched controls for each study.',
 'Saved models, resumable runners and local progress monitors.',
 'Independent feature reconstruction and model-prediction replay.',
 '376 tests passed before the latest classifier study.',
 'Partner-cohort intake is prepared. Real data and clinical review are pending.'
], 'A fresh clone lacks ignoredwaveforms,caches,weightsandpatientpredictions. WindowssetupandCodexhandoffdocsareavailable,butWindowsnumericaltraininghasnotbeenvalidated. Latestdistanceverificationiscompletedandstopped;oldermultiviewsweepremainspaused.Noactiveorqueuednewtrainingatthisevidencesnapshot. Partnerintakesmokeusedsyntheticsignals,notacquiredpatients. SmallMIImagingfixturecontains2publicreferencecases,6.48MB,notanexternalvalidationset. Do not count restorationchecksasnewclinicaldata.', ['docs/CODEX_HANDOFF.md','docs/frozen_distance_readout_findings.md','docs/measured_vcg_partner_cohort.md','docs/WINDOWS_SETUP.md']);
s('Next research decisions',[
 'Complete the blinded review of regional evidence and recording timing.',
 'Audit Shandong signal units, lead order, quality and native label meaning.',
 'Register one bounded source-learning comparison before fitting.',
 'Seek independently labeled measured XYZ patients through a clinical partner.',
 'Reserve external patients before selecting the next model.'
], 'These are proposednextsteps,notactiveexperimentsorauthorizedmessages. Feasibilityintaketarget20to30patients is a first technical/labelpilot,not a sufficientconfirmatorysamplesize. Louvain,CSE,Helsinkiandlocalclinicalroutesremainrequestleadswithcustody,rightsandlabelcoverageunconfirmed.Draftsremainunsent. NewECGsourcedata mustremainseparatefromtargetpatients.Label-equivalence,identityandcompute-budgetdecisionsprecedetraining. Independentdatarequiresaprespecifiedendpointandmodelhashes.', ['docs/data_acquisition_action_plan_20261119.md','docs/measured_vcg_partner_cohort.md','docs/open_dataset_review_20261119.md']);
s('Progress since the previous presentation',[
 'Completed the transfer and distribution studies proposed last time.',
 'Built and tested further tensor, supervision and classifier alternatives.',
 'Reached approximately 0.60 on the existing all-record development task.',
 'Identified recording selection and regional evidence as priorities for review.',
 'Prepared the next data and validation pathway. The 0.80-0.90 target remains open.'
], 'Maintalkends here. Explainprogressasimplementedmethods,controlledcomparisonsandbetterunderstandingoftheendpoint. AvoidclaimingthatmoreexperimentsaloneestablishSOTA. Thehighestall-recordpointsarenotindependentconfirmation,whiletheearliest-recordscoresareadifferentendpoint. Usetheappendixforallscores,uncertainty,seeds,classresultsanddatasetdetails. No small footer text is used in this deck.', ['docs/CODEX_HANDOFF.md','docs/patient_independent_research_audit.md']);
backup('distribution comparisons',[
 ['Representation','Base','Mean','Spread','Quantiles'],
 ['Physical features','0.4344','0.4630','0.4757','0.4871'],
 ['Full convolution base','0.5843','0.5819','0.5941','0.5895'],
 ['Learned embedding additions','0.5843','0.5550','0.5479','0.5809'],
 ['Nonlinear convolution readout','0.5840','0.5995','0.5959','0.5921']
], 'Eachrowisaseparatematchedstudy,all164patientsandthreeseedensembles.Baseforphysicalfeaturesismedianwaveform.Baseforothersisfullconvolutionmorphology.Linearintactbasequantilesminusspread-.004613[-.013297,0].Learnedquantilesminusspread+.033028[-.011283,+.079062].Physicalquantilesminusspread+.011405[-.052573,+.076160].Fixednonlinearcomparisonsallincludezero.Thereisnocommoncausalcomparisonacrossrows.', [R('distribution_pooling_v1'),R('convolution_distribution_v1'),R('learned_distribution_v1'),R('distribution_kernel_v1')],{widths:[340,130,130,130,130],rowH:58,font:20});
backup('kernel members and ensembles',[
 ['Representation','Linear mean','Nonlinear mean','Linear ensemble','Nonlinear ensemble'],
 ['Morphology','0.5729','0.5845','0.5843','0.5840'],
 ['Mean','0.5482','0.5621','0.5819','0.5995'],
 ['Mean + spread','0.5386','0.5636','0.5941','0.5959'],
 ['Quantiles','0.5495','0.5755','0.5895','0.5921'],
 ['Shared inner selection','0.5628','0.5731','0.6034','0.5839']
], 'MeanmemberaveragesthreeF1scores.Ensembleaveragesprobabilitiesbeforemakingclassdecisions.Allseeds42,123,2026remain.Sharedinnerrepresentationchoicecounts16/3/4/7baseline/mean/spread/quantilesareoverlappingtrainingdecisions.Nonlinearprimaryminuslinear-.019441[-.048244,+.005086].', [R('distribution_kernel_v1')],{widths:[260,140,150,155,155],rowH:55,font:19});
backup('source learning comparisons',[
 ['Source recipe','Matched control','Candidate','Ensemble F1 control / candidate'],
 ['PTB-XL diagnostics','Scratch','Certain source','0.5387 / 0.5734'],
 ['Native reconstruction','Target-only','Expanded source','0.4543 / 0.5184'],
 ['Paired ECG teacher','Shuffled','Correct pairs','0.5293 / 0.5643'],
 ['ACS-ECG','OMI only','OMI + artery','0.5704 / 0.5807'],
 ['STAFF III','Pair mean','Pair change','0.5551 / 0.5396'],
 ['Partial native tags','Shuffled','True tags','0.5689 / 0.5759']
], 'Theseareseparatestudieswithdifferentrecipesandmatchedcontrols.Allgainintervalsinthistableincludezero.NativeexpandedpretrainingandbothpairedECGsourcearmsalsocomparedagainst.601801scratchanddidnotimproveit.PTB-XLscratchhasadifferentselectedtrainingrecipe.OrigsourceAPandAUROCarenotthistargetmetric.Sourceidentityremainsunresolvedwhereapplicable.', [R('neural_supervised_transfer_v1'),'docs/vcg_pretraining_pilots_findings.md',R('neural_acs_artery_transfer_v1'),R('neural_staffiii_pair_transfer_v1'),R('neural_native_partial_labels_v1')],{widths:[240,170,185,265],rowH:50,font:20});
backup('other matched readouts',[
 ['Procedure','Control F1','Candidate F1','Candidate minus control interval'],
 ['Subtype graph / ridge','0.5843','0.5914','[-0.0129, +0.0283]'],
 ['Signed / magnitude tensor','0.5179','0.5314','[-0.0275, +0.0544]'],
 ['R-aligned / whole beat','0.5672','0.5772','[-0.0257, +0.0467]'],
 ['Hierarchical / flat','0.5628','0.5868','[-0.0012, +0.0561]'],
 ['Patient / beat CE','0.5101','0.4696','[-0.1037, +0.0221]'],
 ['Equal blend / convolution','0.6034','0.5822','[-0.0878, +0.0436]']
], 'Allscoresarethreeseedensemblesonthe164patientall-recordendpoint,butprotocolsandmatchedcontrolsareseparate.Intervalsaredescriptivepairedpatientbootstrapconditionedonfixedpredictions,anddonotcorrectforrepeateddevelopmentorrefitting.Independentconfirmationremainsneeded.', ['docs/subtype_graph_findings.md','docs/lag_direction_tensor_findings.md',R('r_relative_convolution_v1'),R('probabilistic_subtype_v1'),R('neural_patient_probability_loss_v1'),'results/model_error_comparison_v1/REPORT.md'],{widths:[310,130,140,280],rowH:49,font:20});
backup('matched prediction errors',[
 ['True class','Both correct','Convolution only','Waveform only','Both wrong'],
 ['AMI','4','1','5','7'],['ALMI','5','2','1','8'],['ASMI','16','3','2','6'],['IMI','12','7','3','7'],['ILMI','10','3','4','6'],['HC','46','3','1','2'],['Total','93','19','16','36']
], 'Thetwosourceensemblesexactlyreproduced.603374and.601801.21of36sharedmisseshavethesamewronglabel.Nofittedcombinerorperclassroutingwastrained.Thiscomplementaritydiagnosticdoesnotauthorizeoraclechoicesbetweenmodelsbycorrectness.', ['results/model_error_comparison_v1/REPORT.md'],{widths:[180,160,185,180,155],rowH:44,font:21});
backup('all-record class results',[
 ['Class','Patients','Selected convolution F1','Fixed-eight waveform F1'],
 ['AMI','17','0.3448','0.5294'],['ALMI','16','0.5000','0.4286'],['ASMI','27','0.6333','0.6429'],['IMI','29','0.6129','0.5357'],['ILMI','23','0.6047','0.5957'],['HC','52','0.9245','0.8785']
], 'Same164patients,frozenfolds,andthreeseedensembles.Classoutcomesaredescriptive.Noclassrouting,weightselectionorlabelchangesfollowfromthistable.AMImeansanteriorMIinthisproject.Thefewclasspatientsdrivewideuncertainty.', [R('distribution_kernel_v1'),R('neural_vcg_native_pretraining_v1')],{widths:[100,120,320,320],rowH:49,font:21});
backup('earliest-record endpoints',[
 ['145 dated patients','Ensemble macro F1','Interpretation'],
 ['Frozen all-record model, all visits','0.586275','Archival sensitivity control'],
 ['Same model, earliest visit only','0.515049','No new training'],
 ['Supervised earliest-trained model','0.521335','Fresh matched reference'],
 ['Paired later-record distillation','0.523769','Uncertain gain'],
 ['Inner-selected training duration','0.509127','No demonstrated gain'],
 ['Earliest-record convolution','0.478122','No demonstrated gain']
], 'Allrowshavethesame145datedpatientsbutdifferintrainingandinference,andmustnotbetreatedasonelearningcurve.All164all-record.601801isaseparatereference.Earliestarchivaldatedoesnotestablishclinicalpresentationtime.Theone-selected-recordsensitivityretains164patientsusingdeterministicdatefallbacks.', ['results/index_recording_audit_v1/REPORT.md',R('neural_later_recording_distillation_v1'),R('neural_earliest_training_duration_v1'),R('earliest_record_convolution_v1')],{widths:[400,175,285],rowH:49,font:20});
backup('frozen readout endpoints',[
 ['Procedure','145 earliest records','164 selected records'],
 ['Original probability head','0.521335','0.532437'],['Mean logits','0.517351','0.529068'],['Ridge, 64 features','0.535691','0.546530'],['Ridge, 256 features','0.501020','0.525410'],['Nearest centroid, 64 features','0.507770','0.520508'],['Mahalanobis, 64 features','0.507181','0.519400']
], 'Bothcolumnsuseone recordingperpatient.The164columnisnotall-recordinference.145patientsremainprimary.Readout256minus64-.034671[-.094647,+.021065].Mah-ridge-.028510[-.088700,+.031304].Allheadsuselocalexact-partitionencodersandtraining-onlynormalization.Nocombinedcrossstudywinnerselectionorouterthresholdtuning.', [R('frozen_patient_readout_v1'),R('frozen_distance_readout_v1')],{widths:[420,220,220],rowH:49,font:21});
backup('latest classifier by class',[
 ['Class','Patients','Ridge F1','Centroid F1','Mahalanobis F1'],
 ['AMI','15','0.4286','0.3704','0.2963'],['ALMI','14','0.4444','0.4444','0.4444'],['ASMI','25','0.4815','0.4528','0.4528'],['IMI','29','0.5263','0.5246','0.5424'],['ILMI','22','0.5000','0.4211','0.4500'],['HC','40','0.8333','0.8333','0.8571']
], '145datedearliest-recordpatients.Allthreeclassifiersreceivethesame64features.Mahalanobisusesasharedshrinkagecovariancematrixfitonlyontrainingpatients.Innerlambda1/.9/.5choices14/12/4over30foldseedcontextsarenotindependentreplications.Noexperimenttunesclassroutingfromtheseresults.', [R('frozen_distance_readout_v1')],{widths:[110,130,195,205,220],rowH:49,font:21});
backup('data sources and storage',[
 ['Dataset','Released examples','Research role','Published download size'],
 ['Shandong','25,770 ECGs','Regional statements','2.28 GB signal archive'],
 ['MIMIC-IV-ECG','About 800,000 ECGs','Patient diversity','33.8 GB ZIP'],
 ['MedalCare-XL','16,900 synthetic ECGs','Simulation','9.32 GB ZIP'],
 ['LUDB','200 ECGs','Manual boundaries','13.8 MB ZIP'],
 ['Chapman/Ningbo','45,152 ECGs','Generic pretraining','2.3 GB ZIP']
], 'Thisisthereviewed19Novemberinventory.Downloadsizesarenotexpandedcachesortrainingbudgets.MIMIC90.4GBuncompressed.LUDB23.6MBuncompressed.MedalCareadds0realpatientsandhasartery/transmuralitylabels,notoursixclassendpoint.SPHmostlyoldMIwithmissingALMI/ILMIequivalents.Metadata-onlyscreenhasnopassedsignalQC.AllECGextensionsrequireclearsource-targetacquisitionlabelingandidentitypolicy.', ['docs/open_dataset_review_20261119.md'],{widths:[185,215,220,240],rowH:56,font:20});
backup('verification coverage',[
 ['Completed scope','Independent replay'],
 ['Supervised PTB-XL transfer','630 fits and 90 outer models'],
 ['Physical beat distributions','840 classifiers and all original beats'],
 ['Convolution distribution additions','630 new + 210 source classifiers'],
 ['Linear / nonlinear kernels','2,400 classifiers'],
 ['Frozen patient readouts','120 encoders and 420 ridge heads'],
 ['Frozen distance heads','420 new + 210 source heads, 104 controls']
], 'Countsbelongtoseparatestudiesandmayreusemodels.Theydonotrepresentindependentpatientsoracombinednewfitcount.Restorationrebuildsfeaturesorloadsfittedparameterstoproducepredictionsandrepeatselections,asperstudyscope.Latest376testsmacOSverificationisrecordedbeforethefinalstudy.Windows remainsunvalidated.No numerical artifactsorprivatepredictionsareembeddedinthepresentation.', [R('neural_supervised_transfer_v1'),R('distribution_pooling_v1'),R('convolution_distribution_v1'),R('distribution_kernel_v1'),R('frozen_patient_readout_v1'),R('frozen_distance_readout_v1')],{widths:[410,450],rowH:49,font:21});
s('Appendix: selected references',[
 'Zhang et al. Automated Localization of Myocardial Infarction From Vectorcardiographic via Tensor Decomposition. IEEE TBME, 2023.',
 'Nesaragi et al. Correlation matrix-based tensor decomposition for early prediction of sepsis. 2021.',
 'Tan et al. MultiRocket, 2022. Dempster et al. HYDRA, 2023.',
 'Projected distribution pooling: PSWE, NeurIPS 2021, and FSW, ICLR 2025.',
 'PTB, PTB-XL, STAFF III and LUDB: official PhysioNet releases.',
 'Exact experiment protocols, source publications and image credits are in presenter notes.'
], 'PrimarypaperDOI10.1109/TBME.2022.3202962.SepsispaperDOI10.1016/j.bbe.2021.06.009.Thesearemethodmotivations,notclaimsthatouradaptationsreproduceallupstreamsettingsorobtaintheirbenchmarkperformance.Methodattributionsandlimitationsareineachplan.Referencedeckandallscientificsourcefilesremainunchanged.PublicsignalfigureisprojectgeneratedfromPTB.WikimediaECGbyYitzhakNatCCBY-SA4.0unchanged.', ['https://doi.org/10.1109/TBME.2022.3202962','https://doi.org/10.1016/j.bbe.2021.06.009','https://papers.nips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html','https://proceedings.iclr.cc/paper_files/paper/2025/hash/3dbb8b6b5576b85afb3037e9630812dc-Abstract-Conference.html','https://physionet.org/content/ptbdb/1.0.0/','https://physionet.org/content/ptb-xl/1.0.3/','https://physionet.org/content/staffiii/1.0.0/','https://physionet.org/content/ludb/1.0.1/','https://commons.wikimedia.org/wiki/File:Normal_EKG_wave.svg'],{backup:true,font:21});

// Presenter notes use complete prose. Raw reports retain every numerical detail.
const notesByTitle={
 'Partial regional labels from native PTB': 'The source audit reproduced 34 recordings and 4,644 beats from 14 candidate patients. Thirteen patients had a shared observed region tag and supplied 4,215 supervised beats. Header mentions have not received independent anatomical adjudication. True minus shuffled gave +0.006940 with interval [-0.042964,+0.058767]. True minus scratch gave -0.025914 with interval [-0.085288,+0.035006]. The no-self shuffle changed 10 of 13 tag sets while preserving per-update tag counts and source waveforms. All 120 fits and 30 matched groups replayed. All 30 scratch probability blocks matched the earlier pilot exactly.',
 'Temporal pooling and subtype classifiers': 'The temporal study retained the 3,584-feature base and added 512 columns. Aligned minus whole-beat pooling gave +0.010004 with interval [-0.025707,+0.046669]. Aligned minus shuffled was -0.000221 with interval [-0.041515,+0.039815]. All 630 new fits, 210 source baselines and 120 contexts replayed. In the separate subtype study, hierarchical minus flat classification gave +0.024010 with interval [-0.001207,+0.056060]. The historical selected-linear reference was 0.603374. Both new classifiers used the same representation and matched regularization grids. All 600 procedures replayed. Hierarchy supplied no additional clinical labels.',
 'Patient loss and model averaging': 'The patient-loss study used the same eight-beat bags in every arm. Beat cross-entropy scored 0.510134, patient cross-entropy 0.469596 and their hybrid 0.460082. This sampler differs from the historical waveform recipe that scored 0.601801. All 90 models and 30 groups replayed. Averaging four convolution readouts gave 0.586053 versus the original selector at 0.603374. In the separate convolution-waveform comparison, both models were correct for 93 patients, convolution alone for 19 and waveform alone for 16. Both missed 36, with the same wrong label for 21. The fixed blend scored 0.582209. No weights or patient routing were learned from these outer predictions.',
 'Recording differences became a key question': 'The research audit reproduced convolution macro F1 of 0.603374 and waveform macro F1 of 0.601801. It checked 2,400 fit memberships and all 391 original target headers. Of 52 convolution errors, 44 exchange one MI subtype for another. The 57 of 93 statistic describes disagreement between model predictions across recordings, rather than proven inconsistency in clinical diagnoses. Uniform one-record expected-count sensitivity was about 0.5683. The later earliest-date analysis answers a different question. All original labels remain fixed and independent clinical review is pending.',
 'One earliest recording changes the endpoint': 'This audit fits no models and replays all 30 held-out prediction blocks. Thirteen patients have an earliest-date tie and six have a missing or invalid date, leaving 145 with a unique earliest recording. The main comparison uses those same 145 patients under both inference policies. The all-record result on all 164 patients remains 0.601801. Earliest archived recording does not necessarily mean first acute clinical presentation. Equal weighting across records left all 164 ensemble class decisions unchanged.',
 'Matching the number of beats': 'The matched experiment uses 256 draws per patient, with 64 beats for 78 patients and 50 for one patient. Displayed scores are macro F1 of expected confusion counts. Later minus earliest was +0.095882 with interval [-0.037926,+0.230258]. Half from each minus earliest was +0.084854 with interval [-0.017018,+0.192643]. The wide intervals prevent a firm conclusion. Median archival spacing was four days. Timing, treatment, injury evolution and acquisition could all differ. These 79 repeat-record patients are a separate descriptive subset.',
 'Signal quality did not explain the gap': 'The audit uses zero new fits. Lower drift versus improved correctness had Spearman rho -0.022 with interval [-0.235,+0.185]. The R-detector disagreement association was +0.059 with interval [-0.175,+0.290]. No ADC rail hits occurred. Mean later-minus-earliest drift was -0.03120 mV, with 41 patients showing lower and 38 higher drift. These proxies do not cover all acquisition quality or disease evolution. A private blinded packet of 24 matched review pairs is prepared, with its key separate. Independent clinical review has not happened yet.',
 'Learning earlier signals from later recordings': 'Teachers hold out both the donor and the recipient in the same-class shuffled comparison. The primary endpoint uses 145 dated patients. Paired minus supervised was +0.002434 with interval [-0.079058,+0.085427]. Paired minus shuffled was +0.021697 with interval [-0.062848,+0.102840]. Every paired student seed scored below its supervised counterpart despite the small increase in ensemble point score. All 30 cross-fitted teachers and 90 students replayed, with 30 matched groups. Earliest-record inference uses no later recording.',
 'Longer training did not reliably help': 'On the 145-patient endpoint, inner-selected duration minus fixed eight epochs gave -0.012208 with interval [-0.082185,+0.061027]. The fixed-16 result was a secondary descriptive contrast of +0.031422 with interval [-0.039516,+0.100476]. All 120 trajectories, 360 checkpoints and 30 original baseline blocks replayed. Mean training loss fell from 0.358157 at epoch eight to 0.044369 at epoch 32. A separate earliest-record convolution comparison scored 0.478122 for combined convolution, 0.461037 for DCT and 0.521335 for the matched waveform. It replayed 420 ridge fits and 120 contexts.',
 'Testing the frozen ResNet readout': 'Every readout uses the encoder for its own exact training partition, fixed at epoch eight. Encoder updates are zero. Training patients alone fit scaling and choose ridge alpha from 1 or 100 in the inner folds. On 145 dated patients, original probabilities score 0.521335, mean logits 0.517351, ridge64 0.535691 and ridge256 0.501020. The 256 minus 64 contrast is -0.034671 with interval [-0.094647,+0.021065]. The 64-feature gain over the original is secondary and uncertain. All 120 encoders and 420 ridge heads replayed. The 164-patient sensitivity retains one selected record per person.',
 'Ridge, centroid and Mahalanobis distance': 'The latest completed study finished on 19 November at 13:15 IST. All 420 new distance heads, 210 source ridge heads, 120 contexts and 104 identity controls replayed. All 1,444 protected source files stayed unchanged and encoder updates were zero. Mahalanobis minus ridge was -0.028510 with interval [-0.088700,+0.031304]. Mahalanobis minus centroid was -0.000589 with interval [-0.041155,+0.038686]. AMI F1 fell from 0.4286 to 0.2963. Every Mahalanobis member scored below its paired ridge member. The separate 164-patient selected-record sensitivity is in the appendix. No follow-on training is queued.',
 'Additional data: separate research roles': 'This is the verified 19 November release review. Standard ECG can supply estimated XYZ through the existing Kors conversion. That estimate differs from measured Frank XYZ and cannot create absent clinical region labels. MIMIC waveform access and linked clinical-note access have different conditions. MedalCare adds synthetic signals, with no new real patients. LUDB addresses delineation, rather than immediate regional MI class support. Any source experiment needs an explicit ECG-assisted label. ECG image credit: YitzhakNat, Normal EKG wave, Wikimedia Commons, CC BY-SA 4.0. The image is unchanged and illustrates an ECG, not a measured project recording.',
 'Shandong metadata: what we verified': 'The downloaded metadata and dictionary total 1,001,058 bytes and match publisher MD5 and local SHA-256 hashes. Native region supports are 47 anterior, 117 inferior, 82 anteroseptal and seven extensive-anterior patients. Supports overlap and must not be summed. Ten records carry multiple regional statements. The dictionary lacks ALMI and ILMI counterparts. No waveform download, quality audit, identity clearance, target-patient addition or new source fit has happened. The full signal archive is about 2.28 GB. A normal ECG statement does not establish an independently reviewed healthy control.',
 'Implementation and validation delivered': 'A fresh clone lacks the ignored raw data, feature caches, fitted weights and private patient predictions. Windows setup and handoff documents exist, while Windows numerical training remains unvalidated. The latest completed experiment is stopped, and the older multiview sweep remains paused. The partner intake smoke used synthetic signals and verifies software only. The downloaded MI Imaging format fixture contains two public reference cases totaling 6.48 MB. It is not an external validation cohort. Restoration counts do not add independent patients.',
 'Next research decisions': 'These are proposed steps. No new experiment or external message has been started by this presentation. A first 20-to-30-patient intake is a technical and label feasibility pilot, rather than a sufficient confirmatory cohort. Louvain, CSE, Helsinki and local clinical routes remain leads with access, custody, rights or regional-label coverage unresolved. Request templates remain unsent. Source label meaning, identity policy and laptop budget should be defined before training. A confirmatory external evaluation requires a reserved endpoint and frozen model selection.',
 'Progress since the previous presentation': 'End the main talk here. The project has progressed through implemented methods, controlled comparisons and a clearer definition of the intended examination. The strongest all-record point estimates remain development results, while the earliest-record studies use a different endpoint. The appendix contains detailed comparisons, class results and verification scope. Reaching 0.80 to 0.90 remains an open research ambition. More experiments or a larger tensor alone do not establish state-of-the-art performance.',
 'Appendix: distribution comparisons': 'Rows belong to separate studies. All scores are three-seed ensemble macro F1 on the original 164 patients. The physical-feature base means the median waveform, while the other bases retain the original convolution representation. The empirical-quantile gains over spread remain uncertain. The original convolution spread contrast is a separately labeled post-hoc observation. No between-row comparison isolates a single causal change.',
 'Appendix: kernel members and ensembles': 'Mean member F1 averages the three F1 scores. Ensemble F1 instead scores decisions after averaging the three probability vectors. Both retain seeds 42, 123 and 2026. The primary selected nonlinear-minus-linear contrast is -0.019441 with interval [-0.048244,+0.005086]. All fixed representations and both families remain reported. Inner representation decisions overlap in training patients and are not independent replications.',
 'Appendix: source learning comparisons': 'Each row uses its own matched control and source hypothesis. All displayed primary gain intervals include zero. The expanded native-pretraining and paired ECG procedures also failed to beat their matched scratch ensemble at 0.601801. PTB-XL transfer uses a different target schedule and therefore has a different scratch reference. Source AP and AUROC are separate endpoints. Recording-copy screens do not resolve all cross-dataset patient identity.',
 'Appendix: other matched readouts': 'These are separate experiments on the 164-patient all-record task. Intervals are descriptive paired patient bootstraps conditional on saved predictions. They omit refitting uncertainty and repeated-development effects. The table retains unsuccessful comparisons and the original declared controls. Independent confirmation remains necessary.',
 'Appendix: matched prediction errors': 'The source ensembles reproduce macro F1 of 0.603374 and 0.601801. Of 36 shared errors, 21 have the same wrong prediction. This is a diagnostic of complementary errors. It uses no fitted combiner and gives no permission to choose the correct model per patient using known answers. The fixed equal blend scored 0.582209.',
 'Appendix: all-record class results': 'Both columns use the same 164 patients, original class order and frozen folds. All three seeds contribute equally to each ensemble. These class results describe development predictions. We did not fit class-specific routing, change labels or choose weights using the outer results. AMI denotes anterior MI. Small class counts create substantial uncertainty.',
 'Appendix: earliest-record endpoints': 'Rows use the same 145 dated patients but different training and inference policies. They must not be presented as one continuous improvement curve. The 164-patient all-record score of 0.601801 is separate. Earliest archived date does not establish clinical presentation time. The 164-patient one-selected-record sensitivity uses deterministic fallbacks for missing or tied dates.',
 'Appendix: frozen readout endpoints': 'Both columns use one record per patient. The 164-patient column is not all-record inference. The 145 dated patients remain primary. All heads use encoders from exact training partitions and training-only preprocessing. The 256-versus-64 and Mahalanobis-versus-ridge contrasts remain uncertain and unfavorable in point estimate. No outer threshold tuning or winner selection follows from this table.',
 'Appendix: latest classifier by class': 'The primary cohort has 145 dated earliest-record patients. All three classifiers use the same 64-dimensional feature representation. Mahalanobis uses a shared covariance estimated from training patients with inner-selected shrinkage. Lambda choices across 30 fold-seed contexts overlap in patients. The class table does not justify changing the classifier per class based on outer performance.',
 'Appendix: data sources and storage': 'This is the release inventory reviewed on 19 November. Download size differs from extracted arrays, feature caches and model output. MIMIC is about 90.4 GB uncompressed. LUDB is about 23.6 MB uncompressed. MedalCare supplies no real people and uses artery/transmurality labels. Shandong lacks the full six-class target mapping. Its local screen covers metadata only. A future ECG-assisted study requires its own compatibility and overlap checks.',
 'Appendix: verification coverage': 'These counts belong to separate studies, and some studies reuse source models. They are not one combined new-fit total or independent patient count. Verification reconstructs features or restores saved parameters and repeats predictions and selections, according to the registered scope. The latest macOS suite passed 376 tests before the final study. Windows training remains unvalidated. The presentation includes aggregate evidence only.',
 'Appendix: selected references': 'The main paper DOI is 10.1109/TBME.2022.3202962. The sepsis paper DOI is 10.1016/j.bbe.2021.06.009. These publications motivate methods. Our bounded adaptations do not inherit their benchmark claims. Project protocols disclose implementation differences and attributions. The PTB signal illustration is an existing project figure from public data. The ECG schematic is by YitzhakNat under CC BY-SA 4.0, unchanged. Direct links appear below.'
};
for(const slide of slides)if(notesByTitle[slide.title])slide.notes=notesByTitle[slide.title];

const q=v=>typeof v==='string'?'"'+v.replaceAll('\\','\\\\').replaceAll('"','\\"')+'"':Array.isArray(v)?'{'+v.map(q).join(', ')+'}':String(v);
function prepare(){
 const a=['tell application "Keynote"','set d to make new document with properties {document theme:theme "Basic White",width:960,height:540}'];
 function text(t,x,y,w,h,size=23,bold=false){a.push('tell s',`set t to make new text item with properties {object text:${q(t)},position:{${x},${y}},width:${w},height:${h}}`,`set font of object text of t to ${q(bold?'TimesNewRomanPS-BoldMT':'TimesNewRomanPSMT')}`,`set size of object text of t to ${size}`,'set color of object text of t to {0,0,0}','end tell');}
 slides.forEach((p,i)=>{
  a.push(i===0?'set s to slide 1 of d':'tell d\nset s to make new slide at end of slides with properties {base layout:slide layout "Blank"}\nend tell','set base layout of s to slide layout "Blank" of d');
  a.push(`set presenter notes of s to ${q((p.backup?'APPENDIX':'MAIN PRESENTATION')+'\n\n'+p.notes+'\n\nEvidence and image credits\n'+p.sources.map(x=>x.startsWith('http')?x:'https://github.com/Kumaryan12/major_project/blob/main/'+x).join('\n'))}`);
  if(p.cover){text(p.title,50,52,860,45,32,true);text(p.lines.slice(0,2).join('\n'),50,112,860,82,26,true);text(p.lines[2],50,226,860,32,23);text(p.lines.slice(3,6).join('\n'),50,280,860,100,22);text(p.lines.slice(6).join('\n'),50,422,860,68,23);return;}
  text(p.title,36,20,888,54,p.title.length>48?28:30,true);
  let end=105;
  if(p.table){const y=100,rh=p.rowH||42,ws=p.widths;a.push('tell s',`set tb to make new table with properties {row count:${p.table.length},column count:${ws.length},header row count:1,header column count:0,position:{50,${y}},width:860,height:${rh*p.table.length}}`,'end tell','tell tb','set font name of cell range to "TimesNewRomanPSMT"',`set font size of cell range to ${p.font||22}`,'set text color of cell range to {0,0,0}','set vertical alignment of cell range to center','set text wrap of cell range to true','set background color of cell range to {65535,65535,65535}','set font name of row 1 to "TimesNewRomanPS-BoldMT"');ws.forEach((w,j)=>a.push(`set width of column ${j+1} to ${w}`));p.table.forEach((r,ri)=>{a.push(`set height of row ${ri+1} to ${rh}`);r.forEach((v,ci)=>a.push(`set value of cell "${String.fromCharCode(65+ci)}${ri+1}" to ${q(v)}`));});a.push('end tell');end=y+rh*p.table.length+24;}
  else if(p.chart){a.push(`add chart s row names {"Patient macro F1"} column names ${q(p.chart.labels)} data ${q([p.chart.values])} type horizontal_bar_2d group by chart row`,'set c to last chart of s','set position of c to {45,93}','set width of c to 870',`set height of c to ${p.chartH}`);end=93+p.chartH+20;}
  else if(p.image){let [x,y,w,h]=p.imageBox;a.push('tell s',`make new image with properties {file:POSIX file ${q(path.join(root,p.image))},position:{${x},${y}},width:${w},height:${h}}`,'end tell');end=105;}
  else if(p.diagram)end=p.diagram==='visits'?310:p.diagram==='readout'?300:300;
  const count=p.lines.length;
  const x=p.side?685:50,w=p.side?225:p.sideLeft?520:860;
  const endY=510, available=endY-end;
  const h=count?Math.min(p.side?120:p.sideLeft?77:82,available/count):0;
  p.lines.forEach((line,j)=>text(line,x,end+j*h,w,h-5,p.font||(p.side?22:p.table||p.chart?22:24)));
 });
 a.push(`save d in POSIX file ${q(path.join(build,'native.key'))}`,`export d to POSIX file ${q(path.join(build,'candidate.pptx'))} as Microsoft PowerPoint`,'close d saving no','end tell');
 fs.writeFileSync(path.join(build,'build.applescript'),a.join('\n'));
 fs.writeFileSync(path.join(build,'metadata.json'),JSON.stringify({slides},null,2));
 console.log(`Prepared ${slides.length} slides: ${slides.filter(x=>!x.backup).length} main, ${slides.filter(x=>x.backup).length} appendix.`);
}
const run=(cmd,args)=>{const r=spawnSync(cmd,args,{cwd:root,encoding:'utf8',maxBuffer:4*1024*1024});if(r.status!==0)throw new Error(r.stderr||r.stdout);if(r.stdout)process.stdout.write(r.stdout);};
if(process.argv[2]==='prepare')prepare();
else if(process.argv[2]==='build')run('osascript',[path.join(build,'build.applescript')]);
else if(process.argv[2]==='finish')run('.venv/bin/python',['scripts/finalize_progress_update_20261120.py']);
else if(process.argv[2]==='pdf')run('osascript',[path.join(build,'export_pdf.applescript')]);
else console.log('Use prepare, build, finish or pdf.');
