/** Formal progress deck. Native Keynote fallback when artifact-tool is unavailable.
 * Usage: node scripts/build_progress_presentation_20261116.mjs prepare|build|finalize|pdf
 * Historical experiment code and presentations are read only.
 */
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const root=process.cwd();
const build=path.join(root,'tmp/presentations/progress_20261116');
const ppt=path.join(root,'output/pptx/mi_progress_since_october_2026.pptx');
const pdf=path.join(root,'output/pdf/mi_progress_since_october_2026.pdf');
fs.mkdirSync(build,{recursive:true});
const R=(p)=>`results/${p}/complete/REPORT.md`;
const slides=[];
function s(title,lines,notes,sources,extra={}) {slides.push({title,lines,notes,sources,...extra});}
s('Implementation progress',[
 'Patient-Independent Localization of Myocardial Infarction',
 'from Vectorcardiographic Signals',
 'Aryan Kumar (23ECE1006)',
 'Major project with Kunal Maka (23ECE1017)',
 'Department of Electronics and Communication Engineering',
 'National Institute of Technology Goa',
 'Guide: Dr. Shivnarayan Patidar',
 'Progress since 7 October 2026'
], 'My friend has introduced the research paper and the problem. I will explain what we implemented after the previous presentation, what the completed experiments show, and what is running now. The primary task remains six-class localization from measured Frank XYZ VCG.', ['docs/CODEX_HANDOFF.md','docs/implementation_presentation_notes.md'],{cover:true});
s('Progress since the previous presentation',[
 'The October presentation proposed waveform neural models as future work.',
 'We have now trained them, tested tensor alternatives, and built a source-learning pipeline.',
 'These historical procedures share the patient endpoint but use different models and aggregation.'
], 'In October the strongest nested feature result was 0.487. The later ResNet was trained with all three declared seeds, giving a mean of 0.550. A separate post-hoc uniform ensemble later reached 0.584570. The mean and ensemble are different predictors and should not be treated as one uninterrupted improvement curve. The approximate 0.063 increase from the old feature result to the ResNet mean is historical development evidence, not fresh-cohort confirmation.', ['docs/CODEX_HANDOFF.md','results/neural_repeat_v1/REPORT.md',R('neural_residual_seed_ensemble_v1')],{table:[['Completed procedure','Patient macro F1','Summary'],['October feature baseline','0.487','One nested procedure'],['Waveform ResNet','0.550 ± 0.040','Mean ± seed SD'],['Ordered context + auxiliary loss','0.584570','Three-seed ensemble']],widths:[420,175,265],rowH:39});
s('Unchanged patient-level evaluation',[
 '164 patients, 391 recordings and 55,863 original measured-XYZ beats.',
 'Six classes: AMI, ALMI, ASMI, IMI, ILMI and HC.',
 'Ten frozen outer patient folds. Three inner folds select settings and stopping in later studies.',
 'Training partitions alone fit normalization, references, selectors and model combinations.',
 'Primary endpoint: six-class patient macro F1. Every patient stays wholly within one partition.'
], 'The unit of independence is a patient. More beats or overlapping windows do not add independent patients. Each original beat has 651 samples at 1000 Hz, from 250 ms before R through 400 ms after R. Aggregation is declared separately for each study. We retain every target patient, original label, cached R position and frozen fold. Micro F1 equals accuracy for this single-label problem and remains secondary.', ['AGENTS.md','docs/CODEX_HANDOFF.md','configs/patient_folds.csv'],{footer:'Repeatedly inspected PTB folds remain development data.'});
s('Implemented pipeline after October',[
 '1. Preserve the audited measured-XYZ recordings and original R-aligned beat bank.',
 '2. Construct waveform, multiview, relational, geometric or random-convolution representations.',
 '3. Fit each representation and classifier inside the corresponding patient training partition.',
 '4. Use inner patients for model choices. Predict every held-out patient in the outer folds.',
 '5. Restore saved models and replay decisions before publishing aggregate results.'
], 'The major implementation change is that the project is now a set of controlled, reproducible studies rather than one tensor recipe. Some methods classify beats and average probabilities. Others first build recording features and aggregate records. We compare each method with its matched control rather than attributing every difference to the tensor alone.', ['docs/CODEX_HANDOFF.md'],{footer:'Measured VCG stays primary. ECG-assisted work has a separate scope.'});
s('Waveform ResNet and repeatability',[
 '1D ResNet: widths 16/32/64, a 64-feature embedding and a six-class head.',
 'All held-out beat probabilities contribute to a patient prediction.',
 'Three-seed mean: 0.550 ± 0.040. The spread measures training-seed variation.'
], 'We implemented a compact 1D ResNet that learns waveform features directly. The seed-42 plain model scored 0.586 and its statistics-fusion variant 0.584. We did not stop at that favorable seed. Seeds 123 and 2026 scored 0.557 and 0.507, giving 0.550 mean with sample standard deviation 0.040. This slide reports individual models and their mean, not a probability ensemble.', [R('neural_vcg_v1'),'results/neural_repeat_v1/REPORT.md'],{chart:{labels:['Seed 42','Seed 123','Seed 2026'],values:[0.586,0.557,0.507],max:.7},footer:'Patient macro F1 for each completed training seed.'});
s('Carefully constructed multiview tensors',[
 'Five views: raw R, denoised D, recording median M, residual D−M and residual R−D.',
 'Reconstruction checks preserve R = D + (R−D) and D = M + (D−M).',
 'Feature tensor: beat × five views × XYZ × five bands × time.',
 'Morphology, multiscale and inter-channel features preserve signed orientation and amplitude.',
 'Higher reconstruction fidelity alone did not consistently improve localization.'
], 'We separated the shared recording shape from beat-to-beat residuals and denoising residuals. These are transformations of the same measured signal, not new sensors or extra patients. We checked their numerical reconstruction identities. We then extracted complementary morphology, scale and relation features. The feature-first route completed. The older full multiview neural sweep remains paused and is not a completed result.', ['docs/record_tensor_quality_plan.md','docs/multiview_vcg_plan.md','docs/tensor_feature_design_plan.md',R('tensor_refinement_v1')],{footer:'Feature-first studies completed. The older multiview neural sweep remains paused.'});
s('Feature stability and tabular learning',[
 'Bootstrap feature consensus helped the matched tree procedure.',
 'TabPFN on the resulting tensor features gave a higher point score, with an uncertain difference.',
 'Hard semantic grouping and fixed prior correction did not improve the tested recipe.'
], 'A larger feature bank was not sufficient. The refinement study gave mixed results: expanded linear 0.425 and tree 0.452, versus legacy linear 0.498 and tree 0.420. Bootstrap consensus then improved the matched tree score to 0.513. A local TabPFN-v2 procedure scored 0.542, but its difference from bootstrap XGBoost was uncertain. These are feature procedures and must not be conflated with the neural seed mean or its ensemble.', [R('tensor_refinement_v1'),R('feature_stability_v1'),R('tabpfn_v2_v1')],{table:[['Matched feature procedure','Macro F1','Interpretation'],['Legacy fixed tree','0.420','Stability-study control'],['Bootstrap-consensus tree','0.513','Gain interval +0.027 to +0.158'],['Raw TabPFN-v2','0.542','Gain over tree remains uncertain'],['TabPFN prior correction','0.498','No improvement']],widths:[360,145,355],rowH:38,footer:'Intervals are descriptive and conditional on fitted development predictions.'});
s('Relations, geometry and ordered paths',[
 'We tested physiological structure as a representation hypothesis.',
 'Each study retained its own matched control. None established a reliable overall gain.'
], 'The sepsis-paper idea became correlation and covariance tensors with shared training-only bases. Geometry described signed VCG loops and approximate phase features. Healthy-reference tensors described deviation from training references. Path signatures encoded within-beat ordered paths. Their controls help test whether the added structure is useful, and the results did not establish a consistent gain. The adaptive phase detector accepted only 23.2% of beats and its acceptance differed by class, so it cannot be called validated clinical delineation.', [R('relational_vcg_v1'),R('neural_geometry_v1'),R('deviation_vcg_v1'),R('signature_vcg_v1')],{table:[['Study','Candidate / matched control','Outcome'],['Shared covariance + statistics','0.470 / 0.469','No demonstrated gain'],['Adaptive geometry, linear model','0.467 / 0.445','Uncertain difference'],['Healthy-reference deviation','0.479 / 0.499','No demonstrated gain'],['Third-order path signature','0.508 / 0.515','No demonstrated gain']],widths:[350,300,210],rowH:41,footer:'Separate studies and controls. Scores are patient macro F1.'});
s('Consecutive beats and RR timing',[
 'Real consecutive windows of 4, 8 or 16 beats stay within one recording.',
 'Inner selection chooses one shared length for ordered and shuffled controls.',
 'Shuffling moves each beat with its own RR pair. All original query beats remain covered.'
], 'The hypothesis was that consistency and timing across beats distinguish infarct subtypes. We used a shared beat encoder and a recurrent context model. The matched shuffle destroys ordering while preserving beat/RR pairs. We evaluated 4, 8 and 16 beats with a comparable nominal beat budget and symmetric inner length selection. The ordered-versus-shuffled difference was minus 0.008 with an interval from minus 0.057 to plus 0.039. This did not demonstrate an ordering benefit. All 1200 fitted contexts and 120 outer models were verified.', [R('neural_chronological_lengths_v1'),'docs/chronological_lengths_plan.md'],{table:[['Length-selected procedure','Mean macro F1 ± seed SD'],['Mean pooling + RR','0.485 ± 0.055'],['Ordered context, no RR','0.509 ± 0.039'],['Ordered context + RR','0.524 ± 0.018'],['Shuffled context + RR','0.533 ± 0.013']],widths:[560,300],rowH:35,footer:'Ordered−shuffled: −0.008 [−0.057, +0.039]. No demonstrated ordering gain.'});
s('Direct morphology with residual context',[
 'A direct beat-classification path preserves morphology.',
 'A gated context correction and an auxiliary beat loss test whether context adds useful signal.',
 'Matched mean-seed results did not establish an improvement.'
], 'The chronology diagnostics suggested that context training might sacrifice direct beat morphology. We therefore retained a direct beat path and added context as a residual correction. We tested ordered and shuffled context with and without an auxiliary beat loss. The original trained-model result was 0.539 for ordered plus auxiliary compared with 0.551 for beat-only. All six arms, three seeds, 720 fits, 180 outer restorations and 120 matched checks are retained. The full arm table is in backup.', [R('neural_residual_context_v1'),'docs/chronological_diagnostics_findings.md'],{table:[['Procedure','Mean macro F1 ± seed SD'],['Matched beat-only','0.551 ± 0.027'],['Ordered context + auxiliary loss','0.539 ± 0.032'],['Paired difference','−0.012 [−0.051, +0.027]']],widths:[560,300],rowH:43,footer:'This original study reports mean member F1, before the separate ensemble analysis.'});
s('Uniform seed ensembles',[
 'A separate post-hoc analysis averaged all three probability vectors for every arm.',
 'No new fits, fitted mixing weights, seed subsets or threshold search.',
 'The higher point score remains uncertain and needs independent confirmation.'
], 'An ensemble is one predictor obtained by averaging model probabilities. Its F1 is not the arithmetic mean of member F1 scores. We retained all six arms and all three seeds, restored every outer member and replayed the source checks. Ordered plus auxiliary scored 0.584570, versus 0.545112 for the matched beat-only ensemble. The paired difference interval includes zero. Because we examined ensembling after seeing the source results, it is explicitly post-hoc development and does not rewrite the earlier mean-model conclusion.', [R('neural_residual_seed_ensemble_v1'),'docs/residual_seed_ensemble_plan.md'],{table:[['Procedure','Mean member F1','Ensemble F1'],['Matched beat-only','0.551','0.545112'],['Ordered context + auxiliary loss','0.539','0.584570'],['Ensemble difference','', '+0.039458']],widths:[440,210,210],rowH:43,footer:'Difference interval: [−0.034767, +0.112701]. Three models per fold at inference.'});
s('Fusion and staged context learning',[
 'Neural–tensor late fusion selected a single blend weight using inner patients.',
 'Staged learning first trained morphology, then adapted or froze it while learning context.',
 'Neither follow-up established a gain. The adaptive fusion gate remains deferred.'
], 'We tried to combine complementary neural and tensor predictions with a simple inner-selected scalar. It scored 0.583052 against 0.584570 for neural alone. We also tested continued morphology, joint context, frozen context and shuffled context in a separate staged study. Frozen context scored 0.563539 against matched morphology 0.545112, but its interval included zero and the shuffled control scored 0.565056. Inner validation chose zero adaptation updates in 82 of 120 outer adaptations. These results do not support an ordering benefit or a larger adaptive gate.', [R('neural_tensor_fusion_v1'),R('neural_staged_context_v1')],{table:[['Study comparison','Ensemble F1','Difference interval'],['Inner-selected fusion / neural','0.583052 / 0.584570','[−0.050922, +0.050275]'],['Frozen context / morphology','0.563539 / 0.545112','[−0.021765, +0.061253]'],['Shuffled frozen context','0.565056','No ordering advantage']],widths:[365,245,250],rowH:43,footer:'Each row uses the stated study control. All scores here are ensemble F1.'});
s('Random-convolution features with ridge',[
 'Each recording uses the coordinate-median waveform of all its original beats.',
 'MultiRocket-style and experimental multivariate HYDRA features feed a matched ridge classifier.',
 'Combined features improved the matched DCT control, while losing within-record variation.'
], 'This route extracts many fixed convolutional responses rather than training a deep encoder. Training-only scale and threshold estimation remain inside each inner or outer partition. The combined three-seed ensemble scored 0.584266. Its difference from DCT was plus 0.138485 with a positive descriptive interval. Its gain over MultiRocket alone remained uncertain. It is essentially level with, and slightly below, the separate historical neural ensemble 0.584570. The bounded implementation differs from upstream defaults and the HYDRA multivariate code remains experimental with GPL attribution.', [R('rocket_vcg_v1'),'docs/rocket_vcg_plan.md'],{chart:{labels:['DCT control','MultiRocket-style','HYDRA','Combined'],values:[.445781,.561442,.548100,.584266],max:.7},footer:'Three-seed ensemble F1. Combined−DCT: +0.138485 [+0.047546, +0.228367].'});
s('Selected convolution features and TabPFN',[
 'Training-only bootstrap consensus selected 32 or 64 convolution features.',
 'The selected-feature ridge and PFN procedures both lost score against full-feature ridge.',
 'AMI class F1 fell from 0.345 to 0.080 under both selected procedures.'
], 'The next bounded study asked whether a compact set of stable convolution features would let PFN help. We retained the full ridge reference and used nested budget selection for both classifiers. Neither selected procedure improved it. AMI correctly classified patients fell from 5 of 17 to 1 of 17. Fixed-budget outcomes are retained in backup, but cannot replace the prespecified selector after viewing outer scores. All 660 new classifiers and the reused source features were restored or reconstructed.', [R('rocket_tabpfn_v1'),'docs/rocket_tabpfn_plan.md'],{table:[['Prespecified procedure','Ensemble macro F1'],['Full-feature ridge','0.584266'],['Selected-feature ridge','0.544036'],['Selected-feature PFN','0.530966']],widths:[560,300],rowH:41,footer:'PFN−full ridge: −0.053301 [−0.127591, +0.021462].'});
s('What the data diagnostics showed',[
 'The independent unit remains 164 patients despite tens of thousands of beats.',
 'A training-patient learning curve rose approximately from 0.407 to 0.550.',
 'Error review found 35 patients wrong across all three selected ResNet seeds.',
 'Chronology diagnostics found persistent IMI/ILMI errors and no simple normalization mismatch.',
 'These observations motivate better signal use and source learning. They do not prove label noise or a ceiling.'
], 'We audited errors and training behavior before changing labels or adding complexity. None of the persistent-error observations is expert label adjudication. The chronology review replayed all 1200 corresponding waveform normalizations and found agreement. Equal-record aggregation left the historical ResNet mean close to 0.550, and no selected-length model stopped at the 25-epoch cap. Those checks rule out specific easy explanations but do not establish why context failed. The learning curve is descriptive and cannot predict a guaranteed sample size for 0.80.', ['results/neural_diagnostics_v1/REPORT.md',R('neural_learning_curve_v1'),'docs/chronological_diagnostics_findings.md']);
s('Expanded supervised source cohort',[
 'PTB-XL native diagnostic annotations provide a separate source-learning task.',
 'We include all 1,841 likelihood-100 MI training patients, then fill to 4,096 patients.',
 'All selected visits remain. Source fold 10 stays excluded before label and signal access.'
], 'The native audit examined official training folds 1 through 8 and validation fold 9. The bounded expansion retained 5063 training visits from 4096 patients, with the entire fold-9 validation set of 2183 records from 1942 patients. The source input is Kors-derived XYZ from twelve-lead ECG, not measured Frank XYZ. Every record uses up to four uniformly spaced complete crops, with masks for padding. Source training is additional information and compute, so it requires explicit provenance and a new matched target comparison.', ['results/supervised_source_audit_v1/REPORT.md','results/supervised_source_pilot_v1/REPORT.md'],{table:[['Source partition','Patients','Recordings'],['Training, folds 1–8 subset','4,096','5,063'],['Validation, complete fold 9','1,942','2,183'],['Total retained','6,038','7,246']],widths:[440,210,210],rowH:43,footer:'Source input: ECG-derived XYZ. The measured-VCG target cohort remains unchanged.'});
s('Native source annotation objectives',[
 'Twenty-eight heads meet training-only support in the retained source cohort.',
 'Unlisted codes are annotation negatives under a disclosed research convention.',
 'Listed likelihood 0 is unknown and masked in both objectives.',
 'Primary “certain” objective uses likelihood-100 positives and masks intermediate annotations.',
 'Sensitivity objective also counts listed intermediate annotations as positive.'
], 'We preserve overlapping native diagnoses instead of forcing PTB-XL into our exclusive six target classes. An unlisted code is not adjudicated absence of disease. Likelihood values are not calibrated soft probabilities. Source NORM is not automatically clinical HC, and broad source AMI cannot be silently mapped to the target exclusive AMI class. Both objectives share common validation masks and train-only normalization, sampling and positive weights. Validation macro average precision chooses the checkpoint.', ['results/supervised_source_pilot_v1/REPORT.md','docs/supervised_source_pilot_plan.md'],{footer:'Native source labels support representation learning, with unresolved clinical equivalence.'});
s('Completed supervised source training',[
 'Both source objectives trained 15 epochs and selected epoch 13 using validation AP.',
 'One source training seed, shared crop-pooling ResNet and all 28 heads retained.',
 'Restored models reproduced full validation predictions.'
], 'The primary source model reached macro AP 0.317456 compared with a training-prevalence constant baseline of 0.037470. Its AUROC was 0.908703 and fixed-threshold F1 was 0.290552. The intermediate-positive sensitivity had nearly identical AP. We retain both rather than selecting an objective from this tiny difference. These are patient-weighted record-level multilabel source metrics on model-selection data. They are not target six-class macro F1, and good source learning does not prove transfer.', ['results/supervised_source_pilot_v1/REPORT.md'],{table:[['Source procedure','Macro AP','AUROC','F1 at 0.5'],['Prevalence constant','0.037470','0.500000','0.000000'],['Certain objective','0.317456','0.908703','0.290552'],['Intermediate-positive','0.317306','0.906510','0.287006']],widths:[335,175,175,175],rowH:43,footer:'Different endpoint: native multilabel source learning. No target localization result here.'});
s('Expanded recording-overlap screen',[
 'All 7,246 retained source recordings screened against 391 target recordings.',
 'Independent replay re-decoded signals and repeated every selected continuous comparison.',
 'The recording-copy screen passed. Patient identity across datasets remains unresolved.'
], 'We screened source training and source model-selection validation, not only a convenient subset. Encoded-file matching found zero copies. Template retrieval selected top-eight candidates and the continuous screen checked 57968 pairs out of 2833186 possible. All 20 known-copy controls passed, with zero high-correlation candidates and zero insufficient-active-lead pairs. This checks a declared class of recording copies. Different visits by the same person, short or transformed copies and retrieval misses remain possible, so this cannot be called patient-identity clearance.', ['results/supervised_source_overlap_v1/REPORT.md','docs/supervised_source_overlap_plan.md'],{table:[['Completed check','Result'],['Encoded-file matches','0'],['High-correlation candidates','0'],['Insufficient-active-lead pairs','0'],['Known-copy controls','20 / 20'],['Continuous candidate comparisons','57,968']],widths:[600,260],rowH:31,footer:'No target labels, patients, recordings or thresholds changed in response to model errors.'});
s('Current matched supervised-transfer study',[
 'Three target arms: scratch, certain-source initialization and intermediate-positive initialization.',
 'Only encoder weights transfer. All arms receive fresh matching six-class heads.',
 'Three target seeds, ten outer folds and two learning rates selected in three inner folds.',
 'Declared total: 630 fits, including 90 outer models. Full restoration precedes final reporting.',
 'Training resumed from saved completed fits. No final localization result is available yet.'
], 'The current question is whether supervised source learning improves initialization for measured-VCG localization. Both source objectives enter the comparison regardless of their small source-AP difference. We discard source heads, source normalization and optimizer state, fit target normalization on the exact training partition, and fine-tune the same architecture. All arms share target sampling and head initialization for matched contexts. The primary result will be certain minus scratch for an equal ensemble of all three target seeds. The interrupted MPS run resumed from compatible saved fits; only an unfinished fit repeats. The runner verifies the final study, publishes aggregate results and stops without launching another experiment.', ['docs/supervised_transfer_plan.md','configs/supervised_transfer_v1.yaml'],{footer:'Running study. Partial counters and inner scores are not final performance.'});
s('Reproducibility and project status',[
 'Saved YAML protocols and frozen patient folds define each experiment.',
 'Source, configuration and runtime hashes protect completed numerical artifacts.',
 'Independent reports restore models, replay selections and verify paired training contexts.',
 'Latest full pre-launch macOS verification: 289 tests passed. Windows remains unvalidated.',
 'Completed: source learning and recording screen. Active: supervised transfer. Paused: older multiview neural sweep.'
], 'We expanded the codebase with numerical and integrity checks rather than just reporting better scores. The 289-test suite includes transfer provenance, matching, training-only normalization and selection, tamper checks and small-study restoration. Full real-data studies also have their own independent replay counts, listed in backup. We keep only aggregate metrics and hashed audits public. Signals, fitted weights, patient predictions and source contexts stay local and ignored. A clone does not control another machine’s processes. No following training experiment is queued.', ['AGENTS.md','docs/CODEX_HANDOFF.md'],{footer:'Tests establish implemented behavior. They do not establish clinical validity.'});
s('Current conclusions and next decisions',[
 'Completed waveform learning improved the historical point result beyond the October feature baseline.',
 'Promising development references are the 0.584570 neural ensemble and 0.584266 convolution-feature ensemble.',
 'Added tensor structure, chronology and compression did not consistently beat their matched controls.',
 'Next decision: finish the declared transfer comparison, then consider a bounded distribution-pooling study.',
 'Independent label-compatible patients are still needed for a strong generalization claim.'
], 'The project has progressed substantially in implementation and evidence, but the 0.80 to 0.90 target is still unachieved. A practical next hypothesis is to preserve the distribution of all beat shapes rather than only a recording median or a mean probability. Proposed controls are encoding the median waveform, averaging individual-beat features, pooling moments, and pooling projected quantiles with a comparable feature budget. Paired acquisition alignment is a separate ECG-assisted proposal. Synthetic physiology needs a source-label and anatomy audit first. Within-beat dynamics is another bounded proposal. None is implemented, queued or guaranteed to help. The present transfer study finishes first.', ['docs/CODEX_HANDOFF.md','docs/research_frontiers_20261116.md'],{footer:'Development evidence only. No SOTA, clinical-validity or guaranteed target-score claim.'});
s('Backup: patient bags and contrastive learning',[
 'Unordered bags pool embeddings from up to 16 beats. They are distinct from consecutive-beat sequences.',
 'Contrastive comparisons use their own matched CE sampler controls.'
], 'Patient-bag learning and learned Tucker factors were trained end to end with patient labels. The bag can include separate recordings and has no claimed temporal order. Attention, learned scales and compression did not outperform the single-seed waveform reference. The original contrastive comparison improved its altered-sampler CE control by 0.037 on average, but the interval included zero. A more patient-diverse follow-up reversed the point difference. This does not establish robust superiority.', [R('neural_patient_tensor_v1'),R('neural_contrastive_v1'),R('neural_contrastive_diverse_v1')],{backup:true,table:[['Completed comparison','Candidate / control F1'],['Mean embedding / waveform reference','0.522 / 0.586 (single seed)'],['Gated attention / waveform reference','0.443 / 0.586 (single seed)'],['Learned scales + fixed compression','0.480 (single seed)'],['Learned supervised Tucker + attention','0.507 (single seed)'],['Original SupCon / matched CE','0.516 / 0.479 (seed mean)'],['Diverse SupCon / matched CE','0.515 / 0.531 (seed mean)']],widths:[520,340],rowH:32});
s('Backup: waveform follow-ups',[
 'Each row retains the study’s own protocol and declared seed policy.',
 'These tested recipes did not establish a consistent gain over their controls.'
], 'The preprocessing study tested raw against the established denoised input. The longer-crop study tested more context around R. Bounded regularization selected among four fixed presets using inner patients and shared baseline fits where appropriate. Territory supervision derived labels from the existing class names and therefore supplied no independent anatomy annotation. The error audit and learning curve are diagnostics rather than new clinical ground truth.', [R('neural_preprocessing_v1'),R('neural_context_v1'),R('neural_regularization_v1'),R('neural_territory_v1'),'results/neural_diagnostics_v1/REPORT.md',R('neural_learning_curve_v1')],{backup:true,table:[['Completed recipe or diagnostic','Reported result'],['Raw waveform','Mean F1 about 0.521'],['Longer R-relative crops','Mean F1 about 0.538'],['Inner-selected regularization / baseline','0.550 / 0.550'],['Territory auxiliary / CE reference','0.538 / 0.550'],['Persistent ResNet errors','35 patients wrong across all 3 seeds'],['Training-patient budget curve','Approximately 0.407 to 0.550']],widths:[520,340],rowH:33});
s('Backup: correlation and covariance tensors',[
 'XYZ × eight wavelet slices gives 24 channels. Four approximate R-relative regions form 24 × 24 × 4 tensors.',
 'A training-only shared channel basis compresses both matrix axes. Inner folds select rank 4 or 8.',
 'The sepsis paper motivated this VCG adaptation. Its endpoint and reported scores do not transfer to MI.'
], 'All arms used the same fixed depth-3 XGBoost and training weights. Statistics retain amplitude information. Covariance also retains amplitude relative to training-only channel scales, whereas correlation omits absolute scale. Shared covariance gave 0.470 compared with 0.469 for short-beat statistics, with a paired interval including zero. The approximate regions are not expert QRS, ST or T delineations.', [R('relational_vcg_v1'),'https://doi.org/10.1016/j.bbe.2021.06.009'],{backup:true,table:[['Representation','Patient macro F1'],['Short-beat statistics control','0.469'],['Full-record wavelet statistics','0.421'],['Flat correlations + context statistics','0.459'],['Shared correlation cores + statistics','0.413'],['Shared covariance cores + statistics','0.470']],widths:[600,260],rowH:31});
s('Backup: geometry and reference limitations',[
 'Adaptive boundaries passed strict checks for 12,977 of 55,863 beats, about 23.2%.',
 'Acceptance was class dependent: about 43% for HC and 15–22% for MI classes.',
 'Fixed and adaptive descriptors retained the full target cohort through declared fallback rules.',
 'Healthy references came only from the relevant training patients.',
 'Path signatures summarize within-beat order. They differ from cross-beat chronology.'
], 'Geometry can provide physically interpretable descriptors without proving a clinical delineator. The adaptive linear result 0.467 versus statistics 0.445 had an interval from minus 0.061 to plus 0.107. The tree statistics-plus-geometry result 0.461 versus 0.409 had an interval from minus 0.024 to plus 0.131. Healthy deviation scored 0.479 versus matched morphology 0.499. Signature3 scored 0.508 versus morphology 0.515. We therefore retain these as completed ablations with no demonstrated reliable overall gain.', [R('neural_geometry_v1'),R('deviation_vcg_v1'),R('signature_vcg_v1')],{backup:true,footer:'No expert boundary adjudication or validated physiological explanation follows from these features.'});
s('Backup: earlier transfer and ECG extension',[
 'The earlier masked-reconstruction source pilot used 1,024 source training patients.',
 'The new native-diagnostic supervised study uses a different source objective and expanded cohort.',
 'Measured ECG + VCG is a separately labeled extension, with matched additive and bilinear controls.'
], 'The old source-transfer experiment averaged three target-seed scores: pretrained 0.520 versus matched scratch 0.553. That is not the current supervised transfer result. The ECG extension used simultaneous measured twelve-lead ECG in PTB. Its VCG twin control matched added capacity without adding ECG information. Bilinear fusion scored 0.534 versus additive 0.536 and VCG 0.550, all seed means. The new primary project input remains measured XYZ.', [R('neural_transfer_v1'),R('neural_paired_ecg_v1')],{backup:true,table:[['Completed procedure','Mean patient macro F1'],['Earlier masked-source transfer / scratch','0.520 / 0.553'],['Measured VCG / measured ECG','0.550 / 0.509'],['VCG twin capacity control','0.511'],['ECG + VCG additive fusion','0.536'],['ECG + VCG bilinear fusion','0.534']],widths:[600,260],rowH:33});
s('Backup: feature refinement and budgets',[
 'Fidelity, stable selection and downstream prediction were evaluated separately.',
 'Fixed-budget scores remain disclosed controls. Outer outcomes cannot replace the declared selector.'
], 'Increasing DCT ranks improved reconstruction but did not give consistent localization gains. The early feature-stability result and the later convolution-feature selection result concern different feature banks. We cannot generalize that selection always helps. In the later study both selected procedures reduced AMI F1, while fixed 32 and 64 feature controls showed that the classifier and budget interact. These controls are evidence for a future hypothesis, not permission to select a new outer-score winner retrospectively.', [R('tensor_refinement_v1'),R('feature_stability_v1'),R('tabpfn_v2_v1'),R('rocket_tabpfn_v1')],{backup:true,table:[['Study and procedure','Macro F1'],['Refinement: expanded linear / tree','0.425 / 0.452'],['Refinement: legacy linear / tree','0.498 / 0.420'],['Convolution fixed32: ridge / PFN','0.495415 / 0.510066'],['Convolution fixed64: ridge / PFN','0.576198 / 0.532195'],['Convolution selected: ridge / PFN','0.544036 / 0.530966']],widths:[600,260],rowH:35});
s('Backup: all residual-context arms',[
 'Original mean-model results and the later post-hoc ensembles remain separate.',
 'Every arm retains all three seeds. Ensembling used zero new training fits.'
], 'This complete table prevents a favorable ensemble arm from hiding the original study. Mean member F1 averages three scores, whereas ensemble F1 scores the average probability predictor. Ordered context plus auxiliary has the highest ensemble point score within this table but its matched improvement remains uncertain. The full source study verified 720 fits, 180 outer restorations and 120 matched checks. The follow-up independently restored all 180 members again for inference.', [R('neural_residual_context_v1'),R('neural_residual_seed_ensemble_v1')],{backup:true,table:[['Arm','Mean member F1','Ensemble F1'],['Beat only','0.551','0.545'],['Beat + auxiliary','0.518','0.549'],['Ordered context','0.520','0.554'],['Ordered context + auxiliary','0.539','0.585'],['Shuffled context','0.518','0.573'],['Shuffled context + auxiliary','0.510','0.556']],widths:[440,210,210],rowH:34});
s('Backup: all staged-learning arms',[
 'Morphology teachers are fresh for each exact inner or outer training partition.',
 'Inner selection chose zero adaptation updates for 82 of 120 outer adaptations.',
 'The historical 0.584570 ensemble is a separate reference, rather than this study’s matched control.'
], 'Staged learning protected the morphology encoder while asking whether context could improve it. Continued morphology uses more training without context. Joint and frozen branches separate adaptation from context learning. A shuffled branch tests ordering. Frozen versus morphology had a point gain of 0.018428 with an interval including zero, and shuffled scored slightly higher than ordered frozen. All 600 phase checkpoints, 150 outer checkpoints and 120 matched checks were verified.', [R('neural_staged_context_v1')],{backup:true,table:[['Prespecified procedure','Three-seed ensemble F1'],['Morphology','0.545112'],['Continued morphology','0.549550'],['Joint context','0.559042'],['Frozen context','0.563539'],['Frozen shuffled context','0.565056']],widths:[560,300],rowH:32});
s('Backup: class-level reference results',[
 'Both columns use all original target patients and six-way predictions.',
 'They describe separate development procedures, rather than a matched superiority comparison.'
], 'HC is easier than the small MI subtypes in both references. In the neural ordered-plus-aux ensemble AMI F1 is 0.294, while HC is 0.868. The convolution ensemble has AMI 0.345 and HC 0.925. Patient counts are small, especially AMI and ALMI. Macro F1 gives each class equal weight, so improving healthy-control discrimination alone does not solve subtype localization. These outer class outcomes must not choose class-specific routing or relabeling.', [R('neural_residual_seed_ensemble_v1'),R('rocket_vcg_v1')],{backup:true,table:[['Class','Patients','Neural ensemble F1','Convolution ensemble F1'],['AMI','17','0.294','0.345'],['ALMI','16','0.471','0.483'],['ASMI','27','0.640','0.633'],['IMI','29','0.596','0.542'],['ILMI','23','0.638','0.578'],['HC','52','0.868','0.925']],widths:[120,130,305,305],rowH:34});
s('Backup: completed independent verification',[
 'These are replay and restoration counts, separate from predictive performance.',
 'The current transfer study still requires its final 630-model verification.'
], 'Each completed study has its own manifest and independent report. Restoration means loading saved parameters and reproducing predictions, not merely observing that a filename exists. Scope and numerical hashes keep completed studies immutable. These checks reduce implementation and provenance errors but do not correct repeated development or establish clinical generalization.', [R('neural_chronological_lengths_v1'),R('neural_residual_context_v1'),R('neural_tensor_fusion_v1'),R('neural_staged_context_v1'),R('rocket_vcg_v1'),R('rocket_tabpfn_v1'),'results/supervised_source_pilot_v1/REPORT.md','results/supervised_source_overlap_v1/REPORT.md'],{backup:true,table:[['Completed scope','Verification coverage'],['Chronology 4/8/16','1,200 fits, 120 outer models'],['Direct beat + residual context','720 fits, 180 outer, 120 matched checks'],['Neural–tensor fusion','250 source contexts, 40 outer restorations'],['Staged context','600 phase checkpoints, 150 outer'],['Random convolutions','120 transforms, 840 classifiers'],['Selected convolution / PFN','660 new classifiers, 240 masks'],['Expanded recording screen','7,246 signals and 57,968 comparisons']],widths:[395,465],rowH:29});
s('Backup: selected research references',[
 'Zhang et al. Automated Localization of Myocardial Infarction From Vectorcardiographic via Tensor Decomposition. IEEE TBME, 2023. DOI: 10.1109/TBME.2022.3202962.',
 'Nesaragi, Patidar and Thangaraj. A correlation matrix-based tensor decomposition method for early prediction of sepsis from clinical data. 2021. DOI: 10.1016/j.bbe.2021.06.009.',
 'Ilse et al. Attention-based Deep Multiple Instance Learning. ICML, 2018. Khosla et al. Supervised Contrastive Learning. NeurIPS, 2020.',
 'Tan et al. MultiRocket. 2022. Dempster et al. HYDRA. 2023. Hollmann et al. Accurate predictions on small data with a tabular foundation model. Nature, 2025.',
 'PTB and PTB-XL official PhysioNet documentation. Full implementation evidence and additional method references appear in slide notes and repository reports.'
], 'These references motivate the implemented methods. Our VCG adaptations and bounded budgets differ from upstream settings where disclosed. The sepsis metric, tabular benchmark metrics and time-series benchmark rankings are not our localization scores. Detailed sources for the proposed next methods are in the research shortlist. The repository contains each exact protocol and aggregate report.', ['https://doi.org/10.1109/TBME.2022.3202962','https://doi.org/10.1016/j.bbe.2021.06.009','https://proceedings.mlr.press/v80/ilse18a.html','https://arxiv.org/abs/2004.11362','docs/rocket_vcg_plan.md','docs/tabpfn_v2_plan.md','https://physionet.org/content/ptbdb/1.0.0/','https://physionet.org/content/ptb-xl/1.0.3/'],{backup:true,font:18});

function q(v){return typeof v==='string'?'"'+v.replaceAll('\\','\\\\').replaceAll('"','\\"')+'"':Array.isArray(v)?'{'+v.map(q).join(', ')+'}':String(v);}
const run=(cmd,args)=>{const r=spawnSync(cmd,args,{cwd:root,encoding:'utf8',maxBuffer:10*1024*1024});if(r.status!==0)throw new Error(`${cmd}: ${r.stderr||r.stdout}`);if(r.stdout)process.stdout.write(r.stdout);};
function prepare(){
 let a=['tell application "Keynote"','set d to make new document with properties {document theme:theme "Basic White", width:960, height:540}'];
 function text(t,x,y,w,h,size=22,bold=false,italic=false,center=false){
  let font=italic?'TimesNewRomanPS-ItalicMT':bold?'TimesNewRomanPS-BoldMT':'TimesNewRomanPSMT';
  a.push('tell s',`set t to make new text item with properties {object text:${q(t)}, position:{${x},${y}}, width:${w},height:${h}}`,`set font of object text of t to ${q(font)}`,`set size of object text of t to ${size}`,'set color of object text of t to {0,0,0}','end tell');
 }
 slides.forEach((p,i)=>{
  a.push(i===0?'set s to slide 1 of d':'tell d\nset s to make new slide at end of slides with properties {base layout:slide layout "Blank"}\nend tell','set base layout of s to slide layout "Blank" of d');
  a.push(`set presenter notes of s to ${q((p.backup?'BACKUP':'MAIN PRESENTATION')+'\n\n'+p.notes+'\n\nEvidence\n'+p.sources.map(x=>x.startsWith('http')?x:'https://github.com/Kumaryan12/major_project/blob/main/'+x).join('\n'))}`);
  if(p.cover){
   text(p.title,50,62,860,42,30,true,false,true);
   text(p.lines.slice(0,2).join('\n'),50,108,860,66,24,true,false,true);
   text(p.lines[2],50,220,860,36,24,false,false,true);
   text(p.lines.slice(3,7).join('\n'),50,282,860,110,19,false,false,true);
   text(p.lines[7],50,430,860,35,23,false,false,true);
  }else{
   text(p.title,20,14,920,51,p.title.length>48?27:29,true);
   let end=95;
   if(p.table){
    const y=93,rh=p.rowH||38,ws=p.widths;
    a.push('tell s',`set tb to make new table with properties {row count:${p.table.length},column count:${ws.length},header row count:1,header column count:0,position:{50,${y}},width:860,height:${rh*p.table.length}}`,'end tell','tell tb','set font name of cell range to "TimesNewRomanPSMT"',`set font size of cell range to ${p.table.length>7?18:19}`,'set text color of cell range to {0,0,0}','set vertical alignment of cell range to center','set text wrap of cell range to true','set background color of cell range to {65535,65535,65535}','set font name of row 1 to "TimesNewRomanPS-BoldMT"');
    ws.forEach((w,j)=>a.push(`set width of column ${j+1} to ${w}`));
    p.table.forEach((row,r)=>{a.push(`set height of row ${r+1} to ${rh}`);row.forEach((v,c)=>a.push(`set value of cell "${String.fromCharCode(65+c)}${r+1}" to ${q(v)}`));});a.push('end tell');end=y+rh*p.table.length+20;
   }else if(p.chart){
    a.push(`add chart s row names {"Macro F1"} column names ${q(p.chart.labels)} data ${q([p.chart.values])} type horizontal_bar_2d group by chart row`,'set c to last chart of s','set position of c to {45,85}','set width of c to 870','set height of c to 245');end=347;
   }else end=105;
   const maxY=p.footer?460:487;
   const h=Math.min(65,(maxY-end)/p.lines.length);
   p.lines.forEach((t,j)=>text((/^\d\./.test(t)?'':'•  ')+t,50,end+j*h,860,h,p.font||(p.table||p.chart?20:23)));
   if(p.footer)text(p.footer,50,485,850,32,13,false,true);
  }
  text(String(i+1),903,516,35,20,13);
 });
 a.push(`save d in POSIX file ${q(path.join(build,'native.key'))}`,`export d to POSIX file ${q(path.join(build,'candidate.pptx'))} as Microsoft PowerPoint`,'close d saving no','end tell');
 fs.writeFileSync(path.join(build,'build.applescript'),a.join('\n'));
 fs.writeFileSync(path.join(build,'metadata.json'),JSON.stringify({slides,charts:slides.flatMap((s,i)=>s.chart?[{...s.chart,slide:i+1}]:[])},null,2));
 fs.mkdirSync(path.join(root,'docs'),{recursive:true});
 fs.writeFileSync(path.join(root,'docs/progress_presentation_20261116_notes.md'),'# Implementation progress presentation\n\nFormal black-and-white update to the 7 October presentation. Slides 1–22 form the main talk (about 15 minutes). Slides 23–33 are technical backup. Speaker notes are also embedded in the editable PowerPoint.\n\nThe transfer study was resumed on 16 November 2026. Its counters change during training; slide 20 deliberately gives the fixed study scope and no partial score. Check the live monitor before presenting.\n\n'+slides.map((p,i)=>`## ${i+1}. ${p.title}\n\n${p.notes}\n\nEvidence:\n\n${p.sources.map(x=>'- '+(x.startsWith('http')?x:`[${x}](../${x})`)).join('\n')}\n`).join('\n'));
 console.log(`Prepared ${slides.length} slides (${slides.filter(x=>x.backup).length} backup).`);
}
function finalize(){
 // stdlib OOXML edits preserve native tables, charts, embedded workbooks and notes.
 const py=String.raw`import json,re,xml.etree.ElementTree as E
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path.cwd(); b=root/'tmp/presentations/progress_20261116'
meta=json.loads((b/'metadata.json').read_text()); N={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
for k,v in N.items(): E.register_namespace(k,v)
def sub(p,t,**a): return E.SubElement(p,'{'+N[t.split(':')[0]]+'}'+t.split(':')[1],a)
def fill(p,c):
 for n in list(p):
  if n.tag.split('}')[-1] in ('solidFill','noFill','gradFill','pattFill','blipFill','grpFill'):p.remove(n)
 sub(sub(p,'a:solidFill'),'a:srgbClr',val=c)
def serialize(r): return E.tostring(r,encoding='utf-8',xml_declaration=True)
with ZipFile(b/'candidate.pptx') as z: parts={n:z.read(n) for n in z.namelist()}
for name in list(parts):
 if not re.fullmatch(r'ppt/slides/slide\d+\.xml',name): continue
 r=E.fromstring(parts[name])
 for tbl in r.findall('.//a:tbl',N):
  rows=tbl.findall('a:tr',N)
  for ri,row in enumerate(rows):
   for cell in row.findall('a:tc',N):
    pr=cell.find('a:tcPr',N)
    # Booktabs: no vertical grid, top rule, header rule, final rule.
    for edge in ('lnL','lnR','lnT','lnB'):
     e=pr.find('a:'+edge,N)
     if e is None:e=sub(pr,'a:'+edge)
     e.clear()
     if (edge=='lnT' and ri==0) or (edge=='lnB' and ri in (0,len(rows)-1)):
      e.set('w','9500');fill(e,'000000');sub(e,'a:prstDash',val='solid')
     else: sub(e,'a:noFill')
    fill(pr,'FFFFFF')
 for sp in r.findall('.//p:sp',N):
  body=sp.find('p:txBody/a:bodyPr',N)
  if body is not None:
   body.set('lIns','0');body.set('rIns','0');body.set('tIns','0');body.set('bIns','0')
 # Keynote resizes text around the initial box centre when fonts change.
 # Restore the authored body grid after export, keeping the text editable.
 idx=int(re.search(r'slide(\d+)',name)[1])-1;spec=meta['slides'][idx]
 if not spec.get('cover'):
  end=93+spec.get('rowH',38)*len(spec['table'])+20 if 'table' in spec else 347 if 'chart' in spec else 105
  h=min(65,((460 if 'footer' in spec else 487)-end)/len(spec['lines']))
  shapes=r.findall('.//p:sp',N)
  for j,line in enumerate(spec['lines']):
   sp=shapes[j+1];off=sp.find('p:spPr/a:xfrm/a:off',N);ext=sp.find('p:spPr/a:xfrm/a:ext',N)
   off.set('x','635000');off.set('y',str(round((end+j*h)*12700)))
   ext.set('cx',str(860*12700));ext.set('cy',str(round(h*12700)))
   body=sp.find('p:txBody/a:bodyPr',N);body.set('anchor','t')
   for child in list(body):
    if child.tag.split('}')[-1] in ('spAutoFit','normAutofit','noAutofit'):body.remove(child)
   sub(body,'a:noAutofit')
   ts=sp.findall('p:txBody/a:p/a:r/a:t',N)
   assert ts
   ts[0].text=('' if re.match(r'\d\.',line) else '•  ')+line
   for t in ts[1:]:t.text=''
 if name=='ppt/slides/slide1.xml':
  for sp in r.findall('.//p:sp',N)[:-1]:
   for par in sp.findall('p:txBody/a:p',N):
    pr=par.find('a:pPr',N)
    if pr is None:pr=E.Element('{'+N['a']+'}pPr');par.insert(0,pr)
    pr.set('algn','ctr')
 parts[name]=serialize(r)
names=sorted([n for n in parts if re.fullmatch(r'ppt/charts/chart\d+\.xml',n)],key=lambda n:int(re.search(r'chart(\d+)',n)[1]))
assert len(names)==len(meta['charts'])
for name,spec in zip(names,meta['charts']):
 r=E.fromstring(parts[name]); bar=r.find('c:chart/c:plotArea/c:barChart',N);ser=bar.find('c:ser',N)
 for n in r.findall('.//a:latin',N): n.set('typeface','Times New Roman')
 for n in r.findall('.//a:defRPr',N): n.set('sz','1800');fill(n,'000000')
 for ax in r.findall('.//c:valAx',N):
  sc=ax.find('c:scaling',N)
  for e in list(sc):
   if e.tag.split('}')[-1] in ('min','max'):sc.remove(e)
  sub(sc,'c:max',val=str(spec['max']));sub(sc,'c:min',val='0')
  u=ax.find('c:majorUnit',N)
  if u is not None:u.set('val','0.2')
  f=ax.find('c:numFmt',N)
  if f is not None:f.set('formatCode','0.0')
 for ax in r.findall('.//c:catAx',N)+r.findall('.//c:valAx',N):
  for tag in ('majorGridlines','minorGridlines'):
   n=ax.find('c:'+tag,N)
   if n is not None:ax.remove(n)
 layout=r.find('c:chart/c:plotArea/c:layout/c:manualLayout',N)
 if layout is not None:
  for k,v in {'x':'0.29','y':'0.05','w':'0.63','h':'0.80'}.items():
   n=layout.find('c:'+k,N)
   if n is not None:n.set('val',v)
 fill(ser.find('c:spPr',N),'606060')
 labels=ser.find('c:dLbls',N)
 labels.find('c:showVal',N).set('val','1');labels.find('c:dLblPos',N).set('val','outEnd');labels.find('c:numFmt',N).set('formatCode','0.000')
 for n in labels.findall('.//a:defRPr',N):n.set('sz','1900');n.set('b','1');fill(n,'000000')
 for i in range(len(spec['values'])):
  d=E.Element('{'+N['c']+'}dPt');sub(d,'c:idx',val=str(i));sp=sub(d,'c:spPr');fill(sp,'404040' if spec['slide']==13 and i==len(spec['values'])-1 else '929292');ser.insert(list(ser).index(labels),d)
 g=bar.find('c:gapWidth',N)
 if g is not None:g.set('val','75')
 vals=[float(v.text) for v in r.findall('.//c:val/c:numRef/c:numCache/c:pt/c:v',N)]
 assert all(abs(a-b)<1e-9 for a,b in zip(vals,spec['values']))
 parts[name]=serialize(r)
assert len([n for n in parts if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)])==len(meta['slides'])
assert len([n for n in parts if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)])==len(meta['slides'])
out=root/'output/pptx/mi_progress_since_october_2026.pptx';out.parent.mkdir(parents=True,exist_ok=True)
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for n,v in parts.items():z.writestr(n,v)
print(out)
`;
 const r=spawnSync(path.join(root,'.venv/bin/python'),['-c',py],{cwd:root,encoding:'utf8'});if(r.status!==0)throw new Error(r.stderr);console.log(r.stdout);
}
const stage=process.argv[2];
if(stage==='prepare')prepare();
else if(stage==='build')run('/usr/bin/osascript',[path.join(build,'build.applescript')]);
else if(stage==='finalize')finalize();
else if(stage==='pdf'){
 fs.mkdirSync(path.dirname(pdf),{recursive:true});
 const f=path.join(build,'export.applescript');
 fs.writeFileSync(f,`tell application "Keynote"\nset d to open POSIX file ${q(ppt)}\nexport d to POSIX file ${q(pdf)} as PDF\nclose d saving no\nend tell`);run('/usr/bin/osascript',[f]);
}else throw new Error('Expected prepare, build, finalize or pdf');
