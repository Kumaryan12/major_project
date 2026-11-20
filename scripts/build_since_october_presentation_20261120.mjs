/** Complete progress AFTER the 7 October presentation, through 19 November.
 * Uses the existing formal native Keynote authoring and OOXML validation path.
 * node scripts/build_since_october_presentation_20261120.mjs prepare|build|finish|pdf
 */
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {slides as earlier} from './build_progress_presentation_20261116.mjs';
import {slides as recent,prepare} from './build_progress_update_20261120.mjs';

const root=process.cwd(),build=path.join(root,'tmp/presentations/since_october_20261120');
const pptx=path.join(root,'output/pptx/mi_complete_progress_since_october_20261120.pptx');
const pdf=path.join(root,'output/pdf/mi_complete_progress_since_october_20261120.pdf');
const guide=path.join(root,'docs/complete_progress_since_october_20261120_presenter_guide.md');
const R=n=>`results/${n}/complete/REPORT.md`;
const slides=[];
const clean=p=>{const q=structuredClone(p);delete q.footer;if(q.chart)q.chartH=236;if(q.table){q.font=q.table[0].length>3?20:22;q.rowH=Math.max(q.rowH||40,39);}return q;};
const use=(p,overrides={})=>slides.push({...clean(p),...overrides});
const add=(title,lines,notes,sources,extra={})=>slides.push({title,lines,notes,sources,...extra});
const append=(p,overrides={})=>use(p,{title:p.title.replace(/^Backup: /,'Appendix: '),backup:true,...overrides});

use(recent[0],{
 lines:[...recent[0].lines.slice(0,6),'Progress since 7 October 2026','20 November 2026'],
 notes:'This is the complete implementation update after the presentation dated 7 October 2026. We first recall the tensor reproduction and patient benchmark. We then explain the waveform, tensor, sequence, transfer and diagnostic studies completed since then. Our primary task remains six-class localization from measured Frank XYZ on unseen patients.',
 sources:['docs/implementation_presentation_notes.md','output/latex/mi_implementation.tex','docs/CODEX_HANDOFF.md']
});
add('What we had completed in October',[
 'We reproduced the tensor pipeline and established a patient-disjoint benchmark.',
 'Wavelet statistics gave 0.487 patient macro F1 on PTB.',
 'The external result was weaker. Neural models were still proposed work.'
], 'The original presentation is dated 7 October 2026. Random-beat and patient-grouped rows below are beat accuracy on the same 12-class reproduction task. The six-class patient endpoint is different. The nested wavelet result is 0.4872 and the older external model scored about 0.242 on 1,092 PTB-XL patients. The October presentation reported 15 passing tests. It proposed compact neural models and hybrids, which we subsequently implemented.', ['docs/implementation_presentation_notes.md','output/latex/mi_implementation.tex','RESULTS.md','ABLATION_RESULTS.md','EXTERNAL_RESULTS.md'],{
 table:[['October experiment','Reported result'],['Random-beat reproduction, 12 classes','99.14% beat accuracy'],['Same cohort with patient groups','41.39% beat accuracy'],['Six-class nested Tucker + XGBoost','0.403 patient macro F1'],['Six-class nested wavelet statistics','0.487 patient macro F1'],['PTB-XL external test of the old model','0.242 patient macro F1']],widths:[585,275],rowH:39
});
add('Progress since the October presentation',[
 'These procedures use the same PTB development endpoint with different recipes.',
 'The score is now approximately 0.60. Independent confirmation remains necessary.'
], 'The historical October score was 0.4872. The first ResNet three-seed mean is 0.550 with sample seed SD 0.040. The neural residual-context ensemble is a separate post-hoc procedure at 0.584570. The selected linear convolution ensemble is 0.603374, and the fixed-eight-epoch waveform ensemble is 0.601801. Mean member F1 and ensemble F1 are different endpoints. The numerical point increase from October wavelet to selected convolution is about 0.116, but this historical comparison does not isolate one cause or establish performance on a new cohort. Do not call this SOTA or a guarantee of 0.80 to 0.90.', ['ABLATION_RESULTS.md','results/neural_repeat_v1/REPORT.md',R('neural_residual_seed_ensemble_v1'),R('distribution_kernel_v1'),R('neural_vcg_native_pretraining_v1')],{
 table:[['Completed procedure','Macro F1','Summary'],['October wavelet features','0.487','Nested procedure'],['Waveform ResNet repeat','0.550 ± 0.040','Mean ± seed SD'],['Neural residual-context model','0.584570','Three-seed ensemble'],['Selected linear convolution','0.603374','Three-seed ensemble'],['Fixed-eight-epoch waveform','0.601801','Three-seed ensemble']],widths:[435,170,255],rowH:39,font:21
});
use(recent[3],{lines:[...recent[3].lines.slice(0,2),'Patient macro F1 gives each of the six classes equal weight.',...recent[3].lines.slice(3)]});
use(recent[4]);
add('The implemented research pipeline',[
 'The measured XYZ bank supports both engineered features and learned waveforms.',
 'Training patients alone determine preprocessing, tuning and stopping.',
 'Held-out patient predictions and saved-model replay support each report.'
], 'This diagram summarizes the implementation since October. Some routes classify beats before pooling probabilities. Other routes build recording summaries and then pool records into patient features. Exact aggregation is declared per experiment. Every fitted transform belongs to its corresponding training partition. Inner validation selects settings where the protocol calls for selection. Outer folds retain patients unseen by that fitted model. Independent replay checks implementation, while a reserved external cohort is still needed for confirmation.', ['docs/CODEX_HANDOFF.md','configs/patient_folds.csv'],{diagram:'pipeline'});
use(earlier[4],{title:'Waveform ResNet: proposed in October, now trained'});
use(earlier[5],{
 title:'Carefully constructed multiview tensors',
 lines:['We separated typical recording shape, beat variation and denoising residuals.','Reconstruction checks preserve the original signed signal.','The feature studies completed. The older multiview neural sweep stays paused.'],
 diagram:'multiview'
});
use(earlier[6]);
use(earlier[7]);
use(earlier[8],{title:'Consecutive heartbeats and RR timing'});
add('Direct morphology and residual context',[
 'A direct beat path preserves morphology while context adds a gated correction.',
 'The original mean-model comparison did not establish a gain.',
 'A separate post-hoc ensemble reached 0.584570, with an uncertain paired gain.'
], 'All six original arms, three seeds and 720 fitted contexts remain reported. Mean member F1 averages separate F1 scores. Ensemble F1 averages probabilities first and scores one predictor. The ordered-plus-auxiliary mean is 0.539 versus beat-only 0.551. Its later equal-seed ensemble scores 0.584570 versus matched beat-only ensemble 0.545112. Ensemble difference +0.039458 has interval [-0.034767,+0.112701]. All 180 outer members were restored. The ensemble does not rewrite the source mean-model conclusion or prove ordering is useful.', [R('neural_residual_context_v1'),R('neural_residual_seed_ensemble_v1')],{
 table:[['Matched procedure','Mean member F1','Ensemble F1'],['Beat only','0.551','0.545112'],['Ordered context + auxiliary loss','0.539','0.584570']],widths:[440,210,210],rowH:46
});
use(earlier[11]);
use(earlier[12]);
use(earlier[13]);
use(recent[13],{
 title:'Source learning and supervised transfer',
 lines:['A separate PTB-XL task learned native diagnostic representations.','Target inference uses measured XYZ and fresh six-class heads.','Transfer minus scratch: +0.0346, interval [-0.0465, +0.1138].'],
 notes:'This complete source-to-target route was implemented after October. The bounded supervised source used 4,096 training patients and 1,942 validation patients, with source fold 10 excluded before labels and signals were read. Both source objectives trained 15 epochs, selecting epoch 13. Their source macro AP was about 0.317, a different multilabel endpoint. Source and overlap verification preceded target fitting. The matched target study completed all 630 fits and 90 outer models. The certain-source gain is uncertain. Source-target patient identity remains unresolved despite a passed recording-copy screen. Detailed source cohort, objectives, metrics and overlap results are in the appendix.',
 sources:[...recent[13].sources,'results/supervised_source_pilot_v1/REPORT.md','results/supervised_source_overlap_v1/REPORT.md']
});
use(recent[5],{lines:['A median waveform can hide variation between the individual beats.','We encoded every original beat before pooling.','Mean describes the centre. Spread and quantiles describe variation.']});
add('Distribution tensors: what changed',[
 'We tested physical features, intact convolution features and learned embeddings.',
 'Quantile gains over matched mean/spread controls remained uncertain.',
 'More distribution detail did not consistently improve localization.'
], 'Each table row is a different matched study on the original 164 patients. The physical study keeps a 720-column tensor across all four arms. The convolution and learned studies keep the intact 3,584-column original base and append 144 columns. Their baseline is 0.584266. Every beat was replayed in each respective feature study. The physical quantile-minus-moments interval is [-0.052573,+0.076160]. Intact-convolution quantiles minus spread is -0.004613 with interval [-0.013297,0]. Learned quantiles minus spread is +0.033028 with interval [-0.011283,+0.079062]. Full arm results remain in the appendix.', [R('distribution_pooling_v1'),R('convolution_distribution_v1'),R('learned_distribution_v1')],{
 table:[['Matched feature family','Mean/spread F1','Quantile F1'],['Physical and DCT features','0.475709','0.487114'],['Intact convolution base + summaries','0.594098','0.589485'],['Intact base + learned summaries','0.547867','0.580895']],widths:[460,200,200],rowH:43
});
use(recent[9]);
add('Further tensor hypotheses and controls',[
 'Each method keeps the same patients, original labels and frozen folds.',
 'These tests did not establish a reliable gain or validated clinical timing.'
], 'The within-beat lag tensor compares signed direction, magnitude and balanced sign scrambling. Signed 0.531364 is below the original base 0.584266 and scrambled 0.546633. Signed minus magnitude remains uncertain. Subtype graph regularization compares the clinical graph with ridge and a relabeled graph, keeping HC isolated. Subtype graph minus ridge is +0.007158 with interval [-0.012932,+0.028282]. The local timing audit replays all 391 records and injected controls. None of the three post-R windows met its required patient coverage, so the full timing tensor was deferred before classifier fitting.', ['docs/lag_direction_tensor_findings.md','docs/subtype_graph_findings.md','docs/local_timing_audit_findings.md'],{
 table:[['Hypothesis','Completed result'],['Signed within-beat direction','0.531364 versus base 0.584266'],['Subtype graph regularization','0.591424 versus ridge 0.584266'],['Local timing estimator','Failed the predefined coverage gate']],widths:[415,445],rowH:49
});
add('Additional source-learning pilots',[
 'All four extensions use measured XYZ alone at target inference.',
 'None demonstrated a gain over its matched scratch ensemble of 0.601801.',
 'The source questions and their matched controls differ.'
], 'Native masked reconstruction added 124 unlabeled measured-XYZ patients. It scored 0.518364 versus target-only pretraining 0.454298, but remained below scratch. Correct ECG pairing scored 0.564293 versus shuffled 0.529345, with an uncertain pairing difference. ACS OMI-plus-artery scored 0.580730 versus OMI-only 0.570389. STAFF paired-change scored 0.539589 versus pair-mean 0.555118. Each study has separately replayed matched models. ECG-derived source vectors differ from measured Frank target vectors, and source native labels are not substituted for PTB class labels. Individual sources and controls are detailed in the appendix.', ['docs/vcg_pretraining_pilots_findings.md',R('neural_acs_artery_transfer_v1'),R('neural_staffiii_pair_transfer_v1')],{
 table:[['Source signal or supervision','Target ensemble F1'],['Additional native XYZ reconstruction','0.518364'],['Paired ECG teacher','0.564293'],['ACS-ECG OMI + artery','0.580730'],['STAFF III paired changes','0.539589']],widths:[630,230],rowH:39
});
add('Supervision and patient-level classifiers',[
 'Partial region mentions, hierarchical probabilities and patient losses were tested.',
 'Their declared matched comparisons did not establish a reliable gain.',
 'Frozen score corrections and equal model averaging also failed to improve the references.'
], 'The partial-label study retains positive-only native header mentions from 13 outside-target patients, with a tag-count-matched shuffled control. True 0.575887 versus shuffled 0.568947 is uncertain and below scratch 0.601801. Hierarchical classification scored 0.586828 versus flat 0.562819, with interval for the difference [-0.001207,+0.056060]. Patient-level cross-entropy scored 0.469596 versus beat cross-entropy 0.510134 under the same grouped-batch sampler. This sampler differs from the older 0.601801 waveform procedure. Frozen score corrections leave all original ensemble decisions unchanged. Equal convolution-waveform blend 0.582209 fails to beat either component. R-relative aligned pooling likewise did not beat shuffled pooling. Complete results and matched controls are in the appendix.', [R('neural_native_partial_labels_v1'),R('probabilistic_subtype_v1'),R('neural_patient_probability_loss_v1'),R('distribution_score_correction_v1'),'results/model_error_comparison_v1/REPORT.md',R('r_relative_convolution_v1')],{
 table:[['Matched comparison','Candidate / control F1'],['True / shuffled partial labels','0.575887 / 0.568947'],['Hierarchical / flat classifier','0.586828 / 0.562819'],['Patient / beat cross-entropy','0.469596 / 0.510134']],widths:[550,310],rowH:43
});
use(recent[20]);
use(recent[21]);
use(recent[22],{
 title:'Beat count and signal quality checks',
 lines:['Matching beat counts preserved the higher point score for later records.','The intervals remain broad. The quality proxies did not explain the gap.','Independent clinical and recording-timing review remains pending.'],
 notes:recent[22].notes+'\n\n'+recent[23].notes,
 sources:[...recent[22].sources,...recent[23].sources]
});
add('Training specifically on the earliest record',[
 'The primary endpoint here uses the same 145 dated patients.',
 'Distillation, longer training and convolution features did not establish a gain.',
 'These scores answer a different question from all-record macro F1 near 0.60.'
], recent[24].notes+'\n\n'+recent[25].notes, [...recent[24].sources,...recent[25].sources],{
 table:[['Earliest-record procedure','Ensemble macro F1'],['Eight-epoch supervised reference','0.521335'],['Later-record distillation','0.523769'],['Inner-selected 8/16/32 epochs','0.509127'],['Earliest-record convolution','0.478122']],widths:[630,230],rowH:39
});
use(recent[26],{
 lines:['We froze the ResNet encoder and fitted patient-level classifiers.','Ridge on 64 features scored 0.535691. Retaining 256 features scored 0.501020.','The larger readout did not establish an improvement on the earliest-record task.']
});
use(recent[27]);
use(recent[28]);
use(recent[29]);
use(recent[30]);
use(recent[31],{
 lines:['The package expanded from 15 tested cases in October to 376 before the latest study.','Versioned protocols, frozen folds and matched controls define each experiment.','Independent reports replay saved features, models, selections and predictions.','Partner-cohort intake is prepared. Real data and clinical review are pending.','Completed studies are stopped. The older multiview sweep stays paused.'],
 sources:[...recent[31].sources,'docs/implementation_presentation_notes.md']
});
use(recent[32]);
use(recent[33],{
 title:'What we have completed since October',
 lines:['Trained the neural models that were proposed in the October presentation.','Implemented richer tensors, feature learning, sequence models and source transfer.','Raised historical all-record point results from 0.487 to approximately 0.60.','Audited recording choice and prepared a pathway to additional data.','The 0.80-0.90 target and independent external confirmation remain open.'],
 notes:'The main presentation ends here. October is the baseline for the entire update. The project progressed through model implementation, controlled comparisons and stronger development point results, while preserving patient separation. The historical 0.487-to-0.60 comparison spans different procedures and repeated development. The appendix covers unsuccessful branches, all important controls, member-versus-ensemble distinctions, class results and source evidence. More experiments alone do not establish SOTA or guarantee a clinical result.'
});

// Detailed backup slides preserve the complete research trail without overloading the main talk.
for(const i of [22,23,24,25,26,27,28,29]){
 const p=clean(earlier[i]);delete p.footer;
 append(p,{lines:p.table?[]:p.lines,rowH:Math.max(43,Math.min(50,Math.floor(390/(p.table?.length||6)))),font:p.table?.[0]?.length>3?20:21});
}
for(const i of [34,35,36,37,38,39,40,41,42,43,44])append(recent[i]);
for(const i of [15,16,17,18])append(earlier[i]);
append(recent[11],{title:'Appendix: within-beat direction tensor'});
append(recent[17],{title:'Appendix: all partial-label arms'});
append(recent[10],{title:'Appendix: frozen score corrections'});
append(recent[18],{title:'Appendix: R-relative and subtype readouts'});
append(recent[23],{title:'Appendix: paired recording quality'});
add('Appendix: training-patient learning curve',[
 'More training patients improved the historical waveform result.','The curve does not establish a ceiling or predict the sample size needed for 0.80.','Error bars describe variation across the declared training seeds.'
], 'Existing completed project figure. The plotted patient budgets and fixed held-out folds belong to the original learning-curve study. Approximate macro F1 rose from 0.407 to 0.550 as the training budget rose. This is an observed data-budget experiment, not a forecast for a new dataset. The original figure is unchanged. Source image and experiment metrics are cited below.', [R('neural_learning_curve_v1'),'results/neural_learning_curve_v1/complete/learning_curve.png'],{backup:true,image:'results/neural_learning_curve_v1/complete/learning_curve.png',imageBox:[45,115,600,375],side:true});
add('Appendix: completed source-learning curves',[
 'Both source objectives trained 15 epochs and selected epoch 13.','Source validation AP chooses the source checkpoint.','This figure reports source diagnostics, separate from target localization macro F1.'
], 'Existing unmodified aggregate source-only figure. Both declared annotation objectives and the prevalence-constant comparator remain. The PTB-XL source cohort and 28 diagnostic heads differ from the six-class Frank-XYZ target. Final target transfer results are completed and reported in the main deck. Source macro AP about 0.317 is not a localization score.', ['results/supervised_source_pilot_v1/REPORT.md','results/supervised_source_pilot_v1/learning_curves.png'],{backup:true,image:'results/supervised_source_pilot_v1/learning_curves.png',imageBox:[45,115,600,600*640/1120],side:true});
append(recent[45],{
 sources:[...recent[45].sources,'docs/progress_presentation_20261117_illustrated_notes.md','docs/implementation_presentation_notes.md','https://proceedings.mlr.press/v80/ilse18a.html','https://arxiv.org/abs/2004.11362']
});

// Strip historical bottom notes and keep every plot/table body large in the new deck.
for(const p of slides){delete p.footer;if(p.table&&!p.widths)throw Error('Missing table geometry: '+p.title);}
const scopeDescription='Covers all completed work after the 7 October 2026 presentation through 19 November 2026. The November decks are intermediate artifacts, not the presentation baseline.';
fs.mkdirSync(build,{recursive:true});
const run=(cmd,args)=>{const p=spawnSync(cmd,args,{cwd:root,encoding:'utf8',maxBuffer:4*1024*1024});if(p.status!==0)throw Error(p.stderr||p.stdout);if(p.stdout)process.stdout.write(p.stdout);};
switch(process.argv[2]){
 case 'prepare':prepare({slideSpecs:slides,buildDir:build,scopeDescription});break;
 case 'build':run('osascript',[path.join(build,'build.applescript')]);break;
 case 'finish':run('.venv/bin/python',['scripts/finalize_progress_update_20261120.py','--build-dir',build,'--pptx',pptx,'--pdf',pdf,'--guide',guide]);break;
 case 'pdf':run('osascript',[path.join(build,'export_pdf.applescript')]);break;
 default:throw Error('Expected prepare, build, finish or pdf');
}
