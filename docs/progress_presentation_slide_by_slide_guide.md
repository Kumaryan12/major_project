# Slide-by-slide learning guide: implementation progress

For Aryan's implementation presentation. Read alongside [the illustrated PDF](../output/pdf/mi_progress_since_october_2026_illustrated.pdf). This is an explanation of that saved deck, not an edited presentation. It covers all 35 slides: 24 main slides and 11 backup slides.

Prepared 17 November 2026. Interpret slide numbers as PDF page numbers. Suggested spoken passages are explanations you can adapt, not words you must memorize.

## Before the slides: the story in plain language

We want a computer to predict one of five myocardial-infarction location categories or healthy control from measured three-direction VCG signals. We first reproduced the paper and built patient-separated baselines. Since the October presentation, we have trained waveform neural networks, built several signal/tensor representations, tested beat sequences and ensembles, and investigated learning from a larger source dataset. Some methods improved a matched development result; many added complexities did not. We now have stronger reference models and a reproducible evaluation process, but have not achieved the aspirational 0.80–0.90 macro F1 or independently established clinical generalization.

The presentation is primarily about what was implemented and what the comparisons taught us. It does not repeat the whole paper, and it does not establish that our model is ready for clinical diagnosis.

### Six ideas you must understand first

1. **Patient, recording, beat:** one patient can have multiple recordings; each recording contains many beats. We have 164 independent target patients, not 55,863 independent people.
2. **Representation:** the information supplied to a classifier. Examples are the waveform itself, waveform statistics, a tensor, or responses to many convolution filters.
3. **Patient macro F1:** calculate an F1 score for each of the six classes, then average those six scores equally. It is not accuracy. A 0.58 macro F1 does not mean that exactly 58% of patients were classified correctly.
4. **Matched control:** a comparison designed to keep other important conditions alike. It is stronger evidence about a change than comparing it with a number from a different historical procedure.
5. **Seed mean versus ensemble:** average three model scores to describe individual-run performance; average their probability predictions first to make an ensemble. These are different operations.
6. **Development evidence:** these PTB patients have informed repeated research decisions. Patient-separated evaluation remains important, but these are not fresh, untouched confirmation patients.

### The evaluation, without jargon

For one outer fold, set aside roughly one tenth of the patients. Use only the remaining patients to fit and choose the model. Within those training patients, three inner folds help choose settings and stopping. Refit using the chosen settings on that outer training partition, then predict the held-out patients. Repeat until every patient has a held-out prediction. All beats and recordings of one person stay together.

Training-only also applies to normalization, tensor bases, reference patients, feature selection and learned blends. Computing an examination's own beat summary for inference is different from fitting a population normalizer using test patients.

For a class c:

`precision = TP / (TP + FP)`

`recall = TP / (TP + FN)`

`F1_c = 2 TP / (2 TP + FP + FN)`

`macro F1 = (F1_AMI + F1_ALMI + F1_ASMI + F1_IMI + F1_ILMI + F1_HC) / 6`

Here AMI means **anterior MI** in our target class mapping; do not expand it as acute MI in this six-class table. ALMI is anterolateral, ASMI anteroseptal, IMI inferior, ILMI inferolateral, and HC healthy control. These are the project's dataset categories, not newly adjudicated anatomical labels. See [class mapping](../src/mi_localization/labels.py).

### Small glossary for later slides

| Term | Meaning in this project |
|---|---|
| Morphology | Waveform shape, including amplitude and relative timing |
| Embedding | A numerical feature vector produced by an encoder |
| Head | The final prediction layer; six outputs for the exclusive target classes |
| Tensor | A multi-axis array; the axes describe what each position means |
| Wavelet band | A component describing signal behavior at a particular temporal scale |
| DCT | Discrete cosine transform; represents a signal using cosine-basis coefficients |
| Rank / compression | How many basis directions a compact representation retains |
| Residual | A difference, such as a denoised beat minus its recording median; not automatically noise |
| RR interval | Time between successive detected R peaks |
| Moment pooling | Summarizing a distribution by quantities such as mean and standard deviation |
| Quantile | A value at a specified position in the sorted observations; the median is the 50th percentile |
| Ridge | A linear classifier with a penalty that constrains coefficient size |
| Auxiliary loss | An additional training objective, such as requiring beat embeddings to predict the class |
| Frozen encoder | Encoder weights are held fixed while another component trains |
| Fine-tuning | Continuing parameter learning on the target task after source training |
| Checkpoint | Saved fitted parameters at a particular training stage |
| Bootstrap interval | Here, a descriptive range formed by resampling patients and recomputing a difference using fixed predictions |
| Interval includes zero | The conditional resampling result does not establish a reliably positive difference; it does not prove equality |

### Updates since this PDF was prepared

Keep these separate from the original slide content:

- **Slides 20 and 22–24:** supervised transfer has completed. Ensemble scratch **0.538736**, certain-source **0.573361**, intermediate-positive source **0.549364**. Primary certain−scratch **+0.034625**, descriptive paired interval **[−0.046511,+0.113800]**: no demonstrated reliable gain. All 630 fits / 90 outer models / 210 matched checks verified. [Completed report](../results/neural_supervised_transfer_v1/complete/REPORT.md).
- **Slide 24:** the first compact physical/DCT distribution study has also completed: quantiles **0.487114**, mean/spread **0.475709**, primary difference **+0.011405 [−0.052573,+0.076160]**, uncertain. It did not exceed the separate historical references. [Completed report](../results/distribution_pooling_v1/complete/REPORT.md).
- **Slides 23–24:** a separately authorized stronger convolution-distribution follow-up is now active. It preserves the entire original convolution feature vector and adds mean, moment or quantile summaries of all-beat residuals. It has no validated final score yet. Latest full suite at its launch: **309 tests**, rather than the deck's historical 289. [Protocol](convolution_distribution_plan.md). Use `python scripts/convolution_distribution_status.py` for actual local status; this guide is not a live monitor.
- **Slide 34:** the transfer verification mentioned as pending has now passed. The older multiview neural sweep remains paused.

## Slide 1 — Implementation progress

**Purpose:** introduce your part of the presentation and its time period.

**Understand:** “patient-independent” means the people used to evaluate a fitted model are excluded from that model's training and model selection. “Localization” means predicting the dataset's MI-location category, rather than only deciding MI versus healthy. Your friend explains the original paper and motivation; you explain the implementation and evidence since 7 October.

**Say:** “My friend has explained the paper and the problem. I will now describe what we implemented after our previous presentation, how we evaluated it on separate patients, and what the completed experiments showed.”

**Likely question:** “Is this only reproduction?” Answer: reproduction was the starting point; later work includes controlled extensions in representation, waveform learning, context, ensembling and source transfer. Their benefit has to be demonstrated separately.

**Transition:** first show the broad progress, then explain why the evaluation is credible.

## Slide 2 — Progress since the previous presentation

**Purpose:** give three reference points without implying they are identical experiments.

**Read the table:** October nested feature procedure **0.487**; later plain waveform ResNet **0.550 ± 0.040** across three seeds; separate ordered-context-plus-auxiliary **probability ensemble 0.584570**.

The historical difference 0.550−0.487 is **0.063 macro F1**, or 6.3 points on a 0–100 F1 scale. The 0.550 is an average of individual scores. The 0.584570 is the score of a combined predictor from a different procedure. Do not describe the entire table as one matched ablation or call these numbers accuracy.

**Say:** “Waveform learning was future work in October. We have now trained it across three seeds and investigated stronger representations and ensembles. These are useful historical progress points, while the later slides show each method's matched comparison.”

**Likely question:** “Did a single model improve to 0.585?” Answer: the 0.584570 figure is a three-model probability ensemble; its original mean member score was lower and is reported separately.

## Slide 3 — Unchanged patient-level evaluation

**Purpose:** explain what counts as a fair prediction.

**Understand:** **164 patients → 391 recordings → 55,863 beats**. The 651-sample beat crops contain three measured XYZ channels. Ten outer folds evaluate held-out patients; three inner folds in later studies choose settings. A person's beats must never be scattered between training and testing just because they look like separate rows.

Think of one patient with 200 beats: predicting another beat from that same person is easier than predicting a new person's MI category. Grouping by patient addresses that problem. Macro F1 weights all six classes equally, so strong HC performance cannot hide weak MI-subtype performance.

**Say:** “Our independent unit is the patient. We keep every recording and beat of a patient together, fit preprocessing on training patients only, and report six-class patient macro F1.”

**Likely question:** “Does nested cross-validation make this untouched validation?” Answer: it separates fitting and selection within each experiment, but repeated research on this same cohort makes the overall evidence developmental.

## Slide 4 — ECG illustration and measured VCG

**Purpose:** connect the signal-processing pipeline to a recognizable waveform.

**Look at the left image:** three plots are measured Frank X, Y and Z versus time. Recorded and denoised traces are overlaid. Dashed vertical lines mark detected R peaks. The shaded area marks one beat crop aligned to R: **250 ms before through 400 ms after**, 651 samples including both endpoints at 1,000 Hz.

**Look at the right image:** the generic ECG sketch helps orient the audience to P, QRS and T landmarks. It is not the project's measured input or evidence that we automatically located clinical boundaries correctly. Keep the explanation focused on alignment rather than making a diagnosis from this example.

**Say:** “We locate R peaks and extract consistent three-channel beat windows. The actual task uses measured XYZ VCG; the ECG sketch is included only to explain the timing landmarks.”

**Likely question:** “Is the R here the raw view R?” Answer: R peak is a timing landmark. On the later multiview slide, the symbol R denotes the raw waveform. The notation has two meanings in different contexts.

## Slide 5 — Implemented pipeline after October

**Purpose:** provide the common workflow behind many experiments.

**Understand the sequence:** preserve original signals and beat membership → build a representation → fit training-only preprocessing and classifier → select with inner patients → predict held-out patients → independently restore and check the saved result.

A waveform network learns its representation during training. A feature model may use statistics, tensors or convolution features. Some procedures average beat probabilities; others average recording features. These are declared design choices, so compare each with its own control.

**Say:** “The common pipeline is controlled representation construction, training-only model fitting, inner selection, held-out patient prediction, and independent restoration before reporting.”

**Likely question:** “What is new beyond choosing a classifier?” Answer: the implementation includes signal views, tensor construction, partition-specific learned transforms, aggregation controls and reproducible fitting/reporting—not only changing the final algorithm.

## Slide 6 — Waveform ResNet and repeatability

**Purpose:** explain the first major learned-waveform model and why multiple runs matter.

**Understand:** the input is a **3×651 waveform**. A 1D convolution slides along time and learns local patterns across channels. Residual connections allow a block to learn a correction while retaining a direct route for information. Widths 16/32/64 are learned channel counts, not beat counts. The network produces a 64-number embedding and six-class probabilities.

All held-out beat probabilities contribute to the patient prediction. Seeds **42/123/2026** give approximately **0.586/0.557/0.507**, averaging **0.550 ± 0.040**. Seeds change initialization/sampling randomness, not the frozen patient folds. The SD describes variation across those runs; it is not a patient confidence interval.

**Say:** “The ResNet learns directly from waveform shape. We repeated it with three declared seeds because reporting only the favorable 0.586 run would overstate repeatability.”

**Likely question:** “Why use an embedding?” Answer: it is a compact learned feature vector that the classification head can use; it is not itself a clinical label or guaranteed physiological explanation.

## Slide 7 — Carefully constructed multiview tensors

**Purpose:** explain the tensor construction concretely.

**Understand the five views:** raw signal **R**, denoised **D**, own-recording median beat **M**, beat variation **D−M**, and denoising residual **R−D**. The equalities `R=D+(R−D)` and `D=M+(D−M)` are numerical reconstruction checks. The residuals may contain signal as well as artifacts; calling all residual content noise would be unjustified.

The tensor axes are **beat × view × XYZ × wavelet band × time**. A tensor is a multi-axis array; it need not be a new neural architecture. Wavelet bands retain information at different temporal scales. The views reorganize existing measurements, not add sensors or patients.

**Say:** “We separated common recording morphology, beat variation and denoising residuals, then represented each view across XYZ, frequency scales and time.”

**Likely question:** “Did all multiview models finish?” Answer: the feature-first studies completed; the older full multiview neural sweep remains paused. Better reconstruction alone did not consistently improve classification.

## Slide 8 — Feature stability and tabular learning

**Purpose:** show that feature reliability matters as well as feature count.

**Understand:** bootstrap consensus resamples training patients and looks for features selected consistently. It must be repeated within each relevant training partition. XGBoost builds an ensemble of decision trees. TabPFN is a pretrained tabular prediction method; here it receives feature vectors, not raw ECG/VCG waveforms.

The matched tree goes **0.420→0.513** with bootstrap consensus. Local TabPFN reaches **0.542**, but its difference from the tree remains uncertain. The disclosed prior-correction variant is **0.498**. “Raw TabPFN” here means the uncorrected prediction procedure on the tensor features, not raw signal input.

**Say:** “Stable selection improved this matched tree pipeline. TabPFN gave a higher point score, but the uncertainty did not establish a reliable additional gain.”

**Likely question:** “Is selection always beneficial?” Answer: no; the later convolution-feature selection study reduced performance. The feature bank and procedure matter.

## Slide 9 — Relations, geometry and ordered paths

**Purpose:** summarize four attempts to add structured signal information.

**Understand:** covariance measures joint variation while retaining scale; correlation normalizes scale. Geometry describes the XYZ trajectory's shape. Healthy-reference features describe difference from training reference recordings. Path signatures summarize ordered within-beat trajectories using successive interaction terms.

Read each row as **candidate / its own matched control**: covariance **0.470/0.469**, adaptive geometry **0.467/0.445**, healthy deviation **0.479/0.499**, signature3 **0.508/0.515**. These controls differ between studies; the rows are not a direct common leaderboard.

**Say:** “We tested whether physiological structure improved the representation. None of these comparisons established a reliable overall gain.”

**Likely question:** “Does no gain mean the physiology is useless?” Answer: it means these particular representations and training procedures did not establish benefit; it does not prove a general impossibility.

## Slide 10 — Consecutive beats and RR timing

**Purpose:** test whether temporal ordering adds information beyond having multiple beats.

**Understand:** windows contain **4, 8 or 16 consecutive beats from one recording**. RR is the interval between successive detected R peaks. A shared beat encoder extracts waveform features, and a recurrent context model can combine them in order. Inner selection chooses a shared length for the ordered/shuffled comparison.

The shuffled control moves each beat together with its own RR value, changing ordering without inventing unrelated beat/timing pairs. Ordered+RR mean F1 **0.524** versus shuffled+RR **0.533**; difference **−0.008 [−0.057,+0.039]**. This did not demonstrate an ordering benefit. All original query beats remained covered.

**Say:** “We compared real chronology with the same beats and timing pairs shuffled. Chronology did not reliably outperform the shuffled control.”

**Likely question:** “Are four beats ideal?” Answer: not established. Length was chosen inside training folds from the declared options; the experiment did not demonstrate one universal optimum.

## Slide 11 — Direct morphology with residual context

**Purpose:** retain direct beat information while testing an additional context path.

**Understand:** morphology means waveform shape. The direct path predicts from a beat. The context branch adds a gated correction based on neighboring beats. An auxiliary beat loss encourages the beat representation to remain informative rather than only serving the sequence model.

Original mean-member results: matched beat-only **0.551 ± 0.027**, ordered+auxiliary **0.539 ± 0.032**, difference **−0.012 [−0.051,+0.027]**. This is the original training study. It does not show that the average individual model improved.

**Say:** “We preserved a direct morphology path and added context as a correction. The mean-seed comparison did not establish a gain.”

**Likely question:** “Why does the next slide show 0.585?” Answer: it evaluates a different predictor, the probability ensemble of all three members. It does not change these original individual-model results.

## Slide 12 — Uniform seed ensembles

**Purpose:** explain the higher ensemble score without confusing it with mean seed F1.

**Understand the operations:** mean member F1 is `(F1(model1)+F1(model2)+F1(model3))/3`. An ensemble first calculates `(p1+p2+p3)/3` for every patient, chooses the class with the highest mean probability, then calculates F1 from those labels. F1 and argmax are nonlinear, so the results can differ in either direction.

Beat-only ensemble **0.545112**; ordered+auxiliary ensemble **0.584570**; difference **+0.039458 [−0.034767,+0.112701]**. All three seeds and all six arms were retained. No new fitting or learned blending weights. The analysis is **post-hoc**: it was chosen after seeing earlier study results.

**Say:** “Uniform ensembling gave a promising higher point score, while its matched difference remains uncertain. It is a separate post-hoc development result.”

**Likely question:** “Why can ensembling help a weaker mean model?” Answer: members may make different errors and have different confidence patterns. That is a possible mechanism, not independently established causality in this dataset.

## Slide 13 — Fusion and staged context learning

**Purpose:** test combining useful components and protecting morphology during context training.

**Understand fusion:** blend neural and tensor probabilities using one scalar chosen on inner patients. Selected fusion **0.583052** versus neural **0.584570**, so no demonstrated improvement.

**Understand stages:** first train morphology, then either continue training it, jointly adapt it with context, or freeze its weights while training context. Frozen-context ensemble **0.563539** versus its matched morphology **0.545112**; uncertainty includes zero. Shuffled frozen context **0.565056** also prevents an ordering-benefit claim.

**Say:** “Neither inner-selected late fusion nor staged context established a reliable gain. The staged model's improvement in point score also did not depend on correct beat order.”

**Likely question:** “Why not a larger adaptive gate?” Answer: the simpler matched fusion did not establish a benefit; a more complex gate would need a separately justified study, not a claim based on these outcomes.

## Slide 14 — Random-convolution features with ridge

**Purpose:** introduce a strong alternative to training a waveform encoder.

**Understand:** for each recording, take the median across beats separately at each channel/time coordinate. Apply many fixed/random convolution filters and summarize their responses. Thresholds/scales estimated from data remain training-only. MultiRocket-style and experimental multivariate HYDRA produce complementary features. Ridge is a regularized linear classifier fitted on those features.

Ensemble scores: DCT **0.445781**, MultiRocket-style **0.561442**, HYDRA **0.548100**, combined **0.584266**. Combined−DCT **+0.138485 [+0.047546,+0.228367]** is a positive descriptive development contrast. Combined versus MultiRocket remains uncertain. The historical neural **0.584570** is a separate reference.

**Say:** “A rich fixed convolution feature bank with a simple ridge classifier reached a strong development score. Its recording-median input compresses beat-to-beat variation, which motivates our later distribution work.”

**Likely question:** “Is this entirely data-independent?” Answer: filter generation can be fixed/random, but fitted scales, thresholds and classifier state still use training patients only. Our bounded implementation is not identical to every upstream default.

## Slide 15 — Selected convolution features and TabPFN

**Purpose:** test whether a smaller, stable feature subset improves the strong convolution pipeline.

**Understand:** inside training partitions, bootstrap consensus and inner selection choose a 32- or 64-feature budget. Compare full ridge **0.584266**, selected ridge **0.544036**, selected PFN **0.530966**. AMI class F1 declines **0.345→0.080** under both selected procedures.

More compact does not automatically mean more informative. Features that are weak individually can still matter jointly, and a selector can remove useful information. Those are plausible explanations; the table alone does not prove the exact cause.

**Say:** “Feature compression and PFN did not improve the prespecified pipeline, so we retained the full-feature ridge reference.”

**Likely question:** “Can we just report the favorable fixed-64 result?” Answer: no. Those outer outcomes are disclosed controls; they cannot replace the training-only budget selector after the fact.

## Slide 16 — What the data diagnostics showed

**Purpose:** explain what we checked before blaming labels or adding complexity.

**Read the figure:** the horizontal axis is the training-patient budget. The dashed training line is high; the held-out patient line rises roughly **0.407→0.550** as more training patients are used. Error bars are training-seed SD, not independent-patient confidence intervals. Near-perfect training fit with lower held-out results indicates a generalization gap in this tested setup.

**Read the bullets:** 35 patients were wrong across all three selected ResNet seeds. Chronology checks found no simple normalization mismatch explaining its failure. These are diagnostics, not expert label correction or proof of a dataset ceiling.

**Say:** “More training patients helped within this completed study, but the curve cannot tell us an exact sample size required for 0.80 F1. Persistent errors need investigation, not automatic relabeling.”

**Likely question:** “Are those 35 labels wrong?” Answer: the experiment does not establish that. They may reflect ambiguity, model limitations, difficult signals or other factors.

## Slide 17 — Expanded supervised source cohort

**Purpose:** explain how a larger dataset is used without changing the target task.

**Understand:** PTB-XL is a separate **source** dataset for representation learning. The bounded cohort has **4,096 training patients/5,063 records** and **1,942 validation patients/2,183 records**. All 1,841 training patients with at least one likelihood-100 review-MI annotation are included, then the cohort is filled to the declared training budget. Source fold 10 is excluded.

Source XYZ is derived from twelve-lead ECG using a transformation; target XYZ is measured Frank VCG. Three channels do not make these acquisition methods interchangeable. Source visits remain grouped as declared and native diagnoses are retained rather than forced into target classes.

**Say:** “We expanded source training while keeping the original 164-patient measured-VCG target cohort and folds unchanged.”

**Likely question:** “Did we increase the number of independent target patients?” Answer: no. We added source-learning data, which may or may not transfer usefully to the original target endpoint.

## Slide 18 — Native source annotation objectives

**Purpose:** explain uncertain, overlapping source labels honestly.

**Understand:** a source recording can have several diagnoses, so it uses multiple binary outputs rather than one exclusive six-way target. Twenty-eight output heads meet the training-only support rule. A mask means a label contributes no loss for that record.

The **certain** objective uses listed likelihood-100 annotations as positives and masks intermediate/likelihood-0 entries. The **sensitivity** objective also treats intermediate listed annotations as positive; likelihood 0 remains masked. Unlisted codes are treated as annotation negatives under a disclosed convention, not a claim that disease absence was clinically adjudicated. Likelihood values are annotation states, not calibrated probabilities.

**Say:** “We preserved native overlapping diagnoses and tested two declared uncertainty conventions instead of silently mapping them into our target classes.”

**Likely question:** “Is source NORM equal to HC?” Answer: not automatically. Source clinical equivalence and source-target identity remain unresolved.

## Slide 19 — Completed supervised source training

**Purpose:** show that source learning worked on its own task, separately from transfer.

**Read the table:** certain-source **AP 0.317456**, **AUROC 0.908703**, **F1@0.5 0.290552**; intermediate-positive **0.317306 / 0.906510 / 0.287006**. The training-prevalence constant baseline has AP **0.037470**.

**Understand the metrics:** AP summarizes precision/recall ranking behavior; AUROC measures positive-versus-negative ranking across thresholds; F1@0.5 evaluates one fixed threshold. High AUROC and modest fixed-threshold F1 can coexist, particularly with rare labels and uncalibrated thresholds. These are patient-weighted record-level multilabel source metrics, not target six-class patient macro F1.

**Say:** “Both objectives learned source diagnostic structure above the constant baseline, but this does not establish improved target localization.”

**Likely question:** “Why keep both objectives?” Answer: they are prespecified primary/sensitivity objectives. The tiny source AP difference is not a reason to select a winner using target results.

## Slide 20 — Source-learning curves

**Purpose:** make checkpoint selection visible.

**Read the plot:** x-axis is source epoch, y-axis source-validation macro AP. Two curves correspond to the two annotation conventions; the dotted low baseline is the prevalence-constant predictor. Both trained for 15 epochs and selected **epoch 13** using the declared common validation AP.

An epoch is a training schedule unit, not an extra independent dataset. The best checkpoint can occur before the last epoch. Source validation was used to choose it, so it is model-selection evidence rather than untouched confirmation. These curves do not show target macro F1.

**Say:** “Source AP rose during training. We restored the checkpoints selected by source-validation AP, not whichever checkpoint happened to give the best target score.”

**Status update:** transfer is now completed; use the results listed at the start of this guide instead of saying it is still running.

## Slide 21 — Expanded recording-overlap screen

**Purpose:** investigate one important source of cross-dataset contamination.

**Understand:** screen all **7,246 source recordings** against **391 target recordings**. Exact encoded-file matches were zero. Candidate retrieval followed by continuous signal comparison checked **57,968 pairs**, not every possible pair. All **20/20 known-copy controls** passed; no high-correlation candidates or insufficient-active-lead pairs were found under the declared method.

Positive controls are examples known to contain copies: they test whether the method can find the kind of contamination it claims to screen. However, no detected copied recording is not proof that no individual appears in both datasets. Different visits, transformed/short copies and missed candidates remain possible.

**Say:** “The declared recording-copy screen passed and was independently replayed, while cross-dataset patient identity remains unresolved.”

**Likely question:** “Can we now call transfer completely independent?” Answer: no. Recording-copy screening and patient-identity linkage answer different questions.

## Slide 22 — Current matched supervised-transfer study

**Purpose:** isolate whether source initialization helps the target task.

**Understand the three arms:** scratch, certain-source encoder initialization, intermediate-positive-source encoder initialization. Transfer **encoder weights only**. Discard source diagnostic heads, normalizer and optimizer; use a fresh six-class head and exact target-training normalization. Architecture, initial heads and target sampling are matched where declared.

Three arms × three target seeds × ten outer folds × `(three inner folds × two rates + one outer refit)` = **630 fits**, including **90 outer models**. Outer labels do not choose the learning rate or stopping epoch.

**Say now:** “This comparison has since completed and all 630 fits were restored. The certain-source ensemble scored 0.573361 versus matched scratch 0.538736, but the paired difference interval includes zero.”

**Likely question:** “Is that higher than the historical neural ensemble?” Answer: no; historical 0.584570 remains a separate procedure, while scratch 0.538736 is the proper matched baseline for this transfer experiment.

## Slide 23 — Reproducibility and project status

**Purpose:** explain why reported results can be audited.

**Understand:** YAML records the settings; frozen folds fix patient assignments; hashes detect changes in code/configuration/runtime/artifacts; restoration reloads saved fitted parameters and reproduces predictions. Independent reports repeat selection and matching checks, rather than trusting the original training log.

Tests check defined implementation behaviors, including synthetic numerical oracles, partition isolation and tamper rejection. They are distinct from real-data model replay, and neither establishes clinical validity. Public artifacts contain aggregates and hashes; signals, fitted weights and patient predictions remain local and ignored.

**Say:** “We can trace a reported result to its protocol and saved fitted models, and independently reproduce the predictions and decisions.”

**Status update:** the deck's **289 tests** is historical. The later launch suite passed **309**. Transfer and the first distribution study are completed; the stronger convolution-distribution follow-up is active. Windows remains unvalidated; older multiview neural work remains paused.

## Slide 24 — Current conclusions and next decisions

**Purpose:** finish with the evidence and the specific unresolved question.

**Understand:** historical progress is real implementation work, with useful development reference scores near **0.5846**. Added tensor dimensions, chronology and feature compression did not consistently improve matched controls. Larger or more sophisticated is therefore a hypothesis to test, not a result.

**Say now:** “We have stronger reference procedures and a controlled evaluation pipeline, but the target score remains unachieved. Transfer and the first distribution screen have since completed without demonstrated reliable gains. We are now testing whether preserving the full convolution baseline and adding all-beat distribution information helps.”

The first distribution screen's **0.487114** used a different, compact physical/DCT feature bank; it does not show that every distribution method is ineffective. The active follow-up has no final validated score. Independent label-compatible patients would still be required for stronger generalization evidence.

**Likely question:** “What is the current contribution?” Answer: reproducible patient-level implementation and controlled representation/model comparisons, with explicit positive and negative results; proposed novelty or superiority still needs support.

## Slide 25 — Backup: patient bags and contrastive learning

**Use when asked:** “Did you try attention, learned tensors or contrastive objectives?”

A bag is a set of beats from a patient; it need not be consecutive. Mean pooling averages embeddings; attention learns weights. Tucker-style compression represents a multi-axis tensor using a smaller core and factors. A supervised contrastive objective encourages same-class embeddings to be similar and different-class embeddings to separate; cross-entropy trains the class prediction directly.

The bag experiments are single-seed comparisons: mean embedding **0.522**, attention **0.443**, learned-scale/fixed-compression **0.480**, learned Tucker+attention **0.507**, with waveform reference **0.586**. The contrastive studies instead report seed means and their own CE sampler controls: **0.516/0.479**, then diverse follow-up **0.515/0.531**.

**Say:** “We tested these families, but they did not establish robust superiority. An unordered bag is a different experiment from consecutive-beat chronology.”

**Avoid:** mixing single-seed and mean-seed entries into a common ranking, or calling an attention weight a validated explanation of infarct location.

## Slide 26 — Backup: waveform follow-ups

**Use when asked:** “Could preprocessing, longer context or regularization solve it?”

The tested raw waveform procedure scored about **0.521**, longer crops **0.538**, inner-selected regularization **0.550** versus baseline **0.550**, and territory auxiliary supervision **0.538** versus CE **0.550**. Each row has its own protocol and seed policy.

Regularization changes constraints on fitting; a longer crop adds more signal around R. Territory labels were derived from existing class names, so they are not additional independent anatomical ground truth. More tasks do not automatically mean more independent information.

**Say:** “We checked these plausible alternatives before moving on. Their tested recipes did not establish a consistent gain.”

The 35 persistent errors and patient-budget curve are diagnostics, not additional successful models. Do not reinterpret their existence as proof of incorrect labels.

## Slide 27 — Backup: correlation and covariance tensors

**Use when asked:** “How did you adapt the sepsis-paper idea?”

Combine **three XYZ channels × eight wavelet slices = 24 channels**. For each of four approximate R-relative regions, compute a **24×24** relation matrix; stacking the regions gives **24×24×4**. A shared basis learned from training patients compresses both matrix axes; inner selection chooses rank 4 or 8. Conceptually a region can be summarized as `Uᵀ C U` using the same training basis U.

Correlation divides out scale; covariance retains amplitude relative to declared training scales. This distinction matters because amplitude may contain localization information. Approximate regions are not expert-delineated clinical intervals.

Statistics control **0.469**; shared covariance+statistics **0.470**; shared correlation+statistics **0.413**. The extra structure did not establish a reliable gain.

**Say:** “We implemented a VCG-specific relation-tensor adaptation with training-only shared compression. The sepsis paper motivates the construction, but its endpoint and scores do not transfer to our task.”

## Slide 28 — Backup: geometry and reference limitations

**Use when asked:** “How physiological or clinically interpretable are those features?”

Adaptive boundary checks accepted **12,977/55,863 beats**, about **23.2%**. Acceptance was about **43% HC** but **15–22% MI classes**. Deleting the rejected beats could change class composition and create a misleadingly easier task. The declared fallback rules kept the entire cohort.

Healthy references are drawn only from the relevant training partition. A patient's distance from a healthy reference is not the difference from that person's own counterfactual healthy heart. Path signatures describe time ordering inside a beat; chronology describes ordering between separate beats.

**Say:** “The descriptors are interpretable signal constructions, but they do not validate clinical boundaries or causal disease explanations.”

**Likely question:** “Why not keep only good beats?” Answer: quality filtering would require a separately justified, label-independent protocol and coverage analysis; these results do not authorize score-driven removal.

## Slide 29 — Backup: earlier transfer and ECG extension

**Use when asked:** “Did you already test transfer or twelve-lead ECG?”

The earlier source pilot used masked reconstruction on **1,024 training patients**, rather than native-diagnosis supervised learning on the expanded cohort. Its mean target scores were pretrained **0.520** versus scratch **0.553**. Do not confuse it with the completed supervised transfer from slide 22.

Separate measured-ECG experiments: VCG **0.550**, ECG **0.509**, VCG twin **0.511**, additive ECG+VCG **0.536**, bilinear fusion **0.534**, all as reported seed means. The twin adds a VCG branch as a capacity control. Additive fusion combines branch information simply; bilinear fusion includes interactions.

**Say:** “Both source objectives and ECG extensions were tested under separate scopes. Our primary target remains measured Frank XYZ VCG.”

**Avoid:** implying the new supervised result is confirmation of the old masked objective, or describing ECG-assisted training as VCG-only training provenance.

## Slide 30 — Backup: feature refinement and budgets

**Use when asked:** “Why not increase tensor rank or pick the best feature budget?”

Higher DCT rank can reconstruct a waveform more closely while retaining nuisance variation as well as useful signal. Reconstruction error and diagnosis quality optimize different goals. The refinement results are mixed: expanded linear/tree **0.425/0.452**, legacy linear/tree **0.498/0.420**.

Convolution fixed32 ridge/PFN **0.495415/0.510066**; fixed64 **0.576198/0.532195**; nested selected **0.544036/0.530966**. These show that budget and classifier interact. They do not permit selecting a new fixed budget from held-out outcomes and reporting that as if it had been prespecified.

**Say:** “We evaluated fidelity, stable selection and downstream prediction separately. A better reconstruction or favorable fixed-budget control does not replace the declared training-only selector.”

## Slide 31 — Backup: all residual-context arms

**Use when asked:** “Were all arms reported, or only the favorable ensemble?”

The table keeps six arms: beat only, beat+auxiliary, ordered, ordered+auxiliary, shuffled, shuffled+auxiliary. Each has both mean member and ensemble F1. Ordered+auxiliary has mean **0.539** and ensemble about **0.585**; beat-only has mean **0.551** and ensemble about **0.545**.

This is a direct example of why averaging scores is different from scoring averaged probabilities. The same members produce two different summaries; no additional fitting is implied by the ensemble column. All arms retain all three seeds.

**Say:** “We show the full table so the favorable ensemble point does not hide the weaker average individual-model result.”

**Likely question:** “Did you select the best seed?” Answer: no. These ensembles average all three declared seeds equally; however, the ensemble analysis itself is explicitly post-hoc development.

## Slide 32 — Backup: all staged-learning arms

**Use when asked:** “Was the gain just extra training, or did context help?”

Morphology **0.545112** is this study's control. Continued morphology **0.549550** tests extra training without context; joint **0.559042** allows morphology adaptation; frozen **0.563539** protects the encoder; frozen shuffled **0.565056** tests order. All are three-seed ensemble scores.

For each exact inner/outer training partition, morphology teachers are fitted afresh. Using a teacher trained on the full outer training set inside its inner validation would leak those validation patients. Inner selection chose zero adaptation updates in **82/120 outer adaptations**, so many branches retained their initial state.

**Say:** “These controls separate additional training, freezing and sequence ordering. They did not establish an ordering-dependent gain.”

**Avoid:** replacing the matched morphology control with the historical 0.584570 ensemble from another study.

## Slide 33 — Backup: class-level reference results

**Use when asked:** “Where does the remaining error come from?”

Patient counts are **AMI17, ALMI16, ASMI27, IMI29, ILMI23, HC52**. Neural/convolution ensemble class F1 is **AMI .294/.345**, **ALMI .471/.483**, **ASMI .640/.633**, **IMI .596/.542**, **ILMI .638/.578**, **HC .868/.925**.

HC is considerably stronger than several MI subtypes. Macro F1 gives each class one sixth of the weight, so raising HC alone cannot solve subtype localization. Small MI classes also mean a few patient errors can change class F1 substantially.

**Say:** “The main difficulty remains distinguishing MI subtypes, rather than only identifying healthy controls.”

**Likely question:** “Can we route each class to whichever model did best?” Answer: not using these held-out outcomes. Any routing rule must be fitted and selected within training partitions and evaluated separately.

## Slide 34 — Backup: completed independent verification

**Use when asked:** “How do you know the saved results are valid implementations?”

The table lists coverage of restoration and replay, such as **1,200 chronology fits**, **720 residual-context fits**, **840 random-convolution classifiers** and the expanded signal screen. A fit count is a record of numerical work/verification, not the number of independent patients or a performance metric.

Restoration means loading the saved fitted model and reproducing query predictions. Replaying selections means choosing rates/ranks/budgets again from saved inner evidence and checking the same decision. Hashes detect changed artifacts but are not a substitute for numerical replay.

**Say:** “Each result has a frozen scope and an independent report that reconstructs features or restores models and repeats the relevant decisions.”

**Status update:** the transfer verification shown as pending is now complete: **630 fits, 90 outer models, 210 matched checks**. Implementation verification still cannot establish new-patient clinical validity or undo repeated development on PTB.

## Slide 35 — Backup: selected research references

**Use when asked:** “Which ideas are from existing research?”

Zhang et al. is the original MI-localization/Tucker-decomposition basis. Nesaragi et al. motivates relation tensors through a different sepsis task. Ilse et al. motivates attention-based multiple-instance learning; Khosla et al. supervised contrastive learning. MultiRocket/HYDRA motivate convolution-feature methods; TabPFN motivates tabular prior-based prediction. PTB and PTB-XL documentation establish dataset provenance.

**Say:** “We adapted established methods to our patient-level measured-VCG task and disclosed the implementation differences and matched results.”

References motivate choices; citing a paper does not transfer its score, benchmark ranking, clinical endpoint or theoretical guarantees to our pipeline. A method adaptation is not automatically novel or SOTA. Use the repository protocols and reports for exact implementation evidence.

## Rehearsal plan

For each main slide, practise explaining four things without reading it: the question, what was implemented, which comparison answers it, and the conclusion the evidence supports. If a professor asks for detail, use the relevant backup slide rather than reciting every configuration.

Prioritize slides **3, 6, 7, 10–12, 14, 18–19 and 22**: these contain the most important conceptual distinctions. Before presenting, verbally update slides20/22–24/34 or request a separate deck revision. Do not silently call the old status text current.

A concise ending: “Since October, we implemented waveform learning and several controlled tensor/context/source extensions. We have useful reference procedures around 0.58 ensemble macro F1 on the development cohort, but several sophisticated additions did not consistently improve matched controls. Our next question is whether preserving strong morphology features and all-beat variation helps. We will judge it using the same patient-separated protocol and independent replay.”

## Evidence used

- [Illustrated PDF](../output/pdf/mi_progress_since_october_2026_illustrated.pdf) and its [original detailed speaker notes](progress_presentation_20261117_illustrated_notes.md).
- [Current project handoff](CODEX_HANDOFF.md), [project class mapping](../src/mi_localization/labels.py), [ResNet implementation](../src/mi_localization/neural.py), and [tensor construction plan](tensor_feature_design_plan.md) clarify terminology.
- The original speaker notes link every slide to its completed reports/protocols. Numerical values in the slide sections refer to the saved slide unless marked as an update.
- Updates use the independently validated [supervised-transfer report](../results/neural_supervised_transfer_v1/complete/REPORT.md), [first distribution report](../results/distribution_pooling_v1/complete/REPORT.md) and [active follow-up protocol](convolution_distribution_plan.md). Process counters are deliberately not used as final results.
