# Next experiments using existing measured-XYZ representations

Reviewed 19 November 2026. This research decision follows the completed duration, single-record convolution and recording-quality studies. No classifier fit or new training sweep was started in this review.

**Subsequent outcome, 19 November.** Experiment 1 was authorized and completed independently: original .521335, mean logits .517351, ridge64 .535691, ridge256 .501020 on the 145-patient primary endpoint. The proposed pre-compression gain was not demonstrated; the secondary ridge64 gain remains uncertain. All 420 heads and 120 feature contexts replayed. At that completion review, experiments 2 and 3 remained proposals. [Findings](frozen_patient_readout_findings.md).

**Later authorization, 19 November.** The user approved experiment 2. Its fixed [distance-readout protocol](frozen_distance_readout_plan.md) and [execution instructions](frozen_distance_readout_run.md) supersede only that proposed study's earlier boundary. Implementation `b1916616`, all 376 tests and the real-data inner-only fitting/restoration smoke passed; full fit/independent replay launched at 11:47:30 IST. Experiment 3, the decision offsets, and any all-record follow-up remain unstarted.

## Why revisit the readout

The strongest all-record development procedures remain selected convolution .603374 and waveform ensemble .601801 over 164 patients. The current single-record supervised reference is .521335 over 145 date-eligible patients. More optimization reduced training loss without reliably improving that endpoint. Replacing this single-record waveform model with fixed convolution features scored .478122. These observations support testing how the waveform representation is used for patient classification. [Duration result](../results/neural_earliest_training_duration_v1/complete/REPORT.md) · [Single-record convolution](../results/earliest_record_convolution_v1/complete/REPORT.md)

The current ResNet maps each beat to 64 channels over 41 downsampled time positions. Four-bin average pooling gives 256 values. A learned dense layer and GELU compress those values to 64, followed by dropout and a six-way linear classifier. Evaluation averages the six probabilities across beats. The previous learned-distribution study appended summaries of learned residual embeddings to an intact convolution baseline. It used a different encoder source and inherited inner-validation stopping. The proposed readout comparison uses fixed-eight-epoch waveform checkpoints from exact inner/outer partitions and a standalone patient-level classifier. [Architecture](../src/mi_localization/neural.py#L186) · [Previous learned summary procedure](learned_distribution_plan.md)

A read-only feasibility check restored the epoch-eight encoder for seed 42, outer fold 1, inner fold 1. Its file hash and 98 fitting patients matched the saved receipt. On two fitting beats per patient, both the 256-value time-bin output and the 64-value final embedding were finite. No query patients or labels entered that check. With one float32 row per patient, the 164-patient tables occupy approximately 164 KiB and 41 KiB respectively before metadata. Full encoding can stream one recording and one partition at a time using the existing waveform cache.

## 1. Test the 256-to-64 compression and patient-level readout

**Question.** Can the already learned temporal features support a more stable subtype classifier after patient aggregation?

Freeze the epoch-eight encoder for every exact fitting partition. Average features within the selected recording, using all its retained beats. The source training includes the established fallback policy and the main metric covers the same 145 dated patients. Fit a separate balanced ridge head to 64 values and another to the 256 values preceding compression. Select alpha from [1,100] inside each outer-training partition, using pooled dated inner predictions and the existing inner split rule. Fit normalization using training patient rows only. Average the three seed probability vectors equally for outer inference.

Retain the original average-beat-probability model as a control. Also evaluate the original linear classifier on the mean 64-value embedding at fixed temperature. Its logits equal the mean original per-beat logits in evaluation mode. Its softmax can differ from the original mean of per-beat softmax probabilities. This gives a parameter-free aggregation control before interpreting a new learned head.

The two declared contrasts are 256-ridge minus 64-ridge for the compression question, and 256-ridge minus original model for useful improvement. Report both controls and all class scores, member/ensemble scores and paired patient intervals. If neither contrast supports benefit, retain the original source model. More columns can increase estimation variance, especially with 16–29 MI patients per class.

**Background.** Separating representation learning from classification has established precedent. Kang et al. studied visual long-tailed recognition, with different data and sampling. Their results motivate testing the decision stage and establish no expected MI gain. [Decoupling Representation and Classifier for Long-Tailed Recognition, ICLR 2020](https://openreview.net/pdf?id=r1gRTCVFvB)

**Cost and scope.** Existing duration-study checkpoints supply all 90 inner and 30 outer encoders at eight epochs, without validation-based encoder stopping. New fits are small CPU readouts. An independent replay should restore each encoder, reconstruct features, fit all scalers from its exact training rows and restore every readout. No global patient embedding table may mix encoders from different fitting partitions. Expected dominant cost is streaming frozen encoding.

## 2. Test a shrinkage distance classifier on the same 64 features

**Question.** Does regularized within-class geometry distinguish subtypes better than a linear score head?

Use the same exact-partition patient embedding rows as experiment 1. Compare balanced ridge, equal-prior Euclidean class-centroid distance and equal-prior shared-covariance Mahalanobis distance. For the covariance, pool within-class patient residuals and shrink toward a scaled identity, with fixed candidate shrinkages [0.5,0.9,1.0] selected by inner patient macro F1. Include a positive numerical floor. Each patient contributes once and each class has equal mass in the pooled covariance. A 64-dimensional class covariance estimated from roughly ten patients would be rank-deficient, so begin with a shared strongly regularized covariance. This is a separate head experiment after a completed decision on experiment 1.

The simplest intuition is that some learned feature directions vary greatly among patients of the same subtype. A regularized distance model gives those directions less influence. It may also suppress useful region information. Compare to Euclidean distance and ridge on identical features to test that mechanism. Do not normalize physical XYZ or rotate it as part of the adaptation.

**Background.** Bateni et al. studied regularized Mahalanobis classifiers on image few-shot tasks. Their method also used task-adapted encoders. A frozen ECG embedding readout is a narrower adaptation. [Improved Few-Shot Visual Classification, CVPR 2020](https://arxiv.org/abs/1912.03432) · [Author code and method description](https://github.com/plai-group/simple-cnaps)

**Cost.** CPU operations on at most 164 by 64 patient features and 64 by 64 covariance matrices. The existing frozen encoder pass supplies the features. There is no published MI-specific improvement estimate.

## 3. Test a bounded macro-F1 decision rule

**Question.** Does optimizing the classifier's decision boundary for patient macro F1 improve a fixed source representation?

The current CE and squared-error objectives differ from the project macro F1 endpoint. A conservative first adaptation would add six class offsets to the patient logits, with the sum constrained to zero and shrinkage toward zero. Use only cross-fitted outer-training patient logits. Fit the offsets inside a meta-training split and validate their strength in a separate meta-validation split, or use a predeclared fixed penalty in a bounded pilot. Keep zero offsets as the primary control and compare class F1 as well as overall F1. Calibrating a positive scalar temperature on the final logits alone preserves their argmax.

Choosing offsets using pooled outer predictions would consume the test outcomes. Cross-fitting must include upstream learned choices. A fixed-eight-epoch encoder with fresh exact-partition normalization simplifies that requirement. This method can shift false positives among nearby classes and can overfit even with five effective offset parameters. It supplies no additional signal or independent examples.

**Background.** Pan et al. developed BEAM-F for direct multiclass macro-F optimization. Their confusion-matrix method differs from the restricted offset adaptation above. Avoid transferring guarantees from that full algorithm to this pilot. [Optimizing the Multiclass F-measure via Biconcave Programming, ICDM 2016](https://research.ibm.com/publications/optimizing-the-multiclass-f-measure-via-biconcave-programming)

## Decision and wider endpoint

Prioritize experiment 1 because it tests a concrete compression point using saved encoders, with inexpensive matched readouts. Register a fixed protocol and completion inventory before implementation. Research-stage advancement would require the proposed model to improve over the original supervised reference and show MI-subtype benefit, with uncertainty reported. Keep the experiments sequential.

A positive result on earliest recordings would establish a candidate for the 145-patient archival task. To address the original 164-patient all-record target, run the chosen recipe in a separate matched all-record study, starting from the .601801 fixed-eight waveform schedule. That source retained 30 outer checkpoints and requires new exact-partition inner encoders for head selection. The .603374 convolution result remains a separate all-record reference.

None of these methods offers an evidence-based forecast of .80–.90. The partner cohort and blinded clinical review remain the means to investigate label evidence and independent-patient coverage. [Partner intake](measured_vcg_partner_cohort.md)
