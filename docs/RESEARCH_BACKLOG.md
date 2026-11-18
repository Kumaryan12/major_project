# VCG research backlog

This is the working queue for ideas that could improve six-class patient macro F1 on measured Frank XYZ or strengthen the evidence for the project. Change statuses as results arrive. Work through one idea at a time so we can tell what made a difference. Use the frozen patient folds for PTB development, and keep independent validation separate. [Project handoff](CODEX_HANDOFF.md)

| ID | Idea | Status | Next decision |
|---|---|---|---|
| 1 | Share classifier weights across related MI subtypes | Tested, gain uncertain | Keep the matched ridge reference. Record the result before considering further graph work. |
| 2 | Preserve waveform shape and timing in separate tensor channels | Audited, tensor deferred | No post-R window passed the support gate. Revisit with a stronger way to distinguish timing from shape. |
| 3 | Use documented region mentions from additional native patients | Tested, gain uncertain | True mentions scored above shuffled and below matched scratch. Retain the documented control results. |
| 4 | Obtain an independent measured XYZ cohort with reviewed labels | Awaiting cohort | Identify a partner, agree on label evidence and freeze patient assignments. |
| 5 | Learn from matched simulated infarct perturbations | Feasibility needed | Confirm that matched anatomy and disease-change runs exist before downloading or building a model. |
| 6 | Transfer angiography-linked artery supervision from ACS-ECG | Verified, gain uncertain | Keep the original measured-XYZ references. The artery objective trailed matched scratch. |
| 7 | Compare errors of the strongest saved convolution and waveform ensembles | Diagnostic complete | Keep both references. Defer fusion fitting after the equal blend scored below both. |
| 8 | Preserve direction in a within-beat lead-delay tensor | Tested, no demonstrated gain | Signed features trailed both the sign-scrambled control and the baseline. |
| 9 | Train the waveform ResNet on pooled patient probabilities | Verified, no demonstrated gain | Keep the historical waveform recipe. Both pooled objectives scored below the matched control. |
| 10 | Average the four linear convolution readouts | Verified, gain uncertain | Keep the 0.603374 selector as the historical reference. |
| 11 | Pool convolution responses in fixed R-relative windows | Proposed | Compare aligned and position-shuffled windows with a matched global feature budget. |
| 12 | Separate MI detection and localization in a probabilistic readout | Proposed | Test a binary plus conditional subtype readout on cached patient features. |
| 13 | Learn ECG changes from controlled balloon occlusion | Feasibility needed | Check annotated baseline and inflation pairs with comparable acquisition before fitting. |

## 1. Related MI subtypes share classifier weights

**Hypothesis.** Linking classifiers that share a region in their label names can stabilize estimates for small classes. The classes retain individual decision scores. The graph regularizes the final classifier, whereas the earlier territory study put an auxiliary loss on the neural encoder.

**Controls and result to record.** Run ordinary class-balanced ridge, ridge with the declared region graph, and ridge with the same graph relabeled among MI classes. Keep the full baseline features and the same nested folds. Advance the idea only if it improves patient macro F1 against ridge and the relabeled graph supports the proposed mechanism. Keep all seed and class results. [Protocol](subtype_graph_plan.md) · [Earlier territory loss](territory_vcg_plan.md)

**Result.** The subtype graph reached **0.591424** ensemble macro F1 versus **0.584266** for ordinary ridge and **0.582026** for the relabeled graph. Both differences involving the subtype graph have intervals that include zero. Anterior MI stayed at 0.344828 while anterolateral MI fell from 0.482759 to 0.466667. Verification replayed 120 feature contexts and all declared candidates. The original ridge remains the matched reference. [Findings](subtype_graph_findings.md) · [Validated report](../results/subtype_graph_v1/complete/REPORT.md)

## 2. Shape and timing tensor

**Hypothesis.** A shared time adjustment for X, Y and Z can make recurring local shapes easier to compare. Save the original waveform, aligned waveform and time adjustment so timing remains available. The previous audit found a median whole-beat shift of 0 ms. Check variation in fixed post-R windows first, with bounded adjustments and no new clinical boundary claims.

**Controls and result to record.** Compare original features, aligned features and aligned features plus timing. Report alignment stability on disjoint beats and any change in patient macro F1. Evaluate the smallest representation that answers the hypothesis before expanding. [Alignment audit](record_tensor_quality_plan.md) · [Joint waveform alignment research](https://link.springer.com/article/10.1007/s11222-025-10678-8)

**Audit result.** None of the three post-R windows passed. The strongest covered 20 of 164 patients at the declared support threshold, versus the required 20% of patients. The estimator recovered injected shifts across 81.8–96.7% of post-R controls. Synthetic shape changes sometimes produced apparent shifts despite zero injected delay. Defer full extraction and keep the measurements as feasibility evidence. [Findings](local_timing_audit_findings.md) · [Validated audit](../results/local_timing_audit_v1/REPORT.md)

## 3. Partial region supervision from extra native cases

**Hypothesis.** Additional measured-XYZ patients may help more when they carry location information. Fourteen screened source patient IDs have consistent parsed MI-location mentions outside the six target classes. Review the original evidence before treating these as region targets. Mask ambiguous or unmentioned regions.

**Controls and result to record.** Compare target CE, target descriptor supervision and two source supervision arms with true or shuffled positive mentions. Keep the six-class target endpoint fixed. The audit reproduced all 34 source records for the 14 candidates. Thirteen patients have at least one supported region mention. [Native partial-label protocol](native_partial_labels_plan.md) · [Earlier native pilot](vcg_pretraining_pilots_findings.md)

**Result.** True mentions reached 0.575887 ensemble macro F1, versus 0.568947 for shuffled, 0.592327 for target descriptors and 0.601801 for scratch. The primary true-minus-shuffled interval [−0.042964, +0.058767] includes zero. All 120 models passed independent replay. The source tag distribution is sparse and imbalanced. [Findings](native_partial_labels_findings.md)

## 4. Independent measured XYZ cohort

**Hypothesis.** More independently labeled MI patients give us a chance to improve class coverage and check generalization. Prioritize anterior and anterolateral MI and sufficient coverage of all six classes.

**Controls and result to record.** Confirm clinical label evidence and signal format, reserve the patient test set before choosing the final model, and record the frozen arm and hashes. Report patient-level metrics with uncertainty. The synthetic integration check exercised intake and inference. Real recordings and label review are pending. [Partner protocol and tooling](measured_vcg_partner_cohort.md)

The CSE diagnostic study is a measured orthogonal archive lead, with access and label coverage still to verify. [External data shortlist](EXTERNAL_DATA_SHORTLIST.md)

## 5. Matched synthetic infarct perturbations

**Hypothesis.** Comparing signals from the same simulated anatomy with controlled disease changes could teach features that are less sensitive to anatomy. The first step is to verify that the required matched cases exist and that native simulated labels and acquisition scope are understood.

**Controls and result to record.** Split by anatomy. Compare matched disease-change supervision with ordinary synthetic source training and target-only scratch. Use native source labels and report the target transfer result separately. [Earlier simulation design notes](research_frontiers_20261116.md)

## 6. Angiography-linked pretraining from ACS-ECG

**Hypothesis.** Pretraining on native coronary artery labels may transfer more localization information to measured XYZ than reconstruction or a broad ECG diagnosis objective. The publisher's metadata archive passed its checksum and has 913 patients with at least one OMI recording in a single acute artery group. The released test CSV omits diagnostic labels. [Metadata audit](EXTERNAL_DATA_SHORTLIST.md#acs-ecg-metadata-check)

**Controls and result to record.** Check waveform alignment and prespecify handling of recordings with missing, multivessel or inconsistent artery annotations. Then compare equal source exposure with and without the artery objective and a matched target-only scratch model. Preserve the measured XYZ endpoint and use source artery groups for the auxiliary task. [External data shortlist](EXTERNAL_DATA_SHORTLIST.md)

**Implementation.** Verified two small raw WFDB samples against their metadata references. The bounded pipeline prepares and independently replays the source bank and copy screen, restores both source models, then trains and restores all 90 measured-XYZ target fits. [Protocol and monitoring](acs_artery_transfer_plan.md)

**Result.** OMI-only .570389, OMI-plus-artery .580730, scratch .601801 patient ensemble macro F1. Artery minus OMI +.010341 with descriptive paired 95% interval [−.066726,+.085396]. Both fixed source models, all 90 target fits and all 30 matched contexts replayed. Source validation artery macro F1 .615503 is a separate source endpoint. [Verified report](../results/neural_acs_artery_transfer_v1/complete/REPORT.md)

## 7. Convolution and waveform prediction overlap

**Result.** Reconstructed the verified convolution and waveform ensembles at 0.603374 and 0.601801. The convolution model alone was correct on 19 patients, the waveform model alone on 16, and both missed 36. The fixed equal blend scored 0.582209, a difference of −0.021165 [−0.087770, +0.043554] versus convolution. The waveform ensemble's higher anterior MI F1 was weakened by blending. The diagnostic supports keeping the two component references and gives no demonstrated reason to prioritize a tuned fusion. [Protocol](model_error_comparison_plan.md) · [Verified comparison](../results/model_error_comparison_v1/REPORT.md)

## 8. Within-beat lead direction

**Hypothesis.** Local signed interactions between measured XYZ leads may offer information beyond the convolution representation. Compute lag covariance in four R-relative windows at four time delays. The antisymmetric entries flip sign when the time direction is reversed.

**Controls and result to record.** Keep the full convolution base and the symmetric entries fixed across the three tensor arms. Compare signed, absolute and sign-scrambled entries at the same feature width. Use patient-nested tuning and report whether the signed arm improves both over magnitude and over the original convolution reference. [Protocol](lag_direction_tensor_plan.md)

**Result.** Signed 0.531364, magnitude 0.517919, sign scrambled 0.546633 and baseline 0.584266. The primary difference was +0.013444 [−0.027522, +0.054415]. All three signed members were below their matched scrambled and baseline members. Replayed all 120 contexts and all declared models. [Findings](lag_direction_tensor_findings.md)

## 9. Patient probability loss for the waveform ResNet

**Hypothesis.** The 0.601801 waveform ensemble trains with cross-entropy on each beat and averages beat probabilities for the patient decision. Applying the loss to the average probability of a sampled patient bag could better align training with inference. Earlier patient-tensor models pooled embeddings and changed the model architecture. This comparison keeps the waveform network and inference rule fixed. [Waveform results](vcg_pretraining_pilots_findings.md) · [Earlier patient pooling](../results/neural_patient_tensor_v1/complete/REPORT.md) · [Multiple-instance learning research](https://proceedings.mlr.press/v80/ilse18a.html)

**Proposed comparison.** Use class- and patient-balanced bags with one prespecified beat count and identical beat draws across arms. Compare the mean of per-beat cross-entropies with cross-entropy on the mean beat probability. Include a fixed half-and-half loss as a secondary arm since the pooled objective can emphasize beats already supporting the correct class. Keep the optimizer, beat exposure and all-beat inference identical. Use all frozen folds and three declared seeds. The primary comparison is pooled versus matched beat loss, with the historical 0.601801 waveform and 0.603374 convolution ensemble scores shown as development references. Report paired patient uncertainty and per-class F1.

**Result.** All 90 models and 30 matched fold and seed groups independently verified. Ensemble macro F1 was 0.510134 for beat CE, 0.469596 for patient CE and 0.460082 for hybrid CE. Patient CE minus matched beat CE was −0.040538 with descriptive paired 95% interval [−0.103712, +0.022126]. The eight-beat control also scored below the historical 0.601801 waveform using individual beat draws. The matched comparison isolates the loss under grouped sampling. [Protocol](patient_probability_loss_plan.md) · [Verified report](../results/neural_patient_probability_loss_v1/complete/REPORT.md)

## 10. Fixed average of the linear convolution readouts

**Hypothesis.** Averaging the linear predictions for morphology, mean, mean/spread and quantiles could reduce sensitivity to which representation wins in inner validation. The four representations share the same base features and may add little diversity.

**Proposed comparison.** Replay all four fixed linear readouts for all frozen folds and three seeds. Take the equal probability average and compare it with the verified inner-selected linear procedure at 0.603374. Keep the existing inner regularization choices and probability conversion. Report the paired patient interval and per-class results. This uses the already fitted models and a fixed combination. [Kernel findings](distribution_kernel_findings.md)

**Result.** Verified 120 saved outer probability files and reproduced every source arm, seed and selector metric. Equal average ensemble 0.586053 versus selector 0.603374, difference −0.017321 with descriptive paired 95% interval [−0.042859, +0.005133]. Zero new fits. [Verified comparison](../results/convolution_readout_average_v1/REPORT.md)

## 11. R-relative window pooling for convolution features

**Hypothesis.** The convolution backbone already preserves some timing through the mean index of positive responses. Pooling response statistics within fixed regions before, near and after R could preserve more localization information than whole-beat pooling. Aggregate features from the existing measured XYZ recording templates before considering more expensive all-beat extraction. [Existing backbone](rocket_vcg_plan.md) · [Temporal ROCKET research](https://doi.org/10.1007/s10618-025-01162-y)

**Proposed comparison.** Keep the verified full convolution base. Append a fixed feature budget in every extended arm. Compare R-aligned short-kernel summaries with the same summaries under independent position shuffles per record, and with the same width of extra whole-beat summaries. Freeze window boundaries and filter-count rules before fitting. Fit scaling and regularization inside patient partitions. Report all folds, three seeds, the paired contrast with the equal-width global control and comparison with the historical 0.603374 selector. AMI and ALMI F1 are prespecified descriptive checks.

## 12. Probabilistic detection and localization readout

**Hypothesis.** A readout specialized to the five MI subtypes can focus on subtype distinctions after a binary readout estimates MI versus HC. For each subtype, multiply its conditional probability by the estimated probability of MI, while retaining the binary HC probability. [Separate detection and localization in ECG research](https://pubmed.ncbi.nlm.nih.gov/38599223/)

**Proposed comparison.** Train both stages inside the same patient partition using the cached convolution features. Compare with a flat six-class multinomial logistic model using the same data, feature widths and training-only tuning budget, and show the existing ridge reference. Outer evaluation includes all patients and all six classes. Report within-MI errors and healthy-control errors to show whether the new readout helps localization and how it changes detection.

## 13. Paired ECG changes during coronary balloon occlusion

**Hypothesis.** Within-patient baseline-to-occlusion ECG changes may reveal how waveform morphology responds to controlled ischemia. STAFF III includes 104 patients with timing annotations. This would be an ECG-assisted source experiment with measured XYZ as the sole target input at inference. [PhysioNet dataset](https://physionet.org/content/staffiii/1.0.0/)

**Proposed comparison.** Audit exact annotated intervals, baseline acquisition setting, contrast events and usable vessel support. Split by patient before constructing pairs. Compare matched change supervision with a single-record source objective using the same patients and comparable exposure, with a target scratch control. The source task describes induced ischemia, and target performance must be measured separately.

## Evidence already in hand

- Beat distributions, frozen score corrections and a nonlinear readout did not establish a gain on their primary comparisons. Use their reports to avoid repeating the same hypotheses. [Frozen score correction](distribution_score_correction_findings.md) · [Kernel readouts](distribution_kernel_findings.md)
- Masked native-data and paired-ECG pretraining were tested under an eight-epoch pilot schedule. Neither ensemble beat the matched 0.601801 scratch ensemble. [Pilot findings](vcg_pretraining_pilots_findings.md)
- Before starting a new idea, write its primary comparison, data split, controls and completion criteria into its protocol. After completing it, record the result and link the report here.
