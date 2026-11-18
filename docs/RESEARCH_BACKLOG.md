# VCG research backlog

This is the working queue for ideas that could improve six-class patient macro F1 on measured Frank XYZ or strengthen the evidence for the project. Change statuses as results arrive. Work through one idea at a time so we can tell what made a difference. Use the frozen patient folds for PTB development, and keep independent validation separate. [Project handoff](CODEX_HANDOFF.md)

| ID | Idea | Status | Next decision |
|---|---|---|---|
| 1 | Share classifier weights across related MI subtypes | Tested, gain uncertain | Keep the matched ridge reference. Record the result before considering further graph work. |
| 2 | Preserve waveform shape and timing in separate tensor channels | Audited, tensor deferred | No post-R window passed the support gate. Revisit with a stronger way to distinguish timing from shape. |
| 3 | Use documented region mentions from additional native patients | Tested, gain uncertain | True mentions scored above shuffled and below matched scratch. Retain the documented control results. |
| 4 | Obtain an independent measured XYZ cohort with reviewed labels | Awaiting cohort | Identify a partner, agree on label evidence and freeze patient assignments. |
| 5 | Learn from matched simulated infarct perturbations | Feasibility needed | Confirm that matched anatomy and disease-change runs exist before downloading or building a model. |
| 6 | Transfer angiography-linked artery supervision from ACS-ECG | Source identified | Obtain and audit the small public metadata archive, then verify a waveform sample. |
| 7 | Compare errors of the strongest saved convolution and waveform ensembles | Diagnostic complete | Keep both references. Defer fusion fitting after the equal blend scored below both. |

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

**Hypothesis.** Pretraining on native coronary artery labels may transfer more localization information to measured XYZ than reconstruction or a broad ECG diagnosis objective. ACS-ECG's publication describes real recordings from 18,909 patients with angiography-linked labels. The release metadata is public. The metadata archive download could not be verified from this environment.

**Controls and result to record.** Verify the label archive, patient grouping, acute and chronic annotations, actual vessel support and waveform alignment. Then compare equal source exposure with and without the artery objective and a matched target-only scratch model. Preserve the measured XYZ endpoint, and use the original artery categories for the source. [External data shortlist](EXTERNAL_DATA_SHORTLIST.md)

## 7. Convolution and waveform prediction overlap

**Result.** Reconstructed the verified convolution and waveform ensembles at 0.603374 and 0.601801. The convolution model alone was correct on 19 patients, the waveform model alone on 16, and both missed 36. The fixed equal blend scored 0.582209, a difference of −0.021165 [−0.087770, +0.043554] versus convolution. The waveform ensemble's higher anterior MI F1 was weakened by blending. The diagnostic supports keeping the two component references and gives no demonstrated reason to prioritize a tuned fusion. [Protocol](model_error_comparison_plan.md) · [Verified comparison](../results/model_error_comparison_v1/REPORT.md)

## Evidence already in hand

- Beat distributions, frozen score corrections and a nonlinear readout did not establish a gain on their primary comparisons. Use their reports to avoid repeating the same hypotheses. [Frozen score correction](distribution_score_correction_findings.md) · [Kernel readouts](distribution_kernel_findings.md)
- Masked native-data and paired-ECG pretraining were tested under an eight-epoch pilot schedule. Neither ensemble beat the matched 0.601801 scratch ensemble. [Pilot findings](vcg_pretraining_pilots_findings.md)
- Before starting a new idea, write its primary comparison, data split, controls and completion criteria into its protocol. After completing it, record the result and link the report here.
