# Native region mention results

Using the true source mentions gave **0.575887** patient-level ensemble macro F1. Shuffled source mentions gave **0.568947**, an uncertain difference of **+0.006940 [−0.042964, +0.058767]**. Matched scratch reached **0.601801**. True mentions trailed scratch by **−0.025914 [−0.085288, +0.035006]**. This comparison gives no demonstrated gain from the tested partial-supervision recipe. [Protocol](native_partial_labels_plan.md) · [Validated report](../results/neural_native_partial_labels_v1/complete/REPORT.md)

| Arm | Mean member macro F1 | Ensemble macro F1 |
|---|---:|---:|
| Scratch | 0.541465 | 0.601801 |
| Target descriptors only | 0.544872 | 0.592327 |
| Shuffled source mentions | 0.550108 | 0.568947 |
| True source mentions | 0.545536 | 0.575887 |

Anterior MI class F1 rose from 0.457143 with shuffled mentions to 0.514286 with true mentions. The matched scratch class score was 0.529412. Anterolateral MI reached 0.466667 with true mentions, compared with 0.500000 for target descriptors alone. These class differences describe the held-out development predictions. Mean member and ensemble metrics answer different questions.

The source audit reproduced all 34 records and 4,644 beats from 14 candidate patients. Thirteen patients and 4,215 beats had a shared observed tag. The remaining patient's header named only posterior MI. Among the 13 patients, the observed region counts were anterior 1, inferior 9, lateral 12 and septal 1. Each shuffle reassigned every source patient's tag vector to a different patient, changed 10 of 13 tag sets and preserved the same tag counts per update. Unmentioned source regions contribute zero loss. The header checks confirm consistent documented codes and matched waveforms. Independent anatomical adjudication was unavailable.

All 120 models across ten folds and three seeds passed independent replay. Read-only verification matched 120 fit audit hashes and 240 model/prediction file hashes and confirmed the matching training draws in all 30 groups. The 30 scratch prediction blocks were identical to the earlier pilot's scratch runs. The inner-only four-arm smoke and **338 tests** passed. Keep scratch as the matched waveform reference. [Aggregate results](../results/neural_native_partial_labels_v1/complete/summary.json) · [Backlog](RESEARCH_BACKLOG.md)
