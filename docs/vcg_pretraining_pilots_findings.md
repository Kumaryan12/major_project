# More measured VCG patients and paired ECG training results

Both studies use the same fixed eight-epoch target schedule, all 164 target patients, all ten frozen patient folds and three seeds. These figures measure six-class patient macro F1 on PTB development patients. The comparator for each procedure is its matched control under this schedule. [Protocol](vcg_pretraining_pilots_plan.md)

## Native measured XYZ

The audit admitted **124 additional patients, 156 recordings and 17,573 beats**. One of 157 candidate recordings failed signal variation. The copy screen ran 1,248 continuous comparisons after exact hashes against all 391 target recordings. The 12 known-copy controls passed. No patient IDs were quarantined. The continuous screen covered the shortlisted pairs. [Audit](../results/ptb_native_expansion_v1/REPORT.md)

| Arm | Mean member macro F1 | Ensemble macro F1 |
|---|---:|---:|
| Scratch | 0.541465 | 0.601801 |
| Masked pretraining on target-training patients | 0.472359 | 0.454298 |
| Masked pretraining with additional native patients | 0.495450 | 0.518364 |

Replacing half the examples in each reconstruction batch with additional native patients improved the ensemble point estimate by **+0.064066** over target-only pretraining. The paired patient interval **[−0.003292, +0.133664]** crosses zero. Expanded pretraining trailed scratch by **−0.083436 [−0.160972, −0.007201]** under the declared schedule. Mean member scores show the same ordering. The comparison tests a short masked reconstruction and fine-tuning recipe. The added patients contributed unlabeled signals. [Validated native report](../results/neural_vcg_native_pretraining_v1/complete/REPORT.md)

## Paired ECG-assisted training

| Arm | Mean member macro F1 | Ensemble macro F1 |
|---|---:|---:|
| Scratch | 0.541465 | 0.601801 |
| Shuffled ECG targets | 0.481652 | 0.529345 |
| Correctly paired ECG targets | 0.544613 | 0.564293 |

Correct pairing improved the ensemble point estimate by **+0.034948** over shuffled pairing. The paired interval **[−0.051853, +0.121214]** crosses zero. The paired ensemble trailed scratch by **−0.037508 [−0.125846, +0.049101]**. The paired mean member score was slightly higher, **0.544613 versus 0.541465**, and seed SD was lower. Its anterior MI class F1 fell from **0.529412 to 0.285714**. The original six-class ensemble endpoint shows no demonstrated gain. Both ECG-assisted arms ran inference only on measured XYZ. [Validated bridge report](../results/neural_vcg_ecg_bridge_v1/complete/REPORT.md)

Independent replay restored **90 models per study** and matched all 30 fold/seed groups per study. A separate read-only check matched all 180 fit audits and 360 checkpoint/prediction file hashes. Scratch prediction files matched byte for byte between studies. Both runners finished. Each real inner-only smoke restored all three arms, and the full macOS test suite passed **328 tests**.

## Independent cohort

The intake and evaluation tooling are ready. A synthetic WFDB integration run verified all six label paths and exercised inference with all 30 model checkpoints from one completed arm. Those generated signals verified software behavior. A real external score requires access to an eligible cohort and independent label review. [Partner cohort instructions](measured_vcg_partner_cohort.md)
