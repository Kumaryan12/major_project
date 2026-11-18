# Saved-prediction comparison of convolution and waveform ensembles

Compare two strong completed measured-XYZ models on their exact shared held-out patient assignments. Use the inner-selected linear convolution procedure and the fixed-eight-epoch scratch waveform control from the native pretraining study. This is a retrospective development diagnostic motivated by the original fusion study's different component models. The original fold-specific model selection and all three seeds are retained. [Convolution report](../results/distribution_kernel_v1/complete/REPORT.md) · [Waveform report](../results/neural_vcg_native_pretraining_v1/complete/REPORT.md)

Load the probability files linked to the independently verified outer fit audits. Validate the original selection and patient/row membership checksums. Rebuild each pooled patient prediction using the original aggregation. The two sets must agree on the 164 patient IDs, true class order, fold IDs and published component metrics. Record the source hashes and publish aggregate counts only.

Count both correct, convolution only, waveform only and both wrong, overall and by true class. Count identical wrong predictions within the shared misses. Report component macro F1 and descriptive paired patient intervals. A fixed 50/50 probability average provides one descriptive check of whether the two sets of scores work together. The weights are set before reading paired predictions. There is no weight search, calibration, class routing, model fitting or new waveform extraction in this diagnostic. Since the components were chosen after prior PTB results were reviewed, the fixed average is also retrospective development evidence. No one chooses the better prediction per patient using the true label.

An eventual tuned fusion would require exact inner-split probability generation for both components. The fixed-eight-epoch waveform procedure has outer predictions only. Any fitted combination must be trained and selected within outer-training patients, with an independent cohort as the strongest follow-up check.

```bash
.venv/bin/python -m mi_localization.model_error_comparison
.venv/bin/python -m mi_localization.model_error_comparison --verify
```

## Completed finding

The convolution and waveform ensemble scores reproduced at 0.603374 and 0.601801. Both correctly classified 93 patients. Their correct predictions differed on 35 patients, split 19 convolution-only and 16 waveform-only. Both missed 36, and they agreed on the same wrong prediction for 21 of those. Anterior MI F1 was 0.529412 for waveform, 0.344828 for convolution and 0.375000 for the blend. Anterolateral MI remained difficult for both.

The fixed equal blend scored 0.582209. Its differences versus convolution and waveform were −0.021165 [−0.087770, +0.043554] and −0.019592 [−0.074656, +0.034880], respectively. These are descriptive paired intervals. Distinct errors can be useful, but neither interval demonstrates a gain, and the equal average did not capitalize on the waveform model's stronger anterior MI result. Keep the separate references and defer a tuned blend.

A fresh process reproduced all metrics and checked 180 linked source artifacts for the 60 outer prediction blocks and the original selections. All eight relevant tests passed. [Verified report](../results/model_error_comparison_v1/REPORT.md) · [Aggregate metrics](../results/model_error_comparison_v1/summary.json)
