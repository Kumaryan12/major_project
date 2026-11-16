# Measured-VCG beat-distribution pooling

Prespecified measured-VCG distribution-pooling development on repeatedly inspected PTB patients; no independent confirmation, physiological distribution model or established SOTA.

All 164 patients, 391 recordings and 55,863 original beats retained. Reconstructed every beat and record-median feature, all 120 training transforms, and restored all 840 classifiers including 120 outer models. 120 matched four-arm feature-context checks passed.

| Procedure | Mean member macro F1 ± projection-seed SD | Uniform three-seed ensemble macro F1 |
|---|---:|---:|
| median_waveform | 0.413881 ± 0.015183 | 0.434441 |
| mean_embedding | 0.454664 ± 0.011233 | 0.463013 |
| mean_spread | 0.472406 ± 0.031658 | 0.475709 |
| quantiles | 0.462552 ± 0.035313 | 0.487114 |

## Prespecified comparisons

Primary: quantiles minus mean_spread. All contrasts retained. Intervals are descriptive and conditional.

| Ensemble comparison | Difference | Paired patient 95% interval |
|---|---:|---:|
| quantiles_minus_mean_spread | +0.011405 | [-0.052573, +0.076160] |
| quantiles_minus_mean_embedding | +0.024100 | [-0.022854, +0.074146] |
| quantiles_minus_median_waveform | +0.052672 | [-0.027981, +0.135237] |
| mean_spread_minus_mean_embedding | +0.012696 | [-0.051823, +0.080506] |
| mean_embedding_minus_median_waveform | +0.028572 | [-0.057232, +0.112728] |

## Ensemble class F1

| Class | median_waveform | mean_embedding | mean_spread | quantiles |
|---|---:|---:|---:|---:|
| AMI | 0.228571 | 0.277778 | 0.358974 | 0.473684 |
| ALMI | 0.258065 | 0.444444 | 0.378378 | 0.411765 |
| ASMI | 0.500000 | 0.358974 | 0.372093 | 0.350000 |
| IMI | 0.305085 | 0.392157 | 0.408163 | 0.377358 |
| ILMI | 0.530612 | 0.490566 | 0.489796 | 0.470588 |
| HC | 0.784314 | 0.814159 | 0.846847 | 0.839286 |

## Method and limits

The fixed feature extractor keeps signed XYZ DCT coefficients and RMS, mean absolute amplitude, minimum and maximum in approximate R-relative windows. Those windows are not clinical delineations. The bank has no learned cohort state. Only exact training patients fit feature normalization, with equal patient, record and within-record beat weights. Seeded unit-norm random projections operate on feature space, without rotating or unit-normalizing physical XYZ.

All four arms use the same window/projection/quantile dimensions. Median_waveform encodes each recording's coordinate-median waveform. Mean_embedding repeats its mean projected beat feature on the grid. Mean_spread encodes only mean and population SD as mu + sigma times fixed standard-normal quantiles. This is a moment-control basis, not a claim that physiological beat distributions are Gaussian. Quantiles uses empirical linear-interpolation quantiles of all original beats in that record. Every grid is divided by sqrt(Q); no per-arm StandardScaler undoes that normalization. Recording tensors pool equally per patient. Effective feature ranks differ despite identical stored dimensions.

Class-balanced ridge with intercept and Cholesky solver receives the same two declared alpha candidates in every arm. Three inner patient folds choose alpha using mean inner F1; configured-order ties. Every outer refit excludes its query patients from learned preprocessing and selection. All three projection seeds remain in equal probability ensembles. Softmax of ridge scores at fixed temperature one is an aggregation interface, not calibrated clinical risk. Seed SD measures feature-projection variation, not independent patients or a generalization confidence interval. Per-seed scores and mean-member differences remain in summary.json.

The primary contrast tests quantile shape beyond mean/spread. Other contrasts distinguish encoding-before-pooling and simpler variability effects. No outer-score-based arm, quantile, window, budget, projection, source encoder, threshold or seed-subset selection occurs. The repeatedly inspected PTB cohort remains development data. Patient bootstrap shares class-stratified draws across arms and seeds, with no refits or corrections for repeated development, overlapping training folds or multiple comparisons. More beats are not more independent patients. Neither a high point score nor a positive conditional interval establishes novelty, SOTA, clinical validity or guaranteed 0.80–0.90 performance.

This is a compact projected-quantile adaptation inspired by set/distribution pooling, not an exact implementation of PSWE or FSW and not a claim to their injectivity guarantees. It uses no new patients, ECG input, source pretraining, learned neural encoder or Tucker compression. Historical neural0.584570, random-convolution0.584266 and supervised-transfer results remain separate development references, not the matched controls here.

Protocol: docs/distribution_pooling_plan.md. Research basis: https://papers.nips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html and https://proceedings.iclr.cc/paper_files/paper/2025/hash/3dbb8b6b5576b85afb3037e9630812dc-Abstract-Conference.html.
