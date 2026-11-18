# Fixed average of four linear convolution readouts

Replayed the saved, independently verified outer probabilities for all 164 patients, ten folds and three seeds. Each readout contributes a fixed quarter within each seed. The three seed probability vectors are then averaged. This comparison uses zero new fits.

| Procedure | Mean seed macro F1 | Ensemble macro F1 |
|---|---:|---:|
| baseline | 0.572860 | 0.584266 |
| plus_mean | 0.548247 | 0.581913 |
| plus_spread | 0.538610 | 0.594098 |
| plus_quantiles | 0.549509 | 0.589485 |
| selected_linear | 0.562847 | 0.603374 |
| equal_average | 0.562487 | 0.586053 |

Equal average minus inner selector was **-0.017321** with descriptive paired patient 95% interval [-0.042859, +0.005133].

| Class | Inner selector F1 | Equal average F1 |
|---|---:|---:|
| AMI | 0.344828 | 0.333333 |
| ALMI | 0.500000 | 0.500000 |
| ASMI | 0.633333 | 0.610169 |
| IMI | 0.612903 | 0.557377 |
| ILMI | 0.604651 | 0.590909 |
| HC | 0.924528 | 0.924528 |

This is a development comparison on repeatedly studied PTB patients. The interval describes fixed predictions on this cohort and does not account for refitting or repeated study. The source regularization and shared representation selection were reproduced from their saved inner scores. Artifact verification covers probabilities and receipts from the earlier model restoration. The current comparison does not refit or reload the original classifiers.
