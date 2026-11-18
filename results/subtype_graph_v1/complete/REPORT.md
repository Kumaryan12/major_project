# Subtype graph regularization

All 164 patients, 391 recordings and 55,863 original source beats. Replayed all 120 feature contexts, 210 ordinary ridge references, 360 positive inner candidates and 31 positive outer fits. The original convolution ridge metrics reproduced exactly.

| Arm | Mean member macro F1 ± seed SD | Equal-seed ensemble macro F1 |
|---|---:|---:|
| ridge | 0.572860 ± 0.019471 | 0.584266 |
| subtype_graph | 0.575819 ± 0.025973 | 0.591424 |
| relabeled_graph | 0.576249 ± 0.025384 | 0.582026 |

The primary contrast is subtype graph versus ordinary ridge. The relabeled graph checks whether the declared subtype links improve on the same amount of graph structure.

| Paired comparison | Ensemble macro F1 difference | Descriptive patient 95% interval |
|---|---:|---:|
| subtype_graph_minus_ridge | +0.007158 | [-0.012932, +0.028282] |
| subtype_graph_minus_relabeled_graph | +0.009398 | [-0.001798, +0.026507] |
| relabeled_graph_minus_ridge | -0.002241 | [-0.017136, +0.010498] |

| Arm | Sharing off | Strength 0.5 | Strength 2 |
|---|---:|---:|---:|
| subtype_graph | 13 | 4 | 13 |
| relabeled_graph | 16 | 5 | 9 |

| Class | Ordinary ridge F1 | Subtype graph F1 | Relabeled graph F1 |
|---|---:|---:|---:|
| AMI | 0.344828 | 0.344828 | 0.333333 |
| ALMI | 0.482759 | 0.466667 | 0.482759 |
| ASMI | 0.633333 | 0.644068 | 0.633333 |
| IMI | 0.542373 | 0.526316 | 0.517241 |
| ILMI | 0.577778 | 0.622222 | 0.590909 |
| HC | 0.924528 | 0.944444 | 0.934579 |

Both graphs preserve six independent score columns while penalizing differences between linked weight vectors. Both use the same inner-selected ordinary ridge alpha and the same three graph strengths. Only the MI label assignments on the edges differ. The control preserves topology and keeps HC isolated, as does the subtype graph. Zero strength uses the original ridge probabilities.

All ten original patient folds and all three feature seeds contribute to the results. The study uses verified convolution feature caches with independent classifier replay. The graph reflects relationships proposed from the label names. Patient bootstrap intervals describe fixed predictions on this repeatedly inspected development cohort and omit refitting and repeated-development effects.
