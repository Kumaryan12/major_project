# Probabilistic MI subtype readout on measured XYZ

Replayed 600 matched procedures across 120 training/query partitions, 164 patients, ten frozen folds and three feature seeds.

| Readout | Mean seed macro F1 | Three-seed ensemble macro F1 |
|---|---:|---:|
| flat | 0.546352 | 0.562819 |
| hierarchical | 0.549633 | 0.586828 |
| historical_selector | 0.562847 | 0.603374 |

The primary comparison is hierarchical minus flat.

| Contrast | Ensemble difference | Descriptive paired patient 95% interval |
|---|---:|---|
| hierarchical_minus_flat | +0.024010 | [-0.001207, +0.056060] |
| hierarchical_minus_historical_selector | -0.016546 | [-0.076801, +0.047500] |
| flat_minus_historical_selector | -0.040555 | [-0.094470, +0.014164] |

| Class | Flat | Hierarchical | Historical selector |
|---|---:|---:|---:|
| AMI | 0.242424 | 0.303030 | 0.344828 |
| ALMI | 0.516129 | 0.545455 | 0.500000 |
| ASMI | 0.629630 | 0.666667 | 0.633333 |
| IMI | 0.526316 | 0.526316 | 0.612903 |
| ILMI | 0.565217 | 0.565217 | 0.604651 |
| HC | 0.897196 | 0.914286 | 0.924528 |

| Readout | Within-MI confusions | MI predicted HC | HC predicted MI |
|---|---:|---:|---:|
| flat | 48 | 7 | 4 |
| hierarchical | 47 | 5 | 4 |
| historical_selector | 44 | 5 | 3 |

Both new readouts use the same cached feature matrix in each partition. Their shared representation came from the historical train-only selector. Each readout uses the same three-value C grid and its own inner-score choice. The hierarchical model multiplies MI probability by conditional subtype probability and retains HC probability. It evaluates all six outcomes for every patient.

This is repeatedly inspected PTB development data. Paired intervals condition on these predictions and omit refitting and repeated-development effects. The historical selector is a read-only performance reference.

[Protocol](../../../docs/probabilistic_subtype_plan.md)
