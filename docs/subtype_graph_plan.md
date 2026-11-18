# Graph regularization for the six MI and control scores

Test whether sharing weights between specific VCG subtype classifiers improves patient macro F1. Use the original standardized 3,584-column combined convolution features from the independently replayed convolution distribution bank. The baseline is fixed convolution ridge with a historical ensemble score of 0.584266.

Each classifier minimizes class-balanced squared error with an unpenalized intercept and the existing ridge penalty. The graph adds an L2 penalty on differences between the weight vectors of linked classes. The six original score columns and argmax labels are used throughout. A shared penalty permits nearby classifiers to borrow strength and can also blur distinctions. The graph is an experimental prior derived from class names.

| Arm | Edges | Candidate graph strengths |
|---|---|---|
| Ordinary ridge | None | 0 |
| Subtype graph | AMI–ALMI, AMI–ASMI, IMI–ILMI | 0, 0.5, 2 |
| Relabeled graph | IMI–ALMI, IMI–ASMI, AMI–ILMI | 0, 0.5, 2 |

The relabeling preserves the graph topology and spectrum with HC isolated in both arms. The links are prespecified. Each graph penalty scales by the selected ridge alpha. Eigendecomposing the six-class Laplacian turns the fit into independent ridge solves in a rotated **output** basis. Feature coordinates and physical XYZ do not rotate.

Use the original ten outer folds, three inner folds and three feature seeds. Within each outer-fold/seed group, reproduce the ordinary ridge choice of alpha from [1, 100] on the three inner folds. Use that same alpha for both graph arms and select each arm's graph strength using mean inner macro F1, with ties favoring off, then weaker sharing. Both graph arms have identical tuning budgets. Off restores the exact original classifier. The primary contrast is subtype graph versus ordinary ridge. The contrast against the relabeled graph tests the specificity of the proposed links.

Each run checks 120 frozen feature contexts and 210 source ridge references, computes 360 positive inner graph candidate fits and fits up to 60 selected positive outer models. An independent process reconstructs the graphs, baselines, choice scores and candidate solutions, checks saved coefficients and probabilities, and reports all three seed results, class metrics, uniform probability ensembles and descriptive paired patient intervals. The cohort is the repeatedly inspected development set.

```bash
.venv/bin/python scripts/run_subtype_graph.py --smoke
.venv/bin/python scripts/run_subtype_graph.py
```

Primary precedent for related output functions with kernel regularization. [Evgeniou, Micchelli and Pontil](https://www.jmlr.org/papers/v6/evgeniou05a.html)
