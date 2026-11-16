# What the consecutive-beat diagnostics tell us

Completed **16 November 2026**, without additional neural fits. [Reproducible audit](../results/neural_chronological_diagnostics_v1/REPORT.md), [confusion figure](../results/neural_chronological_diagnostics_v1/class_confusions.png), [training histories](../results/neural_chronological_diagnostics_v1/training_histories.png).

## Findings

- Historical beat ResNet: **0.550 ± 0.040**, ordered+RR: **0.524 ± 0.018**, shuffled+RR: **0.533 ± 0.013**. These are three-seed means and sample SDs. The matched order-control contrast remains negative and uncertain; cross-procedure differences cannot isolate the effect of order or a single architectural change.
- The largest ordered-versus-ResNet class F1 losses are **IMI 0.548→0.471** and **ILMI 0.576→0.525**. HC changes **0.837→0.847**. Chronology does not solve the difficult subtype distinctions under this recipe. Confusion figures show repeated predictions from the same patients, not 492 independent cases.
- **36 patients** are wrong in every ResNet seed, **40** in every ordered+RR seed, with **25 shared**. In each seed, ordered+RR recovers 22/17/16 cases that ResNet misses, while losing 25/24/16 ResNet-correct cases. These counts describe complementarity; no oracle routing or ensemble performance is claimed. Counts and macro F1 weight errors differently.
- Recomputing the exact saved ResNet beat probabilities using equal-record aggregation changes its three-seed mean from **0.550108 to 0.549727**. Individual seeds move in different directions. That aggregation change alone does not explain the average gap. This post-hoc sensitivity does not retroactively change the original inner stopping protocol or establish a new best result.
- Every corresponding waveform normalization matches exactly across **1,200** inner/outer comparisons. Source beats, labels and frozen folds match. That rules out these particular mismatches; sampling and objectives still differ.
- Among the **90 inner fits at the selected lengths**, median best epoch is **4 for ResNet versus 8 for ordered+RR**. **Zero** selected-length fits in either arm peak at epoch 25. Longer optimization alone is not supported as the explanation for the gap. This does not establish a training ceiling or rule out another optimizer/stopping recipe.
- Median final inner loss is **0.039 versus 0.170**, but a beat objective and a window objective are different. The loss gap is compatible with a harder learning problem, not proof of lost information. Later-epoch curves use fewer surviving fits after early stopping; consult the CSV's `n_fits` column.

## Focused next hypothesis

Preserve the direct **64-dimensional beat embedding → class logits** path, and add temporal context as a gated residual rather than making all predictions depend on the **32-dimensional recurrent state**. Retain explicit beat supervision alongside a window objective. This could make useful morphology easier to preserve while the model learns whether context contributes. It is a hypothesis, not a demonstrated explanation or a novel/SOTA claim.

Before fitting, specify a compact comparison with a fresh beat-only control, a residual-context arm, an auxiliary-beat-loss ablation, and corresponding shuffled-context controls. Match the waveform encoder, exposure, patient/record sampling, normalization and inference aggregation across relevant arms. Choose any length/epochs only inside the training folds. Keep every declared seed and original patient; don't route or filter patients using this error analysis. A future comparison on these repeatedly inspected folds remains development work.

Train from scratch within each partition, or pretrain only on that exact partition. The saved outer ResNet checkpoints saw their inner-validation patients and therefore cannot initialize an allegedly patient-disjoint inner selection procedure. No follow-up full training study was launched as part of this diagnostic.

## Reproduce

```sh
.venv/bin/python -m mi_localization.chronology_diagnostics
```

The command reads existing local predictions/audits, verifies cohort and saved-score alignment, reconstructs original beat aggregation, compares normalization and stopping histories, then writes aggregate metrics and figures to `results/neural_chronological_diagnostics_v1`. Source-file hashes pin the analysis; changed code/inputs require a new output namespace. It does not invoke a neural trainer or modify completed study artifacts. Individual patient comparisons remain in the ignored `review/patient_comparison.csv`.
