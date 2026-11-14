# Cross-patient contrastive follow-up: minibatch patient diversity

## Completed first study

All three seeds and both objectives completed the original structured-batch
experiment. Mean patient macro F1 was **0.479 ± 0.042** for its matched CE
control and **0.516 ± 0.050** for cross-patient SupCon. Mean paired delta
was +0.037, with a descriptive conditional patient-bootstrap interval
[-0.015, +0.088]. The interval includes zero, and two of three seeds did
not improve. Neither objective beat the earlier plain ResNet's three-seed
mean of 0.550. Results are retained in
`results/neural_contrastive_v1/complete/REPORT.md` (commit `921fbba`).

## Hypothesis—not an established cause

The original structured batch contained only **12 distinct patients**:
six classes × two patients × eight beats. Its 96 beats are therefore much
more clustered by person than the earlier weighted-random 128-beat batch.
Under that older sampler, the expected number of distinct patients is
81.9–82.2 across the frozen outer training folds. This is an analytical
expectation from the class/patient probabilities, not a measured effective
sample size or evidence that all beats are independent.

The CE result suggests it is worth testing whether low within-batch patient
diversity hurts optimization. It does NOT prove this explanation. The
follow-up uses **48 distinct patients** per batch: eight patients per class
× two beats. All current inner training splits have at least nine patients
in every class, so this is feasible without using validation/test patients.

Both policies have the same expected class/patient-balanced marginal beat
sampling distribution. Their within-batch correlations differ. With eight
patients per class and two beats, each contrastive anchor also has 14
different-patient positives instead of eight. These are coupled consequences
of the new batch composition; this experiment cannot isolate them perfectly.

## Exactly what changes

Only `patients_per_class` (2 → 8), `beats_per_patient` (8 → 2), and the output
directory change relative to `configs/contrastive_vcg.yaml`.

Unchanged: waveform cache, cohort, labels, R peaks, class order, inner/outer
folds, three training seeds, 78k-parameter ResNet, 96-beat batch size,
86 batches/epoch, maximum epochs/patience, optimizer, temperature and loss
coefficient. Both arms receive the new batch policy. There is no new
architecture, projection head, augmentation or pretraining, and existing
results are not overwritten.

Compare the new contrastive model against its NEW matched CE control. Do
not declare a contrastive gain by comparing only with a different sampler's
score. Preserve every seed and report the paired mean difference, not a
best seed or outer-test-optimized ensemble.

This hypothesis was chosen AFTER viewing the previous results. The same
PTB patients remain development data; the follow-up is not independent
confirmation. No gain or 0.80–0.90 macro F1 is promised. The primary report
retains its conditional-bootstrap and repeated-development caveats.

## Run, inspect, and resume

```bash
.venv/bin/python -m mi_localization.contrastive --config configs/contrastive_vcg_diverse.yaml
.venv/bin/python -m mi_localization.contrastive_report --config configs/contrastive_vcg_diverse.yaml
```

Read saved progress without starting training:

```bash
.venv/bin/python -m mi_localization.contrastive --config configs/contrastive_vcg_diverse.yaml --status
```

New results: `results/neural_contrastive_diverse_v1/complete/`. Inner/outer
checkpoints, per-epoch progress, scope guards and aggregate-report validation
use the unchanged contrastive runner. Patient predictions, weights and
live/error/temp files stay local; aggregate completed results can be pushed.
