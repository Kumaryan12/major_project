# Matched cross-patient contrastive VCG experiment

## Question and scope

Does explicitly bringing together same-class examples from **different
training patients** improve six-class MI localization over an otherwise
identical cross-entropy (CE) control?

This tests a learning objective, not a larger model. Both arms use the
same 78,198-parameter measured-XYZ ResNet, 64-dimensional embedding and
six-class head. There is no extra projection head, pretraining, signal
augmentation, tensor branch, ECG input or feature memory bank. Neither
labels nor the 164-patient cohort, 391 recordings, 55,863 beat crops or
frozen outer patient folds are changed. All existing completed reports
remain intact.

## Structured batches shared by both arms

Each batch samples **two distinct patients in each of the six classes**,
then eight beats per patient, for 96 beats. Patients are chosen uniformly
within each class. Beats are drawn without replacement within that
patient's draw if possible, otherwise with replacement. Patients may
reappear in later batches. Every inner training pool is checked to contain
enough distinct patients before the study begins.

Both arms receive 86 such batches per epoch: **8,256 beat presentations**.
This differs from the previous ResNet's 8,192 weighted-random presentations
and batch size 128. Therefore the old three-seed mean of 0.550 is context,
not this experiment's matched control. A new CE model is trained for every
fold and seed; its score must not be assumed to equal the old result.

Dedicated epoch-seeded NumPy generators make sampled rows independent of
optimizer/model gradients. Audits record initial-weight hashes and every
epoch's sampled-row hash. The report checks that initial weights, model
capacity, training-only normalization and all shared-epoch batch sequences
match between arms.

## Loss

The control minimizes ordinary unweighted CE. The experimental arm
minimizes **CE + 0.1 × cross-patient supervised contrastive loss**, with
temperature 0.1. These coefficients are fixed in advance, not selected by
outer test scores.

For a training anchor, positives have the same class but a different
patient identity. All same-patient pairs, including self-pairs, are
excluded from both the positive mask AND contrastive denominator. We do
not inadvertently repel other beats of the same person as negatives.
Different-class, different-patient examples provide negatives. The loss
averages positive log probabilities with a stable masked log-sum-exp.

The loss uses cosine-normalized encoder embeddings; the classifier still
receives its original embeddings. XYZ is normalized with weighted
training-set lead moments, never unit-scaled beat by beat. No patient code
is an input to the encoder/classifier; codes are used only for batching,
contrastive masks and partition bookkeeping. Validation/test examples
never enter a contrastive batch or normalization fit.

The contrastive diagnostic is computed for CE too, but contributes **no
gradient** to that control. Tests verify that a zero-weight experimental
fit reproduces CE exactly on CPU.

## Fixed patient splits, three seeds, nested stopping

- Training seeds: **42, 123, 2026**. Split seed: **42**. Both inner and outer
  patient partitions are fixed across objectives and training seeds.
- Three inner patient folds select stopping using patient macro F1, up to
  25 epochs and patience six. The earliest best epoch is retained.
- A fresh model and normalization fit on all outer training patients uses
  the rounded median inner-selected epoch count. No outer test monitoring.
- Maximum budget and batch prefixes are matched. Actual epochs/updates may
  differ because the two objectives can select different stopping epochs.
- Every test beat is predicted; mean probabilities give one prediction per
  patient, preserving the original patient-level endpoint.
- There is no favorable-seed selection, outer-test-chosen fusion/ensemble,
  relabeling based on errors, or PTB-XL test access/rescoring.

Completed inner fits and outer predictions are checkpointed. Resumes
validate source/configuration/runtime/bank/frozen-fold fingerprints and
stored row coverage. Raw data and earlier results are not overwritten.

## Reporting and uncertainty

The final report requires all ten folds for both arms and all three seeds.
It recomputes patient scores and checks nested stopping, matching initial
weights and sampled-row hashes. Report all seed-wise F1 scores, their
mean/sample standard deviation, and paired objective differences. It also
includes mean class-level F1 across seeds.

For descriptive intervals, 5,000 class-stratified patient-bootstrap draws
are shared across all seeds/arms. Each draw averages seed-wise macro F1;
the seed/patient observations are **not** flattened into 492 independent
patients and predictions are not ensembled. Intervals condition on fitted
models and the fixed seeds: no model refits, seed resampling or adjustment
for repeated development, correlated training folds or multiple comparisons.
Training-seed SD is not a confidence interval for generalization.

All these PTB folds were already inspected. A gain remains development
evidence, not independent confirmation, clinical validation, novelty or
SOTA. Reaching 0.80–0.90 macro F1 is not guaranteed.

## Run, progress and resume

```bash
.venv/bin/python -m mi_localization.contrastive
.venv/bin/python -m mi_localization.contrastive_report
```

Saved-progress inspection, **without starting or interrupting training**:

```bash
.venv/bin/python -m mi_localization.contrastive --status
```

The status lists outer-fold counts for all six seed/arm combinations and
the last saved epoch/checkpoint. Timestamps/PID are recorded, but saved
progress is not a live process check. Most recent caught exceptions are
recorded locally; a forced process kill cannot necessarily log a traceback.

Re-running the training command resumes compatible checkpoints. For a
partial integration run, use `--outer-folds 1`; results are saved separately
and explicitly marked incomplete. A partial run is never complete CV.
Outputs: `results/neural_contrastive_v1/complete/`. Raw signals, waveform
arrays, patient prediction CSVs, fitted weights and temporary/live/error
files stay local. Only aggregate metrics and audits are published.

The existing waveform cache uses full-record denoising and is offline /
noncausal. Seeded MPS is not guaranteed bitwise deterministic; runtime
backend and library/source versions are recorded. CPU deterministic tests
do not establish cross-platform equality.

## Research basis and deferred work

[Khosla et al., Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
motivates learning embeddings with multiple same-class positives. The
cross-patient masking/structured batch policy is a task-specific adaptation,
not a claim to invent supervised contrastive learning. The implementation
does not copy the reference implementation or use its augmented two-view
protocol/projection head.

Geometry-aware masked pretraining, signal preprocessing ablations and the
ECG + VCG extension remain separate, **not implemented** in this phase.
