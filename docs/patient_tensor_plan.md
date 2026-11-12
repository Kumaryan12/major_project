# Modern VCG tensor and patient-level learning: prespecified development plan

## Why this tensor

The completed single-beat ResNet reached 0.586 patient macro F1. Adding
192 wavelet statistics reached 0.584, while the previous correlation and
covariance representations did not establish an improvement over their
matched control. These results motivate preserving waveform morphology
and learning with patient-level labels, rather than assuming that a larger
tensor or another reconstruction-energy decomposition will help.

The new input has meaningful axes: **beats × XYZ × time × scales**. A
training patient bag has 16 beats, each with 3 leads and 651 samples. One
scale is the unmodified, training-normalized waveform. Three additional
shared, trainable temporal filters initialize as local averages of widths
9, 33 and 129 samples. Filters do not mix XYZ. After training they can have
signed responses; they are not guaranteed physiological frequency bands.

The tensor is computed on demand, not saved as another huge feature bank.
Both old waveforms and the frozen clinical labels/folds remain unchanged.

## Four ablations, not an untested stack

| Model | What changes | Parameters (6 classes) |
|---|---|---:|
| `bag_mean` | ResNet embeddings, uniform patient pooling | 78,198 |
| `bag_attention` | Same encoder, gated patient attention | 82,390 |
| `multiscale_attention` | Add learned scale views with fixed temporal compression | 89,657 |
| `tucker_attention` | Replace fixed compression with supervised Tucker mode factors | 89,294 |

All retain the same CNN waveform path. The two scale-view models add a
32-dimensional auxiliary embedding. Their capacities are close, not
perfectly matched. The fixed-compression arm is necessary: any gain from
additional views alone is not evidence that Tucker compression helped.

The auxiliary views first pool time to 64 bins. Learned factors map lead,
time and scale modes to ranks **3, 12, 3**, producing a 3 × 12 × 3 core per
beat. Full lead rank retains capacity across XYZ. Factors initialize with
data-independent cosine bases and are optimized end-to-end by patient
classification loss. An orthogonality penalty of 0.001 regularizes their
Gram matrices; it does not strictly enforce orthogonality or prove
physiological interpretability. The fixed control average-pools time to
12 bins and retains all scales.

**No order-dependent factor contracts the beat axis.** Training bags are
random sets from one patient, possibly across that patient's recordings.
They are not continuous 16-beat temporal sequences. Mean/gated pooling is
permutation-invariant and masks padding. Attention values are learned
weights, not validated explanations or signal-quality annotations.

## Training and patient safety boundaries

- Primary input remains measured Frank XYZ only: 164 patients, 391 records,
  55,863 original beats, six unchanged classes.
- A sampled bag contains one patient's beats only, without replacement
  within that bag. Short bags use padding/masks rather than duplicated
  valid beats. Patients can recur across training draws.
- Each epoch draws 512 class/patient-balanced bags, up to 8,192 beat
  presentations. Batch size eight bags matches the prior nominal batch
  of 128 beat presentations. The loss is one unweighted class prediction
  per bag; it is not the previous per-beat objective.
- All architectures/optimizer/ranks are fixed in `configs/patient_tensor.yaml`.
  There is no outer-fold selection of rank, fusion weight, seed or ensemble.
- Inner patient splits select early stopping (25 epochs maximum, patience
  six). Their rounded median best epoch sets a fresh outer-training refit.
  Normalization, filters, factors, encoder and attention are trained on the
  appropriate training partition only. Existing outer-fold encoder weights
  are NOT reused inside inner validation, which would leak those patients.
- At inference, every eligible patient beat is encoded exactly once in
  bounded batches. All embeddings are pooled together. No random test
  bags, most-confident-beat selection or fixed test subset is used.
- Offline full-record denoising remains noncausal; this is not a streaming
  model. No PTB-XL test records are used, rescored or tuned against.

## Repeatability and honest comparison

`configs/neural_repeat.yaml` runs the plain single-beat ResNet for training
seeds **42, 123 and 2026**, keeping BOTH outer and inner partitions fixed.
The completed seed 42 run is reused after checking its protocol, frozen
cohort, predictions and seed audits. Other seeds are freshly fit. Results
include all seeds and their mean/sample standard deviation, not a selected
best seed or a test-optimized ensemble. Source/backend metadata are retained;
this is not a promise of cross-platform bitwise identity.

The patient-tensor ablation starts with seed 42. Its primary paired reference
is the existing seed-42 ResNet; multi-seed repeatability is a separate
question. If an architectural change warrants more study, a future
preregistered multi-seed comparison must repeat the controls too—not only
the model that happened to score best on development folds.

Reports validate complete prediction coverage, frozen IDs/labels/folds,
inner-only epoch selection and absence of final-fit validation patients.
They contain class-level F1, aggregate confusion review, and paired
bootstrap differences against the ResNet, patient mean/attention and fixed
compression controls. Bootstrap intervals are conditional on fitted
predictions; they do not adjust for correlated training folds, model/seed
selection, repeated development or multiple comparisons.

Errors motivate consistent signal/annotation review, not relabeling cases
to agree with predictions. No clinical labels are changed by these tools.
The same repeatedly studied PTB patients remain development data. Independent
validation is still required, and 0.80–0.90 macro F1 is not guaranteed.

## Commands and artifacts

```bash
.venv/bin/python -m mi_localization.neural_repeat
.venv/bin/python -m mi_localization.patient_tensor
.venv/bin/python -m mi_localization.patient_tensor_report
```

Repetition outputs: `results/neural_repeat_v1/`. Patient models:
`results/neural_patient_tensor_v1/complete/`. The patient runner supports
`--outer-folds 1` for a separately stored partial run, never complete CV.
Inner and outer checkpoints permit continuation. Source/config/runtime/
bank/fold fingerprints refuse incompatible resumes. Original completed
reports are not overwritten; new optional seed controls change the neural
training-module source checksum, so a legacy training resume should use
its original code revision or a new output directory.

Fitted models, patient predictions and feature arrays stay local and
gitignored. Aggregate results and audits can be pushed. Do not describe
any model as fully evaluated until every outer fold finishes.

## Research basis and what is not implemented

- [Attention-based Deep Multiple Instance Learning, Ilse et al. (2018)](https://proceedings.mlr.press/v80/ilse18a.html)
  motivates permutation-invariant gated pooling of instances.
- [Tensor Regression Networks, Kossaifi et al. (2020)](https://www.jmlr.org/papers/v21/18-503.html)
  motivates end-to-end learned low-rank mode contractions, rather than
  treating a separately fitted unsupervised core as a final representation.
- [ST-MEM, Na et al. (ICLR 2024)](https://arxiv.org/abs/2402.09450)
  motivates a possible later lead/time masked-pretraining study. Such
  pretraining is **not implemented here** and would have to exclude the
  appropriate inner/outer held-out patients even without using labels.

This is an evidence-driven VCG adaptation, not an exact reproduction of
those papers, a claim of established novelty, or demonstrated SOTA. A
large Transformer, graph model, masked pretraining, clinical delineation,
and ECG + VCG extension are deliberately not claimed as completed work.
