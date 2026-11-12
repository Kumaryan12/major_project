# Compact VCG waveform development

## Scope and prespecified comparison

Measured Frank XYZ remains the primary input. This phase implements a
compact 1D residual network and a statistics-fusion ablation, not an ECG +
VCG model, pretrained foundation model, attention architecture or SOTA claim.
The existing fixed depth-3 XGBoost with short-wavelet statistics is the
matched patient-cohort control (0.468723 macro F1). The historical 0.487
depth-3/6 nested benchmark has a different model-selection protocol.

The new waveform cache contains the same 55,863 denoised 651-sample XYZ
beats from 391 recordings and 164 patients. Existing R positions, six labels
and ten patient folds are unchanged. Processing is offline/noncausal because
denoising uses the full recording. No detector is recalibrated and no
PTB-XL test record is read or scored in this phase.

## Architecture

The stem is a 15-sample, stride-2 XYZ convolution. Three residual stages
have widths 16, 32 and 64, two 7-sample convolutions per stage, GroupNorm
and GELU activation. Four coarse temporal pooling bins retain more timing
structure than a single global average. A 64-dimensional waveform embedding
feeds the six-class head, with dropout 0.2. The plain model has 78,198
parameters. The fusion variant adds a 32-dimensional embedding of the
existing 192 short-window wavelet statistics, for 84,566 parameters.

The pooling implementation is mathematically equivalent to adaptive
average pooling, including overlapping bins for non-divisible lengths.
Explicit bin means avoid an unsupported native MPS operation; tests compare
both outputs and gradients to PyTorch's CPU reference.

This is a small, fixed architecture before the experiment. The feature
fusion model is trained separately, not chosen or combined using outer
test scores. No data augmentation or self-supervised pretraining is used.

## Patient-only training and stopping

For each outer fold and model:

1. Split the outer training patients into three inner folds.
2. Fit waveform and statistical normalization only on the inner training
   patients. Equal class/patient weights define the moments. Waveforms use
   fixed training-set lead means/scales, preserving relative amplitude;
   they are not normalized to unit amplitude beat by beat. Statistics use
   training moments, with standardized values clipped at ±10.
3. Train with AdamW (learning rate 0.001, weight decay 0.01), batch size 128,
   up to 25 epochs and patience six. Monitor **patient macro F1** on every
   inner validation patient's complete set of beats. Select the earliest
   epoch attaining the best score.
4. Take the rounded median of the three selected epochs. Initialize a new
   model, refit normalization on all outer training patients and train for
   that fixed number of epochs, without an outer-test validation loader.
5. Predict every outer-test beat, average probabilities per patient, and
   score one six-class prediction per patient.

Each training epoch draws 8,192 beats **with replacement**, balancing
classes and patients within class. Unweighted cross entropy is used to
avoid applying these weights twice. There are 55,863 beats but only 164
independent patients. This neural optimization budget differs from
XGBoost's all-training-beat fitting and is disclosed in comparisons.

These repeatedly inspected PTB folds remain a development benchmark.
Neither nesting nor fresh initialization removes prior representation
selection on this cohort. Improvement still needs independent confirmation.

## Reproduce and resume

```bash
.venv/bin/python -m pip install -e '.[neural,dev]'
.venv/bin/mi-localization neural-features
.venv/bin/mi-localization neural-study
.venv/bin/python -m mi_localization.neural_report
```

The configuration is `configs/neural_vcg.yaml`. The waveform bank is
`data/processed/neural_vcg_v1/`. Aggregate results go to
`results/neural_vcg_v1/complete/`. Inner epoch-selection audits and outer
probability checkpoints allow rerunning the same command to resume.
Partial runs, e.g. `--outer-folds 1`, go to a separate directory and are
explicitly labelled incomplete. A partial run is not complete CV.

Study fingerprints include the configuration, feature-bank fingerprint,
frozen-fold file contents, training-module source hash, PyTorch/NumPy
versions, backend and platform. Changed settings/runtime require a new
output directory. Source checksums are not a substitute for validation.

`device: auto` selects available CUDA/MPS, otherwise CPU. CPU/CUDA request
deterministic algorithms; seeded MPS does not provide a bitwise
determinism guarantee. Cross-platform/version agreement is not promised.
All backends must pass finite forward/backward and probability checks.

The report requires complete, matched patient predictions for every
configured model and the fixed control. It checks patient IDs, labels,
folds, selected epochs and recomputed scores. It includes class-level F1
and 5,000 paired class-stratified patient bootstrap draws. These intervals
are conditional on the fitted predictions; there are no bootstrap model
refits or adjustments for repeated development, correlated training folds
or multiple comparisons.

Raw signals, feature arrays, neural weights and patient predictions stay
local and gitignored. Only aggregate audits/results are published. A
completed report is not claimed until the real full experiment finishes.

## Research motivation and future phases

Strodthoff et al., [Deep Learning for ECG Analysis: Benchmarks and Insights
from PTB-XL](https://arxiv.org/abs/2004.13701), motivates residual waveform
baselines, not a predicted score on our VCG task. PyTorch's
[reproducibility guidance](https://docs.pytorch.org/docs/2.15/notes/randomness.html)
describes why seeded runs do not guarantee cross-platform equivalence.

Only after this comparison should we consider learned tensor-core fusion,
multi-beat inputs or training-patient-only self-supervised learning. ECG +
VCG is a separate extension with matched controls. None has been
implemented in this phase. The target of 0.80–0.90 macro F1 is aspirational,
not an expected outcome or a justification for tuning on test patients.
