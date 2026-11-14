# Bounded regularization and augmentation on measured VCG

## Question

Can a small, inner-patient-selected training policy improve the original 651-sample denoised XYZ ResNet? The three-seed baseline averages 0.550 patient macro F1. Removing denoising averaged 0.521; a longer 1,001-sample window averaged 0.538. Neither comparison established a consistent gain. This experiment retains the original input, architecture, cohort and folds and tests training policy.

These same PTB patients have been inspected repeatedly. This is exploratory development, not independent confirmation. A selected policy is not assumed to improve the score, and the project target of 0.80–0.90 is not promised.

## Exactly four presets

| Preset | Dropout | AdamW weight decay | Label smoothing | Common XYZ gain | Additive noise SD / training lead SD |
|---|---:|---:|---:|---|---:|
| baseline | 0.20 | 0.01 | 0 | 1 | 0 |
| regularized | 0.40 | 0.05 | 0.05 | 1 | 0 |
| augmented | 0.20 | 0.01 | 0 | Uniform(0.95, 1.05) | 0.005 |
| combined | 0.40 | 0.05 | 0.05 | Uniform(0.95, 1.05) | 0.005 |

The strengths are declared experimental choices, not tuned from the outer-fold results or claimed clinically validated invariances. The regularization preset changes three settings together; this experiment cannot isolate which of them helps. The augmented preset similarly combines gain and noise. The four presets are a bounded two-by-two design, not a broad hyperparameter search.

## Patient-only selection

For each of seeds 42, 123 and 2026, and each of the ten frozen outer patient folds:

1. Keep the outer-test patients apart. Form the same three frozen-seed inner patient partitions used in the previous studies.
2. Evaluate each of the four presets on all three inner partitions. Fit normalization to unaugmented inner-training patients only. Early stopping uses only inner-validation patient macro F1, with up to 25 epochs and patience 6.
3. Select the preset with the highest arithmetic mean of its three best inner patient macro F1 scores. Exact ties follow the configured order, starting with baseline. Do not choose a seed using outer scores.
4. Set each final refit's epoch count to the rounded median of that preset's three inner selected epochs. Train a fresh fixed baseline and a fresh selected model on all outer-training patients without monitoring outer-test scores. If baseline wins selection, reuse that same final baseline fit and probabilities exactly.
5. Evaluate every original outer-test beat unchanged and average beat probabilities into one prediction per patient. Only the fixed baseline and the selected procedure receive outer evaluation; unselected presets do not receive additional outer tests.

The final comparison is **fixed baseline versus bounded inner-selected procedure**. The selected procedure receives additional tuning compute; it is not an equal-search-budget comparison or an isolated augmentation-effect estimate. Baseline inner fits are shared with the selection procedure. The full study requires 390–420 unique fits: 3 seeds × 10 folds × (12 inner fits + 1 or 2 final refits). The exact count and per-fold selections are reported after completion.

## Matched data and training budget

All 164 patients, 391 recordings and 55,863 original beats remain. Inputs are measured Frank XYZ at 1 kHz with the original full-record wavelet denoising and −250/+400 ms crop, 651 samples. The waveform-only ResNet has 78,198 parameters, widths 16/32/64 and four temporal pooling bins. Learning rate stays 0.001; batch size stays 128. Each epoch draws 8,192 beats with replacement, balancing classes and patients within each class. Cross entropy is otherwise unweighted, avoiding double class balancing.

All candidates use matching initial weights and the same sampled training rows for shared epochs. Their unaugmented training-partition normalization must match exactly. Different stopping epochs are allowed. The baseline invokes the existing frozen trainer directly; a CPU test also verifies exact equivalence between that trainer and the new trainer with all added effects disabled. Model weights, row hashes, fitting counts and stopping histories are validated in the final report.

There is no new dataset, beat exclusion, label change, redetection, longer window, ECG input, pretraining, tensor feature path or architecture change. Existing experiment modules, results and folds remain intact. MPS is not guaranteed bitwise deterministic.

## Augmentation mechanics and assumptions

Augmentation runs on sampled **training occurrences only**, using an independent seeded CPU generator. It does not consume the sampler or model/dropout random streams. Augmented and combined presets share the same augmentation stream for shared epochs. Each occurrence receives fresh draws; no augmented examples are added to the cached dataset or counted as new independent patients. The same 8,192 presentations/epoch budget is retained.

One positive scalar gain is shared by all XYZ leads and times of a beat. It multiplies the physical signal before training-set centering/scaling. If normalized input is `z = (x − mean) / scale`, the implemented equivalent is `z_aug = gain*z + (gain − 1)*mean/scale + epsilon`. The Gaussian noise `epsilon` has standard deviation 0.005 in training-standardized units, independently across samples/leads, so its physical SD is 0.005 times that lead's weighted training-set SD. Moments are never recomputed on individual augmented beats.

Common gain preserves vector direction before added noise, but diagnostic amplitude information can change. Additive noise can also alter small features. These perturbations are small proposed nuisance variations, not proof of preserved MI labels or observed acquisition noise distributions. Inner selection includes the untouched baseline so augmentation is optional. No lead flipping/rotation, time reversal/warping, segment masking or artificial baseline shifts are introduced.

Validation and test loaders never augment. Dropout is disabled by evaluation mode; prediction uses every original beat once. No test-time augmentation or new ensemble is introduced. Original waveform arrays remain unchanged.

## Run and monitor

```bash
# Complete four-preset nested study; compatible checkpoints resume.
.venv/bin/python -m mi_localization.regularization
# Validate every seed/fold and generate the paired report.
.venv/bin/python -m mi_localization.regularization_report
# Read-only status; never starts or resumes training.
.venv/bin/python -m mi_localization.regularization --status
```

Configuration: [`configs/regularization_vcg.yaml`](../configs/regularization_vcg.yaml). New outputs: `results/neural_regularization_v1/complete/`. No new waveform bank is needed. The study hashes the existing bank files, beat table, paper settings, frozen folds, relevant source modules and runtime; changed inputs/settings refuse a silent resume. Inner fits and paired outer predictions are saved atomically. Partial-fold studies have separate output directories and cannot produce a full report. Failures are recorded locally and do not automatically restart training. Status shows saved progress, not proof that the PID is still alive.

## Reporting

The final report requires all folds and all three seeds. It recomputes patient macro F1 from aligned predictions; validates all four inner candidate histories, train/validation fitting scopes, selection rule, epoch budgets, matching initial weights, sampled rows, normalizers and augmentation streams; and checks that outer refits never use validation stopping. A baseline fallback must reuse the same fit and full patient probabilities exactly.

Report all three paired scores, mean/sample SD, per-class F1, selected presets by seed/fold and total fit count. A 5,000-draw class-stratified paired patient bootstrap shares patient resamples across both arms and all fixed seeds. Repeated patient/seed predictions are not independent patients. Intervals condition on fitted predictions/seeds; they do not correct for correlated training folds, repeated development or multiple comparisons. No best seed or unplanned ensemble is selected.

Waveforms, fitted weights, individual predictions and transient logs remain local. Public outputs contain aggregate metrics and selection/training audits. The already used PTB-XL test is not rescored and cannot serve as fresh confirmation.

## Primary sources informing the design

- Raghu et al., [Data Augmentation for Electrocardiograms](https://arxiv.org/abs/2204.04360), CHIL 2022: their empirical work finds augmentation effectiveness task-dependent. This motivates local inner-patient selection, not a claim that their gains transfer to measured VCG localization. We do not implement their TaskAug algorithm here.
- Dade et al., [A Survey of Augmentation Techniques for Enhancing ECG Representation Through Self-Supervised Contrastive Learning](https://www.cinc.org/archives/2024/pdf/CinC2024-223.pdf), Computing in Cardiology 2024: evaluates noise/scaling and other augmentations in a different ECG/low-LVEF setting, reporting dependence on strength/data size and limited gains. It does not validate our VCG strengths, labels or expected performance.
- Official PyTorch documentation for [Dropout](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html), [AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html), and [CrossEntropyLoss label smoothing](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) defines the implemented training operations.
