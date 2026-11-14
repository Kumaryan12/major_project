# Separate measured ECG + VCG extension

## Question and scope

The completed source-only masked encoder did not improve mean target macro F1 (scratch 0.553 ± 0.052 versus pretrained 0.520 ± 0.007 under the matched two-rate procedure). Earlier VCG-only attention, learned tensors, contrastive learning, longer windows and augmentation have not established a repeatable gain. A useful next test is whether the simultaneously measured ECG contains complementary information, rather than adding another VCG transformation and expecting a larger tensor to solve the small-patient problem.

Measured Frank VCG remains the primary benchmark. This is the separately authorized **actual twelve-lead ECG + measured VCG extension**, not a replacement or a claim of VCG-only performance. The same 164 patients, 391 recordings, 55,863 beats, six labels and frozen ten patient folds remain. No external source, PTB-XL record, pretrained checkpoint, age/sex metadata or old external-test predictions enter the new models.

## Freeze the paired signal bank

The ECG channels are I, II, III, aVR, aVL, aVF, V1–V6 in declared order. All fifteen measured channels share a WFDB time axis at 1,000 Hz. Finite physical mV signals, distinct leads, lengths/rates, header labels and complete crops are checked. The ECG uses the same full-record bior6.8 denoising and existing −250/+400 ms R-relative crops, with no new peak detection, resampling, template alignment, patient exclusion or label change.

Every recomputed Frank crop must exactly match the original VCG bank before its paired ECG crop is accepted. This verifies cached row and time alignment without using classifier outputs. ECG normalization is deferred to each training partition; per-beat normalization is not applied. Encoded source files, bank content, preprocessing code and runtime are fingerprinted. Raw inventories and per-record signal flags remain local; aggregate counts are public.

One discovered numerical edge case is explicitly handled in the new ECG denoiser: at wavelet threshold zero, soft-thresholding is mathematically the identity. Keeping coefficients unchanged avoids 0/0 for zero-valued coefficients. Ordinary positive-threshold outputs exactly reproduce the existing denoiser. The completed bank audit found **one recording with a zero noise threshold and zero recordings with entirely constant ECG leads**. A zero wavelet-noise estimate does not imply that the waveform itself is flat. Constant ECG leads, if present, are retained and counted; no signal is fabricated or patient dropped. The original VCG preprocessing and all historical results stay unchanged. Complete-record denoising remains offline/noncausal.

## Five predeclared models

| Model | Inputs and change | Parameters, six classes |
|---|---|---:|
| `vcg` | Original measured-XYZ residual network | 78,198 |
| `ecg` | Same single encoder with twelve-channel input convolution | 80,358 |
| `vcg_twin` | Two encoders, both VCG-only; second gets XYZ tiled four times | 160,694 |
| `joint_additive` | VCG and actual ECG encoders, additive compressed fusion | 160,694 |
| `joint_bilinear` | Same paired inputs and weights, low-rank bilinear interaction | 160,694 |

Each encoder outputs a 64-dimensional embedding. For a,b from the two branches and rank-16 learned factors U,V:

```
bilinear: tanh((U a) * (V b))
additive: tanh((U a + V b) / sqrt(2))
classifier input: concatenate(a, b, interaction)
```

The Hadamard product computes a compressed interaction of the 64 × 64 outer-product tensor without materializing it. Tanh follows the interaction for bounded nonlinear features. This is a task-specific adaptation of low-rank bilinear pooling, not an exact reproduction or a claim of new mathematics. Rank is fixed before the run and never selected on outer predictions.

The three dual models have the exact same initial trainable tensors, parameter counts, classifier and architecture dimensions. `vcg_twin` substitutes a deterministic tiling of normalized XYZ for its twelve-channel second input; this adds no ECG information. Both VCG-only arms do not even read ECG examples in their datasets. `ecg` does not read VCG examples. Shared-input normalization and shared-epoch sampled-row sequences match across arms. The VCG arm is tested to reproduce the original frozen trainer exactly on CPU.

Four contrasts are declared, and all will be reported:

1. `joint_bilinear − vcg`: total extension versus the primary sensor benchmark; model capacity also differs.
2. `joint_bilinear − vcg_twin`: replace the second VCG view with measured ECG at identical parameter count and bilinear architecture.
3. `joint_bilinear − joint_additive`: interaction rule with the same paired inputs and trainable tensors.
4. `joint_bilinear − ecg`: whether the complete fusion procedure improves over ECG alone; capacity also differs.

The tiling control is a deterministic redundant input, not a new physiological lead system. The additive interaction is a nonlinear compressed control, not plain concatenation. These contrasts do not form a full factorial study of every architectural or input choice.

## Equal target protocol

All models start from scratch: AdamW 0.001, decay 0.01, dropout 0.2, CE, gradient clipping 5, batch 128, 8,192 patient/class-balanced beat presentations per epoch, maximum 25 epochs and patience six. Three inner patient folds, fixed across all arms and seeds, select early-stopping epochs. Rounded median best epoch sets a fresh outer-training refit without test monitoring. No rate/rank search, extra loss, source pretraining, augmentation, probability calibration or test-optimized ensemble is included.

Training-only weighted per-lead moments use the same class/patient weights as the existing model. Frozen held-out patients never enter normalization or optimization. All their eligible beats contribute to mean patient probabilities. Training seeds [42, 123, 2026] are all retained. The run consists of **600 fits**: 3 seeds × 10 outer folds × 5 models × (3 inner fits + 1 refit). Larger models cost more compute despite equal presentation and epoch limits; these controls do not guarantee equally optimal hyperparameters for every architecture.

Completed inner and outer fits are checkpointed for resumption. Fingerprints refuse incompatible data/settings/runtime/code changes. The final report rechecks every fit's membership, initialization, stopping, selected epochs, common normalization/sampling, saved prediction hashes and pooled patient metrics before publishing results.

## Run and inspect

```bash
.venv/bin/python -m mi_localization.paired_bank
.venv/bin/python -m mi_localization.paired_study
.venv/bin/python -m mi_localization.paired_report
# Read-only saved progress; a process check separately establishes liveness:
.venv/bin/python -m mi_localization.paired_study --status
tail -f tmp/paired_ecg_vcg_run.log
# Combined runner with validated aggregate-only publication:
caffeinate -i /bin/zsh scripts/run_paired_ecg_vcg.sh --publish > tmp/paired_ecg_vcg_run.log 2>&1
```

Configuration: `configs/paired_ecg_vcg.yaml`. New ECG cache: `data/processed/paired_ecg_v1/`. Public bank report: `results/neural_paired_ecg_v1/bank_audit/REPORT.md`. Study outputs: `results/neural_paired_ecg_v1/complete/`. Partial-fold runs live in separate directories and cannot be represented as complete CV. Waveforms, fitted models, raw inventories, record flags and patient prediction CSVs stay local and ignored.

## What success would mean

Primary endpoint: patient macro F1 across all three target seeds, with sample SD, class F1, confusion tables and all four paired contrasts. The same class-stratified patient bootstrap draws are used across all arms/seeds. Intervals are descriptive and conditional on fitted predictions, without refits or corrections for correlated folds, repeated development or multiple contrasts. Repeated seed/patient predictions are never independent patients.

A favorable result would motivate a fixed new-cohort validation and a more focused ablation. It would not establish SOTA, clinical utility or untouched confirmation. A higher score from an ECG extension cannot be labeled VCG-only improvement. The 0.80–0.90 macro F1 target remains aspirational; no score is promised. Patient labels will not be merged or changed and difficult patients will not be excluded to meet it. Seeded MPS is not guaranteed bitwise deterministic.

## Primary research sources

- Kim et al. (ICLR 2017), [Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325): factorized interactions motivate the tensor branch; its VQA results do not transfer to MI localization.
- Liu et al. (ACL 2018), [Efficient Low-rank Multimodal Fusion With Modality-Specific Factors](https://aclanthology.org/P18-1209/): broader motivation for efficient multimodal interactions, not a reproduced implementation here.
- [PTB Diagnostic dataset documentation](https://physionet.org/content/ptbdb/1.0.0/): simultaneous measured ECG and Frank channels.
- Li et al. (2024), [multi-branch MI detection/localization](https://pubmed.ncbi.nlm.nih.gov/38599223/): an example of domain-informed inter-patient ECG localization. Its reported 67.17% PTB localization accuracy is not our six-class macro F1 and is not a directly comparable target.
