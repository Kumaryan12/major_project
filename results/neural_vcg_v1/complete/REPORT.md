# VCG neural development results

Complete pooled out-of-fold patient predictions: 164 patients, 55,863 beats, 6 classes, 10 frozen patient folds.

**Development only:** these PTB folds were inspected during earlier representation studies. This is not untouched confirmation, a clinical validation or an established SOTA result.

| Model | Patient macro F1 | Accuracy | Δ F1 vs fixed control | Paired 95% interval for Δ |
|---|---:|---:|---:|---:|
| Short-wavelet statistics + fixed XGBoost | 0.469 | 52.4% | +0.000 | — |
| Compact XYZ ResNet | 0.586 | 63.4% | +0.118 | [+0.022, +0.211] |
| Compact XYZ ResNet + short-wavelet statistics | 0.584 | 65.2% | +0.115 | [+0.023, +0.206] |

Descriptive uncertainty: 5,000 paired, class-stratified patient bootstrap draws conditional on fitted predictions. No model refits; intervals do not adjust for repeated development, correlated cross-validation training folds or multiple comparisons.

## Protocol

- Measured Frank XYZ only, same denoised 651-sample beats and R positions. No ECG input or pretraining.
- Small residual encoder (three stages, widths 16/32/64), GroupNorm, coarse temporal pooling, 64-dimensional waveform embedding. Fusion adds a 32-dimensional embedding of the existing 192 short-wavelet statistics.
- Architecture and optimizer were fixed before the run; this is not a large hyperparameter search.
- Up to 25 epochs with patience 6. Early stopping uses patient macro F1 on 3 inner patient folds only. The rounded median best epoch sets the outer-training refit length. Outer test patients are never used for stopping or normalization.
- Each training epoch draws 8,192 beats with replacement, balancing classes and patients within class. Unweighted cross entropy avoids applying the balancing twice. Validation/test predictions use every eligible beat.
- Waveform scaling uses weighted training-set XYZ means/standard deviations, not per-beat amplitude normalization. Statistics are scaled using training-only moments and clipped to ±10 standardized units.
- XGBoost trains on all eligible training beats; neural models use the declared sampling budget. Thus this is a practical model comparison, not an identical optimization budget.
- Mean beat probabilities yield one prediction per patient. The historical nested depth-3/6 score of 0.487 is a different protocol from this fixed depth-3 control.
- Full-record wavelet denoising is offline/noncausal, not a streaming predictor.

## Class-level F1

| Class | Patients | Short-wavelet statistics + fixed XGBoost | Compact XYZ ResNet | Compact XYZ ResNet + short-wavelet statistics |
|---|---:|---:|---:|---:|
| AMI | 17 | 0.357 | 0.438 | 0.385 |
| ALMI | 16 | 0.471 | 0.533 | 0.457 |
| ASMI | 27 | 0.316 | 0.582 | 0.632 |
| IMI | 29 | 0.418 | 0.571 | 0.491 |
| ILMI | 23 | 0.439 | 0.578 | 0.667 |
| HC | 52 | 0.812 | 0.816 | 0.870 |

## Limits and next steps

No arm reached 0.80 patient macro F1.

There are only 164 independent patients in the real PTB task; 55,863 beats do not constitute that many independent cases. A gain here must be confirmed on a genuinely new cohort. Do not repeatedly tune against the previously evaluated PTB-XL test split. ECG + VCG and self-supervised learning remain separate future extensions.

## Reproduce

```bash
.venv/bin/python -m pip install -e '.[neural,dev]'
.venv/bin/mi-localization neural-features
.venv/bin/mi-localization neural-study
.venv/bin/python -m mi_localization.neural_report
```

Checkpoints, fitted weights and patient predictions stay local; only aggregate audits, metrics and figures are published.

Runtime: `{'torch_version': '2.15.0', 'numpy_version': '2.5.3', 'device': 'mps', 'platform': 'macOS-26.5-arm64-arm-64bit', 'torch_threads': 2, 'deterministic_kernels': False, 'training_code_sha256': 'c55ae78b65aca055c0f8f37f73ce7d3387175cbebb311990cc1f060f20967358'}`. CPU/CUDA deterministic algorithms are requested; seeded MPS is not guaranteed bitwise deterministic. Seeds do not guarantee identical results across platforms/PyTorch releases.

Research motivation: Strodthoff et al., [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://arxiv.org/abs/2004.13701). That study motivates testing residual waveform models but does not establish their performance on this VCG MI localization task.
