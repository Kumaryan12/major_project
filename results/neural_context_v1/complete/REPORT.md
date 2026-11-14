# Short versus longer-context denoised VCG

164 unchanged patients, 55,863 original beat positions, 6 classes, 10 patient folds, seeds [42, 123, 2026].

| Training seed | Short macro F1 | Long macro F1 | Long − short |
|---|---:|---:|---:|
| 42 | 0.586 | 0.536 | -0.051 |
| 123 | 0.557 | 0.507 | -0.050 |
| 2026 | 0.507 | 0.572 | +0.064 |

## Across all prespecified seeds

- short: **0.550 ± 0.040** (mean ± sample SD).
- long: **0.538 ± 0.033** (mean ± sample SD).
- Mean paired long − short difference: **-0.012**, descriptive patient-bootstrap interval [-0.064, +0.038].

5,000 class-stratified patient draws are shared across both arms and all fixed seeds. Repeated seed/patient predictions are not independent patients. No refits or seed resampling; no correction for correlated training folds, repeated development or multiple comparisons. Development only; no independent confirmation or SOTA claim.

## Matched protocol

Both inputs are measured Frank XYZ denoised over the full recording with the existing bior6.8 pipeline. Short crops use the original window; long crops span −350 to +650 ms, 1001 samples at 1000 Hz. All original central samples were verified bitwise equal. There is no new peak detection, time resizing, excluded beat/patient, ECG input, augmentation or statistical fusion.

150 longer crops (0.269%) across 95 patients require reflection at recording boundaries. Only the missing samples are synthetic; existing samples are unchanged. Padding metadata stays local and aggregate counts are in the cache audit. Reflected samples are not observed physiology.

Both arms are freshly trained with the same ResNet parameter count and initial weights. Shared-epoch training rows match. Adaptive temporal pooling supports both lengths without time resampling. The longer arm processes more samples and changes the relative position of the central beat in the pooling bins; this is a window comparison, not an equal-FLOP experiment or isolation of a particular cardiac phase.

Class/patient-weighted normalization is fit separately inside each training partition, over that arm's entire window (including reflected samples). There is no per-beat amplitude normalization. Different window moments are allowed and must not be pooled across arms or fit on held-out patients.

Each epoch presents 8,192 beats, batch size 128. Fixed AdamW settings, up to 25 epochs and patience 6. 3 frozen inner patient partitions select the rounded-median epoch count; fresh outer-training refits never monitor outer-test scores. Actual stopping epochs may differ. Every test beat is evaluated; mean probabilities produce one prediction per patient.

The short control is refit within this study, rather than substituting a historical score. Source/runtime/fold/cache hashes guard compatible resumes; MPS is not guaranteed bitwise deterministic. The old raw-versus-denoised experiment remains unchanged.

## Mean class-level F1 across seeds

| Class | Patients | Short | Long |
|---|---:|---:|---:|
| AMI | 17 | 0.373 | 0.360 |
| ALMI | 16 | 0.436 | 0.405 |
| ASMI | 27 | 0.530 | 0.548 |
| IMI | 29 | 0.548 | 0.534 |
| ILMI | 23 | 0.576 | 0.531 |
| HC | 52 | 0.837 | 0.850 |

## Interpretation

Longer windows may include neighboring beats. Neither window has independently adjudicated R positions or P/QRS/ST/T coverage; this experiment cannot attribute any gain to a specific phase. The same repeatedly inspected PTB cohort is development data. This does not establish a dataset ceiling, novelty, clinical validity or a promised 0.80–0.90 macro F1.

Only aggregate metrics and audits are published. Patient predictions, waveforms, padding arrays and fitted weights stay local. Existing PTB-XL tests are not rescored.
