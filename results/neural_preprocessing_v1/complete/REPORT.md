# Raw versus denoised VCG: matched waveform experiment

164 unchanged patients, 55,863 identical beat positions, 6 classes, 10 patient folds and seeds [42, 123, 2026].

| Training seed | Denoised macro F1 | Raw macro F1 | Raw − denoised |
|---|---:|---:|---:|
| 42 | 0.586 | 0.533 | -0.053 |
| 123 | 0.557 | 0.517 | -0.039 |
| 2026 | 0.507 | 0.512 | +0.004 |

## Across all prespecified seeds

- denoised: **0.550 ± 0.040** (mean ± sample SD across training seeds).
- raw: **0.521 ± 0.011** (mean ± sample SD across training seeds).
- Mean paired raw − denoised difference: **-0.029**, descriptive patient-bootstrap interval [-0.078, +0.019].

5,000 class-stratified patient draws are shared across both arms and all fixed seeds. Seed/patient repetitions are not independent cases. No model refits or seed resampling; no correction for correlated training folds, repeated development or multiple comparisons. These are development results, not independent confirmation or a clinical/SOTA claim.

## Matched protocol

The raw input is unfiltered measured XYZ cropped at the original cached R positions. The denoised input is the existing full-record wavelet-denoised cache, independently reproduced exactly before raw cache construction. No new R detection, beat exclusion, label change, longer crop, ECG input or statistical feature fusion is included.

Both arms use the same compact ResNet, initial-weight hashes, class/patient-balanced weighted-random beat sampler and shared-epoch row sequences. Normalization is fit separately using only each training partition's signal; raw and denoised moments must NOT be forced to be equal. Waveform amplitudes are not normalized beat by beat.

Both have 8,192 beat presentations/epoch, batch size 128, up to 25 epochs and patience 6. Architecture/optimizer are fixed. 3 inner patient folds select the rounded-median epoch budget for a fresh outer-training refit; outer test patients never enter scaling or stopping. Actual epochs may differ between arms. Every outer-test beat is evaluated and probabilities are averaged per patient.

The denoised control is freshly refit; the historical ResNet's 0.550 mean is not silently reused. Runtime/source/bank/fold fingerprints and private checkpoints permit compatible resumes. Seeded MPS is not guaranteed bitwise deterministic.

## Mean class-level F1 across seeds

| Class | Patients | Denoised | Raw |
|---|---:|---:|---:|
| AMI | 17 | 0.373 | 0.353 |
| ALMI | 16 | 0.436 | 0.313 |
| ASMI | 27 | 0.530 | 0.493 |
| IMI | 29 | 0.548 | 0.520 |
| ILMI | 23 | 0.576 | 0.595 |
| HC | 52 | 0.837 | 0.852 |

## Interpretation boundaries

The accompanying signal audit verifies consistency and quantifies preprocessing changes; it does not validate R-peak accuracy or physiological coverage. No independent peak/delineation annotations or expert adjudication were supplied. Screening flags did not remove patients or beats. A lower signal-difference metric alone is not proof that denoising preserves all diagnostic information.

This experiment evaluates this fixed processing/model combination. It neither establishes a PTB performance ceiling nor exhausts preprocessing choices. More architectures/longer windows should follow specific evidence; a 0.80–0.90 macro F1 is not promised. Independent validation remains necessary.

Run `.venv/bin/python -m mi_localization.signal_audit`, `.venv/bin/python -m mi_localization.preprocessing_study`, then `.venv/bin/python -m mi_localization.preprocessing_report`. Read progress with `.venv/bin/python -m mi_localization.preprocessing_study --status`.

Waveform arrays, patient predictions, fitted weights, raw review plots and record-level diagnostics remain local. Only aggregate metrics/audits are published. Existing results and PTB-XL tests are untouched.
