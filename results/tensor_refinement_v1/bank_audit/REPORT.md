# Signed tensor compression refinement

Label-free compression diagnostics; reconstruction is not diagnostic accuracy; native bands have different coefficient sampling resolutions.

All 164 patients, 391 records and 55,863 beats retained. Original global DCT and record aggregation reproduce exactly.

Global DCT ranks increase waveform resolution; denoised-only controls match the all-view ranks. Native-wavelet cores first compute A4/D4/D3/D2/D1 coefficients, then use a fixed signed DCT time basis within each band's native coefficient grid. This is a fixed partial time projection, not learned Tucker/HOOI or a shared clinical phase grid.

## Fidelity by physical view

| Method | View | Mean record relative squared error | Central-window RMS error, mV |
|---|---|---:|---:|
| global16 | raw | 0.1551 | 0.14181 |
| global16 | denoised | 0.1527 | 0.14035 |
| global16 | record_median | 0.2233 | 0.13316 |
| global16 | beat_residual | 0.0891 | 0.08099 |
| global16 | filter_residual | 0.8846 | 0.00543 |
| global32 | raw | 0.0261 | 0.05314 |
| global32 | denoised | 0.0242 | 0.05173 |
| global32 | record_median | 0.0376 | 0.04907 |
| global32 | beat_residual | 0.0296 | 0.03413 |
| global32 | filter_residual | 0.7884 | 0.00511 |
| global64 | raw | 0.0037 | 0.01767 |
| global64 | denoised | 0.0025 | 0.01573 |
| global64 | record_median | 0.0038 | 0.01363 |
| global64 | beat_residual | 0.0093 | 0.01322 |
| global64 | filter_residual | 0.6508 | 0.00470 |
| wavelet16 | raw | 0.1843 | 0.15670 |
| wavelet16 | denoised | 0.1821 | 0.15529 |
| wavelet16 | record_median | 0.2680 | 0.14786 |
| wavelet16 | beat_residual | 0.0969 | 0.08710 |
| wavelet16 | filter_residual | 0.8127 | 0.00521 |
| wavelet32 | raw | 0.0396 | 0.06801 |
| wavelet32 | denoised | 0.0383 | 0.06699 |
| wavelet32 | record_median | 0.0571 | 0.06350 |
| wavelet32 | beat_residual | 0.0327 | 0.04201 |
| wavelet32 | filter_residual | 0.6031 | 0.00449 |

Relative squared errors exclude zero-energy lead-beats, then average records equally. Absolute errors retain mV units. Native symmetric wavelet coefficients have boundary redundancy; no variance partition or clean-signal ground truth is claimed. All feature summaries are lossy and fitted selection stays in training partitions. No patient exclusions, R redetection, alignment or relabeling.
