# Longer denoised VCG context: cache audit

164 patients, 391 records, 55,863 original beats retained.

Original: 651 samples (−250 to +400 ms). Longer: 1001 samples (−350 to +650 ms), both at 1000 Hz.

Every original short crop was regenerated exactly, and every embedded central crop is bitwise equal to the retained float32 cache. Both use the original full-record wavelet denoising. No new R detection or beat-wise denoising is used.

Reflection padding affects 150 beats (0.269%) across 95 patients: 42 left boundaries and 108 right boundaries. Maximum missing samples: left 96, right 247. Padding is applied after full-record denoising; no existing samples are replaced.

| Class | Beats | Padded beats |
|---|---:|---:|
| AMI | 6333 | 17 |
| ALMI | 6827 | 21 |
| ASMI | 11489 | 35 |
| IMI | 12410 | 31 |
| ILMI | 8049 | 19 |
| HC | 10755 | 27 |

No redetection, label change, beat exclusion, time resizing or independent physiological coverage validation. Reflection padding is synthetic context only at recording boundaries.

Longer windows may contain neighboring beats; they are not adjudicated P/QRS/ST/T segments or guaranteed full cardiac cycles. Existing peak screening flags do not establish physiological coverage. Classification benefit requires the matched patient-separated study.

Waveforms, per-beat padding and patient identities remain local. Only aggregate cache diagnostics are published.
