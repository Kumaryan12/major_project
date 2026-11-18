# Earliest-recording sensitivity for the frozen measured-XYZ ResNet

Replayed 30 held-out prediction blocks. The 164 patients, 391 recordings and 55,863 beats retain their original labels and folds. All 385 usable dates come from PTB ECG headers. Zero new fits.

| Date eligibility | All patients | Patients with multiple recordings |
|---|---:|---:|
| earliest_date_tie | 13 | 13 |
| eligible | 145 | 79 |
| missing_or_invalid_date | 6 | 1 |

| Cohort and inference | Patient macro F1 | Accuracy | Member F1 in seed order |
|---|---:|---:|---|
| dated_unique_earliest (145) / all_beats | 0.586275 | 0.634483 | 0.591941, 0.540247, 0.473496 |
| dated_unique_earliest (145) / one_record | 0.515049 | 0.572414 | 0.533748, 0.493014, 0.444571 |
| dated_multiple_records (79) / all_beats | 0.564307 | 0.582278 | 0.559798, 0.541743, 0.478352 |
| dated_multiple_records (79) / one_record | 0.458477 | 0.468354 | 0.470474, 0.477534, 0.447743 |
| all_patients (164) / all_beats | 0.601801 | 0.664634 | 0.593102, 0.542634, 0.488658 |
| all_patients (164) / equal_records | 0.601801 | 0.664634 | 0.592545, 0.535003, 0.494981 |
| all_patients (164) / one_record | 0.534194 | 0.603659 | 0.539421, 0.498591, 0.461688 |

| Paired contrast versus all beats | Macro F1 difference | 95% interval | Top label changed |
|---|---:|---|---:|
| dated_unique_earliest / one_record | -0.071226 | [-0.146981, +0.003561] | 25 |
| dated_multiple_records / one_record | -0.105830 | [-0.223805, +0.007884] | 25 |
| all_patients / equal_records | +0.000000 | [+0.000000, +0.000000] | 0 |
| all_patients / one_record | -0.067607 | [-0.136307, +0.001920] | 27 |

## Primary cohort class performance

| Class | Eligible / original support | All beats P / R / F1 | Earliest record P / R / F1 |
|---|---:|---|---|
| AMI | 15 / 17 | 0.533 / 0.533 / 0.533 | 0.400 / 0.267 / 0.320 |
| ALMI | 14 / 16 | 0.455 / 0.357 / 0.400 | 0.350 / 0.500 / 0.412 |
| ASMI | 25 / 27 | 0.615 / 0.640 / 0.627 | 0.462 / 0.480 / 0.471 |
| IMI | 29 / 29 | 0.556 / 0.517 / 0.536 | 0.536 / 0.517 / 0.526 |
| ILMI | 22 / 23 | 0.565 / 0.591 / 0.578 | 0.524 / 0.500 / 0.512 |
| HC | 40 / 52 | 0.814 / 0.875 / 0.843 | 0.850 / 0.850 / 0.850 |

The primary contrast is one earliest dated record minus all beats on the same eligible patients. The multiple-record subset is descriptive. The all-patient one-record policy uses a deterministic fallback when dates are missing or tied. Equal-record inference gives each available record the same weight.

The dates describe available archival recordings. Clinical presentation time and the effects of elapsed time, treatment and recording quality remain unresolved. Intervals condition on the saved predictions and observed class counts from repeatedly inspected PTB development data.

[Registered protocol](../../docs/index_recording_audit_plan.md)
