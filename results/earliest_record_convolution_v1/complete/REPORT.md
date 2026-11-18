# Convolution features on earliest-record measured XYZ

Verified 420 ridge fits and 120 regenerated feature contexts.

| Population | Arm | Ensemble macro F1 | Member macro F1 in seed order |
|---|---|---:|---|
| dated (145) | resnet_8 | 0.521335 | 0.499865, 0.522067, 0.500015 |
| dated (145) | dct | 0.461037 | 0.461037, 0.461037, 0.461037 |
| dated (145) | combined | 0.478122 | 0.487650, 0.477224, 0.405626 |
| all_patients (164) | resnet_8 | 0.532437 | 0.511795, 0.543314, 0.514097 |
| all_patients (164) | dct | 0.459673 | 0.459673, 0.459673, 0.459673 |
| all_patients (164) | combined | 0.496999 | 0.494216, 0.493722, 0.425826 |

| Population | Contrast | Macro F1 difference | Descriptive 95% interval |
|---|---|---:|---|
| dated | combined_minus_resnet_8 | -0.043213 | [-0.150252, +0.064770] |
| dated | combined_minus_dct | +0.017085 | [-0.091500, +0.124470] |
| dated | dct_minus_resnet_8 | -0.060298 | [-0.158879, +0.043186] |
| all_patients | combined_minus_resnet_8 | -0.035439 | [-0.136374, +0.067728] |
| all_patients | combined_minus_dct | +0.037326 | [-0.068857, +0.141046] |
| all_patients | dct_minus_resnet_8 | -0.072765 | [-0.168794, +0.028147] |

| Class on primary dated cohort | Patients | ResNet F1 | DCT F1 | Combined F1 |
|---|---:|---:|---:|---:|
| AMI | 15 | 0.3704 | 0.2222 | 0.2400 |
| ALMI | 14 | 0.4615 | 0.3333 | 0.3478 |
| ASMI | 25 | 0.4727 | 0.3478 | 0.4615 |
| IMI | 29 | 0.5263 | 0.5098 | 0.4839 |
| ILMI | 22 | 0.4500 | 0.5714 | 0.4545 |
| HC | 40 | 0.8471 | 0.7816 | 0.8810 |

The primary contrast is combined convolution versus fixed eight-epoch ResNet on the same 145 earliest-record patients. The DCT contrast controls the ridge readout and record template. No new waveforms enter these results. The 164-patient sensitivity includes deterministic fallbacks. Intervals condition on the saved predictions in repeatedly studied PTB.

[Protocol](../../../docs/earliest_record_convolution_plan.md)
