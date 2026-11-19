# Frozen64 ridge and shared-covariance distance readouts

Verified 120 feature contexts, 420 new distance heads and 210 source ridge heads; 104 identity checks.
Neural encoder updates: zero. All source artifacts retained unchanged.

| Cohort | Arm | Ensemble macro F1 | Member F1 in seed order |
|---|---|---:|---|
| dated (145) | ridge | 0.535691 | 0.501370, 0.568980, 0.500507 |
| dated (145) | centroid | 0.507770 | 0.463396, 0.486684, 0.461864 |
| dated (145) | mahalanobis | 0.507181 | 0.463396, 0.490909, 0.489767 |
| all_patients (164) | ridge | 0.546530 | 0.507333, 0.588467, 0.513662 |
| all_patients (164) | centroid | 0.520508 | 0.465336, 0.501971, 0.477639 |
| all_patients (164) | mahalanobis | 0.519400 | 0.465336, 0.505690, 0.502845 |

| Cohort | Contrast | Difference | Descriptive paired 95% interval |
|---|---|---:|---|
| dated | mahalanobis_minus_ridge | -0.028510 | [-0.088700,+0.031304] |
| dated | mahalanobis_minus_centroid | -0.000589 | [-0.041155,+0.038686] |
| dated | centroid_minus_ridge | -0.027921 | [-0.100774,+0.043663] |
| all_patients | mahalanobis_minus_ridge | -0.027129 | [-0.083811,+0.026831] |
| all_patients | mahalanobis_minus_centroid | -0.001107 | [-0.039825,+0.035408] |
| all_patients | centroid_minus_ridge | -0.026022 | [-0.093167,+0.039304] |

| Primary cohort class | Patients | Ridge | Centroid | Mahalanobis |
|---|---:|---:|---:|---:|
| AMI | 15 | 0.4286 | 0.3704 | 0.2963 |
| ALMI | 14 | 0.4444 | 0.4444 | 0.4444 |
| ASMI | 25 | 0.4815 | 0.4528 | 0.4528 |
| IMI | 29 | 0.5263 | 0.5246 | 0.5424 |
| ILMI | 22 | 0.5000 | 0.4211 | 0.4500 |
| HC | 40 | 0.8333 | 0.8333 | 0.8571 |

Primary: Mahalanobis minus ridge. Covariance control: Mahalanobis minus centroid. All three seeds are equally weighted. Mahalanobis has three inner shrinkage candidates; centroid has one. Intervals condition on fixed predictions from repeatedly studied development patients. The 145-patient endpoint uses earliest dated recordings; the 164 sensitivity retains one selected record each. Historical all-record .60 results are separate.

[Protocol](../../../docs/frozen_distance_readout_plan.md)
