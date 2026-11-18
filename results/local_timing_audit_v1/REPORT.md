# Local timing feasibility in measured XYZ

All 164 patients, 391 recordings and 55,863 original beats. Two disjoint same-record reference templates per query group, with all beats in the denominators.

| R-relative window (ms) | Repeatable beats | Supported local beats | Patients above support threshold | Control acceptance | Correct among accepted | Gate |
|---|---:|---:|---:|---:|---:|---|
| [-50, 70) | 47,840 | 0 | 0/164 | 100.0% | 100.0% | Reference |
| [90, 180) | 9,638 | 2,489 | 20/164 | 81.8% | 100.0% | Below gate |
| [180, 270) | 35,458 | 1,066 | 8/164 | 95.1% | 100.0% | Below gate |
| [270, 360) | 30,427 | 2,142 | 16/164 | 96.7% | 100.0% | Below gate |

| R-relative window (ms) | Zero-delay gain/noise cases with apparent shift | Zero-delay shape-change cases with apparent shift |
|---|---:|---:|
| [-50, 70) | 0/391 | 0/391 |
| [90, 180) | 0/391 | 37/391 |
| [180, 270) | 0/391 | 71/391 |
| [270, 360) | 0/391 | 42/391 |

Positive lag means the observed signal occurs later. Supported local shifts require agreement between disjoint references, departure from the near-R anchor and a cosine improvement over the near-R-only shift against the other reference in both directions.

Signal morphology and local timing can change the same score. The synthetic controls measure recovery of known shifts and sensitivity to one fixed shape deformation. Disjoint templates within a recording can still be correlated. The windows are fixed relative to the cached detector index and have no clinical annotations. This label-blind feasibility report gives no localization score.

See the [protocol](../../docs/local_timing_audit_plan.md) for the exact thresholds and denominators.
