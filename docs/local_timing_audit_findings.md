# Idea 2 local timing audit findings

**None of the three post-R windows passed the predeclared feasibility gate.** The audit found some repeatable offsets relative to the near-R anchor. They were concentrated in too few patients to justify full tensor extraction using this estimator. The analysis used all 164 patients, 391 recordings and 55,863 stored measured XYZ beats. [Protocol](local_timing_audit_plan.md) · [Validated audit](../results/local_timing_audit_v1/REPORT.md) · [Aggregate results](../results/local_timing_audit_v1/summary.json)

| Post-R window | Beats with supported local shift | Patients with support on at least 10% of beats, equal record weighting | Injected delays accepted and recovered | Shape-only controls with apparent shift |
|---|---:|---:|---:|---:|
| 90 to 180 ms | 2,489 / 55,863 (4.5%) | 20 / 164 (12.2%) | 1,280 / 1,564 (81.8%) | 37 / 391 (9.5%) |
| 180 to 270 ms | 1,066 / 55,863 (1.9%) | 8 / 164 (4.9%) | 1,488 / 1,564 (95.1%) | 71 / 391 (18.2%) |
| 270 to 360 ms | 2,142 / 55,863 (3.8%) | 16 / 164 (9.8%) | 1,513 / 1,564 (96.7%) | 42 / 391 (10.7%) |

All accepted injected shifts in the three windows were recovered within 2 ms. For positive controls accepted at both the window and anchor, subtracting the anchor left no apparent local shift of at least 8 ms. The gain and noise control without a shift also produced no accepted apparent shifts. A local amplitude deformation produced an apparent shift in 9.5–18.2% of recording-template controls across the post-R windows. This shows that the estimator can react to a change in shape even with no change to the time grid.

The 90–180 ms window had the strongest local support. The median patient's mean cosine gain versus the near-R-only shift was 0.0117 among comparable beats there, versus 0.0024 and 0.0026 in the later windows. The support requirement used all beats as denominators and required at least 33 of 164 patients to reach the 10% threshold. The gate was descriptive. Disjoint references from the same recording can still be correlated, and the windows refer to cached detector positions. Establishing physical or clinical timing would require additional evidence.

An independent process rebuilt all 15 diagnostic arrays and matched them on all 391 recordings, including the injected and deformation controls. All 32 relevant tests passed. The audit used no classifier fits, and macro F1 remains unmeasured for this representation. The backlog records the timing tensor as deferred and lists review of the extra patients' partial labels as the next candidate. [Research backlog](RESEARCH_BACKLOG.md)
