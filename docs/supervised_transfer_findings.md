# Completed supervised transfer: interpretation and next decision

Reviewed17November2026. Independently published result `46612ff`: [report](../results/neural_supervised_transfer_v1/complete/REPORT.md), [machine-readable results](../results/neural_supervised_transfer_v1/complete/summary.json). All630fits/90outer restorations/210matched checks passed. No live worker is needed to read this completed result.

| Procedure | Mean member macro F1 ± seed SD | Equal-three-seed ensemble F1 |
|---|---:|---:|
| Matched scratch | .553331 ± .051880 | .538736 |
| Certain-source initialization, primary | .569351 ± .035333 | .573361 |
| Intermediate-positive source, sensitivity | .514954 ± .038236 | .549364 |

Primary ensemble difference **+.034625 [−.046511,+.113800]**, descriptive paired patient interval including zero. Sensitivity difference **+.010628 [−.059197,+.078868]**, also uncertain. Do not select a source objective or seed subset from these outer outcomes. The mean-model and ensemble comparisons are separate.

The certain-source ensemble's largest class point increase is ILMI (.533333→.690909), with HC .859813→.895238. AMI declines .333333→.285714. This describes predictions, not causal attribution or permission for class-based routing. Overall point performance remains below the separate historical neural ensemble .584570 and full-convolution ridge .584266. No established new best model or demonstrated gain follows.

Learnable source diagnostic labels therefore did not establish a reliable target localization improvement under this bounded transfer design. The source used one training seed, derived XYZ and overlapping native diagnoses. The recording screen passed, but source-target patient identity and clinical equivalence remain unresolved. All original target patients and folds remain development data.

The user authorized proceeding with the independently motivated [four-arm beat-distribution screen](distribution_pooling_plan.md). It tests whether retaining morphology distributions helps beyond mean/spread, starting with fixed physical/DCT features instead of choosing one of these encoders. No acquisition bridge, synthetic data, new source objective or resumed older multiview sweep is included.
