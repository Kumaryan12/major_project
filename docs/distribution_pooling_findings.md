# Completed beat-distribution screen

Reviewed17November2026. Implementation `4be6eef`, independently published results `ff0712a`: [report](../results/distribution_pooling_v1/complete/REPORT.md), [metrics](../results/distribution_pooling_v1/complete/summary.json), [prespecified protocol](distribution_pooling_plan.md).

All **840classifiers /120outer models /120training-feature contexts** were restored/reconstructed. All55,863original beat features and391median-waveform features were independently re-extracted. All120matched four-arm context checks passed. Original164patients,391records,labels,R positions and frozen folds remain. The runner independently verified, published and stopped. This is a completed result, not an active queue.

| Procedure | Mean member F1 ± projection-seed SD | Equal-three-seed ensemble F1 |
|---|---:|---:|
| Median waveform features | .413881 ± .015183 | .434441 |
| Mean beat embedding | .454664 ± .011233 | .463013 |
| Mean + spread | .472406 ± .031658 | .475709 |
| Empirical quantiles | .462552 ± .035313 | .487114 |

Primary quantile−moment ensemble difference **+.011405 [−.052573,+.076160]**. Mean member difference is **−.009854 [−.041463,+.021645]**. Both conditional descriptive intervals include zero. A favorable ensemble point does not establish a single-model improvement.

Quantile−mean ensemble **+.024100 [−.022854,+.074146]** and quantile−median **+.052672 [−.027981,+.135237]** are also uncertain. Thus this screen did not demonstrate that distribution shape helps beyond simple moments, or establish that retaining variation improves patient localization generally.

Quantile AMI class F1 is higher than the matched moment control (.358974→.473684), while ASMI (.372093→.350000), IMI (.408163→.377358), ILMI (.489796→.470588) and HC (.846847→.839286) decline. These are descriptive development outcomes, not grounds for class-based routing or label changes. All class results and seeds remain in the report.

The .487114 point remains below the separate historical neural ensemble .584570 and full-convolution ridge .584266, and below completed certain-source transfer .573361. Those are different procedures, not this screen's matched controls. The current extractor uses fixed short DCT/physical features and16random projections per window. Its lower performance cannot establish a ceiling for distribution pooling, all-beat random convolutions, learned encoders or this dataset.

**Decision:** retain the completed four-arm result and strongest historical references. No new highest score, demonstrated overall improvement, independent confirmation or SOTA. A possible separately authorized follow-up would retain the exact pooling controls while using a stronger feature extractor; every learned extractor must respect exact inner/outer memberships. No such follow-up, feature-budget change, source encoder selection or old paused sweep was launched. More real independent patients remain necessary for strong confirmation.
