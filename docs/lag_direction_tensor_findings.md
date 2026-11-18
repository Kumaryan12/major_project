# Within-beat lead direction tensor findings

**The signed tensor reached 0.531364 ensemble macro F1 versus 0.584266 for matched convolution ridge.** The magnitude branch reached 0.517919 and the sign-scrambled branch reached 0.546633. No added branch beat the matched baseline. [Protocol](lag_direction_tensor_plan.md) · [Verified report](../results/lag_direction_tensor_v1/complete/REPORT.md)

| Contrast | Ensemble difference | Descriptive paired patient 95% interval |
|---|---:|---:|
| Signed minus magnitude, primary | +0.013444 | [−0.027522, +0.054415] |
| Signed minus sign scrambled | −0.015269 | [−0.066919, +0.034030] |
| Signed minus baseline | −0.052903 | [−0.122082, +0.013902] |

All three individual signed members exceeded their matching magnitude members, and all three trailed their sign-scrambled and baseline counterparts. The results give no demonstrated benefit for preserving the original signs. Signed IMI class F1 rose from 0.542373 to 0.566667 versus baseline, while the other five class scores fell. The comparison concerns the declared windows, lags, pooling and fixed branch scale on development patients.

The audit and tests checked that reversing time preserves the symmetric terms and flips the antisymmetric signs. Independent replay re-encoded all 55,863 beats across 391 recordings, rebuilt the 120 matched feature contexts, restored 210 source references and replayed all 630 new classifiers. The original baseline metrics reproduced exactly. A read-only check matched the 120 context audit and data pairs. The real inner-only smoke and **24 relevant tests** passed. The detached runner completed and exited. [Aggregate metrics](../results/lag_direction_tensor_v1/complete/summary.json) · [Backlog](RESEARCH_BACKLOG.md)
