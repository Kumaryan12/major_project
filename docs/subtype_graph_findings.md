# Subtype graph study findings

The graph that links related MI subtypes scored **0.591424** patient macro F1 in the three-seed ensemble. The matched convolution ridge reproduced **0.584266**, and the relabeled graph reached **0.582026**. The subtype graph changed five patient classifications versus ridge. [Protocol](subtype_graph_plan.md) · [Validated report](../results/subtype_graph_v1/complete/REPORT.md) · [Aggregate metrics](../results/subtype_graph_v1/complete/summary.json)

| Comparison | Ensemble difference | Paired patient 95% interval |
|---|---:|---:|
| Subtype graph minus ridge | +0.007158 | [−0.012932, +0.028282] |
| Subtype graph minus relabeled graph | +0.009398 | [−0.001798, +0.026507] |
| Relabeled graph minus ridge | −0.002241 | [−0.017136, +0.010498] |

Both intervals for the subtype graph include zero. Mean member macro F1 was 0.575819 for subtype sharing, 0.572860 for ridge and 0.576249 for the relabeled graph. The different rankings reflect the difference between averaging predictions and averaging member F1. Inner validation enabled sharing in 17 of 30 fold/seed decisions for the subtype graph and 14 for the relabeled graph.

Anterior MI F1 stayed at 0.344828, while anterolateral MI declined from 0.482759 to 0.466667. Inferolateral MI improved from 0.577778 to 0.622222 and healthy controls from 0.924528 to 0.944444. The observed change falls short of the proposed improvement in the harder anterior subtypes. Keep the ordinary ridge as the matched reference and retain this result in the backlog. The two graphs test a particular weight penalty on PTB development patients.

Independent replay checked all 120 feature contexts, restored 210 original ridge references, reproduced 360 positive inner graph candidates and 31 selected positive outer fits, and recovered baseline metrics exactly. A separate check verified all 120 context and array hashes. All ten outer folds and three seeds were included. The study used existing cached waveform features. The real inner-only smoke passed, and the full macOS test suite passed 330 tests. [Research backlog](RESEARCH_BACKLOG.md)
