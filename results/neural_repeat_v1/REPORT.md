# VCG ResNet training-seed repeatability

Fixed outer AND inner patient partitions. All requested seeds are reported; no best-seed selection or ensemble.

| Training seed | Patient macro F1 | Accuracy |
|---|---:|---:|
| 42 | 0.586 | 63.4% |
| 123 | 0.557 | 62.2% |
| 2026 | 0.507 | 57.9% |

Macro F1 mean ± sample standard deviation: **0.550 ± 0.040**.

The original seed is reused only after checking its model protocol, frozen cohort, predictions and training seed audits. Runtime/backend/source hashes are retained per run; this is a training-seed check, not a guarantee of bitwise reproduction across MPS/PyTorch versions.

These are repeatedly studied PTB development patients. The spread across a few training seeds is not a confidence interval for generalization and does not remove model-selection bias.

## Class-level F1 across seeds

| Class | 42 | 123 | 2026 |
|---|---:|---:|---:|
| AMI | 0.438 | 0.359 | 0.323 |
| ALMI | 0.533 | 0.387 | 0.389 |
| ASMI | 0.582 | 0.609 | 0.400 |
| IMI | 0.571 | 0.581 | 0.491 |
| ILMI | 0.578 | 0.542 | 0.609 |
| HC | 0.816 | 0.863 | 0.833 |
