# Direct beat morphology with gated temporal context

Implemented **16 November 2026**, following the completed [chronology diagnostic](chronological_diagnostics_findings.md). This is a new development experiment, not a result or a SOTA claim. Code/config/result namespaces are separate from the completed chronology/signature studies.

Validation before full launch: **217 tests passed** on macOS, including eight residual-context tests and actual tiny nested fit/restore/resume/tamper checks. The real-data **MPS GPU smoke passed for all six arms**, each with 88,693 registered parameters, matching initial states and restored predictions. Smoke outputs are implementation checks, not diagnostic performance. Windows has not been validated.

## Hypothesis and fixed representation

The preceding sequence recipe replaced direct beat classification with a 32-dimensional context head and did not improve the matched shuffled control. Retaining a direct morphology path and explicit beat supervision may make it easier to learn useful context without making every prediction depend on that bottleneck. This is a testable hypothesis, not an established cause of the earlier result.

Use the same measured, signed XYZ input and 64-dimensional compact ResNet beat encoder. A direct dropout/linear head maps each embedding to six class logits. In parallel, embedding plus embedded preceding RR/availability enters a masked standard GRU with hidden size 32. Its final state produces a contextual six-logit correction. A scalar sigmoid gate, conditioned on mean beat embedding and context state, multiplies that correction. Add it to each beat's direct logits, apply softmax per beat, then average real-beat probabilities within the window.

The gate starts at sigmoid(−3), approximately 0.0474, using zero gate weights and bias −3; its branch receives gradients immediately. Beat-only arms force the gate to zero. Retaining a path is an architectural property, not a guarantee of preserving the old model's accuracy: encoder weights are still jointly trained. Gate magnitude is not physiological importance; contextual head scaling also matters.

## Six fixed arms

| Arm | Context correction | Direct beat auxiliary loss |
|---|---|---:|
| `beat_only` | Disabled | 0 |
| `beat_aux` | Disabled | 0.5 × beat CE |
| `ordered_context` | Original consecutive context | 0 |
| `ordered_context_aux` | Original consecutive context | 0.5 × beat CE |
| `shuffled_context` | Jointly shuffled embedding/RR pairs | 0 |
| `shuffled_context_aux` | Jointly shuffled embedding/RR pairs | 0.5 × beat CE |

Every arm has the same initial state and registered parameters. Inactive context parameters in beat-only arms mean active capacity differs. All arms compute direct logits on the same ordered beats, including matched direct-dropout assignments. Only the contextual embedding/RR pairs are permuted in shuffled arms; no fictitious RR intervals are created. Both shuffled arms use the same deterministic permutation schedule. Query permutations depend on record/window identity, not unrelated query patients or labels.

All arms minimize window negative log probability. Auxiliary arms additionally minimize direct beat cross entropy with coefficient **0.5**, mean over real beats in each window then over windows. The direct auxiliary head receives no contextual correction. The coefficient, gate initialization and arms are fixed before full training; no outer-score coefficient/gate/ensemble selection.

Primary comparisons: combined recipe versus `beat_only`; contextual contribution versus `beat_aux`; chronological order versus `shuffled_context_aux`. Auxiliary effects and the no-aux order control are reported as declared ablations. Improvements against one baseline alone do not prove that chronology, auxiliary supervision or a causal physiological mechanism caused them.

## Data and evaluation

- Retain **164 patients, 391 recordings, 55,863 beats**, original six classes, cached R positions and frozen patient folds. No filtering, relabeling or added ECG inputs.
- Fix **four consecutive retained beats within one recording**. Earlier inner selection preferred 4/8/16 in 18/11/1 cases; this motivates a compact fixed-four architecture comparison, not a universally optimal length or an outer-score winner. No length search occurs in this study.
- Same class/patient-balanced sampling, then uniform record and contiguous start: **2,048 windows / 8,192 nominal beat slots per epoch**, **32 windows / 128 slots per batch**, **64 optimizer steps per epoch**. Arms share windows at common epochs. Mask incomplete tails; never cross record/patient boundaries.
- Fresh initialization per fit; no saved pretrained/fitted encoder is loaded. In particular, an outer-trained ResNet must not initialize an inner fit whose validation patients it has already seen.
- Training-only waveform and RR normalization, exactly the existing chronology normalizer. AdamW, LR 0.001, weight decay 0.01, dropout 0.2, gradient norm clip 5. Up to **25 epochs**, patience **6**, earliest best **inner patient macro F1**. Each arm's rounded median best inner epoch sets its fresh outer-refit duration; outer patients never stop training.
- Ten outer patient folds × three optimization seeds **42/123/2026** × six arms × (three inner fits + one outer refit): **720 fits**, comprising **540 inner and 180 outer**. All seeds/arms are retained.
- Offline inference tiles every query beat once, masks tails, weights window probabilities by valid-beat count within each record, then averages records equally per patient. This is common to all fresh controls. Window context and original preprocessing are not a streaming/causal claim.

The new beat-only control uses the same window objective, sampling and inference as its matched context arms. It differs from the historical per-beat-trained ResNet and is fitted anew. Historical 0.550 is context, not a substitute for this control.

## Verification and reporting

Mathematical/behavior tests cover direct-path equivalence to mean beat probabilities, masked tails, gate/encoder/beat/context gradients, joint context-only permutation, inference independence from query labels and unrelated patients, exact record aggregation, and actual tiny nested training/save/restore/resume. Source/config/runtime hashes prevent incompatible resume.

Independent reporting rebuilds training normalization, original windows and context permutations, inner stopping and outer durations; checks all six controls' matching initial states/direct samples; restores all **180 outer models** and compares their predictions. There are **120 six-arm matched partition checks**. Patient-level metrics, class confusions, MI-only F1 using unchanged six-way predictions and all predeclared contrasts are published only after verification.

Descriptive intervals use 5,000 paired class-stratified patient draws shared across arms and seeds. The vectorized implementation is checked against explicit repeated-patient F1. These intervals condition on fitted predictions; they do not correct overlapping training folds, repeated development or multiple comparisons. Seed SD is not a generalization confidence interval. The PTB cohort remains development data; no guaranteed 0.80–0.90 or SOTA claim.

## Run and monitor

```sh
# Actual-data device/gradient/checkpoint smoke, all six arms.
.venv/bin/python -m mi_localization.residual_context_study --smoke

# Run/resume all fits, independently verify, then publish aggregates.
.venv/bin/python scripts/run_residual_context.py --publish

# Read-only terminal monitor; Ctrl+C stops the monitor only.
.venv/bin/python scripts/live_status.py --watch
```

Runner log: `tmp/residual_context_v1.log`; runner metadata: `tmp/residual_context_v1_runner.json`. The launcher holds an OS file lock and reuses a matching verified device smoke. Its publication step checks a clean main branch, stages only this result namespace and retries GitHub push on failure. Publication failure is distinguished from training verification.

Native Windows uses `.venv\Scripts\python.exe`. Python entry points and lock support are provided, but Windows device/training behavior has not been validated. Raw banks, weights and individual predictions remain local and ignored. Public output: `results/neural_residual_context_v1`. Modules: `residual_context_model.py`, `residual_context_study.py`, `residual_context_report.py`; [configuration](../configs/residual_context_v1.yaml). Existing numerical modules and completed results remain unchanged.
