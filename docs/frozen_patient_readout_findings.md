# Frozen patient readout results

Completed 19 November 2026 at 10:29:54 IST. Implementation `c40fbaf7`, fixed protocol `b1608ba7`. [Protocol](frozen_patient_readout_plan.md) · [Validated report](../results/frozen_patient_readout_v1/complete/REPORT.md) · [Aggregate metrics](../results/frozen_patient_readout_v1/complete/summary.json) · [Completion custody check](../results/frozen_patient_readout_v1/complete/completion_check.json).

## Result

The 256-feature pre-compression ridge did not demonstrate benefit over either the 64-feature ridge or the original probability average. The 64-feature ridge has a modest higher point score than the original model on the declared secondary comparison, with an interval including zero. Retain the original supervised reference and all four declared procedures; do not promote an outer-result-selected winner as confirmed improvement.

| Readout | Primary: 145 dated earliest-record patients | Mean member macro F1 | Sensitivity: 164 selected-record patients |
|---|---:|---:|---:|
| Original mean beat probabilities | .521335 | .507316 | .532437 |
| Original classifier, mean logits | .517351 | .500158 | .529068 |
| Ridge on 64 final features | .535691 | .523619 | .546530 |
| Ridge on 256 pre-compression features | .501020 | .508611 | .525410 |

All ensembles use equal probabilities from all three seeds, 42/123/2026. Mean member F1 is a different statistic. The report retains every seed and class metric.

| Prespecified primary-cohort contrast | Ensemble macro F1 difference | Descriptive paired 95% interval |
|---|---:|---|
| Ridge256 minus ridge64 | −.034671 | [−.094647,+.021065] |
| Ridge256 minus original | −.020315 | [−.081297,+.037948] |
| Mean logits minus original | −.003984 | [−.034376,+.020518] |
| Ridge64 minus original, secondary | +.014356 | [−.046053,+.080129] |

These intervals condition on fixed predictions from repeatedly studied PTB development patients. None excludes zero. All three ridge64 member scores exceed their matched original member scores, but two gains are very small. The experiment establishes no independent superiority, SOTA or expected .80–.90 performance.

## What changed across classes

Under ridge64, anterior MI F1 rises from .3704 to .4286 and inferolateral MI from .4500 to .5000. Anteroseptal rises from .4727 to .4815, inferior is unchanged, anterolateral falls from .4615 to .4444 and HC from .8471 to .8333. More columns did not solve the difficult anterior groups: ridge256 AMI F1 is .2308. These are descriptive fixed-procedure outcomes, not a basis for class routing or post-hoc feature selection.

The results give no support for the proposed benefit of bypassing the 256-to-64 compression under these two ridge heads and training policies. They do not prove compression always helps or that the larger representation contains no relevant information. The patient head's small secondary point gain is compatible with the representation being useful under a different readout, and also with sampling variation. More confident diagnosis of data/label limitations still needs clinical review and independent patients.

## Scope and verification

Exact epoch-eight encoders came from the completed earliest-record duration study, separately for each inner/outer training partition. All selected-record retained beats supplied frozen features. Patient feature normalization and alpha selection stayed inside the corresponding training partition. All ten outer folds, three inner folds and all three seeds were retained. No encoder weights were updated.

Independent verification reconstructed all 120 feature contexts, replayed source query beat probabilities and restored all 420 ridge classifiers, including 60 outer heads, then reproduced the original baseline metrics exactly. The post-completion check matched all 420 classifier receipt hashes, 840 model/probability files, 120 feature receipts, 120 protected source audits and pinned numerical code/protocol/fold hashes. This final custody check was read-only apart from its new aggregate receipt; it did not repeat waveform encoding or fitting. All 371 implementation tests and the inner-only four-head fitting/restoration smoke passed before full fitting.

The user-requested verification restart ran as a one-shot background job. It exited with code 0 after the full replay and report. No worker or further experiment is running from this study. Preserve its numerical module, configuration, protocol, source references and completed outputs.

The 164-patient sensitivity still uses one selected record per patient. Neither table endpoint replaces the historical all-record .603374 convolution / .601801 waveform results. The proposed shared-covariance distance head and bounded decision-rule studies remain separate future decisions; none was started during this completion review. [Research shortlist](next_readout_experiments.md).
