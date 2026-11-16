# Staged morphology learning with frozen-context adaptation

Implemented **16 November 2026** after the user requested the next method following the completed late-fusion experiment. This is a controlled development hypothesis, not a demonstrated improvement or established SOTA. The historical **0.584570** residual seed ensemble stays a separate frozen reference.

## Hypothesis and research basis

Learn a useful beat representation first, then test whether a small context branch can improve it while its morphology encoder/direct classifier remain fixed. The prior jointly trained residual recipe did not demonstrate an individual-model gain, and its small initial gate often grew substantially. These observations do not prove feature drift caused the failures; frozen-versus-joint controls test that hypothesis.

The general adaptation principle is related to [Zhang et al., Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks, ECCV 2020](https://arxiv.org/abs/1912.13503). That paper studies other tasks/domains and does not establish measured-VCG MI performance. Our same-task, partition-local morphology pretraining and temporal correction are an adaptation of the principle, not a reproduction or an originality claim. [Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054) provides related evidence about feature changes during fine-tuning; its setting/theory does not prove the mechanism on this cohort.

## Shared first phase

For each outer-training/inner-training partition and training seed, fit a fresh `beat_only` model using the established measured-XYZ ResNet and window-probability objective. The 64-dimensional embedding feeds a direct six-class head; its contextual correction is disabled. All original six labels, **164 patients, 391 records and 55,863 beats**, cached R positions, signed amplitude/orientation and frozen patient folds remain.

Use four consecutive retained beats within one recording, class/patient-balanced sampling, then uniform record/start. Each epoch has **2,048 windows / 8,192 nominal beat slots**, **32 windows / 128 slots per batch**, and **64 optimizer steps**. Inner patient macro F1 selects the earliest best epoch with patience six and a 25-epoch cap; its rounded median selects a fresh outer teacher duration. This phase uses no auxiliary beat loss and no external pretraining.

Save each partition's teacher weights, normalization, train/query membership hashes, training-label hash and scope. A teacher is shared among that partition's adaptation controls. The study rejects a teacher from any other partition, even if parameter shapes or patient counts match. Existing outer-trained checkpoints are never reused across inner validation patients. All source numerical modules/results from earlier experiments remain pinned and unchanged.

## Five declared procedures

| Procedure | Second phase | Trainable parameters at default widths |
|---|---|---:|
| `morphology` | Teacher prediction unchanged | No second phase |
| `continued_morphology` | Continue encoder + direct head, context disabled | 78,198 |
| `joint_context` | Adapt encoder, direct head and context together | 88,693 |
| `frozen_context` | Keep encoder/direct head fixed; adapt RR/GRU/gate/context head | 10,495 |
| `frozen_shuffled` | Same frozen model with joint waveform/RR context permutation | 10,495 |

Registered architecture size is **88,693** for all second-phase controls; active capacity differs. Continued morphology controls extra optimization exposure. Frozen versus joint tests trainability under the same warm start; frozen versus shuffled tests chronological order under freezing. Baseline versus frozen also changes capacity/exposure, so it is not a pure causal effect of one module.

## Second-phase initialization and selection

Copy only the encoder and direct head from the correct teacher. Initialize RR/GRU/gate parameters freshly with the same seed across controls (`phase-one fit seed + 500000003`). Set the contextual output layer's weights and bias to **zero**, retaining the gate's sigmoid initialization at bias −3. The initial correction is exactly zero, so all adaptation arms initially reproduce the teacher. The nonzero gate allows the context output layer to receive gradients immediately; gradients reach earlier context layers after that layer updates.

All adaptation arms disable direct-head dropout, including the joint/continued controls; eval mode does not block gradients in the trainable controls. Context-head dropout remains enabled during training. Use a fresh AdamW optimizer, LR 0.001, weight decay 0.01, gradient clipping at five, and window negative log probability with no auxiliary term. The config's inherited auxiliary coefficient is unused because no `_aux` arm is invoked. The second phase uses the same nominal window/beat/update budget as the first, with a separate deterministic sampling stream. All adaptation controls share sampled windows/direct beats at common positive epochs; only contextual embedding/RR pairs are jointly permuted for the shuffled arm.

**Epoch zero is an inner candidate.** Evaluate the unchanged teacher before adaptation. The earliest-best policy can select zero adaptation epochs, with the same patience-six / 25-epoch cap. Each adaptation arm's rounded median inner best epoch (including zero) sets its outer refit duration. Outer patients never choose a phase duration or gate value. A zero-update outer adaptation reproduces the teacher; this inner fallback does not guarantee a better outer F1. Total selected exposure can differ between arms, and the baseline has no second phase.

Frozen encoder/head weights are hashed initially, at every adaptation epoch, and in the selected saved checkpoint. The report checks those hashes and trainable parameter counts, zero-initialized context, per-partition teacher identity, all normalization/window/permutation/stopping audits and restored predictions.

## Budget and endpoint

- Ten outer folds × three seeds **42/123/2026** × (three inner fits + one outer refit) = **120 partition/seed contexts**.
- One shared teacher plus four adaptation procedures per context = **600 phase-fit/checkpoint contexts**: 120 teachers and 480 adaptations.
- **150 outer checkpoints**, plus inner phase checkpoints. Independently restore **all 600 selected phase models**, with 120 matched four-adaptation partition checks.
- Some outer adaptation checkpoints may select zero updates. Report their count explicitly; they are not gradient-trained models or extra independent patients.
- **Equal-probability ensembles across all three seeds are declared before this study**, for every arm. Also report all member metrics and their mean/SD. Ensemble F1 is a separate predictor, without a seed SD or seed/arm subset selection.

Primary comparisons are frozen context versus its fresh teacher, continued morphology, joint adaptation and shuffled frozen context. Preserve all outcomes and all six original class F1 values, including AMI/ALMI weaknesses. Historical 0.584570 uses another training recipe and remains context rather than the matched control. No score-driven cohort filtering, clinical phase relabeling, ECG extension or prior probability correction is introduced.

Inference covers every query beat exactly once, masks incomplete tails, weights window probabilities by valid-beat count within each recording, then averages recordings equally per patient. It remains offline analysis. Descriptive uncertainty uses 5,000 class-stratified paired patient draws conditional on fitted ensemble predictions, without refits or correction for overlapping training folds, repeated development or multiple comparisons. These repeatedly inspected PTB patients are development data; independent confirmation remains necessary for generalization/SOTA claims.

## Validation and commands

Eight targeted tests passed before launch: exact zero-residual prediction preservation, real frozen/joint gradient behavior, unchanged teacher state, deterministic direct-head dropout policy, epoch-zero fallback, actual tiny two-phase nested fit/restore/resume, outer-label counterfactuals, tamper rejection, strict teacher partition checks and monitor/verification guards. The actual-data MPS smoke passed every adaptation arm, weight-freeze invariants, restored predictions and the zero-correction teacher equivalence. These checks establish implementation behavior, not a classification gain. Windows is not numerically validated.

The complete macOS regression suite passed **234 tests** before full launch.

```sh
.venv/bin/python -m mi_localization.staged_context_study --smoke
.venv/bin/python scripts/run_staged_context.py --publish
.venv/bin/python scripts/live_status.py --watch
```

The launcher uses an OS lock, reuses a matching passed smoke, runs full fits plus independent reporting, and publishes only aggregate results from a clean main branch. Log: `tmp/staged_context_v1.log`; runner metadata: `tmp/staged_context_v1_runner.json`; output: `results/neural_staged_context_v1`. The monitor shows the active phase, arm, fold, seed, epoch and gate; completed phases include zero-update checkpoints. Ctrl+C stops only the monitor. Direct modules are `staged_context_study` and `staged_context_report`; native Windows uses `.venv\Scripts\python.exe` and requires compatible source banks in its own namespace.

[Configuration](../configs/staged_context_v1.yaml). Fitted weights, per-patient probabilities and arrays remain local/ignored; aggregate metrics and hashed audits may be published. Do not edit pinned numerical code/config during a run or resume the older paused multiview sweep.

Full study launched **16 November 2026, 14:10 IST**, from implementation commit `a8140ed`. Both launcher and worker were observed running and the first inner teacher completed an epoch. This launch observation is historical; training and verification subsequently completed in `a56fe23`. No worker/launcher was alive at completion review.

## Completed outcome — 16 November 2026

[Verified report](../results/neural_staged_context_v1/complete/REPORT.md) and [machine-readable summary](../results/neural_staged_context_v1/complete/summary.json), published automatically in `a56fe23`. All 600 phase checkpoints restored, including 150 outer models; all 120 matched checks and query-beat coverage passed. The completion review matched the current scope and rechecked all 600 checkpoint/probability artifact hashes without new training.

| Procedure | Three-seed ensemble macro F1 | Zero-update outer adaptations |
|---|---:|---:|
| Morphology | 0.545112 | No second phase |
| Continued morphology | 0.549550 | 23/30 |
| Joint context | 0.559042 | 20/30 |
| Frozen context | 0.563539 | 21/30 |
| Frozen shuffled | 0.565056 | 18/30 |

Frozen−morphology is +0.018428 [−0.021765,+0.061253]; frozen−joint +0.004498 [−0.044728,+0.053981]; frozen−shuffled −0.001517 [−0.021361,+0.017507]. These descriptive patient intervals all include zero. The frozen mean member F1 is 0.555054 ± 0.013390 versus teacher 0.550704 ± 0.027315; this is a different endpoint from ensemble F1 and SD across only three optimization seeds.

The point improvement over the matched morphology ensemble does not establish a gain, and chronological order did not outperform its shuffled control. The historical 0.584570 ensemble remains a separate reference; this study did not exceed it. Inner selection selected epoch zero in 215/360 adaptation fits, leading to 82/120 outer adaptations with no gradient updates. In particular, only 9/30 frozen-context outer models adapted at all. This records frequent rejection of this adaptation recipe by inner validation; it does not prove temporal information is inherently useless or that more patients cannot help. AMI ensemble F1 rose from 0.286 to 0.400, while ALMI and IMI fell slightly, so class changes are mixed.

No new training was started during completion review. Keep completed numerical code/config/results pinned, retain the prior reference, and require a separately declared hypothesis and matched controls for any follow-up.
