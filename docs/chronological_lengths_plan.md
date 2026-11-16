# Nested comparison of 4, 8 and 16 consecutive beats

Revised 15 November 2026 at the user's request, **before any full chronological fit started**. This replaces the waiting fixed-four queue, under a new experiment namespace. Fixed-four and fixed-sixteen configs/history remain available but are not queued. The existing signature study is undisturbed. No beat-length classification result is available at implementation time.

**Completed 16 November 2026:** all **1,200 fits**, **120 outer model restorations**, **300 matched recurrent-partition checks** and complete query-beat coverage passed verification. Mean ± seed SD macro F1: mean+RR **0.485 ± 0.055**, ordered/no-RR **0.509 ± 0.039**, ordered+RR **0.524 ± 0.018**, shuffled+RR **0.533 ± 0.013**. Ordered minus shuffled: **−0.008 [−0.057,+0.039]**, descriptive paired patient interval; no demonstrated ordering benefit. RR versus no-RR: **+0.016 [−0.029,+0.061]**, also uncertain and conditional on the shared length policy. Lengths 4/8/16 were selected 18/11/1 times in the 30 overlapping training decisions; these are not per-length outer scores or a universally optimal length. The historical ResNet mean remains 0.550 under a different procedure. [Validated report](../results/neural_chronological_lengths_v1/complete/REPORT.md), result commit `e4377f4`. No new full sweep is started by reviewing this result.

## What is being compared

Use the existing measured-XYZ ResNet/GRU, preceding-RR features, original patients/labels/R positions and ten outer/three inner patient folds. The same four arms are evaluated: mean_rr, ordered_no_rr, ordered_rr, shuffled_rr. See [the base chronology protocol](chronological_vcg_plan.md) for the model, masks, source metadata and inference aggregation.

Every inner training partition tests every arm at each candidate length. Within any one length, controls share patient/record/window draws and recurrent initialization. Windows never cross recording or patient boundaries. Shuffling moves waveform/RR pairs jointly, preserving their contents without inventing intervals. Normalization and early stopping remain training-only.

| Consecutive beats | Windows/epoch | Windows/batch | Nominal beat slots/epoch | Optimizer batches/epoch | Inference windows/batch |
|---:|---:|---:|---:|---:|---:|
| 4 | 2,048 | 32 | 8,192 | 64 | 64 |
| 8 | 1,024 | 16 | 8,192 | 64 | 32 |
| 16 | 512 | 8 | 8,192 | 64 | 16 |

All lengths use 128 nominal beat slots per optimizer batch and 256 per inference batch. Model parameter shapes are unchanged. Equal beat exposure does not make the optimization identical: shorter windows have more window labels and different recurrent context. Different lengths do not contain exactly the same sampled windows. Actual valid-beat presentation counts are audited, with no dropped tails/patients; this cohort supports every full training length.

## Choosing length without outer-test selection

For each **outer fold and training seed**, calculate the three-inner-fold mean patient macro F1 for ordered_rr and shuffled_rr at each length. Choose the shared length maximizing:

```
shared_length_score(L) = (mean_inner_F1(ordered_rr, L)
                         + mean_inner_F1(shuffled_rr, L)) / 2
```

Configured-order ties choose the first of 4/8/16. Selection is symmetric in the primary order-control pair rather than favoring the main model's validation fluctuations. All four outer arms use this same chosen length; every arm gets a fresh outer refit for its own rounded-median inner selected epochs at that length. No candidate's outer probabilities are computed to choose the length.

Primary contrast: ordered_rr versus shuffled_rr, at a matched shared length. Secondary RR/no-RR and mean-pooling comparisons are **conditional on that primary-pair length-selection rule**; they are not independently length-optimized competitors. All four arms have the same candidate grid and inner fit count, but only the primary pair's inner scores define the shared choice.

A chosen length may differ by outer fold/seed. Selection frequencies are training diagnostics, not independent proof that a single length is universally optimal. Inner curves are optimized scores; do not present the highest as project performance or retrospectively choose the highest per-length outer result.

## Fixed budget and verification

Three optimization seeds 42/123/2026, ten outer folds, four arms, three lengths and three inner folds give:

- **1,080 inner fits:** 3 seeds × 10 outer folds × 4 arms × 3 lengths × 3 inner folds.
- **120 fresh outer fits:** one chosen length × 4 arms × 3 seeds × 10 outer folds.
- **1,200 total fits**, followed by independent report verification and all 120 outer model restorations.

The report reconstructs all train-only moments, windows, permutations, inner stopping and length choices. It checks matched recurrent initialization/schedules and complete query-beat coverage. It publishes all training seeds, patient macro F1, each class F1, MI-only scores with unchanged six-way predictions, descriptive paired intervals, and shared selected-length counts. Seed variation and conditional bootstrap intervals do not establish independent confirmation on these repeatedly inspected PTB development patients.

No encoder trained on outer-training patients initializes an inner fit. Every model begins fresh. There is no single global best length selected from all outer labels, new patient filtering, class change or guaranteed score gain. A gain over shuffled control alone need not exceed the historical ResNet 0.550.

## Files and execution

- [Grid config](../configs/chronological_lengths_v1.yaml) references the unchanged [base config](../configs/chronological_vcg.yaml).
- Source: `chronology_lengths_study.py`, `chronology_lengths_report.py`; shared numeric model and metadata modules remain unchanged.
- New results: `results/neural_chronological_lengths_v1`. Old fixed-four outputs are not overwritten.
- Local log: `tmp/chronological_lengths_v1.log`.
- Queue: `python scripts/run_chronological_lengths.py --wait-for-signature --publish`. It waits for the current complete, fingerprint-matched signature report and stops on dependency failure. A clone does not inherit the originating Mac process.
- Direct prepared-machine execution: `python scripts/run_chronological_lengths.py --publish`.

GPU smoke checks cover forward/backward and disk state restoration at **every candidate length**, without reporting classification improvement. Native Windows must pass its own environment tests/smoke; use a new compatible config/output namespace when source/runtime fingerprints differ. Arrays, private predictions and `.pt` models remain ignored.

## Terminal live monitor

From the repository root on the current Mac:

```sh
.venv/bin/python scripts/live_status.py --watch
```

It refreshes every five seconds. For a single snapshot, omit `--watch`; for a machine-readable snapshot use `--json`. It shows the signature prerequisite and the new beat-length job separately, including fit counts, queue/training/verification state, matching live worker/launcher commands, selected/current length, fold, seed, epoch, training loss and last-update age. Inner validation F1 is labeled as such, not a final model score. Counters are completed-fit counts, not elapsed-time estimates.

On Windows use `.venv\Scripts\python.exe scripts/live_status.py --watch`. Process inspection is read-only (`ps` on POSIX, CIM command-line metadata on Windows); it never uses a Windows kill signal as a status check. If process information is unavailable, the monitor says so rather than treating a saved PID as proof of activity. It reports local state only.

**Ctrl+C stops only the monitor**, not the detached experiment. Alternatively, follow logs with `tail -f tmp/chronological_lengths_v1.log`. The generic `project_status.py` remains a saved-file inventory rather than a live process check.
