# Queued next process: expanded source recording-overlap audit

Authorized 16 November 2026 by “please queue the next best process.” This follows the native supervised-source pilot and supersedes the earlier instruction to wait for a next part **only for this audit**. The original source runner still stops as designed; this separately authorized queue starts the next process. No target classification/transfer study or subsequent process is included.

## Why this process comes next

The supervised pilot selected 5,063 training records from 4,096 patients and the complete native validation fold (2,183 records / 1,942 patients). Its encoder and chosen epoch can be influenced by both the training and model-selection-validation recordings. The historical copy screen covered only 1,152 source recordings. It cannot support claims about this expanded cohort.

The useful next gate before a matched scratch-versus-supervised-pretrained target comparison is therefore a new recording-level audit covering **every retained recording used by the new source models**, against all **391 recordings from the original 164 PTB target patients**. This is necessary provenance work, not a new representation, model improvement or additional target patients.

## Queue and completion gates

1. Wait for the existing source-pilot launcher and its active bank/training worker to finish. Do not duplicate preparation, train a missing model, restart paused work, or stop the current pilot.
2. Require a complete source report, training scope, source bank and every declared objective/seed. Independently invoke the source trainer with `--verify` to restore models, recompute predictions/metrics and verify hashes. Only then start the expanded audit.
3. Pin the verified source reports/bank/cohort, every selected input header/data hash, original target beat/fold checksums and cohort counts, code/config/runtime, and a new output/cache namespace. Reject any changed inputs/code rather than editing manifests.
4. Run the expanded screen; independently rerun it with `--verify`. Publish aggregate results even when a completed screen finds candidates or insufficient active leads, with its passed flag false. Failed known-copy controls prevent a completed/passed audit. Stop afterward in all cases; review is required before any later transfer claim or study.

The queue waits for the active source launcher PID on the originating Mac and also checks the actual command of the worker in the source progress file. A missing/reused PID alone cannot establish completion. If the source process ends incomplete, the queue records an error and starts no audit or fitting. An OS-held queue lock prevents duplicate launchers and releases on crash.

## Numerical screen

Reuse the existing pinned `overlap_screen.py` numerical functions without modifying the historical study. Source metadata folds remain1–8/9; fold10 is rejected before waveform access. This source inventory is the new verified bank's retained recording IDs/groups/roles, not an old one-record-per-patient pilot or a union of selected records that never reached fitting.

Read the eight independent ECG leads I/II/V1–V6 in physical mV from source and target recordings, validating lead identity, rate, gain, units and local data paths. Resample to250Hz. Build four normalized morphology descriptors per record: median/first/middle/last detector-centered beat templates,66points per lead. Descriptor normalization is solely for copy retrieval, not a change to the signed/amplitude-preserving classification input.

Use seed42 and20 synthetic known-copy controls made from ten-second target segments, with a gain/offset and small added noise. Choose the first successful retrieval count from8/16/32 based only on retrieving and confirming every known control. This is not selected using target classification labels/errors. Retrieve in128-source blocks to avoid the full expanded feature-similarity matrix. Missing source templates fall back to every target; missing target templates are always included.

For each retrieved/fallback pair, correlate the source's central eight seconds against all valid target lags. Require at least six active leads (SD≥0.00001mV), mean correlation≥0.995 and minimum active-lead correlation≥0.98. Record full ten-second comparisons when the implied segment fits. Also check encoded data-file hash matches. Run two CPU workers. Report actual pair coverage against all possible pairs; this is not an exhaustive all-pair all-lag search. At top8 and roughly7,246 retained sources the nominal budget is about58,000 continuous comparisons before fallbacks, versus about2.83million possible recording pairs.

The recording-screen passed flag requires every synthetic control to succeed, no encoded-file matches, no high-correlation candidate pairs, and no selected pairs with insufficient active leads. A candidate is a review flag, not a diagnosis or proof that two records identify the same person. The audit never removes/relabels target patients to obtain a passed flag or score.

## Independent verification and privacy

Each record cache and target-pair checkpoint has a pinned context/hash. Resume uses only compatible work. Full independent `--verify` re-decodes **every** original common-lead waveform, reconstructs/resolves its cache, repeats all synthetic controls/retrieval, and recomputes **every** selected continuous pair. Re-read input hashes at the end to detect mid-run changes. Hash and check private provenance, controls, retrieval, pair checkpoints and public results.

Record/patient IDs, matching pairs, waveforms and caches stay in ignored review/data directories. Publish only aggregate counts/report and hashed manifests. No individual classification predictions/errors enter screening. Source training, original target folds/labels/R positions and all prior completed studies remain unchanged.

Even a passed screen does not resolve the identity of different visits, copies shorter than eight seconds, transformations outside the controls or retrieval misses. It cannot establish clean patient-independent cross-dataset transfer, clinical equivalence, independent confirmation or SOTA. A later target comparison would remain explicitly exploratory until stronger provenance evidence exists. No new macro-F1 result can come from this audit alone.

## Commands

Validation before queue registration: **281 tests passed** on the originating Mac, including eight new prerequisite/completion, copy-control, chunked-retrieval/fallback, cache/pair replay, tamper and waiting-queue checks. An I/O smoke on three real cached source and three original PTB target records passed header/common-lead validation and raw cache decode/replay. These checks produce no target score or completed full-screen result.

```bash
# Queue behind the actual source launcher; substitute its verified local PID.
.venv/bin/python scripts/queue_supervised_source_overlap.py --wait-for-pilot-pid PID --publish

# Monitor both the native pilot and the queued audit.
.venv/bin/python scripts/supervised_source_next_status.py --watch

# Once source artifacts are complete, run/replay directly in a compatible scope.
.venv/bin/python -m mi_localization.supervised_source_overlap
.venv/bin/python -m mi_localization.supervised_source_overlap --verify
```

Source pilot logs: `tmp/supervised_source_pilot_v1_prepare.log` and `..._train.log`. Queue log: `tmp/supervised_source_overlap_v1_queue.log`. Outputs: `results/supervised_source_overlap_v1`; caches: `data/processed/supervised_source_overlap_v1`. Use `.venv\Scripts\python.exe` and fresh output/cache namespaces on Windows; a cloned queue/PID cannot control the originating Mac's process.

Numerical reference: [SciPy cross-correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html). Protocol/settings are project choices; the official correlation definition does not establish retrieval sensitivity or patient independence.
