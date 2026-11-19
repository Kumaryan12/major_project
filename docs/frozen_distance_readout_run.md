# Frozen64 distance comparison: execution and monitoring

[Fixed protocol](frozen_distance_readout_plan.md) · [Configuration](../configs/frozen_distance_readout_v1.yaml). This comparison restores the completed ridge64 baseline and fits equal-prior centroid and shared-covariance Mahalanobis heads. The 64-feature encoders and their exact inner/outer fitting patients remain fixed. This is a classifier study with zero neural weight updates.

Full fitting launched on the originating Mac on 19 November 2026 at 11:47:30 IST, from implementation `b1916616`. The worker and launcher commands were checked alive. This dated launch note does not establish current activity or completed performance.

The primary comparison is Mahalanobis−ridge on the same 145 dated earliest-record patients. The verified ridge reference is .535691. The 164-patient fallback sensitivity also uses one selected record per patient. Neither is the historical all-record .60 benchmark.

Inventory: 420 new distance heads, 210 restored source ridge heads, 120 independently reconstructed feature contexts, at least 90 full-shrinkage identity controls and 60 new outer heads. The real-data inner-only smoke restored both source ridge candidates and all four distance candidates and passed its identity control. All 1,444 source files remained unchanged. Six focused tests passed, including query-label isolation, class-balanced residual covariance, analytic distance probabilities, degenerate features and corrupted-file rejection. The full macOS suite passed all 376 tests before launch.

From the repository root:

```sh
.venv/bin/python scripts/run_frozen_distance_readout.py --publish
```

The launcher fits, starts a separate independent verification process, checks the declared completion inventory, publishes only its aggregate manifest/summary/report if requested, and exits. Any failure stops the launcher with an error rather than skipping checks. No next experiment or old paused sweep follows. Completed heads can be restored under an unchanged scope; source/protocol/runtime changes require a separate namespace.

Monitor the actual local process and counters:

```sh
.venv/bin/python scripts/frozen_distance_readout_status.py --watch
```

The log for the originating Mac launch is `tmp/frozen_distance_readout_v1.log`. Its one-shot macOS launch job uses explicit `KeepAlive=false`; local job metadata is stored under ignored `tmp/`. It is loaded only for the current session, not installed for login startup. The same Python runner can be used directly on another compatible platform; Windows uses `.venv\Scripts\python.exe`, needs its own environment/data migration, and has not been validated.

Saved counters do not measure independent verification completion. The final report appears only after all declared models, features, selections, identity controls and source/baseline equivalences replay successfully. Model files, compact feature matrices, per-patient predictions and launcher state stay private under ignored `review/`. A clone cannot monitor or control another machine's process.

Most work is streaming existing waveforms through frozen encoders twice. Small head fitting is fast. Based on the previous full feature/replay run, allow roughly 1–1.5 hours on this laptop for fitting and verification together; this is a planning estimate, not a guaranteed ETA or a partial performance result.
