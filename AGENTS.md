# Project instructions for coding agents

Read [docs/CODEX_HANDOFF.md](docs/CODEX_HANDOFF.md) first. It contains the project goal, experiment history, evidence links, active work and known limitations. For a new Windows machine also read [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md). Run `python scripts/project_status.py` for a dependency-free, read-only inventory of this checkout. That command does not establish whether a saved training PID is alive.

## Scientific contract

- Primary target: **six-class patient-level macro F1** on measured Frank XYZ VCG. Classes in configured order: AMI, ALMI, ASMI, IMI, ILMI, HC. Micro F1 equals accuracy for this single-label task and is secondary, not a replacement target.
- Current target cohort: 164 patients, 391 records, 55,863 beats. Keep the original labels, R positions and `configs/patient_folds.csv`. Beats/records from the same patient must stay together.
- Normalization, reference libraries, selection, stopping and any learned combination belong inside the relevant training partition. Outer-test labels must not choose models, epochs, thresholds, feature budgets or seeds.
- Report all declared folds/seeds and patient-level results. A best seed, partial fold result, inner-validation score, reconstruction score or random-beat accuracy is not the final project macro F1.
- This PTB cohort has been repeatedly inspected and is development data. Do not call it untouched validation or claim SOTA, clinical validity, a demonstrated ceiling or a guaranteed 0.80–0.90 score.
- Keep measured VCG primary. ECG-assisted/ECG+VCG work is a separately labeled extension. Preserve signed XYZ orientation and amplitude; arbitrary rotations and per-beat unit-amplitude normalization are not assumed safe for localization.

## Work safely with existing experiments

- Completed study/bank modules are pinned by source/config/runtime hashes in manifests. Before editing an old experiment, inspect its manifests and dependencies. Prefer a new module/config/output namespace for a changed hypothesis. Never disable integrity checks to make a cache load.
- Do not overwrite completed reports or mix Windows-generated runs into a Mac output namespace. A file with `complete` in its directory name is not proof of completion; check the validated report/summary and scope.
- The healthy-reference study completed and was published in `3a0caf2`: 0.479 mean macro F1 versus matched ensemble 0.499, with no demonstrated gain. The current phase is the separately controlled ordered-signature study; check local status and latest Git history for actual run completion. A cloned repository does not contain/control another computer's process. Do not stop, restart or duplicate a full sweep merely to learn the project.
- The older multiview neural sweep was paused. Do not resume it unless the current user asks. Ordered signatures are implemented, with performance pending until their completed report exists. The consecutive-beat/RR study is now implemented and approved for queued execution after signature verification; see `docs/chronological_vcg_plan.md`. Queue readiness, active training and a completed score are distinct states.
- Data, caches, fitted weights, fitted PFN contexts and individual patient predictions are deliberately ignored. Do not commit them, credentials, private review packets or `.venv`. Public aggregate metrics and hashed audits may be committed.
- Stage only task-owned files; do not accidentally include untracked partial experiment outputs. Follow the user's current Git/push instructions and do not change collaborators, permissions or protected branches on your own.

## Implementation and validation

- Python package: `src/mi_localization`; protocol files: `configs`; meaningful synthetic/integrity tests: `tests`. Run from the repository root. Use `rg` for search.
- Test the changed behavior, then the relevant suite. Last complete macOS verification at the handoff: **199 tests passed**. This is historical evidence, not a claim that Windows has passed.
- Local TabPFN-v2 inference must use `tabpfn_backend.run_fit/run_predict`, which spawn isolated workers. Direct TabPFN/LightGBM numerical loading in the feature-selection process previously caused a macOS OpenMP failure. Do not change installed numerical binaries or use duplicate-OpenMP override hacks.
- For the established tree procedures, preserve the existing single-thread XGBoost/threadpool contexts. Do not silently change numerical dependencies, checkpoint versions or search budgets during a resume.
- Keep text/source files in LF form under `.gitattributes`. Windows uses `.venv\Scripts\python.exe`; macOS/Linux use `.venv/bin/python`. The zsh launchers and Keynote/AppleScript scripts are platform-specific conveniences; use Python entry points for portable work.
- Prefer reports and machine-readable metrics over old presentation text. Update the handoff when a completed result or research decision changes, clearly separating completed, active, paused and proposed work.
