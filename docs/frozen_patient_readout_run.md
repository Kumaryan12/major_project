# Running the frozen patient readout study

Implementation: `c40fbaf7`. [Fixed protocol](frozen_patient_readout_plan.md) · [Configuration](../configs/frozen_patient_readout_v1.yaml).

The full CPU run started on the originating Mac on 19 November 2026. The launcher first fits the patient classifiers, then starts a separate verification process. No following experiment is queued. The encoders are already trained: this stage fits new patient-level ridge classifiers on their frozen features. It does not update neural weights. The inner-only fitting/restoration smoke, six focused tests and the full 371-test macOS suite passed. These checks are implementation evidence, not a completed study or a score improvement.

From the repository root, monitor local command-checked progress:

```sh
.venv/bin/python scripts/frozen_patient_readout_status.py --watch
```

Read the log:

```sh
tail -f tmp/frozen_patient_readout_v1.log
```

Windows uses `.venv\Scripts\python.exe` instead. A clone does not contain the originating computer's checkpoints or control its processes. The recorded CPU runtime and source hashes must match for a completed-fit resume.

There are 120 encoder feature contexts and 420 ridge classifiers. Saved counts measure fitting artifacts, not independently verified performance. During verification the process reconstructs those same contexts and restores all classifiers; the fitting counters therefore stay at their totals. A final `complete/summary.json` and `complete/REPORT.md` appear only after the full replay and baseline-equivalence checks pass. Individual feature and model artifacts remain private under the ignored `review/` directory.

The primary endpoint is the original 145 dated earliest-record patients, with a matched original-model macro F1 of .521335. The 164-patient fallback sensitivity still uses one selected recording each. Neither is the historical all-record .60 endpoint.

For reproduction or a completed-fit resume on a compatible machine, after checking that no matching worker is active:

```sh
.venv/bin/python -m mi_localization.frozen_patient_readout
.venv/bin/python -m mi_localization.frozen_patient_readout --verify
```

Do not launch another copy to check progress, edit the pinned numerical code/configuration during a run, or treat smoke results as full-study results.
