# Expanded supervised-source recording-overlap audit

Expanded recording-copy screen only; no patient-identity clearance, PTB classification, localization gain or SOTA claim.

Covered **7,246 retained source recordings / 6,038 source patients**, including training and source model-selection validation, against **391 PTB recordings / 164 target patients**. The pilot originally selected 7,246 records; source-only QC/exact-copy exclusions are inherited from its verified bank. This replaces no historical result and extends no old subset clearance.

| Check | Result |
|---|---:|
| Encoded-file source–target matches | 0 |
| High-correlation candidate pairs | 0 |
| Pairs with fewer than six active leads | 0 |
| Known-copy controls retrieved and confirmed | 20/20 |
| Retrieved/fallback continuous pairs checked | 57,968 / 2,833,186 |
| Recording screen passed | True |

Common physical-mV I/II/V1–V6 ECG is resampled to 250Hz. Median/first/middle/last morphology templates retrieve top-8 targets per source, escalating through prespecified counts only on synthetic copy controls. Missing source templates use all targets; missing target templates are always included. The source central eight seconds is compared at every valid target lag. At least six active leads, mean correlation≥0.995 and each active lead≥0.98 define a candidate. Full ten-second checks are reported privately when the aligned segment exists. Maximum checked query mean correlation: 0.858734.

Every input file, numerical implementation, runtime, source-bank/source-model provenance and protected target beat/fold checksum is pinned. Private recording caches and pair checkpoints support resume; independent --verify re-decodes every common-lead waveform, repeats retrieval/controls and every selected continuous comparison, and checks all saved artifact hashes. A failed copy-control retrieval/confirmation blocks this screen; any match or insufficient active leads blocks a passed recording-screen flag. Source/target patients are not deleted or relabeled in response to target outcomes.

This is not a patient-identity clearance: different visits by the same person, shorter copies, transformations outside the controls and missed retrieval remain unresolved. Morphological similarity alone is not proof of duplicate people. The original PTB folds/labels/beats are unchanged, and no target classification predictions, errors or outer scores choose screening thresholds or retrieval budgets.

No target training or transfer comparison is queued by this audit. After independent verification and aggregate publication, the runner stops. A later matched scratch-versus-supervised-pretrained study would remain exploratory development with unresolved cross-dataset identity, and must be specified separately.

Protocol: [expanded audit](../../docs/supervised_source_overlap_plan.md). Numerical method: existing project overlap_screen.py; [SciPy correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).
