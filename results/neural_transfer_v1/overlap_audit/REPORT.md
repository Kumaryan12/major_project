# Source–target recording-overlap screen

1,152 retained PTB-XL source records (training and source validation) versus 391 PTB records from 164 target patients.

Exact encoded data-file matches: **0**. High continuous-correlation candidate pairs: **0**. Positive copied-segment controls retrieved and confirmed: **20/20**.

Checked 9,216 retrieved/fallback pairs out of 450,432 possible record pairs. Candidate retrieval uses median/first/middle/last common-ECG beat templates; predeclared top-k escalation selects 8 after testing synthetic copied controls. Missing templates use exhaustive candidate fallback. At least six active leads are required. Incomplete active-lead pairs: 0.

The eight independent ECG leads are read in common physical mV units and resampled to 250 Hz. Continuous FFT cross-correlation searches all valid lags of the source's central eight seconds in each candidate PTB recording. A candidate requires mean lead correlation ≥0.995 and every active lead ≥0.98. Full source-segment correlation/equality is recorded where its alignment fits. Positive controls are ten-second PTB copies with known gain, offset and small noise, used only to test the screen, never for pretraining or classification.

## Interpretation and gate

Recording screen passed under this bounded design: **True**. A detected match or incomplete screen stops the target experiment pending source review/quarantine and, if needed, source retraining. No target patient is removed or relabeled.

Passing does not identify people or establish that source and target patients are disjoint. Similar beats are not duplicate evidence by themselves; only continuous-segment confirmations are flagged. The top-k design and synthetic controls do not measure population sensitivity, and shorter/modified copies or different recordings of the same person can be missed. Patient identity remains unresolved even if there are zero candidates.

Any subsequent scratch-versus-pretrained comparison remains explicitly exploratory development on repeatedly inspected PTB folds, with the unresolved source-target identity limitation. It cannot be presented as clean patient-independent confirmation or SOTA. Existing source pretraining declarations remain unchanged.

Only aggregate results/provenance are public. Patient/record candidate pairs, controls, raw signals and per-record caches remain local in `review/` and ignored data caches. No PTB classification predictions or error flags enter the screen, and no fold-10 PTB-XL waveforms are accessed.

Method reference: [SciPy cross-correlation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).
