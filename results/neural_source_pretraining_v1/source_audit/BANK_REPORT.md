# Derived-XYZ source pilot bank

1,152/1,152 selected recordings retained; one 651-sample crop per unique retained source patient.

| Role | Retained patients |
|---|---:|
| train | 1024 |
| validation | 128 |

Exclusions: `{}`. Exact cross-patient source-file duplicate groups: 0; every selected member of a detected group is quarantined.

Waveforms are checked for 500 Hz, 5000 samples, 12 unique leads, mV units and finite values. Eight independent leads feed the existing Kors matrix; polyphase resampling produces 1000 Hz XYZ. Existing bior6.8 full-record denoising and detector settings are reused. One median eligible detector crop is retained. Flat/nonfinite signals and missing complete crops are recorded as exclusions. Malformed files fail the build rather than being silently skipped.

No diagnostic labels enter selection or pretraining. Source training and validation patients are disjoint; fold 10 waveform access is rejected before I/O. Target PTB data, folds, labels and scores are untouched.

Detector positions and derived-vector fidelity are not independently adjudicated on PTB-XL. Local train/validation example plots and per-record audits are in `review/`. No sensitivity/PPV, physiological coverage, cross-dataset patient independence, localization improvement or SOTA is claimed. Comprehensive PTB-versus-PTB-XL waveform/identity overlap remains unresolved; downstream PTB benchmarking has not been cleared by this audit.
