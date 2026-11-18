# More measured VCG patients and paired ECG training

The objective is six-class patient macro F1 on the unchanged measured-Frank XYZ target. The two studies use ten frozen patient folds and seeds 42, 123 and 2026. They compare separate hypotheses with matched scratch controls. The fixed target schedule uses eight epochs and 2,048 beat presentations per epoch for a bounded pilot. Each pretrained arm adds 64 updates with batches of 128. Schedules and preprocessing are set before the full runs. The shorter scratch control applies to the pilot, while earlier scores used their respective historical recipes.

## Native XYZ pretraining

Audit official PTB header IDs outside all target folds, retaining the paper's patient294 exclusion. Copy locally available XYZ files, fetch missing official files into a separate namespace and verify rate, leads and physical units. Basic signal checks use per-lead variation, finite values and crop coverage. Exact raw decoded XYZ hashes screen the whole target inventory. The continuous screen retrieves up to eight targets by beat templates and checks a six-second excerpt at 100 Hz for consistent shifted correlation on all three leads. Twelve paired retrieval/shift/gain controls must pass. Exact source duplicates under different IDs and potential target copies quarantine source IDs. This is an exact-copy and shortlisted near-copy screen. Clinical identity requires evidence beyond the recordings.

The audit admitted 124 source patients, 156 recordings and 17,573 beats. There were 157 candidate recordings from 125 IDs after the frozen paper exclusion. One recording failed the variation check. All 12 controls passed, with 1,248 continuous comparisons and zero quarantined IDs. [Audit report](../results/ptb_native_expansion_v1/REPORT.md)

Compare scratch, masked reconstruction on target-training patients, and matched masked reconstruction replacing half of each batch with additional native patients. Both pretraining arms start from identical weights, share half the target samples and use matched time masks, scaling, batch size and update count. Sampling gives an equal chance to each patient within its pool before choosing a beat. Scaling uses the current target training patients. The additional patients' diagnostic labels play no role. Transfer the encoder to a fresh six-class head and use the same target fit in all arms. The primary contrast is expanded versus target-only pretraining.

## ECG-assisted training extension

The same model predicts compact signed DCT descriptors of simultaneously measured 12-lead ECG while observing measured XYZ. Use the independently aligned ECG cache and exact training-only patient weighted descriptor scaling. Each pretraining batch draws one beat per distinct training patient. The shuffled arm uses exactly the same XYZ beats and ECG targets with a nonzero cyclic pairing shift, making every pairing cross-patient. The paired arm preserves the original recording time and beat position. Both arms transfer only the XYZ encoder to a fresh classifier. Held-out inference reads XYZ alone. The primary contrast is paired versus shuffled, with both versus scratch reported. This evaluates the training value of paired ECG and does not establish transfer from an ECG-derived source dataset.

## Verification and usage

Each stage saves private models, probabilities and audits keyed to exact fold, seed, row hashes, sample plans and source scope. A separate process reconstructs normalizers and sampling plans, checks matched initialization and exposure, restores every declared checkpoint and reruns held-out inference. The report uses all 90 models per pilot, reports all three seed results, the equal-probability ensembles, class metrics and paired patient intervals. Both studies are development analyses on previously inspected PTB patients.

```bash
.venv/bin/python scripts/run_vcg_pretraining_pilot.py audit --download
.venv/bin/python scripts/run_vcg_pretraining_pilot.py native --smoke
.venv/bin/python scripts/run_vcg_pretraining_pilot.py bridge --smoke
.venv/bin/python scripts/run_vcg_pretraining_pilot.py native
.venv/bin/python scripts/run_vcg_pretraining_pilot.py bridge
.venv/bin/python scripts/run_vcg_pretraining_pilot.py native --status
.venv/bin/python scripts/run_vcg_pretraining_pilot.py bridge --status
```

Each smoke fits three models and replays the checkpoints with one inner partition, excluding that outer fold's held-out patients. Full runs use separate output scopes. The independent clinical cohort uses a separate [intake and evaluation protocol](measured_vcg_partner_cohort.md).
