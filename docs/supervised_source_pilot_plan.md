# Bounded supervised native-diagnosis source pilot

Authorized 16 November 2026: implement the next source-data expansion and training step, then **stop and wait for the next part**. No measured-VCG target fitting, transfer evaluation, new target score or subsequent experiment is included.

## Frozen design

- PTB-XL 1.0.3 official pinned metadata, native folds 1–8 training, complete fold9 validation, fold10 excluded before label/waveform access. Existing source audit and all target/model artifacts remain unchanged.
- Include every training patient with at least one likelihood100 AMI, ALMI, ASMI, IMI or ILMI annotation. Deterministically fill to 4,096 training patients with seed42. All selected visits remain, with their own native labels. No validation annotation, target error, demographic or metadata quality flag chooses training patients.
- Actual metadata selection: **1,841 mandatory MI-positive training patients**, **4,096 total training patients / 5,063 records**, **1,942 validation patients / 2,183 records**. Training is enriched; validation keeps native prevalence.
- Download/copy only these 7,246 source pairs into `data/raw/ptbxl_supervised_v1`, distinct from the frozen older cache. Reuse eligible old pairs by copying, never symlinking or modifying them. Raw and derived banks are ignored. The source-download budget is approximately 870 MB of uncompressed waveform data before cache reuse, plus small headers/crop caches.
- Validate native header/rate/shape/leads/units/gains/encoding/size, decode and check finite/nonflat XYZ, resample 500→1000Hz and apply the existing bior6.8 denoiser/detector. Retain up to four uniformly spaced complete R−250…R+400ms crops per recording. Fewer valid crops use masks, not fabricated beats. Fail malformed downloads; explicitly record signal-QC failures.
- Screen raw encoded-file hashes and decoded full-record derived-XYZ hashes for exact selected-source copies. Quarantine every cross-patient member, including cross-role groups; same-patient copies retain the lowest recording ID. Preserve duplicate identities/QC privately. This does not perform a near-copy screen or prove cross-dataset patient independence.
- Preserve all 44 diagnostic state columns and likelihoods. Train only native diagnostic heads with at least 20 distinct **retained training patients** at likelihood100. Head eligibility is chosen after QC from training only; validation support never selects heads. No exclusive six-way remapping and no automatic NORM→clinical-HC mapping.

## Two declared annotation conventions

Unlisted codes act as **annotation negatives** in this benchmark. That is an explicit research convention, not clinically adjudicated absence of disease. Likelihood0 is documented as unknown and is masked in both arms. Intermediate values are not calibrated probabilities.

1. `certain` (primary): likelihood100 positives, unlisted annotation negatives; mask listed unknown/intermediate entries.
2. `positive_intermediate` (sensitivity): intermediate and100 entries are positive; unknown0 masked; unlisted annotation negatives.

Both arms use the **same validation mask/labels**:100 positive, unlisted annotation-negative,0/intermediate masked. Report the common valid observations and per-code support. This is a supervision sensitivity check, not a causal test of label noise.

## Representation and optimization

Reuse the established small ResNet encoder (widths16/32/64, four temporal bins,64-dimensional representation), shared across each recording's valid crops. Average crop features, then apply dropout0.2 and a multilabel head. Pooling is unordered; there is no new chronology claim. Native record labels supervise the recording, not each individual heartbeat.

Fit signed per-axis means/scales using training records only, weighting patients equally, records equally within each patient and valid crops equally within each record. Preserve amplitude/orientation; no per-beat amplitude normalization, augmentation/rotation or target statistics. Training samples records with replacement at inverse-visit-count weights, so patients have equal sampling mass. Training-only per-code negative/positive loss ratios are clipped to [1,20]. Masked BCE averages observed labels within each sampled recording.

One prespecified feasibility seed **42**, both objectives, AdamW learning-rate0.001/weight-decay0.01, batch64,4,096 presentations/epoch, at most15epochs, patience4, gradient-norm cap5. Use common source-validation macro AP for stopping, with earliest ties and epoch-zero fallback. Do not select seeds or alter budgets after seeing source/target outcomes. MPS is seeded but not bitwise deterministic.

The primary source metric is **patient-weighted record-level annotation macro average precision**. Record weights are1/(retained visits for that patient), applied before per-code masks; record-specific labels remain distinct. Also report macro AUROC and fixed0.5-threshold macro F1, plus per-code counts/AP/AUROC/F1 and masked entries. Codes lacking both positive and negative validation observations are listed but excluded from macro metrics. No threshold tuning or independent-test claim. Constant training-prevalence and random-initialization controls are reported.

These source metrics are a different endpoint from six-class patient macro F1 on measured Frank VCG. A source AP/F1 cannot replace the project score or establish transfer benefit.

## Completion and verification

Source banks/fit scopes hash code, config, metadata, source files, arrays, normalization and runtime. Epoch checkpoints save optimizer, best model and sampler/loader/torch/device RNG states. Every completed run is restored and its full validation predictions recomputed in a separate verification invocation before publication. Report all declared fits, including epoch-zero fallback/failures; keep weights and record-level labels/predictions private.

Update progress in `results/supervised_source_pilot_v1/progress.json`; logs in ignored `tmp/`. Public reports live in `results/supervised_source_pilot_v1/`. Windows uses a fresh namespace/runtime and `.venv\Scripts\python.exe`; do not override integrity guards to reuse a Mac run.

```bash
.venv/bin/python -m mi_localization.supervised_source_bank --prepare
.venv/bin/python scripts/run_supervised_source_pilot.py --publish
.venv/bin/python -m mi_localization.supervised_source_train --verify
.venv/bin/python scripts/supervised_source_status.py --watch
```

The launcher has **no downstream transfer step**. After source results and verification, stop for the user's next instruction. Any later target comparison must separately address expanded recording-overlap/provenance limits and retain the original patients, folds, labels and all training-only selection rules.

Native dataset/fold/likelihood semantics: [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/). Architecture is the project's existing `neural.VCGResNet`, not a new published SOTA model.
