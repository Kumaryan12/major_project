# Exploratory source-to-target transfer

This phase asks whether the completed, source-only masked-reconstruction encoder helps measured-Frank VCG localization relative to fresh scratch training. It does not claim new tensor novelty or established SOTA. Measured VCG stays the primary target input; a joint ECG + VCG classifier remains a separate extension.

## Recording-overlap screen

`configs/source_target_overlap.yaml` freezes a bounded recording-copy screen. It uses all 1,152 retained source records, including source validation, and all 391 target recordings from the unchanged 164-patient cohort. No target predictions, persistent-error lists, diagnostic selection or label changes enter this screen. PTB-XL fold 10 is excluded before source waveform access.

Eight common independent ECG leads (I, II, V1–V6) are read in physical mV and resampled to 250 Hz. Four normalized morphology templates per record (median, first, middle, last eligible beat) retrieve candidate target records. Top-k choices [8, 16, 32] are declared in advance; the smallest passing 20 known-copy controls is used. The controls are random ten-second target segments altered by 1.03 gain, lead offsets and 0.0001 mV Gaussian noise. They test the screen only and are never source-training samples. Missing templates trigger exhaustive candidate fallback.

For each retrieved pair, FFT cross-correlation searches every valid target lag for the source's central eight seconds. At least six source leads must have SD ≥0.00001 mV. A candidate requires mean correlation ≥0.995 and every active lead ≥0.98. Full ten-second segment correlation/equality is recorded when its alignment fits. Exact encoded `.dat` hashes are also compared, although different encoding/rates make this a limited additional check.

The completed screen checked **9,216 of 450,432 possible pairs**, detected **20/20** altered-copy controls, found **zero** high-correlation candidates and encoded-file matches, and had zero unavailable-template/inactive-lead fallbacks. Maximum checked query mean correlation was 0.851. See [screen report](../results/neural_transfer_v1/overlap_audit/REPORT.md).

This is not an exhaustive all-pair waveform search, an estimate of population sensitivity, or patient identity linkage. Shorter/modified copies, candidates outside retrieval, and different recordings of the same person may be missed. Unrelated anonymized identifiers cannot establish patient disjointness. **Patient identity remains unresolved.** Passing allows only an explicitly exploratory comparison. Any detected match/incomplete screen blocks transfer pending source review/quarantine and possible source retraining; target patients are never removed to improve scores.

## Frozen target comparison

`configs/transfer_vcg.yaml` declares both arms and all target training seeds [42, 123, 2026]. Same measured XYZ bank, original 651-sample beats, six classes, labels, R positions and ten fixed outer patient folds. Three inner patient splits, fixed across rates, arms and training seeds, stay wholly within outer-training patients.

| Setting | Scratch | Pretrained |
|---|---|---|
| ResNet architecture | 78,198 parameters | Same |
| Encoder initialization | Seeded random | Fixed source encoder, 77,808 parameters |
| Classifier initialization | Fresh seeded head | Identical fresh head |
| Target learning-rate candidates | 0.001, 0.0001 | Same |
| Normalization | Target-training-only weighted moments | Same; source moments discarded |
| Objective/optimizer | CE, AdamW, decay 0.01 | Same; all encoder parameters fine-tuned |
| Dropout / augmentation | 0.2 / none | Same |
| Sampling | 8,192 weighted beat draws per epoch | Identical shared-epoch draws |
| Maximum epochs / patience | 25 / 6 | Same |

Each arm independently selects the rate with the largest mean best inner-patient macro F1; exact ties favor the configured order. Rounded median best inner epoch at the chosen rate sets the outer refit length. Fresh outer fits use all outer-training patients and never monitor outer test performance. All eligible held-out beats are averaged into one probability vector per patient. There is no best-seed reporting, unplanned ensemble or test-based rate selection.

This is **420 target fits**: three target seeds × ten outer folds × two arms × (two rates × three inner fits + one refit). Source pretraining adds compute beyond the equal target tuning budgets. One fixed source-pretraining seed is reused across three target seeds; these are not independent source-pretraining replications. Historical 0.550 ± 0.040 is context, not the fresh control for this new rate search.

The encoder export SHA, pilot/source-bank manifests, overlap-screen provenance and raw input hashes are checked before training. All new output is versioned separately. Inner-fit JSONs and completed outer-fold predictions support resumption. Changed settings, source weights, input data, runtime or trainer code reject reuse. Report validation rechecks every fit, classifier/encoder initializations, target membership, training-only stopping, rate selection, shared sampling, prediction hashes, pooled patient probabilities and reported metrics.

## Run and monitor

```bash
.venv/bin/python -m mi_localization.overlap_screen
.venv/bin/python -m mi_localization.transfer
.venv/bin/python -m mi_localization.transfer_report
# Read-only checkpoint progress (check process separately for liveness):
.venv/bin/python -m mi_localization.transfer --status
tail -f tmp/transfer_vcg_run.log
# Optional end-to-end runner; --publish commits/pushes aggregate results only
# after report validation and only on main with a clean tracked working tree:
caffeinate -i /bin/zsh scripts/run_transfer_vcg.sh --publish > tmp/transfer_vcg_run.log 2>&1
```

Reports go to `results/neural_transfer_v1/complete/`. Partial-fold experiments use separate directories and cannot produce the final report. Pair identities, per-record caches, patient predictions, raw waveforms and fitted weights stay local and ignored; public reports contain aggregate metrics/provenance.

## Interpretation

The primary comparison is mean patient macro F1 across all three prespecified target seeds, with sample SD and a class-stratified paired patient bootstrap using the same patient draws across both arms and all fixed seeds. Patients are not multiplied by the number of seeds. Intervals are descriptive and conditional on fixed predictions and one source encoder, with no training refits, seed resampling or adjustment for correlated folds, repeated development or multiple comparisons.

Kors-derived source XYZ differs from measured Frank XYZ. Better reconstruction can fail to transfer. PTB folds have been repeatedly inspected; cross-dataset identity remains unresolved even after the bounded recording screen. Neither this experiment nor a favorable score establishes independent patient-level confirmation, clinical validity, novelty or SOTA. No 0.80–0.90 macro F1 is promised. Full-record denoising remains offline/noncausal and seeded MPS is not guaranteed bitwise deterministic.

Cross-correlation reference: [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html). Source dataset: [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/).
