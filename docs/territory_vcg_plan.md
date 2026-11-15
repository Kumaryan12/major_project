# Shared-label auxiliary supervision: isolated VCG test

## Why this next step

The completed geometry screen did not establish an improvement over its own controls: linear adaptive geometry was 0.467 versus statistics 0.445 (difference interval −0.061 to +0.107), and tree statistics+geometry was 0.461 versus statistics 0.409 (−0.024 to +0.131). Strict automatic phase consensus covered only 23.2% of beats. These findings do not justify stacking an unvalidated geometry branch into the neural model or claiming a breakthrough.

The separate proposed objective can be tested without any new boundaries. The hypothesis is that sharing label descriptors may help the small six-class task: anterior examples occur in AMI/ALMI/ASMI, while lateral examples occur in ALMI/ILMI. This is an experimental inductive bias, not additional clinical ground truth or a claim of novel mathematics.

## Original endpoint and derived targets

The original six-way classifier and evaluation stay unchanged. A four-output auxiliary head predicts binary **mentions derived from the source label names**:

| Source label | Anterior | Inferior | Lateral | Septal |
|---|---:|---:|---:|---:|
| AMI | 1 | 0 | 0 | 0 |
| ALMI | 1 | 0 | 1 | 0 |
| ASMI | 1 | 0 | 0 | 1 |
| IMI | 0 | 1 | 0 | 0 |
| ILMI | 0 | 1 | 1 | 0 |
| HC | 0 | 0 | 0 | 0 |

A zero denotes absence from this encoding, **not independently verified absence of anatomical involvement**. These targets cannot fix clinical annotation ambiguity. The source six-class labels are never merged, changed or replaced with auxiliary predictions. At inference, the original six-class logits alone determine the prediction.

## Matched comparison

Both arms use the same original 651-sample denoised measured-XYZ input, residual encoder, six-class classifier and four-output auxiliary linear head. Production parameter count is **78,458 in both arms**, 260 more than the original model. Both have the same initial trainable tensors, fresh head, target-training-only normalization and shared-epoch sampled beats. The extra head is initialized inside a CPU RNG fork, preserving the original model/dropout stream. A CPU regression test verifies that the zero-weight control reproduces the original trainer's encoder/classifier weights and CE history exactly; MPS bitwise identity is not promised.

```
Control:       loss = CE(six_class_logits, source_label) + 0.0 * BCE(auxiliary, mentions)
Experimental:  loss = CE(six_class_logits, source_label) + 0.2 * BCE(auxiliary, mentions)
```

BCE is averaged across the four descriptors and sampled beats. Both branches share the encoder. No additional class weights are applied inside either loss: the existing sampler already balances original classes and patients within class. Control BCE is diagnostic and contributes zero gradient. The experimental coefficient is fixed at 0.2 before observing this run; there is no outer-selected auxiliary strength, projection, hierarchy or ensemble.

Unchanged optimizer/budget: AdamW 0.001, decay 0.01, dropout 0.2, gradient clipping 5, batch 128, 8,192 beat draws per epoch, max 25 epochs and patience six. Three inner patient folds, fixed across arms and seeds, select epochs using **six-class patient macro F1**. Rounded median best epoch sets a fresh outer-training refit with no outer-test monitoring. All held-out beats are averaged to one patient probability vector.

Same 164 patients, 391 recordings, 55,863 beats, original R positions, class order and ten frozen outer folds. Training seeds [42, 123, 2026] are all reported. Total: **240 fits** = 3 seeds × 10 folds × 2 arms × (3 inner fits + 1 refit). No geometry features/tensors, delineation, ECG, external source, pretraining, augmentation or label adjudication is included. This isolates the objective from the geometry hypothesis.

## Validation, progress and artifacts

```bash
.venv/bin/python -m mi_localization.territory_study
.venv/bin/python -m mi_localization.territory_report
.venv/bin/python -m mi_localization.territory_study --status
tail -f tmp/territory_vcg_run.log
# Combined run with aggregate publication after full report validation:
caffeinate -i /bin/zsh scripts/run_territory_vcg.sh --publish > tmp/territory_vcg_run.log 2>&1
```

Configuration: `configs/territory_vcg.yaml`. Separate outputs: `results/neural_territory_v1/complete/`. Completed inner fits and outer predictions support resumes; an interrupted individual fit restarts from its seed. Changed model/config/data/runtime fingerprints refuse incompatible checkpoint reuse. Final validation checks all seed/fold histories, coefficients/objective arithmetic, initialized weights, normalization, sampled-row sequences, six-class stopping and pooled prediction metrics.

Every seed and mean/sample SD are reported, together with a paired patient bootstrap using identical class-stratified patient draws across both arms and all fixed seeds. Repeated seed/patient predictions are not independent patients. Intervals are descriptive and conditional on fitted predictions, with no model/seed resampling or corrections for correlated folds, repeated development or multiple experiments. The repeatedly inspected PTB cohort remains development data, not independent confirmation.

Any improvement still needs a fixed new-cohort validation. No 0.80–0.90 macro F1, anatomical correctness, clinical utility, novelty or SOTA is promised. No difficult patients/labels are changed. Full-record denoising remains offline/noncausal; seeded MPS is not guaranteed bitwise deterministic. Individual predictions, fitted weights and waveform caches stay local; only aggregate reports and fit audits are public.
