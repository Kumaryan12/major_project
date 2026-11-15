# Information-preserving measured-VCG tensor comparison

## Decision from completed output

The [shared-label study](../results/neural_territory_v1/complete/REPORT.md) completed all 240 fits: CE **0.550 ± 0.040**, auxiliary descriptors **0.538 ± 0.026**, mean paired difference **−0.012**, descriptive interval **[−0.056, +0.032]**. AMI/ALMI losses offset ILMI/HC gains. It did not establish an overall improvement. Use ordinary six-class CE for this next experiment.

The [record audit](../results/neural_record_tensor_v1/audit/REPORT.md) found strong existing within-record QRS agreement (median patient statistic 0.9951); this is neither clinical quality nor SNR. Strict automatic phase consensus covered only 23.2% of beats in the separate geometry audit. Therefore test raw/template/residual information first, with temporal alignment and quality weighting reserved for later ablations. Do not combine several unvalidated changes into one model.

## Five physical views

For each original 651-sample measured Frank XYZ beat:

| View | Definition | Intended information |
|---|---|---|
| R | Existing unfiltered matched crop | Measurements before wavelet shrinkage |
| D | Existing denoised crop | Established baseline signal |
| M | Coordinate-wise median of D beats in the same recording | Stable within-exam morphology |
| D−M | Beat residual | Variation around that recording's template |
| R−D | Filter residual | Information removed by the existing filter |

The conceptual tensor has shape **55,863 × 5 × 3 × 651** (beat × view × XYZ × time). Views are constructed on demand from the two existing waveform banks, 391 cached templates and a beat-to-record mapping. Training does not require a separate multi-gigabyte five-view array. Raw and denoised caches retain their original data and hashes.

Templates use every unlabeled beat in **that recording only**, including a held-out recording's own beats at inference. No training/test patients, separate recordings, class-conditioned dictionaries or cohort statistics are mixed into a template. This is offline within-exam processing, not causal real-time prediction. It adds record context but creates no new patients. Both residuals can contain pathology and are not labeled noise.

The [bank audit](../results/neural_multiview_v1/bank_audit/REPORT.md) checks matching labels, patient rows, original beat assignments, source hashes, finite values, template membership and reconstruction: M+(D−M)=D and D+(R−D)=R within float32 tolerance. Maximum observed errors are 9.54×10⁻⁷ mV and 3.73×10⁻⁹ mV respectively. Reconstruction proves data preservation, not clinical accuracy or denoising quality.

## Six predefined arms

Each bracketed slot contains XYZ channels in the stated order.

| Arm | Input slots | Parameters | Purpose |
|---|---|---:|---|
| vcg | D | 78,198 | Fresh original baseline |
| capacity_control | D, D, D, D, D | 81,078 | Expanded-input control without added views |
| raw_denoised | R, D, D, D, D | 81,078 | Restore raw measurements |
| template_only | M, M, M, M, M | 81,078 | Test template averaging alone |
| template_residual | D, D, M, D−M, D | 81,078 | Record context without raw restoration |
| full_tensor | R, D, M, D−M, R−D | 81,078 | Combined representation |

The four non-template expanded arms form a controlled raw-restoration × record-context comparison. The full representation also explicitly exposes the deterministic filter residual. All five-view models share exactly the same initial tensors, body, classifier and parameter count. Their stem copies the original XYZ kernel into view two and initializes the other view weights to zero. Except template_only, they start with D as their active input. CPU RNG initialization is preserved; original classifier hashes match across all arms. A CPU regression test checks exact equality of the new original-input trainer with the frozen trainer. Seeded MPS is not promised bitwise deterministic.

The five view/lead axes are flattened to 15 channels for the first convolution. This stage tests input representation and record context; it is **not a learned Tucker contraction or chronological sequence model**. Residuals are deterministic linear combinations, so benefits would reflect context/inductive bias, not newly created sensor information. Duplicate channels also change optimization and effective first-layer updates: matching capacity does not make the control optimization-neutral. Hence retain both the duplicate-input control and original three-channel baseline. Expanded inputs add compute despite equal training budgets.

## Fixed evaluation and comparisons

Keep all **164 patients, 391 recordings, 55,863 beats**, six original classes, original R positions, ten frozen patient folds and seeds **42, 123, 2026**. No patient/beat deletion, label replacement, geometry, auxiliary targets, ECG, augmentation, external pretraining, temporal realignment, rotation or quality weighting enters this stage.

Same AdamW rate 0.001, weight decay 0.01, dropout 0.2, gradient clipping 5, batch 128, 8,192 balanced beat draws per epoch, max 25 epochs and patience six. Sampling balances original classes and patients within class. Three fixed inner patient partitions select epochs using six-class patient macro F1. Their rounded median best epoch determines a fresh outer-training refit without outer monitoring. Mean probabilities across all held-out beats give one prediction per patient. Total: **720 fits = 3 seeds × 10 outer folds × 6 arms × (3 inner fits + 1 refit)**.

Each physical view/lead uses weighted **training-partition-only** mean and scale. D moments exactly follow the old normalizer. Sharing these moments across arms and seeds is valid because the train rows and weights are identical; a guarded cache avoids repeated full-bank scans. Final reporting independently recomputes every training partition's moments. Separate residual standardization can affect low-amplitude variation and optimization; it is not physiological amplitude normalization. No inference exam contributes to fitted normalization.

Eight prespecified descriptive contrasts (second minus first):

1. Original vs full tensor.
2. Capacity control vs full tensor.
3. Raw+denoised vs full tensor.
4. Capacity control vs template+residual.
5. Template+residual vs full tensor.
6. Capacity control vs raw+denoised.
7. Capacity control vs template only.
8. Template only vs full tensor.

Report every seed, mean/sample SD and class F1. Paired class-stratified patient bootstrap draws are shared across models and all fixed training seeds; patient×seed rows are not independent patients. Intervals condition on fitted predictions, without refits or adjustments for correlated training folds, multiple contrasts or repeated development. They are descriptive, not independent statistical confirmation. Favorable results require a fixed new-cohort validation; no 0.80–0.90 score, novelty, SOTA or clinical validity is promised.

## Run, resume and inspect

```bash
.venv/bin/python -m mi_localization.multiview_bank
.venv/bin/python -m mi_localization.multiview_study
.venv/bin/python -m mi_localization.multiview_report
.venv/bin/python -m mi_localization.multiview_study --status
tail -f tmp/multiview_vcg_run.log

# Combined resumable run with publication after full validation:
caffeinate -i /bin/zsh scripts/run_multiview_vcg.sh --publish > tmp/multiview_vcg_run.log 2>&1
```

Config: `configs/multiview_vcg.yaml`. Separate output: `results/neural_multiview_v1/complete/`. Source/config/runtime fingerprints reject incompatible resumes. Completed inner selections and outer fits resume; an interrupted individual fit restarts from its seed. Status prints saved progress, not proof of a live process; compare its PID and timestamp with the operating system. A partial-fold smoke run has a separate directory and cannot generate a complete report.

Reporting validates source hashes, frozen membership, training-only normalization, initialized weights, parameter counts, matched sampled-row histories, stopping budgets, saved model weights and complete pooled predictions before publishing the aggregate report. Source arrays, templates, record mappings, normalizer caches, weights, record review details and individual predictions remain local. Public artifacts are aggregate reports and fit audits. The publish wrapper refuses automatic commits if tracked work or the main branch changed during training.
