# Data fidelity and within-record tensor exploration

## Evidence before another model

The existing dataset has 164 independent patients, not 55,863 independent cases. Its learning curve improved from 0.407 at 25% training patients to 0.550 at 100%, under the declared fixed sampling/training policy. It does not establish a dataset ceiling or a sample-size requirement for 0.80–0.90.

Our automatic phase estimator passed strict multi-axis checks for only 23.2% of beats, with class-dependent coverage. Its feature screen did not establish a gain. Raw-vs-denoised, longer windows and prior learned tensors also did not establish consistent gains. Further tensor work should therefore measure information preservation and compare specific representations, rather than equate extra axes or a smoother waveform with better data.

The new [within-record feasibility audit](../results/neural_record_tensor_v1/audit/REPORT.md) examines all 391 records and 164 patients without changing source data or ongoing training. Median patient-level QRS cosine agreement with a cross-beat template was **0.9951** before new alignment. The median selected absolute shift was **0 ms**; the median patient's record-median fraction preferring shifts of at least 10 ms was **12.5%**. This does not diagnose missed peaks or quantify clinical quality. The same score selects the shift, so a higher post-shift agreement is optimistic by construction.

The audit uses odd/even chronological beats as disjoint query/reference halves within each recording, a common XYZ shift in ±20 ms, no rotations/scaling/time warping, and a shared comparison core. It supplies no external clean-signal or clinical boundary ground truth. A whole-beat residual may contain real disease-related variation as well as acquisition artifacts.

## Proposed first tensor

Use one recording at a time, preserving its actual chronological order:

`T[beat, view, XYZ, original_time]`

For a measured denoised beat X and that recording's median template M:

- View 1: original full 651-sample waveform X.
- View 2: M broadcast across beats, representing the stable component.
- View 3: residual R = X − M, retaining beat-to-beat differences.

X = M + R mathematically; the prototype's reconstruction error was below 1.5e−8 mV. The representation is redundant and creates no new sensor information. Its hypothesis is that a finite model may learn stable morphology and variability more easily when exposed to both explicitly. It is not robust PCA, a verified noise decomposition or established novelty.

The original view remains because templates can blur rare abnormalities or differences in T timing. The residual remains because variation is not necessarily noise. Templates are estimated from the same recording's unlabeled input only, never from another patient or a label-conditioned cohort. Using all beats of a test recording is offline within-exam processing, not evidence of streaming prediction. Any cohort-learned basis, template dictionary, quality threshold or scaling must be fitted inside the relevant training fold.

Here X is the **existing denoised measured-VCG beat**, not independently known clean physiology. The raw source is retained separately. A subsequent raw-aware version could have five views: raw, current denoised, recording template, denoised−template, and raw−denoised. This would expose both beat variation and the component removed by preprocessing, without declaring either residual to be noise. That five-view bank/classifier is not implemented or evaluated by this three-view feasibility prototype. It requires its own duplicate-input capacity control and raw+denoised intermediate ablation; the earlier raw-only study does not answer whether the views are complementary.

This differs from the previous patient tensor: those bags were unordered, potentially combining recordings, and their views were learned temporal filters. This proposal uses recording-specific stable/residual context. A later chronological sequence model must use contiguous beats from one recording, include padding masks/RR context, and avoid falsely treating arbitrary bag order as time.

## Controls that distinguish causes

| Model/input | Question |
|---|---|
| Original waveform reference | Where do we start? |
| Duplicate-waveform views with the new architecture | Does extra channel/model capacity alone explain a gain? |
| Recording template only | Does averaging morphology help or remove useful variability? |
| Original + template + residual | Does explicit stable/variable decomposition add value? |

The prototype is implemented; these classifier comparisons have **not** been trained. Small temporal shifts and quality weighting should be separate later ablations, not silently bundled into the new tensor. Alignment should initially use a common XYZ translation in time with a narrow declared range; preserve physical amplitude, coordinate direction, duration and estimated shift. Avoid per-axis alignment, arbitrary spatial rotations or unrestricted temporal warping, which may alter useful structure. Compare before/after morphology and classification independently.

## What high-quality data means here

1. **Acquisition integrity:** units, lead identity, synchronous sampling, encoded clipping/flatline candidates, source hashes and original record/patient membership. Many consistency checks already pass; this does not certify diagnostic quality.
2. **Morphology fidelity:** retain the original branch when testing denoising, averaging or compression. Assess changes in low-amplitude waveform structure, timing, vector orientation and rare beat variation, not merely reconstruction MSE or smoothness.
3. **Boundary reliability:** carry missing/uncertain estimates explicitly; do not present current adaptive markers as clinical truth or require successful delineation to keep a patient in evaluation.
4. **Annotation reliability:** obtain independent expert review of a class-balanced sample, with error-selected cases analyzed separately. Matching parsed headers is not clinical adjudication. Source labels should never be automatically changed to agree with predictions.
5. **Independent diversity:** add genuinely new, appropriately labeled patients/sites when possible. Synthetic/noisy copies add views, not independent patients. A new split of the already inspected cohort is not new validation.

Signal quality indices can confuse pathological morphology with artifacts; rare beats and hard patients should not be discarded to raise the score. A first quality experiment should retain every patient and use any weighting only under a declared training-fold policy. All-beat evaluation and class-stratified coverage/weight diagnostics are necessary controls. A smooth template may have high apparent SNR while losing diagnostically useful variability.

## Research references and limitations

- [Åström et al., VCG loop alignment](https://pubmed.ncbi.nlm.nih.gov/10763295/): spatial/temporal alignment and morphology interact; this motivates measuring fidelity, not assuming alignment helps localization.
- [Morphology-preserving ECG denoising](https://pmc.ncbi.nlm.nih.gov/articles/PMC11100958/): emphasizes preserving low-amplitude content and inter-beat variation; its performance is not evidence for this VCG task.
- [Assessing SQIs on pathological ECGs](https://pubmed.ncbi.nlm.nih.gov/27454007/): shows that artifact discrimination can depend on pathology; applying ECG SQIs to our VCG task would itself require validation.
- [LVCG preprint](https://arxiv.org/abs/2605.31249) and [official code](https://github.com/BosonHwang/LVCG): a separate later route for physically structured representation learning at larger scale. Its ECG-to-latent-VCG setup, datasets and metrics differ from measured-Frank six-class localization. A checkpoint requires provenance/overlap checks before use; no transfer gain is inferred here.

## Reproduce the feasibility audit

```bash
.venv/bin/python -m mi_localization.record_tensor_audit
.venv/bin/python -m pytest tests/test_record_tensor.py -q
```

Config: `configs/record_tensor_audit.yaml`. Aggregate results: `results/neural_record_tensor_v1/audit/`. Per-record/patient diagnostics and the example prototype remain local under ignored `review/`. The shared-label experiment continues unchanged. No classifier has been evaluated on the prototype tensor; no macro-F1 gain, clinical quality certification, novelty or SOTA is claimed.
