# Next research direction: healthy-reference and ordered-trajectory VCG tensors

Research review: 15 November 2026. **Proposal only: no new model has been trained by this review.** Keep measured Frank XYZ primary; ECG-assisted training remains a separate extension. The existing chronological/geometric roadmap remains available, rather than being a completed experiment.

## Why change the representation question?

The completed [TabPFN benchmark](../results/tabpfn_v2_v1/complete/REPORT.md) reached patient macro F1 **0.542**, versus matched bootstrap-XGBoost **0.513**. Its paired descriptive difference interval includes zero. The historical waveform model reached **0.550 ± 0.040** across three seeds under a different procedure. Finer tensor compression improved reconstruction without a consistent classification gain. Neither result supports a claim of 0.80–0.90 or SOTA.

Two specific representation hypotheses remain untested:

1. Comparing a recording with a small healthy reference library may expose useful patterns beyond its own-record template/residual features.
2. Explicit ordered interactions in a signed XYZ trajectory may add information that zero-lag covariance and simple geometric descriptors do not capture well. Existing waveform/DCT inputs already encode time; the hypothesis concerns a useful inductive bias, not information absent from every current model.

Do not merge these into a large architecture before testing the parts.

## A. First experiment: healthy-reference deviation features

### Research basis and novelty boundary

[Yan et al., *Deviation-Aware Pretraining for Cardiovascular Disease Detection from Electrocardiograms*, submitted 10 November 2026](https://arxiv.org/abs/2611.14141), propose normal-morphology codebooks and deviation-guided cross-lead representation learning with large-scale ECG pretraining. This is recent preprint evidence, not established performance on our cohort. No downloadable model implementation/checkpoint was verified in this review.

Our proposed adaptation uses measured XYZ, a small training-only reference library, signed multiscale deviations, and small-sample classifiers. It is **not** their Mamba model or a reproduction of their results. Healthy-reference learning itself is prior art. The potential contribution is the adaptation and evidence for patient-independent MI subtype localization; priority or publishable novelty has not been established by this search.

### Representation

Start from existing 391 recording templates, retaining all 164 patients and their original labels. Form patient summaries by giving each recording equal weight, so a patient with many beats does not dominate the atlas. Use the original R alignment, physical mV and signed XYZ coordinates. Initially use denoised XYZ only, without DTW, arbitrary rotations, per-beat amplitude normalization, new delineation or filtering patients.

Fit exactly two healthy medoids to reference-patient morphology in a fixed DCT-32 XYZ representation. Use deterministic initialization/ties and preserve the corresponding physical waveform summaries. Two prototypes are a deliberately small first hypothesis, not a tuned assertion that two normal cardiac types exist.

For query recording template x and reference waveform h_k, construct:

```
deviation[k, XYZ, band, time] = wavelet_band(x - h_k)
axes: recording × reference(2) × XYZ(3) × band(5) × time(651)
```

Use the established sym4, four-level additive wavelet bands and four fixed R-relative windows. These windows are not independently measured clinical QRS/ST/T boundaries. Retain signed means and RMS values: 2 references × 3 axes × 5 bands × 4 windows × 2 summaries = **240 features per recording**. Add the two prototype distances and a fixed nearest-distance margin; aggregate recordings equally at patient level. Keep the existing morphology features as an optional, separately declared combination arm. Select 24/64 columns within training data using the established selector; do not globally cache atlas-derived features across folds.

Simply subtracting one fixed healthy mean and applying a linear DCT is an affine transformation. After centering it can be equivalent to the original features. Useful differences must come from nonlinear distance/energy terms, reference structure or interactions, and must survive controls. Residuals are differences from a reference, not proven infarct tissue, noise-free signals, or a patient's counterfactual healthy heart.

### Avoid self-reference and label leakage

The reference fit uses HC labels, so this is **supervised reference construction**, not label-free pretraining. A healthy training patient's own waveform must not become its reference while an unseen patient's cannot.

Use a predeclared three-member reference ensemble inside each relevant training partition:

1. Stratify its patients into three disjoint reference parts using a fixed seed.
2. For member j, fit the healthy atlas from HC patients in part j only. Fit the classifier and its feature selector on patients in the other two parts only. All feature normalization/ranking and any reference-derived scales stay within their allowed subsets.
3. Predict the untouched inner validation or outer test patients with each member, then average the three probability vectors. Every original training patient participates in classifier training in two members; no cohort member is removed from evaluation.

Build this ensemble afresh for every inner/outer training partition. Never use an outer-training atlas inside its inner validation fits. References may be reused across that member's classifier bootstrap draws because they are a separate, fixed reference subset; classifier bootstrap stability then does not measure uncertainty in the atlas. Assess reference-partition variability separately.

Feasibility check on the frozen folds: **46–47 HC patients** per outer-training set; **30–32** per inner-training set; **10–11** HC patients per reference part at the inner-training stage. Those are small reference libraries. This strict design reduces classifier rows per member, which may outweigh any representation benefit.

### Required controls

| Arm | Purpose |
|---|---|
| Fresh established full-training morphology procedure | Practical reference against the current feature model |
| Same three-part ensemble, morphology only | Controls the smaller classifier sets and probability averaging |
| Same ensemble, morphology + healthy deviation features | Tests the additional reference representation |
| Same ensemble, morphology + size-matched mixed-class reference features | Tests whether healthy specificity helps beyond a generic reference-distance expansion |

For the mixed-class control, sample the same number of reference patients from the same reference part with a fixed policy independent of model scores, allowing its natural class mix; use identical medoid count and feature construction. No classifier patient is in its member's reference set in any arm. Keep classifier family, selector budget and inner search space matched between the three ensemble arms. Choose one classifier family for the first controlled study rather than immediately searching several; pinned raw TabPFN-v2 is a reasonable starting point, through the existing isolated local backend.

Use frozen outer folds and the established three inner patient folds, with all decisions inside training contexts. Repeat reference-split seeds [42, 123, 2026] and report each, not a chosen seed. Report six-class macro F1, every class F1, and MI-only five-class macro F1 computed on the original MI patients without changing their six-way predictions. This helps distinguish better HC separation from better localization. These are development comparisons, not independent confirmation.

## B. Second experiment: signed, time-augmented path-signature tensors

[Feng et al., *PathFusion-Net*, accepted manuscript listed by UCL (2026)](https://discovery.ucl.ac.uk/id/eprint/10216614/) combines signature/path-development methods with neural ECG arrhythmia classification under an inter-patient split. That motivates a representation experiment; its arrhythmia task and accuracy do not establish our MI macro F1. Signature methods for ECG are already prior art.

Our adaptation would treat each measured trajectory as (time, X, Y, Z). Truncated signatures encode ordered iterated interactions; antisymmetric second-order terms describe signed areas, while higher orders describe additional ordered relationships. Time augmentation retains timing relative to the explicit clock. Translation invariance of increments would lose absolute offsets, so preserve physical waveform means/levels as parallel features. Do not impose spatial rotation invariance or arbitrary axis rotations.

Start with levels 1–3 on the whole beat and fixed windows, using fixed unit conversions for time/mV and training-only feature scaling. A four-coordinate path has 4 + 16 + 64 = **84 raw signature coordinates** through level three, with algebraic redundancies; do not interpret them as 84 independent biomarkers. Compare with existing morphology and matched simple signed-area/geometry features. Use linear/shallow models before adding a large neural encoder. Audit sensitivity to denoising, sampling and small R jitter; high-order terms can amplify noise.

This differs from our completed geometry audit through higher-order, explicitly ordered interactions. It is not guaranteed to add useful signal. Test it alone before adding it to the healthy-reference tensor.

## C. Only after the feature hypotheses: chronological neural context

The [existing roadmap](modern_architecture_roadmap.md) already specifies true contiguous beats from one recording versus a matched shuffled-order control. This has not been trained. It remains a separate experiment, rather than justification to resume the paused multiview sweep automatically.

[PACE-ECG, submitted 9 November 2026](https://arxiv.org/abs/2611.12307), studies temporally neighbouring recordings from the same patient for pretraining and external prognostic transfer. It is relevant to objective design, but **longitudinal recordings are not consecutive beats**, and its prognosis results do not prove that beat chronology will improve MI localization.

## Decision rule and immediate next action

Implement the healthy-reference transformer and reference-disjoint ensemble first, with synthetic membership/self-reference checks and a finite local inference smoke test. Then run the complete declared comparison; do not interpret an attractive partial fold as a gain. Preserve source modules/manifests from completed studies and use a new experiment/output namespace.

Proceed to combination only if the new representation improves its matched ensemble control consistently across seeds and the benefit includes MI subtypes. If healthy references and mixed references perform similarly, describe generic reference features rather than a healthy-specific mechanism. If there is no gain, keep the result and move to the path-signature comparison. A larger tensor or more faithful reconstruction is not itself success.

All repeatedly inspected PTB folds remain development data. Seek a label-compatible independent cohort for final confirmation. The target 0.80–0.90 remains an aspiration, not a score forecast or a reason to change patients, labels or evaluation rules.
