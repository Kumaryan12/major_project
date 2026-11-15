# Modern architectures for the next VCG experiments

Research review: 15 November 2026. This document records architecture proposals and feasibility findings; experiment completion is tracked separately in the follow-up below. It does not establish a SOTA model or start training by itself.

Follow-up status: the [TabPFN-v2 feature benchmark](tabpfn_v2_plan.md) has now completed, reaching 0.542 macro F1 versus matched XGBoost 0.513, with a descriptive difference interval including zero. The chronological/geometric neural directions below remain proposals. This is not a new SOTA claim.

## Recommendation

Use a pretrained tabular foundation model as the immediate benchmark on our tensor-derived features. For the main neural research contribution, test a compact, pretrained geometric and chronological VCG model in controlled stages. Keep measured Frank XYZ as the primary input. ECG-assisted or ECG+VCG experiments remain a separate extension.

An architecture can use modern methods before we know whether it performs well on this task. Calling the resulting MI model state of the art requires comparable patient-disjoint results against appropriate baselines, ideally on an independent cohort. A model's performance on other diagnoses, AUROC or accuracy is not our six-class patient macro F1.

## Evidence from this project

- Only **164 independent patients**, despite 55,863 beats. The [learning curve](../results/neural_learning_curve_v1/complete/REPORT.md) rose from 0.407 at the smallest patient subset to 0.550 with all training patients under its fixed policy. Full training resubstitution was about 0.970. This supports concern about generalization, without proving a dataset ceiling or a required sample count.
- Plain VCG ResNet: **0.550 ± 0.040** across three training seeds. [Report](../results/neural_repeat_v1/REPORT.md).
- Bootstrap-selected features + shallow trees: **0.513**, versus matched tree control 0.420, one fixed resampling/model-seed policy. [Report](../results/feature_stability_v1/complete/REPORT.md). This promising feature-pipeline gain still needs resampling-seed checks.
- Finer tensor compression improved reconstruction but did not improve localization consistently. [Report](../results/tensor_refinement_v1/complete/REPORT.md).
- Unordered patient bags and learned Tucker/attention were already tested; they did not beat the matched waveform reference. [Report](../results/neural_patient_tensor_v1/complete/REPORT.md). A chronological model must use real within-record order, not rename these bags as sequences.
- The source reconstruction pilot used only 1,024 PTB-XL training patients. Its transferred encoder reached 0.520 versus matched scratch 0.553 across target seeds. [Report](../results/neural_transfer_v1/complete/REPORT.md). This rejects that tested transfer recipe, not all large-scale pretrained encoders. Derived source XYZ and measured Frank XYZ differ, and source–target patient identity remains unresolved.

## 1. Immediate benchmark: TabPFN on the existing features

[TabPFN, Nature 2025](https://doi.org/10.1038/s41586-024-08328-6), is a pretrained transformer for small tabular prediction tasks. Its synthetic-task pretraining lets it infer predictions from a labeled training table rather than train a large task-specific network from 164 patients. This motivates a direct experiment; it does not establish a score on our VCG task.

Use the existing morphology/combined feature pools and training-only bootstrap selector, with feature budgets 24/64. One row remains one patient. A matched bootstrap-XGBoost control isolates the classifier change. Keep checkpoint, inference settings and candidate count fixed; select feature family/budget inside inner patient folds. Classifier context must contain only training patients and labels. Audit the backend's preprocessing, class weighting support, class order and inference behavior before evaluation; do not assume another estimator's preprocessing defaults apply.

The [official repository](https://github.com/PriorLabs/TabPFN) currently defaults to TabPFN-3.5 and provides explicit older-version selection. Recent weights require model-specific terms/access; v2 has a different attribution license. Pin the chosen checkpoint/version and run locally. Newer must not silently replace older weights during a resume. No external prediction API or patient-feature upload is required by the proposed benchmark.

[TabICLv2](https://arxiv.org/abs/2602.11139) is a possible second tabular foundation-model comparison. However, the [official README](https://github.com/soda-inria/tabicl#faq) says its authors have not tested generalization below 300 samples, and our train folds have fewer than 164 patients. Its released checkpoint supports inference, but it is not the best-evidenced first choice for our sample regime. Do not duplicate patients to make the table appear larger.

This track tests whether the features can support stronger nonlinear prediction with a broadly pretrained classifier. It provides a strong baseline, not architectural novelty specific to VCG.

## 2. Main neural direction: chronological VCG representation learning

[LVCG (2026)](https://arxiv.org/abs/2605.31249) learns cardiac representations in latent VCG space from 12-lead ECG. Its beat morphology and inter-beat temporal modeling are directly relevant to our representation question. The [released architecture description](https://github.com/BosonHwang/LVCG/blob/main/docs/ARCHITECTURE.md) uses a StateGRU temporal module by default; a modern model need not use a large Transformer everywhere.

The repository describes training and downstream use of `final.pt`, but I did not verify a public pretrained-weight download from the README. Code availability alone is not checkpoint availability. Its inputs are ECG-derived latent VCG, not our measured Frank XYZ; dimensional equality is not proof of aligned coordinates/distribution. Source weights/data and target adaptation need verification before claiming pretrained transfer. The architecture notes also describe a difference between paper and released embedding pooling, so any reproduction must pin the revision.

The following is **our proposed adaptation**, not the published LVCG model or an established novel/SOTA result:

```
measured XYZ with original R positions
        |
contiguous beats from one recording + RR intervals + masks
        |
waveform encoder  ||  direction-preserving trajectory encoder
        |                         |
        +------ gated feature fusion ------+
                         |
              compact chronological module
                         |
               recording representation
                         |
          patient aggregation and six-class head
```

Represent input as **record × ordered beat × physical view × XYZ × original time**. Use the original signal plus declared median/residual views as optional ablations. A sequence never mixes different recordings or patients. Retain original R timing, RR intervals, amplitude and XYZ direction. Handle short sequences with masks. At inference, cover all eligible beats/recordings with a declared aggregation rule; no most-confident-window selection.

Begin with the existing compact waveform encoder and a small GRU over true contiguous beat embeddings. Compare it with an order-shuffled control using the same records, parameters and sampling budget. This tests chronological information separately from capacity. Learned temporal context differs from the prior unordered MIL experiment. A pretrained encoder can be added only after its actual source/checkpoint compatibility is verified; target-trained weights from a fold must never initialize that fold's inner validation models.

Then test a separate trajectory branch. Keep signed XYZ coordinates and time indices, with optional physical-time derivatives. A graph/point encoder can model local trajectory relationships; include temporal adjacency so loop crossings do not erase phase order. Preserve absolute orientation and amplitude in the input. Pure rotation-invariant descriptors or arbitrary spatial rotations are not assumed valid for infarct localization. Geometry-derived values must be computed before any scaling that would distort the intended physical relationship, with learned scaling fitted on training patients only.

Use a small gated fusion first; test low-rank tensor interactions only as a subsequent ablation. Do not repeat the earlier failed Tucker/attention stack and claim it is a new representation. Do not impose one shared decision/feature per corresponding view: the recent hard semantic-group constraint hurt performance, and view complementarity remains a hypothesis to learn rather than force.

The [2026 3D point-cloud MI localization paper](https://doi.org/10.1016/j.asoc.2026.114896) describes VCG point-cloud encoding with multiscale residual and temporal-attention components. This is a direct architectural comparator. I could verify the publisher's title/abstract/highlights through indexed material, but full-text access was blocked and I did not verify released code. Its reported accuracy cannot be treated as our six-class patient macro F1; cohort, labels, splits and aggregation must be checked before a replication or SOTA comparison.

## 3. Separate ECG-assisted extension

[ECG-FM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12530324/) supplies an open pretrained ECG encoder, with [released weights](https://github.com/bowang-lab/ECG-FM). It is a potential frozen-feature benchmark or teacher on paired 12-lead ECG, followed by a small target head/adapter. Its standard-ECG input semantics differ from XYZ: do not put VX/VY/VZ into arbitrary ECG slots and claim an exact pretrained model.

The paper reports PTB excluded from its pretraining collection but PTB-XL included. Dataset-name exclusion does not resolve the project's existing patient-identity uncertainty. PTB-XL cannot be called an untouched external test for an encoder pretrained on it. If distilling ECG knowledge into a VCG-only student, explicitly label this ECG-assisted training even if inference uses only VCG. Keep this route separate from primary measured-VCG-only claims.

## Experimental order and success criteria

1. Confirm the promising bootstrap-tree procedure across sampling seeds while retaining its fixed feature pool and selection policy. Do not select a favorable seed.
2. Benchmark pinned local TabPFN on the same patient feature task, with matched feature selection and controls. Start with a finite inference/checkpoint smoke test; report all planned outer folds before claiming a gain.
3. Test real chronological context versus a matched shuffled-order control on measured VCG. Keep geometry/pretraining out of that first temporal ablation.
4. Verify the chosen pretrained source/checkpoint and source–target overlap, then compare scratch versus pretrained versions with matched target tuning budgets.
5. Add the direction-preserving trajectory branch only as a separately controlled comparison. A final combined architecture is justified by these ablations, not by its component names.

Use the six original classes, patient-level macro F1, frozen outer folds and training-only normalization/stopping/selection. Repeat neural models across the existing three training seeds and report all class F1 values. These folds are repeatedly inspected development data. Preserve an independently sourced, label-compatible cohort for final confirmation when available. More public unlabeled ECGs may support pretraining, but do not create more independently labeled target patients or solve label mismatch automatically.

Success is a reproducible improvement over a matched control and stronger generalization evidence. A particular architecture does not make 0.80–0.90 achievable by guarantee, and this review has not established a SOTA model.
