# Healthy-reference VCG deviation implementation

Implemented 15 November 2026. This is the first experiment in the [research plan](deviation_tensor_research_plan.md), not its later path-signature or chronological neural extensions. Implementation and smoke completion do not establish a classification gain.

## Completed result

All **1,960 fit contexts**, 100 outer arm/seed evaluations and the independent report have completed. Every outer model restored and every relevant bootstrap selector cache replayed. [Full validated report](../results/deviation_vcg_v1/complete/REPORT.md).

| Procedure | Patient macro F1 |
|---|---:|
| Full-training morphology, one fixed seed policy | 0.513 |
| Matched morphology ensemble, mean ± reference-seed sample SD | 0.499 ± 0.019 |
| Healthy-reference features, mean ± reference-seed sample SD | 0.479 ± 0.034 |
| Mixed-reference features, mean ± reference-seed sample SD | 0.463 ± 0.047 |

Healthy minus matched morphology: **−0.020**, descriptive paired patient interval **[−0.061,+0.017]**. The tested recipe did not demonstrate an improvement. Healthy references also did not establish superiority over mixed references: +0.017 [−0.022,+0.056]. Do not choose the favorable reference seed or infer that all healthy-reference methods fail. Preserve this result and move to the separately controlled [ordered-trajectory study](signature_vcg_plan.md).

## Frozen first comparison

Measured Frank XYZ, the same 164 patients, 391 recording templates, six original classes and frozen ten outer/three inner patient folds. Two feature budgets (24/64) are selected using mean inner patient macro F1, separately for each arm/reference seed. No outer-score selection.

- `full_morphology`: fresh full-training 480-column morphology procedure, with its two-budget inner search. Fitted once per fold; this is not identical to the previous morphology/combined family search that reached 0.542.
- `ensemble_morphology`: three reference-disjoint members with morphology only. Controls the smaller classifier contexts and probability averaging.
- `healthy`: the same ensemble with morphology plus healthy-reference deviation features.
- `mixed`: the same ensemble with morphology plus size-matched mixed-class reference features.

Reference-partition seeds are 42, 123 and 2026; all are reported. They change the reference partitions, not the fixed classifier/inference seed policy, and reuse the same patients. They are not independent cohort replications or extra training patients.

## Tensor and safeguards

Each member splits its training patients into a one-third reference part and a two-thirds classifier part, stratified by class. The atlas, selector and classifier never use validation/test patients. No classifier patient appears in its member's atlas. Every training patient participates in classifier fitting in two members; all original patients receive out-of-fold predictions.

Healthy medoids are fitted only to HC patients from the reference part. Mixed medoids use an equal number sampled from that same part, with identities selected independently of class labels/model scores. Both solve the exact two-medoids objective with squared Euclidean signed DCT-32 XYZ distances, deterministic patient-order ties, and no rotation/warping/amplitude normalization. Labels determine the healthy library and shared size; this is supervised reference construction.

Reference waveforms are equal-record coordinate medians of the corresponding patient's recording templates. A query record's deviation from each reference is decomposed into the existing five additive sym4 wavelet bands. Signed mean/RMS in four fixed R-relative windows gives 240 features, plus two DCT distances and an absolute distance margin: **243 additional columns**. Patient aggregation is an equal-record coordinate median. Windows are not clinical delineations and residuals are not verified pathology.

The original morphology pool stays in every arm. The established 64-resample bootstrap consensus selector, 32 separate audit resamples and Pearson redundancy pruning use classifier-training rows only. The reference part stays fixed during those classifier bootstrap draws: those draws do not measure atlas uncertainty. Ranking uses class-balanced training scaling; the frozen local TabPFN-v2 receives raw selected columns and applies its own training-context preprocessing. It uses the existing isolated worker backend, eight fixed inference members and raw probabilities, without inverse-prior correction.

Recording templates use each examination's own beats, so inference is offline. No new clinical labels, filtering, R detection or streaming claims are introduced. Source modules and manifests from completed studies are unchanged.

## Execution and progress

```sh
# Finite real-data backend and save/load check; no score claim.
.venv/bin/python -m mi_localization.deviation_study --smoke

# Resumes matching cached fits, then independently verifies the complete study.
zsh scripts/run_deviation_vcg.sh --publish

# Read saved counters; they do not establish whether a process is alive.
.venv/bin/python -m mi_localization.deviation_study --status

# For the background run launched from this session:
tail -n 12 tmp/deviation_vcg_v1.log
pgrep -fl 'mi_localization.deviation|run_deviation_vcg'
```

There are **1,960 numerical fit contexts**: 60 full-morphology inner + 10 outer; 1,620 ensemble inner + 270 ensemble outer. There are 100 outer arm/seed evaluations (ten full-morphology and ninety ensemble). Full-model weights are frozen; a PFN context fit is not gradient training of the foundation model. No completion-time estimate is assumed before observing throughput.

The runner stops on failure. Matching configuration/source/code hashes permit resume; changed study settings require a new output namespace. Arrays, patient predictions, selectors, medoid waveforms and fitted contexts remain local under ignored paths. Public artifacts contain aggregate results and hashed provenance. The publication option commits/pushes the new result directory only after the full independent report passes and the tracked working tree is clean.

## Verification and reporting

Synthetic tests exercise exact medoid selection, disjoint participation, query/classifier perturbation invariance of the atlas, signed feature layout, physical distances, equal-record aggregation, MI-only scoring, a complete nested study, cache resume, restored models and corruption rejection. A numerical stand-in is used only for PFN inference in the fast nested test; the separate real-data smoke exercises the actual pinned backend.

The final report reconstructs every reference/feature context, replays the relevant bootstrap selector caches, verifies every inner selection, and restores all **280 outer member contexts**. It reports six-class and MI-only macro F1, per-class scores, all reference seeds and their mean/sample SD. MI-only scoring retains six-way predictions, so an MI patient predicted HC stays an error. Paired descriptive intervals resample the same patients jointly across seeds; they do not count seeds as independent subjects or refit models.

The completed report is in `results/deviation_vcg_v1/complete/REPORT.md`. These repeatedly inspected PTB folds are development data. An apparent gain requires repeatability and independent cohort confirmation; the 0.80–0.90 target and SOTA status remain unproven.
