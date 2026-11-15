# Consecutive VCG beats and RR timing

**Superseded before full training:** the user requested a 4/8/16 comparison. The active queued protocol is now [nested length selection](chronological_lengths_plan.md), with a new config/output namespace and live monitor. The fixed-four setup below remains historical context; its full sweep was never started.

Implemented 15 November 2026, after the user approved testing true chronology as the next patient-aggregation experiment. This study is separate from the active path-signature comparison and the older paused multiview sweep. Full execution is queued behind the signature study's complete independent verification. There is no new classification score at implementation time.

The user clarified that the desired group is **four consecutive heartbeats within one recording**. The primary configuration was changed from 16 to four before any full chronological fit or study manifest existed. The waiting queue was stopped and re-established; no active signature fit was changed. The 16-beat alternative remains in [chronological_vcg_16.yaml](../configs/chronological_vcg_16.yaml) under a separate, unrun output namespace. Its earlier finite hardware smoke is historical evidence, not a completed 16-versus-four performance comparison.

## Hypotheses and four controls

The existing patient feature summaries and old unordered bags discard real beat order. Test whether a small chronological module can use information beyond the morphology and RR content of the same consecutive window.

| Arm | Aggregation | RR input | Purpose |
|---|---|---|---|
| mean_rr | Mean of projected beat/RR embeddings | Preceding RR + availability flag | Secondary simple aggregation reference |
| ordered_no_rr | GRU over chronological beats | Zeroed RR channels | RR ablation with the same recurrent architecture |
| ordered_rr | GRU over chronological beats | Preceding RR + availability flag | Main chronological candidate |
| shuffled_rr | Identical GRU over permuted beat/RR pairs | Original RR remains attached to its beat | Main matched order control |

The three recurrent arms have identical parameter shapes and initial weights for each fit. Mean pooling has different active capacity; a comparison against it alone cannot isolate chronology. No pretrained target-fold encoder is reused: every inner fit and outer refit starts fresh.

Primary contrast: **ordered_rr versus shuffled_rr**. Additional contrasts: ordered_rr versus ordered_no_rr (RR contribution) and mean_rr (practical complexity comparison). A gain over a weak shuffled model does not itself exceed the historical ResNet 0.550 or establish a better overall project model.

## Original data and metadata audit

Keep all **164 patients, 391 recordings, 55,863 cached XYZ beats**, original labels, R positions and frozen patient folds. Each beat remains the original 651-sample denoised waveform at 1,000 Hz. No new R detector, delineation, rotation, amplitude normalization, interval clipping or quality-based removal.

Record metadata is sorted by original cached R positions, with strict checks that each recording belongs to one patient/label and retained beat indices are consecutive. Actual recording lengths range from **33 to 269 beats**, median **141**. Every recording supports a full four-beat training window. Padding remains implemented for short windows and inference tails.

RR is the difference between adjacent **retained detector R positions**, in seconds. It is not independently annotated clinical RR. The first retained beat of each recording has missing preceding RR, represented by a zero-valued normalized input and availability flag zero; subsequent beats retain their original positive intervals. No RR threshold excludes a beat. Only the preceding gap is supplied, not both neighboring gaps or absolute elapsed time.

## Model

Each beat uses the existing compact XYZ ResNet encoder (widths 16/32/64, four temporal bins, 64-dimensional embedding). A small two-input MLP maps normalized preceding RR and its availability flag to eight dimensions. Concatenated 72-dimensional tokens feed a 32-state GRU and dropout/six-class head. Explicit standard GRU equations support masking without packed/platform-specific RNN kernels. In the no-RR arm both RR channels are zeroed, with the architecture retained.

The controls never feed patient IDs, record identifiers, beat indices or absolute timestamps into the network. Those fields only construct/audit windows and deterministic permutation keys. Preserve signed XYZ amplitude through training-set scaling, not per-beat normalization.

## Training windows and shuffled control

For each epoch, sample **2,048 windows of four consecutive beats**. Sampling is class/patient balanced, then uniform among that patient's recordings, then uniform among valid contiguous starts. A window never crosses a recording or patient boundary. All arms share the same sampled-window plan at every common epoch. Window cross-entropy is optimized; it is not a differentiable patient-macro-F1 loss.

The shuffled control permutes only valid tokens, moving each waveform and its original preceding-RR/missingness pair together. It does not compute fictitious gaps between shuffled timestamps. Permutation RNG is separate from the sampling/model random streams. This tests the added value of order **conditional on the same token-level RR information**, not removal of every temporal clue.

Training permutations vary deterministically by fit seed, epoch and sampled-window position. Inference permutations are fixed by seed and source-record/window identity, so unrelated query patients do not change an existing window's permutation. There is no favorable-permutation selection or test-time confidence-based window choice.

## Inference and masks

Tile every query recording into nonoverlapping windows of up to four beats. Pad the final incomplete window on the right, with explicit masks. GRU updates and mean pooling ignore padded tokens. Every original query beat appears exactly once.

Weight each window's probabilities by its valid-beat count within the recording, then average recordings equally within each patient. This is a declared aggregation policy; it differs from the historical beat-probability mean and the old all-patient embedding bag. The examination's own full recording is available, so the procedure is offline, not a streaming clinical predictor.

## Patient-nested training

- Ten frozen outer folds, three fixed inner patient folds; six original classes and patient macro F1.
- Training seeds **42, 123, 2026**, all retained. These are optimization seeds, unlike the signature study's selector-resampling seeds.
- Waveform and valid-RR moments use training patients only, with class/patient-balanced weights. RR availability is explicit; no learned imputation uses validation/test patients.
- AdamW, rate 0.001, weight decay 0.01, batch 32 four-beat windows, gradient clipping at five. Up to 25 epochs; patience six on inner patient macro F1. Earliest best epoch wins.
- Rounded median of three inner selected epochs sets a fresh outer refit's fixed training length. No outer monitoring or outer-trained initialization enters inner fits.
- **480 fits:** four arms × three seeds × ten outer folds × (three inner + one outer). Max epoch/patience caps match; selected lengths and actual compute may differ by arm.

The nominal epoch still has **8,192 beat slots** and 64 optimizer batches: 2,048×4 slots, with 32×4 beats per batch. This matches the original 512×16/eight-window-batch plan's nominal beat exposure and update count. Inference batches are 64 four-beat windows, retaining 256 beat slots like the original 16-window batches. A four-versus-16 comparison would still change sequence context and number of window labels; it is not a perfectly isolated change of one computational quantity. The alternative is not run or selected from outer scores. Short future windows with padding would reduce valid-beat presentations, which are audited rather than equated to slots.

## Verification and reporting

Synthetic tests cover original RR construction, recording boundaries, paired token permutation, training-only moments, masks, GRU equivalence to PyTorch's GRUCell equations, matched recurrent initialization, all-beat coverage, query-label/permutation independence, actual nested CPU training, cache resume, saved-model restoration and corruption rejection. A separate real-data MPS smoke runs a finite forward/backward training check and disk state restoration. A smoke is not a performance result.

The full report rebuilds normalization and window/permutation schedules, verifies inner stopping and median-epoch refits, checks shared schedules/initial weights across recurrent controls, and restores every saved outer model. MPS is not guaranteed bitwise identical across machines/versions; restoration requires declared probability tolerance and identical predicted classes.

Report patient macro F1, each class, all seeds and mean/sample SD, plus MI-only macro F1 using unchanged six-way predictions. Paired class-stratified patient draws share resamples across seeds and condition on fitted predictions; they do not refit models or correct repeated development/correlated folds/multiple comparisons. Seed SD is not a generalization confidence interval. These repeatedly inspected PTB patients remain development data.

## Execution and queue

```sh
# Finite actual-data/device/restore check, not the complete study.
.venv/bin/python -m mi_localization.chronology_study --smoke

# The approved current-machine queue: waits for the verified signature report.
.venv/bin/python scripts/run_chronological_vcg.py --wait-for-signature --publish

# Direct execution on a prepared machine, without a dependency queue.
.venv/bin/python scripts/run_chronological_vcg.py --publish

.venv/bin/python -m mi_localization.chronology_study --status
tail -n 12 tmp/chronological_vcg_v1.log
```

The queue pins the current signature fingerprint and requires its complete report, all bootstrap replays and all 150 outer context restorations. It stops if the local dependency fails or ends without verification. It does not change or stop the signature process. The wait option concerns a local run; a clone does not inherit that process or its ignored runner metadata.

Windows can use `.venv\Scripts\python.exe` and the direct runner after compatible source banks/configs are prepared. Never infer a Mac process from a saved PID on another machine. Metadata/cache: `data/processed/chronological_vcg_v1`; results: `results/neural_chronological_v1`; source configuration: [chronological_vcg.yaml](../configs/chronological_vcg.yaml). Raw metadata arrays, weights and individual predictions remain ignored. Only verified aggregates are auto-published, with a clean tracked working tree.

## Decision rule

Attribute a benefit to chronology only if ordered_rr improves over the matched shuffled_rr control consistently across seeds, with informative class-level results. Attribute an RR benefit only if the no-RR contrast supports it. Prefer a simpler mean model if it performs as well. A result below the historical stronger benchmark is not a project breakthrough merely because its local control is weaker.

No new cohort, label, phase ground truth, independent validation or promised 0.80–0.90 score is introduced. This experiment tests a specific information/aggregation hypothesis before any larger sequence architecture or combined model is justified.
