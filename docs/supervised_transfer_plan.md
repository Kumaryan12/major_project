# Matched supervised-source transfer to measured Frank VCG

Authorized 16 November 2026 after completed source learning (`78aa1bf`) and independently replayed expanded recording-overlap screen (`6b892d1`). The user accepted a controlled three-arm target study. This experiment is prospective from this authorization on repeatedly studied **development** patients, not a new untouched confirmation cohort.

## Fixed hypothesis and arms

Does native-diagnostic supervised source learning give a more useful initialization for the original measured-VCG six-class task than a newly trained matched scratch model?

| Arm | Encoder initialization | Role |
|---|---|---|
| `scratch` | Fresh random VCG ResNet | Matched target-training control |
| `source_certain` | Fixed source seed42, likelihood100/unlisted objective | Primary transfer procedure |
| `source_positive_intermediate` | Fixed source seed42, intermediate-positive sensitivity | Sensitivity transfer procedure |

Use both native source objectives irrespective of their tiny source-validation AP difference (0.317456/0.317306). Both were selected at source epoch13; neither is selected using target outcomes. Their source diagnostic heads, source normalization/optimizer and source positive weights are discarded. Only encoder weights transfer; initialize matching fresh six-class target heads and fine-tune the whole same network. There is no automatic source-NORM→target-HC mapping or change to target territories.

Primary endpoint: patient macro F1 of the prespecified **equal-probability ensemble of all three target seeds**, with `source_certain`−matched `scratch` the primary comparison. Report the intermediate-positive comparison separately as sensitivity, with no source-objective/seed-subset/weight selection from outer scores. Also report every seed's patient metrics, mean seed F1±sample SD and ensemble per-class results. Mean seed F1 is not ensemble F1.

Historical neural ensemble0.584570 and full-feature ridge0.584266 remain separate development references. The source pilot's multilabel AP/AUROC/F1 do not replace target localization metrics. The earlier masked-reconstruction transfer result is a different source objective/cohort and is not reused as the new scratch control.

## Target data and matched protocol

- Same original **164 patients / 391 records / 55,863 beats**, six classes in order AMI/ALMI/ASMI/IMI/ILMI/HC, cached signed measured XYZ and R positions,651samples at1000Hz. Original `configs/patient_folds.csv` and beat hashes must match the expanded screen's protected target cohort.
- Ten frozen patient outer folds, three patient inner folds within each outer-training partition. Target seeds42/123/2026, same inner split seeds as the established transfer protocol. No inner teacher/normalizer/selection state is trained on outer-query patients.
- Same small ResNet widths16/32/64 and four temporal bins, dropout0.2, ordinary six-class CE/AdamW, weight decay0.01, gradient norm cap5. All source layers may adapt; no ECG input enters target inference. Source crop-feature averaging is not added to the target beat model.
- Fit target axis normalization on each exact target-training partition only, with identical patient/class weights across arms. No source normalizer, target-query statistics, arbitrary rotations or per-beat unit-amplitude normalization.
- Both learning-rate candidates `[0.001,0.0001]`, batch128,8,192 sampled beats/epoch, at most25epochs, patience6 in every arm. Same target fit seeds, fresh initial classifier and patient/class-weighted sampled rows for shared epochs; source encoder weights intentionally differ. No unequal per-arm tuning expansion.
- Choose the rate with the highest mean best inner patient macro F1 independently per arm; exact ties use configured order. Refit a fresh model on all outer-training patients for the rounded median best epoch of that chosen rate. No outer test monitoring/stopping/threshold/epoch/budget selection.
- Average probabilities across **all original held-out beats per patient**, then average per-patient probability vectors equally over all three target seeds for each arm. Do not choose favorable records/beats/patients or use globally pooled outer predictions to fit a combiner.

Declared total: **630 completed fit contexts** =3arms ×3seeds ×10outer folds ×(2rates ×3inner folds +1outer refit). Of those,540are inner fits and90outer models. There are210 three-arm matched context checks, plus within-arm cross-rate initialization/sampling checks. Source pretraining adds compute beyond the equal target fit budgets; the three target seeds do not replicate source pretraining.

## Prerequisites and integrity

The launcher independently runs source-model `--verify` and expanded-audit `--verify` before target fitting. Pin a preflight receipt to current code/config/runtime, both source exports, source results, expanded screen, target bank/folds and claim scope. Resume requires matching receipt/scope and rechecks current hashes; no integrity check is disabled or historical manifest edited.

Require both source model exports with the declared objective/source seed and exact architecture, finite weights and source-only/unresolved-identity flags. Validate all source report/member hashes. Require the new expanded screen's passed flag, every known-copy control, no copy candidate/exact match/insufficient-active-lead pair, all input/cached/pair artifacts and current code scope. Verify that its source inventory covers all retained source training **and model-selection validation** records, and its protected target is the unchanged original cohort.

Passed recording screening does **not** clear patient independence: different visits, shorter/modified copies and retrieval misses remain unresolved. Derived source XYZ differs from measured Frank XYZ; clinical source/target annotation equivalence is not adjudicated. This is explicitly exploratory transfer under those limits, not clean external confirmation.

## Checkpoints and independent report

Use a new namespace `results/neural_supervised_transfer_v1`. Keep source/earlier numerical modules/configs/results unchanged. Reuse the existing pinned transfer trainer functions without editing their two-arm historical study.

For every inner and outer fit, atomically save the complete target model, exact normalization, context/row hashes, fitted audit and query probabilities in ignored `review/`. Compatible resume restores completed fits and re-infers their predictions; interruptions during an unfinished fit repeat that one fit. No per-epoch optimizer-resume claim is made. MPS is seeded but not guaranteed bitwise-deterministic training.

Final reporting runs in a separate process and restores **all630 models**, recomputes every exact query prediction, verifies inner selected-checkpoint F1, repeats every rate/epoch decision and checks matching heads/sampling/normalization/parameter counts/memberships. Outer refits are checked for zero validation/test monitoring. Verify pooled patient probabilities/metrics, complete seed/fold/arm coverage and all model/prediction/selection hashes. Models and patient-level data remain ignored; public results contain aggregate scores, selection summaries, plots and hashed audits.

Use5,000 class-stratified paired patient bootstrap draws, shared across arms and fixed seeds; report ensemble differences and mean-seed differences separately. Each patient is resampled once per draw, never treated as three independent seed-patient samples. Intervals are descriptive/conditional on these fitted predictions and two fixed source encoders, without refitting or correction for overlapping training folds, repeated development or multiple comparisons. No universal ceiling, guaranteed0.80–0.90, clinical validity or SOTA claim follows from this experiment.

## Commands and stopping boundary

Pre-launch validation: **289 tests passed**, including eight new encoder/provenance/record-screen/preflight-gate, three-arm matching, training-only selection/normalization, full small-study135model restoration/report, tamper and paired-ensemble bootstrap checks. The actual input loader confirmed the original164/391/55,863target cohort and both fixed source encoders. A real MPS smoke fitted/restored all three arms on six inner-training and six inner-query patients, with matching heads/sampling/normalization and no outer-test patients. It is not a final performance result.

Launch observation: implementation `4a41e32` was pushed before the full sweep. Fresh independent source-model verification and the full57,968pair expanded-screen replay passed; the matching private preflight receipt was frozen. The detached launcher started on16November2026 at16:40UTC (22:10IST), PID48689, with macOS idle-sleep inhibition. The monitor subsequently confirmed an active target worker on seed42 / outer fold1 / scratch / rate0 / inner1. This is active training, not a completed result. `launch.json` and the public study manifest record the declared scope; use the live monitor for current phase/counts. No following experiment is queued.

```bash
.venv/bin/python scripts/run_supervised_transfer.py --preflight-only
.venv/bin/python scripts/run_supervised_transfer.py --publish
.venv/bin/python scripts/supervised_transfer_status.py --watch
# Independent completed-result replay/report:
.venv/bin/python -m mi_localization.supervised_transfer_report
```

Logs: `tmp/supervised_transfer_v1_preflight.log`, `tmp/supervised_transfer_v1_train.log`. The runner reuses only a matching verified preflight, trains every declared context, independently reports/publishes the result, then stops. No ensemble fusion, source expansion, new architecture or further study is automatically launched.

Windows uses `.venv\Scripts\python.exe`; use a separately configured namespace and inspect/rebuild prerequisite banks/scopes as needed. Changing only the final output directory cannot make a Mac-pinned predecessor bank compatible. Follow `AGENTS.md`/the collaborator handoff rather than changing historical manifests.
