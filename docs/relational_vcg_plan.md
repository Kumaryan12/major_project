# VCG-first relational tensor development

Decision: measured Frank XYZ remains the primary input. ECG + VCG is a
separate future extension, not an added input to the primary benchmark.

## Motivation and first implementation

The sepsis paper by Nesaragi, Patidar and Thangaraj (2021), DOI
10.1016/j.bbe.2021.06.009, motivates stacking correlation matrices and
compressing them. Its AUROC/utility results are not evidence of 0.80-0.90
macro F1 for MI localization. This implementation is an adaptation, not an
exact reproduction of that sepsis study or a claim of established novelty.

For each aligned beat, original + D3-D9 signals in XYZ produce 24 channels.
Four symmetric matrices describe the whole beat and approximate R-relative
QRS / early post-QRS / late post-QRS regions. These are not verified waveform
delineations. Inter-channel Pearson correlations omit absolute amplitude;
the parallel covariance view and unchanged 192 amplitude-bearing statistics
retain complementary information. Flat channels are handled explicitly.

The bank computes wavelet bands over the full denoised PTB recording before
cropping the original 651-sample beats. The paper-faithful short-beat
statistics remain a separate control. Full-record processing is offline and
noncausal; it must not be described as a streaming prediction system.
The context construction reduces short-window extension effects, but it is
not guaranteed to improve classification. Record-edge effects still exist.

The shared projection learns a patient/class-weighted channel second-moment
basis on training data. Both matrix axes use the same basis; the region axis
is never mixed. Covariance scales are estimated from training data only.
Core upper triangles are flattened and concatenated with statistics.
Ranks 4/8 yield 40/144 relational features, respectively, plus 192 statistics.
Uncompressed matrices have 1,200 features, including diagonal entries, plus
192 statistics. Correlation diagonal entries are largely redundant; this
control is intentionally simple and exposes whether compression helps.

## Protocol

- Same 164 patients, six classes, cached R positions and 55,863 test beats.
- Previously inspected frozen 10 patient folds remain a DEVELOPMENT benchmark.
- All arms use the same pre-existing fixed depth-3, 250-tree XGBoost.
- Inner three-fold patient validation chooses shared-core rank 4 or 8.
- Every inner fit learns its own projection/scales, without validation patients.
- Equal class weights and equal patients within each training class.
- Mean beat probabilities give one prediction per test patient.
- Select neither ranks nor ensembles using outer test performance.
- No PTB-XL reads, refits or rescoring. No ECG channels enter this experiment.
- This fixed-classifier study differs from the earlier depth-3/6 nested model
  benchmark; compare improvements to its own short-statistics control.

Five arms: short statistics, context statistics, uncompressed correlation +
context statistics, shared correlation cores + context statistics, and shared
covariance cores + context statistics. Feature caches and checkpoint
fingerprints detect configuration/source changes. Partial runs have separate
directories and explicitly incomplete metrics; never call their scores
complete cross-validation. Fold checkpoints permit safe continuation.

## Run

```bash
.venv/bin/mi-localization relational-features
.venv/bin/mi-localization relational-study
.venv/bin/python -m mi_localization.relational_report
```

For a partial integration check:

```bash
.venv/bin/mi-localization relational-study --arms correlation_shared --outer-folds 1
```

Configuration: `configs/relational_vcg.yaml`. Cache:
`data/processed/relational_vcg_v1/`. New results:
`results/relational_vcg_v1/complete/`. Existing reports remain unchanged.
Raw signals, feature arrays, fold beat probabilities and patient prediction
files stay local; small aggregate metrics/tuning audits are committed.
The final reporting command requires all five completed paired arms and
checks predictions against the frozen patient labels/folds. Bootstrap
intervals resample patients within classes using identical draws for every
arm. They are conditional on fitted predictions, without model refits or
correction for repeated development, shared training folds or multiple
comparisons; they are descriptive uncertainty, not independent confirmation.

## Separate extension, not yet implemented

An ECG + VCG hybrid may use actual simultaneously measured PTB ECG, rather
than reconstructed leads. It needs matched patients/folds, a VCG-only
comparator, separate artifact names and input disclosures. Larger-cohort
pretraining and learned waveform encoders can then be studied. Held-out
patients must be excluded even from unsupervised pretraining. Examples to
review include Mehari and Strodthoff (2022), DOI
10.1016/j.compbiomed.2021.105114, and Guo et al. (2025), DOI
10.1016/j.ymeth.2025.01.015. Their tasks/metrics do not establish our target.

An independent improvement claim still requires a fresh untouched cohort,
class-level results and patient-level uncertainty. A new random split of
the same repeatedly studied patients is not fresh confirmation. The goal
of 0.80-0.90 macro F1 is aspirational, not an expected result or stopping rule.
