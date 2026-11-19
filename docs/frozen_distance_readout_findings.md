# Frozen64 distance readout findings

Completed and independently verified on 19 November 2026 at 13:15:08 IST. Published automatically in `ee8aeca9`; implementation `b1916616`, protocol `9e20f805`. [Validated report](../results/frozen_distance_readout_v1/complete/REPORT.md) · [Metrics](../results/frozen_distance_readout_v1/complete/summary.json) · [Fixed protocol](frozen_distance_readout_plan.md) · [Post-completion custody check](../results/frozen_distance_readout_v1/complete/completion_check.json).

## Result and decision

The shared-covariance Mahalanobis procedure did not demonstrate an improvement over the matched ridge64 procedure or ordinary centroids. Retain ridge as this comparison's reference. No further head tuning, data download or training run was started during completion review.

| Readout | Primary: 145 dated earliest-record patients, ensemble F1 | Primary mean member F1 | Sensitivity: 164 selected-record patients, ensemble F1 |
|---|---:|---:|---:|
| Ridge64 | .535691 | .523619 | .546530 |
| Euclidean centroids | .507770 | .470648 | .520508 |
| Inner-selected Mahalanobis | .507181 | .481357 | .519400 |

All three source seeds 42/123/2026 and all ten folds are retained. Equal-seed probability ensembles and mean member F1 are different summaries. Each of the three Mahalanobis members scores below its paired ridge member. Relative to centroid members, Mahalanobis has a higher mean, while its probability ensemble is slightly lower; do not describe the mean-member difference as an ensemble improvement.

| Prespecified primary-cohort contrast | Ensemble F1 difference | Descriptive paired 95% interval |
|---|---:|---|
| Mahalanobis minus ridge, primary | −.028510 | [−.088700,+.031304] |
| Mahalanobis minus centroid, covariance control | −.000589 | [−.041155,+.038686] |
| Centroid minus ridge, secondary | −.027921 | [−.100774,+.043663] |

The point estimates fall below ridge. None of the intervals excludes zero, so they do not establish a general superiority or inferiority claim. The intervals condition on fixed predictions from repeatedly studied PTB development patients. The 164-patient selected-record sensitivity has the same direction and uncertain contrasts.

## Subtypes and controls

Relative to ridge, Mahalanobis anterior-MI F1 falls from .4286 to .2963, anteroseptal from .4815 to .4528, and inferolateral from .5000 to .4500. Inferior rises from .5263 to .5424 and HC from .8333 to .8571; anterolateral is unchanged at .4444. Thus the tested covariance readout did not resolve the hard MI subtype distinctions. This is a descriptive score pattern, not proof that particular physiological information was removed.

Inner selection chose shrinkage 1/.9/.5 in 14/12/4 of 30 outer fold/seed decisions. These are overlapping training decisions, not independent observations or a universally optimal shrinkage. The lambda1/centroid equivalence check passed for all 90 inner contexts and 14 selected outer contexts, 104 in total. Keep the declared inner rule; do not choose another shrinkage after examining outer scores.

## Completion evidence and scope

Independent verification reconstructed all 120 exact-partition source feature contexts, restored all 420 new distance heads (including 60 outer heads) and 210 frozen ridge heads, rebuilt training-only scaler/centroid/covariance/precision statistics, and replayed all decisions and predictions. Ridge baseline metrics reproduced exactly. All 1,444 source files remained unchanged. There were zero encoder updates. The 376-test suite and inner-only real-data fitting/restoration/identity smoke passed before full fitting.

The additional completion check matched 420 distance audit hashes, 840 model/prediction files, 120 feature receipts and caches, 210 source ridge receipts, 104 identity controls, all 1,444 protected source files and the manifest verification digest. Numerical source/protocol/fold hashes were unchanged. This was a read-only custody check after the independent waveform/feature replay, not another fitting or inference run.

The one-shot job exited with code 0, with one launch and no live worker or launcher at completion review. It published the final aggregate report and summary and stopped. Preserve `frozen_distance_model.py`, `frozen_distance_readout.py`, its config/protocol and completed numerical outputs.

This comparison used the frozen 64-feature earliest-record encoders. It does not evaluate every possible metric/classifier, identify a single data-versus-representation cause, or change the historical all-record .603374 convolution / .601801 waveform endpoint. With no demonstrated gain from this recipe, the proposed next priority is a bounded source-label/waveform feasibility audit, such as the SPH regional ECG candidate already screened at metadata level. That is a separate decision with ECG-assisted labeling and no guaranteed transfer benefit. [Open-data review](open_dataset_review_20261119.md).
