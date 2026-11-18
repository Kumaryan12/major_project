# Matched ResNet beat and patient probability loss

The existing measured XYZ waveform ResNet trains on individual beat cross-entropy and predicts by averaging all eligible beat probabilities per patient. This study tests whether training on the pooled patient probability gives a better patient decision. The matched control uses the same grouped training batches and per-beat cross-entropy.

| Arm | Loss per sampled patient bag |
|---|---|
| Beat CE | Mean negative log true-class probability across eight beats |
| Patient CE | Negative log of the mean true-class probability across eight beats |
| Hybrid CE | Equal average of the two losses |

Draw a class uniformly, a training patient uniformly within that class and eight beats uniformly with replacement across that patient's original beat rows. Beats may come from multiple recordings of the same patient. A batch contains 16 bags, for 128 beat presentations per update. The historical scratch recipe provides the signed waveform bank, ResNet, dropout, class- and patient-weighted training normalization, AdamW settings, gradient clipping, and eight epochs of 2,048 beat presentations. The historical scratch result used individual beat draws. All three new arms share the sampled bags, initial weights and update counts within each fold and seed. Stable log-space pooling preserves the probability average for very small probabilities.

Primary contrast is the patient CE minus beat CE three-seed probability ensemble. Hybrid minus beat CE is secondary. Evaluate all ten original patient folds and seeds 42, 123 and 2026. Each arm keeps the fixed eight-epoch budget. Held-out inference averages every original beat probability per patient. Report all seed and ensemble metrics, per-class F1 and paired descriptive patient intervals. The existing 0.601801 waveform and 0.603374 selected linear convolution scores are historical development references, with the new beat CE as the matched loss control.

The real-data smoke uses an inner split of outer fold 1's training patients, one seed and two updates per arm. For the full run, save 90 fits and replay every saved model in a separate process. Reconstruct normalization, the initial state and sampling schedules and confirm the saved probabilities for every fit. Completed fits can be restored after interruption. Raw waveforms, checkpoints and patient predictions remain local. The runner stops after verification and optional publication of aggregate evidence.

Run the smoke with `.venv/bin/python scripts/run_patient_probability_loss.py --smoke`. Launch full training with `.venv/bin/python scripts/run_patient_probability_loss.py --publish` from the repository root. On the launching Mac, `tail -F tmp/patient_probability_loss_v1.log` shows progress for the detached run. The completed result is `results/neural_patient_probability_loss_v1/complete/REPORT.md` after the separate replay.

[Historical waveform pilot](vcg_pretraining_pilots_findings.md) · [Research backlog](RESEARCH_BACKLOG.md)
