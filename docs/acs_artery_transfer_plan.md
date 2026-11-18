# ACS-ECG artery-supervised transfer to measured XYZ

This ECG-assisted training extension tests whether angiography-linked artery supervision improves six-class localization from measured Frank XYZ at inference. Primary contrast is OMI-plus-artery pretraining versus OMI-only pretraining. Both use identical ACS-ECG records and update schedules. Fresh measured-XYZ scratch is the third arm. The frozen PTB cohort has 164 patients, 391 records, 55,863 beats, the original labels and ten patient folds.

## Source cohort and labels

Use ACS-ECG v1 `CSV.zip` and `ECG_row_data.zip`. Verify the versioned published size and checksum. Exclude the publisher's unlabeled test file before waveform decoding. Verify that its patient and waveform identifiers are disjoint from the labeled file. Select all 1,139 patients with any published `OMI = 1` and 1,139 seeded sampled patients without a positive OMI visit. Divide patients 80/20 with stratification by available artery group, other OMI, or OMI-negative status. Retain all visits of selected patients before shared waveform quality checks.

The binary task uses the published record-level OMI flag. The auxiliary task uses LAD, LCX or RCA for OMI records with one acute parent group, based on the article's segment codes. LM, unlabeled and multigroup acute records, and patients with contradictory single-group annotations have masked artery loss. Chronic segment annotations remain distinct from acute annotations. The vessel task describes source angiographic labels. No artery label maps to a PTB territory label. The source AP and artery F1 report performance within the source validation partition and do not choose epochs.

## Waveforms and checks

Verify archive member alignment with the metadata and the WFDB header before decoding. Require 12 named leads, ten seconds at 500 Hz, millivolts and finite data. Apply the existing Kors transform, resample to 1,000 Hz and use target-compatible denoising, detection and signed R-relative crops. Pad to four evenly selected complete beats per record with a valid-beat mask. Both arms share QC exclusions and exact source copy quarantine. Independently re-decode the selected source records and reproduce each retained bank array.

The full bank uses the established common-lead PTB recording-copy screen, with 20 known-copy controls and eight-second continuous confirmation of retrieved candidates. Screen against all 391 original target records. Encoded matches, flagged candidates or insufficient active leads block training. The screen checks recording overlap and leaves cross-dataset patient identity unresolved. The separate source-only/inner-target smoke skips copy screening and does not enter the full study.

## Training and replay

Use one fixed source seed 42. Both source models have the same ResNet initialized identically, with record-level mean embedding pooling and one OMI logit plus three artery logits. Use source-train-only patient-weighted normalization, OMI-balanced and within-class patient-weighted record draws, 2,048 record presentations per epoch, batch size 32, and 12 fixed epochs. Loss is binary cross-entropy plus masked artery cross-entropy weighted 0 or 0.5. AdamW uses the historical waveform optimizer settings. Restore both models and reproduce held-out source probabilities before exporting both encoders.

Transfer encoder weights only. Each target arm starts with a fresh matching six-class head and an identical target-only normalizer, individual-beat schedule, optimizer and eight fixed epochs. Use training seeds 42, 123 and 2026 with every frozen fold. The target decision averages all original beat probabilities per patient and then averages across the three seeds. Report both seed means and the ensemble, per-class scores, and paired descriptive patient intervals. The scratch and 0.603374 selected-convolution references remain available for interpretation. The target test folds are repeatedly used development data.

## Running

From the repository root on macOS/Linux, place the pinned archives in `data/raw/acs_ecg_v1/`. The official release lists [metadata](https://ndownloader.figshare.com/files/62951134) and [raw waveforms](https://ndownloader.figshare.com/files/62951395). Use `.venv/bin/python scripts/run_acs_artery_transfer.py --smoke` for the small real source and inner-target check, or `.venv/bin/python scripts/run_acs_artery_transfer.py --publish` to run smoke and full verification, publish aggregate results, and exit. The detached Mac run logs to `tmp/acs_artery_transfer_v1.log`. Check `results/neural_acs_artery_transfer_v1/complete/REPORT.md` after replay. Fitted models, source archives, and patient predictions stay local.

[ACS-ECG paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13347058/) · [Versioned deposit](https://doi.org/10.6084/m9.figshare.29925314.v1) · [Metadata findings](EXTERNAL_DATA_SHORTLIST.md#acs-ecg-metadata-check)
