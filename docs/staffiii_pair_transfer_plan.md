# Paired STAFF III source transfer

The source task predicts LAD, LCX or RCA from before/during balloon-occlusion ECG. The target stays six-class patient macro F1 from measured Frank XYZ. This is a bounded ECG-assisted extension. Compare a source model trained on the difference between a pair's embeddings with the same model trained on their mean. Target arms are scratch and the two source initializations.

## Source metadata and input

Use PhysioNet's STAFF III v1.0.0 release and its checksum inventory. Read the fixed annotation workbook's ordered BI slots, native vessel labels and inflation starts. Join to the first balloon-inflation event in each file. Use [-20, -10) and [+30, +40) seconds from the marker in the same recording, with the second window bounded by the next inflation or deflation marker. Reject annotated contrast inside [-20, +40) and check agreement with the workbook start within 1.1 s. Use the three major vessel groups and interpret the workbook's `circ` as LCX. Exclude source patients 1, 4, 5, 6 and 89 because the workbook identifies possible lead errors. A file can begin later in a procedure and contrast recording can be incomplete.

Check all headers and event files against the published checksums. Read the paired digital windows from complete checksum-verified records where locally available. For remaining records, require the exact byte range, content length and two matching copies from PhysioNet's release and archive URLs. Compare the range decoder to full-file WFDB decoding on the available complete files. Pin the excerpt and its local receipt for independent decode replay. Publisher whole-file hashes apply directly to the full files. The partial files have matching range evidence and local hashes.

Decode the eight common ECG leads in mV using the WFDB gains and baselines. Source limb electrodes use the Mason–Likar setup. Apply the existing Kors map for derived XYZ and the existing signed XYZ denoising and peak detector at 1,000 Hz. Use four uniformly selected complete beats per window and skip pairs lacking sufficient usable beats. Retain the first quality-passing pair per patient in workbook inflation order. Split retained patients using a stratified 25% source validation set, seed 42. Require at least five training and two validation patients per vessel for the bounded pilot.

Screen the two retained waveform windows per source patient against all 391 original target recordings using the pinned common-lead copy screen and 20 positive controls. A copy flag or insufficient active leads blocks fitting. The copy screen examines recording overlaps, with cross-dataset identity unresolved.

## Matched source and target models

Use one fixed source seed, 42. Both source arms share a ResNet, native three-class head, joint training-only normalization, training pair draws, optimizer and 12 epochs of eight updates with 16 pairs per update. Encode all four beats in each window, average beat embeddings within the window, then combine the two window embeddings by `during - before` or `(during + before) / 2`. Validate the fixed final checkpoints on held-out source patients using vessel macro F1. Independently replay both source probabilities and export encoder weights only.

Target uses fixed patient folds, seeds 42, 123 and 2026, and the original signed VCG bank of 164 patients, 391 records and 55,863 beats. Reuse the historic eight-epoch waveform protocol, with a new six-class head, exact target-train normalization, and matching sample draws across arms. Target inference averages all measured-XYZ beat probabilities per patient and then probabilities across the three seeds. The primary paired comparison is change minus mean. Both arms are also compared with scratch. Report per-class and per-seed results and a descriptive paired patient interval. The fixed PTB cohort is development data.

## Run

From the repository root use `.venv/bin/python scripts/run_staffiii_pair_transfer.py --publish`. Use `--bank-only` for independently replayed source preparation. The full runner performs inner-only smoke training and replay before 90 full target fits, then publishes only verified aggregate evidence. Detached logs are in `tmp/staffiii_pair_transfer_v1.log`. Source workbooks, waveforms, patient predictions and weights stay local.

[PhysioNet data and annotation workbook](https://physionet.org/content/staffiii/1.0.0/) · [Earlier timing screen](EXTERNAL_DATA_SHORTLIST.md#staff-iii-timing-check)
