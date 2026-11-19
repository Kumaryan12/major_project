# Open datasets that could support the measured-VCG project

Reviewed 19 November 2026. This is a source-data feasibility review; no new model training, target relabeling or automatic experiment queue was started. The already authorized frozen-distance study continues separately. Measured Frank XYZ remains the primary target input. Any standard-ECG or ECG-derived-XYZ source experiment must be labeled ECG-assisted training, with a separately reported ECG+VCG extension if ECG is also needed at inference.

The search did not establish a large, immediately downloadable six-class measured-Frank cohort beyond our existing PTB material. Open standard-ECG datasets offer new patients or auxiliary supervision, but they cannot be pooled into the six-class Frank cohort without checking acquisition and label meaning. [Archive/partner routes](data_acquisition_action_plan_20261119.md).

## Recommended shortlist

| Dataset and direct source | Confirmed public material | Use worth testing | Size and practical boundary |
|---|---|---|---|
| **Shandong Provincial Hospital (SPH)**: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9174207/), [metadata](https://springernature.figshare.com/articles/dataset/The_attributes_of_ECG_records/17912441), [dictionary](https://springernature.figshare.com/articles/dataset/diagnostic_statement_dictionary/17912507), [signals](https://springernature.figshare.com/articles/dataset/ECG_records/17912444) | 25,770 12-lead records from 24,666 patients. The downloaded dictionary contains four MI-region statements. Our metadata replay finds 260 records /242 patients with one or more such statements. | First new regional-supervision source to audit. It explicitly distinguishes anterior and anteroseptal ECG patterns. Retain native, potentially overlapping source statements. Mostly old MI, incomplete target-region coverage and ECG acquisition are material mismatches. | Full signal archive 2,281,799,680 bytes, about 2.28 GB. Metadata/dictionary total only 1,001,058 bytes; both downloaded and checksum-verified. No signal download in this review. Publisher deposits list CC0; the paper is CC BY 4.0. |
| **MIMIC-IV-ECG v1.0**: [official release](https://physionet.org/content/mimic-iv-ecg/1.0/) | Roughly 800,000 ten-second 12-lead ECGs from nearly 160,000 patients, plus public machine statements and links to clinical notes. The waveform release is openly accessible under ODbL 1.0. | Patient-diverse morphology learning, or carefully screened regional weak supervision. Machine statements and uncertainty/negation need their own audit; linked human/clinical information has separate access conditions. | 90.4 GB uncompressed, 33.8-GB full ZIP. Start with metadata; a selected 1,000-record int16 waveform payload at 500 Hz is roughly 120 MB before headers/format variation. Process recordings in batches. |
| **MedalCare-XL v1.3**: [official release](https://zenodo.org/records/8068944) | 16,900 synthetic 12-lead ECGs, supplied with simulation information and raw/noisy/filtered variants. MI subclasses are artery LAD/LCX/RCA crossed with two transmurality levels. CC BY 4.0. | The most distinctive simulation direction: investigate controlled disease and noise changes, after auditing shared anatomy/run identifiers. It adds no real patients and its labels differ from the PTB endpoint. | Official ZIP is 9,316,411,006 bytes, about 9.32 GB, before extraction. Audit archive access/contents and a storage-limited subset before a full download. Same-anatomy healthy/MI pairing is a question to check, not an established matched source bank. |
| **LUDB v1.0.1**: [official release](https://physionet.org/content/ludb/1.0.1/) | 200 ten-second 12-lead ECGs with cardiologist-annotated P/QRS/T boundaries and diagnoses. ODC Attribution 1.0. | A small source for supervised delineation and phase-feature quality. This tests a different bottleneck from adding MI-class examples. Transfer to measured XYZ needs a defined validation path and appropriate inference labeling. | 23.6 MB uncompressed, 13.8-MB ZIP. Particularly suitable for a laptop-scale feasibility study. No LUDB download or delineator fitting was started here. |

Rank these by question. SPH is the first new candidate for explicit regional ECG supervision; MIMIC is the larger diversity option; MedalCare supplies controlled simulated mechanisms; LUDB supplies manual timing supervision. None has demonstrated an improvement on our measured-XYZ target.

## What the SPH metadata actually supports

The two files matched the publisher's computed and supplied MD5s and were also recorded with SHA-256. Raw metadata remain under ignored `data/raw/open_dataset_screen_20261119/sph/`. [Aggregate screen](../results/open_dataset_screen_20261119/sph_metadata_summary.json) · [Publisher file inventory](../results/open_dataset_screen_20261119/release_inventory.json) · [Replay script](../scripts/screen_sph_metadata.py).

| Native code / statement | Recordings | Unique patients | Recorded age modifiers, per statement-record |
|---|---:|---:|---|
| 160 / Anterior MI | 52 | 47 | Acute 1, old 47, unspecified 4 |
| 161 / Inferior MI | 120 | 117 | Acute 2, recent 3, old 114, unspecified 1 |
| 165 / Anteroseptal MI | 91 | 82 | Acute 4, recent 9, old 75, unspecified 3 |
| 166 / Extensive anterior MI | 7 | 7 | Old 7 |

Supports overlap: there are 260 MI-statement recordings from 242 patients, and ten recordings carry more than one MI-region statement. Do not sum the rows to obtain a unique-patient denominator. The dictionary does not provide ALMI/ILMI counterparts; extensive anterior cannot simply be converted into either current anterior subtype. Unspecified age does not establish acute or old MI. The 13,905 normal-ECG-statement-only recordings are not independently reviewed clinical healthy controls.

The script checks all codes against the released dictionary, unique recording IDs and nonempty patient IDs, then independently reproduces MI statement parsing with a boundary-aware regex. Both passes reproduced the aggregate output. Duration from metadata is 10–56 seconds at the declared 500 Hz, while the paper describes 10–60 seconds. No raw HDF5 file has been decoded, so duration/unit/lead/QC checks remain pending. No PTB identity or waveform-overlap clearance, six-class mapping, or new target patient count follows from this metadata screen.

Reproduce locally with the downloaded metadata:

```sh
.venv/bin/python scripts/screen_sph_metadata.py
```

The script checks the exact release hashes and refuses to overwrite changed results. It fits no models and downloads no signals. Only aggregate results are tracked.

## Other open releases and their limitations

- **PTB-XL, ACS-ECG and STAFF III** remain important source datasets. Bounded versions have already been trained and transferred in this project, without a demonstrated gain over their matched references. A larger source experiment needs a new hypothesis and controlled compute budget. Preserve the PTB-XL fold-10 exclusion and all earlier source namespaces. [Existing source audits/results](EXTERNAL_DATA_SHORTLIST.md).
- **Chapman/Shaoxing/Ningbo** has [45,152 standard-ECG records](https://physionet.org/content/ecg-arrhythmia/1.0.0/), 5.1 GB uncompressed and a 2.3-GB ZIP under CC BY 4.0. Its downloaded 2.5-kB condition dictionary includes a general MI code, not the current regional endpoint. It could support generic pretraining, with lower localization priority than a region-annotated source. Earlier 10,646-person releases overlap this expanded archive; do not count both as independent patients.
- **CODE-15%** has [345,779 exams from 233,770 patients](https://zenodo.org/records/4916206). The released diagnostic columns cover six rhythm/conduction abnormalities rather than MI regions. Its 46.3-GB collection is a possible generic source, with lower immediate value for our current endpoint and laptop budget.
- **PhysioNet/CinC 2007 MI Imaging** supplies only four averaged cases, with two public reference interpretations. We already obtained the two labeled cases as a 6.48-MB format/reference fixture. This does not supply a larger six-class training cohort. [Local acquisition review](measured_vcg_data_acquisition.md).

## Availability traps checked

**CODE-II-open** is promising on paper: 15,000 unique-patient ECGs and 66 diagnostic classes. However, its [paper](https://www.nature.com/articles/s41746-026-02704-4) says the public data are available while the freshly fetched [author repository](https://github.com/PetrusAbreu/code-ii) still lists both PhysioNet links as “coming soon.” README Git blob `75f31812881766145da010f0d42b8b2bcec01b4d` was recorded in the release inventory. No working waveform/label release was established in this search. Keep it on the watchlist, not the download-ready list. Its proposed test labels are reserved for blinded benchmarking, not source training.

**Harvard–Emory ECG Database (HEEDB)** contains millions of ECGs, but its [actual v5.0 access page](https://bdsp.io/content/heedb/5.0/) requires credentialing, training and a data-use agreement. An open-access paper does not make the dataset an anonymous open download. It is a separate longer-term credentialed route, with no account application or download started here.

The [Mendeley cardiac-patient release](https://data.mendeley.com/datasets/gwbz3fsgp8/2) used in some high-scoring papers is an ECG image dataset. It is a different input task from signed native numerical XYZ. Unpaired MRI datasets also do not directly add VCG training examples. Repackaged PTB/PhysioNet challenge files can duplicate existing patients and held-out sources.

## Next bounded decisions

1. Finish the current distance study. No source training is queued behind it.
2. For regional data, first audit SPH signals and label compatibility with a capped download/QC plan. Preserve its native multi-label statements and infarct age. Choose source development/validation patients before any source fitting; separately screen source/target overlap.
3. If scale is the goal, perform a metadata-first MIMIC regional/uncertainty support audit, then define a capped 500–1,000-patient signal pilot. Prefer a deliberate source task over another generic pretraining run.
4. For better phase features, LUDB is an inexpensive alternative research path. A delineator's source accuracy or cleaner reconstruction is not target localization macro F1. If ECG supplies timing at target inference, report that as an ECG+VCG extension. XYZ-only inference requires its own validated student/transfer method.
5. For simulation, verify MedalCare anatomy-group partitions and disease/noise pairing, then register a compact matched control study. Keep simulated and clinical patients separate in reporting.

Use equal-compute scratch/source controls and the fixed target folds for any later study. Never select source cases or label mappings using target outer outcomes. Independent regional clinical evidence and real measured-XYZ patients remain necessary for direct confirmation of the primary project endpoint.
