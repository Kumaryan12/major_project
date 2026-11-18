# External data shortlist for measured VCG infarct localization

Reviewed 18 November 2026. The current PTB cohort supports controlled development experiments with patient-disjoint folds. Its 55,863 beats come from 164 patients, including 17 anterior and 16 anterolateral cases. The ResNet learning curve rose from 0.472 to 0.550 mean patient macro F1 as training grew from about 113 to 148 patients. This supports seeking more patients for training and a separate cohort for validation. It does not determine how many patients are needed or forecast an attainable score. [Learning curve](../results/neural_learning_curve_v1/complete/REPORT.md)

| Dataset | What is available | Project use |
|---|---|---|
| ACS-ECG | 19,955 ten-second, 500 Hz 12-lead ECGs from 18,909 coronary angiography patients. The descriptor documents patient IDs, acute versus chronic occlusion, and vessel segment labels. The Figshare API lists publicly released diagnostic metadata, median ECGs and raw WFDB archives under CC0. | Best new candidate for pretraining on clinically documented occlusion and artery location. Transfer to measured XYZ can be tested separately against scratch. [Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC13347058/) · [Publisher deposit](https://doi.org/10.6084/m9.figshare.29925314.v1) |
| STAFF III | Open PhysioNet data from 104 patients, with 1,000 Hz 12-lead ECGs, 152 annotated coronary balloon occlusions and timing annotations. | Controlled paired baseline, inflation and recovery study. Separate ischemia and vessel endpoints. Screen contrast injections, timing uncertainty and room/acquisition differences. [PhysioNet data](https://physionet.org/content/staffiii/1.0.0/) |
| PTB-XL 1.0.3 | Open 21,799 12-lead records from 18,869 patients with potentially overlapping diagnostic ECG statements and patient splits. | Existing source benchmark and control. Prior supervised transfer did not demonstrate reliable improvement in the measured-XYZ target. Preserve native labels and the historical holdout. [Dataset](https://physionet.org/content/ptb-xl/1.0.3/) · [Prior transfer](supervised_transfer_findings.md) |
| CSE diagnostic study | Published evidence of simultaneously recorded orthogonal XYZ at 500 Hz in 1,220 clinically validated adults, with broad anterior, inferior and combined MI categories. | Best archive lead for additional measured orthogonal data. No public paired waveform and case-label download located. First verify acquisition geometry, access to waveforms and per-case labels, use terms, and overlap. Its published categories allow only a separate coarse task. [Original CSE diagnostic study](https://www.nejm.org/doi/full/10.1056/NEJM199112193252503) |
| PhysioNet/CinC 2007 MI imaging | Four cases with Frank XYZ as an averaged 2 kHz beat within a body-surface map. Reference MRI ground truth is publicly supplied for the two training cases. | Small waveform and localization sanity check. Insufficient labeled coverage for six-class training or benchmark evaluation. [Challenge data](https://physionet.org/content/challenge-2007/1.0.0/) |

## ACS-ECG metadata check

The official v1 API download for `CSV.zip` succeeded on 18 November 2026. All 296,769 bytes match the published MD5 `1cb46279c0e68e6512bd39214fc56528`. The local SHA-256 is `675a57e90eea1175259301293812ec0a23c73f83ec7fb352a8309f05843d0005`. The raw waveform archive also matched its published checksum. Both archives are stored locally under ignored raw data. [Release metadata](https://api.figshare.com/v2/articles/29925314/versions/1)

The labeled training CSV contains 17,960 ECGs from 17,018 patients, including 855 with multiple ECGs. The released test CSV contains 1,995 ECGs from 1,891 patients and omits the diagnosis and vessel columns. Patient IDs and waveform IDs are disjoint across the two files. Source model selection can use a patient-disjoint split within the labeled release.

Among 1,151 labeled acute occlusion MI (`OMI = 1`) recordings from 1,139 patients, 916 recordings from 913 patients have exactly one acute artery group. The counts below follow Table 2, using segment value 1 for acute occlusion and value 2 for chronic occlusion. Count a group as acute when any of its segments has value 1. Keep left main separate, and aggregate branches into their parent groups. [Data dictionary in the descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC13347058/#Tab2)

| Single acute group | Segment codes | Recordings | Patients |
|---|---|---:|---:|
| LM | LM | 5 | 5 |
| LAD | PLAD, MLAD, DLAD, DB | 347 | 347 |
| LCX | PLCX, MLCX, DLCX, OM | 209 | 209 |
| RCA | PRCA, MRCA, DRCA | 355 | 354 |

Of the remaining OMI recordings, 96 have no acute artery group and 139 name multiple acute groups. There are 13 recordings with an acute segment code outside `OMI = 1`, and 116 OMI recordings have a coexisting chronic segment code. Two patients have different single-artery labels across ECGs. The labeled OMI ECG-to-angiography interval ranges from 43 to 10,066 minutes, with a median of 474 minutes. These counts are metadata feasibility evidence. Check timing and ambiguous annotations before locking the source target.

The completed pilot independently re-decoded all 2,481 selected raw recordings and used 1,822 source training and 456 source validation patients. Its matched transfer arms reached .570389 and .580730 measured-XYZ patient macro F1, with .601801 for scratch. ACS-ECG's `AMI` means acute MI, while our six-class `AMI` means anterior MI. Vessel labels stayed native to the source task, and the PTB territory labels and frozen folds stayed fixed. [Verified report](../results/neural_acs_artery_transfer_v1/complete/REPORT.md)

The primary prospect for independent six-class validation is still a partner cohort with measured Frank XYZ and compatible adjudication. The intake and evaluation tools are already available. [Partner cohort protocol](measured_vcg_partner_cohort.md)

## STAFF III timing check

On 18 November 2026, checked all 142 WFDB event annotations and their 142 matching headers against the publisher's v1.0.0 SHA-256 inventory. The inventory file SHA-256 is `e4c71aa1b47b8fd17b1549b5f650770df63f4cb795388d129c6ca78cd7a78269`. The annotations contained 152 inflation markers, 152 deflation markers and 210 contrast markers across 104 patient prefixes. All checked headers report nine stored channels at 1 kHz, including I, II and V1–V6. [Versioned dataset and annotations](https://physionet.org/content/staffiii/1.0.0/)

For the first annotated inflation in each file, check that the same recording contains 10-second windows at [−20, −10) seconds and [+30, +40) seconds relative to its marker. Require the latter window to end by the next inflation or deflation marker and reject a recorded contrast marker anywhere in [−20, +40). Of 142 recordings, 74 had too little time before inflation and three failed the recorded contrast check. The remaining 65 recordings represent 61 patient prefixes.

These are timing candidates. Waveform quality, vessel support and possible procedural effects still need review. Injections can be unannotated, and a file can start later in a procedure. Limb leads use the Mason–Likar placement. Any encoder transfer would be an ECG-assisted extension with separate source validation and measured Frank XYZ as the sole target inference input. [Dataset protocol](https://physionet.org/content/staffiii/1.0.0/)
