# Measured-VCG data acquisition status

**19 November 2026.** Obtained a bounded public acquisition from the PhysioNet/CinC 2007 MI Imaging Challenge. The two labeled cases contain averaged body-surface waveforms and published gadolinium-enhanced MRI reference interpretations. The archive also supplies separate signed XYZ plots for these cases. The download was kept under the ignored local directory `data/raw/physionet_mi_imaging_2007_v1`. [Publisher dataset and license](https://physionet.org/content/challenge-2007/1.0.0/) · [MRI reference interpretations](https://physionet.org/content/challenge-2007/1.0.0/training-set.txt)

## Verified downloads

All seven files (6,476,508 bytes total) matched byte for byte across the versioned publisher endpoint and its official S3 mirror. The hashes below allow repeat verification. The two MATLAB arrays decode as 352 by 1,639 and 352 by 970, in microvolts. Their body-surface columns match the ASCII signals exactly after the first six fields. The ASCII layout includes three additional waveform columns. Their exact mapping to the signed XYZ plots is unverified. Preserve the original release files. The publication describes archived chronic infarction complexes and MRI segment annotations. Confirm the numeric XYZ map and sampling metadata before importing. The challenge page describes 2 kHz, and the follow-up methods paper reports 1 kHz. The local release has a time-field increment of 1 and needs an explicit unit and sampling-rate check before direct comparison. Site identity, acquisition phase and class semantics require separate review before model use. With two labeled participants, this is a data-format and reference-label fixture. [Data organization](https://physionet.org/content/challenge-2007/1.0.0/data/) · [Primary scar study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4646064/)

| File | Bytes | Local SHA-256 |
|---|---:|---|
| `case0001.dat` | 3,520,685 | `3a12ab5c9783c4ea67dd5fbcc779e2d1d20221fccdf2491263d733b7b72ddbeb` |
| `case0001_dat.mat` | 536,092 | `f309596be5b2652e46313a23c1e24e3751be24d9eae995d4eebc81dfdbdd82d1` |
| `case0001_xyz.pdf` | 18,982 | `42037b75276bc1a9022193be8be8bd4b8808458988eb8a8de500115d652f1b8f` |
| `case0002.dat` | 2,083,673 | `a4e0a6404b13c5cac6271d5a184785aecf449beee974d07b370a8e9e2d94be66` |
| `case0002_dat.mat` | 305,889 | `8cc851f9237367310ad98872e2c1567ce953e73352778490760ab3e7f8aa8a20` |
| `case0002_xyz.pdf` | 10,769 | `48bbbe6dbe3101e2e99c538cb80222b7e5a4897e758b48016eecfd9c5cc06f52` |
| `training-set.txt` | 418 | `8d2c9c8e73530e8bffdd0e901d0ccc93877da7c6acdc6e1cf280c88fde340aad` |

Publisher base `https://physionet.org/files/challenge-2007/1.0.0/`. The TXT file is at the root and the MAT, DAT and PDF files are under `data/`. The equivalent mirror base is `https://physionet-open.s3.amazonaws.com/challenge-2007/1.0.0/`. Retain the publisher's attribution with any reuse.

## Larger cohorts to ask about

| Lead | Evidence and access |
|---|---|
| Glasgow MALT cohort | The primary study describes four selected participants from a 38-patient cohort with body-surface mapping and delayed-enhancement MRI. The public challenge supplies four waveform cases and two public reference interpretations. Ask whether additional paired data and recorded or reproducibly reconstructed orthogonal leads can be released. The published cohort concerns chronic scar. [Primary report](https://pmc.ncbi.nlm.nih.gov/articles/PMC4646064/) |
| CSE diagnostic archive | The primary CSE study documents simultaneous orthogonal XYZ in 1,220 cases and broad anterior, inferior and combined MI categories. Public paired waveform and clinical reference downloads were not located. Confirm waveform and label access with the custodian, and treat this as a coarse regional endpoint if acquired. [CSE diagnostic study](https://www.nejm.org/doi/full/10.1056/NEJM199112193252503) |
| Helsinki/Aalto | The study includes 57 patients with body-surface mapping at multiple stages and gadolinium-enhanced MRI measurements. The publisher's available supplements are aggregate tables. A signed Frank XYZ export or defensible orthogonal derivation needs confirmation from the authors. [Primary report](https://pmc.ncbi.nlm.nih.gov/articles/PMC6931515/) |

## Laptop-sized partner intake

Use the [existing advisor brief](measured_vcg_advisor_brief.md) to request aggregate counts first. Ask for a feasibility batch of about **20–30 patients**, with a native measured XYZ recording, a pseudonymous patient ID, recording timing, waveform units and polarity, and a small table of independently reviewed region labels. Prioritize anterior, anterolateral and anteroseptal cases. Ask for a representative acquisition and original duration, with repeat recordings linked by patient if shared. Keep the imaging report or adjudication reference with the clinical partner when a de-identified regional label table suffices. Reserve separate patients or acquisition sites for later validation. A feasibility batch is for checking format, signal quality and label coverage.

Three channels at 1 kHz stored as 16-bit integers require about **360 kB per minute**. With one minute per patient, **30 patients are about 11 MB** and **100 are about 36 MB**, before headers and variation in sampling or encoding. A record-template workflow can process one patient at a time and retain only the required compact covariates. Current local free space was about 168–169 GiB during this review. [Partner intake protocol](measured_vcg_partner_cohort.md)
