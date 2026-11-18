# Acquiring additional MI localization data

Reviewed 19 November 2026. This is an acquisition plan, not a new dataset or a training authorization. The ongoing frozen patient readout study continues unchanged. [Existing intake](measured_vcg_partner_cohort.md) · [Previously obtained small fixture](measured_vcg_data_acquisition.md) · [Unsent request templates](data_access_request_templates_20261119.md).

The search found additional measured-orthogonal archives worth requesting, but did not establish a large, currently downloadable, independently labeled six-class Frank-XYZ cohort. Archive survival, access, use rights and case-level compatibility remain questions for their custodians. Counts describe published studies, not files currently available to this project.

## Priority routes

| Priority | Route | Verified evidence | Next action and unresolved points |
|---|---|---|---|
| 1, in parallel with archives | Advisor introduction to a local cardiology partner, starting with Goa Medical College | Its official site lists a cardiology department. It does not establish measured Frank availability. | Ask Dr. Patidar for an introduction. Ask whether raw native measured XYZ exists, how many unique patients have region evidence, and whether a de-identified 20–30-person feasibility export is possible. If only standard ECG exists, record that as the separate ECG-assisted extension. |
| 1, new archive lead | UCLouvain / Cliniques universitaires Saint-Luc historical Louvain archive | Li et al. (1994) describe 2,810 patients with simultaneous standard ECG and Frank XYZ. The group without major conduction defects includes 279 anterior MI, 589 inferior MI and 1,042 normals. | Ask whether the original digital archive still exists and who holds it. Confirm raw signals, clinical references, finer regions, permission to train and patient overlap with other archives. Broad anterior/inferior categories cannot be assumed to match all five PTB subtypes. |
| 2 | CSE diagnostic archive | The original study describes 1,220 adults with simultaneous ECG/orthogonal XYZ at 500 Hz and independent clinical evidence. A 2017 primary paper gives a named access-request contact. | Request the diagnostic archive with measured XYZ and case-level clinical references; clarify whether training and reference-label release are allowed. Distinguish it from the smaller measurement/delineation library. Its broad MI groups need a separate task or independently reviewed finer labels. |
| 3 | Helsinki / HUS / Aalto BSPM-CMR cohort | Kylmälä et al. describe 57 patients with serial 120-lead body-surface recordings and CMR, including 44 with visible final infarct scar. | Ask the corresponding author about raw waveforms, regional CMR annotations and any simultaneously measured Frank channels. A BSPM export is not automatically native Frank XYZ. A documented reconstruction would require its own domain label and validation. |
| 3 | Glasgow electrocardiology / MI Imaging Challenge contributors | The public challenge provides four cases and two public MRI reference interpretations; Glasgow publishes a current enquiry route. | Ask about the parent cohort and additional exportable recordings/region labels. Confirm which orthogonal signals were physically measured versus reconstructed. Do not treat a contact listing as permission or a promise of data. |

Primary sources: [GMC departments](https://gmcgoa.edu.in/departments/); [Li et al., original Louvain paper](https://dial.uclouvain.be/pr/boreal/object/boreal%3A246773/datastream/PDF_01/view); [Saint-Luc research enquiry route](https://www.saintluc.be/fr/b-ctc-organisation-et-contacts); [original CSE diagnostic study](https://www.nejm.org/doi/full/10.1056/NEJM199112193252503); [CSE access statement](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0175087); [Helsinki paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6931515/); [challenge description](https://physionet.org/content/challenge-2007/1.0.0/); [Glasgow enquiries](https://www.gla.ac.uk/schools/healthwellbeing/research/robertsoncentreforbiostatistics/electrocardiology/contact/).

The Louvain cohort details were checked in the search-indexed text of the original institutional PDF. Direct PDF retrieval timed out; no local waveform archive or PDF download was obtained from that repository. The 1994 publication date must not be confused with a recent repository crawl date. Current archive custody is unverified.

## Public contact routes

| Lead | Public route | What it establishes |
|---|---|---|
| Local partner | Dr. Patidar introduction to GMC cardiology; [official department listing](https://gmcgoa.edu.in/departments/) | A plausible local institutional route. No verified Frank recorder or releasable cohort yet. |
| Louvain | `clinicaltrialcenter@saintluc.uclouvain.be`, from the [institutional research contact page](https://www.saintluc.be/fr/b-ctc-organisation-et-contacts) | A general routing address. Ask to be directed to cardiology/electrocardiology and the historical archive custodian. It is not a verified archive-access officer. |
| CSE | `prubel.lyon@gmail.com`, explicitly supplied in the [2017 paper's data-availability statement](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0175087) | A historical request address for Prof. Paul Rubel. Current responsibility/deliverability and license conditions remain unverified. |
| Helsinki | `minna.kylmala@hus.fi`, in the [original paper's correspondence details](https://pmc.ncbi.nlm.nih.gov/articles/PMC6931515/) | A publication-era author contact; current deliverability and sharing scope require confirmation. |
| Glasgow | `Peter.Macfarlane@glasgow.ac.uk`, on the [official electrocardiology enquiry page](https://www.gla.ac.uk/schools/healthwellbeing/research/robertsoncentreforbiostatistics/electrocardiology/contact/) | A published enquiry route; no demonstrated data-sharing agreement. |

Use an advisor-backed institutional enquiry. Begin with aggregate availability and a tiny example export rather than a full-archive request. These addresses were researched only; no correspondence has been sent.

## Open ECG alternative

[MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/) provides roughly 800,000 ten-second 12-lead ECGs from nearly 160,000 patients. The official page currently permits open waveform access under its license; the release removed sensitive cardiologist free text. Public machine-generated statements and note links are distinct from independently adjudicated infarct-region labels. Access to linked clinical modules has separate terms. This is a source-learning/ECG-extension candidate, not additional measured-Frank target data. Full uncompressed size is 90.4 GB.

If selected for a later bounded feasibility audit, start with metadata and a few headers. Audit unique-patient support, regional statement semantics, uncertainty/negation, acquisition, existing-source overlap and permissible labels. Only after that decision, consider 500–1,000 unique-patient signals. At 12 channels, 500 Hz, ten seconds and 16 bits, the uncompressed waveform payload is about 120 kB per ECG, or 120 MB for 1,000, before headers and encoding differences. Process in batches. No MIMIC download or fitting was started during this search.

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), ACS-ECG and STAFF III remain available sources with distinct annotations. We already tested bounded transfers from them without a demonstrated improvement over the fixed waveform reference. Expanding a source requires a new specific hypothesis and matched control; source size alone does not establish target benefit. [Existing source evidence](EXTERNAL_DATA_SHORTLIST.md).

## Minimal first request

Ask for aggregate counts of unique patients by region/evidence type, acquisition system and native measured-XYZ availability. If feasible, request one sample file and a de-identified 20–30-patient pilot with:

- Native signed X/Y/Z, units, sampling rate, electrode convention and any processing; explicitly identify derived channels.
- A pseudonymous patient ID linked across visits and a recording ID; recording timing relative to the clinical event.
- Original regional findings and their evidence source, infarct age, multiple/uncertain territories, and how controls were reviewed.
- A custodian-approved use scope covering model development or locked evaluation, storage and reporting.

Retain original labels and provenance. Have the partner/advisor agree on a clinical mapping before using model outcomes; keep indeterminate and multi-territory cases visible. Broad artery or anterior/inferior labels can support a separate source task without inventing the five PTB subtypes.

At 1 kHz, three channels and 16 bits, a one-minute XYZ record is 360 kB. Thirty such records are about 10.8 MB; 100 are about 36 MB, excluding headers and alternate encodings. The partner can retain raw images while sharing an auditable de-identified label/evidence table. A 20–30-person pilot tests feasibility, not six-class generalization or a promised score gain.

## Execution order

1. Use the advisor draft for a local introduction, and use the archive-specific templates for Louvain and CSE enquiries. The user/advisor sends them after selecting the recipient and checking institutional details.
2. Track availability replies, ownership, use terms, signal type and label coverage. An unanswered email is not evidence that an archive no longer exists.
3. Inspect one permitted example, then the bounded pilot using the existing private intake. Do not place patient data or clinical documents in Git.
4. Assign new development patients separately from a locked external test cohort before model feedback. Check identities/copies, sites, acquisition and label mapping. Keep the original PTB folds and historical results fixed.
5. Run a controlled patient learning curve only after eligible development data exist. Report site/class support and a matched baseline. A different label task or source metric cannot replace six-class measured-XYZ macro F1.

Repackaged PTB files and additional crops of existing patients do not increase independent patient coverage. Synthetic records can test mechanisms but cannot supply real external-patient confirmation. Prospective collection is a clinician-led option if no compatible archive survives; equipment availability and an institutional study pathway need feasibility confirmation first.
