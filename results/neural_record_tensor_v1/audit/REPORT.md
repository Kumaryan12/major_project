# Within-record template/residual tensor feasibility

Audited 391 records, 164 patients and 55,863 original beats. No waveform, cached R position, source label or patient eligibility changed. No running training process was modified.

Median across patients of their record-median QRS cosine agreement was **0.9951** before any shifts. The median selected absolute shift was **0.0 ms**. The median patient's record-median fraction choosing |lag| ≥10 ms was **12.5%**.

Median patient-level full-core template-relative difference was 0.151 before and 0.129 after the diagnostic shift. These are geometric differences, not measured noise ratios or SNR. The same QRS score selects and evaluates a lag, so its improvement is optimistic and cannot establish benefit.

Each recording is split into odd/even chronological beats. Each half is compared with the other half's median, avoiding direct self-inclusion in the reference. ±20 ms shifts in 2 ms steps act jointly on XYZ; no spatial rotation, amplitude scaling or time warping is performed. Exact ties favor zero shift. Comparisons use an approximate R-relative QRS region and a common 611-sample core; no claimed clinical delineation is used. Serially adjacent beats remain correlated.

The unaligned prototype preserves all 651 original samples and has axes beat × view × XYZ × time. Its views are X, the same recording's median M, and residual R=X−M. Mathematically X=M+R (within floating-point tolerance). This representation is redundant and creates no new patient information. Residuals can include pathology and physiological variability, not only artifacts. Both template and residual would need to be retained and tested.

A first-record local prototype has maximum reconstruction error 1.49e-08 mV. It is not a full evaluated feature bank, a quality-controlled clinical dataset, or proof of improved localization. Record/patient diagnostics and example arrays remain in ignored `review/`.

Proposed controlled comparison: (1) existing waveform model, (2) duplicate-waveform input with the new architecture's channel count, (3) template-only view, (4) original+template+residual. Temporal alignment and quality weighting should each be separate ablations after signal-preservation checks. Every patient remains in outer testing; all cohort-fitted choices belong inside training folds. Ordered beat modeling must use contiguous beats within a recording, not arbitrary bags or concatenated recordings.

High QRS agreement does not prove diagnostic quality, and filtering/averaging cannot create missing labels or independent patients. Our earlier phase-consensus coverage was only 23.2%; this prototype deliberately does not depend on those phase estimates. Source-label reliability still needs independent review.

Research basis: [VCG loop alignment](https://pubmed.ncbi.nlm.nih.gov/10763295/), [morphology-preserving ECG denoising](https://pmc.ncbi.nlm.nih.gov/articles/PMC11100958/), [pathology-sensitive quality assessment](https://pubmed.ncbi.nlm.nih.gov/27454007/). These motivate controls; their findings do not establish accuracy for this project.
