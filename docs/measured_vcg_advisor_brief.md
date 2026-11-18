# Measured VCG cohort feasibility request

**For Dr. Shivnarayan Patidar**

We would like your help identifying a hospital or research group that records measured Frank X, Y and Z and can independently document myocardial infarct regions. Our six-class PTB study contains 164 patients and reached 0.6034 patient macro F1. Most errors were between MI locations. Additional patients with independently supported location labels would let us test whether training data scale and label quality improve localization.

For a first feasibility check, could the partner provide aggregate answers to the following?

- Which recordings contain measured Frank X, Y and Z? Please describe the acquisition system, signed lead convention, sampling rate, waveform units, typical duration and export format. Our current intake expects WFDB with measured VX, VY and VZ at 1,000 Hz in millivolts. We can review other native rates or formats before setting a conversion protocol.
- How many unique patients have late-gadolinium-enhancement cardiac MRI or independent clinical adjudication of infarct region? How many appear to cover anterior, anterolateral, anteroseptal, inferior, and inferolateral MI, and how many are independently reviewed healthy controls? Please also indicate counts with multiple or uncertain territories.
- Are the VCG acquisition time, infarct timing and label evidence available? Can repeated recordings be linked by patient pseudonym? Are there different sites or acquisition periods suitable for separate development and locked validation?
- Who would oversee waveform export, label adjudication and research access at the partner institution?

Aggregate availability counts are enough for this first discussion. We can agree on the class mapping and the treatment of unclear or multi-territory cases before modeling. If a cohort is feasible, we propose designating development patients separately from the external test patients before reviewing model predictions.

For error review, a clinician could assess a mixed set of cases without seeing model predictions or which cases the models missed. We would collect whether the original region is supported, whether more than one territory is documented, the recording timing and any signal concerns. These findings would guide data collection while preserving the frozen PTB evaluation.

[Technical intake and validation protocol](measured_vcg_partner_cohort.md)
