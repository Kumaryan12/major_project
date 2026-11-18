# Independent measured Frank VCG cohort

Request de-identified waveform recordings from a clinical or research partner that can document patient-level infarct location from cardiac MRI or independent clinical adjudication. Ask separately about confirmed control recordings. Prioritize anterior and anterolateral MI, and seek coverage of AMI, ALMI, ASMI, IMI, ILMI and HC. Agree on how to handle infarct age, multiple territories and unclear evidence before looking at any model predictions. The six labels follow the project's current endpoint. The partner should retain an auditable mapping from original evidence to the assigned class and handle project access and approvals locally.

Preserve patient pseudonyms across visits. Reserve the external test patients and their fixed labels before choosing the model, feature processing, training budget or decision rule. Give a separate development assignment to any patients supplied for improving the model. Plan participant counts and uncertainty targets with the advisor before evaluation based on the numbers of eligible patients per class. A single case of each class only makes computation possible. Prefer an independently acquired site for the locked evaluation where feasible.

For the current intake, export WFDB recordings with measured VX, VY, VZ at 1,000 Hz in mV. Preserve acquisition orientation and amplitude. Keep the waveform directory and manifest under local ignored data storage. Use [the manifest header](partner_vcg_manifest_template.csv) and these rules.

- `record` is a unique relative WFDB path without the extension. `patient_id` and `site` are pseudonymous identifiers. Keep a consistent patient assignment across all rows.
- `partition` is `development` or `external_test`. This assignment controls patient use, rather than individual recording use.
- `review_status` is `confirmed`, `indeterminate` or `outside_endpoint`. Retain the latter categories in the intake receipt with empty label and evidence fields.
- For confirmed cases, set `label` to one of the six project classes and `evidence_type` to `cardiac_mri` or `clinical_adjudication` for MI and `control_review` for HC. These categories record the supplied review status. The software checks the format and waveforms.

```bash
.venv/bin/python -m mi_localization.vcg_external_intake \
  data/raw/partner_vcg_v1/manifest.csv data/raw/partner_vcg_v1/waveforms
```

The intake records class/partition coverage and checks unique record paths, consistent patient labels and splits, finite measured XYZ, basic duration and variation, matching PTB directory IDs and exact decoded copies against locally available PTB signals. Partner adjudication, site independence and identity assurance remain separate tasks. Keep the intake receipt private until partner release terms permit aggregate reporting.

## First pilot intake

Use a de-identified feasibility batch of about 20–30 patients. Start with one representative recording per patient at the original duration and use stable pseudonyms for any repeat visits. Ask the partner for acquisition details, study timing where available and their evidence for region labels. Keep timing as a separate table keyed by the same pseudonyms if the partner supplies it. Agree on whether the batch is available for later development. Use the development partition for the intake smoke and for any batch already inspected during feasibility. A future locked evaluation uses separate patient IDs agreed with the partner before model feedback.

The first intake receipt checks basic waveform eligibility and shows counts by label and review status. A real pilot can have partial class coverage. Confirm the native sampling rate and sign convention with the partner before conversion if the export differs from the currently supported WFDB format.

The one-minute pilot-size smoke test processed 30 synthetic WFDB records and the manifest in 10,808,181 bytes. The existing PTB exact-waveform screen read 448 locally cached unique XYZ signals. Both passes reproduced the same intake receipt. This fixture measured storage and exercised the loader.

## External evaluation

Once a real cohort is available, freeze its receipt, model hashes and preprocessing before reading external outcomes. The target inference model should use the frozen measured XYZ inputs and patient pooling. Report external six-class patient macro F1, class F1, patient counts and uncertainty, keeping development scores explicitly separate. Real participant access, labels and an external score remain pending.

The evaluation command loads one declared arm from a completed pilot and requires all 30 verified fold/seed checkpoints with their original training normalizers. It saves the model and cohort scope before inference, pools measured-XYZ probabilities across eligible beats and reports aggregate patient metrics. Choose the arm using development evidence and record it before the external run. Use a private output directory.

```bash
.venv/bin/python -m mi_localization.vcg_external_evaluation \
  data/raw/partner_vcg_v1/manifest.csv data/raw/partner_vcg_v1/waveforms \
  --study results/neural_vcg_native_pretraining_v1/complete --arm YOUR_LOCKED_ARM \
  --output data/processed/partner_vcg_intake_v1/evaluation
```
