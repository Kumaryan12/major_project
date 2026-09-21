"""Build a professor-facing snapshot from checked-in experiment results."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/mi_localization_progress_report.pdf"
NAVY = colors.HexColor("#13263B")
TEAL = colors.HexColor("#087E8B")
PALE = colors.HexColor("#EAF4F4")
GRAY = colors.HexColor("#536170")
LIGHT = colors.HexColor("#F1F4F7")


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pct(value: float, places: int = 2) -> str:
    return f"{100 * value:.{places}f}%"


def number(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


STYLE = {
    "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=NAVY, spaceAfter=9),
    "subtitle": ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=10.3, leading=15, textColor=GRAY, spaceAfter=12),
    "h1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=NAVY, spaceBefore=8, spaceAfter=7),
    "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=10.7, leading=14, textColor=TEAL, spaceBefore=9, spaceAfter=5),
    "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=9.2, leading=13.4, textColor=NAVY, spaceAfter=7),
    "small": ParagraphStyle("Small", fontName="Helvetica", fontSize=8, leading=11.2, textColor=GRAY, spaceAfter=5),
    "table": ParagraphStyle("Table", fontName="Helvetica", fontSize=8.25, leading=10.8, textColor=NAVY),
    "table_head": ParagraphStyle("TableHead", fontName="Helvetica-Bold", fontSize=8.1, leading=10.5, textColor=colors.white),
    "callout": ParagraphStyle("Callout", fontName="Helvetica-Bold", fontSize=9.4, leading=14, textColor=NAVY),
    "footer": ParagraphStyle("Footer", fontName="Helvetica", fontSize=7.3, leading=9, textColor=GRAY, alignment=TA_CENTER),
}


def para(text: str, kind: str = "body") -> Paragraph:
    return Paragraph(text, STYLE[kind])


def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    data = [[para(value, "table_head") for value in headers]]
    data += [[para(str(value), "table") for value in row] for row in rows]
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#D5DEE5")),
    ]))
    return result


def callout(text: str, width: float) -> Table:
    box = Table([[para(text, "callout")]], colWidths=[width])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("LINEBEFORE", (0, 0), (0, 0), 3, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return box


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setTitle("Patient-Independent MI Localization - Major Project Progress Report")
    canvas.setAuthor("Major Project Research Team")
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D5DEE5"))
    canvas.line(20 * mm, height - 19 * mm, width - 20 * mm, height - 19 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(20 * mm, height - 16 * mm, "MI LOCALIZATION | RESEARCH PROGRESS")
    canvas.line(20 * mm, 17 * mm, width - 20 * mm, 17 * mm)
    canvas.drawString(20 * mm, 12 * mm, "Research use only - not for clinical diagnosis")
    canvas.drawRightString(width - 20 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    beat = load("results/metrics_beat.json")
    patient = load("results/metrics_patient.json")
    rf = load("results/patient_benchmark/metrics_rf.json")
    svm = load("results/patient_benchmark/metrics_svm.json")
    xgb = load("results/patient_benchmark/metrics_xgboost.json")
    wavelet = load("results/wavelet_statistics_confirm/metrics_xgboost.json")
    external = load("results/ptbxl_external/metrics.json")
    audit = load("results/ptbxl_audit/summary.json")
    paired = load("results/domain_shift/metrics.json")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=24 * mm, bottomMargin=23 * mm,
        title="Patient-Independent MI Localization - Major Project Progress Report",
    )
    usable = A4[0] - 40 * mm
    story = []

    # Page 1: executive summary.
    story += [
        Spacer(1, 8 * mm),
        para("Major project progress report", "title"),
        para("Patient-Independent Localization of Myocardial Infarction from Vectorcardiographic Signals Using Tensor Decomposition and Machine Learning", "subtitle"),
        para(f"Status snapshot: {date(2026, 9, 22).strftime('%d %B %Y')}  |  Repository: github.com/Kumaryan12/major_project  |  Verified commit: a939096", "small"),
        HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=5, spaceAfter=14),
        para("Project question", "h1"),
        para("Can a VCG-based myocardial infarction (MI) localization method that scores highly on random heartbeat splits also classify <b>completely unseen patients</b>? We reproduced the published tensor-decomposition baseline, changed the evaluation unit to patients, investigated stronger representations and models, and tested transportability to another dataset."),
        para("Results at a glance", "h1"),
        table(
            ["Experiment", "Cohort / protocol", "Main result"],
            [
                ["Paper-style reproduction", "PTB; 178 patients; random beat 10-fold CV", f"{pct(beat['accuracy'], 2)} beat accuracy"],
                ["Unseen-patient test", "PTB; 178 patients; patient-grouped 10-fold CV", f"{pct(patient['accuracy'], 2)} beat accuracy"],
                ["Best six-class PTB model", "PTB; 164 patients; nested patient CV", f"{number(wavelet['patient_metrics']['macro_f1'])} patient macro F1"],
                ["External validation", "PTB to PTB-XL; 1,092 held-out patients", f"{number(external['patient_metrics']['macro_f1'])} patient macro F1"],
            ],
            [usable * .30, usable * .43, usable * .27],
        ),
        Spacer(1, 9 * mm),
        callout("Main finding: the paper-style beat-level score is nearly reproduced, but it does not translate to comparable performance on unseen patients. The current system is a research prototype, not a clinically reliable diagnostic model.", usable),
        Spacer(1, 8 * mm),
        para("What has been completed", "h1"),
        para("An end-to-end, reproducible codebase now includes PTB data preparation; wavelet denoising and R-peak detection; VCG beat tensor construction and Tucker features; random-beat and patient-independent evaluation; nested six-class model benchmarks; 12 representation ablations; a strict PTB-XL cohort audit; external testing; and a paired Frank-versus-Kors VCG domain-shift experiment."),
        para("What remains", "h1"),
        para("The planned improvement phase - deep signal models and tensor/deep hybrid models - has <b>not</b> been implemented. No state-of-the-art or clinical-performance claim is currently justified. The supervisor should review the six-class endpoint, label mapping, and next experiment before model development."),
        PageBreak(),
    ]

    # Page 2: reproduction and the generalization gap.
    story += [
        para("1. Paper reproduction and the generalization gap", "h1"),
        para("The reference study by Zhang et al. (IEEE TBME, 2023) combines wavelet-based VCG tensors, Tucker compression and a TreeBagger classifier. It reports <b>99.80% accuracy</b> on normal and 11 MI-location categories in the PTB database [1]. Our implementation follows that pipeline using Frank X/Y/Z signals."),
        table(
            ["Processing stage", "Implemented choice"],
            [
                ["Signal preparation", "1,000 Hz Frank XYZ; bior6.8 denoising; Pan-Tompkins-style R-peak detection"],
                ["Beat and tensor", "651 samples per beat (250 before, 400 after R); 3 leads x 651 time x 8 signal/subband slices"],
                ["Tensor feature", "D3-D9 wavelet details; Tucker time rank 3; 72 features per beat"],
                ["Classifier", "500-tree bootstrap random forest as the TreeBagger analogue"],
            ],
            [usable * .27, usable * .73],
        ),
        Spacer(1, 5 * mm),
        para("Cohort audit", "h2"),
        para("The matched PTB cohort has <b>178 subjects</b> (126 MI, 52 healthy), 425 VCG recordings and 60,507 detected beats. The paper reports 60,527 beats - a difference of 20 (0.033%). Exact beat annotations and several preprocessing details were unpublished, so this is a close reimplementation rather than a bitwise replication."),
        para("Evaluation changes the conclusion", "h2"),
        table(
            ["Evaluation", "Patient overlap", "Accuracy", "Macro F1"],
            [
                ["Random beat 10-fold CV", "Same patients can appear in train and test", pct(beat["accuracy"], 2), "0.991"],
                ["Patient-grouped 10-fold CV", "No patient shared across train and test", pct(patient["accuracy"], 2), "0.191"],
                ["Published paper result", "Beat-level protocol", "99.80%", "Not directly comparable"],
            ],
            [usable * .29, usable * .39, usable * .14, usable * .18],
        ),
        Spacer(1, 5 * mm),
        callout("The accuracy difference between our two protocols is 57.75 percentage points. This is evidence of severe within-patient dependence in the beat-level task, not proof that the original authors intentionally introduced leakage.", usable),
        Spacer(1, 6 * mm),
        para("Why we changed the primary task", "h2"),
        para("Several of the 12 classes have only one independent patient. Holding out that patient leaves no training example of the class. We therefore defined a six-class endpoint with at least ten patients per class: AMI, ALMI, ASMI, IMI, ILMI and healthy controls (HC). It contains 164 patients and 55,863 beats; patient-level macro F1 is the primary metric."),
        PageBreak(),
    ]

    # Page 3: six-class benchmark and ablations.
    story += [
        para("2. Patient-independent model and feature studies", "h1"),
        para("The six-class benchmark uses fixed patient-disjoint outer folds, inner patient-disjoint tuning, patient-balanced training weights and patient-level aggregation of beat probabilities. A healthy-only model reaches 31.71% patient accuracy on PTB; accuracy alone is not sufficient."),
        table(
            ["Model / representation", "Patients", "Accuracy", "Macro F1", "Macro ROC-AUC"],
            [
                ["Tucker + random forest", "164", pct(rf["patient_metrics"]["accuracy"]), number(rf["patient_metrics"]["macro_f1"]), number(rf["patient_metrics"]["macro_roc_auc_ovr"])],
                ["Tucker + linear SVM", "164", pct(svm["patient_metrics"]["accuracy"]), number(svm["patient_metrics"]["macro_f1"]), number(svm["patient_metrics"]["macro_roc_auc_ovr"])],
                ["Tucker + XGBoost", "164", pct(xgb["patient_metrics"]["accuracy"]), number(xgb["patient_metrics"]["macro_f1"]), number(xgb["patient_metrics"]["macro_roc_auc_ovr"])],
                ["Wavelet statistics + XGBoost", "164", pct(wavelet["patient_metrics"]["accuracy"]), number(wavelet["patient_metrics"]["macro_f1"]), number(wavelet["patient_metrics"]["macro_roc_auc_ovr"])],
            ],
            [usable * .40, usable * .10, usable * .16, usable * .16, usable * .18],
        ),
        Spacer(1, 6 * mm),
        para("Representation ablation", "h2"),
        para("We evaluated 12 representations on the same frozen PTB patient folds: Tucker ranks 1/2/3/5/8; raw versus denoised beats; amplitude normalization; individual leads; no wavelet subbands; and wavelet statistics without Tucker compression. The latter was strongest in the exploratory comparison. Its 192 features summarize each lead and subband with eight statistical descriptors of shape, amplitude and energy."),
        para("Nested confirmation", "h2"),
        para(f"With the same nested selection protocol, wavelet statistics increased patient macro F1 from <b>{number(xgb['patient_metrics']['macro_f1'])}</b> to <b>{number(wavelet['patient_metrics']['macro_f1'])}</b>, and patient accuracy from <b>{pct(xgb['patient_metrics']['accuracy'])}</b> to <b>{pct(wavelet['patient_metrics']['accuracy'])}</b>. The paired bootstrap interval for the macro-F1 difference is +0.003 to +0.165 (10,000 class-stratified patient resamples)."),
        callout("Important qualification: the representation was selected after viewing exploratory results on these PTB folds. The nested rerun helps compare model selection but is not an independent confirmation of the representation choice.", usable),
        Spacer(1, 7 * mm),
        para("What this means for the project title", "h2"),
        para("Tensor decomposition remains the reproduced baseline and a central scientific comparator. The strongest current six-class PTB result comes from wavelet statistics <b>without</b> Tucker compression. This is an informative finding, not a reason to relabel the paper baseline as the improved model."),
        PageBreak(),
    ]

    # Page 4: external test and domain-shift controls.
    story += [
        para("3. External validation and domain-shift analysis", "h1"),
        para("PTB-XL v1.0.3 provides 12-lead ECG rather than measured Frank VCG. We converted I, II and V1-V6 to estimated XYZ using the Kors regression, resampled from 500 to 1,000 Hz, and applied one fixed PTB-trained wavelet-statistics model to fold 10 without PTB-XL tuning [2, 3]."),
        table(
            ["Strict PTB-XL test cohort", "Value"],
            [
                ["Records / independent patients", f"{audit['locked_test_records']:,} / {audit['locked_test_patients']:,}"],
                ["Patients per class", "AMI 19; ALMI 5; ASMI 104; IMI 154; ILMI 24; HC 786"],
                ["Patient accuracy / macro F1", f"{pct(external['patient_metrics']['accuracy'])} / {number(external['patient_metrics']['macro_f1'])}"],
                ["Balanced accuracy / macro ROC-AUC", f"{pct(external['patient_metrics']['balanced_accuracy'])} / {number(external['patient_metrics']['macro_roc_auc_ovr'])}"],
            ],
            [usable * .45, usable * .55],
        ),
        Spacer(1, 5 * mm),
        para("The strict cohort accepts one target MI code or isolated NORM, excludes ambiguous/non-target MI records, and removes patients whose records have conflicting status. This improves label specificity but does not make PTB and PTB-XL diagnoses equivalent.", "small"),
        para("External result interpretation", "h2"),
        para("The cohort is 72.0% HC. An always-HC prediction would obtain 71.98% accuracy, above our 56.23%; the low 0.242 macro F1 is therefore the more informative summary. ALMI has only five test patients and zero correct classifications. The model is <b>not</b> suitable for diagnostic use."),
        para("Paired PTB control: measured versus derived VCG", "h2"),
        para("To isolate the VCG transformation from other differences, we compared simultaneous ECG and Frank recordings for the same 391 PTB records. We held patient folds, labels and 55,863 beat positions constant."),
        table(
            ["Training to testing domain", "Patient accuracy", "Patient macro F1"],
            [
                ["Measured to measured", pct(paired["conditions"]["measured_to_measured"]["patient_metrics"]["accuracy"]), number(paired["conditions"]["measured_to_measured"]["patient_metrics"]["macro_f1"])],
                ["Measured to Kors-derived", pct(paired["conditions"]["measured_to_derived"]["patient_metrics"]["accuracy"]), number(paired["conditions"]["measured_to_derived"]["patient_metrics"]["macro_f1"])],
                ["Kors-derived to Kors-derived", pct(paired["conditions"]["derived_to_derived"]["patient_metrics"]["accuracy"]), number(paired["conditions"]["derived_to_derived"]["patient_metrics"]["macro_f1"])],
            ],
            [usable * .51, usable * .24, usable * .25],
        ),
        Spacer(1, 5 * mm),
        para("Median record-level measured-versus-derived signal correlation was X "
             f"{number(paired['signal_medians']['correlation_x'])}, Y "
             f"{number(paired['signal_medians']['correlation_y'])}, and Z "
             f"{number(paired['signal_medians']['correlation_z'])}. The median matched-feature correlation was "
             f"{number(paired['paired_feature_shift']['median_feature_correlation'])}.", "small"),
        para("The measured-to-derived macro-F1 change is -0.046, with paired bootstrap 95% interval -0.108 to +0.017. It is consistent with a transform mismatch, but the interval includes zero. The much larger PTB-to-PTB-XL loss cannot be assigned to this factor alone; device, 10-second duration, label definitions and case mix also change.", "small"),
        PageBreak(),
    ]

    # Page 5: next phase and reproducibility.
    story += [
        para("4. Current contribution, limitations and next phase", "h1"),
        para("Current research contribution", "h2"),
        para("The project demonstrates that a near-reproduced high beat-level result is not evidence of unseen-patient reliability; provides a patient-disjoint six-class benchmark; tests a concrete feature improvement; and documents failure to transport well to a second dataset. All of these are completed experimental results, including the negative findings."),
        para("Limitations to state in the viva/report", "h2"),
        table(
            ["Issue", "Implication"],
            [
                ["Small PTB class cohorts", "The primary six-class task has 164 patients; uncertainty remains large."],
                ["Selection on PTB folds", "Wavelet-statistics gain is promising but needs a new untouched confirmation set."],
                ["PTB versus PTB-XL mismatch", "Measured versus derived VCG and clinical labels differ; scores are not controlled comparisons."],
                ["Sparse external classes", "Only five ALMI patients in the PTB-XL fold-10 strict test cohort."],
                ["No clinical validation", "The code is for research, not deployment or diagnosis."],
            ],
            [usable * .32, usable * .68],
        ),
        Spacer(1, 5 * mm),
        para("Proposed next milestone - not yet completed", "h2"),
        para("Freeze the target labels, patient splits and primary metric with the supervisor. Then implement a compact 1D CNN/ResNet VCG baseline and a tensor-plus-learned-feature hybrid, using patient-disjoint model selection and the same held-out-patient comparison. Only claim a substantial improvement if paired uncertainty and a genuinely fresh external evaluation support it. Published PTB-XL deep-learning numbers are for different label tasks and cannot be used as direct SOTA comparators [4]."),
        para("Reproducibility and meeting handoff", "h2"),
        para("Repository: <link href='https://github.com/Kumaryan12/major_project'>github.com/Kumaryan12/major_project</link>. The README contains setup and command sequences; all experiment configurations and machine-readable metrics are checked in. Raw datasets, cached features and fitted models are not committed. Latest verified state: commit a939096, clean working tree before this report, and 15 passing automated tests."),
        para("Suggested decisions for the guide meeting: confirm the six-class patient-level endpoint; review the strict PTB-XL label mapping; agree on the deep-model comparison and available compute; decide what independent validation will support the final claim.", "small"),
        HRFlowable(width="100%", thickness=.6, color=colors.HexColor("#D5DEE5"), spaceBefore=7, spaceAfter=7),
        para("References", "h2"),
        para("[1] Zhang et al., IEEE Transactions on Biomedical Engineering 70(3), 812-823 (2023), DOI: 10.1109/TBME.2022.3202962. <link href='https://pubmed.ncbi.nlm.nih.gov/36040933/'>PubMed record</link>.", "small"),
        para("[2] Wagner et al., PTB-XL v1.0.3, PhysioNet. <link href='https://physionet.org/content/ptb-xl/1.0.3/'>Dataset and fold protocol</link>. [3] Tereshchenko Laboratory, <link href='https://github.com/Tereshchenkolab/Pacing_spike_removal_v2/blob/main/kors.m'>Kors transform implementation</link>.", "small"),
        para("[4] Strodthoff et al., Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. <link href='https://github.com/helme/ecg_ptbxl_benchmarking'>Reference benchmark repository</link>.", "small"),
    ]

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(OUTPUT)


if __name__ == "__main__":
    build()
