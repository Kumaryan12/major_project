"""Build a concise professor-facing summary of completed project work."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/mi_localization_summary_report.pdf"

NAVY = colors.HexColor("#142A43")
BLUE = colors.HexColor("#246B8E")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5D6A76")
PALE_BLUE = colors.HexColor("#EEF5F8")
PALE_GRAY = colors.HexColor("#F6F7F8")
GRID = colors.HexColor("#D9D9D9")
WHITE = colors.white


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pct(value: float, places: int = 2) -> str:
    return f"{100 * value:.{places}f}%"


def dec(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


STYLES = {
    "title": ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=23, leading=27,
        textColor=colors.black, spaceAfter=7,
    ),
    "subtitle": ParagraphStyle(
        "Subtitle", fontName="Helvetica", fontSize=11.2, leading=16,
        textColor=MUTED, spaceAfter=12,
    ),
    "meta": ParagraphStyle(
        "Meta", fontName="Helvetica", fontSize=8.5, leading=12,
        textColor=MUTED, spaceAfter=5,
    ),
    "h1": ParagraphStyle(
        "Heading 1", fontName="Helvetica-Bold", fontSize=15, leading=19,
        textColor=colors.black, spaceBefore=6, spaceAfter=7, keepWithNext=True,
    ),
    "h2": ParagraphStyle(
        "Heading 2", fontName="Helvetica-Bold", fontSize=11.2, leading=14,
        textColor=colors.black, spaceBefore=7, spaceAfter=4, keepWithNext=True,
    ),
    "body": ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=9.6, leading=14,
        textColor=INK, spaceAfter=7,
    ),
    "small": ParagraphStyle(
        "Small", fontName="Helvetica", fontSize=8.1, leading=11.2,
        textColor=MUTED, spaceAfter=4,
    ),
    "key": ParagraphStyle(
        "Key", fontName="Helvetica-Bold", fontSize=11.2, leading=16,
        textColor=NAVY, spaceBefore=7, spaceAfter=8,
    ),
    "table": ParagraphStyle(
        "Table", fontName="Helvetica", fontSize=8.35, leading=11,
        textColor=INK,
    ),
    "table_head": ParagraphStyle(
        "Table Head", fontName="Helvetica-Bold", fontSize=8.25, leading=10.5,
        textColor=WHITE,
    ),
    "footer": ParagraphStyle(
        "Footer", fontName="Helvetica", fontSize=7.2, leading=9,
        textColor=MUTED, alignment=TA_CENTER,
    ),
}


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(
        text,
        ParagraphStyle(
            "Bullet", parent=STYLES["body"], leftIndent=14, firstLineIndent=-8,
            bulletIndent=0, spaceAfter=5,
        ),
        bulletText="•",
    )


def data_table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    data = [[p(item, "table_head") for item in headers]]
    data.extend([[p(str(item), "table") for item in row] for row in rows])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def metric_strip(items: list[tuple[str, str]], width: float) -> Table:
    cells = []
    for value, label in items:
        cells.append(p(f"<font size='16'><b>{value}</b></font><br/><font color='#5D6A76'>{label}</font>", "table"))
    table = Table([cells], colWidths=[width / len(cells)] * len(cells), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return table


def page_frame(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.6)
    canvas.line(19 * mm, height - 17 * mm, width - 19 * mm, height - 17 * mm)
    canvas.line(19 * mm, 16 * mm, width - 19 * mm, 16 * mm)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(19 * mm, height - 13.5 * mm, "MAJOR PROJECT SUMMARY  |  MI LOCALIZATION")
    canvas.drawString(19 * mm, 11.5 * mm, "Research use only  |  Not for clinical diagnosis")
    canvas.drawRightString(width - 19 * mm, 11.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    beat = load("results/metrics_beat.json")
    grouped = load("results/metrics_patient.json")
    tucker_xgb = load("results/patient_benchmark/metrics_xgboost.json")
    wavelet = load("results/wavelet_statistics_confirm/metrics_xgboost.json")
    external = load("results/ptbxl_external/metrics.json")
    paired = load("results/domain_shift/metrics.json")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=22 * mm,
        bottomMargin=21 * mm,
        title="Patient Independent MI Localization Project Summary",
        author="Aryan Kumar and Kunal Maka",
        subject="Summary of completed major project work",
    )
    usable = A4[0] - 38 * mm
    story = []

    # Page 1
    story.extend([
        Spacer(1, 6 * mm),
        p("Patient Independent MI Localization Project Summary", "title"),
        p("Implementation and evaluation of vectorcardiographic tensor decomposition and machine learning", "subtitle"),
        p("Aryan Kumar 23ECE1006  |  Kunal Maka 23ECE1017  |  Department of Electronics and Communication Engineering  |  NIT Goa", "meta"),
        p("Guide Dr Shivnarayan Patidar  |  Status 1 October 2026  |  Repository github.com/Kumaryan12/major_project", "meta"),
        Spacer(1, 4 * mm),
        p("Executive summary", "h1"),
        p(
            "We implemented an end-to-end reproduction of Zhang et al. for myocardial infarction location classification from vectorcardiographic signals. We then replaced random heartbeat splitting with patient-disjoint evaluation, compared classifiers and feature representations, tested the selected model on PTB-XL, and investigated measured versus ECG-derived VCG. The project now has a reproducible research baseline and a clear improvement path, but it has not yet produced a clinically reliable or state-of-the-art model."
        ),
        p(
            "The central result is that evaluation design changes the conclusion. The reproduced tensor pipeline reaches 99.14% accuracy when beats from the same patients can appear in training and testing, but only 41.39% when test patients are completely unseen.",
            "key",
        ),
        metric_strip([
            ("99.14%", "Random beat accuracy"),
            ("41.39%", "Unseen patient split beat accuracy"),
            ("0.487", "Best PTB patient macro F1"),
            ("0.242", "External PTB XL patient macro F1"),
        ], usable),
        Spacer(1, 7 * mm),
        p("Work completed", "h1"),
        bullet("Reproduced the paper cohort and preprocessing on the PTB Diagnostic ECG Database."),
        bullet("Implemented wavelet denoising, R peak detection, beat extraction, VCG tensors, Tucker features and a 500 tree random forest."),
        bullet("Added leakage-free patient grouped evaluation and a six-class patient-level benchmark."),
        bullet("Compared random forest, linear SVM and XGBoost with nested patient-disjoint tuning."),
        bullet("Evaluated 12 signal and feature representations and confirmed the strongest PTB candidate."),
        bullet("Completed locked external testing on PTB-XL and a paired Frank versus Kors VCG study."),
        bullet("Added automated tests, fixed experiment configurations, machine-readable outputs, documentation, a professor report and a presentation."),
        Spacer(1, 5 * mm),
        p("Current status", "h2"),
        p("The research pipeline and all experiments listed above are complete and pushed to GitHub through commit <b>34eea1c</b>. The deep-learning and tensor-plus-learned-feature improvement phase remains future work."),
        PageBreak(),
    ])

    # Page 2
    story.extend([
        p("Implemented method and reproducibility", "h1"),
        p("The implementation follows the published signal-processing and tensor-decomposition design while recording choices that the paper did not fully specify."),
        data_table(
            ["Stage", "Implemented method", "Output"],
            [
                ["Data", "PTB measured Frank X Y Z signals at 1000 Hz", "178 patients and 425 records"],
                ["Preprocessing", "bior6.8 denoising and Pan Tompkins style R peak detection", "60507 complete beats"],
                ["Beat tensor", "651 samples by 3 leads by 8 original and wavelet slices", "3 x 651 x 8 tensor"],
                ["Paper feature", "Tucker compression of the time mode to rank 3", "72 features per beat"],
                ["Paper classifier", "500 tree bootstrap random forest", "Beat and patient grouped evaluation"],
                ["Improved feature", "Eight statistics for every lead and wavelet slice", "192 features per beat"],
            ],
            [usable * 0.20, usable * 0.55, usable * 0.25],
        ),
        Spacer(1, 6 * mm),
        p("Reproduction quality", "h2"),
        p("The final cohort contains 126 MI patients and 52 healthy controls. We detected 60507 beats, while the paper reports 60527, a difference of 20 beats or 0.033%. The paper does not publish its code, exact R peak annotations, wavelet threshold rule, fold assignments or all TreeBagger options, so exact numerical identity is not possible."),
        p("The implementation is deterministic and configuration-driven. Patient folds, experiment settings, metrics, predictions and confusion matrices are stored in the repository. Raw datasets, feature caches and fitted models are excluded because of size."),
        p("Evaluation protocols", "h2"),
        data_table(
            ["Protocol", "Purpose", "Patient overlap"],
            [
                ["Random beat 10 fold", "Reproduce the paper-style result", "Almost every patient appears on both sides"],
                ["Patient grouped 10 fold", "Measure generalization to unseen patients", "Zero overlap"],
                ["Nested six-class benchmark", "Tune models without using outer test patients", "Zero overlap in outer and inner splits"],
                ["Locked PTB-XL fold 10", "Test transportability without external tuning", "External patients only"],
            ],
            [usable * 0.27, usable * 0.44, usable * 0.29],
        ),
        Spacer(1, 6 * mm),
        p("Why the six-class task was required", "h2"),
        p("The original 12-class task includes MI locations represented by only one patient. A model cannot learn such a class when its only patient is held out. The primary patient benchmark therefore retains AMI, ALMI, ASMI, IMI, ILMI and healthy control, each with at least 10 patients. This gives 164 patients and 55863 beats. Patient-level macro F1 is the primary endpoint because it weights all six classes equally."),
        PageBreak(),
    ])

    # Page 3
    story.extend([
        p("Experimental results", "h1"),
        data_table(
            ["Experiment", "Cohort and protocol", "Accuracy", "Macro F1"],
            [
                ["Paper-style tensor baseline", "178 PTB patients random beat folds", pct(beat["accuracy"]), "0.991"],
                ["Same tensor baseline", "178 PTB patients grouped by patient", pct(grouped["accuracy"]), "0.191"],
                ["Tucker plus XGBoost", "164 PTB patients nested six-class CV", pct(tucker_xgb["patient_metrics"]["accuracy"]), dec(tucker_xgb["patient_metrics"]["macro_f1"])],
                ["Wavelet statistics plus XGBoost", "164 PTB patients nested six-class CV", pct(wavelet["patient_metrics"]["accuracy"]), dec(wavelet["patient_metrics"]["macro_f1"])],
                ["PTB to PTB-XL external test", "1092 PTB-XL patients locked fold 10", pct(external["patient_metrics"]["accuracy"]), dec(external["patient_metrics"]["macro_f1"])],
            ],
            [usable * 0.30, usable * 0.39, usable * 0.15, usable * 0.16],
        ),
        Spacer(1, 6 * mm),
        p("Patient-independent benchmark", "h2"),
        p("On the same six-class PTB task, random forest obtained 0.380 patient macro F1, linear SVM 0.366 and XGBoost 0.403. Their uncertainty intervals overlap, so the current cohort does not prove that one classifier is definitively superior. Healthy controls are consistently easier than MI locations."),
        p("Feature study", "h2"),
        p("Twelve representations were evaluated on frozen patient folds. Wavelet statistics without Tucker compression performed best. In the nested comparison, patient macro F1 increased from 0.403 to 0.487 and patient accuracy increased from 46.95% to 54.88%. The paired bootstrap interval for the macro F1 improvement was +0.003 to +0.165. Because the representation was selected after exploratory PTB comparisons, the improvement still needs independent confirmation."),
        p("External validation", "h2"),
        p("The fixed PTB model was tested once on 1185 ECG records from 1092 PTB-XL patients. Twelve-lead ECG was converted to estimated VCG with the Kors regression before applying the same feature pipeline. Patient macro F1 fell to 0.242. The cohort is 72.0% healthy, and an always-healthy classifier would achieve 71.98% accuracy, higher than the model's 56.23%. This confirms that raw accuracy is misleading and that the current model does not transport reliably."),
        p("Measured and derived VCG analysis", "h2"),
        data_table(
            ["Training domain", "Test domain", "Patient macro F1"],
            [
                ["Measured Frank VCG", "Measured Frank VCG", dec(paired["conditions"]["measured_to_measured"]["patient_metrics"]["macro_f1"])],
                ["Measured Frank VCG", "Kors-derived VCG", dec(paired["conditions"]["measured_to_derived"]["patient_metrics"]["macro_f1"])],
                ["Kors-derived VCG", "Kors-derived VCG", dec(paired["conditions"]["derived_to_derived"]["patient_metrics"]["macro_f1"])],
            ],
            [usable * 0.39, usable * 0.39, usable * 0.22],
        ),
        Spacer(1, 5 * mm),
        p("The measured-to-derived reduction is 0.046 macro F1, with a 95% interval from -0.108 to +0.017. It may contribute to the external gap, but it does not explain the much lower PTB-XL result. Device, recording length, labels, prevalence and case mix also change."),
        PageBreak(),
    ])

    # Page 4
    story.extend([
        p("Conclusions and next research phase", "h1"),
        p("What the project establishes", "h2"),
        bullet("The published pipeline has been reproduced closely under its beat-level evaluation setup."),
        bullet("Random beat splitting substantially overestimates performance for the intended unseen-patient task."),
        bullet("Wavelet statistics improve the current PTB patient-level result, but the choice is not independently confirmed."),
        bullet("The PTB-trained model performs poorly on a locked PTB-XL cohort and is not suitable for clinical use."),
        bullet("The Kors conversion explains only part of the domain difference within the limits of the paired PTB analysis."),
        p("Limitations", "h2"),
        data_table(
            ["Limitation", "Effect on interpretation"],
            [
                ["Small MI classes", "Patient-level estimates remain uncertain, especially for rare locations."],
                ["PTB-guided feature selection", "The wavelet-statistics gain requires a new untouched confirmation set."],
                ["Different databases", "PTB and PTB-XL differ in signals, labels, duration, devices and prevalence."],
                ["Sparse external labels", "The locked test includes only five ALMI patients."],
                ["Research prototype", "No diagnostic, deployment or state-of-the-art claim is justified."],
            ],
            [usable * 0.31, usable * 0.69],
        ),
        Spacer(1, 6 * mm),
        p("Recommended next milestone", "h2"),
        p("Freeze the six-class label definition, patient folds and patient macro F1 endpoint with the project guide. Then implement a compact 1D CNN or ResNet on VCG beats and a hybrid model that combines learned signal features with tensor or wavelet features. Compare all models on exactly the same patient folds, report paired uncertainty, and reserve a genuinely untouched dataset for confirmation."),
        p("Decisions needed from the project guide", "h2"),
        bullet("Confirm whether the final project will retain the six-class endpoint or use a different clinical label grouping."),
        bullet("Approve the deep-learning baseline, hybrid design and available compute budget."),
        bullet("Agree on the independent validation plan before further model selection."),
        p("Repository and supporting material", "h2"),
        p("Repository: <link href='https://github.com/Kumaryan12/major_project'>github.com/Kumaryan12/major_project</link>. The repository contains setup instructions, reproducible commands, fixed configurations, results, tests, the detailed progress report and the presentation deck. Latest included commit: 34eea1c."),
        Spacer(1, 4 * mm),
        p("Reference", "h2"),
        p("Zhang et al. Automated Localization of Myocardial Infarction From Vectorcardiographic via Tensor Decomposition. IEEE Transactions on Biomedical Engineering. 2023;70(3):812-823. DOI 10.1109/TBME.2022.3202962.", "small"),
    ])

    document.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    print(OUTPUT)


if __name__ == "__main__":
    build()
