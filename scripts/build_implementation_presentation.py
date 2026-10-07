"""Prepare an implementation-only presentation for Aryan's project talk.

Uses local Keynote for native editable slide text, tables and charts. Run prepare,
then run its generated AppleScript with osascript, then run finalize. Rendering
the final PPTX in Keynote provides a separate visual inspection PDF.
"""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
import os
import posixpath
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "tmp/presentations/implementation"
OUTPUT = ROOT / "output/pptx/mi_implementation_presentation.pptx"
GUIDE = ROOT / "docs/implementation_presentation_notes.md"
NAVY = "17324D"
TEAL = "087F8C"
GRAY = "536778"
ORANGE = "AB562B"
FONT = "Avenir Next"
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def as_literal(value) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", '" & return & "') + '"'
    if isinstance(value, (tuple, list)):
        return "{" + ", ".join(as_literal(v) for v in value) + "}"
    return str(value)


def rgb(value: str) -> list[int]:
    return [int(value[i : i + 2], 16) * 257 for i in (0, 2, 4)]


def evidence_plot() -> str:
    """Plot real signals and the detector output, rather than simulated VCG."""
    os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mi_localization.data import read_vcg
    from mi_localization.signal import denoise_vcg, detect_r_peaks

    record = "patient001/s0010_re"
    raw, fs = read_vcg(ROOT / "data/raw/ptbdb", record, ["vx", "vy", "vz"])
    clean = denoise_vcg(raw)
    peaks = detect_r_peaks(clean, fs, threshold_scale=0.06)
    center = int(peaks[3])
    left, right = max(0, center - 1000), center + 1000
    t = np.arange(left, right) / fs
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 6.5), sharex=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for i, ax in enumerate(axes):
        ax.plot(t, raw[left:right, i], color="#C2CBD4", lw=1.15, label="Recorded signal")
        ax.plot(t, clean[left:right, i], color="#087F8C", lw=1.7, label="After denoising")
        for r in peaks[(peaks >= left) & (peaks < right)]:
            ax.axvline(r / fs, color="#AB562B", lw=0.9, ls="--", alpha=0.6)
        ax.axvspan((center - 250) / fs, (center + 400) / fs, color="#087F8C", alpha=0.09)
        ax.set_ylabel(f"{'XYZ'[i]}  (mV)", fontsize=13, color="#17324D")
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#BBC6D0")
        ax.tick_params(labelsize=11, colors="#536778")
        ax.grid(axis="y", color="#E9EEF2", lw=0.6)
    axes[0].legend(loc="upper right", frameon=False, fontsize=11, ncol=2)
    axes[-1].set_xlabel("Time in the PTB recording (seconds)", fontsize=13, color="#17324D")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.98, bottom=0.11, hspace=0.18)
    target = BUILD / "real_vcg_processing.png"
    fig.savefig(target, dpi=160, facecolor="white")
    plt.close(fig)
    return record


class Deck:
    def __init__(self):
        self.lines = [
            'tell application "Keynote"',
            'set d to make new document with properties {document theme:theme "Basic White", width:1920, height:1080}',
        ]
        self.slides = []
        self.charts = []

    def slide(self, title: str, speech: str, transition: str, source: str, time: int = 45, backup: bool = False):
        number = len(self.slides) + 1
        if number == 1:
            self.lines += ['set s to slide 1 of d', 'set base layout of s to slide layout "Blank" of d']
        else:
            self.lines += ['tell d', 'set s to make new slide at end of slides with properties {base layout:slide layout "Blank"}', 'end tell']
        self.slides.append({"number": number, "title": title, "speech": speech, "transition": transition, "source": source, "time": time, "backup": backup})
        notes = f"{'BACKUP SLIDE' if backup else f'SUGGESTED TIME {time} seconds'}\n\nSAY\n{speech}\n\nTRANSITION\n{transition}\n\nEVIDENCE\n{source}"
        self.lines.append(f"set presenter notes of s to {as_literal(notes)}")
        if number > 1:
            self.text(title, 100, 70, 1720, 125, 66, NAVY, True)
            self.text(f"{'Backup ' if backup else ''}{number:02}", 1640, 1000, 180, 45, 23, GRAY)
        return self

    def text(self, content: str, x: int, y: int, w: int, h: int, size: int = 38, color: str = NAVY, bold: bool = False):
        font = "AvenirNext-DemiBold" if bold else FONT
        self.lines += [
            'tell s',
            f'set t to make new text item with properties {{object text:{as_literal(content)}, position:{{{x}, {y}}}, width:{w}, height:{h}}}',
            f'set size of object text of t to {size}',
            f'set font of object text of t to {as_literal(font)}',
            f'set color of object text of t to {as_literal(rgb(color))}',
            'end tell',
        ]
        return self

    def table(self, rows, widths, x=100, y=270, row_h=92, size=32):
        self.lines += [
            'tell s',
            f'set tb to make new table with properties {{row count:{len(rows)}, column count:{len(widths)}, header row count:1, header column count:0, position:{{{x}, {y}}}, width:{sum(widths)}, height:{row_h * len(rows)}}}',
            'end tell',
            'tell tb',
            f'set font name of cell range to {as_literal(FONT)}',
            f'set font size of cell range to {size}',
            f'set text color of cell range to {as_literal(rgb(NAVY))}',
            'set vertical alignment of cell range to center',
            'set text wrap of cell range to true',
        ]
        for i, width in enumerate(widths, 1):
            self.lines.append(f'set width of column {i} to {width}')
        for r, row in enumerate(rows, 1):
            self.lines.append(f'set height of row {r} to {row_h}')
            self.lines.append(f'set background color of row {r} to {as_literal(rgb(NAVY if r == 1 else ("F1F6F8" if r % 2 == 0 else "FFFFFF")))}')
            if r == 1:
                self.lines.append(f'set text color of row {r} to {{65535, 65535, 65535}}')
            for c, value in enumerate(row):
                self.lines.append(f'set value of cell "{chr(65+c)}{r}" to {as_literal(str(value))}')
        self.lines.append('end tell')
        return self

    def chart(self, labels, values, max_value, fmt="0.000", highlight=0, x=100, y=245, w=1720, h=510):
        self.lines += [
            f'add chart s row names {{"Result"}} column names {as_literal(labels)} data {as_literal([values])} type horizontal_bar_2d group by chart row',
            'set c to last chart of s',
            f'set position of c to {{{x}, {y}}}',
            f'set width of c to {w}',
            f'set height of c to {h}',
        ]
        self.charts.append({"labels": labels, "values": values, "max": max_value, "format": fmt, "highlight": highlight, "slide": len(self.slides)})
        return self

    def footer(self, text):
        return self.text(text, 100, 925, 1600, 65, 27, GRAY)

    def finish(self):
        self.lines += [
            f'export d to POSIX file {as_literal(str(BUILD / "candidate.pptx"))} as Microsoft PowerPoint',
            'close d saving no',
            'end tell',
        ]
        (BUILD / "build.applescript").write_text("\n".join(self.lines))
        (BUILD / "metadata.json").write_text(json.dumps({"slides": self.slides, "charts": self.charts}, indent=2))
        GUIDE.parent.mkdir(parents=True, exist_ok=True)
        guide = [
            "# Implementation presentation speaking notes", "",
            "Presenter: Aryan Kumar. Project with Kunal Maka, under Dr Shivnarayan Patidar at NIT Goa.", "",
            "Present slides 1–14 for a roughly 10–12 minute implementation section. Slides 15–17 are backup material for questions. Your friend covers the paper and problem statement before your section.", "",
            "For an 8 minute slot, shorten slides 2 and 3, mention the ablation slide briefly, and leave slide 12 for discussion. The key story is the pipeline, the patient split, the feature improvement, and external validation.", "",
            "Use ‘we’ for the project work. The presentation does not assign individual coding contributions.", "",
        ]
        for s in self.slides:
            guide += [f"## Slide {s['number']} {s['title']}", "", f"Suggested time: {'discussion only' if s['backup'] else str(s['time']) + ' seconds'}.", "", s["speech"], "", f"Transition: {s['transition']}", "", f"Evidence: {s['source']}", ""]
        guide += [
            "## Questions to rehearse", "",
            "**Why is 99% accuracy not the final project result?** It comes from random beat folds where the same patients contribute beats to training and testing. Our intended task involves patients the model has never seen. The same 12-class baseline gets 41.39% beat accuracy under patient grouping.", "",
            "**Is the original paper wrong?** We closely reproduce the result under its stated evaluation regime. Our work asks an additional generalization question. The score does not establish performance on unseen patients.", "",
            "**Why did you move to six classes?** Some original classes have just one patient. Holding that patient out leaves no training patient for that class. We kept classes with at least ten patients for the primary endpoint.", "",
            "**Why macro F1?** It computes F1 for each class and averages the six values equally. It prevents the large healthy class from dominating the summary.", "",
            "**Does wavelet statistics prove a new state of the art?** It improves our PTB estimate by 0.084 macro F1. We selected the representation after looking at PTB folds, and external macro F1 is only 0.242. A fresh confirmation set and comparisons on identical label tasks are still needed.", "",
            "**Why 0.487 in one PTB slide and 0.469 in the domain slide?** The first uses nested selection between two XGBoost candidates. The paired domain experiment freezes a depth-3 model. The paired comparison must use its internal 0.469 control.", "",
            "**Did Kors conversion cause the external failure?** The paired PTB estimate decreases by 0.046 macro F1 when test signals switch to derived VCG, but its interval includes zero. Several other factors change in PTB-XL, so we cannot assign the external loss to conversion alone.", "",
            "**What is complete and what remains?** The reproduction, patient benchmark, ablations, external evaluation, and paired domain analysis are complete. The proposed 1D CNN/ResNet and hybrid are future work.", "",
            "## Delivery tips", "",
            "Open Presenter View to use the notes. Show the plots and explain the finding instead of reading every number. Pause on slide 6, which establishes why our evaluation matters. Keep the difference between beat accuracy and patient macro F1 explicit. End slide 14 with the next research milestone and invite questions.", "",
        ]
        GUIDE.write_text("\n".join(guide))


def prepare():
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    beat = load("results/metrics_beat.json")
    patient = load("results/metrics_patient.json")
    models = {m: load(f"results/patient_benchmark/metrics_{m}.json") for m in ["rf", "svm", "xgboost"]}
    wave = load("results/wavelet_statistics_confirm/metrics_xgboost.json")
    external = load("results/ptbxl_external/metrics.json")
    paired = load("results/domain_shift/metrics.json")
    record = evidence_plot()
    d = Deck()

    d.slide("Implementation and results",
        "My friend has explained the paper and the problem statement. I will now explain what we implemented and what the experiments showed. We built the published pipeline, evaluated generalization to unseen patients, compared feature representations, and tested the selected model on another database.",
        "First, here is the complete system we built.", "README.md and saved experiment reports.", 20)
    d.text("Implementation\nand results", 120, 270, 1660, 310, 114, "FFFFFF", True)
    d.text("Patient-independent MI localization", 125, 625, 1600, 100, 48, "FFFFFF")
    d.text("Presented by Aryan Kumar\nMajor project with Kunal Maka, NIT Goa", 125, 865, 1400, 105, 33, "CFDEE8")

    d.slide("The system we implemented",
        "Our implementation has four parts. The first prepares the PTB cohort and processes Frank VCG. The second extracts the paper's tensor features and reproduces its random beat evaluation. The third adds patient-disjoint model selection and representation comparisons. The fourth performs locked external testing and the paired acquisition-domain experiment. All of these run through our Python command-line interface.",
        "I will follow this sequence, starting with the data.", "README.md; src/mi_localization/cli.py", 35)
    d.table([
        ["Implemented stage", "What the code produces"],
        ["Data and signal processing", "Audited patient cohort, clean VCG, aligned beats"],
        ["Paper reproduction", "Wavelet tensors, 72 features, random forest results"],
        ["Patient benchmark and ablations", "Nested model comparison and 12 representations"],
        ["External and domain studies", "PTB-XL predictions and measured/derived comparison"],
    ], [620, 1100], y=240, row_h=115, size=38)
    d.footer("One Python pipeline with saved configurations, predictions and metrics")

    d.slide("PTB data preparation",
        "We use the measured Frank X, Y and Z signals from PTB, sampled at one thousand hertz. We matched the paper's cohort: 178 patients, including 126 with MI and 52 healthy controls, across 425 recordings. We excluded the noisy patient 294 and MI patients without acute location annotations. We detected 60,507 complete beats, twenty fewer than the paper. The total is close, but our class counts differ because the paper did not publish its exact R-peak annotations.",
        "The next slide shows how a recording becomes individual beats.", "RESULTS.md; configs/paper.yaml; data/raw/ptbdb/manifest.csv", 45)
    for x, value, label in [(100,"178","patients"),(705,"425","VCG recordings"),(1310,"60,507","complete beats")]:
        d.text(value, x, 270, 520, 150, 106, TEAL, True)
        d.text(label, x, 430, 520, 65, 40)
    d.text("Measured Frank X, Y and Z at 1,000 Hz",100,575,1650,70,46,NAVY,True)
    d.text("126 MI patients and 52 healthy controls\nAcute-location labels and paper cohort exclusions",100,680,1680,150,38)
    d.footer("Paper total: 60,527 beats. Our difference: 20 beats, or 0.033%")

    d.slide("Signal processing and beat extraction",
        "This is a real PTB recording processed by our code. Gray is the recorded signal and teal is the denoised signal. We use bior6.8 wavelet denoising, then a Pan-Tompkins-style detector on the three-dimensional vector magnitude. The dashed lines mark detected peaks, and the shaded window shows one beat. We retain 250 samples before the R peak and 400 after it, including the peak itself. That gives 651 samples for each lead. This trace illustrates the processing output, rather than serving as independently annotated detector validation.",
        "Every extracted beat then becomes the tensor used by the paper.", f"Actual record {record}; src/mi_localization/signal.py; configs/paper.yaml. Detector threshold 0.06 is calibrated to published beat counts.", 55)
    d.lines += ['tell s', f'set im to make new image with properties {{file:POSIX file {as_literal(str(BUILD / "real_vcg_processing.png"))}, position:{{80, 235}}, width:1230, height:640}}', 'end tell']
    d.text("bior6.8",1370,270,450,65,48,TEAL,True)
    d.text("Wavelet denoising",1370,355,420,70,36)
    d.text("R peaks",1370,480,420,65,48,TEAL,True)
    d.text("Vector-magnitude detector",1370,565,420,100,36)
    d.text("651 samples",1370,705,420,70,48,TEAL,True)
    d.text("250 before, R, 400 after",1370,795,440,80,34)
    d.footer("Real PTB example. Shading marks one beat window. Dashed lines mark detected peaks")

    d.slide("Tensor construction and Tucker features",
        "For each three-lead beat, we retain the original signal and reconstruct wavelet detail subbands D3 through D9. This gives eight slices and a tensor of shape three by 651 by eight. We compress only the time mode to rank three using truncated SVD. The lead and subband modes remain intact. The resulting core is three by three by eight, which we flatten into 72 values. Our code fixes SVD sign ambiguity and uses column-major vectorization for deterministic features. A 500-tree bootstrap random forest is our scikit-learn analogue of MATLAB TreeBagger.",
        "We first checked how close this implementation came to the published result.", "src/mi_localization/features.py; src/mi_localization/evaluation.py; configs/paper.yaml", 60)
    d.table([
        ["Stage", "Dimensions", "Implementation"],
        ["Wavelet tensor", "3 × 651 × 8", "X/Y/Z, time, original + D3–D9"],
        ["Tucker core", "3 × 3 × 8", "Time rank 3 through truncated SVD"],
        ["Classifier input", "72 features", "Flattened core for each beat"],
    ], [490,470,760],y=245,row_h=125,size=38)
    d.text("500-tree bootstrap random forest",100,815,1650,75,46,TEAL,True)
    d.footer("Lead and subband modes stay intact. SVD sign handling and vectorization are deterministic")

    d.slide("The evaluation split changes the result",
        "The published accuracy is 99.80%. Our random beat ten-fold reproduction reaches 99.136%, only 0.664 percentage points lower. But almost every patient appears in both training and testing in those folds. When we group all recordings and beats of a patient together, the same 12-class model reaches only 41.385% beat accuracy. The drop is 57.75 percentage points. This is the main reason our project focuses on unseen patients. We closely reproduce the paper's reported regime, but that regime does not answer our patient-generalization question.",
        "To make the patient task learnable, we defined a better-supported six-class benchmark.", "RESULTS.md; results/metrics_beat.json; results/metrics_patient.json. All three bars are beat accuracy on the 12-class task.", 65)
    d.chart(["Published paper","Our random beat CV","Our patient grouped CV"],[99.80,beat["accuracy"]*100,patient["accuracy"]*100],100,"0.00",1)
    d.text("57.75 percentage-point drop",100,820,1650,70,45,ORANGE,True)
    d.footer("Beat accuracy (%). Random beat folds share patients. Patient grouped folds have zero overlap")

    d.slide("Patient benchmark design",
        "Several original locations have only one patient. If that patient is held out, the class has no training examples. We therefore retain six classes with at least ten patients each. The cohort has 164 patients and 55,863 beats. We freeze ten outer patient folds, and choose hyperparameters within each training fold using three inner patient folds. Training weights balance classes and patients. At testing, we average all beat probabilities for each patient before making one prediction. Macro F1 averages the six class F1 scores equally, so the large healthy class cannot dominate the endpoint.",
        "This protocol lets us compare classifiers fairly.", "BENCHMARK_RESULTS.md; configs/patient_folds.csv; src/mi_localization/benchmark.py", 60)
    d.table([["Class","Patients"],["AMI",17],["ALMI",16],["ASMI",27],["IMI",29],["ILMI",23],["HC",52]], [440,260],y=255,row_h=79,size=33)
    d.text("164 patients and 55,863 beats",900,260,920,80,46,TEAL,True)
    d.text("10 outer patient folds\n3 inner folds for tuning\nEqual class and patient weights\nMean beat probabilities per patient",900,395,880,360,40)
    d.text("Primary endpoint: patient macro F1",900,820,920,100,38,NAVY,True)
    d.footer("Every outer test fold contains all six classes. No test patient contributes to model selection")

    d.slide("Classifier comparison on Tucker features",
        "We compared random forest, linear SVM and XGBoost using the same 72 tensor features and the same patient folds. XGBoost leads on our primary metric at 0.403 macro F1. Random forest gets 0.380 and SVM 0.366. Random forest has slightly higher accuracy, but we chose macro F1 because classes are unequal in size. The bootstrap intervals overlap substantially, so this cohort does not prove that one classifier is definitively superior. The modest results suggested that changing the representation might help more than changing the classifier.",
        "We therefore ran a representation ablation study.", "BENCHMARK_RESULTS.md; results/patient_benchmark/metrics_{rf,svm,xgboost}.json", 45)
    d.chart(["Random forest","Linear SVM","XGBoost"],[models[m]["patient_metrics"]["macro_f1"] for m in ["rf","svm","xgboost"]],0.6,"0.000",2)
    d.text("XGBoost leads on the chosen endpoint",100,830,1670,65,45,TEAL,True)
    d.footer("Patient macro F1, six classes. Confidence intervals overlap across all three classifiers")

    d.slide("Representation study with 12 ablations",
        "An ablation changes one aspect of the features while holding the evaluation setup fixed. We tested Tucker ranks, denoising, amplitude normalization, individual leads, wavelet removal, and a statistics-based representation. This chart shows five examples from all twelve. Wavelet statistics led the exploratory comparison at 0.464 macro F1. Removing wavelet slices was substantially worse. More Tucker rank added little. These are exploratory results with one fixed XGBoost model, which is why they differ from the nested benchmark numbers.",
        "Next we reran the strongest representation using the nested benchmark protocol.", "ABLATION_RESULTS.md; results/ablation/comparison.csv. Fixed depth-6 XGBoost. Full 12-row table in backup slide 16.", 50)
    ablation = list(csv.DictReader((ROOT / "results/ablation/comparison.csv").open()))
    lookup = {r["representation"]: r for r in ablation}
    keys=["wavelet_statistics","tucker_clean_r5","tucker_clean_r3","tucker_vector_norm_r3","tucker_no_wavelet_r3"]
    d.chart(["Wavelet statistics","Tucker rank 5","Tucker rank 3","Amplitude normalization","No wavelet subbands"],[float(lookup[k]["patient_macro_f1"]) for k in keys],0.6,"0.000",0,h=650,y=225)
    d.footer("Exploratory patient macro F1. Same frozen folds and one fixed XGBoost classifier")

    d.slide("Wavelet statistics improved the PTB result",
        "This representation keeps the same three leads and eight wavelet slices, but summarizes each slice with eight statistics instead of a low-rank Tucker core. Three times eight times eight gives 192 features. Using the nested protocol, macro F1 rises from 0.403 to 0.487 and accuracy rises from 46.95% to 54.88%. The paired bootstrap interval for the macro-F1 increase is 0.003 to 0.165. The result is encouraging within PTB, but we chose the representation after inspecting these folds. The nested rerun does not remove that selection effect. We therefore need external confirmation.",
        "We tested the fixed PTB model on the locked PTB-XL cohort.", "ABLATION_RESULTS.md; results/wavelet_statistics_confirm/metrics_xgboost.json; src/mi_localization/features.py", 65)
    d.text("3 leads × 8 slices × 8 statistics = 192 features",100,235,1720,95,53,TEAL,True)
    d.text("Mean, standard deviation, RMS, peak-to-peak, mean absolute value,\nlog energy, skewness and kurtosis",100,360,1670,130,36)
    d.table([["Nested PTB result","Tucker + XGBoost","Statistics + XGBoost"],["Patient macro F1",f'{models["xgboost"]["patient_metrics"]["macro_f1"]:.3f}',f'{wave["patient_metrics"]["macro_f1"]:.3f}'],["Patient accuracy",f'{100*models["xgboost"]["patient_metrics"]["accuracy"]:.2f}%',f'{100*wave["patient_metrics"]["accuracy"]:.2f}%']], [580,560,580],y=535,row_h=105,size=35)
    d.text("Gain: +0.084 macro F1",100,860,1650,50,37,TEAL,True)
    d.footer("Representation selection used these PTB folds. Fresh confirmation is still needed")

    d.slide("Locked external validation on PTB-XL",
        "We train the selected wavelet-statistics model on all 164 eligible PTB patients and test it once on recommended PTB-XL fold ten. The strict test contains 1,185 records from 1,092 patients. PTB-XL provides 12-lead ECG, so we derive XYZ with the Kors matrix and resample from 500 to one thousand hertz. There is no external tuning. The patient macro F1 is only 0.242. The cohort is 72% healthy, and an always-healthy classifier has higher accuracy than our model. This is a useful negative result: the current localization model does not transfer reliably. Differences in signals, labels and population make it difficult to assign the gap to one cause.",
        "We investigated one specific cause using simultaneous PTB signals.", "EXTERNAL_RESULTS.md; results/ptbxl_external/metrics.json; configs/ptbxl_external.yaml. PTB-XL v1.0.3. Cross-database identity overlap has not been independently verified.", 65)
    d.text("1,092",100,245,760,140,98,TEAL,True)
    d.text("external test patients",100,400,760,65,39)
    d.text(f'{external["patient_metrics"]["macro_f1"]:.3f}',1070,245,720,140,98,ORANGE,True)
    d.text("patient macro F1",1070,400,720,65,39)
    d.text("12-lead ECG, Kors-derived XYZ, shared beat and feature pipeline",100,535,1700,100,38)
    d.table([["Patient accuracy","Model","Always healthy"],["PTB-XL fold 10",f'{external["patient_metrics"]["accuracy"]*100:.2f}%',"71.98%"]],[660,530,530],y=665,row_h=102,size=38)
    d.footer("No PTB-XL tuning. Healthy controls comprise 72.0% of the strict cohort")

    d.slide("Paired measured and derived VCG study",
        "PTB has simultaneous measured Frank VCG and 12-lead ECG. We derive Kors VCG from the ECG for the same 164 patients, 391 records and original beat positions. This holds patient identity, labels and timing constant. The fixed measured-to-measured model reaches 0.469 macro F1. Testing it on derived VCG gives 0.422, and training and testing in the derived domain gives 0.467. The estimated reduction is 0.046, but its interval includes zero. The pattern suggests acquisition mismatch may contribute. It cannot fully explain the external result. These scores use a fixed depth-three model, whereas the earlier 0.487 score used nested selection.",
        "Alongside the experiments, we built a reproducible codebase.", "DOMAIN_SHIFT_RESULTS.md; results/domain_shift/metrics.json. Fixed depth-3 XGBoost and shared measured-domain R positions.", 60)
    d.chart(["Measured / measured","Measured / derived","Derived / derived"],[paired["conditions"][k]["patient_metrics"]["macro_f1"] for k in ["measured_to_measured","measured_to_derived","derived_to_derived"]],0.6,"0.000",0)
    d.text("Estimated change: −0.046 macro F1",100,825,1720,70,45,ORANGE,True)
    d.footer("Training / test domains. Paired 95% interval: −0.108 to +0.017. Same patients and beats")

    d.slide("Codebase and reproducible experiments",
        "The implementation is a Python package with a command-line interface. It separates data preparation, signal processing, features and evaluation. YAML configurations fix processing parameters and model settings, and the patient assignments are saved in a CSV. Outputs include out-of-fold probabilities, metrics, confusion matrices and tuning decisions. We cache features to avoid repeating signal processing. We have fifteen passing automated tests. The repository also contains the experiment reports and presentation material. Large raw signals, caches and fitted models stay local.",
        "I will close with the completed contribution and the next development step.", "README.md; pyproject.toml; src/mi_localization/; tests/. Test verification: 15 passed, 6 October 2026.", 40)
    d.text("Python package",100,250,750,65,47,TEAL,True)
    d.text("data.py and signal.py\nfeatures.py and evaluation.py\nbenchmark.py and ablation.py\nexternal.py and domain_shift.py",100,355,840,360,37)
    d.text("Repeatable experiments",1040,250,780,65,47,TEAL,True)
    d.text("Saved YAML settings and patient folds\nCached feature banks\nPredictions, metrics and confusion matrices\n15 automated tests passing",1040,355,780,350,37)
    d.text("mi-localization run --mode both\nmi-localization benchmark --models rf svm xgboost",100,795,1710,105,32,NAVY)
    d.footer("github.com/Kumaryan12/major_project")

    d.slide("Completed work and next milestone",
        "We have completed the paper reproduction, established a patient-disjoint benchmark, found a stronger PTB feature representation, and documented its external limitations. The main conclusion is that nearly reproducing 99% beat accuracy does not establish reliability on new patients. The next milestone is a compact 1D CNN or ResNet and a model that combines learned signal features with tensor or wavelet features. Those models are proposed work. We should evaluate them using the same patient endpoint and reserve fresh validation for any improvement claim. This gives us a concrete path from the completed implementation to the next project phase.",
        "That completes the implementation section. I am happy to discuss the design and results.", "All five experiment reports. Deep models and hybrid remain unimplemented as of 7 October 2026.", 50)
    d.text("Completed",100,270,790,80,52,TEAL,True)
    d.text("Paper pipeline and close reproduction\nEvaluation on unseen patients\nClassifier and representation studies\nExternal and paired domain experiments",100,405,810,345,39)
    d.text("Next milestone",1050,270,770,80,52,TEAL,True)
    d.text("Compact 1D CNN or ResNet\nTensor or wavelet plus learned features\nThe same patient endpoint and folds\nFresh independent confirmation",1050,405,770,345,39)
    d.text("The main contribution so far is a tested research baseline",100,825,1720,85,47,NAVY,True)
    d.footer("Current evidence supports research development. SOTA and clinical use remain unestablished")

    d.slide("Class-level results", "These values make clear that the improvement is uneven. AMI and ILMI improve on PTB, whereas ASMI declines. External performance is especially weak for AMI, ALMI and ILMI, and ALMI has only five external test patients.", "Use this slide only if asked about specific MI locations.", "ABLATION_RESULTS.md; saved classification reports for Tucker, wavelet statistics and PTB-XL", 0, True)
    rows=[["Class","PTB patients","Tucker F1","Statistics F1","External F1"]]
    for cl in models["xgboost"]["classes"]:
        baseline=models["xgboost"]["patient_metrics"]["classification_report"][cl]
        rows.append([cl,int(baseline["support"]),f'{baseline["f1-score"]:.3f}',f'{wave["patient_metrics"]["classification_report"][cl]["f1-score"]:.3f}',f'{external["patient_metrics"]["classification_report"][cl]["f1-score"]:.3f}'])
    d.table(rows,[260,350,350,400,360],y=250,row_h=86,size=34)
    d.footer("PTB scores use 164 patients. External scores use a different cohort and label mapping")

    d.slide("Full exploratory ablation results", "This full table contains all twelve representations. These scores use one fixed classifier and are exploratory. They should not be mixed with the nested model-selection estimates.", "Use this table to answer questions about alternative features and ranks.", "results/ablation/comparison.csv; ABLATION_RESULTS.md", 0, True)
    friendly={"wavelet_statistics":"Wavelet statistics","tucker_clean_r5":"Tucker rank 5","tucker_clean_r8":"Tucker rank 8","tucker_clean_r3":"Tucker rank 3","tucker_raw_r3":"Raw signal, rank 3","tucker_clean_r2":"Tucker rank 2","tucker_vector_norm_r3":"Amplitude normalized","tucker_z_r3":"Z lead only","tucker_clean_r1":"Tucker rank 1","tucker_x_r3":"X lead only","tucker_y_r3":"Y lead only","tucker_no_wavelet_r3":"No wavelet subbands"}
    rows=[["Representation","Features","Patient accuracy","Patient macro F1"]]+[[friendly[r["representation"]],r["feature_count"],f'{100*float(r["patient_accuracy"]):.2f}%',f'{float(r["patient_macro_f1"]):.3f}'] for r in ablation]
    d.table(rows,[760,280,340,340],y=210,row_h=53,size=28)
    d.footer("Frozen PTB folds with fixed depth-6 XGBoost. Representation choice is exploratory")

    d.slide("Technical choices and reproduction limits", "The paper omits several implementation choices. We made them deterministic and recorded them in configuration. In particular the detector threshold of 0.06 is calibrated to published beat counts. Matching the total does not recover the unpublished annotations. SVD sign fixing and column-major flattening help portable results. Some wavelet details depend heavily on boundary extension because nine levels is long for a 651-sample beat.", "Use this slide for questions about exact reproducibility.", "README.md Reproduction boundaries; configs/paper.yaml; src/mi_localization/features.py", 0, True)
    d.table([["Choice","Implementation and limit"],["Wavelet boundaries","Symmetric extension and soft thresholding"],["R peak detection","Vector magnitude, 0.06 threshold calibrated to beat counts"],["Feature determinism","SVD sign fixing and column-major core flattening"],["Classifier analogue","scikit-learn bootstrap forest instead of MATLAB TreeBagger"],["Exact replication","Unpublished annotations, folds and options limit exact matching"]],[550,1170],y=235,row_h=104,size=34)
    d.footer("Every choice is documented. Near matching the total beat count does not match every annotation")
    d.finish()
    print(BUILD / "build.applescript")


def fill(parent, color):
    for child in list(parent):
        if child.tag.rsplit("}",1)[-1] in {"solidFill","noFill","gradFill","pattFill","blipFill","grpFill"}:
            parent.remove(child)
    solid=ET.Element(f'{{{NS["a"]}}}solidFill')
    ET.SubElement(solid,f'{{{NS["a"]}}}srgbClr',val=color)
    parent.insert(0,solid)


def xml_bytes(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def finalize():
    meta=json.loads((BUILD / "metadata.json").read_text())
    with ZipFile(BUILD / "candidate.pptx") as z:
        parts={n:z.read(n) for n in z.namelist()}
    chart_names=sorted([n for n in parts if re.fullmatch(r"ppt/charts/chart\d+\.xml",n)],key=lambda n:int(re.search(r"chart(\d+)",n).group(1)))
    assert len(chart_names)==len(meta["charts"])
    for name, spec in zip(chart_names,meta["charts"]):
        root=ET.fromstring(parts[name])
        bar=root.find("c:chart/c:plotArea/c:barChart",NS)
        ser=bar.find("c:ser",NS)
        for node in root.findall(".//a:latin",NS):node.set("typeface",FONT)
        for ax in root.findall(".//c:valAx",NS):
            scale=ax.find("c:scaling",NS)
            for node in list(scale):
                if node.tag.rsplit("}",1)[-1] in {"min","max"}:scale.remove(node)
            # CT_Scaling requires orientation, max, min in this order.
            ET.SubElement(scale,f'{{{NS["c"]}}}max',val=str(spec["max"]))
            ET.SubElement(scale,f'{{{NS["c"]}}}min',val="0")
            unit=ax.find("c:majorUnit",NS)
            if unit is not None:unit.set("val","25" if spec["max"]==100 else "0.2")
            minor=ax.find("c:minorUnit",NS)
            if minor is not None:ax.remove(minor)
            fmt=ax.find("c:numFmt",NS)
            fmt.set("formatCode","0" if spec["max"]==100 else "0.0")
            # Give the number labels space below the axis.
            ax.find("c:tickLblPos",NS).set("val","low")
        for ax in root.findall(".//c:catAx",NS)+root.findall(".//c:valAx",NS):
            for tag in ["majorGridlines","minorGridlines"]:
                node=ax.find(f"c:{tag}",NS)
                if node is not None:ax.remove(node)
            for run in ax.findall(".//a:defRPr",NS):
                run.set("sz","3000")
                fill(run,GRAY)
        # Compact category labels aligned right beside their own bars.
        manual=root.find("c:chart/c:plotArea/c:layout/c:manualLayout",NS)
        if manual is not None:
            for k,v in {"x":"0.36","y":"0.055","w":"0.58","h":"0.82"}.items():
                node=manual.find(f"c:{k}",NS)
                if node is not None:node.set("val",v)
        fill(ser.find("c:spPr",NS),"91ACB9")
        # Native data point fills preserve editable source values and labels.
        for idx in range(len(spec["values"])):
            node=ET.Element(f'{{{NS["c"]}}}dPt')
            ET.SubElement(node,f'{{{NS["c"]}}}idx',val=str(idx))
            sp=ET.SubElement(node,f'{{{NS["c"]}}}spPr')
            fill(sp,TEAL if idx==spec["highlight"] else (ORANGE if spec["slide"]==6 and idx==2 else "91ACB9"))
            position=list(ser).index(ser.find("c:dLbls",NS))
            ser.insert(position,node)
        labels=ser.find("c:dLbls",NS)
        labels.find("c:showVal",NS).set("val","1")
        labels.find("c:dLblPos",NS).set("val","outEnd")
        labels.find("c:numFmt",NS).set("formatCode",spec["format"])
        for run in labels.findall(".//a:defRPr",NS):
            run.set("sz","3400")
            run.set("b","1")
            fill(run,NAVY)
        gap=bar.find("c:gapWidth",NS)
        if gap is not None:gap.set("val","85")
        parts[name]=xml_bytes(root)
    cover=ET.fromstring(parts["ppt/slides/slide1.xml"])
    csld=cover.find("p:cSld",NS)
    bg=csld.find("p:bg",NS)
    if bg is None:
        bg=ET.Element(f'{{{NS["p"]}}}bg')
        csld.insert(0,bg)
    bg.clear()
    props=ET.SubElement(bg,f'{{{NS["p"]}}}bgPr')
    fill(props,NAVY)
    ET.SubElement(props,f'{{{NS["a"]}}}effectLst')
    parts["ppt/slides/slide1.xml"]=xml_bytes(cover)
    # Native table border style: visible but restrained instead of black grid.
    for name in list(parts):
        if not re.fullmatch(r"ppt/slides/slide\d+\.xml",name):continue
        root=ET.fromstring(parts[name])
        for prop in root.findall(".//a:tcPr",NS):
            for edge in ["lnL","lnR","lnT","lnB"]:
                node=prop.find(f"a:{edge}",NS)
                if node is not None:
                    node.set("w","6350")
                    fill(node,"D7E1E7")
        parts[name]=xml_bytes(root)
    with ZipFile(OUTPUT,"w",ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    # Verify values and native chart workbooks against literal inputs.
    for name,spec in zip(chart_names,meta["charts"]):
        root=ET.fromstring(parts[name])
        categories=[v.text for v in root.findall(".//c:cat/c:strRef/c:strCache/c:pt/c:v",NS)]
        vals=[float(v.text) for v in root.findall(".//c:val/c:numRef/c:numCache/c:pt/c:v",NS)]
        assert categories==spec["labels"]
        assert all(abs(a-b)<1e-5 for a,b in zip(vals,spec["values"]))
        # The editable Excel workbook must agree with the cached chart data.
        rels=ET.fromstring(parts[posixpath.join(posixpath.dirname(name),"_rels",posixpath.basename(name)+".rels")])
        workbook_rel=next(r for r in rels if r.attrib["Type"].endswith("/package"))
        workbook_path=posixpath.normpath(posixpath.join(posixpath.dirname(name),workbook_rel.attrib["Target"]))
        xns={"x":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with ZipFile(BytesIO(parts[workbook_path])) as book:
            shared=ET.fromstring(book.read("xl/sharedStrings.xml"))
            strings=["".join(si.itertext()) for si in shared]
            sheet=ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
            cells={cell.attrib["r"]: (strings[int(cell.find("x:v",xns).text)] if cell.attrib.get("t")=="s" else float(cell.find("x:v",xns).text)) for cell in sheet.findall(".//x:c",xns)}
            for i,(label,value) in enumerate(zip(spec["labels"],spec["values"])):
                column=chr(66+i)
                assert cells[f"{column}1"]==label
                assert abs(cells[f"{column}2"]-value)<1e-10
    assert len([n for n in parts if re.fullmatch(r"ppt/slides/slide\d+\.xml",n)])==17
    assert len([n for n in parts if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml",n)])==17
    for i in range(1,18):
        assert b"SAY" in parts[f"ppt/notesSlides/notesSlide{i}.xml"]
    print(OUTPUT)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","finalize"])
    args=parser.parse_args()
    prepare() if args.stage=="prepare" else finalize()
