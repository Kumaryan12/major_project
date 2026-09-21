-- Build the research progress deck as editable Keynote slides and export PPTX.
-- Run: osascript scripts/build_professor_slides.applescript

set projectRoot to "/Users/aryansatyendrakumar/Projects/Major_Project"
set draftPdf to projectRoot & "/tmp/presentations/mi_project_progress_review.pdf"
set finalPptx to projectRoot & "/output/pptx/mi_project_progress_review.pptx"

tell application "Keynote"
	set d to make new document with properties {document theme:theme "Academy", width:1920, height:1080}
		tell d
			set titleSlide to slide 1
			set object text of default body item of titleSlide to "MAJOR PROJECT PROGRESS REVIEW"
			set object text of default title item of titleSlide to "Patient-Independent MI Localization"
			set presenter notes of titleSlide to "Aryan Kumar and Kunal Maka. Project progress as of 22 September 2026. Research code: https://github.com/Kumaryan12/major_project"
		end tell
		tell titleSlide
			set teamLine to make new text item with properties {object text:"Aryan Kumar  •  Kunal Maka     |     NIT Goa     |     September 2026", position:{50, 955}, width:1800, height:65}
			set size of object text of teamLine to 30
			set color of object text of teamLine to {65535, 65535, 65535}
		end tell
end tell

my addBullets(d, "Research question", "The published method reports 99.80% accuracy using heartbeat-level evaluation." & return & "We ask whether the same VCG features work on completely unseen patients." & return & "The project reproduces the tensor method, then evaluates patient-level generalization.", "Source: Zhang et al., IEEE TBME 2023, DOI 10.1109/TBME.2022.3202962. The paper's headline result is under its beat-level setup.", 42)

my addBullets(d, "Reproduced tensor pipeline", "PTB measured Frank X, Y and Z signals at 1,000 Hz" & return & "Wavelet denoising, R-peak detection and 651-sample beats" & return & "Lead x time x subband tensor: 3 x 651 x 8" & return & "Tucker rank 3 across time gives 72 features" & return & "500-tree random forest as the TreeBagger analogue", "Matched PTB cohort: 178 subjects, 425 records, 60,507 detected beats. Paper reports 60,527. See RESULTS.md and configs/paper.yaml.", 35)

my addChartSlide(d, "The split changes the result", {"Random beats", "Unseen patients"}, {99.14, 41.39}, "Same PTB cohort and 72-feature model. Beat accuracy (%).", "Both values are beat-level accuracy on the 178-subject PTB cohort. Random beat CV lets the same patient contribute to train and test. Patient-grouped CV has zero patient overlap. Full precision: 99.136% and 41.385%. The published result is 99.80%.", 305, 270, 1370, 500)

my addBullets(d, "A learnable patient task", "The 12-class task includes MI locations represented by only one patient." & return & "The primary benchmark keeps six classes with at least 10 patients each." & return & "164 patients and 55,863 beats; no patient crosses a test fold." & return & "Primary endpoint: patient-level macro F1.", "Retained classes: AMI, ALMI, ASMI, IMI, ILMI, HC. See BENCHMARK_RESULTS.md and configs/patient_folds.csv.", 38)

my addChartSlide(d, "Wavelet statistics lead on PTB", {"Tucker + SVM", "Tucker + RF", "Tucker + XGBoost", "Wavelet stats + XGBoost"}, {0.366, 0.380, 0.403, 0.487}, "Six-class patient macro F1; representation choice needs external confirmation.", "All four results use 164 PTB patients under patient-disjoint evaluation. The wavelet-statistics representation was selected after exploratory ablations on these PTB folds. Nested rerun macro F1: Tucker XGBoost 0.4027; wavelet statistics XGBoost 0.4872. See BENCHMARK_RESULTS.md and ABLATION_RESULTS.md.", 420, 240, 1280, 510)

my addBullets(d, "What the feature study showed", "Twelve representations tested on frozen patient folds" & return & "Wavelet statistics retain lead and subband information without Tucker compression." & return & "Macro F1 improves from 0.403 to 0.487 in the nested PTB comparison." & return & "The tensor model remains the reproduced baseline and scientific comparator.", "Paired class-stratified bootstrap interval for macro-F1 difference: +0.003 to +0.165. This is not independent validation of the representation choice. See ABLATION_RESULTS.md.", 37)

my addBullets(d, "External PTB-XL result", "PTB-trained model tested on 1,092 PTB-XL patients with derived VCG" & return & "Patient macro F1: 0.242; patient accuracy: 56.23%" & return & "Healthy controls are 72.0% of this cohort." & return & "Always predicting healthy would reach 71.98% accuracy.", "Strict PTB-XL fold-10 test: 1,185 ECG records, 1,092 patients. Only 5 ALMI test patients and 0 correct ALMI classifications. No PTB-XL tuning. PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/. See EXTERNAL_RESULTS.md.", 38)

my addBullets(d, "Derived VCG explains only part of the gap", "Matched PTB patients: measured → measured VCG macro F1 = 0.469" & return & "Measured → derived VCG macro F1 = 0.422" & return & "Derived → derived VCG macro F1 = 0.467" & return & "The measured-to-derived change is −0.046; its 95% interval includes zero.", "All comparisons use the same 164 PTB patients and matched beat positions. The paired bootstrap interval is -0.108 to +0.017. This isolates the Kors conversion within PTB; it cannot explain all PTB-XL differences. See DOMAIN_SHIFT_RESULTS.md.", 37)

my addBullets(d, "Limits of the current model", "The PTB cohort is small for six-class patient prediction." & return & "External labels, recording length and measured versus derived VCG differ." & return & "Some external MI classes have very few patients." & return & "Current results do not support diagnostic use or a state-of-the-art claim.", "Cross-database identity overlap has not been independently verified. External fold 10 has been inspected; any later use should be labeled exploratory unless a new untouched set is obtained.", 38)

my addBullets(d, "Next research phase", "Build a compact 1D CNN/ResNet baseline on VCG signals." & return & "Test a tensor-plus-learned-feature hybrid under the same patient folds." & return & "Compare patient macro F1 with uncertainty and seek fresh external validation." & return & "Guide decisions: confirm labels, endpoint, compute and validation plan.", "The deep models and hybrid are proposed work, not yet implemented. Repository: https://github.com/Kumaryan12/major_project. Professor progress report: output/pdf/mi_localization_progress_report.pdf.", 38)

tell application "Keynote"
	export d to POSIX file draftPdf as PDF
	export d to POSIX file finalPptx as Microsoft PowerPoint
	close d saving no
end tell

on addBullets(d, slideTitle, bulletText, notesText, bodySize)
	tell application "Keynote"
		tell d
			set s to make new slide at end of slides with properties {base layout:slide layout "Title & Bullets"}
			set object text of default title item of s to slideTitle
			set size of object text of default title item of s to 66
			set object text of default body item of s to bulletText
			set size of object text of default body item of s to bodySize
			set presenter notes of s to notesText
		end tell
	end tell
end addBullets

on addChartSlide(d, slideTitle, labels, values, subtitleText, notesText, chartX, chartY, chartWidth, chartHeight)
	tell application "Keynote"
		tell d
			set s to make new slide at end of slides with properties {base layout:slide layout "Title Only"}
			set object text of default title item of s to slideTitle
			set size of object text of default title item of s to 66
			add chart s row names {"Result"} column names labels data {values} type horizontal_bar_2d group by chart row
			set c to last chart of s
			set position of c to {chartX, chartY}
			set width of c to chartWidth
			set height of c to chartHeight
			set presenter notes of s to notesText
		end tell
		tell s
			set t to make new text item with properties {object text:subtitleText, position:{80, 870}, width:1760, height:68}
			set size of object text of t to 28
			set font of object text of t to "Avenir Next"
		end tell
	end tell
end addChartSlide
