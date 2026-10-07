# Implementation presentation speaking notes

Presenter: Aryan Kumar. Project with Kunal Maka, under Dr Shivnarayan Patidar at NIT Goa.

Present slides 1–14 for a roughly 10–12 minute implementation section. Slides 15–17 are backup material for questions. Your friend covers the paper and problem statement before your section.

For an 8 minute slot, shorten slides 2 and 3, mention the ablation slide briefly, and leave slide 12 for discussion. The key story is the pipeline, the patient split, the feature improvement, and external validation.

Use ‘we’ for the project work. The presentation does not assign individual coding contributions.

## Slide 1 Implementation and results

Suggested time: 20 seconds.

My friend has explained the paper and the problem statement. I will now explain what we implemented and what the experiments showed. We built the published pipeline, evaluated generalization to unseen patients, compared feature representations, and tested the selected model on another database.

Transition: First, here is the complete system we built.

Evidence: README.md and saved experiment reports.

## Slide 2 The system we implemented

Suggested time: 35 seconds.

Our implementation has four parts. The first prepares the PTB cohort and processes Frank VCG. The second extracts the paper's tensor features and reproduces its random beat evaluation. The third adds patient-disjoint model selection and representation comparisons. The fourth performs locked external testing and the paired acquisition-domain experiment. All of these run through our Python command-line interface.

Transition: I will follow this sequence, starting with the data.

Evidence: README.md; src/mi_localization/cli.py

## Slide 3 PTB data preparation

Suggested time: 45 seconds.

We use the measured Frank X, Y and Z signals from PTB, sampled at one thousand hertz. We matched the paper's cohort: 178 patients, including 126 with MI and 52 healthy controls, across 425 recordings. We excluded the noisy patient 294 and MI patients without acute location annotations. We detected 60,507 complete beats, twenty fewer than the paper. The total is close, but our class counts differ because the paper did not publish its exact R-peak annotations.

Transition: The next slide shows how a recording becomes individual beats.

Evidence: RESULTS.md; configs/paper.yaml; data/raw/ptbdb/manifest.csv

## Slide 4 Signal processing and beat extraction

Suggested time: 55 seconds.

This is a real PTB recording processed by our code. Gray is the recorded signal and teal is the denoised signal. We use bior6.8 wavelet denoising, then a Pan-Tompkins-style detector on the three-dimensional vector magnitude. The dashed lines mark detected peaks, and the shaded window shows one beat. We retain 250 samples before the R peak and 400 after it, including the peak itself. That gives 651 samples for each lead. This trace illustrates the processing output, rather than serving as independently annotated detector validation.

Transition: Every extracted beat then becomes the tensor used by the paper.

Evidence: Actual record patient001/s0010_re; src/mi_localization/signal.py; configs/paper.yaml. Detector threshold 0.06 is calibrated to published beat counts.

## Slide 5 Tensor construction and Tucker features

Suggested time: 60 seconds.

For each three-lead beat, we retain the original signal and reconstruct wavelet detail subbands D3 through D9. This gives eight slices and a tensor of shape three by 651 by eight. We compress only the time mode to rank three using truncated SVD. The lead and subband modes remain intact. The resulting core is three by three by eight, which we flatten into 72 values. Our code fixes SVD sign ambiguity and uses column-major vectorization for deterministic features. A 500-tree bootstrap random forest is our scikit-learn analogue of MATLAB TreeBagger.

Transition: We first checked how close this implementation came to the published result.

Evidence: src/mi_localization/features.py; src/mi_localization/evaluation.py; configs/paper.yaml

## Slide 6 The evaluation split changes the result

Suggested time: 65 seconds.

The published accuracy is 99.80%. Our random beat ten-fold reproduction reaches 99.136%, only 0.664 percentage points lower. But almost every patient appears in both training and testing in those folds. When we group all recordings and beats of a patient together, the same 12-class model reaches only 41.385% beat accuracy. The drop is 57.75 percentage points. This is the main reason our project focuses on unseen patients. We closely reproduce the paper's reported regime, but that regime does not answer our patient-generalization question.

Transition: To make the patient task learnable, we defined a better-supported six-class benchmark.

Evidence: RESULTS.md; results/metrics_beat.json; results/metrics_patient.json. All three bars are beat accuracy on the 12-class task.

## Slide 7 Patient benchmark design

Suggested time: 60 seconds.

Several original locations have only one patient. If that patient is held out, the class has no training examples. We therefore retain six classes with at least ten patients each. The cohort has 164 patients and 55,863 beats. We freeze ten outer patient folds, and choose hyperparameters within each training fold using three inner patient folds. Training weights balance classes and patients. At testing, we average all beat probabilities for each patient before making one prediction. Macro F1 averages the six class F1 scores equally, so the large healthy class cannot dominate the endpoint.

Transition: This protocol lets us compare classifiers fairly.

Evidence: BENCHMARK_RESULTS.md; configs/patient_folds.csv; src/mi_localization/benchmark.py

## Slide 8 Classifier comparison on Tucker features

Suggested time: 45 seconds.

We compared random forest, linear SVM and XGBoost using the same 72 tensor features and the same patient folds. XGBoost leads on our primary metric at 0.403 macro F1. Random forest gets 0.380 and SVM 0.366. Random forest has slightly higher accuracy, but we chose macro F1 because classes are unequal in size. The bootstrap intervals overlap substantially, so this cohort does not prove that one classifier is definitively superior. The modest results suggested that changing the representation might help more than changing the classifier.

Transition: We therefore ran a representation ablation study.

Evidence: BENCHMARK_RESULTS.md; results/patient_benchmark/metrics_{rf,svm,xgboost}.json

## Slide 9 Representation study with 12 ablations

Suggested time: 50 seconds.

An ablation changes one aspect of the features while holding the evaluation setup fixed. We tested Tucker ranks, denoising, amplitude normalization, individual leads, wavelet removal, and a statistics-based representation. This chart shows five examples from all twelve. Wavelet statistics led the exploratory comparison at 0.464 macro F1. Removing wavelet slices was substantially worse. More Tucker rank added little. These are exploratory results with one fixed XGBoost model, which is why they differ from the nested benchmark numbers.

Transition: Next we reran the strongest representation using the nested benchmark protocol.

Evidence: ABLATION_RESULTS.md; results/ablation/comparison.csv. Fixed depth-6 XGBoost. Full 12-row table in backup slide 16.

## Slide 10 Wavelet statistics improved the PTB result

Suggested time: 65 seconds.

This representation keeps the same three leads and eight wavelet slices, but summarizes each slice with eight statistics instead of a low-rank Tucker core. Three times eight times eight gives 192 features. Using the nested protocol, macro F1 rises from 0.403 to 0.487 and accuracy rises from 46.95% to 54.88%. The paired bootstrap interval for the macro-F1 increase is 0.003 to 0.165. The result is encouraging within PTB, but we chose the representation after inspecting these folds. The nested rerun does not remove that selection effect. We therefore need external confirmation.

Transition: We tested the fixed PTB model on the locked PTB-XL cohort.

Evidence: ABLATION_RESULTS.md; results/wavelet_statistics_confirm/metrics_xgboost.json; src/mi_localization/features.py

## Slide 11 Locked external validation on PTB-XL

Suggested time: 65 seconds.

We train the selected wavelet-statistics model on all 164 eligible PTB patients and test it once on recommended PTB-XL fold ten. The strict test contains 1,185 records from 1,092 patients. PTB-XL provides 12-lead ECG, so we derive XYZ with the Kors matrix and resample from 500 to one thousand hertz. There is no external tuning. The patient macro F1 is only 0.242. The cohort is 72% healthy, and an always-healthy classifier has higher accuracy than our model. This is a useful negative result: the current localization model does not transfer reliably. Differences in signals, labels and population make it difficult to assign the gap to one cause.

Transition: We investigated one specific cause using simultaneous PTB signals.

Evidence: EXTERNAL_RESULTS.md; results/ptbxl_external/metrics.json; configs/ptbxl_external.yaml. PTB-XL v1.0.3. Cross-database identity overlap has not been independently verified.

## Slide 12 Paired measured and derived VCG study

Suggested time: 60 seconds.

PTB has simultaneous measured Frank VCG and 12-lead ECG. We derive Kors VCG from the ECG for the same 164 patients, 391 records and original beat positions. This holds patient identity, labels and timing constant. The fixed measured-to-measured model reaches 0.469 macro F1. Testing it on derived VCG gives 0.422, and training and testing in the derived domain gives 0.467. The estimated reduction is 0.046, but its interval includes zero. The pattern suggests acquisition mismatch may contribute. It cannot fully explain the external result. These scores use a fixed depth-three model, whereas the earlier 0.487 score used nested selection.

Transition: Alongside the experiments, we built a reproducible codebase.

Evidence: DOMAIN_SHIFT_RESULTS.md; results/domain_shift/metrics.json. Fixed depth-3 XGBoost and shared measured-domain R positions.

## Slide 13 Codebase and reproducible experiments

Suggested time: 40 seconds.

The implementation is a Python package with a command-line interface. It separates data preparation, signal processing, features and evaluation. YAML configurations fix processing parameters and model settings, and the patient assignments are saved in a CSV. Outputs include out-of-fold probabilities, metrics, confusion matrices and tuning decisions. We cache features to avoid repeating signal processing. We have fifteen passing automated tests. The repository also contains the experiment reports and presentation material. Large raw signals, caches and fitted models stay local.

Transition: I will close with the completed contribution and the next development step.

Evidence: README.md; pyproject.toml; src/mi_localization/; tests/. Test verification: 15 passed, 6 October 2026.

## Slide 14 Completed work and next milestone

Suggested time: 50 seconds.

We have completed the paper reproduction, established a patient-disjoint benchmark, found a stronger PTB feature representation, and documented its external limitations. The main conclusion is that nearly reproducing 99% beat accuracy does not establish reliability on new patients. The next milestone is a compact 1D CNN or ResNet and a model that combines learned signal features with tensor or wavelet features. Those models are proposed work. We should evaluate them using the same patient endpoint and reserve fresh validation for any improvement claim. This gives us a concrete path from the completed implementation to the next project phase.

Transition: That completes the implementation section. I am happy to discuss the design and results.

Evidence: All five experiment reports. Deep models and hybrid remain unimplemented as of 7 October 2026.

## Slide 15 Class-level results

Suggested time: discussion only.

These values make clear that the improvement is uneven. AMI and ILMI improve on PTB, whereas ASMI declines. External performance is especially weak for AMI, ALMI and ILMI, and ALMI has only five external test patients.

Transition: Use this slide only if asked about specific MI locations.

Evidence: ABLATION_RESULTS.md; saved classification reports for Tucker, wavelet statistics and PTB-XL

## Slide 16 Full exploratory ablation results

Suggested time: discussion only.

This full table contains all twelve representations. These scores use one fixed classifier and are exploratory. They should not be mixed with the nested model-selection estimates.

Transition: Use this table to answer questions about alternative features and ranks.

Evidence: results/ablation/comparison.csv; ABLATION_RESULTS.md

## Slide 17 Technical choices and reproduction limits

Suggested time: discussion only.

The paper omits several implementation choices. We made them deterministic and recorded them in configuration. In particular the detector threshold of 0.06 is calibrated to published beat counts. Matching the total does not recover the unpublished annotations. SVD sign fixing and column-major flattening help portable results. Some wavelet details depend heavily on boundary extension because nine levels is long for a 651-sample beat.

Transition: Use this slide for questions about exact reproducibility.

Evidence: README.md Reproduction boundaries; configs/paper.yaml; src/mi_localization/features.py

## Questions to rehearse

**Why is 99% accuracy not the final project result?** It comes from random beat folds where the same patients contribute beats to training and testing. Our intended task involves patients the model has never seen. The same 12-class baseline gets 41.39% beat accuracy under patient grouping.

**Is the original paper wrong?** We closely reproduce the result under its stated evaluation regime. Our work asks an additional generalization question. The score does not establish performance on unseen patients.

**Why did you move to six classes?** Some original classes have just one patient. Holding that patient out leaves no training patient for that class. We kept classes with at least ten patients for the primary endpoint.

**Why macro F1?** It computes F1 for each class and averages the six values equally. It prevents the large healthy class from dominating the summary.

**Does wavelet statistics prove a new state of the art?** It improves our PTB estimate by 0.084 macro F1. We selected the representation after looking at PTB folds, and external macro F1 is only 0.242. A fresh confirmation set and comparisons on identical label tasks are still needed.

**Why 0.487 in one PTB slide and 0.469 in the domain slide?** The first uses nested selection between two XGBoost candidates. The paired domain experiment freezes a depth-3 model. The paired comparison must use its internal 0.469 control.

**Did Kors conversion cause the external failure?** The paired PTB estimate decreases by 0.046 macro F1 when test signals switch to derived VCG, but its interval includes zero. Several other factors change in PTB-XL, so we cannot assign the external loss to conversion alone.

**What is complete and what remains?** The reproduction, patient benchmark, ablations, external evaluation, and paired domain analysis are complete. The proposed 1D CNN/ResNet and hybrid are future work.

## Delivery tips

Open Presenter View to use the notes. Show the plots and explain the finding instead of reading every number. Pause on slide 6, which establishes why our evaluation matters. Keep the difference between beat accuracy and patient macro F1 explicit. End slide 14 with the next research milestone and invite questions.
