# Research shortlist: preserving distributions, acquisition pairing and simulated interventions

Research review:16November2026. These are **proposed experiments**, not completed implementations, queued fits, established original methods or SOTA results. The running matched supervised-transfer study stays unchanged. Its current scores must not choose these hypotheses or be called untouched confirmation.

## Recommendation and project-specific gaps

The first practical candidate is **distribution-aware pooling of all measured-VCG beats**. A more ambitious thesis direction is physiology-based synthetic source learning with explicit anatomy/label audits. Paired acquisition alignment is a separately labeled ECG-assisted extension; compact within-beat dynamical features are another exploratory representation.

Project-specific reasons:

- The random-convolution bank encodes each recording's coordinate-median waveform. It compresses within-record variation before nonlinear feature extraction. The current transfer study classifies beats and averages their probabilities per patient.
- Unordered mean/attention bags, chronology4/8/16+RR, geometry descriptors, healthy-reference deviations, path signatures, staged context, contrastive objectives and several feature classifiers were already tested. A new proposal must distinguish itself from these controls; adding a Transformer or renaming views is not evidence of novelty.
- Measured Frank XYZ and ECG-derived XYZ are paired in PTB but differ as acquisition representations. Existing acquisition-domain comparisons and additive/bilinear ECG fusion did not establish consistent improvement; no successful paired shared/private alignment is established by those results.
- There are only164real target patients. More crops/projections or synthetic records are not additional independent target patients. Original folds, labels and R positions remain fixed.

## 1. Distribution-aware patient morphology tensors — first practical screen

**Question:** Does classifying a distribution of beat shapes preserve useful information that a recording median, latent mean or probability mean loses?

Encode every original beat into a compact signed/amplitude-sensitive vector, initially with data-independent physical/DCT features so no target-trained encoder can leak between partitions. A fixed external source encoder is an alternative with its existing provenance/identity limits. Compare per-record distributions using projected quantiles or a correctly implemented sliced-Wasserstein embedding; give recordings equal total weight when pooling per patient.

A proposed compact tensor is **record × fixed R-relative window × feature projection × quantile**. Its entries describe the spread of projected beat features within a recording. The quantile axis summarizes beats; it is not an extra independent sample axis. Mathematical projections in feature space are not arbitrary rotations/normalization of physical XYZ. Fixed R-relative windows must not be called clinically validated QRS/ST/T segments.

Start with one small classifier family and the same feature extractor/partitions:

1. Encode the recording median waveform, matching the current random-convolution compression point.
2. Encode individual beats, then pool their mean.
3. Pool individual-beat mean/spread as a moment control.
4. Pool compact projected quantiles/distribution embeddings, with a comparable feature budget.

This separates encoding-before-pooling from a richer pooling operator. Use all eligible beats; do not select confident/clean/class-favorable beats or remove hard patients. Fit any normalization, reference distribution, projection learning, factorization, selector and classifier only on the relevant training patients. If target-trained neural encoders are later used, their inner/outer memberships must match exactly; an outer-trained encoder is forbidden inside its inner validation partitions.

This differs from the previous jointly trained16-beat mean/attention bag study: it explicitly measures the within-patient distribution, can use all beats deterministically, and begins with a small-head feature experiment. It differs from healthy-reference subtraction: no healthy counterfactual is asserted. A compact finite-projection quantile representation must not claim the theoretical injectivity guarantees of a different full embedding.

Prior art: [Pooling by Sliced-Wasserstein Embedding, NeurIPS2021](https://papers.nips.cc/paper_files/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html) treats a set as a sampled distribution. [Fourier Sliced-Wasserstein Embedding, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3dbb8b6b5576b85afb3037e9630812dc-Abstract-Conference.html) provides distribution/multiset embeddings. Their published demonstrations are not MI-localization results. Our VCG tensor/aggregation design is a proposed adaptation, not an established novel technique or guaranteed gain.

## 2. Paired acquisition alignment with a preserved measured-only path — separate extension

**Question:** Can paired recordings teach a source-derived representation how to handle the measured-Frank acquisition domain without discarding its additional information?

Use each PTB training recording's simultaneous measured XYZ and its ECG-derived XYZ at the same original R positions. A small shared-component adapter/consistency objective aligns their embeddings while a separate measured-only component retains information that a Kors projection cannot represent. Target inference receives measured XYZ only. This must still be called **ECG-assisted training**, in the separate extension the user accepted.

Initial comparisons: matched measured-only model; correct-pair shared-component alignment; same-budget shuffled-pair alignment; and alignment without the preserved measured component as an information-loss ablation. Use only exact inner/outer training patients for all paired losses, normalization and stopping. Never use held-out patients' unlabeled paired signals to fit an adapter. Do not force waveform equality or declare all subject/axis variation irrelevant; such a constraint could erase location information.

This is different from concatenating ECG and VCG embeddings or adding a generic cross-patient contrastive loss. The intended intervention is paired acquisition consistency with an explicit path retaining measured-specific features. The terms "shared" and "private" denote model components, not clinically proven physiological/anatomical factors.

Research motivation: [Gyawali et al., Learning to Disentangle Inter-Subject Anatomical Variations in Electrocardiographic Data, IEEE TBME2022](https://pubmed.ncbi.nlm.nih.gov/34460360/) investigates anatomical variability and task-relevant ECG representations. It concerns a different localization task and does not validate this adapter or establish MI subtype gains.

## 3. Mechanistic synthetic source learning — strongest longer-term intervention idea

**Question:** Can controlled simulations teach an encoder which changes track infarction territory/severity and which reflect anatomy/acquisition/noise?

[MedalCare-XL v1.3, primary Zenodo deposit](https://zenodo.org/records/8068944) supplies mechanistic synthetic12-lead ECGs, simulation parameters and underlying-run identifiers. Its MI subclasses are **LAD/LCX/RCA occlusion × transmurality0.3/1.0**. These are not the six PTB classes AMI/ALMI/ASMI/IMI/ILMI/HC and must not be silently mapped by matching the number of labels. The deposit documents raw/noisy/filtered versions and anatomy-model-separated train/validation/test splits. The ZIP is9.3GB; this review did not download it.

First audit the schema, actual file counts, units/leads, simulation identifiers, anatomical grouping, and whether parameter files support matched healthy/infarct contrasts under the same geometry. If those contrasts exist, train on disease-change representations while retaining source anatomical variation; otherwise do not fabricate matched counterfactual pairs. Raw/noisy views of the same simulation can support a separately declared noise-consistency control.

Convert synthetic ECG to estimated XYZ only as an explicitly labeled source domain. Pretrain on its native verified occlusion/severity labels or controlled contrasts, then use a new matched target-transfer study. Group all noise variants/scenarios from one anatomy according to the documented native split, and inspect correspondence before merging any source partitions. Simulated anatomies/records are not new real PTB patients. Neither good synthetic discrimination nor reconstruction quality establishes real localization transfer.

Prior art: [MedalCare-XL primary paper/preprint](https://arxiv.org/abs/2211.15997) and [Improving Myocardial Infarction Detection via Synthetic ECG Pretraining,2025preprint](https://arxiv.org/abs/2506.23259). Synthetic pretraining is therefore not a first-ever contribution. The research opportunity is a carefully specified, validated intervention/source-transfer design for our endpoint, with simulation-to-real controls and no guaranteed0.80–0.90.

A full anatomical digital twin is a larger resource commitment: [PAA-Net/MICCAI2026 preprint](https://arxiv.org/abs/2605.22044) jointly models3Dmyocardial anatomy and multi-lead ECG. Its infarct-segmentation endpoint and data requirements differ from our three measured channels and six coarse classes; its Dice scores cannot be used as our macro F1 or imply that our labels support voxel-level scar supervision.

## 4. Within-beat dynamics: delay tensors and Koopman-style features

**Question:** Do within-beat transition dynamics add a useful bias beyond waveform shape, spatial summaries and ordered-path signatures?

Build a declared XYZ × time-delay × R-relative-time tensor from original beats. With fixed compact lag/rank budgets, extract regularized delay/DMD-style transition features or prediction residuals. Preserve signed XYZ and physical time/amplitude. A first screen needs no large Transformer. Any shared dictionary, reference, scaling or basis belongs to training patients only; a per-record computation must be declared as inference-time signal feature extraction, not a globally fitted model.

Compare signed waveform/DCT controls, lagged-linear controls with comparable feature budgets, and dynamic features. Include an appropriate order-destroyed control to test the claimed temporal mechanism. This concerns progression *inside* one beat, not cross-beat ordering already tested with4/8/16beats and RR. Nonstationary short cardiac windows and unstable operators may make this fail; modes/eigenvalues are not automatically physiological mechanisms.

Prior art: [Ghosh and Monfared, Electrocardiogram Classification with Transformers Using Koopman and Wavelet Features,2026preprint](https://arxiv.org/abs/2603.08339) evaluates normal/abnormal and rhythm classes, not six-way MI localization. It also reports that a simple hybrid did not improve accuracy. It motivates a bounded hypothesis, not importing its score or calling the idea new.

## Order and evidence required

Finish the declared transfer study unchanged. Then prioritize the small distribution-pooling screen. Keep paired alignment separate as ECG-assisted; perform only a metadata/feasibility audit before a large synthetic-data download or simulator investment. Delay/dynamical features are a bounded alternative, not another automatically queued sweep.

For every study, prespecify controls/seeds/budgets and report all patient folds/class results, matched macro-F1 differences and limitations. A novelty claim needs a closer prior-art comparison of the final implemented method. A SOTA claim additionally needs comparable evaluation and independent data; this shortlist establishes neither. No new model fitting, downloads, queue or numerical changes to active/pinned experiments are part of this review.
