# Windows collaborator setup

Read [CODEX_HANDOFF.md](CODEX_HANDOFF.md) before training. The research code has been tested on macOS; **this guide is not evidence that a native Windows run has passed**. The Keynote/AppleScript presentation builders require macOS. The zsh `.sh` launchers are conveniences; use Python commands in native PowerShell.

## Install and inspect without training

Install Python **3.12** and [Git for Windows](https://git-scm.com/install/win). The project supports Python >=3.11,<3.14; 3.12 matches the originating major/minor version. Run from a writable directory in PowerShell:

```powershell
git clone https://github.com/Kumaryan12/major_project.git
cd major_project
git switch -c work/windows-onboarding

py -3.12 -m venv .venv
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[neural,geometry,foundation,dev]"

.\.venv\Scripts\python.exe scripts/project_status.py
.\.venv\Scripts\mi-localization.exe --help
.\.venv\Scripts\python.exe -m pytest -q
```

The full extras install research/test dependencies. For only the original feature pipeline, `.[dev]` is smaller, but it is not sufficient for every neural/geometry/foundation-model test. Do not copy a macOS virtual environment or blindly install a macOS `pip freeze` containing Apple-specific packages.

Direct interpreter paths avoid PowerShell activation-policy changes. Set `PYTHONUTF8=1` in each new terminal before using Python 3.12 because reports contain Unicode. Source/config files are kept LF by the root `.gitattributes`, avoiding accidental CRLF changes to hashes. Do not renormalize the entire historical repository as an onboarding step.

For NVIDIA GPU training, use the command produced by the [official PyTorch installer selector](https://pytorch.org/get-started/locally/) for Windows and the installed GPU/driver, then install the project extras. Do not assume a generic pip install provides the right CUDA build. Check with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The neural configurations use `device: auto`, which chooses CUDA when available, otherwise CPU on Windows. Apple MPS is not available there. The current pinned TabPFN comparison uses CPU deliberately; changing it to GPU changes its declared runtime and requires a separate experiment. CPU-only work is possible; do not assume a full neural sweep will be fast.

## What a fresh clone contains

It contains source code, configurations, frozen folds, tests, documentation and public aggregate reports. It does **not** contain:

- `data/raw/ptbdb/`: raw PTB signals/headers and preparation metadata.
- `data/processed/`: beat tables, waveform/statistics/template/feature banks and their manifests.
- `data/models/`: public foundation checkpoint downloaded for local inference.
- Ignored result `.npz`, `.pt`, `.joblib`, `.tabpfn_fit`, patient prediction CSVs and bootstrap-cache files.
- Local `tmp/` logs or the process running on Aryan's Mac.

Use `scripts/project_status.py` to distinguish absent artifacts from a failed experiment. Reading public reports and developing/testing code does not require downloading every dataset.

## Data and old experiments

Raw data can be downloaded using the existing preparation command, **when the task requires it**:

```powershell
.\.venv\Scripts\mi-localization.exe prepare --config configs/paper.yaml
```

That is a network/data operation, not a quick status command. Feature extraction is a separate potentially expensive step:

```powershell
.\.venv\Scripts\mi-localization.exe features --config configs/paper.yaml
```

Do not immediately run `run --mode both` or every README experiment. Several historical default result paths already contain tracked reports/manifests. A new reproduction should use a new config/output namespace and must preserve original patients/labels/folds. The basic `features` command does not create all advanced banks or completed prerequisite studies.

For example, the original neural bank depends on existing beat rows and a relational statistics bank. The latest deviation loader also verifies completed refinement and stability predecessors. Inspect `load_inputs`, `prepare_bank` and the matching configs before constructing a rebuild plan. The full dependency chain is in [the handoff](CODEX_HANDOFF.md#6-repository-map-and-missing-local-state).

Raw/processed artifacts may be shared separately by the project owners, preserving their metadata and keeping them outside Git. Transfer does not guarantee resume compatibility: some fingerprints include file paths/mtimes, package versions, platform or numerical hashes. On a mismatch, identify the incompatible ancestor, rebuild or explicitly migrate under new configs/namespaces, and rerun verification. Never rewrite a manifest or remove a guard to force a match. Do not put Windows-generated results into an old Mac namespace.

## Continuing the completed deviation experiment

The deviation study has completed and been published with no demonstrated gain. A fresh clone cannot reproduce it from public aggregate JSON alone. First arrange the required artifact handoff or implement a separate, reviewed cold-start rebuild if a new reproduction is needed. Do not duplicate the full 1,960-fit sweep during orientation.

Once all prerequisites are genuinely available and matched under the chosen configs, the portable equivalents of its zsh launcher are the following **execution commands**, not installation checks:

```powershell
.\.venv\Scripts\python.exe -m mi_localization.deviation_study --smoke --config configs/deviation_vcg.yaml
.\.venv\Scripts\python.exe -m mi_localization.deviation_study --config configs/deviation_vcg.yaml
.\.venv\Scripts\python.exe -m mi_localization.deviation_report --config configs/deviation_vcg.yaml
```

Those default config paths describe the original experiment; use the prepared Windows config path when a new namespace/dependency chain is required. The Python entry points do not themselves auto-publish Git changes. Preserve completed reports, use task-specific branches and push only intended source/docs or validated aggregate results. Check Git status before staging so partial results, local patient data and model files are not accidentally included.

## Current ordered-signature study

The [signature study](signature_vcg_plan.md) is the current approved phase. It consumes verified original morphology and raw/denoised/template banks directly, without requiring completed refinement/stability classifier studies. Those source banks/data must still be transferred or built compatibly; a fresh clone alone is insufficient. Its source config is `configs/tensor_features.yaml`, and the 1,050-fit run uses `configs/signature_vcg.yaml`. Inspect local availability with `scripts/project_status.py` before doing any expensive work.

After the source-bank and namespace plan is ready, the native Python execution commands are:

```powershell
.\.venv\Scripts\python.exe -m mi_localization.signature_study --smoke --config configs/signature_vcg.yaml
.\.venv\Scripts\python.exe -m mi_localization.signature_study --config configs/signature_vcg.yaml
.\.venv\Scripts\python.exe -m mi_localization.signature_report --config configs/signature_vcg.yaml
```

Use the prepared Windows config path when rebuilding under a new namespace. Run the smoke first; the full study is not an onboarding check. A Mac background process is not transferred through Git, and model predictions need not be bitwise identical across platforms.

## Approved next: consecutive beats and RR

The [chronological study](chronological_vcg_plan.md) uses the existing denoised beat bank plus original beat metadata and patient folds. It does not need signature features or a completed signature run to train on a separate prepared machine. The signature dependency queue is only our current-machine sequencing policy.

On Windows, after source banks and a fresh compatible config/output namespace are ready:

```powershell
.\.venv\Scripts\python.exe -m mi_localization.chronology_study --smoke
.\.venv\Scripts\python.exe -m mi_localization.chronology_study
.\.venv\Scripts\python.exe -m mi_localization.chronology_report
```

Pass `--config` for the prepared Windows configuration. The portable runner `python scripts/run_chronological_vcg.py` combines these default-config steps. Do not use `--wait-for-signature` on a clone to wait on Aryan's Mac process. The full neural run is 480 fits, not an installation check. Native Windows remains unvalidated until its own tests/smoke pass.
