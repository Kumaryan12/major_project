# Formal implementation presentation

`mi_implementation.tex` is a simple monochrome Beamer presentation matching
the implementation deck. Slides 1–14 are the main talk, and slides 15–17
contain technical backup. It includes the completed experiments, results,
qualifications, and the proposed next milestone. Speaker notes are embedded
with `\note{...}` and hidden from the audience PDF.

## Overleaf

Upload the ZIP as a new Overleaf project, select `mi_implementation.tex` as
the main document, and use the pdfLaTeX compiler. The `assets/` directory
must remain next to the `.tex` file. No shell escape or external font is
required.

## Local compilation

From this directory:

```bash
pdflatex mi_implementation.tex
pdflatex mi_implementation.tex
```

Alternatively, from the repository root:

```bash
tectonic output/latex/mi_implementation.tex --outdir output/pdf
```

To display notes on a second screen, replace `hide notes` in the source with
`show notes on second screen=right`. Notes pages change the compiled output,
so retain the audience version for distribution.

The grayscale waveform is a real PTB example generated with the project's
denoising and detector code. Recreate the asset with:

```bash
.venv/bin/python scripts/prepare_beamer_assets.py
```

That command requires the local PTB raw files. The uploaded project already
includes the figure and does not need the raw data to compile.
