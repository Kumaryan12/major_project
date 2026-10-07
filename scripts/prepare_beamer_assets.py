"""Generate a monochrome scientific figure and package the Beamer source."""

from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/latex"
BUILD = ROOT / "tmp/presentations/beamer"


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    assets = SOURCE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mi_localization.data import read_vcg
    from mi_localization.signal import denoise_vcg, detect_r_peaks

    raw, fs = read_vcg(ROOT / "data/raw/ptbdb", "patient001/s0010_re", ["vx", "vy", "vz"])
    clean = denoise_vcg(raw)
    peaks = detect_r_peaks(clean, fs, threshold_scale=0.06)
    center = int(peaks[3])
    left, right = max(0, center - 1000), center + 1000
    time = np.arange(left, right) / fs
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 5.2), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time, raw[left:right, i], color="0.70", lw=1.5, label="Recorded")
        ax.plot(time, clean[left:right, i], color="black", lw=1.0, label="Denoised")
        for r in peaks[(peaks >= left) & (peaks < right)]:
            ax.axvline(r / fs, color="0.4", lw=0.8, ls="--")
        ax.axvspan((center - 250) / fs, (center + 400) / fs, color="0.94")
        ax.set_ylabel(f"{'XYZ'[i]} (mV)", fontsize=12)
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("0.65")
        ax.tick_params(labelsize=11)
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=11)
    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.98, bottom=0.12, hspace=0.20)
    fig.savefig(assets / "vcg_processing_grayscale.png", dpi=190, facecolor="white")
    plt.close(fig)
    bundle = ROOT / "output/mi_implementation_latex.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as z:
        for path in [SOURCE / "mi_implementation.tex", SOURCE / "README.md", assets / "vcg_processing_grayscale.png"]:
            z.write(path, path.relative_to(SOURCE))
    print(bundle)


if __name__ == "__main__":
    main()
