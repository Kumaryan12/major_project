"""Plot label-free global-compression fidelity from the saved bank audit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-dir",default="results/tensor_refinement_v1/bank_audit")
args = parser.parse_args()
root = Path(args.audit_dir)
summary = json.loads((root/"summary.json").read_text())
fidelity = summary["fidelity"]
ranks = sorted(int(name.removeprefix("global")) for name in fidelity if name.startswith("global"))
fig,axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
labels = {"raw":"Raw", "denoised":"Denoised", "record_median":"Recording median", "beat_residual":"Beat residual", "filter_residual":"Filter residual"}
for view,label in labels.items():
    axes[0].semilogy(ranks,[fidelity[f"global{rank}"][view]["mean_record_relative_squared_error"] for rank in ranks],marker="o",label=label)
    axes[1].semilogy(ranks,[fidelity[f"global{rank}"][view]["window_rms_error_mv"]["central"] for rank in ranks],marker="o",label=label)
for ax in axes:
    ax.set_xlabel("Global DCT coefficients per view / lead")
    ax.set_xticks(ranks)
    ax.grid(True,which="major",alpha=.25)
axes[0].set_ylabel("Mean record relative squared error")
axes[1].set_ylabel("Central-window RMS reconstruction error (mV)")
axes[0].set_title("Relative waveform loss")
axes[1].set_title("Absolute waveform loss")
axes[1].legend(fontsize=8)
fig.suptitle("Compression fidelity across all 391 recordings\nSignal reconstruction is not diagnostic accuracy",fontsize=11)
fig.savefig(root/"compression_fidelity.png",dpi=180)
plt.close(fig)
print(root/"compression_fidelity.png")
