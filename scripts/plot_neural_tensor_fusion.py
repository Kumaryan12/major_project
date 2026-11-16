"""Plot verified aggregate neural/tensor fusion metrics; no patient inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
import numpy as np


def plot(root):
    s=json.loads((root/"summary.json").read_text())
    if s.get("complete") is not True or s.get("frozen_references_reproduced") is not True:
        raise ValueError("Require a verified fusion report")
    names={"neural_only":"Frozen neural ensemble","tensor_only":"Frozen tensor / TabPFN", "equal_blend":"Fixed 50:50 blend", "selected_blend":"Inner-selected blend"}
    keys=list(names);values=[s["metrics"][k]["macro_f1"] for k in keys];y=np.arange(len(keys))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    axes[0].barh(y,values,color=[".65",".8",".5",".2"])
    for i,v in enumerate(values):axes[0].text(v+.015,i,f"{v:.4f}",va="center",fontsize=10)
    axes[0].set(yticks=y,yticklabels=[names[k] for k in keys],xlim=(0,1),xlabel="Patient macro F1",title="All 164 patients; four declared procedures")
    axes[0].invert_yaxis();axes[0].grid(axis="x",alpha=.2)
    pairs=[("selected_blend_minus_neural_only","Selected blend − neural"),("equal_blend_minus_neural_only","Equal blend − neural"),("selected_blend_minus_equal_blend","Selected blend − equal blend")]
    for i,(key,_) in enumerate(pairs):
        row=s["contrasts"][key]["six_class"];v=row["delta_macro_f1"];lo,hi=row["descriptive_patient_interval"]
        axes[1].plot([lo,hi],[i,i],color=".4",linewidth=2);axes[1].plot(v,i,"o",color="black")
        axes[1].annotate(f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]",(v,i),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
    axes[1].axvline(0,color=".6",linestyle="--")
    axes[1].set(yticks=range(3),yticklabels=[v[1] for v in pairs],xlabel="Difference in patient macro F1",title="Descriptive paired patient intervals",ylim=(2.6,-.6))
    axes[1].grid(axis="x",alpha=.2)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.suptitle("Patient-nested neural–tensor late fusion on inspected PTB development folds")
    fig.savefig(root/"fusion_comparison.png",dpi=170);fig.savefig(root/"fusion_comparison.pdf");plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("results/neural_tensor_fusion_v1/complete"))
    args=parser.parse_args();plot(args.results)
