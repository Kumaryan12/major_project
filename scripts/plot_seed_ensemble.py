"""Plot aggregate seed-ensemble results; never read patient predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    summary=json.loads((root/"summary.json").read_text())
    if summary.get("complete") is not True: raise ValueError("Require completed verified ensemble results")
    names={"beat_only":"Beat only", "beat_aux":"Beat + auxiliary loss", "ordered_context":"Ordered context",
           "ordered_context_aux":"Ordered context + auxiliary", "shuffled_context":"Shuffled context",
           "shuffled_context_aux":"Shuffled context + auxiliary"}
    arms=list(names);y=np.arange(len(arms))
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True,gridspec_kw={"width_ratios":[1,1.2]})
    axes[0].plot([summary["mean_member_f1"][a] for a in arms],y,"o",color=".6",label="Mean individual-model F1")
    values=[summary["metrics"][a]["macro_f1"] for a in arms]
    axes[0].plot(values,y,"D",color="black",label="Uniform-ensemble F1")
    for i,v in enumerate(values): axes[0].annotate(f"{v:.3f}",(v,i),xytext=(5,-10),textcoords="offset points",fontsize=9)
    axes[0].set(yticks=y,yticklabels=[names[a] for a in arms],xlabel="Patient macro F1",title="Three declared seeds; all six arms")
    axes[0].invert_yaxis()
    axes[0].set_xlim(min([*values,*summary["mean_member_f1"].values()])-.01,max(values)+.015)
    axes[0].legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.16))
    axes[0].grid(axis="x",alpha=.2)
    pairs=[("ordered_context_aux_minus_beat_only","Combined recipe − beat only"),
           ("ordered_context_aux_minus_beat_aux","Context − beat only (auxiliary matched)"),
           ("ordered_context_aux_minus_shuffled_context_aux","Ordered − shuffled (auxiliary matched)")]
    for i,(key,label) in enumerate(pairs):
        r=summary["contrasts"][key]["six_class"];v=r["ensemble_delta_macro_f1"];lo,hi=r["descriptive_patient_interval"]
        axes[1].plot([lo,hi],[i,i],color=".4",linewidth=2);axes[1].plot(v,i,"o",color="black")
        axes[1].annotate(f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]",(v,i),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
    axes[1].axvline(0,color=".6",linestyle="--")
    axes[1].set(yticks=range(3),yticklabels=[p[1] for p in pairs],xlabel="Difference in ensemble macro F1",title="Descriptive paired patient intervals",ylim=(-.7,2.7))
    axes[1].invert_yaxis();axes[1].grid(axis="x",alpha=.2)
    fig.suptitle("Post-hoc development analysis: ensemble F1 differs from mean member F1")
    fig.savefig(root/"ensemble_comparison.png",dpi=170)
    fig.savefig(root/"ensemble_comparison.pdf")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("results/neural_residual_seed_ensemble_v1/complete"))
    args=parser.parse_args();plot(args.results)
