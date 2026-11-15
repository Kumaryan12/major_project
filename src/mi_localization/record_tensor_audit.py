"""Feasibility of within-record template/residual tensors; no quality ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .neural import _hash, _save_json
from .regularization import load_bank
from .signal_audit import file_hash


def prototype_views(beats):
    """Preserve the complete measured waveform, template and physiological residual.

    No residual is called noise. A template is computed from this recording's
    own unlabeled beats only. Values/axes/time ordering are not normalized away.
    """
    beats = np.asarray(beats, dtype=np.float32)
    if beats.ndim != 3 or beats.shape[1] != 3 or not len(beats) or not np.isfinite(beats).all():
        raise ValueError("Expected finite beat × measured XYZ × time")
    template = np.median(beats, axis=0)
    return np.stack([beats, np.broadcast_to(template, beats.shape), beats-template], axis=1)


def diagnose(beats, config):
    a = np.asarray(beats, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (3,651) or not np.isfinite(a).all():
        raise ValueError("Expected original finite 651-sample XYZ beats")
    if len(a) < 4:
        return {"evaluated":False,"beats":len(a)}, None
    # Baseline subtraction is for this diagnostic only. The prototype retains
    # the original full physical waveform and its original 651-sample extent.
    a = a-np.median(a[:,:,50:130],axis=2,keepdims=True)
    lags = np.arange(-config["maximum_lag_samples"],config["maximum_lag_samples"]+1,config["lag_step_samples"])
    zero = int(np.flatnonzero(lags==0)[0])
    start,stop,trim = config["qrs_start"],config["qrs_stop"],config["core_trim"]
    selected,before,after,r0,r1 = [np.zeros(len(a)) for _ in range(5)]
    for parity in [0,1]:
        query = np.flatnonzero(np.arange(len(a))%2==parity)
        reference = np.flatnonzero(np.arange(len(a))%2!=parity)
        template = np.median(a[reference],axis=0)
        t = template[:,start:stop]
        tn = max(np.linalg.norm(t),1e-12)
        scores=[]
        for lag in lags:
            values=a[query,:,start+lag:stop+lag]
            scores.append(np.einsum("nct,ct->n",values,t)/np.maximum(np.linalg.norm(values.reshape(len(values),-1),axis=1)*tn,1e-12))
        scores=np.stack(scores,axis=1)
        best=scores.argmax(1)
        # Exact ties favor zero shift; there is no reason to move a flat trace.
        unchanged=np.isclose(scores[:,zero],scores.max(1),atol=1e-12,rtol=0)
        best[unchanged]=zero
        selected[query]=lags[best]
        before[query]=scores[:,zero]
        after[query]=scores[np.arange(len(query)),best]
        core=template[:,trim:651-trim]
        denominator=max(np.linalg.norm(core),1e-12)
        r0[query]=np.linalg.norm((a[query,:,trim:651-trim]-core).reshape(len(query),-1),axis=1)/denominator
        shifted=np.stack([a[i,:,trim+int(selected[i]):651-trim+int(selected[i])] for i in query])
        r1[query]=np.linalg.norm((shifted-core).reshape(len(query),-1),axis=1)/denominator
    summary={"evaluated":True,"beats":len(a),"median_abs_selected_lag_ms":float(np.median(np.abs(selected))),
        "fraction_abs_lag_ge10ms":float(np.mean(np.abs(selected)>=10)),"fraction_lag_search_boundary":float(np.mean(np.abs(selected)==config["maximum_lag_samples"])),
        "median_qrs_cosine_before":float(np.median(before)),"median_qrs_cosine_after":float(np.median(after)),"median_qrs_cosine_gain":float(np.median(after-before)),
        "median_fullcore_template_relative_difference_before":float(np.median(r0)),"median_fullcore_template_relative_difference_after":float(np.median(r1))}
    return summary,selected.astype(int)


def run_audit(config):
    if config != dict(config,maximum_lag_samples=20,lag_step_samples=2,qrs_start=170,qrs_stop=351,core_trim=20):
        raise ValueError("Freeze the ±20 ms alignment diagnostic and common-core design")
    source=yaml.safe_load(Path(config["source_config"]).read_text())
    waves,_,_,bank=load_bank(source)
    if waves.shape[1:]!=(3,651):
        raise ValueError("Require the original measured XYZ bank")
    beats=pd.read_csv(source["beats_path"])
    beats=beats.loc[beats.label.isin(source["classes"])].reset_index(drop=True)
    scope={"config":config,"bank":bank,"code_sha256":file_hash(__file__),"numpy":np.__version__}
    fingerprint=_hash(scope)
    root=Path(config["output_dir"])
    root.mkdir(parents=True,exist_ok=True)
    manifest=dict(scope,audit_fingerprint=fingerprint)
    guard=root/"audit_manifest.json"
    if guard.exists() and json.loads(guard.read_text())!=manifest:
        raise ValueError("Record tensor audit source/settings changed")
    _save_json(guard,manifest)
    review=root/"review"
    review.mkdir(exist_ok=True)
    rows=[]
    for number,(record,part) in enumerate(beats.groupby("record",sort=True),1):
        values=np.asarray(waves[part.index.to_numpy()])
        summary,_=diagnose(values,config)
        summary.update(record=record,patient_id=part.patient_id.iloc[0],label=part.label.iloc[0])
        rows.append(summary)
        if number==1:
            # A local executable prototype demonstrates axes and preservation;
            # it is not an evaluated model input bank or a denoised ground truth.
            views=prototype_views(values)
            np.savez_compressed(review/"prototype_views.npz",views=views)
            reconstruction=float(np.max(np.abs(views[:,1]+views[:,2]-views[:,0])))
        if number%100==0:
            print(f"Record tensor audit: {number}/{beats.record.nunique()}",flush=True)
    records=pd.DataFrame(rows)
    records.to_csv(review/"record_diagnostics.csv",index=False)
    valid=records.loc[records.evaluated]
    columns=[key for key in valid if key.startswith("median_") or key.startswith("fraction_")]
    patients=valid.groupby(["patient_id","label"])[columns].median().reset_index()
    patients.to_csv(review/"patient_diagnostics.csv",index=False)
    class_table=patients.groupby("label")[columns].median().reindex(source["classes"])
    class_table.to_csv(root/"class_diagnostics.csv")
    summary={"audit_fingerprint":fingerprint,"records":len(records),"evaluated_records":len(valid),"patients":int(patients.patient_id.nunique()),"beats":int(valid.beats.sum()),
        "patient_median_diagnostics":{key:float(patients[key].median()) for key in columns},"prototype_axes":["chronological_beat_within_one_record","original_template_residual","XYZ","original_time"],
        "prototype_reconstruction_max_abs_error_mv":reconstruction,"classification_evaluated":False,"source_data_changed":False,
        "limits":"Templates use disjoint odd/even beats within each recording. Lags maximize the reported QRS agreement metric, so improvement is optimistic by construction. Agreement/residuals are not noise, SNR, clinical quality or localization accuracy. No classification or learned quality weighting is evaluated."}
    _save_json(root/"summary.json",summary)
    d=summary["patient_median_diagnostics"]
    text=["# Within-record template/residual tensor feasibility","",
        f"Audited {len(records)} records, {len(patients)} patients and {len(beats):,} original beats. No waveform, cached R position, source label or patient eligibility changed. No running training process was modified.","",
        f"Median across patients of their record-median QRS cosine agreement was **{d['median_qrs_cosine_before']:.4f}** before any shifts. The median selected absolute shift was **{d['median_abs_selected_lag_ms']:.1f} ms**. The median patient's record-median fraction choosing |lag| ≥10 ms was **{d['fraction_abs_lag_ge10ms']:.1%}**.","",
        f"Median patient-level full-core template-relative difference was {d['median_fullcore_template_relative_difference_before']:.3f} before and {d['median_fullcore_template_relative_difference_after']:.3f} after the diagnostic shift. These are geometric differences, not measured noise ratios or SNR. The same QRS score selects and evaluates a lag, so its improvement is optimistic and cannot establish benefit.","",
        "Each recording is split into odd/even chronological beats. Each half is compared with the other half's median, avoiding direct self-inclusion in the reference. ±20 ms shifts in 2 ms steps act jointly on XYZ; no spatial rotation, amplitude scaling or time warping is performed. Exact ties favor zero shift. Comparisons use an approximate R-relative QRS region and a common 611-sample core; no claimed clinical delineation is used. Serially adjacent beats remain correlated.","",
        "The unaligned prototype preserves all 651 original samples and has axes beat × view × XYZ × time. Its views are X, the same recording's median M, and residual R=X−M. Mathematically X=M+R (within floating-point tolerance). This representation is redundant and creates no new patient information. Residuals can include pathology and physiological variability, not only artifacts. Both template and residual would need to be retained and tested.","",
        f"A first-record local prototype has maximum reconstruction error {reconstruction:.3g} mV. It is not a full evaluated feature bank, a quality-controlled clinical dataset, or proof of improved localization. Record/patient diagnostics and example arrays remain in ignored `review/`.","",
        "Proposed controlled comparison: (1) existing waveform model, (2) duplicate-waveform input with the new architecture's channel count, (3) template-only view, (4) original+template+residual. Temporal alignment and quality weighting should each be separate ablations after signal-preservation checks. Every patient remains in outer testing; all cohort-fitted choices belong inside training folds. Ordered beat modeling must use contiguous beats within a recording, not arbitrary bags or concatenated recordings.","",
        "High QRS agreement does not prove diagnostic quality, and filtering/averaging cannot create missing labels or independent patients. Our earlier phase-consensus coverage was only 23.2%; this prototype deliberately does not depend on those phase estimates. Source-label reliability still needs independent review.","",
        "Research basis: [VCG loop alignment](https://pubmed.ncbi.nlm.nih.gov/10763295/), [morphology-preserving ECG denoising](https://pmc.ncbi.nlm.nih.gov/articles/PMC11100958/), [pathology-sensitive quality assessment](https://pubmed.ncbi.nlm.nih.gov/27454007/). These motivate controls; their findings do not establish accuracy for this project.",""]
    (root/"REPORT.md").write_text("\n".join(text))
    return root/"REPORT.md"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/record_tensor_audit.yaml")
    args=parser.parse_args()
    print(run_audit(yaml.safe_load(Path(args.config).read_text())))
