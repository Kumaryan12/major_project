"""Label-free multiscale VCG tensors and interpretable feature families."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy
import yaml
from scipy.fft import dct

from . import multiview_bank
from .contrastive import _progress
from .neural import _hash, _save_json
from .signal_audit import file_hash

FAMILIES = ("statistics", "morphology", "multiscale", "relations", "combined")
VIEWS = multiview_bank.VIEW_NAMES
LEADS = ("X", "Y", "Z")
PAIRS = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
CROSS = ((0,1),(0,2),(1,2))


def validate_config(config, length):
    wavelet = pywt.Wavelet(config["wavelet"])
    if not 1 <= config["level"] <= pywt.dwt_max_level(length, wavelet.dec_len):
        raise ValueError("Tensor wavelet level exceeds supported crop length")
    if not 1 <= config["dct_coefficients"] <= length or config["chunk_size"] < 1:
        raise ValueError("Invalid tensor compression/chunk size")
    for name, (start, stop) in config["windows"].items():
        if not name or not 0 <= start < stop <= length or stop-start < 2:
            raise ValueError("Invalid fixed tensor window")
    if not config["windows"]:
        raise ValueError("Tensor windows are required")


def bands(config):
    return [f"A{config['level']}"]+[f"D{i}" for i in range(config["level"],0,-1)]


def feature_names(config, statistic_dimensions):
    names = {"statistics":[f"statistics/column_{i:03d}" for i in range(statistic_dimensions)]}
    if statistic_dimensions==192:
        names["statistics"] = [f"baseline/{lead}/{band}/{stat}" for lead in LEADS
            for band in ("original","D3","D4","D5","D6","D7","D8","D9")
            for stat in ("mean","std","rms","range","mean_abs","log_energy","skew","kurtosis")]
    names["morphology"] = [f"{v}/{lead}/dct_{k:02d}" for v in VIEWS for lead in LEADS for k in range(config["dct_coefficients"])]
    names["multiscale"] = [f"{v}/{lead}/{band}/{window}/{stat}" for v in VIEWS for lead in LEADS
        for band in bands(config) for window in config["windows"] for stat in ("mean","rms")]
    names["relations"] = [f"{v}/{band}/{window}/{kind}_{LEADS[i]}{LEADS[j]}" for v in VIEWS for band in bands(config)
        for window in config["windows"] for kind,pairs in (("cov",PAIRS),("corr",CROSS)) for i,j in pairs]
    return names


def tensorize(views, config):
    """N × view × XYZ × band × time; bands sum to the input views."""
    x = np.asarray(views,dtype=np.float64)
    if x.ndim != 4 or x.shape[1:3] != (5,3) or not np.isfinite(x).all():
        raise ValueError("Expected finite beat × five views × XYZ × time")
    validate_config(config,x.shape[-1])
    tensor = np.stack(pywt.mra(x,config["wavelet"],level=config["level"],axis=-1,
        transform="dwt",mode="symmetric"),axis=-2)
    error = float(np.max(np.abs(tensor.sum(-2)-x)))
    if not np.allclose(tensor.sum(-2),x,atol=1e-8,rtol=1e-8):
        raise ValueError("Multiscale tensor does not reconstruct its input")
    return tensor,error


def extract(views, config):
    x = np.asarray(views,dtype=np.float64)
    tensor,error = tensorize(x,config)
    coeff = dct(x,type=2,axis=-1,norm="ortho")[...,:config["dct_coefficients"]]
    morphology = coeff.reshape(len(x),-1)
    # Each cell retains signed mean plus physical RMS, without per-beat scaling.
    cells, relations = [], []
    for start,stop in config["windows"].values():
        window = tensor[...,start:stop]
        cells.append(np.stack([window.mean(-1),np.sqrt(np.mean(window**2,-1))],axis=-1))
        centered = window-window.mean(-1,keepdims=True)
        cov = np.einsum("nvlbt,nvmbt->nvblm",centered,centered)/(stop-start)
        diagonal = np.maximum(np.diagonal(cov,axis1=-2,axis2=-1),0)
        denom = np.sqrt(diagonal[..., :,None]*diagonal[...,None,:])
        corr = np.divide(cov,denom,out=np.zeros_like(cov),where=denom>1e-24)
        corr = np.clip(corr,-1,1)
        relations.append(np.stack([cov[...,i,j] for i,j in PAIRS]+[corr[...,i,j] for i,j in CROSS],axis=-1))
    values = {"morphology":morphology,"multiscale":np.stack(cells,axis=-2).reshape(len(x),-1),
        "relations":np.stack(relations,axis=-2).reshape(len(x),-1)}
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Nonfinite tensor features")
    total = np.sum(x*x,axis=-1)
    retained = np.divide(np.sum(coeff*coeff,axis=-1),total,out=np.ones_like(total),where=total>1e-24)
    return values,{"reconstruction_max_abs_mv":error,"dct_retained_energy_mean":float(retained.mean())}


def robust_record_summary(values):
    return np.concatenate([np.median(values,axis=0),np.quantile(values,.75,axis=0)-np.quantile(values,.25,axis=0)])


def prepare_bank(config):
    d,raw,templates,indices,y,groups,source = multiview_bank.prepare_bank(config)
    validate_config(config,d.shape[-1])
    statistics_path = Path(config["bank_dir"])/"statistics.npy"
    statistics = np.load(statistics_path,mmap_mode="r")
    if statistics.ndim != 2 or len(statistics)!=len(d) or not np.isfinite(statistics).all():
        raise ValueError("Invalid matched baseline statistics")
    scope = {"source":source,"statistics_sha256":file_hash(statistics_path),"code_sha256":file_hash(__file__),
        "construction":{key:config[key] for key in ("wavelet","level","dct_coefficients","windows","chunk_size")},
        "runtime":{"numpy":np.__version__,"pywt":pywt.__version__,"scipy":scipy.__version__},
        "aggregation":"within-record beat median/IQR; then equal-record coordinate median per patient"}
    fingerprint = _hash(scope)
    root = Path(config["feature_bank_dir"])
    root.mkdir(parents=True,exist_ok=True)
    audit_root = Path(config["output_dir"])/"bank_audit"
    audit_root.mkdir(parents=True,exist_ok=True)
    manifest_path = root/"build_manifest.json"
    build = dict(scope,bank_fingerprint=fingerprint)
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=build:
        raise ValueError("Tensor feature construction/source changed; use a new feature_bank_dir")
    _save_json(manifest_path,build)
    if (root/"manifest.json").exists():
        manifest = json.loads((root/"manifest.json").read_text())
        if manifest["bank_fingerprint"]!=fingerprint or any(file_hash(root/name)!=digest for name,digest in manifest["files"].items()):
            raise ValueError("Tensor feature cache changed")
        return root
    names = feature_names(config,statistics.shape[1])
    patient_rows = {}
    record_audits = []
    viewbank = multiview_bank.ViewBank(d,raw,templates,indices,"full_tensor")
    beats = pd.read_csv(config["beats_path"])
    beats = beats.loc[beats.label.isin(config["classes"])].reset_index(drop=True)
    cache = root/"records"
    cache.mkdir(exist_ok=True)
    for record_number,(record,part) in enumerate(beats.groupby("record",sort=True),1):
        rows = part.index.to_numpy()
        cache_path = cache/f"{_hash(record)}.npz"
        if cache_path.exists():
            with np.load(cache_path) as saved:
                if str(saved["fingerprint"].item())!=fingerprint or not np.array_equal(saved["rows"],rows):
                    raise ValueError("Record feature cache scope changed")
                summaries = {family:saved[family] for family in names}
                audit = json.loads(str(saved["audit"].item()))
                digest = str(saved["content_sha256"].item())
            if digest!=_hash([multiview_bank.array_hash(summaries[key]) for key in names]+[audit]):
                raise ValueError("Record feature cache content changed")
        else:
            chunks = {key:[] for key in names}
            diagnostics = []
            for start in range(0,len(rows),config["chunk_size"]):
                selected = rows[start:start+config["chunk_size"]]
                x = viewbank[selected].reshape(len(selected),5,3,d.shape[-1])
                values,diagnostic = extract(x,config)
                values["statistics"] = np.asarray(statistics[selected])
                for key in names:
                    chunks[key].append(values[key])
                diagnostics.append((len(selected),diagnostic))
            summaries = {key:robust_record_summary(np.concatenate(chunks[key])).astype(np.float64) for key in names}
            audit = {"beats":len(rows),"reconstruction_max_abs_mv":max(item["reconstruction_max_abs_mv"] for _,item in diagnostics),
                "dct_retained_energy_mean":sum(n*item["dct_retained_energy_mean"] for n,item in diagnostics)/len(rows)}
            temporary = cache_path.with_suffix(".partial.npz")
            np.savez_compressed(temporary,rows=rows,fingerprint=fingerprint,audit=json.dumps(audit),
                content_sha256=_hash([multiview_bank.array_hash(summaries[key]) for key in names]+[audit]),**summaries)
            temporary.replace(cache_path)
        for family in names:
            if summaries[family].shape!=(2*len(names[family]),) or not np.isfinite(summaries[family]).all():
                raise ValueError("Invalid record feature summary")
        patient = str(part.patient_id.iloc[0])
        patient_rows.setdefault(patient,[]).append(summaries)
        record_audits.append(audit)
        _progress(audit_root,state="constructing_features",records_complete=record_number,total_records=source["source"]["denoised"]["n_records"])
        if record_number%25==0:
            print(f"tensor features: {record_number} records",flush=True)
    ids = np.array(sorted(patient_rows))
    mapping = {str(patient):np.unique(y[groups==patient]) for patient in np.unique(groups)}
    if any(len(mapping[patient])!=1 for patient in ids):
        raise ValueError("Inconsistent source patient labels")
    labels = np.array([mapping[patient][0] for patient in ids],dtype=np.int64)
    values = {key:np.stack([np.median(np.stack([row[key] for row in patient_rows[patient]]),axis=0) for patient in ids]) for key in names}
    schema = {key:[f"record_{summary}/{name}" for summary in ("median","iqr") for name in names[key]] for key in names}
    values["combined"] = np.concatenate(list(values.values()),axis=1)
    schema["combined"] = [f"{family}/{name}" for family in names for name in schema[family]]
    np.savez_compressed(root/"patient_features.npz",patient_id=ids,y=labels,**values)
    _save_json(root/"feature_names.json",schema)
    summary = {"bank_fingerprint":fingerprint,"n_patients":len(ids),"n_records":len(record_audits),"n_beats":len(d),
        "tensor_axes":["beat","view","XYZ","band","original_time"],"tensor_shape":[len(d),5,3,config["level"]+1,d.shape[-1]],
        "feature_dimensions":{key:value.shape[1] for key,value in values.items()},
        "max_tensor_reconstruction_error_mv":max(row["reconstruction_max_abs_mv"] for row in record_audits),
        "mean_record_dct_retained_energy":float(np.mean([row["dct_retained_energy_mean"] for row in record_audits])),
        "nonfinite_features":sum(int(np.sum(~np.isfinite(value))) for value in values.values()),
        "globally_constant_columns_audit_only":{key:int(np.sum(np.ptp(value,axis=0)==0)) for key,value in values.items()},
        "scope":"Label-free construction and fidelity diagnostics only; summaries are lossy; neither reconstruction nor retained energy is diagnostic accuracy."}
    _save_json(root/"manifest.json",dict(summary,files={name:file_hash(root/name) for name in ("patient_features.npz","feature_names.json")}))
    _save_json(audit_root/"summary.json",summary)
    lines = ["# Tensor construction audit","",summary["scope"],"",f"Retained {len(ids)} patients, {len(record_audits)} records, {len(d):,} beats.","",
        f"Tensor shape: {summary['tensor_shape']}; wavelet {config['wavelet']}, complete bands {bands(config)}, symmetric extension. Maximum additive reconstruction error {summary['max_tensor_reconstruction_error_mv']:.3g} mV.","",
        "| Feature family | Patient features |","|---|---:|"]
    lines += [f"| {key} | {value.shape[1]} |" for key,value in values.items()]
    lines += ["","Within-record beat median/IQR, then equal-record coordinate median within each patient. Existing wavelet statistics use the same aggregation. No learned cohort transform, feature selection or classifier enters this stage. Full time traces remain in the source bank; feature summaries are deliberately lossy. Full-data constant-column counts are audit diagnostics only, never a selection mask.","",
        "DCT coefficients share a fixed signed basis across every beat; truncation loses temporal detail and energy retention does not certify disease preservation. Covariance retains amplitude; off-diagonal correlations add normalized XYZ relationships, with undefined zero-variance correlations encoded as zero. Fixed R-relative windows are not clinical boundaries. No per-beat amplitude division, quality deletion, alignment, phase warping or new labels.",""]
    (audit_root/"REPORT.md").write_text("\n".join(lines))
    _progress(audit_root,state="complete",records_complete=len(record_audits))
    return root


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/tensor_features.yaml")
    args = parser.parse_args()
    print(prepare_bank(yaml.safe_load(Path(args.config).read_text())))
