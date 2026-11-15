"""Higher-resolution and native-wavelet signed tensor cores, before fitting."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy
import yaml
from scipy.fft import dct, idct

from . import tensor_feature_bank as original
from .multiview_bank import ViewBank, array_hash, prepare_bank as prepare_views
from .neural import _hash, _save_json
from .signal_audit import file_hash
from .contrastive import _progress


def families(config):
    return tuple(f"{views}_dct{rank}" for views in ("denoised","all") for rank in config["global_ranks"])+tuple(f"wavelet_core{rank}" for rank in config["wavelet_ranks"])


def validate_config(config, source, length):
    original.validate_config(source,length)
    for key in ("global_ranks","wavelet_ranks"):
        values = config[key]
        if not values or len(set(values))!=len(values) or any(type(r)!=int or r<1 for r in values):
            raise ValueError("Require distinct positive refinement ranks")
    if max(config["global_ranks"]+[source["dct_coefficients"]])>length or config["chunk_size"]<1:
        raise ValueError("Invalid global compression rank/chunk size")
    lengths = [len(c) for c in pywt.wavedec(np.zeros(length),source["wavelet"],level=source["level"],mode="symmetric")]
    if max(config["wavelet_ranks"])>min(lengths):
        raise ValueError("Native wavelet compression rank exceeds coefficient length")


def extract(x,config,source):
    x = np.asarray(x,dtype=np.float64)
    if x.ndim!=4 or x.shape[1:3]!=(5,3) or not np.isfinite(x).all():
        raise ValueError("Expected finite beat × five views × XYZ × time")
    length = x.shape[-1]
    validate_config(config,source,length)
    global_coeff = dct(x,type=2,norm="ortho",axis=-1)
    features, reconstructions = {},{}
    for rank in sorted(set(config["global_ranks"]+[source["dct_coefficients"]])):
        coeff = global_coeff[...,:rank]
        reconstructions[f"global{rank}"] = idct(coeff,n=length,type=2,norm="ortho",axis=-1)
        if rank in config["global_ranks"]:
            features[f"denoised_dct{rank}"] = coeff[:,1].reshape(len(x),-1)
            features[f"all_dct{rank}"] = coeff.reshape(len(x),-1)
    wavelet_coeff = pywt.wavedec(x,source["wavelet"],level=source["level"],mode="symmetric",axis=-1)
    transformed = [dct(c,type=2,norm="ortho",axis=-1) for c in wavelet_coeff]
    for rank in config["wavelet_ranks"]:
        core = np.stack([c[...,:rank] for c in transformed],axis=-2)
        features[f"wavelet_core{rank}"] = core.reshape(len(x),-1)
        restored = [idct(c[...,:rank],n=original_c.shape[-1],type=2,norm="ortho",axis=-1) for c,original_c in zip(transformed,wavelet_coeff)]
        reconstructions[f"wavelet{rank}"] = pywt.waverec(restored,source["wavelet"],mode="symmetric",axis=-1)[...,:length]
    baseline = global_coeff[...,:source["dct_coefficients"]].reshape(len(x),-1)
    diagnostics = {}
    energy = np.sum(x*x,axis=-1)
    active = energy>1e-24
    for method,reconstructed in reconstructions.items():
        error = reconstructed-x
        # Aggregate within each record later; undefined relative errors excluded.
        relative = np.divide(np.sum(error*error,axis=-1),energy,out=np.zeros_like(energy),where=active)
        per_view = {}
        for v,name in enumerate(original.VIEWS):
            per_view[name] = {"relative_squared_error_sum":float(relative[:,v][active[:,v]].sum()),
                "active_lead_beats":int(active[:,v].sum()),
                "window_mse_mv2":{window:float(np.mean(error[:,v,:,start:stop]**2)) for window,(start,stop) in source["windows"].items()}}
        diagnostics[method] = per_view
    if any(not np.isfinite(value).all() for value in features.values()):
        raise ValueError("Nonfinite refined features")
    return features,baseline,diagnostics


def feature_names(config,source):
    names = {}
    for rank in config["global_ranks"]:
        for label,views in (("denoised",("denoised",)),("all",original.VIEWS)):
            names[f"{label}_dct{rank}"] = [f"{v}/{lead}/dct_{k:02d}" for v in views for lead in original.LEADS for k in range(rank)]
    for rank in config["wavelet_ranks"]:
        names[f"wavelet_core{rank}"] = [f"{v}/{lead}/{band}/native_time_dct_{k:02d}" for v in original.VIEWS
            for lead in original.LEADS for band in original.bands(source) for k in range(rank)]
    return names


def prepare_bank(config):
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    old_root = original.prepare_bank(source)
    d,raw,templates,record_indices,y,groups,provenance = prepare_views(source)
    validate_config(config,source,d.shape[-1])
    scope = {"source_config":source,"source_manifest_sha256":file_hash(old_root/"manifest.json"),"views":provenance,
        "construction":{key:config[key] for key in ("global_ranks","wavelet_ranks","chunk_size")},
        "code_sha256":file_hash(__file__),"original_code_sha256":file_hash(original.__file__),
        "numpy":np.__version__,"scipy":scipy.__version__,"pywt":pywt.__version__}
    fingerprint = _hash(scope)
    root = Path(config["refined_bank_dir"])
    root.mkdir(parents=True,exist_ok=True)
    audit_root = Path(config["output_dir"])/"bank_audit"
    audit_root.mkdir(parents=True,exist_ok=True)
    guard = root/"build_manifest.json"
    build = dict(scope,bank_fingerprint=fingerprint)
    if guard.exists() and json.loads(guard.read_text())!=build:
        raise ValueError("Refinement source/construction changed; use a new refined_bank_dir")
    _save_json(guard,build)
    if (root/"manifest.json").exists():
        manifest = json.loads((root/"manifest.json").read_text())
        if manifest["bank_fingerprint"]!=fingerprint or any(file_hash(root/name)!=digest for name,digest in manifest["files"].items()):
            raise ValueError("Refinement bank content changed")
        return root
    names = feature_names(config,source)
    all_names = dict(names,baseline16=original.feature_names(source,0)["morphology"])
    bank = ViewBank(d,raw,templates,record_indices,"full_tensor")
    beats = pd.read_csv(source["beats_path"])
    beats = beats.loc[beats.label.isin(source["classes"])].reset_index(drop=True)
    cache = root/"records"
    cache.mkdir(exist_ok=True)
    patients,diagnostics = {},[]
    for number,(record,part) in enumerate(beats.groupby("record",sort=True),1):
        rows = part.index.to_numpy()
        path = cache/f"{_hash(record)}.npz"
        if path.exists():
            with np.load(path) as saved:
                summaries = {name:saved[name] for name in all_names}
                diagnostic = json.loads(str(saved["diagnostic"].item()))
                if str(saved["fingerprint"].item())!=fingerprint or not np.array_equal(saved["rows"],rows) or str(saved["digest"].item())!=_hash([array_hash(summaries[name]) for name in all_names]+[diagnostic]):
                    raise ValueError("Refined record cache changed")
        else:
            chunks = {name:[] for name in all_names}
            diagnostic_chunks = []
            for start in range(0,len(rows),config["chunk_size"]):
                selected = rows[start:start+config["chunk_size"]]
                x = bank[selected].reshape(len(selected),5,3,d.shape[-1])
                values,baseline,diag = extract(x,config,source)
                values["baseline16"] = baseline
                for name in all_names:
                    chunks[name].append(values[name])
                diagnostic_chunks.append((len(selected),diag))
            summaries = {name:original.robust_record_summary(np.concatenate(chunks[name])) for name in all_names}
            diagnostic = {}
            for method in diagnostic_chunks[0][1]:
                diagnostic[method] = {}
                for view in original.VIEWS:
                    active = sum(diag[method][view]["active_lead_beats"] for _,diag in diagnostic_chunks)
                    diagnostic[method][view] = {"mean_relative_squared_error":None if not active else sum(diag[method][view]["relative_squared_error_sum"] for _,diag in diagnostic_chunks)/active,
                        "active_lead_beats":active,"window_mse_mv2":{window:sum(n*diag[method][view]["window_mse_mv2"][window] for n,diag in diagnostic_chunks)/len(rows) for window in source["windows"]}}
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary,rows=rows,fingerprint=fingerprint,diagnostic=json.dumps(diagnostic),
                digest=_hash([array_hash(summaries[name]) for name in all_names]+[diagnostic]),**summaries)
            temporary.replace(path)
        if any(summaries[name].shape!=(2*len(all_names[name]),) or not np.isfinite(summaries[name]).all() for name in all_names):
            raise ValueError("Invalid refined record features")
        patients.setdefault(str(part.patient_id.iloc[0]),[]).append(summaries)
        diagnostics.append(diagnostic)
        _progress(audit_root,state="constructing_refined_features",records_complete=number)
        if number%50==0:
            print(f"refined tensor features: {number} records",flush=True)
    ids = np.array(sorted(patients))
    values = {name:np.stack([np.median(np.stack([row[name] for row in patients[patient]]),axis=0) for patient in ids]) for name in all_names}
    with np.load(old_root/"patient_features.npz") as saved:
        if not np.array_equal(ids,saved["patient_id"]) or not np.array_equal(values.pop("baseline16"),saved["morphology"]):
            raise ValueError("Refined bank failed exact original DCT/aggregation control")
        labels = saved["y"]
    schema = {name:[f"record_{summary}/{feature}" for summary in ("median","iqr") for feature in names[name]] for name in names}
    np.savez_compressed(root/"patient_features.npz",patient_id=ids,y=labels,**values)
    _save_json(root/"feature_names.json",schema)
    fidelity = {}
    for method in diagnostics[0]:
        fidelity[method] = {}
        for view in original.VIEWS:
            active = [diag[method][view]["mean_relative_squared_error"] for diag in diagnostics if diag[method][view]["mean_relative_squared_error"] is not None]
            fidelity[method][view] = {"mean_record_relative_squared_error":None if not active else float(np.mean(active)),
                "window_rms_error_mv":{window:float(np.sqrt(np.mean([diag[method][view]["window_mse_mv2"][window] for diag in diagnostics]))) for window in source["windows"]}}
    summary = {"bank_fingerprint":fingerprint,"n_patients":len(ids),"n_records":len(diagnostics),"n_beats":len(d),
        "feature_dimensions":{name:value.shape[1] for name,value in values.items()},"original_dct_exact_match":True,"fidelity":fidelity,
        "scope":"Label-free compression diagnostics; reconstruction is not diagnostic accuracy; native bands have different coefficient sampling resolutions."}
    _save_json(root/"manifest.json",dict(summary,files={name:file_hash(root/name) for name in ("patient_features.npz","feature_names.json")}))
    _save_json(audit_root/"summary.json",summary)
    lines = ["# Signed tensor compression refinement","",summary["scope"],"",f"All {len(ids)} patients, {len(diagnostics)} records and {len(d):,} beats retained. Original global DCT and record aggregation reproduce exactly.","",
        "Global DCT ranks increase waveform resolution; denoised-only controls match the all-view ranks. Native-wavelet cores first compute A4/D4/D3/D2/D1 coefficients, then use a fixed signed DCT time basis within each band's native coefficient grid. This is a fixed partial time projection, not learned Tucker/HOOI or a shared clinical phase grid.","",
        "## Fidelity by physical view","","| Method | View | Mean record relative squared error | Central-window RMS error, mV |","|---|---|---:|---:|"]
    for method,views in fidelity.items():
        for view,value in views.items():
            relative = value["mean_record_relative_squared_error"]
            window = "central" if "central" in source["windows"] else next(iter(source["windows"]))
            lines.append(f"| {method} | {view} | {'undefined' if relative is None else f'{relative:.4f}'} | {value['window_rms_error_mv'][window]:.5f} |")
    lines += ["","Relative squared errors exclude zero-energy lead-beats, then average records equally. Absolute errors retain mV units. Native symmetric wavelet coefficients have boundary redundancy; no variance partition or clean-signal ground truth is claimed. All feature summaries are lossy and fitted selection stays in training partitions. No patient exclusions, R redetection, alignment or relabeling.",""]
    (audit_root/"REPORT.md").write_text("\n".join(lines))
    _progress(audit_root,state="complete",records_complete=len(diagnostics))
    return root


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/tensor_refinement.yaml")
    args = parser.parse_args()
    print(prepare_bank(yaml.safe_load(Path(args.config).read_text())))
