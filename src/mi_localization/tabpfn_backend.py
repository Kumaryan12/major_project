"""Pinned local TabPFN-v2 checkpoint and inference; no hosted API."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import urllib.request

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

from .neural import _hash
from .neural import _save_json
from .signal_audit import file_hash

VERSION = "9.2.0"
REVISION = "275f19d2add9c2c0de38f9ebaded3856599fa2e7"
SHA256 = "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
BYTES = 29009539
URL = f"https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/{REVISION}/tabpfn-v2-classifier-v2_default.ckpt"


def prepare_checkpoint(config):
    if config["package_version"]!=VERSION or config["checkpoint_sha256"]!=SHA256 or config["checkpoint_revision"]!=REVISION or config["checkpoint_bytes"]!=BYTES or config["checkpoint_url"]!=URL:
        raise ValueError("Require the pinned public TabPFN-v2 checkpoint/package")
    installed = importlib.metadata.version("tabpfn")
    if installed!=VERSION:
        raise ValueError(f"Require tabpfn=={VERSION}; found {installed}")
    path = Path(config["checkpoint_path"])
    if path.name!="tabpfn-v2-classifier-v2_default.ckpt":
        raise ValueError("Preserve the v2 filename used for architecture resolution")
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(".partial")
        with urllib.request.urlopen(URL,timeout=120) as response,temporary.open("wb") as target:
            while chunk:=response.read(1024*1024):
                target.write(chunk)
        if temporary.stat().st_size!=BYTES or file_hash(temporary)!=SHA256:
            raise ValueError("Downloaded TabPFN checkpoint failed size/checksum validation")
        temporary.replace(path)
    if path.stat().st_size!=BYTES or file_hash(path)!=SHA256:
        raise ValueError("Local TabPFN checkpoint changed")
    return {"package_version":installed,"architecture":"TabPFN-v2","checkpoint_revision":REVISION,
        "checkpoint_sha256":SHA256,"checkpoint_bytes":BYTES,"checkpoint_url":URL,
        "checkpoint_model_card":"https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/"+REVISION+"/README.md",
        "inference":"local, frozen foundation weights; no external prediction service"}


def make_classifier(config,seed):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(config["torch_threads"])
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2,
        model_path=str(Path(config["checkpoint_path"]).resolve()),device=config["device"],n_estimators=config["n_estimators"],
        softmax_temperature=config["softmax_temperature"],balance_probabilities=False,
        average_before_softmax=False,categorical_features_indices=[],inference_precision=torch.float32,
        fit_mode="fit_preprocessors",n_preprocessing_jobs=1,random_state=seed,enable_torch_compile=False)


def runtime_info():
    from tabpfn import classifier,model_loading,preprocessing
    packages = ("tabpfn","torch","numpy","scipy","scikit-learn","pandas","joblib","skrub","lightgbm","einops","pydantic","safetensors")
    return {"versions":{name:importlib.metadata.version(name) for name in packages},"python":sys.version,"platform":platform.platform(),
        "classifier_code_sha256":file_hash(classifier.__file__),"loader_code_sha256":file_hash(model_loading.__file__),"preprocessing_code_sha256":file_hash(preprocessing.__file__)}


def fit_context(config,seed,X,y):
    if X.ndim!=2 or len(X)!=len(y) or not np.isfinite(X).all():
        raise ValueError("Invalid TabPFN training context")
    model = make_classifier(config,seed)
    # The selector's mask is reused, but TabPFN receives raw feature values;
    # its own fitted preprocessing is appropriate for its pretrained prior.
    with threadpool_limits(limits=config["torch_threads"]),torch.inference_mode():
        model.fit(np.asarray(X,dtype=np.float64),np.asarray(y,dtype=np.int64))
    return model


def predict(model,X,config):
    with threadpool_limits(limits=config["torch_threads"]),torch.inference_mode():
        return model.predict_proba(np.asarray(X,dtype=np.float64))


def balance(p,labels,n_classes):
    counts = np.bincount(labels,minlength=n_classes).astype(float)
    if np.any(counts<=0):
        raise ValueError("Every TabPFN training-context class is required")
    weighted = np.asarray(p,dtype=np.float64)/counts
    return weighted/weighted.sum(axis=1,keepdims=True)


def settings_audit(model,config):
    from tabpfn import classifier,model_loading,preprocessing
    inference = dataclasses.asdict(model.inference_config_) if dataclasses.is_dataclass(model.inference_config_) else model.inference_config_.model_dump(mode="json")
    # Some dataclass settings contain enums/objects; canonicalize to JSON text.
    inference = json.loads(json.dumps(inference,default=str))
    return {"architecture":"TabPFN-v2","package_version":importlib.metadata.version("tabpfn"),
        "n_estimators":model.n_estimators,"fitted_ensemble_members":len(model.ensemble_configs_),
        "classes":model.classes_.tolist(),"n_features_in":int(model.n_features_in_),
        "random_state":int(model.random_state),"balance_probabilities":bool(model.balance_probabilities),
        "softmax_temperature":model.softmax_temperature,"fit_mode":model.fit_mode,"n_preprocessing_jobs":model.n_preprocessing_jobs,
        "inference_config_sha256":_hash(inference),"inference_config":inference,
        "classifier_code_sha256":file_hash(classifier.__file__),"loader_code_sha256":file_hash(model_loading.__file__),
        "preprocessing_code_sha256":file_hash(preprocessing.__file__),"backend_device":config["device"],
        "precision":"torch.float32","input_scaling":"raw selected columns; backend preprocessing fitted to training context"}


def save(model,path):
    path = Path(path)
    temporary = path.with_name(path.stem+".partial.tabpfn_fit")
    model.save_fit_state(temporary)
    temporary.replace(path)


def load(path,config):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier.load_from_fit_state(path,device=config["device"])


def smoke_test(config):
    checkpoint = prepare_checkpoint(config)
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(6),[9,14,7,13,10,19])
    X = rng.normal(size=(len(labels),24));X[:,0]+=labels
    queries = rng.normal(size=(12,24))
    started = time.monotonic()
    model = fit_context(config,42,X,labels)
    p = predict(model,queries,config)
    if p.shape!=(12,6) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError("Real TabPFN smoke probabilities failed")
    single = predict(model,queries[:1],config)[0]
    altered = queries.copy();altered[1:]+=10000
    other = predict(model,altered,config)[0]
    if not np.allclose(single,p[0],rtol=1e-5,atol=1e-6) or not np.allclose(other,p[0],rtol=1e-5,atol=1e-6):
        raise ValueError("TabPFN prediction depends on unrelated query rows")
    root = Path(config["output_dir"])/"backend_audit"
    root.mkdir(parents=True,exist_ok=True)
    state = root/"smoke.tabpfn_fit"
    save(model,state)
    restored = load(state,config)
    if not np.allclose(predict(restored,queries,config),p,rtol=1e-6,atol=1e-7):
        raise ValueError("TabPFN fit-state restoration failed")
    model.balance_probabilities=True
    if not np.allclose(predict(model,queries,config),balance(p,labels,6),rtol=1e-6,atol=1e-7):
        raise ValueError("TabPFN manual correction differs from official balancing")
    import xgboost
    with xgboost.config_context(nthread=1),threadpool_limits(limits=1):
        tree = xgboost.XGBClassifier(n_estimators=2,max_depth=1,n_jobs=1,objective="multi:softprob",num_class=6,random_state=42)
        tree.fit(X,labels)
        tree_p = tree.predict_proba(queries)
    if not np.isfinite(tree_p).all() or not np.allclose(tree_p.sum(1),1,atol=1e-6):
        raise ValueError("TabPFN/XGBoost runtime coexistence smoke failed")
    result = {"passed":True,"scope":"Synthetic runtime smoke only, not clinical performance", "checkpoint":checkpoint,"backend":settings_audit(restored,config),
        "runtime":runtime_info(),"fit_and_checks_seconds":time.monotonic()-started,"training_rows":len(labels),"query_rows":len(queries),
        "maximum_query_batch_difference":float(max(np.max(np.abs(single-p[0])),np.max(np.abs(other-p[0])))),
        "save_load_passed":True,"official_balancing_equivalence":True,"xgboost_coexistence_passed":True}
    _save_json(root/"summary.json",result)
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/tabpfn_v2.yaml")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(smoke_test(config) if args.smoke else prepare_checkpoint(config),indent=2))
