import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mi_localization import multiview_bank as bank, multiview_study as study
from mi_localization.benchmark import patient_balanced_weights
from mi_localization.contrastive import initial_state_hash
from mi_localization.preprocessing_study import fit_waves
from test_regularization import tiny_bank


def fixture(tmp_path):
    old,_,_,_=tiny_bank(tmp_path)
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/multiview_vcg.yaml").read_text())
    for key in ["seed","split_seed","training_seeds","classes","folds_path","output_dir","outer_splits","inner_splits","device","torch_threads","widths","pool_bins","batch_size","samples_per_epoch","max_epochs","patience","bank_dir","paper_config","beats_path"]:
        cfg[key]=old[key]
    cfg.update(raw_bank_dir=str(tmp_path/"raw"),template_bank_dir=str(tmp_path/"templates"))
    return cfg


def test_templates_are_per_record_and_views_preserve_both_sources(tmp_path):
    cfg=fixture(tmp_path)
    beats=pd.read_csv(cfg["beats_path"])
    beats.loc[2,"record"]="p00/second"
    beats.to_csv(cfg["beats_path"],index=False)
    d,raw,templates,indices,y,groups,scope=bank.prepare_bank(cfg)
    np.testing.assert_array_equal(templates[indices[0]],np.median(d[:2],axis=0))
    np.testing.assert_array_equal(templates[indices[2]],d[2])
    assert indices[0]==indices[1] and indices[2]!=indices[0]
    full=bank.ViewBank(d,raw,templates,indices,"full_tensor")
    values=full[np.arange(3)].reshape(3,5,3,65)
    np.testing.assert_array_equal(values[:,0],raw[:3])
    np.testing.assert_array_equal(values[:,1],d[:3])
    np.testing.assert_allclose(values[:,2]+values[:,3],d[:3],atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(values[:,1]+values[:,4],raw[:3],atol=1e-7,rtol=1e-6)
    assert full[0].shape==(15,65) and full.shape==(81,15,65)
    assert bank.prepare_bank(cfg)[-1]==scope
    original=np.load(Path(cfg["template_bank_dir"])/"templates.npy")
    original[0,0,0]+=1
    np.save(Path(cfg["template_bank_dir"])/"templates.npy",original)
    with pytest.raises(ValueError,match="content changed"):
        bank.prepare_bank(cfg)


def test_each_ablation_has_the_declared_information_and_raw_integrity_is_checked(tmp_path):
    cfg=fixture(tmp_path)
    d,raw,m,indices,*_=bank.prepare_bank(cfg)
    values={arm:bank.ViewBank(d,raw,m,indices,arm)[0] for arm in bank.ARMS}
    np.testing.assert_array_equal(values["capacity_control"],np.tile(d[0],(5,1)))
    np.testing.assert_array_equal(values["template_only"],np.tile(m[indices[0]],(5,1)))
    np.testing.assert_array_equal(values["raw_denoised"][:3],raw[0])
    np.testing.assert_array_equal(values["template_residual"][:6],np.tile(d[0],(2,1)))
    class Forbidden:
        def __getitem__(self,key):
            pytest.fail("D-only view must not fetch raw measurements")
    np.testing.assert_array_equal(bank.ViewBank(d,Forbidden(),m,indices,"capacity_control")[0],values["capacity_control"])
    changed=np.load(Path(cfg["raw_bank_dir"])/"waveforms.npy")
    changed[0,0,0]+=1
    np.save(Path(cfg["raw_bank_dir"])/"waveforms.npy",changed)
    with pytest.raises(ValueError,match="provenance or content changed"):
        bank.prepare_bank(cfg)


def test_multiview_initialization_is_matched_and_starts_with_original_d_only(tmp_path):
    cfg=fixture(tmp_path)
    d,raw,m,indices,*_=bank.prepare_bank(cfg)
    baseline=study.initial_model(cfg,42,"vcg").eval()
    models={arm:study.initial_model(cfg,42,arm).eval() for arm in bank.ARMS[1:]}
    assert len({initial_state_hash(model) for model in models.values()})==1
    x=torch.tensor(np.asarray(d[:4]).copy())
    duplicate=x.repeat(1,5,1)
    torch.testing.assert_close(baseline(x),models["capacity_control"](duplicate),atol=1e-6,rtol=1e-6)
    full=torch.tensor(bank.ViewBank(d,raw,m,indices,"full_tensor")[np.arange(4)])
    torch.testing.assert_close(baseline(x),models["full_tensor"](full),atol=1e-6,rtol=1e-6)
    first=models["full_tensor"].encoder[0].weight
    assert torch.count_nonzero(first[:,:3])==0 and torch.count_nonzero(first[:,6:])==0


def test_normalization_is_training_only_cached_and_matches_frozen_ce(tmp_path):
    cfg=fixture(tmp_path)
    views,y,groups,_=study.load_inputs(cfg)
    full=views["full_tensor"]
    train,valid=np.arange(54),np.arange(54,81)
    weights=patient_balanced_weights(y[train],groups[train])
    mean,scale=bank.fit_normalization(full,y,groups,train,weights)
    raw_changed=np.asarray(full.raw).copy();raw_changed[valid]+=1000
    d_changed=np.asarray(full.d).copy();d_changed[valid]+=2000
    alternate=bank.ViewBank(d_changed,raw_changed,full.templates,full.record_indices,"full_tensor")
    other=bank.fit_normalization(alternate,y,groups,train,weights)
    np.testing.assert_array_equal(mean,other[0]);np.testing.assert_array_equal(scale,other[1])
    root=tmp_path/"run"
    model,norm,audit=study.fit(views["vcg"],full,y,groups,train,valid,2,cfg,42,"vcg",root,"scope")
    old,_,history=fit_waves(full.d,y,groups,train,valid,2,cfg,42)
    for key,value in old.state_dict().items():
        torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
    assert history["history"]==audit["history"]
    fits=[audit]
    for arm in bank.ARMS[1:]:
        model,norm,audit=study.fit(views[arm],full,y,groups,train,valid,2,cfg,42,arm,root,"scope")
        study.validate_fit(audit,cfg,42,arm,train,valid,groups,norm)
        study.check_pair(fits[0],audit)
        if len(fits)>1:
            study.check_pair(fits[1],audit)
        fits.append(audit)
    path=next((root/"normalization").glob("*.npz"))
    with np.load(path) as saved:
        payload={key:saved[key] for key in saved.files}
    payload["mean"][0,0]+=1
    np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match="normalization cache changed"):
        study.normalizer(full,y,groups,train,weights,root,"scope","full_tensor")


def test_complete_nested_multiview_run_resume_and_validated_report(tmp_path,monkeypatch):
    from mi_localization import multiview_report as report
    cfg=fixture(tmp_path)
    calls=[]
    original=study.fit
    def tracked(waves,full,y,groups,train,valid,*args,**kwargs):
        calls.append((set(groups[train]),None if valid is None else set(groups[valid])))
        return original(waves,full,y,groups,train,valid,*args,**kwargs)
    monkeypatch.setattr(study,"fit",tracked)
    table=study.run_study(cfg)
    assert len(calls)==162 and table.complete.all() and len(table)==18
    frozen=pd.read_csv(cfg["folds_path"])
    for i,(train,valid) in enumerate(calls):
        fold=(i//18)%3+1
        test=set(frozen.loc[frozen.fold==fold,"patient_id"])
        assert not train&test
        if valid is not None:
            assert not train&valid and not valid&test
    root=Path(cfg["output_dir"])/"complete"
    assert len(list((root/"normalization").glob("*.npz")))==9
    path=report.summarize_study(cfg,draws=100)
    assert path.exists()
    summary=json.loads((root/"summary.json").read_text())
    assert summary["unique_fit_count"]==162 and len(summary["prespecified_contrasts"])==8
    monkeypatch.setattr(study,"fit",lambda *a,**k:pytest.fail("Completed fits must resume without retraining"))
    pd.testing.assert_frame_equal(table,study.run_study(cfg))
    with pytest.raises(ValueError,match="changed"):
        study.run_study(dict(cfg,dropout=.3))
    audit_path=root/"seed_42/fold_01_full_tensor.json"
    audit=json.loads(audit_path.read_text())
    audit["final_fit"]["normalization"]["wave_mean_"][0]+=1
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="scope/initialization/normalization"):
        report.summarize_study(cfg,draws=100)


def test_partial_runs_keep_their_scope_and_invalid_protocols_are_rejected(tmp_path):
    from mi_localization import multiview_report as report
    cfg=fixture(tmp_path)
    cfg["training_seeds"]=[42]
    table=study.run_study(cfg,outer_folds=[1])
    assert not table.complete.any()
    assert not (Path(cfg["output_dir"])/"complete").exists()
    assert (Path(cfg["output_dir"])/"partial_folds_1/study_manifest.json").exists()
    with pytest.raises(ValueError,match="three seeds"):
        report.summarize_study(cfg,draws=100)
    with pytest.raises(ValueError,match="six multiview arms"):
        study.validate_config(dict(cfg,arms=["vcg","full_tensor"]))
