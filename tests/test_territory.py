import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mi_localization import territory_study as study, territory_report as report
from mi_localization.preprocessing_study import fit_waves
from test_regularization import tiny_bank


def fixture(tmp_path):
    cfg,waves,y,groups = tiny_bank(tmp_path)
    cfg.update(arms=list(study.ARMS),territory_weight=.2,dropout=.2,weight_decay=.01)
    return cfg,waves,y,groups


def test_original_labels_define_mentions_without_changing_class_endpoint():
    cfg = {"classes":["AMI","ALMI","ASMI","IMI","ILMI","HC"]}
    targets = study.codebook(cfg)
    np.testing.assert_array_equal(targets,[[1,0,0,0],[1,0,1,0],[1,0,0,1],[0,1,0,0],[0,1,1,0],[0,0,0,0]])
    assert cfg["classes"] == ["AMI","ALMI","ASMI","IMI","ILMI","HC"]
    with pytest.raises(ValueError,match="Unknown original label"):
        study.codebook({"classes":["other"]})


def test_matched_control_reproduces_original_ce_and_objective_shares_rows(tmp_path):
    cfg,waves,y,groups = fixture(tmp_path)
    train,valid = np.arange(54),np.arange(54,81)
    ce,norm,audit = study.fit(waves,y,groups,train,valid,2,cfg,42,"ce")
    original,_,old = fit_waves(waves,y,groups,train,valid,2,cfg,42)
    for key,value in original.state_dict().items():
        torch.testing.assert_close(ce.state_dict()[key],value,atol=0,rtol=0)
    assert [r["training_loss"] for r in old["history"]] == [r["training_loss"] for r in audit["history"]]
    auxiliary,other_norm,other = study.fit(waves,y,groups,train,valid,2,cfg,42,"territory")
    study.check_pair(audit,other)
    study.validate_fit(audit,cfg,42,"ce",train,valid,groups)
    study.validate_fit(other,cfg,42,"territory",train,valid,groups)
    assert any(not torch.equal(value,auxiliary.encoder.state_dict()[key]) for key,value in ce.encoder.state_dict().items())
    assert ce(torch.tensor(waves[:2])).shape == (2,3)
    assert auxiliary(torch.tensor(waves[:2]),return_auxiliary=True)[1].shape == (2,4)
    corrupted = copy.deepcopy(other)
    corrupted["history"][0]["descriptor_bce"] += 1
    with pytest.raises(ValueError,match="objective/budget"):
        study.validate_fit(corrupted,cfg,42,"territory",train,valid,groups)
    with pytest.raises(ValueError,match="overlapping"):
        study.fit(waves,y,groups,train,train,1,cfg,42,"territory")


def test_complete_nested_comparison_resume_report_and_tamper_guards(tmp_path,monkeypatch):
    cfg,_,_,_ = fixture(tmp_path)
    calls=[]
    original=study.fit
    def tracked(waves,y,groups,train,valid,*args,**kwargs):
        calls.append((set(groups[train]),None if valid is None else set(groups[valid])))
        return original(waves,y,groups,train,valid,*args,**kwargs)
    monkeypatch.setattr(study,"fit",tracked)
    table=study.run_study(cfg)
    assert len(calls)==54 and table.complete.all()
    frozen=pd.read_csv(cfg["folds_path"])
    for i,(train,valid) in enumerate(calls):
        fold=(i//6)%3+1
        test=set(frozen.loc[frozen.fold==fold,"patient_id"])
        assert not train&test
        if valid is not None:
            assert not train&valid and not valid&test
    path=report.summarize_study(cfg,draws=100)
    assert path.exists()
    assert json.loads((path.parent/"summary.json").read_text())["unique_fit_count"]==54
    monkeypatch.setattr(study,"fit",lambda *a,**k:pytest.fail("Completed fits must resume"))
    pd.testing.assert_frame_equal(table,study.run_study(cfg))
    with pytest.raises(ValueError,match="changed"):
        study.run_study(dict(cfg,territory_weight=.3))
    audit_path=path.parent/"seed_42/fold_01_territory.json"
    audit=json.loads(audit_path.read_text())
    audit["final_fit"]["auxiliary_weight"]=0
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="scope/initialization"):
        report.summarize_study(cfg,draws=100)
