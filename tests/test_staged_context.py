import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mi_localization import staged_context_model as models,staged_context_study as study,staged_context_report as report
from mi_localization import residual_context_model as base,chronology_model as chronology
from mi_localization.contrastive import initial_state_hash
from test_residual_context import setup as residual_setup


def setup(tmp_path):
    bank,assigned,cfg=residual_setup(tmp_path)
    cfg.update(arms=list(models.ARMS),output_dir=str(tmp_path/'staged'),stage2_seed_offset=500000003)
    return bank,assigned,cfg


def teacher(bank,assigned,cfg):
    train,query=bank.ids[assigned!=1],bank.ids[assigned==1]
    model,norm,_,_,_=base.fit(bank,train,None,cfg,43,'beat_only',1)
    return train,query,model,norm


def test_default_phase_counts_and_shared_initial_states(tmp_path):
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/staged_context_v1.yaml').read_text())
    phases=cfg['outer_splits']*len(cfg['training_seeds'])*(cfg['inner_splits']+1)
    assert phases==120 and phases*len(cfg['arms'])==600
    assert cfg['outer_splits']*len(cfg['training_seeds'])*len(cfg['arms'])==150
    assert cfg['sequence_length']*cfg['windows_per_epoch']==8192
    bank,assigned,small=setup(tmp_path);_,_,t,_=teacher(bank,assigned,small)
    states=[models.initialize(small,500000046,a,t.state_dict()) for a in models.ADAPTER_ARMS]
    assert len({initial_state_hash(m) for m in states})==1
    assert len({models.morphology_hash(m) for m in states})==1
    assert models.morphology_hash(states[0])==models.morphology_hash(t)
    counts=[sum(p.numel() for p in m.parameters() if p.requires_grad) for m in states]
    assert counts[1]>counts[0]>counts[2] and counts[2]==counts[3]


def test_zero_residual_and_zero_update_outer_preserve_teacher_predictions(tmp_path):
    bank,assigned,cfg=setup(tmp_path);train,query,t,norm=teacher(bank,assigned,cfg)
    expected,_=base.predict(t,norm,bank,query,cfg,43)
    for arm in models.ADAPTER_ARMS:
        model,_,audit,_,_=models.fit_adapter(bank,train,None,cfg,500000046,arm,0,t.state_dict(),norm.audit())
        actual,_=base.predict(model,norm,bank,query,cfg,500000046)
        np.testing.assert_allclose(actual,expected,atol=1e-7)
        assert audit['selected_epoch']==0 and not audit['history'] and audit['optimizer_steps']==0


def test_real_frozen_adaptation_updates_only_context_and_matches_direct_dropout_policy(tmp_path):
    bank,assigned,cfg=setup(tmp_path);train,query,t,norm=teacher(bank,assigned,cfg)
    state={k:v.detach().clone() for k,v in t.state_dict().items()}
    frozen,_,audit,_,_=models.fit_adapter(bank,train,None,cfg,500000046,'frozen_context',2,state,norm.audit())
    assert models.morphology_hash(frozen)==models.morphology_hash(t)
    assert all(r['morphology_sha256']==audit['initial_morphology_sha256'] for r in audit['history'])
    assert frozen.context_head[1].weight.abs().sum()>0
    assert not frozen.encoder.training and not frozen.beat_head.training and frozen.context_head.training
    assert all(not p.requires_grad for p in frozen.encoder.parameters())
    joint,_,_,_,_=models.fit_adapter(bank,train,None,cfg,500000046,'joint_context',1,state,norm.audit())
    assert models.morphology_hash(joint)!=models.morphology_hash(t)
    assert not joint.encoder.training and not joint.beat_head.training
    assert all(p.requires_grad for p in joint.encoder.parameters())
    for k,v in state.items():torch.testing.assert_close(v,t.state_dict()[k])


def test_inner_epoch_zero_fallback_is_eligible_and_restores_initial_state(tmp_path,monkeypatch):
    bank,assigned,cfg=setup(tmp_path);train,query,t,norm=teacher(bank,assigned,cfg)
    cfg=dict(cfg,max_epochs=3,patience=1)
    truth=bank.labels[np.searchsorted(bank.ids,query)];perfect=np.eye(6)[truth];bad=np.roll(perfect,1,axis=1)
    calls=[]
    def predict(*args):
        calls.append(1);return (perfect if len(calls)==1 else bad),{'fixture':True}
    monkeypatch.setattr(base,'predict',predict)
    model,_,audit,p,_=models.fit_adapter(bank,train,query,cfg,500000046,'frozen_context',3,t.state_dict(),norm.audit())
    assert audit['selected_epoch']==0 and audit['best_validation_macro_f1']==1 and len(audit['history'])==1
    assert audit['optimizer_steps']==cfg['windows_per_epoch']//cfg['batch_size']
    np.testing.assert_array_equal(p,perfect)
    assert torch.equal(model.context_head[1].weight,torch.zeros_like(model.context_head[1].weight))


def test_actual_nested_stages_restore_all_models_resume_and_reject_tamper(tmp_path,monkeypatch):
    bank,assigned,cfg=setup(tmp_path)
    monkeypatch.setattr(study,'load_inputs',lambda config:(bank,assigned))
    root=study.run(cfg);summary=report.summarize(cfg)
    assert summary['phase_fits']==30 and summary['all_phase_models_restored']==30
    assert summary['outer_models_restored']==10 and summary['matched_adaptation_partition_checks']==6
    assert len(summary['ensemble_metrics'])==5 and len(summary['contrasts'])==6
    monkeypatch.setattr(base,'fit',lambda *a,**k:pytest.fail('Completed teachers must resume'))
    monkeypatch.setattr(models,'fit_adapter',lambda *a,**k:pytest.fail('Completed adaptation must resume'))
    assert study.run(cfg)==root and report.summarize(cfg)==summary
    with pytest.raises(ValueError,match='namespace'):study.run(dict(cfg,gate_initial_bias=-4.))
    # Outer-query label changes cannot affect stage choices or model inference.
    changed=copy.deepcopy(bank);ids=bank.ids[assigned==1];rows=bank.beats_for(ids)
    changed.y[rows]=(changed.y[rows]+1)%6;changed.labels[assigned==1]=(changed.labels[assigned==1]+1)%6
    scope=study.make_scope(cfg,bank)
    _,first=study.evaluate_fold(bank,assigned,cfg,scope,root,42,1,verify=True)
    _,second=study.evaluate_fold(changed,assigned,cfg,scope,root,42,1,verify=True)
    for arm in cfg['arms']:np.testing.assert_allclose(first[arm],second[arm])
    path=study.path_for(root,1,42,1,'frozen_context').with_suffix('.pt')
    path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='checkpoint changed'):report.summarize(cfg)


def test_frozen_hash_and_epoch_zero_stopping_audit_cannot_be_changed(tmp_path,monkeypatch):
    bank,assigned,cfg=setup(tmp_path)
    monkeypatch.setattr(study,'load_inputs',lambda config:(bank,assigned))
    root=study.run(cfg)
    p=study.path_for(root,1,42,1,'frozen_context');audit=json.loads(p.read_text())
    audit['fit']['history'][0]['morphology_sha256']='tamper';p.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='drifted'):report.summarize(cfg)


def test_teacher_partition_mismatch_is_rejected_even_if_shapes_and_counts_match(tmp_path):
    bank,assigned,cfg=setup(tmp_path);scope=study.make_scope(cfg,bank)
    root=tmp_path/'checks';root.mkdir()
    train,query=bank.ids[assigned!=1],bank.ids[assigned==1]
    teacher=study.phase_fit(bank,cfg,scope,root,42,1,0,'morphology',train,query,1,choices=[])
    wrong=dict(teacher,train_membership_sha256='another partition')
    with pytest.raises(ValueError,match='exact patient partition'):
        study.phase_fit(bank,cfg,scope,root,42,1,0,'frozen_context',train,query,0,teacher=wrong,choices=[])


def test_live_monitor_staged_phase_and_complete_verification_guard(tmp_path,monkeypatch,capsys):
    from test_chronology_lengths import script
    live=script('live_status.py');runner=script('run_staged_context.py')
    p=tmp_path/'results/neural_staged_context_v1/complete';p.mkdir(parents=True)
    (p/'progress.json').write_text(json.dumps(dict(pid=12,state='inner_training',phase='adaptation',gate_mean=.05,training_loss=.8)))
    monkeypatch.setattr(live,'process_command',lambda pid:('python -m mi_localization.staged_context_study' if pid==12 else '', 'checked'))
    s=live.snapshot(tmp_path);assert s['staged_context']['planned_fits']==600 and s['staged_context']['worker_process']=='running'
    live.display(s);assert 'Mean gate' in capsys.readouterr().out
    assert not runner.verified_ready(tmp_path)
    (p/'REPORT.md').write_text('verified');(p/'study_manifest.json').write_text(json.dumps(dict(study_fingerprint='fixture')))
    (p/'summary.json').write_text(json.dumps(dict(complete=True,study_fingerprint='fixture',phase_fits=600,all_phase_models_restored=600,
        outer_models_restored=150,matched_adaptation_partition_checks=120,all_query_beats_covered=True)))
    assert runner.verified_ready(tmp_path)
