"""Bounded source cohort, annotation semantics and training integrity."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from mi_localization import supervised_source_bank as bank
from mi_localization import supervised_source_audit as audit
from test_supervised_source_audit import tiny_config as audit_fixture, make_pair


def tiny_config(tmp_path):
    ac, database, statements=audit_fixture(tmp_path)
    ap=tmp_path/'audit_config.yaml';ap.write_text(yaml.safe_dump(ac))
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/supervised_source_pilot_v1.yaml').read_text())
    config.update(audit_config=str(ap),waveform_dir=str(tmp_path/'expanded'),reuse_waveform_dir=ac['waveform_dir'],
        bank_dir=str(tmp_path/'bank'),output_dir=str(tmp_path/'pilot'),training_patients=12,
        minimum_training_positive_patients=2,device='cpu',widths=[4,8,8],pool_bins=2,
        max_epochs=2,patience=2,samples_per_epoch=16,batch_size=4,training_seeds=[42],torch_threads=1)
    return config,ac,database,statements


def test_mi_enrichment_keeps_visits_and_is_independent_of_validation_annotations(tmp_path):
    config,_,database,statements=tiny_config(tmp_path)
    ac,frame,labels,states,_=bank.source_metadata(config)
    first,indices,mandatory=bank.select_cohort(frame,labels,states,config)
    assert mandatory==8 and first.loc[first.role.eq('train'),'patient_id'].nunique()==12
    assert first.loc[first.role.eq('validation'),'patient_id'].nunique()==2
    assert set(first.loc[first.patient_id.eq(1),'ecg_id'])=={1,21}
    changed=states.copy();changed[frame.role.eq('validation')]=0
    again=bank.select_cohort(frame,labels,changed,config)[0]
    pd.testing.assert_frame_equal(first,again)
    with pytest.raises(ValueError,match='budget'):
        bank.select_cohort(frame,labels,states,dict(config,training_patients=7))
    assert not first.strat_fold.eq(10).any()


def test_frozen_cohort_refuses_changed_settings_or_private_records(tmp_path):
    config,_,_,_=tiny_config(tmp_path)
    cohort,*_,scope=bank.freeze(config)
    assert bank.freeze(config)[-1]==scope
    with pytest.raises(ValueError,match='scope changed'):
        bank.freeze(dict(config,cohort_seed=123))
    (Path(config['output_dir'])/'review/cohort.csv').write_text('changed')
    with pytest.raises(ValueError,match='cohort changed'):
        bank.freeze(config)


def test_excluded_role_rejected_before_copy_download_or_decode(tmp_path,monkeypatch):
    config,_,_,_=tiny_config(tmp_path)
    monkeypatch.setattr(bank.audit,'safe_path',lambda *a:pytest.fail('waveform access'))
    row=SimpleNamespace(role='validation',strat_fold=10,filename_hr='records500/00000/00019_hr',ecg_id=19)
    with pytest.raises(ValueError,match='Refusing'):
        bank.download_row(row,config)
    with pytest.raises(ValueError,match='Refusing'):
        bank.convert(row,config,'test')


def test_raw_and_physical_copy_groups_quarantine_every_cross_patient_member():
    records=pd.DataFrame(dict(ecg_id=[1,2,3,4,5],patient_id=[1,2,3,4,4],role=['train','validation','train','train','train'],
        data_sha256=['a','a','b','c','d'],physical_xyz_sha256=['x','y','y','z','z'],accepted=True,reason='accepted'))
    result,groups=bank.quarantine(records)
    assert set(result.loc[result.reason.eq('cross_patient_exact_duplicate'),'ecg_id'])=={1,2,3}
    assert result.set_index('ecg_id').loc[5,'reason']=='same_patient_exact_duplicate'
    assert result.set_index('ecg_id').loc[4,'accepted']
    assert any(g['cross_role'] for g in groups)


def test_conversion_real_cached_header_controls_crops_and_refuses_mutation(tmp_path,monkeypatch):
    config,ac,_,_=tiny_config(tmp_path)
    make_pair(ac,1)
    row=bank.source_metadata(config)[1].iloc[0]
    bank.download_row(row,config)
    t=np.arange(5000)/500
    ecg=np.sin(2*np.pi*1.2*t[:,None]+np.arange(12)[None,:])*0.2
    monkeypatch.setattr(bank.wfdb,'rdrecord',lambda *a:SimpleNamespace(fs=500,p_signal=ecg,sig_name=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']))
    waves,peaks,item=bank.convert(row,config,'scope')
    assert item['accepted'] and waves.shape==(config['beats_per_record'],3,651)
    assert len(set(peaks))==len(peaks) and np.all(np.diff(peaks)>0)
    assert np.array_equal(waves,bank.convert(row,config,'scope')[0])
    assert Path(ac['waveform_dir'],'records500/00000/00001_hr.dat').stat().st_size==120000
    Path(config['waveform_dir'],'records500/00000/00001_hr.dat').write_bytes(b'changed')
    with pytest.raises(ValueError,match='cache changed'):
        bank.convert(row,config,'scope')


def test_flat_signals_are_explicit_qc_failures_not_normal_labels(tmp_path,monkeypatch):
    config,ac,_,_=tiny_config(tmp_path)
    make_pair(ac,1)
    row=bank.source_metadata(config)[1].iloc[0];bank.download_row(row,config)
    monkeypatch.setattr(bank.wfdb,'rdrecord',lambda *a:SimpleNamespace(fs=500,p_signal=np.zeros((5000,12)),sig_name=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']))
    waves,peaks,item=bank.convert(row,config,'scope')
    assert not item['accepted'] and item['reason']=='nonfinite_or_flat_xyz'
    assert waves.shape==(0,3,651) and len(peaks)==0


def synthetic_bank(tmp_path,monkeypatch):
    config,ac,_,_=tiny_config(tmp_path)
    def local_download(row,c):
        bank.assert_row(row)
        make_pair(dict(ac,waveform_dir=c['waveform_dir']),row.ecg_id,content=bytes([int(row.ecg_id)])*120000)
    def local_record(path):
        ecg_id=int(Path(path).name.split('_')[0]);t=np.arange(5000)/500
        ecg=np.sin(2*np.pi*(1+ecg_id/100)*t[:,None]+np.arange(12)[None,:])*(0.1+ecg_id/100)
        return SimpleNamespace(fs=500,p_signal=ecg,sig_name=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'])
    monkeypatch.setattr(bank,'download_row',local_download)
    monkeypatch.setattr(bank.wfdb,'rdrecord',local_record)
    return config,bank.prepare(config)


def test_prepared_bank_restores_hashes_groups_native_states_and_qc(tmp_path,monkeypatch):
    config,root=synthetic_bank(tmp_path,monkeypatch)
    assert bank.prepare(config)==root
    roles=np.load(root/'roles.npy');patients=np.load(root/'patient_ids.npy')
    assert len(np.unique(patients[roles=='train']))==12
    assert not set(patients[roles=='train']) & set(patients[roles=='validation'])
    assert np.load(root/'waves.npy').shape==(15,4,3,651)
    assert set(np.load(root/'ecg_ids.npy')).isdisjoint({19,20})
    assert np.load(root/'states.npy').shape==(15,7)
    assert np.isnan(np.load(root/'likelihood.npy')).any()
    assert np.all(np.load(root/'counts.npy')>0)
    private=Path(config['output_dir'])/'review/retained_records.csv'
    private.write_text('changed')
    with pytest.raises(ValueError,match='report/review content changed'):
        bank.prepare(config)


def test_native_mask_conventions_have_zero_gradients_for_unknown_entries():
    import torch
    from mi_localization import supervised_source_train as train
    train.neural._seed(42,dict(device='cpu',torch_threads=1))
    states=np.array([[0,1,2,3]],np.uint8)
    y,mask=train.targets(states,'certain')
    assert y.tolist()==[[0,0,0,1]] and mask.tolist()==[[1,0,0,1]]
    yy,mm=train.targets(states,'positive_intermediate')
    assert yy.tolist()==[[0,0,1,1]] and mm.tolist()==[[1,0,1,1]]
    logits=torch.zeros((1,4),requires_grad=True)
    train.masked_loss(logits,torch.tensor(y),torch.tensor(mask),torch.ones(4)).backward()
    assert logits.grad[0,1]==logits.grad[0,2]==0
    assert logits.grad[0,0]>0 and logits.grad[0,3]<0


def test_normalization_and_active_heads_ignore_validation_values():
    from mi_localization import supervised_source_train as train
    waves=np.zeros((3,4,3,651),np.float32)
    waves[0]=1;waves[1]=3;waves[2]=1000
    waves[0,1:]=9000 # padding must not influence fitted means
    counts=np.array([1,4,4]);groups=np.array([1,1,2]);indices=np.array([0,1])
    first=train.fit_normalizer(waves,counts,groups,indices)
    assert np.allclose(first['mean'],[2,2,2])
    waves[2]=-100000
    assert train.fit_normalizer(waves,counts,groups,indices)==first
    states=np.array([[3,3],[3,0],[0,3]],np.uint8)
    assert train.active_columns(states,groups,indices,2).size==0 # visits are not patients
    states[2]=3
    assert train.active_columns(states,groups,indices,2).size==0


def test_pooling_masks_padding_and_source_metrics_weight_patients_equally():
    import torch
    from mi_localization import supervised_source_train as train
    train.neural._seed(42,dict(device='cpu',torch_threads=1))
    config=dict(widths=[4,8,8],pool_bins=2,dropout=0)
    model=train.RecordModel(2,config).eval();x=torch.randn(2,4,3,651);mask=torch.tensor([[1,1,0,0],[1,0,0,0]],dtype=torch.bool)
    with torch.no_grad():
        expected=model(x,mask);x[~mask]=1000
        assert torch.allclose(expected,model(x,mask),atol=1e-6)
    y=np.array([[1],[0],[1]],np.float32);m=np.ones_like(y);p=np.array([[.8],[.6],[.9]])
    a=train.metrics(y,m,p,np.array([1,2,3]),['AMI'])
    b=train.metrics(y[[0,0,1,2]],m[[0,0,1,2]],p[[0,0,1,2]],np.array([1,1,2,3]),['AMI'])
    for k in ['macro_average_precision','macro_auroc','macro_f1_at_05']:
        assert a[k]==b[k]
    assert not a['target_six_class_metric'] and not a['clinical_negatives']


def test_both_objectives_train_restore_export_and_refuse_tampered_predictions(tmp_path,monkeypatch):
    from mi_localization import supervised_source_train as train
    config,root=synthetic_bank(tmp_path,monkeypatch)
    results=train.run(config)
    assert len(results)==2 and {r['objective'] for r in results}=={'certain','positive_intermediate'}
    assert results[0]['initial_hash']==results[1]['initial_hash']
    assert all(r['epochs_completed']==2 for r in results)
    assert all(not r['target_evaluated'] for r in results)
    assert train.run(config,verify=True)==results
    assert train.run(config)==results
    public=json.loads((Path(config['output_dir'])/'summary.json').read_text())
    assert not public['next_stage_automatically_launched'] and not public['cross_dataset_identity_cleared']
    member=Path(config['output_dir'])/'runs/certain_42/summary.json';original=member.read_bytes()
    altered=json.loads(original);altered['metrics']['macro_auroc']=-1;member.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='Restored source metrics'):
        train.run(config,verify=True)
    member.write_bytes(original)
    prediction=Path(config['output_dir'])/'runs/certain_42/validation.npz'
    prediction.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        train.run(config,verify=True)
