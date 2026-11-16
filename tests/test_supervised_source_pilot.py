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
