"""Expanded coverage, prerequisite gates, bounded retrieval and independent replay."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly
import wfdb
import yaml

from mi_localization import supervised_source_overlap as screen
from mi_localization.supervised_source_audit import canonical,file_hash


def config_for(tmp_path):
    c=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/supervised_source_overlap_v1.yaml').read_text())
    c.update(output_dir=str(tmp_path/'audit'),cache_dir=str(tmp_path/'cache'),positive_controls=2,candidate_batch_size=2)
    return c


def test_incomplete_source_pilot_cannot_start_preparation_or_screen(tmp_path,monkeypatch):
    c=config_for(tmp_path)
    pilot=dict(output_dir=str(tmp_path/'source'),bank_dir=str(tmp_path/'bank'))
    source_config=tmp_path/'source.yaml';source_config.write_text(yaml.safe_dump(pilot));c['source_config']=str(source_config)
    monkeypatch.setattr(screen.source,'prepare',lambda *a:pytest.fail('attempted source preparation'))
    with pytest.raises(ValueError,match='pilot is incomplete'):
        screen.load_context(c)
    assert not Path(c['output_dir']).exists()


def test_completed_pilot_gate_checks_all_objectives_weights_and_code(tmp_path,monkeypatch):
    from test_supervised_source_pilot import synthetic_bank
    from mi_localization import supervised_source_train
    c,_=synthetic_bank(tmp_path,monkeypatch)
    supervised_source_train.run(c)
    s,evidence=screen.completed_pilot(c)
    assert s['complete'] and not s['target_evaluated'] and evidence['summary_sha256']
    root=Path(c['output_dir'])
    model=root/'runs/positive_intermediate_42/encoder.pt';original=model.read_bytes();model.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact mismatch'):
        screen.completed_pilot(c)
    model.write_bytes(original)
    training=root/'training_manifest.json';scope=json.loads(training.read_text());scope['code_sha256']='0'*64;training.write_text(json.dumps(scope))
    with pytest.raises(ValueError,match='scope/endpoint mismatch'):
        screen.completed_pilot(c)


def test_chunked_retrieval_equals_original_with_missing_template_fallback(tmp_path):
    c=config_for(tmp_path);rng=np.random.default_rng(42)
    sf=rng.normal(size=(9,4,528)).astype(np.float32);tf=rng.normal(size=(7,4,528)).astype(np.float32)
    sa=np.ones(9,bool);sa[2]=False;ta=np.ones(7,bool);ta[5]=False
    actual=screen.retrieve(sf,tf,sa,ta,2,c)
    expected=screen.numeric.rank_candidates(sf,tf,sa,ta,2)
    assert all(np.array_equal(a,b) for a,b in zip(actual,expected))
    assert np.array_equal(actual[2],np.arange(7))
    assert all(5 in a for a in actual)


def test_pair_checkpoints_replay_known_copy_and_reject_forged_partial(tmp_path):
    c=config_for(tmp_path);rng=np.random.default_rng(9)
    target=rng.normal(0,.2,(6000,8));source=target[500:3000]*1.03+.02
    tp=tmp_path/'target.npz';sp=tmp_path/'source.npz';np.savez(tp,ecg=target);np.savez(sp,ecg=source)
    rows=screen.scan_one(0,[0],[str(tp)],[str(sp)],c,'scope',False)
    assert rows[0]['candidate'] and rows[0]['full_source_available']
    assert rows[0]['query_mean_correlation']>.9999
    assert screen.scan_one(0,[0],[str(tp)],[str(sp)],c,'scope',True)==rows
    path=Path(c['output_dir'])/'review/pairs/00000.json';altered=json.loads(path.read_text())
    altered['rows'][0]['candidate']=False;altered['rows_sha256']=canonical(altered['rows']);path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='does not reproduce'):
        screen.scan_one(0,[0],[str(tp)],[str(sp)],c,'scope',True)
    with pytest.raises(ValueError,match='scope/content changed'):
        screen.scan_one(0,[0],[str(tp)],[str(sp)],c,'different',False)


def test_replay_cache_redecodes_original_waveforms(tmp_path,monkeypatch):
    c=config_for(tmp_path);rng=np.random.default_rng(7);ecg=rng.normal(0,.1,(5000,8))
    path=tmp_path/'raw';names=list(screen.numeric.ECG_LEADS)
    wfdb.wrsamp(path.name,500,units=['mV']*8,sig_name=names,p_signal=ecg,fmt=['16']*8,write_dir=str(tmp_path))
    Path(c['cache_dir']).mkdir()
    item=dict(id='private_source',path=str(path),sampling_rate=500)
    cached,features,available=screen.numeric._cache_record(item,c,'scope')
    screen.replay_cache(item,cached,features,available,c)
    with np.load(cached) as a: changed={k:a[k] for k in a.files}
    changed['ecg'][0,0]+=1;np.savez_compressed(cached,**changed)
    with pytest.raises(ValueError,match='does not reproduce'):
        screen.replay_cache(item,cached,features,available,c)


def tiny_context(tmp_path,c):
    rng=np.random.default_rng(17);sources=[];targets=[]
    names=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    for role,rate,length,items in [('source',500,5000,sources),('target',1000,20000,targets)]:
        for index in range(2):
            path=tmp_path/f'{role}_{index}'
            wfdb.wrsamp(path.name,rate,units=['mV']*12,sig_name=names,p_signal=rng.normal(0,.2,(length,12)),
                        fmt=['16']*12,write_dir=str(tmp_path))
            item=dict(id=f'private_{role}_{index}',patient_id=f'{role}_patient_{index}',path=str(path),sampling_rate=rate)
            if role=='source': item['role']='train' if index==0 else 'validation'
            items.append(item)
    provenance=dict(pilot_evidence={'stub':'isolated numerical integration fixture'},source_bank_manifest_sha256='stub',
        source_training_patients=1,source_validation_patients=1,target=dict(patients=2,records=2),selected_source_records=2)
    return sources,targets,provenance


def test_full_expanded_audit_verifies_all_pairs_and_blocks_tampering(tmp_path,monkeypatch):
    c=config_for(tmp_path);context=tiny_context(tmp_path,c)
    monkeypatch.setattr(screen,'load_context',lambda config:context)
    result=screen.run(c)
    assert result['complete'] and result['recording_screen_passed']
    assert result['source_records']==result['target_records']==2
    assert result['continuous_pairs_checked']==result['all_possible_pairs']==4
    assert not result['patient_identity_independence_cleared'] and not result['target_classification_launched']
    assert screen.run(c,verify=True)==result==screen.run(c)
    root=Path(c['output_dir']);public=(root/'REPORT.md').read_text()+(root/'summary.json').read_text()
    assert 'source_patient_0' not in public and 'private_source_0' not in public
    report=root/'REPORT.md';original=report.read_bytes();report.write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):
        screen.run(c,verify=True)
    report.write_bytes(original)
    raw=Path(context[0][0]['path']).with_suffix('.dat');raw.write_bytes(b'changed')
    with pytest.raises(ValueError,match='inputs/code/runtime changed'):
        screen.run(c,verify=True)


def test_copy_control_failure_cannot_produce_a_passed_audit(tmp_path,monkeypatch):
    c=config_for(tmp_path);context=tiny_context(tmp_path,c)
    monkeypatch.setattr(screen,'load_context',lambda config:context)
    monkeypatch.setattr(screen.numeric,'_controls',lambda *args:(None,[dict(retrieved_known_record=False,confirmed_copied_segment=False)]))
    with pytest.raises(ValueError,match='controls failed'):
        screen.run(c)
    assert not (Path(c['output_dir'])/'summary.json').exists()
    assert json.loads((Path(c['output_dir'])/'control_failure.json').read_text())['known_copy_controls_passed'] is False


def test_queue_waits_for_worker_and_never_bypasses_incomplete_source(tmp_path,monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('source_overlap_queue',scripts/'queue_supervised_source_overlap.py')
    queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    root=tmp_path/'next';pilot=tmp_path/'pilot';pilot.mkdir()
    (pilot/'progress.json').write_text(json.dumps(dict(pid=321,state='training_source_only')))
    pc=tmp_path/'pilot.yaml';pc.write_text(yaml.safe_dump(dict(output_dir=str(pilot),bank_dir=str(tmp_path/'bank'))))
    calls=[];responses=iter(['python scripts/run_supervised_source_pilot.py','python -m mi_localization.supervised_source_train','',''])
    monkeypatch.setattr(queue,'command_for_pid',lambda pid:next(responses))
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:calls.append(seconds))
    monkeypatch.setattr(queue.subprocess,'run',lambda *args,**kwargs:pytest.fail('launched fitting/screen before source completion'))
    with pytest.raises(RuntimeError,match='without complete artifacts'):
        queue.wait_for_pilot(dict(source_config=str(pc)),root,123)
    assert calls==[20]
    assert json.loads((root/'queue_progress.json').read_text())['state']=='waiting_for_source_pilot'
