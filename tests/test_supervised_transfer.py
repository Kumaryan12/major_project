"""Matched nested transfer, encoder-only provenance and full checkpoint restoration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mi_localization import supervised_transfer as study,supervised_transfer_report as report
from mi_localization.supervised_source_audit import canonical,file_hash
from test_regularization import tiny_bank

REAL_LOAD_SOURCES=study.load_sources


def fixture(tmp_path,monkeypatch):
    old,waves,y,groups=tiny_bank(tmp_path)
    c=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/supervised_transfer_v1.yaml').read_text())
    for k in ['seed','split_seed','training_seeds','paper_config','beats_path','bank_dir','folds_path','output_dir',
              'classes','outer_splits','inner_splits','device','torch_threads','widths','pool_bins','batch_size','samples_per_epoch','max_epochs','patience']:
        c[k]=old[k]
    c.update(bootstrap_draws=100,source_config=str(tmp_path/'source.yaml'),overlap_config=str(tmp_path/'overlap.yaml'))
    states={};exports={}
    for arm,seed in zip(study.ARMS[1:],[17,19]):
        model=study.backend.initial_model(c,seed,'scratch',None)
        states[arm]={k:v.clone() for k,v in model.encoder.state_dict().items()}
        exports[arm]=dict(encoder_state_sha256=study.initial_state_hash(model.encoder))
    provenance=dict(exports=exports,source_training_seed=42,source_training_patients=4,source_validation_patients=2,
        patient_identity_independence_cleared=False,expanded_recording_screen=dict(source_records=6,target_records=27,
        continuous_pairs_checked=12,positive_controls=2,positive_controls_retrieved_and_confirmed=2))
    monkeypatch.setattr(study,'load_sources',lambda config,bank:(states,provenance))
    d=study.load_inputs(c);scope=study.make_scope(c,d,list(range(1,c['outer_splits']+1)))
    receipt=Path(c['output_dir'])/'complete/review/preflight.json';receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(dict(scope=scope,source_models_independently_verified=True,expanded_screen_independently_verified=True)))
    return c,waves,y,groups,states,provenance


def export_fixture(tmp_path,config,states):
    root=tmp_path/'pilot';root.mkdir();training=dict(fixture='native source training scope')
    pilot=dict(output_dir=str(root),widths=config['widths'],pool_bins=config['pool_bins']);runs=[]
    for arm,objective in study.OBJECTIVES.items():
        member=root/'runs'/f'{objective}_42';member.mkdir(parents=True)
        scope=dict(training,objective=objective,seed=42);fingerprint=canonical(scope)
        (member/'manifest.json').write_text(json.dumps(scope))
        torch.save(dict(encoder_state_dict=states[arm],fingerprint=fingerprint,widths=config['widths'],pool_bins=config['pool_bins'],
                        source_only=True,ptb_patient_independence_cleared=False,best_epoch=1,
                        normalization={'mean':[1000]*3,'scale':[100]*3}),member/'encoder.pt')
        r=dict(objective=objective,seed=42,complete=True,fingerprint=fingerprint,best_epoch=1,files={'encoder.pt':file_hash(member/'encoder.pt')})
        (member/'summary.json').write_text(json.dumps(r));runs.append(r)
    return pilot,dict(runs=runs),training


def test_checked_exports_load_both_objectives_without_source_normalization(tmp_path,monkeypatch):
    c,_,_,_,states,_=fixture(tmp_path,monkeypatch)
    pilot,summary,training=export_fixture(tmp_path,c,states)
    loaded,evidence=study.checked_exports(c,pilot,summary,training)
    assert set(loaded)==set(study.ARMS[1:])
    assert all(not v['source_only_normalization_transferred'] for v in evidence.values())
    for arm in loaded:
        assert all(torch.equal(v,states[arm][k]) for k,v in loaded[arm].items())
    with pytest.raises(ValueError,match='architecture mismatch'):
        study.checked_exports(dict(c,widths=[4,8,16]),pilot,summary,training)
    p=Path(pilot['output_dir'])/'runs/certain_42/encoder.pt';p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='export changed'):
        study.checked_exports(c,pilot,summary,training)


def test_three_arms_match_target_initial_heads_sampling_and_training_only_norm(tmp_path,monkeypatch):
    c,_,_,_,_,_=fixture(tmp_path,monkeypatch);data=study.load_inputs(c)
    train=np.arange(54);query=np.arange(54,81);fits=[];outputs=[]
    for arm in study.ARMS:
        fit,p=study.member(c,data,tmp_path/'members','scope',arm,train,query,42,.001,2,
            dict(training_seed=42,outer_fold=1,fit_id='outer'),False)
        fits.append(fit);outputs.append(p)
    study.check_matched(fits)
    assert len({f['initial_classifier_sha256'] for f in fits})==1
    assert len({f['initial_encoder_sha256'] for f in fits})==3
    assert all(max(abs(x) for x in f['normalization']['wave_mean_'])<1 for f in fits)
    monkeypatch.setattr(study.backend,'fit_transfer',lambda *a,**k:pytest.fail('restoration trained a model'))
    for arm,expected in zip(study.ARMS,outputs):
        _,p=study.member(c,data,tmp_path/'members','scope',arm,train,query,42,.001,2,
            dict(training_seed=42,outer_fold=1,fit_id='outer'),False,verify=True)
        np.testing.assert_allclose(p,expected,atol=0,rtol=0)
    with pytest.raises(ValueError,match='overlapping target patients'):
        study.member(c,data,tmp_path/'bad','scope','scratch',train,train[:3],42,.001,2,{},False)


def test_full_nested_study_restores_every_fit_and_repeats_report(tmp_path,monkeypatch):
    c,_,_,_,_,_=fixture(tmp_path,monkeypatch)
    calls=[];real=study.backend.fit_transfer
    def tracked(w,y,g,train,valid,*a,**kw):
        calls.append((set(g[train]),None if valid is None else set(g[valid])))
        return real(w,y,g,train,valid,*a,**kw)
    monkeypatch.setattr(study.backend,'fit_transfer',tracked)
    _,result,_,root,partial=study.run_study(c)
    assert not partial and len(calls)==135 and result['completed_fit_count']==135
    assert result['matched_context_checks']==45
    monkeypatch.setattr(study.backend,'fit_transfer',lambda *a,**k:pytest.fail('completed fit was retrained'))
    s=report.summarize(c)
    assert s['restored_fit_count']==135 and s['outer_model_restorations']==27
    assert set(s['equal_probability_seed_ensembles'])==set(study.ARMS)
    assert not s['patient_identity_independence_cleared'] and not s['next_experiment_automatically_launched']
    assert report.summarize(c)==s
    original=(root/'per_seed.csv').read_bytes();(root/'per_seed.csv').write_text('changed')
    with pytest.raises(ValueError,match='report/metrics/artifacts'):
        report.summarize(c)
    (root/'per_seed.csv').write_bytes(original)
    p=root/'review/selections/s42_f01_source_certain.json';selection=json.loads(p.read_text())
    selection['selection']['selected']['final_epochs']+=1;p.write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='Inner-only'):
        study.run_study(c,verify=True)


def test_fit_artifact_and_membership_tampering_fail_before_inference(tmp_path,monkeypatch):
    c,_,_,_,_,_=fixture(tmp_path,monkeypatch);data=study.load_inputs(c)
    context=dict(training_seed=42,outer_fold=1,fit_id='outer')
    study.member(c,data,tmp_path/'member','scope','scratch',np.arange(54),np.arange(54,81),42,.001,2,context,False)
    root=tmp_path/'member/review/fits/s42_f01_scratch_outer'
    (root/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='scope/artifact changed'):
        study.member(c,data,tmp_path/'member','scope','scratch',np.arange(54),np.arange(54,81),42,.001,2,context,False,True)
    with pytest.raises(ValueError,match='Incomplete'):
        study.member(c,data,tmp_path/'missing','scope','source_certain',np.arange(54),np.arange(54,81),42,.001,2,context,False,True)


def test_inner_selection_does_not_consume_outer_scores_and_rejects_missing_arms(tmp_path,monkeypatch):
    c,_,_,_,_,_=fixture(tmp_path,monkeypatch)
    candidates={f'rate_{i}':[dict(best_validation_macro_f1=.5,selected_epoch=e,outer_f1=float(i)) for e in [1,2]] for i in range(2)}
    result=study.backend.choose_rate(candidates,c)
    assert result['selected']['learning_rate']==.001 and result['selected']['final_epochs']==2
    with pytest.raises(ValueError,match='both declared'):
        study.validate_config(dict(c,arms=['scratch','source_certain']))
    with pytest.raises(ValueError,match='at least three'):
        report.summarize(dict(c,training_seeds=[42]))
    receipt=Path(c['output_dir'])/'complete/review/preflight.json';original=receipt.read_bytes();receipt.unlink()
    with pytest.raises(ValueError,match='preflight required'):
        study.run_study(c)
    altered=json.loads(original);altered['expanded_screen_independently_verified']=False;receipt.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='preflight scope changed'):
        study.run_study(c)
    receipt.write_bytes(original)
    Path(c['folds_path']).unlink()
    with pytest.raises(ValueError,match='Frozen patient folds required'):
        study.load_inputs(c)


def test_ensemble_bootstrap_counts_patients_once_and_preserves_zero_differences():
    truth=np.repeat(np.arange(3),4);p=np.eye(3)[truth]*.8+.2/3
    all_p={arm:p.copy() for arm in study.ARMS}
    result=report.ensemble_intervals(truth,all_p,['AMI','IMI','HC'],100,42)
    assert result['unique_patients']==12
    assert all(v==[1.,1.] for v in result['arm_macro_f1_intervals'].values())
    assert all(v==[0.,0.] for v in result['paired_difference_intervals'].values())


def test_source_gate_blocks_failed_screen_identity_claims_and_target_drift(tmp_path,monkeypatch):
    c,_,_,_,states,_=fixture(tmp_path,monkeypatch)
    pilot,source_summary,training=export_fixture(tmp_path,c,states)
    ac=tmp_path/'source_audit.yaml';ac.write_text(yaml.safe_dump(dict(target_classes=c['classes'])));pilot['audit_config']=str(ac)
    Path(c['source_config']).write_text(yaml.safe_dump(pilot));root=tmp_path/'screen';root.mkdir()
    (Path(pilot['output_dir'])/'training_manifest.json').write_text(json.dumps(training))
    sc=dict(source_config=c['source_config'],output_dir=str(root),cache_dir=str(tmp_path/'cache'))
    Path(c['overlap_config']).write_text(yaml.safe_dump(sc))
    bank=study.regularization.load_bank(c)[3];source_evidence={'fixture':'source public provenance'}
    sources=[dict(id='source',path='source path')];targets=[dict(id='target',path='target path')]
    provenance=dict(pilot_evidence=source_evidence,source_training_patients=4,source_validation_patients=2,
        target=dict(patients=bank['n_patients'],records=bank['n_records'],beats=bank['n_beats'],
                    beats_sha256=bank['beats_sha256'],folds_sha256=file_hash(c['folds_path'])))
    scope=dict(config=sc,provenance=provenance,source_inventory_sha256=canonical(sources),target_inventory_sha256=canonical(targets),
        code_sha256=file_hash(study.overlap.__file__),numerical_code_sha256=file_hash(study.overlap.numeric.__file__),
        signal_code_sha256=file_hash(study.overlap.numeric.signal.__file__),kors_code_sha256=file_hash(study.overlap.numeric.external.__file__))
    fp=canonical(scope);(root/'screen_manifest.json').write_text(json.dumps(dict(scope,screen_fingerprint=fp)))
    screened=dict(complete=True,screen_fingerprint=fp,recording_screen_passed=True,patient_identity_independence_cleared=False,
        target_classification_launched=False,near_recording_candidate_pairs=0,exact_encoded_data_matches=0,incomplete_active_lead_pairs=0,
        positive_controls=2,positive_controls_retrieved_and_confirmed=2,source_records=6,target_records=27,continuous_pairs_checked=12)
    def write_screen(changes):
        (root/'summary.json').write_text(json.dumps(dict(screened,**changes)))
        (root/'output_manifest.json').write_text(json.dumps(dict(screen_fingerprint=fp,
            files={'summary.json':file_hash(root/'summary.json')},cache_files={})))
    write_screen({})
    monkeypatch.setattr(study.overlap,'completed_pilot',lambda p:(source_summary,source_evidence))
    monkeypatch.setattr(study.overlap,'load_context',lambda cfg:(sources,targets,provenance))
    monkeypatch.setattr(study.overlap,'inventory',lambda items:items)
    assert set(REAL_LOAD_SOURCES(c,bank)[0])==set(study.ARMS[1:])
    for changes in [dict(near_recording_candidate_pairs=1),dict(recording_screen_passed=False),
                    dict(patient_identity_independence_cleared=True),dict(positive_controls_retrieved_and_confirmed=1)]:
        write_screen(changes)
        with pytest.raises(ValueError,match='did not pass'):
            REAL_LOAD_SOURCES(c,bank)
    write_screen({})
    with pytest.raises(ValueError,match='cohort/folds differ'):
        REAL_LOAD_SOURCES(c,dict(bank,n_patients=bank['n_patients']+1))


def test_source_export_finiteness_is_checked_after_provenance(tmp_path,monkeypatch):
    c,_,_,_,states,_=fixture(tmp_path,monkeypatch);pilot,summary,training=export_fixture(tmp_path,c,states)
    path=Path(pilot['output_dir'])/'runs/certain_42/encoder.pt';export=torch.load(path,weights_only=True)
    first=next(iter(export['encoder_state_dict']));export['encoder_state_dict'][first].view(-1)[0]=float('nan')
    torch.save(export,path);summary['runs'][0]['files']['encoder.pt']=file_hash(path)
    with pytest.raises(ValueError,match='Invalid source encoder weights'):
        study.checked_exports(c,pilot,summary,training)
