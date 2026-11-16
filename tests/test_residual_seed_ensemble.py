import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mi_localization import residual_context_study as study, residual_context_report as report
from mi_localization import residual_context_model as models, residual_seed_ensemble as ensemble
from mi_localization.signal_audit import file_hash
from test_residual_context import setup


def test_uniform_average_preserves_probabilities_and_is_not_hard_voting():
    p=np.array([[[.6,.4],[.1,.9]],[[.51,.49],[.3,.7]],[[0.,1.],[.5,.5]]])
    result=ensemble.uniform_probability_mean(p)
    np.testing.assert_allclose(result,[[.37,.63],[.3,.7]])
    assert result[0].argmax()==1  # Two of three member hard votes chose zero.
    np.testing.assert_allclose(ensemble.uniform_probability_mean(p[::-1]),result)
    np.testing.assert_allclose(ensemble.uniform_probability_mean(p[:,:,::-1]),result[:,::-1])
    for bad in [p[:1],p*2,np.full_like(p,np.nan),p-1]:
        with pytest.raises(ValueError):ensemble.uniform_probability_mean(bad)


def test_contrasts_preserve_paired_ensemble_estimand_and_all_arms():
    base=dict(arms=list(models.ARMS),classes=['MI','HC'],training_seeds=[42,123,2026])
    y=np.array([0,0,1,1]);p=np.eye(2)[y]*.8+.1
    members={(a,s):p.copy() for a in base['arms'] for s in base['training_seeds']}
    values={a:ensemble.uniform_probability_mean([members[a,s] for s in base['training_seeds']]) for a in base['arms']}
    contrasts,gains=ensemble.ensemble_contrasts(y,values,members,dict(bootstrap_draws=100,bootstrap_seed=42),base)
    assert len(contrasts)==8 and len(gains)==6
    for row in contrasts.values():
        for value in row.values():
            assert value['ensemble_delta_macro_f1']==0 and value['descriptive_patient_interval']==[0.,0.]


def test_actual_three_seed_inference_reverification_and_source_immutability(tmp_path,monkeypatch):
    bank,assigned,base=setup(tmp_path);base['training_seeds']=[42,123,2026]
    monkeypatch.setattr(study,'load_inputs',lambda config:(bank,assigned))
    source=study.run_study(base);report.summarize_study(base)
    cfg_path=tmp_path/'source.yaml';cfg_path.write_text(yaml.safe_dump(base))
    cfg=dict(source_config=str(cfg_path),output_dir=str(tmp_path/'ensembles'),operation='uniform_probability_mean',bootstrap_draws=100,bootstrap_seed=42)
    source_hashes={str(p):file_hash(p) for p in source.iterdir() if p.is_file()}
    monkeypatch.setattr(models,'fit',lambda *a,**k:pytest.fail('Ensemble analysis must never train'))
    summary=ensemble.run(cfg)
    assert summary['complete'] and summary['new_training_fits']==0
    assert summary['outer_models_restored_for_ensemble']==36 and summary['source_contexts_reverified']==108
    assert summary['source_matched_partition_checks']==18 and len(summary['metrics'])==6
    assert source_hashes=={str(p):file_hash(p) for p in source.iterdir() if p.is_file()}
    _,_,_,_,scope,_=ensemble.source_inputs(cfg)
    rows,p,audits=ensemble.predict_heldout_fold(bank,assigned,base,source,scope,'ordered_context_aux',1)
    assert len(audits)==3 and all(a['patient_overlap']==0 for a in audits)
    altered=copy.deepcopy(bank);query=bank.ids[rows];qrows=bank.beats_for(query)
    altered.y[qrows]=(altered.y[qrows]+1)%6
    np.testing.assert_allclose(ensemble.predict_heldout_fold(altered,assigned,base,source,scope,'ordered_context_aux',1)[1],p)
    with pytest.raises(ValueError,match='config'):
        ensemble.predict_heldout_fold(bank,assigned,dict(base,training_seeds=[42,123]),source,scope,'ordered_context_aux',1)
    with pytest.raises(ValueError,match='namespace'):
        ensemble.run(dict(cfg,bootstrap_seed=123))
    with pytest.raises(ValueError,match='options'):
        ensemble.source_inputs(dict(cfg,weights=[.2,.4,.4]))
    path=source/'outer_01_ordered_context_aux_s42.pt';path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='checkpoint'):
        ensemble.predict_heldout_fold(bank,assigned,base,source,scope,'ordered_context_aux',1)
