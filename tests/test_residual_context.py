import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from sklearn.metrics import f1_score

from mi_localization import chronology_model as chronology
from mi_localization import residual_context_model as models
from mi_localization import residual_context_study as study, residual_context_report as report
from mi_localization.contrastive import initial_state_hash
from test_chronology import fixture
from test_chronology_lengths import script


def setup(tmp_path):
    bank, assigned, config = fixture(tmp_path)
    config.update(arms=list(models.ARMS), output_dir=str(tmp_path/"residual"), sequence_length=4,
                  auxiliary_weight=.5, gate_initial_bias=-3.)
    return bank, assigned, config


def test_default_protocol_counts_exposure_and_all_six_matched_initializations():
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/residual_context_v1.yaml").read_text())
    study.validate_config(config)
    assert config["sequence_length"] == 4 and config["auxiliary_weight"] == .5
    assert config["windows_per_epoch"]*config["sequence_length"] == 8192
    assert config["batch_size"]*config["sequence_length"] == 128
    assert config["windows_per_epoch"]//config["batch_size"] == 64
    outer = config["outer_splits"]*len(config["training_seeds"])*len(config["arms"])
    assert outer == 180 and outer*(config["inner_splits"]+1) == 720
    states = [initial_state_hash(models.initial_model(config, 42, arm)) for arm in config["arms"]]
    assert len(set(states)) == 1
    with pytest.raises(ValueError): study.validate_config(dict(config, auxiliary_weight=0))


def test_direct_path_is_exact_beat_mean_and_ignores_rr_and_context_permutations(tmp_path):
    bank, assigned, config = setup(tmp_path)
    norm = chronology.SequenceNormalizer().fit(bank, bank.ids[assigned != 1])
    plan = [(0, 0, 4), (1, 0, 3)]
    dataset = models.ResidualDataset(bank, norm, plan, 4, "beat_only", 42, 1)
    x, rr, mask, _, order = [torch.stack([torch.as_tensor(dataset[i][j]) for i in range(2)]) for j in range(5)]
    model = models.initial_model(config, 42, "beat_only").eval()
    logp, beat, gate = model(x, rr, mask, order)
    expected = (beat.softmax(-1)*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
    torch.testing.assert_close(logp.exp(), expected)
    assert torch.equal(gate, torch.zeros_like(gate))
    changed = order.flip(1)
    other, _, _ = model(x, rr+100, mask, changed)
    torch.testing.assert_close(logp, other)
    padded_x = x.clone();padded_rr = rr.clone()
    padded_x[~mask] = 100;padded_rr[~mask] = 100
    after, _, _ = model(padded_x, padded_rr, mask, order)
    torch.testing.assert_close(logp, after)


def test_only_context_is_shuffled_and_wave_rr_pairs_stay_attached(tmp_path):
    bank, assigned, config = setup(tmp_path)
    train = bank.ids[assigned != 1]
    norm = chronology.SequenceNormalizer().fit(bank, train)
    plan = chronology.training_schedule(bank, train, 4, 16, 42, 1)
    a = models.ResidualDataset(bank, norm, plan, 4, "ordered_context_aux", 42, 1)
    b = models.ResidualDataset(bank, norm, plan, 4, "shuffled_context_aux", 42, 1)
    assert a.audit()["direct_token_order_sha256"] == b.audit()["direct_token_order_sha256"]
    assert a.audit()["window_plan_sha256"] == b.audit()["window_plan_sha256"]
    assert b.audit()["nonidentity_window_orders"] > 0
    model = models.initial_model(config, 42, "ordered_context_aux").eval()
    for i in range(3):
        ax, ar, am, al, ao = a[i];bx, br, bm, bl, bo = b[i]
        assert torch.equal(ax, bx) and torch.equal(ar, br) and torch.equal(am, bm) and al == bl
        size = int(am.sum());assert sorted(bo[:size].tolist()) == list(range(size))
        assert bo[size:].tolist() == list(range(size, 4))
        values = []
        handle = model.input_gates.register_forward_pre_hook(lambda _, args: values.append(args[0].detach().clone()))
        _, ordered_beats, _ = model(ax[None], ar[None], am[None], ao[None])
        ordered_tokens = torch.stack(values, dim=1);values.clear()
        _, shuffled_beats, _ = model(bx[None], br[None], bm[None], bo[None])
        shuffled_tokens = torch.stack(values, dim=1);handle.remove()
        torch.testing.assert_close(shuffled_tokens, ordered_tokens[:, bo])
        torch.testing.assert_close(ordered_beats, shuffled_beats)
        model.train()
        torch.manual_seed(123)
        direct_a = model(ax[None], ar[None], am[None], ao[None])[1]
        torch.manual_seed(123)
        direct_b = model(bx[None], br[None], bm[None], bo[None])[1]
        torch.testing.assert_close(direct_a, direct_b)
        model.eval()
    all_plan, _ = chronology.inference_schedule(bank, bank.ids[:3], 4)
    single, _ = chronology.inference_schedule(bank, bank.ids[1:2], 4)
    full = models.ResidualDataset(bank, norm, all_plan, 4, "shuffled_context", 42, 0)
    one = models.ResidualDataset(bank, norm, single, 4, "shuffled_context", 42, 0)
    for i, window in enumerate(single):
        assert np.array_equal(full.order(all_plan.index(window)), one.order(i))


def test_context_gate_gradients_auxiliary_loss_and_masked_tails(tmp_path):
    bank, assigned, config = setup(tmp_path)
    norm = chronology.SequenceNormalizer().fit(bank, bank.ids[assigned != 1])
    data = models.ResidualDataset(bank, norm, [(0, 0, 4), (1, 0, 3)], 4, "ordered_context_aux", 42, 1)
    x, rr, mask, labels, order = [torch.stack([torch.as_tensor(data[i][j]) for i in range(2)]) for j in range(5)]
    model = models.initial_model(config, 42, "ordered_context_aux").eval()
    logp, beat, gate = model(x, rr, mask, order)
    assert torch.allclose(gate, torch.full_like(gate, torch.sigmoid(torch.tensor(-3.)).item()))
    loss, window, auxiliary = models.objective(logp, beat, labels, mask, .5)
    torch.testing.assert_close(loss, window+.5*auxiliary)
    loss.backward()
    for parameter in [model.encoder[0].weight, model.beat_head[1].weight, model.context_head[1].weight,
                      model.input_gates.weight, model.rr_encoder[0].weight, model.gate.bias]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    modified = beat.detach().clone();modified[~mask] += 1000
    torch.testing.assert_close(models.objective(logp.detach(), modified, labels, mask, .5)[2], auxiliary)
    changed_x=x.clone();changed_x[~mask] += 100
    changed_rr=rr.clone();changed_rr[~mask] += 100
    torch.testing.assert_close(model(changed_x, changed_rr, mask, order)[0], logp)
    with torch.no_grad(): model.gate.bias.fill_(-1000)
    closed = model(x, rr, mask, order)[0]
    direct = (beat.detach().softmax(-1)*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
    torch.testing.assert_close(closed.exp(), direct)


def test_actual_patient_inference_is_label_independent_and_record_weighted(tmp_path):
    bank, assigned, config = setup(tmp_path)
    norm = chronology.SequenceNormalizer().fit(bank, bank.ids[assigned != 1])
    model = models.initial_model(config, 42, "beat_only")
    patients = bank.ids[:2]
    p, coverage = models.predict(model, norm, bank, patients, config, 42)
    assert coverage["all_beats_exactly_once"] and coverage["query_beats"] == 16
    plan, _ = chronology.inference_schedule(bank, patients, 4)
    dataset = models.ResidualDataset(bank, norm, plan, 4, "beat_only", 42, 0)
    record_scores = {}
    model.eval()
    with torch.no_grad():
        for i, (record, _, size) in enumerate(plan):
            x, rr, mask, _, order = dataset[i]
            q = model(x[None], rr[None], mask[None], order[None])[0].exp()[0].numpy()
            record_scores[record] = record_scores.get(record, 0)+q*size/len(bank.rows(record))
    expected = np.stack([np.mean([v for r,v in record_scores.items() if bank.record_patients[r] == patient],axis=0) for patient in patients])
    np.testing.assert_allclose(p, expected, atol=1e-7)
    changed = copy.deepcopy(bank);changed.y[changed.beats_for(patients)] = (changed.y[changed.beats_for(patients)]+1)%6
    np.testing.assert_allclose(models.predict(model, norm, changed, patients, config, 42)[0], p)


def test_vectorized_patient_bootstrap_matches_explicit_repeated_patients():
    y=np.array([0,0,1,1,2,2]);prediction=np.array([0,1,1,2,0,2])
    for labels in [np.array([0,1,2]),np.array([0,1])]:
        weights=report.patient_weights(y,labels,50,42)
        fast=report.weighted_f1(y,prediction,labels,weights)
        for i,w in enumerate(weights):
            rows=np.repeat(np.arange(len(y)),w)
            assert fast[i] == pytest.approx(f1_score(y[rows],prediction[rows],labels=labels,average="macro",zero_division=0))
            assert all(w[y==c].sum()==2 for c in labels)
    assert np.array_equal(weights,report.patient_weights(y,np.array([0,1]),50,42))


def test_nested_actual_training_restore_resume_and_tamper_guards(tmp_path,monkeypatch):
    bank,assigned,config=setup(tmp_path)
    monkeypatch.setattr(study,"load_inputs",lambda cfg:(bank,assigned))
    root=study.run_study(config)
    summary=report.summarize_study(config)
    assert summary["fits"]==36 and summary["outer_models_restored"]==12 and summary["matched_partition_checks"]==6
    assert summary["all_query_beats_covered"] and len(summary["contrasts"])==8
    monkeypatch.setattr(models,"fit",lambda *a,**k:pytest.fail("Completed fits must resume without training"))
    assert study.run_study(config)==root and report.summarize_study(config)==summary
    with pytest.raises(ValueError,match="changed"):
        study.run_study(dict(config,auxiliary_weight=.7))
    target=root/"inner_01_ordered_context_aux_s42_i1.json"
    value=json.loads(target.read_text());value["fit"]["history"][0]["beat_loss"]+=1
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="loss/gate"):
        report.summarize_study(config)


def test_live_monitor_new_worker_and_launcher_lock_and_completion(tmp_path,monkeypatch):
    live=script("live_status.py");runner=script("run_residual_context.py")
    handle=runner.acquire_lock(tmp_path/"runner.lock")
    with pytest.raises(RuntimeError,match="already"):
        runner.acquire_lock(tmp_path/"runner.lock")
    handle.close();runner.acquire_lock(tmp_path/"runner.lock").close()
    target=tmp_path/"results/neural_residual_context_v1/complete";target.mkdir(parents=True)
    (target/"progress.json").write_text(json.dumps({"pid":12,"state":"inner_training"}))
    monkeypatch.setattr(live,"process_command",lambda pid:("python -m mi_localization.residual_context_study" if pid==12 else "","checked"))
    value=live.snapshot(tmp_path)["residual_context"]
    assert value["phase"]=="inner_training" and value["planned_fits"]==720
    assert not runner.verified_ready(tmp_path)
    (target/"study_manifest.json").write_text(json.dumps({"study_fingerprint":"fixture"}))
    (target/"REPORT.md").write_text("verified")
    (target/"summary.json").write_text(json.dumps({"study_fingerprint":"fixture","complete":True,"fits":720,
        "outer_models_restored":180,"matched_partition_checks":120,"all_query_beats_covered":True}))
    assert runner.verified_ready(tmp_path)
    assert live.snapshot(tmp_path)["residual_context"]["report_present"]
