import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mi_localization import chronology_bank as data_module, chronology_model as models
from mi_localization import chronology_study as study, chronology_report as report
from mi_localization.contrastive import initial_state_hash


def fixture(tmp_path):
    rng = np.random.default_rng(44)
    classes = ["AMI", "ALMI", "ASMI", "IMI", "ILMI", "HC"]
    ids = np.array([f"patient{i:03d}" for i in range(24)])
    labels = np.repeat(np.arange(6), 4)
    groups, ys, rows, offsets, patients, rr, valid, frames = [], [], [], [0], [], [], [], []
    for index, patient in enumerate(ids):
        for local, size in enumerate((5, 3)):
            record = f"record{index:03d}_{local}"
            start = len(rows);record_rows = np.arange(start, start+size)
            rows.extend(record_rows);offsets.append(len(rows));patients.append(patient)
            gaps = .7+rng.random(size-1)*.3
            peaks = np.r_[400, 400+np.cumsum(np.round(gaps*1000))].astype(int)
            rr.extend(np.r_[0, np.diff(peaks)/1000]);valid.extend([False]+[True]*(size-1))
            groups.extend([patient]*size);ys.extend([labels[index]]*size)
            frames.extend(dict(record=record, patient_id=patient, label=classes[labels[index]], beat_index=i, r_peak=int(p)) for i, p in enumerate(peaks))
    waves = rng.normal(size=(len(rows), 3, 65)).astype(np.float32)
    bank = data_module.RecordBank(waves, np.array(ys), np.array(groups), np.array(rows), np.array(offsets),
                np.array(patients), np.array(rr, np.float32), np.array(valid, bool), {"fixture": True})
    assigned = np.tile([1, 2, 1, 2], 6)
    beats = tmp_path/"beats.csv";pd.DataFrame(frames).to_csv(beats, index=False)
    folds = tmp_path/"folds.csv";pd.DataFrame(dict(patient_id=ids, label=np.asarray(classes)[labels], fold=assigned)).to_csv(folds, index=False)
    cfg = dict(seed=42, split_seed=42, training_seeds=[42], arms=list(models.ARMS), classes=classes,
               outer_splits=2, inner_splits=2, folds_path=str(folds), beats_path=str(beats),
               sequence_bank_dir=str(tmp_path/"metadata"), output_dir=str(tmp_path/"study"),
               device="cpu", torch_threads=1, widths=[4, 4, 4], pool_bins=1, dropout=.1,
               rr_embedding=2, hidden_size=4, sequence_length=3, windows_per_epoch=2, batch_size=1,
               inference_batch_size=16, max_epochs=1, patience=1, learning_rate=.001, weight_decay=.01, bootstrap_draws=100)
    return bank, assigned, cfg


def test_metadata_preserves_original_rr_and_detects_corruption(tmp_path, monkeypatch):
    bank, assigned, cfg = fixture(tmp_path)
    monkeypatch.setattr(data_module, "load_bank", lambda config: (bank.waves, bank.y, bank.groups, {"fixture": True}))
    result = data_module.prepare_bank(cfg)
    np.testing.assert_array_equal(result.record_rows, bank.record_rows)
    np.testing.assert_array_equal(result.rr, bank.rr)
    assert len(result.ids) == 24 and result.rr_valid.sum() == len(bank.waves)-48
    assert data_module.prepare_bank(cfg).provenance == result.provenance
    path = Path(cfg["sequence_bank_dir"])/"metadata.npz";path.write_bytes(path.read_bytes()+b"bad")
    with pytest.raises(ValueError, match="cache changed"):
        data_module.prepare_bank(cfg)


def test_window_boundaries_padding_joint_permutation_and_query_independence(tmp_path):
    bank, assigned, cfg = fixture(tmp_path)
    norm = models.SequenceNormalizer().fit(bank, bank.ids[assigned != 1])
    plan = models.training_schedule(bank, bank.ids[assigned != 1], 3, 20, 42, 1)
    assert all(bank.record_patients[r] in set(bank.ids[assigned != 1]) for r, _, _ in plan)
    ordered = models.WindowDataset(bank, norm, plan, 3, "ordered_rr", 42, 1)
    shuffled = models.WindowDataset(bank, norm, plan, 3, "shuffled_rr", 42, 1)
    assert ordered.audit()["window_plan_sha256"] == shuffled.audit()["window_plan_sha256"]
    for i in range(len(plan)):
        x, rr, mask, label = ordered[i];sx, sr, sm, sl = shuffled[i]
        size = int(mask.sum());order = shuffled.order(i)
        torch.testing.assert_close(sx[:size], x[order]);torch.testing.assert_close(sr[:size], rr[order])
        assert label == sl and torch.equal(mask, sm)
    all_plan, audit = models.inference_schedule(bank, bank.ids[:3], 3)
    assert audit["query_beats"] == 24 and audit["all_beats_exactly_once"]
    assert any(size < 3 for _, _, size in all_plan)
    single, _ = models.inference_schedule(bank, bank.ids[1:2], 3)
    full = models.WindowDataset(bank, norm, all_plan, 3, "shuffled_rr", 42, 0)
    one = models.WindowDataset(bank, norm, single, 3, "shuffled_rr", 42, 0)
    for i, window in enumerate(single):
        location = all_plan.index(window)
        assert np.array_equal(full.order(location), one.order(i))
        torch.testing.assert_close(full[location][0], one[i][0])


def test_normalizer_is_training_only(tmp_path):
    bank, assigned, cfg = fixture(tmp_path)
    train, query = bank.ids[assigned != 1], bank.ids[assigned == 1]
    a = models.SequenceNormalizer().fit(bank, train)
    altered = copy.deepcopy(bank);rows = altered.beats_for(query)
    altered.waves[rows] += 1000;altered.rr[rows] += 10
    b = models.SequenceNormalizer().fit(altered, train)
    assert a.audit() == b.audit()
    assert models.SequenceNormalizer.restore(a.audit()).audit() == a.audit()


def test_gru_equations_controls_rr_ablation_padding_and_gradients(tmp_path):
    bank, assigned, cfg = fixture(tmp_path)
    hashes = [initial_state_hash(models.initial_model(cfg, 42, arm)) for arm in ("ordered_no_rr", "ordered_rr", "shuffled_rr")]
    assert len(set(hashes)) == 1
    model = models.initial_model(cfg, 42, "ordered_rr").eval()
    x = torch.randn(2, 3, 3, 65);rr = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    cell = torch.nn.GRUCell(64+cfg["rr_embedding"], cfg["hidden_size"])
    with torch.no_grad():
        cell.weight_ih.copy_(model.input_gates.weight);cell.bias_ih.copy_(model.input_gates.bias)
        cell.weight_hh.copy_(model.hidden_gates.weight);cell.bias_hh.copy_(model.hidden_gates.bias)
    tokens = torch.cat([model.encoder(x.flatten(0, 1)).reshape(2, 3, 64), model.rr_encoder(rr)], -1)
    h = torch.zeros(2, cfg["hidden_size"])
    for i in range(3): h = torch.where(mask[:, i, None], cell(tokens[:, i], h), h)
    torch.testing.assert_close(model(x, rr, mask), model.head(h), rtol=1e-5, atol=1e-6)
    changed = x.clone();changed[0, 2] = 1000
    changed_rr = rr.clone();changed_rr[0, 2] = 1000
    torch.testing.assert_close(model(x, rr, mask), model(changed, changed_rr, mask))
    no_rr = models.initial_model(cfg, 42, "ordered_no_rr").eval()
    torch.testing.assert_close(no_rr(x, rr, mask), no_rr(x, rr+100, mask))
    loss = model(x, rr, mask).square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    mean = models.initial_model(cfg, 42, "mean_rr").eval();swap = [1, 0, 2]
    torch.testing.assert_close(mean(x, rr, mask), mean(x[:, swap], rr[:, swap], mask[:, swap]), rtol=1e-5, atol=1e-6)


def test_inference_all_beats_equal_record_weight_and_query_label_independence(tmp_path):
    bank, assigned, cfg = fixture(tmp_path)
    model = models.initial_model(cfg, 42, "shuffled_rr").eval()
    norm = models.SequenceNormalizer().fit(bank, bank.ids[assigned != 1])
    query = bank.ids[assigned == 1][:2]
    p, audit = models.predict(model, norm, bank, query, cfg, 42)
    single, _ = models.predict(model, norm, bank, query[1:], cfg, 42)
    np.testing.assert_allclose(p[1:], single, rtol=1e-5, atol=1e-6)
    assert audit["query_beats"] == 16 and audit["query_records"] == 4
    altered = copy.deepcopy(bank);altered.y[altered.beats_for(query)] = 0
    changed, _ = models.predict(model, norm, altered, query, cfg, 42)
    np.testing.assert_array_equal(p, changed)
    # A long record must not outweigh a short record at the patient step.
    class KnownLogits(torch.nn.Module):
        arm = "ordered_rr"
        def forward(self, x, rr, mask):
            result = x.new_zeros((len(x), 6));result[:, 0] = x[:, 0, 0, 0]
            return result
    toy = copy.deepcopy(bank);toy.waves[:8] = 0
    toy.waves[0] = 1;toy.waves[3] = 2;toy.waves[5] = 3
    identity = copy.deepcopy(norm);identity.wave.wave_mean_[:] = 0;identity.wave.wave_scale_[:] = 1
    actual, _ = models.predict(KnownLogits(), identity, toy, toy.ids[:1], cfg, 42)
    probabilities = np.exp(np.array([[1., 0, 0, 0, 0, 0], [2., 0, 0, 0, 0, 0], [3., 0, 0, 0, 0, 0]]))
    probabilities /= probabilities.sum(1, keepdims=True)
    expected = .5*((3*probabilities[0]+2*probabilities[1])/5+probabilities[2])
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-7)


def test_complete_nested_actual_training_resume_restore_and_tamper(tmp_path, monkeypatch):
    bank, assigned, cfg = fixture(tmp_path)
    monkeypatch.setattr(study, "load_inputs", lambda config: (bank, assigned))
    root = study.run_study(cfg)
    summary = report.summarize_study(cfg)
    assert summary["fits"] == 24 and summary["outer_models_restored"] == 8
    assert summary["matched_gru_partition_checks"] == 6 and len(summary["contrasts"]) == 3
    monkeypatch.setattr(models, "fit", lambda *a, **k: pytest.fail("Completed fits must resume"))
    assert study.run_study(cfg) == root and report.summarize_study(cfg) == summary
    with pytest.raises(ValueError, match="changed"):
        study.run_study(dict(cfg, hidden_size=5))
    path = next(root.glob("outer_*ordered_rr*.json"))
    saved = json.loads(path.read_text());saved["coverage"]["all_beats_exactly_once"] = False
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="coverage changed"):
        report.summarize_study(cfg)


def test_queue_requires_complete_verified_signature_dependency(tmp_path):
    source = Path(__file__).resolve().parents[1]/"scripts/run_chronological_vcg.py"
    spec = importlib.util.spec_from_file_location("chronology_queue", source)
    queue = importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    root = tmp_path/"results/signature_vcg_v1/complete";root.mkdir(parents=True)
    (root/"study_manifest.json").write_text(json.dumps({"study_fingerprint": "expected"}))
    assert not queue.dependency_ready(tmp_path, "expected")
    (root/"REPORT.md").write_text("done")
    (root/"summary.json").write_text(json.dumps({"study_fingerprint": "expected", "complete": True, "total_contexts": 1050,
        "outer_contexts": 150, "all_outer_contexts_restored": True, "all_bootstrap_caches_replayed": True}))
    assert queue.dependency_ready(tmp_path, "expected")
    with pytest.raises(RuntimeError, match="manifest changed"):
        queue.dependency_ready(tmp_path, "wrong")
