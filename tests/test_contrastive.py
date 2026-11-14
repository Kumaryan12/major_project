import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mi_localization import contrastive as cp
from mi_localization.benchmark import create_fixed_folds
from mi_localization.neural import FoldNormalizer, VCGResNet, _seed


def tiny_bank(tmp_path):
    root = tmp_path / "bank"
    root.mkdir()
    rng = np.random.default_rng(19)
    groups = np.repeat([f"p{i:02d}" for i in range(27)], 3)
    y = np.repeat(np.repeat(np.arange(3), 9), 3)
    classes = ["AMI", "IMI", "HC"]
    waves = rng.normal(size=(81, 3, 65)).astype(np.float32)
    statistics = rng.normal(size=(81, 4)).astype(np.float32)
    for name, values in [("waveforms", waves), ("statistics", statistics), ("groups", groups), ("y", np.array(classes)[y])]:
        np.save(root / f"{name}.npy", values)
    (root / "manifest.json").write_text(json.dumps({"classes": classes, "shape": [81, 3, 65],
        "statistics": 4, "n_patients": 27, "n_beats": 81, "fingerprint": "tiny-contrastive"}))
    config = {"seed": 42, "split_seed": 42, "training_seeds": [42, 123, 2026],
        "bank_dir": str(root), "folds_path": str(tmp_path / "folds.csv"), "output_dir": str(tmp_path / "results"),
        "classes": classes, "outer_splits": 3, "inner_splits": 2, "arms": list(cp.ARMS),
        "device": "cpu", "torch_threads": 1, "widths": [4, 8, 12], "pool_bins": 2,
        "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.01,
        "patients_per_class": 2, "beats_per_patient": 2, "batches_per_epoch": 2,
        "batch_size": 12, "max_epochs": 2, "patience": 1, "temperature": 0.1, "contrastive_weight": 0.1}
    create_fixed_folds(y, groups, classes, Path(config["folds_path"]), 3, 42)
    return config, waves, statistics, y, groups


def test_masks_exclude_same_patient_from_positives_and_denominator():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    patients = torch.tensor([0, 0, 1, 2, 2, 3])
    allowed, positives = cp.cross_patient_masks(labels, patients)
    assert not allowed[0, 0] and not allowed[0, 1] and not positives[0, 1]
    assert allowed[0, 2] and positives[0, 2]
    assert allowed[0, 3] and not positives[0, 3]
    assert not allowed.diagonal().any()


def test_loss_matches_explicit_log_probability_and_has_finite_gradients():
    features = torch.tensor([[1., 0], [0.9, 0.1], [-1, 0], [-0.9, -0.1]], dtype=torch.float64, requires_grad=True)
    labels, patients = torch.tensor([0, 0, 1, 1]), torch.arange(4)
    actual = cp.cross_patient_loss(features, labels, patients, 0.3)
    vectors = features.detach().numpy().copy()
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    logits = vectors @ vectors.T / 0.3
    losses = []
    for i in range(4):
        denominator = math.log(sum(math.exp(logits[i, j]) for j in range(4) if i != j))
        positives = [j for j in range(4) if i != j and labels[i] == labels[j]]
        losses.append(-np.mean([logits[i, j] - denominator for j in positives]))
    assert float(actual.detach()) == pytest.approx(np.mean(losses))
    actual.backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("same_patient", [True, False])
def test_loss_no_cross_patient_positives_is_differentiable_zero(same_patient):
    x = torch.randn(4, 8, requires_grad=True)
    labels = torch.zeros(4, dtype=torch.long) if same_patient else torch.arange(4)
    patients = torch.zeros(4, dtype=torch.long) if same_patient else torch.arange(4)
    loss = cp.cross_patient_loss(x, labels, patients)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(x.grad).all() and (x.grad == 0).all()


def test_loss_zero_features_and_partial_eligible_anchors_are_finite():
    x = torch.zeros(4, 8, requires_grad=True)
    loss = cp.cross_patient_loss(x, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 2]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match="temperature"):
        cp.cross_patient_loss(x, torch.arange(4), torch.arange(4), 0)


def test_sampler_structured_balanced_train_only_reproducible_across_epochs(tmp_path):
    config, _, _, y, groups = tiny_bank(tmp_path)
    indices = np.concatenate([np.arange(18), np.arange(27, 45), np.arange(54, 72)])
    first = cp.StructuredPatientBatches(y, groups, indices, 3, 2, 2, 3, 42)
    second = cp.StructuredPatientBatches(y, groups, indices, 3, 2, 2, 3, 42)
    epochs = [list(first), list(first)]
    assert epochs == [list(second), list(second)]
    assert epochs[0] != epochs[1]
    for epoch in epochs:
        for positions in epoch:
            rows = indices[positions]
            assert len(rows) == 12 and set(rows) <= set(indices)
            for label in range(3):
                label_rows = rows[y[rows] == label]
                assert len(label_rows) == 4
                _, counts = np.unique(groups[label_rows], return_counts=True)
                assert counts.tolist() == [2, 2]
    with pytest.raises(ValueError, match="fewer"):
        cp.StructuredPatientBatches(y, groups, np.array([0, 27, 54]), 3, 2, 2, 1, 42)


def test_real_matched_training_initialization_batches_and_zero_weight_control(tmp_path):
    config, waves, statistics, y, groups = tiny_bank(tmp_path)
    train = np.arange(81)
    config["dropout"] = 0
    ce_model, ce_norm, ce = cp.fit_contrastive("matched_ce", waves, statistics, y, groups, train, None, 2, config, 42)
    cp_model, cp_norm, supcon = cp.fit_contrastive("cross_patient_supcon", waves, statistics, y, groups, train, None, 2, config, 42)
    assert ce["initial_state_sha256"] == supcon["initial_state_sha256"]
    assert ce["parameter_count"] == supcon["parameter_count"]
    assert [row["batch_indices_sha256"] for row in ce["history"]] == [row["batch_indices_sha256"] for row in supcon["history"]]
    assert ce["normalization"] == supcon["normalization"]
    assert all(row["positive_anchor_fraction"] == 1 and row["beat_presentations"] == 24 for row in supcon["history"])
    assert all(row["training_loss"] == pytest.approx(row["cross_entropy"]) for row in ce["history"])
    assert all(row["training_loss"] == pytest.approx(row["cross_entropy"] + 0.1 * row["cross_patient_supcon"], abs=1e-6) for row in supcon["history"])
    assert not torch.equal(ce_model.classifier[-1].weight, cp_model.classifier[-1].weight)
    zero_model, _, zero = cp.fit_contrastive("cross_patient_supcon", waves, statistics, y, groups, train, None, 2, dict(config, contrastive_weight=0), 42)
    for key, value in ce_model.state_dict().items():
        torch.testing.assert_close(value, zero_model.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="overlap"):
        cp.fit_contrastive("matched_ce", waves, statistics, y, groups, train, train, 1, config, 42)


def fake_fit(calls):
    def fit(arm, waves, statistics, y, groups, train, valid, epochs, config, seed, progress=None):
        calls.append((arm, set(groups[train]), None if valid is None else set(groups[valid]), seed))
        _seed(seed, config)
        normalizer = FoldNormalizer().fit(waves, statistics, train, np.ones(len(train)))
        model = VCGResNet(n_classes=3, widths=(4, 8, 12), pool_bins=2, dropout=config["dropout"])
        weight = 0 if arm == "matched_ce" else config["contrastive_weight"]
        history = [{"epoch": epoch, "training_loss": 0.7 + weight, "cross_entropy": 0.7,
            "cross_patient_supcon": 1, "batch_indices_sha256": cp._hash([seed, train.tolist(), epoch]),
            "batches": config["batches_per_epoch"], "beat_presentations": config["batch_size"] * config["batches_per_epoch"],
            "positive_anchor_fraction": 1} for epoch in range(1, epochs + 1)]
        if progress:
            progress(history[-1])
        return model, normalizer, {"seed": seed, "selected_epoch": 1 if valid is not None else epochs,
            "best_validation_macro_f1": 0.3 if valid is not None else None, "history": history,
            "initial_state_sha256": cp.initial_state_hash(model), "contrastive_weight": weight,
            "patient_overlap": 0, "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_patients": len(np.unique(groups[train])),
            "validation_patients": 0 if valid is None else len(np.unique(groups[valid])),
            "normalizer_fit_beats": len(train), "normalization": normalizer.audit()}
    return fit


def test_nested_scope_seed_split_matching_resume_partial_and_status(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    calls = []
    monkeypatch.setattr(cp, "fit_contrastive", fake_fit(calls))
    result = cp.run_study(config, [1])
    assert result.complete.tolist() == [False] * 6
    frozen = pd.read_csv(config["folds_path"])
    test_ids = set(frozen.loc[frozen.fold == 1, "patient_id"])
    assert len(calls) == 18
    for _, train, valid, _ in calls:
        assert not train & test_ids
        if valid is not None:
            assert not valid & test_ids and not train & valid
    # Both arm and seed changes preserve the fixed inner patient memberships.
    memberships = [(train, valid) for _, train, valid, _ in calls]
    assert all(memberships[:3] == memberships[offset:offset + 3] for offset in range(0, 18, 3))
    calls.clear()
    cp.run_study(config, [1])
    assert calls == []
    assert "partial_complete" in cp.status(config, [1])
    assert "Final report ready: False" in cp.status(config, [1])
    assert not (Path(config["output_dir"]) / "complete").exists()
    with pytest.raises(ValueError, match="settings/runtime changed"):
        cp.run_study(dict(config, temperature=0.2), [1])


def test_inner_patient_shortage_fails_before_any_fit(tmp_path, monkeypatch):
    config, *_ = tiny_bank(tmp_path)
    calls = []
    monkeypatch.setattr(cp, "fit_contrastive", fake_fit(calls))
    with pytest.raises(ValueError, match="fewer"):
        cp.run_study(dict(config, patients_per_class=4, batch_size=24), [1])
    assert calls == []


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_cross_patient_gradient_finite():
    x = torch.randn(12, 64, device="mps", requires_grad=True)
    labels = torch.arange(3, device="mps").repeat_interleave(4)
    patients = torch.arange(6, device="mps").repeat_interleave(2)
    loss = cp.cross_patient_loss(x, labels, patients)
    loss.backward()
    assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()


def test_paired_seed_bootstrap_keeps_patient_count_and_identical_arms_zero():
    from mi_localization.contrastive_report import paired_seed_bootstrap
    truth = np.repeat(np.arange(3), 9)
    pred = truth.copy()
    pred[0] = 1
    values = {seed: {arm: pred.copy() for arm in cp.ARMS} for seed in [42, 123, 2026]}
    result = paired_seed_bootstrap(truth, values, 3, draws=100)
    assert result["unique_patients"] == 27  # not 81 independent patient/seed repetitions
    assert result["mean_paired_macro_f1_difference_95_interval"] == [0, 0]
    assert result == paired_seed_bootstrap(truth, values, 3, draws=100)
    with pytest.raises(ValueError, match="Every class"):
        paired_seed_bootstrap(truth, values, 4, draws=100)


def test_complete_matched_report_checks_draws_and_uses_every_seed(tmp_path, monkeypatch):
    from mi_localization.contrastive_report import summarize_study
    config, *_ = tiny_bank(tmp_path)
    monkeypatch.setattr(cp, "fit_contrastive", fake_fit([]))
    result = cp.run_study(config)
    assert result.complete.all() and len(result) == 6
    report = summarize_study(config, draws=100)
    root = report.parent
    assert "27 unique patients, 81 beats, 3 unchanged classes" in report.read_text()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] is True and summary["training_seeds"] == [42, 123, 2026]
    assert len(summary["per_seed"]) == 3
    assert (root / "paired_bootstrap.json").exists()
    path = root / "seed_42/metrics_cross_patient_supcon.json"
    tampered = json.loads(path.read_text())
    tampered["folds"][0]["inner_selection"][0]["history"][0]["batch_indices_sha256"] = "changed"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="identical batch draws"):
        summarize_study(config, draws=100)
