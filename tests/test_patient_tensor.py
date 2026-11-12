import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mi_localization import patient_tensor as pt
from mi_localization.benchmark import create_fixed_folds
from mi_localization.neural import FoldNormalizer


def configuration():
    return {"seed": 42, "split_seed": 42, "training_seed": 42,
        "classes": ["AMI", "IMI", "HC"], "outer_splits": 3, "inner_splits": 2,
        "models": list(pt.MODELS), "device": "cpu", "torch_threads": 1,
        "widths": [4, 8, 12], "pool_bins": 2, "dropout": 0.1, "attention_hidden": 8,
        "bag_size": 4, "bags_per_epoch": 8, "batch_size": 2, "inference_batch_size": 4,
        "max_epochs": 2, "patience": 1, "learning_rate": 0.001, "weight_decay": 0.01,
        "scale_windows": [3, 9, 17], "tensor_time_bins": 16, "tensor_ranks": [3, 4, 3],
        "tensor_embedding": 8, "orthogonality_weight": 0.001}


def bank(tmp_path):
    config = configuration()
    destination = tmp_path / "bank"
    destination.mkdir()
    rng = np.random.default_rng(19)
    groups = np.repeat([f"p{i:02d}" for i in range(27)], 3)
    y = np.repeat(np.repeat(np.arange(3), 9), 3)
    waves = rng.normal(size=(81, 3, 65)).astype(np.float32)
    statistics = rng.normal(size=(81, 4)).astype(np.float32)
    for name, values in [("waveforms", waves), ("statistics", statistics), ("groups", groups),
                         ("y", np.array(config["classes"])[y])]:
        np.save(destination / f"{name}.npy", values)
    (destination / "manifest.json").write_text(json.dumps({"classes": config["classes"],
        "shape": [81, 3, 65], "n_beats": 81, "n_patients": 27, "fingerprint": "tiny-patient-bank"}))
    config.update(bank_dir=str(destination), output_dir=str(tmp_path / "results"),
                  folds_path=str(tmp_path / "folds.csv"), control_dir=str(tmp_path / "control"))
    create_fixed_folds(y, groups, config["classes"], Path(config["folds_path"]), 3, 42)
    return config, waves, statistics, y, groups


def test_multiscale_preserves_raw_xyz_and_does_not_mix_leads():
    torch.set_num_threads(1)
    views = pt.MultiscaleViews((3, 9, 17))
    x = torch.randn(2, 3, 65)
    before = views(x)
    assert before.shape == (2, 3, 65, 4)
    torch.testing.assert_close(before[..., 0], x)
    changed = x.clone()
    changed[:, 0] *= 10
    torch.testing.assert_close(views(changed)[:, 1:], before[:, 1:])
    with pytest.raises(ValueError, match="odd"):
        pt.MultiscaleViews((2, 4))


def test_tucker_mode_products_and_gradients_are_supervised_trainable():
    config = configuration()
    module = pt.TensorFeatures(config, tucker=True)
    x = torch.randn(2, 3, 65, requires_grad=True)
    views = module.time_pool(module.views(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    reference = torch.einsum("nlts,la,tb,sc->nabc", views, module.lead_factor, module.time_factor, module.scale_factor)
    torch.testing.assert_close(module.core(x), reference)
    assert reference.shape == (2, 3, 4, 3)
    module(x).square().mean().backward()
    for parameter in [module.views.kernels, module.lead_factor, module.time_factor, module.scale_factor]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert abs(float(module.penalty().detach())) < 1e-10


@pytest.mark.parametrize("name", pt.MODELS)
def test_patient_models_permutation_padding_invariance_and_finite_gradients(name):
    config = configuration()
    model = pt.PatientTensorModel(name, config).eval()
    x = torch.randn(2, 4, 3, 65)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    order = torch.tensor([3, 1, 0, 2])
    expected = model(x, mask)
    torch.testing.assert_close(model(x[:, order], mask[:, order]), expected)
    changed = x.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 100
    torch.testing.assert_close(model(changed, mask), expected)
    expected.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="valid beat"):
        model(x, torch.zeros_like(mask))


def test_patient_bags_never_cross_patients_and_masks_short_bags(tmp_path):
    config, waves, statistics, y, groups = bank(tmp_path)
    indices = np.arange(81)
    normalizer = FoldNormalizer().fit(waves, statistics, indices, np.ones(81))
    dataset = pt.PatientBags(waves, y, groups, indices, normalizer, 8, 42)
    x, mask, label = dataset[0]
    assert x.shape == (8, 3, 65) and mask.sum() == 3 and label == y[0]
    valid_values = x[mask].numpy()
    expected = normalizer.transform(waves[:3], statistics[:3])[0]
    assert all(any(np.allclose(row, candidate) for candidate in expected) for row in valid_values)


@pytest.mark.parametrize("name", ["bag_mean", "tucker_attention"])
def test_inference_uses_all_beats_and_is_encoder_batch_size_invariant(tmp_path, name, monkeypatch):
    config, waves, statistics, y, groups = bank(tmp_path)
    normalizer = FoldNormalizer().fit(waves, statistics, np.arange(81), np.ones(81))
    model = pt.PatientTensorModel(name, config)
    counts = []
    original_encode = model.encode

    def encode(x):
        counts.append(len(x))
        return original_encode(x)

    monkeypatch.setattr(model, "encode", encode)
    ids, truth, probabilities = pt.predict_patients(model, normalizer, waves, y, groups, np.arange(81), config)
    assert sum(counts) == 81
    other = pt.predict_patients(model, normalizer, waves, y, groups, np.arange(81), dict(config, inference_batch_size=1))
    np.testing.assert_allclose(probabilities, other[2], atol=1e-6)
    assert len(ids) == 27 and np.allclose(probabilities.sum(1), 1)
    x = normalizer.transform(waves[:3], statistics[:3])[0]
    with torch.no_grad():
        direct = model(torch.from_numpy(x[None]), torch.ones((1, 3), dtype=torch.bool)).softmax(1).numpy()[0]
    np.testing.assert_allclose(probabilities[0], direct, atol=1e-6)


def fake_fit(calls):
    def fit(name, waves, statistics, y, groups, train, valid, epochs, config, seed):
        calls.append((name, set(groups[train]), None if valid is None else set(groups[valid]), seed))
        normalizer = FoldNormalizer().fit(waves, statistics, train, np.ones(len(train)))
        model = pt.PatientTensorModel(name, config)
        return model, normalizer, {"seed": seed, "selected_epoch": 1 if valid is not None else epochs,
            "best_validation_macro_f1": 0.3 if valid is not None else None, "patient_overlap": 0,
            "validation_patients": 0 if valid is None else len(np.unique(groups[valid])),
            "parameter_count": sum(p.numel() for p in model.parameters()), "normalization": normalizer.audit()}
    return fit


def test_patient_nested_scopes_checkpoints_partial_and_runtime_guard(tmp_path, monkeypatch):
    config, *_ = bank(tmp_path)
    calls = []
    monkeypatch.setattr(pt, "fit_patient_model", fake_fit(calls))
    result = pt.run_patient_study(config, [1])
    assert result.complete.tolist() == [False] * 4
    assert result.n_patients.tolist() == [9] * 4
    frozen = pd.read_csv(config["folds_path"])
    test_ids = set(frozen.loc[frozen.fold == 1, "patient_id"])
    assert len(calls) == 12
    for _, train, valid, seed in calls:
        assert not (train & test_ids)
        assert valid is None or (not (valid & test_ids) and not (train & valid))
    calls.clear()
    pt.run_patient_study(config, [1])
    assert calls == []
    assert not (Path(config["output_dir"]) / "complete").exists()
    with pytest.raises(ValueError, match="settings/runtime changed"):
        pt.run_patient_study(dict(config, bag_size=8), [1])


@pytest.mark.parametrize("name", ["bag_mean", "tucker_attention"])
def test_tiny_real_patient_training_and_no_outer_test_stopping(tmp_path, name):
    config, waves, statistics, y, groups = bank(tmp_path)
    train, valid = np.arange(54), np.arange(54, 81)
    model, normalizer, audit = pt.fit_patient_model(name, waves, statistics, y, groups, train, None, 1, config, 42)
    assert audit["selected_epoch"] == 1 and audit["validation_patients"] == 0
    assert np.isfinite(audit["history"][0]["training_loss"])
    with pytest.raises(ValueError, match="overlap"):
        pt.fit_patient_model(name, waves, statistics, y, groups, train, train, 1, config, 42)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_tensor_attention_finite_forward_backward():
    config = configuration()
    config.update(device="mps", scale_windows=[9, 33, 129], tensor_time_bins=64, tensor_ranks=[3, 12, 3])
    model = pt.PatientTensorModel("tucker_attention", config).to("mps")
    value = model(torch.randn(2, 4, 3, 651, device="mps"), torch.ones(2, 4, dtype=torch.bool, device="mps"))
    loss = value.square().mean() + 0.001 * model.penalty()
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)


def test_complete_patient_report_preserves_cohort_and_rejects_false_epoch_budget(tmp_path, monkeypatch):
    from mi_localization.patient_tensor_report import summarize_patient_study
    config, *_ = bank(tmp_path)
    monkeypatch.setattr(pt, "fit_patient_model", fake_fit([]))
    result = pt.run_patient_study(config)
    assert result.complete.all() and result.n_patients.tolist() == [27] * 4
    output = Path(config["output_dir"]) / "complete"
    control = Path(config["control_dir"])
    control.mkdir()
    saved = json.loads((output / "metrics_bag_mean.json").read_text())
    saved["model"] = "resnet"  # synthetic reference fixture, not a real result
    (control / "metrics_resnet.json").write_text(json.dumps(saved))
    (control / "study_manifest.json").write_text((output / "study_manifest.json").read_text())
    (control / "patient_predictions_resnet.csv").write_text((output / "patient_predictions_bag_mean.csv").read_text())
    report = summarize_patient_study(config, draws=100)
    assert "27 patients, 81 beats, 3 classes" in report.read_text()
    uncertainty = json.loads((output / "paired_bootstrap.json").read_text())
    assert "bag_attention" in uncertainty["arms"]["tucker_attention"]["paired_comparisons"]
    assert (output / "aggregate_error_audit.json").exists()
    path = output / "metrics_tucker_attention.json"
    tampered = json.loads(path.read_text())
    tampered["folds"][0]["final_fit"]["validation_patients"] = 1
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="without test stopping"):
        summarize_patient_study(config, draws=100)
