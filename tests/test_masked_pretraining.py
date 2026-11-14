import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mi_localization import masked_pretraining as pilot
from mi_localization.neural import VCGResNet
from test_pretraining_source import tiny_config


def tiny_bank(tmp_path, monkeypatch):
    config, _ = tiny_config(tmp_path)
    config.update(device="cpu", torch_threads=1, widths=[4, 8, 12], pool_bins=2,
                  samples_per_epoch=8, batch_size=4, max_epochs=3, patience=3)
    bank = Path(config["bank_dir"])
    bank.mkdir()
    waves = np.random.default_rng(3).normal(size=(6, 3, 651)).astype(np.float32)
    waves[4:] += 0.3
    np.save(bank / "waveforms.npy", waves)
    np.save(bank / "groups.npy", [f"ptbxl:{index}" for index in range(6)])
    np.save(bank / "roles.npy", ["train"] * 4 + ["validation"] * 2)
    (bank / "manifest.json").write_text(json.dumps({"fingerprint": "tiny", "shape": [6, 3, 651]}))
    monkeypatch.setattr(pilot.pretraining_source, "prepare_bank", lambda config: bank)
    return config, waves


def test_masks_are_joint_xyz_contiguous_reproducible_and_do_not_consume_global_rng():
    state = torch.get_rng_state().clone()
    first = pilot.make_masks(8, 651, 31, 6, torch.Generator().manual_seed(7))
    assert torch.equal(state, torch.get_rng_state())
    second = pilot.make_masks(8, 651, 31, 6, torch.Generator().manual_seed(7))
    assert torch.equal(first, second)
    assert torch.equal(first[:, 0], first[:, 1]) and torch.equal(first[:, 1], first[:, 2])
    assert first.sum((1, 2)).tolist() == [6 * 31 * 3] * 8
    patches = first[:, 0].reshape(8, 21, 31)
    assert torch.equal(patches, patches[:, :, :1].expand_as(patches))
    with pytest.raises(ValueError, match="proper subset"):
        pilot.make_masks(1, 651, 31, 21, torch.Generator())


def test_mse_uses_only_masked_positions_and_has_finite_gradients():
    prediction = torch.tensor([[[1., 100., 3.]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[True, False, True]]])
    loss = pilot.masked_mse(prediction, target, mask)
    assert float(loss.detach()) == 5
    loss.backward()
    assert prediction.grad.tolist() == [[[1., 0., 3.]]]


def test_source_only_training_export_and_train_only_scaling(tmp_path, monkeypatch):
    config, waves = tiny_bank(tmp_path, monkeypatch)
    report = pilot.run_pilot(config)
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["complete"] and summary["source_training_patients"] == 4 and summary["source_validation_patients"] == 2
    assert summary["normalization_fit_patients"] == 4 and summary["ptb_classification_evaluated"] is False
    assert summary["ptb_patient_independence_cleared"] is False and summary["training_presentations"] == 24
    exported = torch.load(report.parent / "encoder_source_only.pt", map_location="cpu", weights_only=True)
    expected_mean = waves[:4].astype(np.float64).mean((0, 2)).astype(np.float32)
    np.testing.assert_allclose(exported["normalization"]["wave_mean_"], expected_mean, atol=1e-7)
    target = VCGResNet(n_classes=6, widths=(4, 8, 12), pool_bins=2)
    target.encoder.load_state_dict(exported["encoder_state_dict"], strict=True)
    assert "Source-pilot report ready: True" in pilot.status(config)
    assert (report.parent / "reconstruction_curve.png").exists()
    with pytest.raises(ValueError, match="settings/runtime changed"):
        pilot.run_pilot(dict(config, learning_rate=0.002))


def test_checkpoint_resume_reproduces_uninterrupted_cpu_training(tmp_path, monkeypatch):
    config, _ = tiny_bank(tmp_path, monkeypatch)
    full = dict(config, output_dir=str(tmp_path / "full"))
    pilot.run_pilot(full)
    resumed = dict(config, output_dir=str(tmp_path / "resume"))
    original = pilot._progress
    def interrupt(root, **fields):
        if fields.get("epoch") == 1:
            raise RuntimeError("intentional interruption after checkpoint")
        return original(root, **fields)
    monkeypatch.setattr(pilot, "_progress", interrupt)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        pilot.run_pilot(resumed)
    assert (Path(resumed["output_dir"]) / "checkpoint.pt").exists()
    monkeypatch.setattr(pilot, "_progress", original)
    pilot.run_pilot(resumed)
    first = torch.load(Path(full["output_dir"]) / "encoder_source_only.pt", weights_only=True)
    second = torch.load(Path(resumed["output_dir"]) / "encoder_source_only.pt", weights_only=True)
    for key, value in first["encoder_state_dict"].items():
        torch.testing.assert_close(value, second["encoder_state_dict"][key], rtol=0, atol=0)
    a = json.loads((Path(full["output_dir"]) / "summary.json").read_text())
    b = json.loads((Path(resumed["output_dir"]) / "summary.json").read_text())
    assert a["history"] == b["history"] and a["best_epoch"] == b["best_epoch"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_production_shape_masked_gpu_backward_checkpoint_and_export(tmp_path, monkeypatch):
    config, _ = tiny_bank(tmp_path, monkeypatch)
    config.update(device="mps", widths=[16, 32, 64], pool_bins=4, samples_per_epoch=128, batch_size=128, max_epochs=1, patience=1)
    report = pilot.run_pilot(config)
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["encoder_parameters"] == 77808
    assert np.isfinite(summary["best_masked_validation_mse"])
    assert np.isfinite(summary["trained_model_zero_input_masked_validation_mse"])
