import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mi_localization import neural
from mi_localization.benchmark import create_fixed_folds
from mi_localization.neural import FoldNormalizer, VCGResNet, run_neural_study


def tiny_config(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    rng = np.random.default_rng(17)
    groups = np.repeat([f"p{i:02d}" for i in range(27)], 2)
    labels = np.repeat(np.repeat(["AMI", "IMI", "HC"], 9), 2)
    waves = rng.normal(size=(54, 3, 65)).astype(np.float32)
    stats = rng.normal(size=(54, 4)).astype(np.float32)
    stats[:, -1] = np.repeat(np.arange(27), 2)
    for name, value in [("waveforms", waves), ("statistics", stats), ("y", labels), ("groups", groups)]:
        np.save(bank / f"{name}.npy", value)
    (bank / "manifest.json").write_text(json.dumps({"fingerprint": "tiny-bank", "shape": [54, 3, 65],
        "n_beats": 54, "n_patients": 27, "statistics": 4, "classes": ["AMI", "IMI", "HC"]}))
    config = {"seed": 42, "bank_dir": str(bank), "output_dir": str(tmp_path / "results"),
        "folds_path": str(tmp_path / "folds.csv"), "classes": ["AMI", "IMI", "HC"],
        "models": ["resnet", "resnet_statistics"], "outer_splits": 3, "inner_splits": 2,
        "device": "cpu", "torch_threads": 1, "widths": [4, 8, 12], "pool_bins": 2,
        "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.01,
        "max_epochs": 2, "patience": 1, "batch_size": 8, "samples_per_epoch": 16}
    mapping = {label: i for i, label in enumerate(config["classes"])}
    y = np.array([mapping[label] for label in labels])
    create_fixed_folds(y, groups, config["classes"], Path(config["folds_path"]), 3, 42)
    return config, waves, stats, y, groups


def test_resnet_shapes_batch_independence_and_plain_model_ignores_statistics():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    x, s = torch.randn(3, 3, 651), torch.randn(3, 192)
    plain, fusion = VCGResNet().eval(), VCGResNet(statistics=192).eval()
    with torch.no_grad():
        assert plain(x, s).shape == fusion(x, s).shape == (3, 6)
        torch.testing.assert_close(plain(x, s), plain(x, s * 1e6))
        torch.testing.assert_close(plain(x[:1]), plain(x)[:1], atol=1e-6, rtol=1e-5)
    assert sum(p.numel() for p in plain.parameters()) < 150000
    with pytest.raises(ValueError, match="requires statistics"):
        fusion(x)
    with pytest.raises(ValueError, match="divisible"):
        VCGResNet(widths=(3, 8, 16))


@pytest.mark.parametrize("length,bins", [(41, 4), (65, 4), (64, 4), (3, 4)])
def test_temporal_pool_matches_adaptive_pool_outputs_and_gradients(length, bins):
    x = torch.randn(2, 3, length, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.adaptive_avg_pool1d(x, bins)
    actual = neural.TemporalBinPool(bins)(x)
    torch.testing.assert_close(actual, expected)
    expected_grad, = torch.autograd.grad(expected.sum(), x, retain_graph=True)
    actual_grad, = torch.autograd.grad(actual.sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_resnet_forward_backward_finite():
    model = VCGResNet().to("mps")
    loss = torch.nn.functional.cross_entropy(model(torch.randn(3, 3, 651, device="mps")),
                                             torch.tensor([0, 1, 2], device="mps"))
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)


def test_normalizer_fits_training_only_preserves_amplitude_and_is_weighted():
    waves = np.stack([np.ones((3, 9)) * value for value in [1, 3, 100000]]).astype(np.float32)
    stats = np.array([[1, 2], [3, 6], [1e8, 1e8]], dtype=np.float32)
    normalizer = FoldNormalizer().fit(waves, stats, [0, 1], [1, 3])
    np.testing.assert_allclose(normalizer.wave_mean_, 2.5)
    np.testing.assert_allclose(normalizer.wave_scale_, np.sqrt(0.75))
    before = normalizer.audit()
    x, s = normalizer.transform(waves[2], stats[2])
    assert np.isfinite(x).all() and np.isfinite(s).all()
    assert np.max(abs(s)) <= 10
    assert normalizer.audit() == before
    scaled, _ = normalizer.transform(waves[1] * 2, stats[1])
    original, _ = normalizer.transform(waves[1], stats[1])
    assert not np.allclose(scaled, original)  # not per-beat unit-amplitude normalization
    waves[2] *= 1e6
    other = FoldNormalizer().fit(waves, stats, [0, 1], [1, 3])
    assert other.audit() == before
    with pytest.raises(ValueError, match="weights"):
        FoldNormalizer().fit(waves, stats, [0, 1], [1, -1])


def test_real_tiny_training_reproducible_finite_and_no_test_monitoring(tmp_path):
    config, waves, stats, y, groups = tiny_config(tmp_path)
    train = np.arange(0, 54, 2)
    config["dropout"] = 0
    first, norm1, audit1 = neural._fit_network("resnet", waves, stats, y, groups, train, None, 2, config, 42)
    second, norm2, audit2 = neural._fit_network("resnet", waves, stats, y, groups, train, None, 2, config, 42)
    np.testing.assert_array_equal(neural._predict_network(first, norm1, waves, stats, y, train, config),
                                  neural._predict_network(second, norm2, waves, stats, y, train, config))
    assert audit1 == audit2
    assert audit1["selected_epoch"] == 2 and audit1["validation_patients"] == 0
    assert all("validation_patient_macro_f1" not in row for row in audit1["history"])
    with pytest.raises(ValueError, match="overlap"):
        neural._fit_network("resnet", waves, stats, y, groups, train, train, 1, config, 42)


def fake_fit(calls):
    def fit(name, waves, statistics, y, groups, train, valid, epochs, config, seed):
        calls.append((name, set(groups[train]), None if valid is None else set(groups[valid])))
        norm = FoldNormalizer().fit(waves, statistics, train, np.ones(len(train)))
        model = VCGResNet(n_classes=3, widths=(4, 8, 12), pool_bins=2,
                          statistics=statistics.shape[1] if name == "resnet_statistics" else 0)
        return model, norm, {"seed": seed, "selected_epoch": 1 if valid is not None else epochs,
            "best_validation_macro_f1": 0.25 if valid is not None else None,
            "normalizer_fit_beats": len(train), "parameter_count": 100, "normalization": norm.audit()}
    return fit


def test_nested_neural_scopes_partial_results_resume_and_scope_guard(tmp_path, monkeypatch):
    config, _, _, _, _ = tiny_config(tmp_path)
    calls = []
    monkeypatch.setattr(neural, "_fit_network", fake_fit(calls))
    result = run_neural_study(config, [1])
    assert result.complete.tolist() == [False, False]
    assert result.n_patients.tolist() == [9, 9]
    frozen = pd.read_csv(config["folds_path"])
    test = set(frozen.loc[frozen.fold == 1, "patient_id"])
    training = set(frozen.loc[frozen.fold != 1, "patient_id"])
    assert len(calls) == 6  # two models, each two inner fits + one outer refit
    for _, train, valid in calls:
        assert not (train & test)
        if valid is not None:
            assert not (valid & test) and not (train & valid)
            assert train | valid == training
        else:
            assert train == training
    output = Path(config["output_dir"]) / "partial_folds_1"
    saved = json.loads((output / "metrics_resnet.json").read_text())
    assert saved["complete"] is False
    assert saved["folds"][0]["final_epochs"] == 1
    assert not (output.parent / "complete").exists()
    calls.clear()
    run_neural_study(config, [1])
    assert calls == []
    with pytest.raises(ValueError, match="settings/runtime changed"):
        run_neural_study(dict(config, max_epochs=3), [1])
    with pytest.raises(ValueError, match="fold IDs"):
        run_neural_study(config, [4])


def test_complete_neural_cohort_preserved_and_frozen_folds_required(tmp_path, monkeypatch):
    from mi_localization.neural_report import summarize_neural_study
    config, _, _, _, _ = tiny_config(tmp_path)
    config["control_dir"] = str(tmp_path / "control")
    monkeypatch.setattr(neural, "_fit_network", fake_fit([]))
    result = run_neural_study(config)
    assert result.complete.all() and result.n_patients.tolist() == [27, 27]
    saved = pd.read_csv(Path(config["output_dir"]) / "complete/patient_predictions_resnet.csv")
    assert saved.patient_id.nunique() == 27
    assert np.allclose(saved.filter(like="probability_").sum(axis=1), 1)
    control = Path(config["control_dir"])
    control.mkdir()
    output = Path(config["output_dir"]) / "complete"
    (control / "metrics_statistics_short.json").write_text((output / "metrics_resnet.json").read_text())
    (control / "patient_predictions_statistics_short.csv").write_text((output / "patient_predictions_resnet.csv").read_text())
    report = summarize_neural_study(config, draws=100)
    assert "27 patients, 54 beats, 3 classes" in report.read_text()
    assert "not untouched confirmation" in report.read_text()
    assert (output / "paired_bootstrap.json").exists()
    tampered = json.loads((output / "metrics_resnet_statistics.json").read_text())
    tampered["folds"][0]["final_epochs"] += 1
    (output / "metrics_resnet_statistics.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="inner-only selection"):
        summarize_neural_study(config, draws=100)
    with pytest.raises(ValueError, match="existing frozen"):
        run_neural_study(dict(config, folds_path=str(tmp_path / "absent.csv")))


def test_waveform_cache_matches_existing_r_positions(tmp_path, monkeypatch):
    config, waves, stats, y, groups = tiny_config(tmp_path)
    relational = Path(config["bank_dir"])
    (relational / "statistics.npy").rename(relational / "statistics_short.npy")
    frame = pd.DataFrame({"patient_id": groups, "record": "r", "label": np.load(relational / "y.npy"),
                          "r_peak": np.arange(54) + 2})
    frame.to_csv(tmp_path / "beats.csv", index=False)
    raw = tmp_path / "raw"
    raw.mkdir()
    for extension in ["hea", "xyz"]:
        (raw / f"r.{extension}").write_text("test")
    paper = {"data": {"raw_dir": str(raw), "leads": ["vx", "vy", "vz"], "sampling_rate": 1000,
                      "before_r": 1, "after_r": 2}, "signal": {"denoise_wavelet": "bior6.8", "denoise_level": 1}}
    import yaml
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper))
    config.update(paper_config=str(tmp_path / "paper.yaml"), beats_path=str(tmp_path / "beats.csv"),
                  statistics_bank_dir=str(relational), bank_dir=str(tmp_path / "new_bank"), feature_n_jobs=1)
    raw_waves = np.arange(180, dtype=float).reshape(60, 3)
    monkeypatch.setattr(neural, "read_vcg", lambda *args: (raw_waves, 1000))
    monkeypatch.setattr(neural, "denoise_vcg", lambda value, *args: value)
    bank = neural.build_waveform_bank(config)
    cached = np.load(bank / "waveforms.npy")
    assert cached.shape == (54, 3, 4)
    np.testing.assert_array_equal(cached[0], raw_waves[1:5].T)
    assert neural.build_waveform_bank(config) == bank
    (raw / "r.xyz").write_text("changed-source")
    with pytest.raises(ValueError, match="sources/settings changed"):
        neural.build_waveform_bank(config)


def test_training_seeds_do_not_change_inner_patient_partitions(tmp_path, monkeypatch):
    config, *_ = tiny_config(tmp_path)
    observations = []
    for seed in [42, 123]:
        calls, seeds = [], []
        implementation = fake_fit(calls)

        def fit(*args):
            seeds.append(args[-1])
            return implementation(*args)

        monkeypatch.setattr(neural, "_fit_network", fit)
        run_neural_study(dict(config, models=["resnet"], split_seed=42, training_seed=seed,
                              output_dir=str(tmp_path / f"seed_{seed}")), [1])
        observations.append((calls, seeds))
    assert observations[0][0] == observations[1][0]
    assert np.array_equal(np.array(observations[1][1]) - observations[0][1], [81] * 3)


def test_repeat_study_reuses_reference_reports_all_seeds_and_guards_scope(tmp_path, monkeypatch):
    import yaml
    from mi_localization.neural_repeat import run_repeats
    config, *_ = tiny_config(tmp_path)
    monkeypatch.setattr(neural, "_fit_network", fake_fit([]))
    run_neural_study(config)
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(config))
    repeated = {"base_config": str(path), "output_dir": str(tmp_path / "repeats"),
                "reference_dir": str(Path(config["output_dir"]) / "complete"),
                "reference_seed": 42, "training_seeds": [42, 123, 2026], "split_seed": 42}
    result = run_repeats(repeated)
    assert result.training_seed.tolist() == [42, 123, 2026]
    summary = json.loads((Path(repeated["output_dir"]) / "summary.json").read_text())
    assert summary["complete"] is True
    assert np.isclose(summary["macro_f1_mean"], result.patient_macro_f1.mean())
    assert np.isclose(summary["macro_f1_sample_std"], result.patient_macro_f1.std(ddof=1))
    assert not (Path(repeated["output_dir"]) / "seed_42").exists()
    with pytest.raises(ValueError, match="Reference split/training seed"):
        run_repeats(dict(repeated, split_seed=17))
