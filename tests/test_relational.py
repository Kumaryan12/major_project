import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mi_localization import relational
from mi_localization.relational import (
    SharedRelationalProjection, pack_symmetric, recording_tensor,
    relational_matrices, run_relational_study, unpack_symmetric,
)


def test_symmetric_roundtrip_and_dimensions():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(5, 4, 24, 24))
    values = values + values.swapaxes(-1, -2)
    packed = pack_symmetric(values)
    assert packed.shape == (5, 1200)
    np.testing.assert_allclose(unpack_symmetric(packed, 24, 4), values)
    assert pack_symmetric(values[0]).shape == (1200,)


def test_relational_handles_flat_channels_and_keeps_amplitude_in_covariance():
    time = np.linspace(0, 2 * np.pi, 651)
    tensor = np.stack([np.sin(time), np.cos(time), np.ones_like(time)])[:, :, None]
    windows = [[-250, 401], [-80, 100]]
    corr, cov = relational_matrices(tensor, windows, 250, shrinkage=0)
    scaled_corr, scaled_cov = relational_matrices(tensor * 3, windows, 250, shrinkage=0)
    assert corr.shape == (2, 3, 3)
    assert np.isfinite(corr).all()
    np.testing.assert_allclose(corr, corr.swapaxes(1, 2))
    assert np.all(corr[:, 2] == 0)
    np.testing.assert_allclose(corr, scaled_corr, atol=1e-12)
    np.testing.assert_allclose(scaled_cov, cov * 9, atol=1e-12)
    assert np.linalg.eigvalsh(corr).min() >= -1e-10


@pytest.mark.parametrize("windows", [[[400, 402]], [[-251, 30]], [[0, 1]], []])
def test_bad_regions_rejected(windows):
    with pytest.raises(ValueError):
        relational_matrices(np.zeros((3, 651, 8)), windows, 250)


def test_recording_wavelets_keep_original_and_reject_short_high_levels():
    rng = np.random.default_rng(8)
    signal = rng.normal(size=(16384, 3))
    result = recording_tensor(signal, "bior6.8", 9, (3, 4, 5, 6, 7, 8, 9))
    assert result.shape == (3, 16384, 8)
    np.testing.assert_allclose(result[:, :, 0], signal.T, rtol=1e-6)
    assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="too short"):
        recording_tensor(signal[:651], "bior6.8", 9, (3, 4))


def _projection_inputs(n=8):
    rng = np.random.default_rng(19)
    wave = rng.normal(size=(n, 2, 3, 10))
    matrices = np.einsum("nrit,nrjt->nrij", wave, wave)
    return np.column_stack([pack_symmetric(matrices), np.arange(n)])


def test_projection_preserves_region_axis_and_statistics():
    X = _projection_inputs()
    projection = SharedRelationalProjection(rank=2, channels=3, regions=2, statistics=1)
    features = projection.fit_transform(X)
    assert features.shape == (8, 2 * 3 + 1)
    np.testing.assert_allclose(features[:, -1], X[:, -1])
    np.testing.assert_allclose(projection.basis_.T @ projection.basis_, np.eye(2), atol=1e-12)
    assert 0 <= projection.energy_fraction_ <= 1


def test_projection_test_values_do_not_change_training_basis_or_scaling():
    X = _projection_inputs()
    projection = SharedRelationalProjection(rank=2, channels=3, regions=2, statistics=1, covariance=True)
    projection.fit(X[:5], sample_weight=np.array([1, 1, 2, 2, 1]))
    basis, scales = projection.basis_.copy(), projection.channel_scale_.copy()
    projection.transform(X[5:] * 100000)
    np.testing.assert_array_equal(projection.basis_, basis)
    np.testing.assert_array_equal(projection.channel_scale_, scales)
    assert projection.n_fit_samples_ == 5


def test_projection_zero_weight_rows_do_not_affect_basis():
    X = _projection_inputs()
    left = SharedRelationalProjection(rank=2, channels=3, regions=2, statistics=1, covariance=True)
    right = SharedRelationalProjection(rank=2, channels=3, regions=2, statistics=1, covariance=True)
    left.fit(X[:5])
    right.fit(np.vstack([X[:5], X[5:] * 10000]), sample_weight=np.array([1] * 5 + [0] * 3))
    np.testing.assert_allclose(left.basis_, right.basis_, atol=1e-10)
    np.testing.assert_allclose(left.channel_scale_, right.channel_scale_)


def _tiny_bank(tmp_path: Path):
    bank = tmp_path / "bank"
    bank.mkdir()
    ids = np.repeat([f"p{i:02d}" for i in range(27)], 3)
    labels = np.repeat(np.repeat(["AMI", "IMI", "HC"], 9), 3)
    values = _projection_inputs(len(ids))
    values[:, -1] = np.repeat(np.arange(27), 3)
    np.save(bank / "groups.npy", ids)
    np.save(bank / "y.npy", labels)
    np.save(bank / "statistics_context.npy", values[:, -1:])
    np.save(bank / "statistics_short.npy", values[:, -1:])
    np.save(bank / "correlation.npy", values[:, :-1])
    np.save(bank / "covariance.npy", values[:, :-1])
    (bank / "manifest.json").write_text(json.dumps({"fingerprint": "test-bank", "channels": 3,
                                                   "regions": 2, "statistics": 1}))
    return {"seed": 42, "bank_dir": str(bank), "folds_path": str(tmp_path / "folds.csv"),
            "output_dir": str(tmp_path / "results"), "classes": ["AMI", "IMI", "HC"],
            "outer_splits": 3, "inner_splits": 2, "ranks": [1, 2], "model_n_jobs": 1,
            "model_candidates": [{"n_estimators": 3, "max_depth": 1}]}


def test_nested_fit_scopes_and_partial_results_are_explicit(tmp_path, monkeypatch):
    config = _tiny_bank(tmp_path)
    calls = []
    original_fit = SharedRelationalProjection.fit

    def observed_fit(self, X, y=None, sample_weight=None):
        calls.append(set(X[:, -1].astype(int)))
        return original_fit(self, X, y, sample_weight)

    monkeypatch.setattr(SharedRelationalProjection, "fit", observed_fit)
    result = run_relational_study(config, ["correlation_shared"], [1])
    assert result.complete.tolist() == [False]
    folds = pd.read_csv(config["folds_path"])
    test_ids = {int(pid[1:]) for pid in folds.loc[folds.fold == 1, "patient_id"]}
    assert len(calls) == 5  # 2 ranks x 2 inner fits + one final outer fit
    assert all(not (fit_ids & test_ids) for fit_ids in calls)
    assert any(len(fit_ids) < len(calls[-1]) for fit_ids in calls[:-1])
    output = Path(config["output_dir"]) / "partial_folds_1"
    metrics = json.loads((output / "metrics_correlation_shared.json").read_text())
    assert metrics["complete"] is False
    assert metrics["n_patients"] == 9
    assert metrics["folds"][0]["patient_overlap"] == 0
    assert not (Path(config["output_dir"]) / "complete").exists()
    calls.clear()
    run_relational_study(config, ["correlation_shared"], [1])
    assert not calls  # resumes without refitting
    config["ranks"] = [1]
    with pytest.raises(ValueError, match="settings changed"):
        run_relational_study(config, ["correlation_shared"], [1])


def test_complete_statistics_study_preserves_entire_patient_cohort(tmp_path):
    config = _tiny_bank(tmp_path)
    result = run_relational_study(config, ["statistics_context"])
    assert result.complete.tolist() == [True]
    assert result.n_patients.tolist() == [27]
    saved = pd.read_csv(Path(config["output_dir"]) / "complete/patient_predictions_statistics_context.csv")
    assert saved.patient_id.nunique() == 27
    assert np.allclose(saved.filter(like="probability_").sum(axis=1), 1)


def test_bank_reuses_manifest_and_rejects_changed_sources(tmp_path, monkeypatch):
    config = _tiny_bank(tmp_path)
    # A complete mock cache; source fingerprint is independently checked.
    bank = Path(config["bank_dir"])
    for name in ["paper.yaml", "beats.csv"]:
        path = tmp_path / name
        if name.endswith("yaml"):
            path.write_text("{}")
        else:
            pd.DataFrame({"record": ["r"] * 81, "patient_id": np.load(bank / "groups.npy"),
                          "label": np.load(bank / "y.npy")}).to_csv(path, index=False)
    config.update(paper_config=str(tmp_path / "paper.yaml"), beats_path=str(tmp_path / "beats.csv"))
    monkeypatch.setattr(relational, "_fingerprint", lambda *args: "test-bank")
    assert relational.build_relational_bank(config) == bank
    monkeypatch.setattr(relational, "_fingerprint", lambda *args: "changed")
    with pytest.raises(ValueError, match="sources/settings changed"):
        relational.build_relational_bank(config)
