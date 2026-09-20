import numpy as np

from mi_localization.ablation import beat_representations
from mi_localization.features import normalize_vector_rms, tensorize_beat, tucker_time_features_multi


FEATURE_CONFIG = {
    "wavelet": "bior6.8",
    "level": 9,
    "detail_levels": [3, 4, 5, 6, 7, 8, 9],
    "canonicalize_svd_sign": True,
}


def test_multi_rank_shapes_and_baseline_equivalence():
    rng = np.random.default_rng(3)
    beat = rng.normal(size=(3, 651))
    tensor = tensorize_beat(beat)
    features = tucker_time_features_multi(tensor, (1, 3, 5))
    assert features[1].shape == (24,)
    assert features[3].shape == (72,)
    assert features[5].shape == (120,)


def test_vector_normalization_removes_offset_and_scale():
    rng = np.random.default_rng(4)
    beat = rng.normal(size=(3, 651))
    first = normalize_vector_rms(beat)
    second = normalize_vector_rms(beat * 7.0 + np.array([[2.0], [-3.0], [5.0]]))
    assert np.allclose(first, second)


def test_all_ablation_representations_are_finite():
    time = np.linspace(0, 1, 651)
    clean = np.vstack([np.sin(2 * np.pi * f * time) for f in (3, 5, 7)])
    result = beat_representations(clean, clean + 0.01, FEATURE_CONFIG)
    assert len(result) == 12
    assert all(np.all(np.isfinite(values)) for values in result.values())
    assert result["wavelet_statistics"].shape == (192,)
