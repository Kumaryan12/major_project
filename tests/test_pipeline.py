import numpy as np

from mi_localization.features import extract_features, tensorize_beat
from mi_localization.labels import location_to_class
from mi_localization.signal import extract_beats


def test_label_variants():
    assert location_to_class("antero-septo-lateral") == "ASLMI"
    assert location_to_class("infero-posterior") == "IPMI"
    assert location_to_class("infero-poster-lateral") == "IPLMI"
    assert location_to_class("infero-latera") == "ILMI"
    assert location_to_class("postero-lateral") == "PLMI"


def test_beat_window_is_651_samples():
    signal = np.zeros((1000, 3))
    beats, peaks = extract_beats(signal, np.array([100, 300, 800]))
    assert beats.shape == (1, 3, 651)
    assert peaks.tolist() == [300]


def test_tensor_and_feature_shapes_are_paper_shapes():
    time = np.linspace(0, 1, 651, endpoint=False)
    beat = np.vstack([np.sin(2 * np.pi * frequency * time) for frequency in (5, 7, 11)])
    tensor = tensorize_beat(beat)
    feature = extract_features(beat)
    assert tensor.shape == (3, 651, 8)
    assert feature.shape == (72,)
    assert np.all(np.isfinite(feature))
