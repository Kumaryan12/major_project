import numpy as np

from mi_localization.domain_shift import _paired_macro_f1_intervals, _peak_recall


def test_peak_recall_matches_nearest_derived_peak():
    reference = np.array([100, 300, 500])
    candidate = np.array([105, 390, 700])
    assert _peak_recall(reference, candidate, 10) == 1 / 3
    assert _peak_recall(reference, np.array([], dtype=int), 10) == 0.0
    assert np.isnan(_peak_recall(np.array([], dtype=int), candidate, 10))


def test_paired_macro_f1_bootstrap_uses_same_patients():
    truth = np.array([0, 0, 1, 1])
    perfect = np.eye(2)[truth]
    wrong = np.eye(2)[1 - truth]
    scores = {
        "measured_to_measured": perfect,
        "measured_to_derived": wrong,
        "derived_to_derived": perfect,
    }
    result = _paired_macro_f1_intervals(truth, scores, 2, seed=42, draws=10)
    first = result["measured_to_derived_minus_measured_to_measured"]
    second = result["derived_to_derived_minus_measured_to_derived"]
    assert first["observed_delta"] == -1.0
    assert first["bootstrap_95_percentile_interval"] == [-1.0, -1.0]
    assert second["observed_delta"] == 1.0
