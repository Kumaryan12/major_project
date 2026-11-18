import numpy as np
import pytest
from scipy.stats import spearmanr

from mi_localization.paired_recording_quality import (
    association, raw_peak_disagreement, sampled_means,
)


def test_peak_match_checks_nearest_detection_and_accepts_exact_tolerance():
    cached = np.array([40, 60, 160, 241, 360, 441])
    detected = np.array([100, 200, 400])
    np.testing.assert_array_equal(
        raw_peak_disagreement(cached, detected, 40),
        [True, False, False, True, False, True],
    )
    assert raw_peak_disagreement(cached, np.array([], dtype=int), 40).all()


def test_sample_means_respect_global_beat_rows_and_missing_preceding_rr():
    rows = np.array([12, 13, 15])
    features = np.array([[3., np.nan], [1., 1.], [5., 0.]])
    samples = np.array([[12, 13], [12, 15]])
    np.testing.assert_allclose(sampled_means(features, samples, rows), [3., .5])
    with pytest.raises(ValueError, match='outside'):
        sampled_means(features, np.array([[12, 99]]), rows)


def test_association_matches_spearman_with_ties_and_keeps_rails_undefined():
    x = np.array([1., 1., 2., 4., 5., np.nan])
    outcome = np.array([0., 1., -1., 0., 1., 1.])
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = association(x, outcome, labels, 100, 42)
    assert result['patients'] == 5
    assert np.isclose(result['rho'], spearmanr(x[:-1], outcome[:-1]).statistic)
    assert 0 < result['finite_draws'] <= 100
    assert association(np.zeros_like(x), outcome, labels, 100, 42) == {
        'patients': 6, 'rho': None, 'interval': None, 'finite_draws': 0,
    }
