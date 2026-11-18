import numpy as np
import pytest

from mi_localization.benchmark import patient_balanced_weights
from mi_localization.probabilistic_subtype import combine_probabilities, error_breakdown, predict_arms


def test_probability_factorization_uses_the_declared_class_positions():
    p = combine_probabilities(np.array([0.8, 0.3]), np.array([[0.75, 0.25], [0.2, 0.8]]), [2, 0], 1)
    np.testing.assert_allclose(p, [[0.2, 0.2, 0.6], [0.24, 0.7, 0.06]])
    np.testing.assert_allclose(p.sum(1), 1)
    with pytest.raises(ValueError):
        combine_probabilities(np.array([0.8]), np.array([[0.75, 0.25]]), [2, 2], 1)


def test_both_arms_preserve_original_class_prior_under_zero_features():
    y = np.repeat(np.arange(6), [2, 4, 1, 1, 3, 6])
    x = np.zeros((len(y), 2))
    weights = patient_balanced_weights(y, np.arange(len(y)))
    p, state = predict_arms(x, y, weights, x[:2], 5, 0.01, dict(max_iterations=1000, tolerance=1e-9))
    np.testing.assert_allclose(p['flat'], 1.0 / 6, atol=1e-6)
    np.testing.assert_allclose(p['hierarchical'], 1.0 / 6, atol=1e-6)
    np.testing.assert_array_equal(state['subtype']['classes'], np.arange(5))


def test_error_categories_distinguish_detection_and_subtype_confusions():
    y = np.array([0, 1, 5, 2, 5])
    predictions = np.array([1, 5, 3, 2, 5])
    assert error_breakdown(y, predictions, 5) == dict(within_mi=1, mi_to_hc=1, hc_to_mi=1)
