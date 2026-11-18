import numpy as np

from mi_localization.distribution_score_correction import BRANCHES, choose, fit_corrections


def test_correction_matches_weighted_ridge_and_has_exact_off_candidate():
    x = np.array([[1., 1.], [2., -1.], [0., 1.]])
    logits = np.array([[0.4, -0.2], [-0.5, 0.6], [0.3, 0.]])
    labels = np.array([0, 1, 0])
    weights = np.array([.75, 1.5, .75])
    mean, coefs = fit_corrections(x, logits, labels, [None, 1., .1])

    np.testing.assert_allclose(mean, (weights @ x) / weights.sum())
    np.testing.assert_array_equal(coefs[0], np.zeros((2, 2)))
    centered = x - mean
    residuals = np.array([[1, -1], [-1, 1], [1, -1]]) - logits
    gram = centered.T @ np.diag(weights) @ centered
    for ratio, coef in zip([1., .1], coefs[1:]):
        np.testing.assert_allclose(
            (gram + ratio * np.trace(gram) * np.eye(2)) @ coef,
            centered.T @ np.diag(weights) @ residuals,
        )


def test_inner_selection_preserves_baseline_choice_and_prefers_no_correction_on_ties():
    inner = [
        ([.3, .6], {branch: [[.3, .8, .9], [.6, .6, .5]] for branch in BRANCHES}),
        ([.4, .7], {branch: [[.4, .9, 1.], [.7, .7, .6]] for branch in BRANCHES}),
    ]
    selected = choose(inner)
    assert selected['baseline_candidate'] == 1
    assert selected['corrections'] == dict.fromkeys(BRANCHES, 0)
