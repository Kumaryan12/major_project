from datetime import date
from types import SimpleNamespace

import numpy as np

from mi_localization.blinded_recording_review import match_cases
from mi_localization.matched_recording_beats import (
    ARMS, DRAWS, RecordPair, decisions, order_pair, sample_indices, summarize,
)


def test_pair_requires_unambiguous_earliest_and_uses_next_dated_record():
    first, second, third = [date(1993, 4, day) for day in (1, 2, 3)]
    assert order_pair({'z': first, 'a': third, 'b': second, 'c': second}) == ('z', 'b')
    assert order_pair({'z': first, 'c': first, 'b': second}) is None
    assert order_pair({'z': first, 'b': None}) is None


def test_each_matched_arm_uses_the_same_even_budget_with_shared_prefixes():
    pair = RecordPair('p', 'a', 'b', date(1993, 4, 1), date(1993, 4, 2),
                      np.arange(7), np.arange(7, 17))
    early, later = sample_indices(pair, np.random.default_rng(42))
    assert early.shape == later.shape == (DRAWS, 6)
    assert np.all(np.diff(np.sort(early), axis=1) > 0)
    assert np.all(np.diff(np.sort(later), axis=1) > 0)
    assert set(np.unique(early)) == set(pair.early_rows)
    assert set(np.unique(later)) == set(pair.later_rows)

    p = np.random.default_rng(42).dirichlet(np.ones(3), size=17)
    pooled = decisions(p, pair, early, later)
    combined = np.concatenate([early[:, :3], later[:, :3]], axis=1)
    np.testing.assert_array_equal(pooled[2], p[combined].mean(1).argmax(1))


def test_expected_confusion_counts_each_patient_once():
    truth = np.array([0, 1])
    calls = np.array([[0, 1], [1, 1]])
    results, contrasts = summarize(truth, np.repeat(calls[None, :, :], len(ARMS), axis=0),
                                   ['a', 'b'], 30, 42)
    np.testing.assert_array_equal(results['earliest']['confusion'], [[.5, .5], [0, 1]])
    assert np.isclose(results['earliest']['macro_f1'], (2 / 3 + .8) / 2)
    assert contrasts['half_from_each_minus_earliest'] == {'delta': 0., 'interval': [0., 0.]}


def test_review_match_uses_class_and_globally_minimizes_gap_difference():
    pairs = [SimpleNamespace(days=days) for days in (12, 100, 90, 11, 2, 2, 120)]
    truth = np.array([0, 0, 0, 0, 1, 1, 0])
    changed = np.array([True, True, False, False, True, False, True])
    assert set(match_cases(pairs, truth, changed)) == {(0, 3), (1, 2), (4, 5)}
