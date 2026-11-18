import numpy as np
import pytest

from mi_localization.model_error_comparison import checked_probabilities, error_overlap
from mi_localization.signal_audit import file_hash


def test_error_overlap_separates_shared_same_and_different_mistakes():
    truth = np.array([0, 0, 0, 2, 2, 0])
    convolution = np.array([0, 0, 1, 0, 0, 0])
    waveform = np.array([0, 1, 0, 0, 1, 0])
    assert error_overlap(truth, convolution, waveform) == {
        'patients': 6,
        'both_correct': 2,
        'convolution_only_correct': 1,
        'waveform_only_correct': 1,
        'both_wrong_same_prediction': 1,
        'both_wrong_different_predictions': 1,
        'both_wrong': 2,
    }


def test_probability_replay_checks_both_hash_and_held_out_row_order(tmp_path):
    path = tmp_path / 'saved.npz'
    rows = np.array([4, 6])
    p = np.array([[.75, .25], [.4, .6]])
    np.savez(path, rows=rows, probabilities=p)
    good_hash = file_hash(path)
    np.testing.assert_array_equal(checked_probabilities(path, good_hash, rows, 2), p)
    with pytest.raises(ValueError, match='held-out membership'):
        checked_probabilities(path, good_hash, rows[::-1], 2)
    np.savez(path, rows=rows, probabilities=p[::-1])
    with pytest.raises(ValueError, match='hash changed'):
        checked_probabilities(path, good_hash, rows, 2)
