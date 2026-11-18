import numpy as np
import pytest

from mi_localization.earliest_training_duration import select_epochs


def test_selection_uses_patient_macro_f1_and_favors_shorter_ties():
    labels = np.array([0, 0, 0, 1])
    accurate = np.array([[.9, .1], [.8, .2], [.8, .2], [.2, .8]])
    majority = np.tile([.9, .1], (4, 1))
    chosen, scores = select_epochs(labels, {32: accurate, 8: majority, 16: accurate}, 2)
    assert chosen == 16
    assert scores['16'] == scores['32'] == 1
    assert scores['8'] < 1


def test_selection_rejects_incomplete_validation_predictions():
    with pytest.raises(ValueError, match='Incomplete'):
        select_epochs(np.array([0, 1]), {8: np.array([[1., 0.], [np.nan, np.nan]])}, 2)
