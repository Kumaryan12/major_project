from datetime import date

import numpy as np
import pandas as pd

from mi_localization.index_recording_audit import pool_predictions, recording_date, select_record


def test_dates_use_day_first_and_require_one_valid_header_field():
    assert recording_date('# ECG date: 03/04/1993') == date(1993, 4, 3)
    assert recording_date('# ECG date: 29/02/1996') == date(1996, 2, 29)
    assert recording_date('# ECG date: 29/02/1993') is None
    assert recording_date('# ECG date: n/a') is None
    assert recording_date('# ECG date: 03/04/1993\n# ECG date: 03/04/1993') is None
    assert recording_date('# Infarction date (acute): 03/04/1993') is None


def test_selection_uses_dates_before_record_names_and_tracks_fallbacks():
    earlier, later = date(1993, 4, 3), date(1993, 4, 4)
    assert select_record({'a': later, 'b': earlier, 'c': later}) == ('b', 'eligible')
    assert select_record({'z': earlier, 'y': earlier, 'a': later}) == ('y', 'earliest_date_tie')
    assert select_record({'a': later, 'b': earlier, 'z': None}) == ('a', 'missing_or_invalid_date')
    assert select_record({'a': None}) == ('a', 'missing_or_invalid_date')


def test_one_record_per_patient_and_record_equal_pooling_with_uneven_beat_counts():
    beats = pd.DataFrame({'patient_id': ['p1', 'p1', 'p1', 'p2'],
                          'record': ['a', 'a', 'z', 'b']})
    probabilities = np.array([[1., 0.], [1., 0.], [0., 1.], [.3, .7]])
    pooled = pool_predictions(probabilities, beats, np.array(['p1', 'p2']),
                              np.array([0, 1]), np.array(['z', 'b']))
    np.testing.assert_allclose(pooled['all_beats'], [[2 / 3, 1 / 3], [.3, .7]])
    np.testing.assert_allclose(pooled['equal_records'], [[.5, .5], [.3, .7]])
    np.testing.assert_allclose(pooled['one_record'], [[0., 1.], [.3, .7]])
