from types import SimpleNamespace

import numpy as np

from mi_localization.staffiii_pair_bank import paired_event, physical_samples


WINDOWS = [[-20, -10], [30, 40]]


def annotation(events):
    return SimpleNamespace(sample=[p for p, _ in events], aux_note=[name for _, name in events])


def test_pair_uses_same_record_and_bounds_at_first_inflation():
    events = annotation([(25001, 'balloon inflation'), (80000, 'balloon deflation'),
                         (145000, 'balloon inflation'), (190000, 'balloon deflation')])
    assert paired_event(events, 190001, WINDOWS) == (
        (25001, [5001, 15001], [55001, 65001]), 'timing_candidate')
    assert paired_event(events, 60000, WINDOWS) == (None, 'unbounded_inflation')


def test_recorded_contrast_between_windows_blocks_pair():
    events = annotation([(24000, 'balloon inflation'), (30000, 'contrast injection'), (70000, 'balloon deflation')])
    assert paired_event(events, 70000, WINDOWS) == (None, 'annotated_contrast')


def test_decode_uses_signed_gain_and_baseline_with_requested_channel_order():
    x = np.array([[5, -10, 17], [15, 10, 3]], dtype='<i2')
    header = SimpleNamespace(baseline=[5, 0, 13], adc_gain=[10, 20, 2])
    np.testing.assert_array_equal(physical_samples(x, header, [2, 0, 1]), [[2, 0, -.5], [-5, 1, .5]])
