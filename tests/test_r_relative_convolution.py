from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mi_localization.r_relative_convolution_features import permute_windows, signed_rms, validate_windows


def test_signed_rms_retains_positive_and_negative_components():
    values = np.array([[0., 2., -4., 0.], [0., 0., 0., 0.]])
    np.testing.assert_allclose(signed_rms(values), [[1., 2.], [0., 0.]])


def test_shuffle_changes_position_while_preserving_all_window_values():
    windows = np.arange(48).reshape(2, 4, 6)
    order = np.array([[3, 1, 0, 2], [1, 3, 2, 0]])
    shuffled = permute_windows(windows, order)
    np.testing.assert_array_equal(shuffled[0], windows[0, order[0]])
    np.testing.assert_array_equal(shuffled[1], windows[1, order[1]])
    with pytest.raises(ValueError):
        permute_windows(windows, np.tile([0, 1, 1, 3], (2, 1)))


def test_short_window_contract_rejects_padding_and_unequal_support():
    config = dict(windows_ms=[[-200, -64], [-64, 72], [72, 208], [208, 344]],
                  r_index=250, dilations=[1, 4], kernels_per_window=16)
    validate_windows(config, 651)
    with pytest.raises(ValueError, match='valid raw'):
        validate_windows(dict(config, r_index=200), 651)
    with pytest.raises(ValueError, match='equal-duration'):
        validate_windows(dict(config, windows_ms=[[-200, -63], [-63, 72], [72, 208], [208, 344]]), 651)


def test_same_pulse_changes_aligned_slot_and_preserves_whole_beat_features():
    code = '''
from types import SimpleNamespace
import numpy as np
from mi_localization.neural import _seed
from mi_localization.r_relative_convolution_features import convolution_blocks

_seed(42, {'device':'cpu', 'torch_threads':2})
weight = np.zeros((84,3,9), np.float32)
weight[:,0,4] = 1
rocket = SimpleNamespace(dilations=[[1,4],[1,4]], weights=[weight]*4,
                         biases=[np.zeros(84,np.float32)]*4)
transform = SimpleNamespace(scale=1, multirocket=rocket, config={'chunk_size':8})
config = dict(windows_ms=[[-200, -64], [-64, 72], [72, 208], [208, 344]],
              r_index=250, dilations=[1,4], kernels_per_window=16, kernel_selection_seed=4)
x = np.zeros((2,3,651), np.float32)
x[0,0,100] = x[1,0,360] = 1
b, ids = convolution_blocks(x, transform, np.tile([1,0,3,2],(2,1)), config)
assert all(v.shape == (2,512) for v in b.values())
assert all(len(set(k)) == 64 for k in ids)
np.testing.assert_array_equal(b['whole_beat'][0], b['whole_beat'][1])
a = b['aligned'].reshape(2,4,4,16,2)
assert a[0,0,0,0,0] > 0 and a[1,0,2,0,0] > 0
np.testing.assert_array_equal(b['shuffled'].reshape(a.shape), a[:,:,[1,0,3,2]])
'''
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, timeout=90)
    assert p.returncode == 0, p.stdout + '\n' + p.stderr
