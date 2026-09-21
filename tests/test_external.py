import numpy as np
import pytest

from mi_localization.external import ECG_LEADS, KORS_MATRIX, kors_transform


def test_kors_matrix_matches_eight_independent_leads():
    ecg = np.zeros((3, 12))
    names = list(ECG_LEADS) + ["iii", "avr", "avl", "avf"]
    ecg[0, 0] = 1.0
    ecg[1, 1] = 1.0
    ecg[2, 7] = 1.0
    result = kors_transform(ecg, names)
    assert result.shape == (3, 3)
    assert np.allclose(result[0], KORS_MATRIX[:, 0])
    assert np.allclose(result[1], KORS_MATRIX[:, 1])
    assert np.allclose(result[2], KORS_MATRIX[:, 7])
    with pytest.raises(ValueError, match="missing"):
        kors_transform(ecg[:, 1:], names[1:])
