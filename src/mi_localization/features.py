"""Paper-faithful lead x time x subband tensors and Tucker-2 compression."""

from __future__ import annotations

import numpy as np
import pywt
import warnings
from scipy.linalg import svd


def reconstruct_detail(signal: np.ndarray, wavelet: str, level: int, detail_level: int) -> np.ndarray:
    # MATLAB accepts nine levels for a 651-sample bior6.8 signal. PyWavelets
    # reproduces that extension but warns that boundary coefficients dominate.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Level value of .* is too high")
        coeffs = pywt.wavedec(signal, wavelet, level=level, mode="symmetric")
    # wavedec ordering: [A_level, D_level, ..., D_1].
    selected = [np.zeros_like(c) for c in coeffs]
    selected[1 + level - detail_level] = coeffs[1 + level - detail_level]
    return pywt.waverec(selected, wavelet, mode="symmetric")[: signal.size]


def tensorize_beat(
    beat: np.ndarray,
    wavelet: str = "bior6.8",
    level: int = 9,
    detail_levels: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9),
) -> np.ndarray:
    """Return X with shape (3 leads, 651 time samples, 8 subbands)."""
    if beat.ndim != 2 or beat.shape[0] != 3:
        raise ValueError(f"Expected beat shape (3, time), got {beat.shape}")
    bands = [beat]
    for detail in detail_levels:
        bands.append(np.vstack([reconstruct_detail(row, wavelet, level, detail) for row in beat]))
    return np.stack(bands, axis=2)


def _canonicalize_columns(u: np.ndarray) -> np.ndarray:
    # SVD vectors are sign-indeterminate. Make the largest-magnitude element in
    # every time factor positive so identical inputs have portable features.
    u = u.copy()
    for column in range(u.shape[1]):
        pivot = int(np.argmax(np.abs(u[:, column])))
        if u[pivot, column] < 0:
            u[:, column] *= -1
    return u


def tucker_time_features(tensor: np.ndarray, rank: int = 3, canonicalize_sign: bool = True) -> np.ndarray:
    """Compress only time mode, matching A=I3 and C=I8 in the paper.

    X_(2) has shape time x (lead*subband); truncated SVD gives B, and the
    core unfolding is B.T @ X_(2), with shape rank x 24 = 72 values.
    """
    time_unfolding = np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], -1, order="F")
    u, _, _ = svd(time_unfolding, full_matrices=False, lapack_driver="gesdd")
    basis = u[:, :rank]
    if canonicalize_sign:
        basis = _canonicalize_columns(basis)
    core_unfolding = basis.T @ time_unfolding
    return core_unfolding.reshape(-1, order="F").astype(np.float32)


def extract_features(beat: np.ndarray, **kwargs: object) -> np.ndarray:
    rank = int(kwargs.pop("time_rank", 3))
    canonical = bool(kwargs.pop("canonicalize_svd_sign", True))
    return tucker_time_features(tensorize_beat(beat, **kwargs), rank, canonical)
