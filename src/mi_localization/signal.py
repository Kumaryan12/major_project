"""Wavelet denoising, Pan-Tompkins-style R-peak detection, and segmentation."""

from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, find_peaks, sosfiltfilt


def wavelet_denoise(signal: np.ndarray, wavelet: str = "bior6.8", level: int = 9) -> np.ndarray:
    maximum = pywt.dwt_max_level(signal.size, pywt.Wavelet(wavelet).dec_len)
    used_level = min(level, maximum)
    coeffs = pywt.wavedec(signal, wavelet, level=used_level, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(signal.size))
    cleaned = [coeffs[0], *[pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]]
    return pywt.waverec(cleaned, wavelet, mode="symmetric")[: signal.size]


def denoise_vcg(vcg: np.ndarray, wavelet: str = "bior6.8", level: int = 9) -> np.ndarray:
    return np.column_stack([wavelet_denoise(vcg[:, lead], wavelet, level) for lead in range(vcg.shape[1])])


def detect_r_peaks(
    vcg: np.ndarray,
    fs: int,
    refractory_ms: int = 250,
    integration_ms: int = 150,
    threshold_scale: float = 0.35,
) -> np.ndarray:
    """Pan-Tompkins-style detector generalized to the three-lead vector magnitude."""
    magnitude = np.linalg.norm(vcg, axis=1)
    sos = butter(2, (5.0, min(20.0, fs * 0.45)), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, magnitude)
    derivative = np.gradient(filtered)
    squared = derivative * derivative
    window = max(1, int(round(integration_ms * fs / 1000)))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")
    threshold = np.median(integrated) + threshold_scale * (
        np.percentile(integrated, 95) - np.median(integrated)
    )
    candidates, _ = find_peaks(
        integrated,
        height=threshold,
        distance=max(1, int(round(refractory_ms * fs / 1000))),
    )
    radius = max(1, int(round(0.12 * fs)))
    refined: list[int] = []
    for peak in candidates:
        left, right = max(0, peak - radius), min(len(magnitude), peak + radius + 1)
        refined.append(left + int(np.argmax(magnitude[left:right])))
    if not refined:
        return np.empty(0, dtype=int)
    peaks = np.unique(refined)
    keep = [int(peaks[0])]
    minimum = int(round(refractory_ms * fs / 1000))
    for peak in peaks[1:]:
        if peak - keep[-1] >= minimum:
            keep.append(int(peak))
        elif magnitude[peak] > magnitude[keep[-1]]:
            keep[-1] = int(peak)
    return np.asarray(keep, dtype=int)


def extract_beats(vcg: np.ndarray, peaks: np.ndarray, before: int = 250, after: int = 400) -> tuple[np.ndarray, np.ndarray]:
    valid = peaks[(peaks >= before) & (peaks + after < len(vcg))]
    beats = np.stack([vcg[peak - before : peak + after + 1].T for peak in valid]) if len(valid) else np.empty((0, vcg.shape[1], before + after + 1))
    return beats.astype(np.float32), valid

