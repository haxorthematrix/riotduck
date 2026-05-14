"""Numeric helpers: windows, PSD, decimation, dBFS conversion."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


def make_window(kind: str, n: int) -> np.ndarray:
    if kind == "rect":
        return np.ones(n, dtype=np.float32)
    return get_window(kind, n).astype(np.float32)


def fft_power_dbfs(iq: np.ndarray, window: np.ndarray) -> np.ndarray:
    """One-shot PSD in dBFS for a single I/Q frame.

    Returns an FFT-shifted power spectrum so bin 0 is the most negative
    frequency offset from the tune center.
    """
    n = len(iq)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if len(window) != n:
        window = make_window("hann", n)
    x = iq * window
    spec = np.fft.fftshift(np.fft.fft(x))
    # power normalized by (window energy * N) — bin power in linear units
    norm = (window @ window) * n
    p = (spec.real**2 + spec.imag**2) / max(norm, 1e-30)
    # complex64 I/Q from SoapySDR/pyrtlsdr is already scaled to [-1, 1].
    # dBFS = 10 log10(p), referenced to full-scale = 1.
    p = np.maximum(p, 1e-30)
    return (10.0 * np.log10(p)).astype(np.float32)


def average_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Power-domain average across multiple PSD frames (in dB).

    Converts to linear, averages, converts back to dB.
    """
    if not frames:
        return np.empty(0, dtype=np.float32)
    stack = np.stack(frames, axis=0)
    lin = 10.0 ** (stack / 10.0)
    mean_lin = np.mean(lin, axis=0)
    return (10.0 * np.log10(np.maximum(mean_lin, 1e-30))).astype(np.float32)


def decimate_to_bin_width(
    freqs_hz: np.ndarray, power_dbfs: np.ndarray, target_bin_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Group adjacent FFT bins so the resulting bin width >= target.

    Power-domain averaging (linear), then back to dB. If the FFT
    already exceeds the target width, returns the input unchanged.
    """
    if len(freqs_hz) == 0:
        return freqs_hz, power_dbfs
    current_bin = float(freqs_hz[1] - freqs_hz[0]) if len(freqs_hz) > 1 else target_bin_hz
    group = max(1, int(round(target_bin_hz / current_bin)))
    if group <= 1:
        return freqs_hz, power_dbfs
    n = (len(freqs_hz) // group) * group
    f = freqs_hz[:n].reshape(-1, group).mean(axis=1)
    lin = 10.0 ** (power_dbfs[:n] / 10.0)
    p = lin.reshape(-1, group).mean(axis=1)
    return f.astype(np.float64), (10.0 * np.log10(np.maximum(p, 1e-30))).astype(np.float32)


def usable_bw(samp_rate: float, fraction: float = 0.75) -> float:
    """How much of `samp_rate` we treat as usable bandwidth.

    Edges of the FFT are contaminated by anti-alias filter rolloff,
    DC spur, and IQ imbalance image. Default keeps the inner 75%.
    """
    return samp_rate * fraction
