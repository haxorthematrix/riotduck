import numpy as np

from riotduck.dsp import (
    average_frames,
    decimate_to_bin_width,
    fft_power_dbfs,
    make_window,
)


def test_fft_power_dbfs_full_scale_tone():
    n = 1024
    sr = 2.4e6
    f = 100e3
    t = np.arange(n) / sr
    iq = np.exp(2j * np.pi * f * t).astype(np.complex64)
    win = make_window("hann", n)
    psd = fft_power_dbfs(iq, win)
    peak_bin = int(np.argmax(psd))
    bin_hz = sr / n
    bin_center_hz = (peak_bin - n / 2) * bin_hz
    assert abs(bin_center_hz - f) < 2 * bin_hz
    assert psd[peak_bin] > -10.0     # full-scale tone, should be loud


def test_average_frames_recovers_mean_db():
    a = np.full(8, -50.0, dtype=np.float32)
    b = np.full(8, -50.0, dtype=np.float32)
    out = average_frames([a, b])
    assert np.allclose(out, -50.0, atol=1e-3)


def test_decimate_to_bin_width_groups_correctly():
    freqs = np.linspace(100e6, 100.001e6, 11)   # 100 Hz bins
    power = np.arange(11, dtype=np.float32) * -1.0
    f2, p2 = decimate_to_bin_width(freqs, power, target_bin_hz=500.0)
    assert len(f2) == 2     # 11 // 5 = 2 groups of 5
    assert len(p2) == 2
