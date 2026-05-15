"""Tests for the unknown-signal classifier.

Synthesized signals of known type → expect the right modulation
label, sensible bandwidth, and plausible symbol rates.
"""
from __future__ import annotations

import numpy as np
import pytest

from riotduck.analysis.classifier import (
    analyze,
    burst_segments,
    classify_modulation,
    estimate_freq_offset,
    estimate_symbol_rate,
    measure_bandwidth,
)


# ---------- signal generators ----------

def gen_cw(samp_rate: float, freq_offset_hz: float, duration_s: float,
           amp: float = 0.5) -> np.ndarray:
    n = int(samp_rate * duration_s)
    t = np.arange(n) / samp_rate
    iq = amp * np.exp(2j * np.pi * freq_offset_hz * t)
    return iq.astype(np.complex64)


def gen_ook(samp_rate: float, freq_offset_hz: float, symbol_rate: float,
            duration_s: float, amp: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(samp_rate * duration_s)
    t = np.arange(n) / samp_rate
    carrier = amp * np.exp(2j * np.pi * freq_offset_hz * t)
    samples_per_sym = int(samp_rate / symbol_rate)
    n_symbols = n // samples_per_sym + 1
    bits = rng.integers(0, 2, size=n_symbols)
    mask = np.repeat(bits, samples_per_sym).astype(np.float32)[:n]
    return (carrier * mask).astype(np.complex64)


def gen_fsk(samp_rate: float, freq_offset_hz: float, deviation_hz: float,
            symbol_rate: float, duration_s: float, amp: float = 0.5,
            seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(samp_rate * duration_s)
    samples_per_sym = int(samp_rate / symbol_rate)
    n_symbols = n // samples_per_sym + 1
    bits = rng.integers(0, 2, size=n_symbols)
    freqs = freq_offset_hz + (bits * 2 - 1) * deviation_hz
    inst_freq = np.repeat(freqs, samples_per_sym).astype(np.float64)[:n]
    phase = 2 * np.pi * np.cumsum(inst_freq) / samp_rate
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _add_noise(iq: np.ndarray, sigma: float = 1e-3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))) * sigma
    return (iq + noise).astype(np.complex64)


# ---------- bandwidth tests ----------

def test_bandwidth_of_cw_is_narrow():
    sr = 2.4e6
    iq = _add_noise(gen_cw(sr, freq_offset_hz=100e3, duration_s=0.02))
    bw = measure_bandwidth(iq, sr)
    # A pure tone (FFT bin) should be very narrow.
    assert bw[3.0] < sr / 1024
    assert bw[6.0] < sr / 512


def test_bandwidth_of_ook_scales_with_symbol_rate():
    sr = 1.024e6
    sym_rate = 2000.0
    iq = _add_noise(gen_ook(sr, freq_offset_hz=80e3, symbol_rate=sym_rate,
                            duration_s=0.1))
    bw = measure_bandwidth(iq, sr)
    # OOK's carrier dominates the PSD so -3 dB BW is essentially the
    # FFT bin width. Modulation sidebands carrying the symbol rate
    # show up at the -20 dB level instead.
    assert 1000.0 < bw[20.0] < 50_000.0


# ---------- frequency offset ----------

def test_freq_offset_recovers_tone_offset():
    sr = 2.4e6
    target = 230_000.0
    iq = _add_noise(gen_cw(sr, target, 0.01))
    off = estimate_freq_offset(iq, sr)
    df = sr / 8192
    assert abs(off - target) < 3 * df


# ---------- burst segmentation ----------

def test_burst_segments_finds_a_gated_pulse():
    sr = 2.4e6
    duration = 0.05    # 50 ms total
    n = int(sr * duration)
    iq = np.zeros(n, dtype=np.complex64)
    burst_start = int(sr * 0.020)
    burst_len = int(sr * 0.010)         # 10 ms burst
    t = np.arange(burst_len) / sr
    iq[burst_start:burst_start + burst_len] = (0.5 * np.exp(2j * np.pi * 100e3 * t)).astype(np.complex64)
    iq = _add_noise(iq, sigma=5e-4)
    bursts = burst_segments(iq, sr)
    assert len(bursts) == 1
    b = bursts[0]
    # Allow some slop on the edges.
    assert abs(b.start_sample - burst_start) < sr * 1e-3
    assert abs(b.length - burst_len) < sr * 2e-3


# ---------- modulation classification ----------

def test_classify_cw():
    sr = 2.4e6
    iq = _add_noise(gen_cw(sr, freq_offset_hz=10e3, duration_s=0.02))
    bw = measure_bandwidth(iq, sr)
    mod, conf, _ = classify_modulation(iq, sr, bw_3db_hz=bw[3.0])
    assert mod == "CW", f"got {mod}"
    assert conf >= 0.6


def test_classify_ook():
    sr = 1.024e6
    iq = _add_noise(gen_ook(sr, freq_offset_hz=80e3, symbol_rate=2000.0,
                            duration_s=0.1), sigma=1e-3)
    bw = measure_bandwidth(iq, sr)
    mod, conf, _ = classify_modulation(iq, sr, bw_3db_hz=bw[3.0])
    assert mod == "OOK", f"got {mod}"


def test_classify_fsk():
    sr = 1.024e6
    iq = _add_noise(gen_fsk(sr, freq_offset_hz=0, deviation_hz=20e3,
                            symbol_rate=2400.0, duration_s=0.05), sigma=1e-3)
    bw = measure_bandwidth(iq, sr)
    mod, conf, _ = classify_modulation(iq, sr, bw_3db_hz=bw[3.0])
    # FSK distinguishing from FM is fuzzy with random data; accept either.
    assert mod in ("FSK", "FM"), f"got {mod}"


# ---------- symbol rate ----------

def test_symbol_rate_ook_within_15_percent():
    sr = 1.024e6
    target_sym = 2000.0
    iq = _add_noise(gen_ook(sr, freq_offset_hz=80e3, symbol_rate=target_sym,
                            duration_s=0.5), sigma=5e-4)
    est = estimate_symbol_rate(iq, sr, min_rate_hz=500, max_rate_hz=10_000)
    assert est is not None
    assert abs(est - target_sym) / target_sym < 0.15


# ---------- full pipeline ----------

def test_analyze_full_pipeline_on_ook():
    sr = 1.024e6
    iq = _add_noise(gen_ook(sr, freq_offset_hz=80e3, symbol_rate=2000.0,
                            duration_s=0.3), sigma=5e-4)
    r = analyze(iq, sr)
    assert r.modulation == "OOK"
    assert r.bw_3db_hz is not None and r.bw_3db_hz > 0
    assert r.symbol_rate_hz is not None
    assert abs(r.symbol_rate_hz - 2000.0) / 2000.0 < 0.2


def test_analyze_handles_empty():
    r = analyze(np.empty(0, dtype=np.complex64), 2.4e6)
    assert r.modulation == "unknown"
    assert r.bursts == []
