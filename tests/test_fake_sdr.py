"""Tests for the synthetic SDR backend."""
from __future__ import annotations

import os

import numpy as np
import pytest

from riotduck.dsp import fft_power_dbfs, make_window
from riotduck.sdr.fake import (
    DEFAULT_PROFILE,
    Emitter,
    FakeBackend,
    fake_available,
    load_profile,
)


def test_fake_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RIOTDUCK_FAKE_DEVICES", raising=False)
    assert not fake_available()
    assert FakeBackend().discover() == []


def test_fake_devices_count_from_env(monkeypatch):
    monkeypatch.setenv("RIOTDUCK_FAKE_DEVICES", "3")
    assert fake_available()
    infos = FakeBackend().discover()
    assert [d.serial for d in infos] == ["fake-0001", "fake-0002", "fake-0003"]


def test_open_session_tunes_and_reads(monkeypatch):
    monkeypatch.setenv("RIOTDUCK_FAKE_DEVICES", "1")
    backend = FakeBackend()
    sess = backend.open("fake-0001")
    sr = sess.set_samp_rate(2.4e6)
    assert sr == 2.4e6
    # Tune to 433.92 MHz; default profile has a burst there. Read just
    # enough samples to see at least some activity in the FFT.
    sess.set_center_hz(433.92e6)
    iq = sess.read_iq(8192)
    assert iq.dtype == np.complex64
    assert len(iq) == 8192


def test_emitter_active_at_burst():
    e = Emitter(kind="burst", hz=433e6, amp=1.0, period_s=1.0, duty=0.1, phase_s=0.0)
    assert e.active_at(0.05)       # within first 10% of the period
    assert not e.active_at(0.5)
    assert e.active_at(1.05)       # wraps


def test_drift_walks_within_range():
    e = Emitter(kind="drift", hz=900e6, drift_hz=10e6, drift_period_s=10.0)
    samples = [e.freq_at(t) for t in np.linspace(0, 10, 21)]
    assert min(samples) >= 900e6 - 1.0
    assert max(samples) <= 910e6 + 1.0


def test_load_profile_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("RIOTDUCK_FAKE_PROFILE", raising=False)
    p = load_profile()
    assert p == DEFAULT_PROFILE

    monkeypatch.setenv("RIOTDUCK_FAKE_PROFILE", str(tmp_path / "nope.yaml"))
    p2 = load_profile()
    assert p2 == DEFAULT_PROFILE


def test_load_profile_from_yaml(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text(
        "emitters:\n"
        "  - kind: carrier\n"
        "    hz: 100.0e+6\n"
        "    amp: 0.5\n"
        "  - kind: burst\n"
        "    hz: 200.0e+6\n"
        "    period_s: 2.0\n"
        "    duty: 0.1\n"
    )
    emitters = load_profile(f)
    assert len(emitters) == 2
    assert emitters[0].kind == "carrier"
    assert emitters[0].hz == 100e6
    assert emitters[1].kind == "burst"


def test_emitter_shows_up_in_fft(monkeypatch):
    # A single carrier inside the tuned passband should produce a
    # measurable peak above the noise floor in an FFT of read_iq().
    monkeypatch.setenv("RIOTDUCK_FAKE_DEVICES", "1")
    backend = FakeBackend(emitters=[Emitter(kind="carrier", hz=100.2e6, amp=0.5)])
    sess = backend.open("fake-0001")
    sess.set_samp_rate(2.4e6)
    sess.set_center_hz(100.0e6)
    sess.set_gain({"tuner": 28})
    n = 4096
    iq = sess.read_iq(n)
    win = make_window("hann", n)
    psd = fft_power_dbfs(iq, win)
    # Peak well above median.
    peak = float(np.max(psd))
    median = float(np.median(psd))
    assert peak - median > 30.0
    # Peak bin should land near the 200 kHz offset.
    bin_hz = 2.4e6 / n
    peak_bin = int(np.argmax(psd))
    peak_offset = (peak_bin - n / 2) * bin_hz
    assert abs(peak_offset - 200e3) < 5 * bin_hz
