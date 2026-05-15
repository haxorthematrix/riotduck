"""Tests for the buffered-capture path."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from riotduck.capture import capture_from_buffer
from riotduck.events import Detection
from riotduck.scanner import Scanner, TuneCapture


def _det(center_hz: float, ts: float = 0.0) -> Detection:
    return Detection.new(
        type="appearance",
        ts=ts,
        range_name="t",
        device_serial="test",
        center_hz=center_hz,
        bw_hz=8000.0,
        power_dbfs=-40.0,
        snr_db=30.0,
        bins=[100],
        first_seen_ts=ts,
        last_seen_ts=ts,
    )


def test_capture_from_buffer_writes_exact_iq(tmp_path: Path):
    n = 1024
    iq = np.arange(n, dtype=np.complex64) + 1j * np.arange(n, dtype=np.complex64)
    tune = TuneCapture(
        center_hz=433.92e6, samp_rate=2.4e6, usable_bw_hz=1.8e6, iq=iq,
    )
    det = _det(center_hz=433.92e6)
    cap = capture_from_buffer(tune, det, tmp_path)
    assert cap is not None
    written = np.fromfile(cap.path, dtype=np.complex64)
    assert np.array_equal(written, iq)
    # CaptureResult metadata reflects the BUFFER's tune, not the detection's.
    assert cap.samp_rate == 2.4e6
    assert cap.center_hz == 433.92e6
    assert abs(cap.duration_s - n / 2.4e6) < 1e-9


def test_capture_from_buffer_empty(tmp_path: Path):
    tune = TuneCapture(
        center_hz=433.92e6, samp_rate=2.4e6, usable_bw_hz=1.8e6,
        iq=np.empty(0, dtype=np.complex64),
    )
    det = _det(center_hz=433.92e6)
    assert capture_from_buffer(tune, det, tmp_path) is None


def test_tune_capture_covers():
    tune = TuneCapture(
        center_hz=100e6, samp_rate=2.4e6, usable_bw_hz=1.8e6,
        iq=np.empty(0, dtype=np.complex64),
    )
    assert tune.covers(100e6)
    assert tune.covers(100.5e6)
    assert tune.covers(100.9e6)
    assert not tune.covers(101.0e6)     # exactly at the edge, outside


def test_find_capture_for_freq_picks_closest():
    """Scanner.find_capture_for_freq should choose the tune that covers
    the freq, preferring the closer center if several overlap."""
    s = object.__new__(Scanner)        # don't open an SDR
    s.last_captures = [
        TuneCapture(center_hz=100e6, samp_rate=2.4e6, usable_bw_hz=1.8e6,
                    iq=np.empty(1, dtype=np.complex64)),
        TuneCapture(center_hz=101.5e6, samp_rate=2.4e6, usable_bw_hz=1.8e6,
                    iq=np.empty(1, dtype=np.complex64)),
        TuneCapture(center_hz=103e6, samp_rate=2.4e6, usable_bw_hz=1.8e6,
                    iq=np.empty(1, dtype=np.complex64)),
    ]
    # 100.1 MHz is covered by the first tune only.
    got = s.find_capture_for_freq(100.1e6)
    assert got is not None and got.center_hz == 100e6
    # 102 MHz: covered by both 101.5 and 103, closer to 101.5.
    got = s.find_capture_for_freq(102e6)
    assert got is not None and got.center_hz == 101.5e6
    # 104.5 MHz: outside everyone's usable bw.
    assert s.find_capture_for_freq(104.5e6) is None
