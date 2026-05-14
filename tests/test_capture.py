"""Tests for inline capture: helpers + ScannerAgent integration."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from riotduck.capture import capture_for_detection, choose_capture_samp_rate
from riotduck.events import Detection
from riotduck.sdr.base import DeviceInfo, SDRSession
from riotduck.storage.files import capture_path, read_iq_cf32


class FakeSession(SDRSession):
    """Test double — records calls, returns synthetic I/Q."""

    def __init__(self, sr: float = 2.4e6, payload: np.ndarray | None = None) -> None:
        self._sr = sr
        self._info = DeviceInfo(
            serial="fake-001",
            type="rtlsdr",
            driver="fake",
            tuning_range_hz=(24e6, 1.7e9),
            samp_rates=(250e3, 1.024e6, 2.4e6),
            gain_stages=("tuner",),
        )
        self._payload = payload
        self.center_calls: list[float] = []
        self.sr_calls: list[float] = []
        self.gain_calls: list[dict] = []
        self.read_sizes: list[int] = []

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def set_center_hz(self, hz: float) -> None:
        self.center_calls.append(hz)

    def set_samp_rate(self, sps: float) -> float:
        self.sr_calls.append(sps)
        self._sr = sps
        return sps

    def set_gain(self, stages):
        self.gain_calls.append(dict(stages))

    def read_iq(self, n_samples: int) -> np.ndarray:
        self.read_sizes.append(n_samples)
        if self._payload is not None:
            return self._payload[:n_samples]
        # Deterministic synthetic complex tone for verifiability.
        t = np.arange(n_samples) / max(self._sr, 1.0)
        return (0.5 * np.exp(2j * np.pi * 100e3 * t)).astype(np.complex64)

    def close(self) -> None:
        pass


def _det(ts: float = 0.0, **kw) -> Detection:
    defaults = dict(
        type="appearance",
        ts=ts,
        range_name="t",
        device_serial="fake-001",
        center_hz=433.92e6,
        bw_hz=10e3,
        power_dbfs=-40.0,
        snr_db=30.0,
        bins=[100],
        first_seen_ts=ts,
        last_seen_ts=ts,
    )
    defaults.update(kw)
    return Detection.new(**defaults)


def test_capture_path_partitions_by_day(tmp_path: Path):
    ts = time.mktime(time.strptime("2026-05-13", "%Y-%m-%d"))
    p = capture_path(tmp_path, "abc123", ts)
    assert p.parent.name == "2026-05-13" or p.parent.name.startswith("2026-05-1")
    assert p.name == "abc123.cf32"
    assert p.parent.exists()


def test_choose_capture_samp_rate_prefers_range_default():
    d = _det(bw_hz=10e3)
    sr = choose_capture_samp_rate(d, range_samp_rate=2.4e6, supported_rates=(250e3, 1e6, 2.4e6))
    assert sr == 2.4e6


def test_choose_capture_samp_rate_picks_min_feasible_when_no_range_sr():
    d = _det(bw_hz=100e3)   # needs >= 400 kS/s
    sr = choose_capture_samp_rate(d, range_samp_rate=None, supported_rates=(250e3, 1e6, 2.4e6))
    assert sr == 1e6


def test_choose_capture_samp_rate_falls_back_to_max():
    d = _det(bw_hz=10e6)    # needs more than any supported rate
    sr = choose_capture_samp_rate(d, range_samp_rate=None, supported_rates=(250e3, 1e6, 2.4e6))
    assert sr == 2.4e6


def test_capture_for_detection_writes_cf32(tmp_path: Path):
    sess = FakeSession(sr=2.4e6)
    d = _det()
    cap = capture_for_detection(
        sess, d, captures_dir=tmp_path, capture_ms=10.0, samp_rate=2.4e6,
        gain={"tuner": 28},
    )
    assert cap is not None
    assert sess.center_calls == [d.center_hz]
    assert sess.sr_calls == [2.4e6]
    assert sess.gain_calls == [{"tuner": 28}]
    # 10 ms @ 2.4 MS/s = 24000 samples
    assert sess.read_sizes == [24000]
    assert cap.path.endswith(".cf32")
    assert Path(cap.path).exists()
    iq = read_iq_cf32(cap.path)
    assert iq.dtype == np.complex64
    assert len(iq) == 24000
    assert abs(cap.duration_s - 10e-3) < 1e-6


def test_capture_for_detection_handles_empty_read(tmp_path: Path):
    sess = FakeSession(sr=2.4e6, payload=np.empty(0, dtype=np.complex64))
    d = _det()
    cap = capture_for_detection(
        sess, d, captures_dir=tmp_path, capture_ms=10.0, samp_rate=2.4e6,
    )
    assert cap is None
