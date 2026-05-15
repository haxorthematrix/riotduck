"""Tests for the capture .meta.json sidecar.

The sidecar lets `riotduck analyze <cf32>` work without explicit
`--samp-rate` / `--center` flags and lets `library add --from-capture`
re-tune the analyzer correctly on an already-on-disk capture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from riotduck.capture import capture_for_detection, capture_from_buffer
from riotduck.events import Detection
from riotduck.scanner import TuneCapture
from riotduck.storage.files import (
    META_SCHEMA_VERSION,
    read_capture_meta,
    sidecar_path_for,
    write_capture_meta,
)
from tests.test_capture import FakeSession


def _det(center_hz: float = 433.92e6, ts: float = 0.0) -> Detection:
    return Detection.new(
        type="appearance",
        ts=ts,
        range_name="t",
        device_serial="abc",
        center_hz=center_hz,
        bw_hz=8000.0,
        power_dbfs=-42.0,
        snr_db=33.0,
        bins=[100],
        first_seen_ts=ts,
        last_seen_ts=ts,
    )


def test_sidecar_path_is_meta_json():
    p = Path("/tmp/captures/2026-05-13/abcd.cf32")
    assert sidecar_path_for(p).name == "abcd.meta.json"
    assert sidecar_path_for(p).parent == p.parent


def test_write_and_read_round_trip(tmp_path: Path):
    iq_path = tmp_path / "x.cf32"
    iq_path.write_bytes(b"")  # the bytes don't matter for this test
    det = _det(center_hz=433.928e6, ts=1234567890.0)
    sidecar = write_capture_meta(
        iq_path,
        detection=det,
        samp_rate=2.4e6,
        capture_center_hz=433.92e6,
        duration_s=0.025,
    )
    assert sidecar.exists()
    assert sidecar.name == "x.meta.json"

    meta = read_capture_meta(iq_path)
    assert meta is not None
    assert meta["schema_version"] == META_SCHEMA_VERSION
    assert meta["samp_rate"] == 2.4e6
    assert meta["capture_center_hz"] == 433.92e6
    assert meta["duration_s"] == 0.025
    assert meta["iq_path"] == "x.cf32"
    d = meta["detection"]
    assert d["id"] == det.id
    assert d["type"] == "appearance"
    assert d["center_hz"] == 433.928e6
    assert d["snr_db"] == 33.0


def test_read_meta_returns_none_when_missing(tmp_path: Path):
    iq_path = tmp_path / "no_sidecar.cf32"
    iq_path.write_bytes(b"")
    assert read_capture_meta(iq_path) is None


def test_read_meta_returns_none_on_garbage(tmp_path: Path):
    iq_path = tmp_path / "broken.cf32"
    iq_path.write_bytes(b"")
    sidecar_path_for(iq_path).write_text("{not valid json")
    assert read_capture_meta(iq_path) is None


def test_capture_for_detection_writes_sidecar(tmp_path: Path):
    sess = FakeSession(sr=2.4e6)
    det = _det()
    cap = capture_for_detection(
        sess, det, captures_dir=tmp_path, capture_ms=5.0, samp_rate=2.4e6,
    )
    assert cap is not None
    sidecar = Path(cap.path).with_suffix(".meta.json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["samp_rate"] == 2.4e6
    assert meta["capture_center_hz"] == det.center_hz
    assert meta["detection"]["id"] == det.id


def test_capture_from_buffer_writes_sidecar(tmp_path: Path):
    iq = np.ones(2048, dtype=np.complex64)
    tune = TuneCapture(
        center_hz=433.92e6, samp_rate=2.4e6, usable_bw_hz=1.8e6, iq=iq,
    )
    det = _det(center_hz=433.93e6)   # slightly off center to verify both
    cap = capture_from_buffer(tune, det, tmp_path)
    assert cap is not None
    meta = read_capture_meta(cap.path)
    assert meta is not None
    # The buffered path records the tune center (not the detection center).
    assert meta["capture_center_hz"] == 433.92e6
    assert meta["samp_rate"] == 2.4e6
    assert meta["detection"]["center_hz"] == 433.93e6
