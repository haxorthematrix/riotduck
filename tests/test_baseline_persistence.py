"""Persistent baselines: snapshot round-trip + .npz file I/O."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from riotduck.baseline import S_ABSENT, S_PRESENT, BaselineEngine
from riotduck.config import DetectionConfig, RangeConfig
from riotduck.events import SweepFrame
from riotduck.storage.baselines import (
    baseline_path,
    load_baseline,
    save_baseline,
)


def _frame(power: np.ndarray, freqs: np.ndarray, ts: float = 0.0) -> SweepFrame:
    return SweepFrame(
        range_name="t",
        device_serial="dev",
        ts=ts,
        freqs_hz=freqs,
        power_dbfs=power.astype(np.float32),
        bin_hz=float(freqs[1] - freqs[0]),
    )


def _warmed_engine(seed: int = 0, n_bins: int = 32, frames: int = 60) -> tuple[BaselineEngine, np.ndarray]:
    bin_hz = 4000.0
    freqs = np.linspace(433.0e6, 433.0e6 + n_bins * bin_hz, n_bins)
    rng = RangeConfig(name="t", f_start=float(freqs[0]),
                      f_end=float(freqs[-1]), bin_hz=bin_hz)
    det = DetectionConfig(warmup_min=20, n_up=2, n_down=2, k_up=6, k_down=2,
                          window_size=64)
    eng = BaselineEngine(range_cfg=rng, detect_cfg=det)
    rgen = np.random.default_rng(seed)
    for ts in range(frames):
        noise = -90.0 + rgen.normal(0, 1.0, size=n_bins)
        eng.ingest(_frame(noise, freqs, ts=float(ts)))
    return eng, freqs


# ---------- snapshot round-trip ----------

def test_load_snapshot_round_trip():
    src, freqs = _warmed_engine(seed=0)
    snap = src.snapshot()

    # Fresh engine, same range/detection cfg.
    dst = BaselineEngine(range_cfg=src.range_cfg, detect_cfg=src.detect_cfg)
    assert dst.load_snapshot(snap)

    np.testing.assert_array_equal(dst._bin_centers_hz, src._bin_centers_hz)
    np.testing.assert_array_equal(dst._state, src._state)
    np.testing.assert_array_equal(dst._ring, src._ring)
    np.testing.assert_array_equal(dst._median, src._median)
    np.testing.assert_array_equal(dst._mad, src._mad)
    assert dst._ring_pos == src._ring_pos
    assert dst._ring_filled == src._ring_filled
    assert dst._initialized


def test_load_snapshot_rejects_wrong_bin_count():
    src, _ = _warmed_engine(n_bins=32)
    snap = src.snapshot()
    # Different bin grid → must reject.
    rng = RangeConfig(name="t", f_start=433e6, f_end=433.5e6, bin_hz=4000.0)
    det = DetectionConfig(window_size=64)
    dst = BaselineEngine(range_cfg=rng, detect_cfg=det)
    # Force n_bins via shape — `dst` hasn't initialized yet, but snapshot
    # has 32 freqs while a 433–433.5 MHz range at 4 kHz is 125 bins.
    snap["freqs_hz"] = np.linspace(433e6, 433.5e6, 125)
    assert not dst.load_snapshot(snap)


def test_load_snapshot_rejects_wrong_window_size():
    src, _ = _warmed_engine()
    snap = src.snapshot()
    # Engine with a different window_size cannot consume this ring shape.
    det = DetectionConfig(window_size=128, warmup_min=20)
    dst = BaselineEngine(range_cfg=src.range_cfg, detect_cfg=det)
    assert not dst.load_snapshot(snap)


def test_loaded_engine_continues_detecting():
    """A loaded baseline should fire appearance on the next strong frame
    without needing to re-warm."""
    src, freqs = _warmed_engine()
    snap = src.snapshot()
    dst = BaselineEngine(range_cfg=src.range_cfg, detect_cfg=src.detect_cfg)
    assert dst.load_snapshot(snap)

    n_bins = freqs.size
    rgen = np.random.default_rng(99)
    detections: list = []
    for ts in range(100, 105):
        s = -90.0 + rgen.normal(0, 1.0, size=n_bins)
        s[15] = -40.0   # strong tone
        detections.extend(dst.ingest(_frame(s, freqs, ts=float(ts))))
    appearances = [d for d in detections if d.type == "appearance"]
    assert len(appearances) >= 1
    assert abs(appearances[0].center_hz - freqs[15]) < (freqs[1] - freqs[0])


# ---------- file I/O ----------

def test_baseline_path_is_filesystem_safe(tmp_path: Path):
    p = baseline_path(tmp_path, "ism/433 mhz", "ser:001")
    assert "/" not in p.name
    assert p.name == "ism_433_mhz__ser_001.npz"


def test_save_then_load_round_trip(tmp_path: Path):
    src, _ = _warmed_engine(seed=1)
    path = tmp_path / "b.npz"
    out = save_baseline(path, src)
    assert out == path and path.exists()

    snap = load_baseline(path)
    assert snap is not None

    dst = BaselineEngine(range_cfg=src.range_cfg, detect_cfg=src.detect_cfg)
    assert dst.load_snapshot(snap)
    np.testing.assert_array_equal(dst._state, src._state)
    np.testing.assert_array_equal(dst._ring, src._ring)


def test_save_baseline_skips_uninitialized(tmp_path: Path):
    rng = RangeConfig(name="t", f_start=433e6, f_end=433.5e6, bin_hz=4000.0)
    det = DetectionConfig()
    eng = BaselineEngine(range_cfg=rng, detect_cfg=det)
    assert save_baseline(tmp_path / "x.npz", eng) is None
    assert not (tmp_path / "x.npz").exists()


def test_load_baseline_missing_returns_none(tmp_path: Path):
    assert load_baseline(tmp_path / "nope.npz") is None


def test_load_baseline_too_old(tmp_path: Path):
    src, _ = _warmed_engine()
    path = tmp_path / "b.npz"
    save_baseline(path, src)
    # Backdate the snapshot's ts by editing the file.
    with np.load(str(path)) as data:
        contents = {k: np.array(data[k]) for k in data.files}
    contents["ts"] = np.array([time.time() - 99999.0])
    np.savez_compressed(str(path), **contents)

    assert load_baseline(path, max_age_s=3600.0) is None
    # Without max_age it still loads.
    assert load_baseline(path) is not None


def test_load_baseline_corrupt_returns_none(tmp_path: Path):
    path = tmp_path / "broken.npz"
    path.write_bytes(b"not a real npz file at all")
    assert load_baseline(path) is None
