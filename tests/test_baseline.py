"""End-to-end test of the baseline detector with synthetic sweeps."""
import numpy as np

from riotduck.baseline import BaselineEngine
from riotduck.config import DetectionConfig, RangeConfig
from riotduck.events import SweepFrame


def _frame(power: np.ndarray, freqs: np.ndarray, ts: float = 0.0) -> SweepFrame:
    return SweepFrame(
        range_name="t",
        device_serial="test",
        ts=ts,
        freqs_hz=freqs,
        power_dbfs=power.astype(np.float32),
        bin_hz=float(freqs[1] - freqs[0]),
    )


def test_appearance_fires_after_warmup():
    n_bins = 64
    freqs = np.linspace(433.0e6, 433.0e6 + 64 * 4e3, n_bins)
    rng = RangeConfig(name="t", f_start=float(freqs[0]), f_end=float(freqs[-1]), bin_hz=4000.0)
    det = DetectionConfig(warmup_min=20, n_up=2, n_down=2, k_up=6, k_down=2, window_size=64)
    eng = BaselineEngine(range_cfg=rng, detect_cfg=det)

    rng_rand = np.random.default_rng(0)
    # Warmup: noise floor at -90 dBFS, MAD ~ 1 dB.
    for ts in range(30):
        noise = -90.0 + rng_rand.normal(0, 1.0, size=n_bins)
        eng.ingest(_frame(noise, freqs, ts=float(ts)))

    # Inject a strong tone at bin 30 for a few consecutive sweeps.
    detections = []
    for ts in range(30, 35):
        sample = -90.0 + rng_rand.normal(0, 1.0, size=n_bins)
        sample[30] = -40.0
        detections.extend(eng.ingest(_frame(sample, freqs, ts=float(ts))))

    appearances = [d for d in detections if d.type == "appearance"]
    assert len(appearances) >= 1
    a = appearances[0]
    assert abs(a.center_hz - freqs[30]) < rng.bin_hz
    assert a.snr_db > 20


def test_disappearance_fires_for_steady_emitter_going_silent():
    n_bins = 64
    freqs = np.linspace(433.0e6, 433.0e6 + 64 * 4e3, n_bins)
    rng = RangeConfig(name="t", f_start=float(freqs[0]), f_end=float(freqs[-1]), bin_hz=4000.0)
    det = DetectionConfig(warmup_min=30, n_up=2, n_down=2, k_up=6, k_down=2, window_size=64)
    eng = BaselineEngine(range_cfg=rng, detect_cfg=det)

    rng_rand = np.random.default_rng(1)
    # Warmup with a *persistently present* emitter at bin 20.
    for ts in range(40):
        sample = -90.0 + rng_rand.normal(0, 1.0, size=n_bins)
        sample[20] = -50.0 + rng_rand.normal(0, 0.5)
        eng.ingest(_frame(sample, freqs, ts=float(ts)))

    detections = []
    # Emitter goes silent.
    for ts in range(40, 50):
        sample = -90.0 + rng_rand.normal(0, 1.0, size=n_bins)
        detections.extend(eng.ingest(_frame(sample, freqs, ts=float(ts))))

    disappearances = [d for d in detections if d.type == "disappearance"]
    assert len(disappearances) >= 1
    d = disappearances[0]
    assert abs(d.center_hz - freqs[20]) < rng.bin_hz
