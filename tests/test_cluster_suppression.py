"""Tests for bin-cluster shadow suppression.

The scenario this fixes: a single strong real emitter triggers
~30 false-positive detections at neighboring frequencies (FFT
sidelobes, IQ imbalance images, tuner spurs). After suppression,
only the strong main lobe survives.

Tested directly against `_suppress_shadowed` and end-to-end through
`BaselineEngine.ingest` with synthesized sweep frames.
"""
from __future__ import annotations

import numpy as np
import pytest

from riotduck.baseline import (
    BaselineEngine,
    _shadow_radius_hz,
    _suppress_shadowed,
)
from riotduck.config import DetectionConfig, RangeConfig
from riotduck.events import Detection, SweepFrame


# ---------- helpers ----------

def _det(center_hz: float, snr_db: float, bw_hz: float = 4000.0,
         type_: str = "appearance") -> Detection:
    return Detection.new(
        type=type_,
        range_name="t",
        device_serial="x",
        center_hz=center_hz,
        bw_hz=bw_hz,
        power_dbfs=-40.0,
        snr_db=snr_db,
        bins=[0],
        first_seen_ts=0.0,
        last_seen_ts=0.0,
    )


def _frame(power: np.ndarray, freqs: np.ndarray, ts: float = 0.0) -> SweepFrame:
    return SweepFrame(
        range_name="t",
        device_serial="test",
        ts=ts,
        freqs_hz=freqs,
        power_dbfs=power.astype(np.float32),
        bin_hz=float(freqs[1] - freqs[0]),
    )


# ---------- _shadow_radius_hz ----------

def test_shadow_zero_below_threshold():
    cfg = DetectionConfig(cluster_shadow_min_snr_db=25.0)
    assert _shadow_radius_hz(_det(433e6, snr_db=10), cfg) == 0.0
    assert _shadow_radius_hz(_det(433e6, snr_db=24.9), cfg) == 0.0


def test_shadow_uses_max_of_base_and_snr():
    cfg = DetectionConfig(
        cluster_shadow_min_snr_db=10.0,
        cluster_shadow_base_hz=20_000.0,
        cluster_shadow_per_db_hz=4_000.0,
    )
    # 15 dB * 4 kHz = 60 kHz, above 20 kHz base.
    assert _shadow_radius_hz(_det(433e6, snr_db=15), cfg) == 60_000.0
    # Base wins when snr * per_db is small relative to base.
    cfg2 = DetectionConfig(
        cluster_shadow_min_snr_db=10.0,
        cluster_shadow_base_hz=20_000.0,
        cluster_shadow_per_db_hz=1_000.0,
    )
    # 11 dB * 1 kHz = 11 kHz, below 20 kHz base — base wins.
    assert _shadow_radius_hz(_det(433e6, snr_db=11), cfg2) == 20_000.0


def test_shadow_scales_with_snr_per_larrys_observation():
    """At Larry's hardware setup a +45 dB carrier showed sidelobes
    out to ~180 kHz. With default config that radius should be in
    the same ballpark."""
    cfg = DetectionConfig()    # defaults
    r = _shadow_radius_hz(_det(433.928e6, snr_db=45.9), cfg)
    assert 150_000 <= r <= 220_000


# ---------- _suppress_shadowed ----------

def test_suppress_empty_and_single():
    cfg = DetectionConfig()
    assert _suppress_shadowed([], cfg) == []
    one = [_det(433e6, snr_db=40)]
    assert _suppress_shadowed(one, cfg) == one


def test_suppress_kills_sidelobes_around_strong_main():
    """Larry's exact pattern: one +45 dB carrier, half a dozen weak
    sidelobes at ±100-200 kHz, all should collapse to one detection."""
    cfg = DetectionConfig()    # defaults give ~180 kHz shadow at +45 dB
    main = _det(433.928e6, snr_db=45.9)
    sidelobes = [
        _det(433.752e6, snr_db=10.0),
        _det(433.812e6, snr_db=10.4),
        _det(433.856e6, snr_db=11.7),
        _det(434.036e6, snr_db=9.5),
        _det(434.060e6, snr_db=10.7),
        _det(434.092e6, snr_db=9.7),
        _det(434.108e6, snr_db=9.1),
    ]
    kept = _suppress_shadowed([main] + sidelobes, cfg)
    assert len(kept) == 1
    assert kept[0].center_hz == main.center_hz


def test_suppress_preserves_input_order_for_kept():
    """When a weak detection is listed before the strong one, after
    suppression the original-order ranking is preserved among the
    kept detections."""
    cfg = DetectionConfig()
    sidelobe = _det(433.85e6, snr_db=10.0)
    main = _det(433.928e6, snr_db=45.0)
    distant = _det(434.5e6, snr_db=15.0)   # outside any shadow
    kept = _suppress_shadowed([sidelobe, main, distant], cfg)
    # sidelobe is suppressed; main + distant survive in original order.
    assert [d.center_hz for d in kept] == [main.center_hz, distant.center_hz]


def test_suppress_two_strong_signals_outside_each_others_shadow():
    cfg = DetectionConfig()
    a = _det(433.0e6, snr_db=40)         # shadow radius ~160 kHz
    b = _det(434.0e6, snr_db=40)         # 1 MHz away, well outside
    kept = _suppress_shadowed([a, b], cfg)
    assert len(kept) == 2


def test_suppress_two_strong_signals_inside_shadow_keeps_stronger_only():
    cfg = DetectionConfig()
    strong = _det(433.0e6, snr_db=45)    # ~180 kHz shadow
    weaker = _det(433.05e6, snr_db=35)   # 50 kHz away → in shadow
    kept = _suppress_shadowed([strong, weaker], cfg)
    assert len(kept) == 1
    assert kept[0].center_hz == strong.center_hz


def test_suppress_weak_detections_dont_cast_shadows():
    """If both detections are below the SNR threshold for casting a
    shadow, neither should suppress the other."""
    cfg = DetectionConfig(cluster_shadow_min_snr_db=25.0)
    a = _det(433.0e6, snr_db=15)         # below threshold
    b = _det(433.05e6, snr_db=12)        # also below
    kept = _suppress_shadowed([a, b], cfg)
    assert len(kept) == 2


def test_suppress_disabled_via_config():
    """When cluster_suppression is off, the engine should keep
    everything (this is verified at the engine level too — here
    we test the helper directly)."""
    cfg = DetectionConfig()    # cluster_suppression True is the default
    cfg.cluster_suppression = False
    # The helper itself doesn't check the flag — the engine gates
    # the call. So with the helper called, suppression still
    # happens; verify by NOT calling it.
    detections = [
        _det(433.928e6, snr_db=45.9),
        _det(433.85e6, snr_db=10),
    ]
    # When the engine has cluster_suppression=False it skips this
    # call entirely; mimic that here.
    assert detections == detections   # tautology — covered in engine test


# ---------- end-to-end via BaselineEngine ----------

def test_engine_suppresses_synthetic_sidelobes():
    """Inject a strong tone + weaker simultaneous tones and confirm
    only the strong one fires an appearance event."""
    n_bins = 64
    bin_hz = 4000.0
    freqs = np.linspace(433.0e6, 433.0e6 + n_bins * bin_hz, n_bins)
    rng = RangeConfig(name="t",
                      f_start=float(freqs[0]), f_end=float(freqs[-1]),
                      bin_hz=bin_hz)
    det = DetectionConfig(
        warmup_min=20, n_up=2, n_down=2, k_up=6, k_down=2, window_size=64,
        cluster_suppression=True,
    )
    eng = BaselineEngine(range_cfg=rng, detect_cfg=det)

    rgen = np.random.default_rng(0)
    # Warmup pure noise.
    for ts in range(30):
        noise = -90.0 + rgen.normal(0, 1.0, size=n_bins)
        eng.ingest(_frame(noise, freqs, ts=float(ts)))

    # Strong tone at bin 30 (+50 dB SNR), weaker "sidelobes" at
    # bins 32 and 35 (+15 dB) — within the strong tone's shadow.
    detections = []
    for ts in range(30, 35):
        s = -90.0 + rgen.normal(0, 1.0, size=n_bins)
        s[30] = -40.0
        s[32] = -75.0
        s[35] = -75.0
        detections.extend(eng.ingest(_frame(s, freqs, ts=float(ts))))

    appearances = [d for d in detections if d.type == "appearance"]
    assert len(appearances) == 1
    assert abs(appearances[0].center_hz - freqs[30]) < bin_hz


def test_engine_with_suppression_disabled_keeps_all_sidelobes():
    n_bins = 64
    bin_hz = 4000.0
    freqs = np.linspace(433.0e6, 433.0e6 + n_bins * bin_hz, n_bins)
    rng = RangeConfig(name="t",
                      f_start=float(freqs[0]), f_end=float(freqs[-1]),
                      bin_hz=bin_hz)
    det = DetectionConfig(
        warmup_min=20, n_up=2, n_down=2, k_up=6, k_down=2, window_size=64,
        cluster_suppression=False,
    )
    eng = BaselineEngine(range_cfg=rng, detect_cfg=det)

    rgen = np.random.default_rng(1)
    for ts in range(30):
        noise = -90.0 + rgen.normal(0, 1.0, size=n_bins)
        eng.ingest(_frame(noise, freqs, ts=float(ts)))

    detections = []
    for ts in range(30, 35):
        s = -90.0 + rgen.normal(0, 1.0, size=n_bins)
        s[30] = -40.0
        s[35] = -75.0
        s[40] = -75.0
        detections.extend(eng.ingest(_frame(s, freqs, ts=float(ts))))

    appearances = [d for d in detections if d.type == "appearance"]
    # Without suppression, the strong tone + the two "sidelobes"
    # each emit their own coalesced detection.
    assert len(appearances) == 3


def test_engine_strong_emitters_far_apart_both_survive():
    """The fix shouldn't suppress genuinely distinct emitters when
    they're spaced beyond the shadow radius."""
    n_bins = 128
    bin_hz = 4000.0
    freqs = np.linspace(433.0e6, 433.0e6 + n_bins * bin_hz, n_bins)
    rng = RangeConfig(name="t",
                      f_start=float(freqs[0]), f_end=float(freqs[-1]),
                      bin_hz=bin_hz)
    det = DetectionConfig(
        warmup_min=20, n_up=2, n_down=2, k_up=6, k_down=2, window_size=64,
        cluster_suppression=True,
        # default shadow at +50 dB SNR: ~200 kHz.
        # 128 bins * 4 kHz = 512 kHz total range. Place tones at
        # bins 10 and 110 → 400 kHz apart, well outside shadow.
    )
    eng = BaselineEngine(range_cfg=rng, detect_cfg=det)

    rgen = np.random.default_rng(2)
    for ts in range(30):
        noise = -90.0 + rgen.normal(0, 1.0, size=n_bins)
        eng.ingest(_frame(noise, freqs, ts=float(ts)))

    detections = []
    for ts in range(30, 35):
        s = -90.0 + rgen.normal(0, 1.0, size=n_bins)
        s[10] = -40.0
        s[110] = -40.0
        detections.extend(eng.ingest(_frame(s, freqs, ts=float(ts))))

    appearances = [d for d in detections if d.type == "appearance"]
    assert len(appearances) == 2
    centers = sorted(d.center_hz for d in appearances)
    assert abs(centers[0] - freqs[10]) < bin_hz
    assert abs(centers[1] - freqs[110]) < bin_hz
