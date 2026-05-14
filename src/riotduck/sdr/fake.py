"""Synthetic SDR backend.

Useful for end-to-end testing and demos without real hardware. Emits
an additive-noise channel plus a configurable population of emitters:

- ``carrier``  — steady complex exponential at frequency ``hz``,
  amplitude ``amp`` (linear, 1.0 = full scale).
- ``burst``    — same as carrier but gated on/off on a duty cycle
  with ``period_s`` and ``duty``.
- ``drift``    — carrier whose frequency walks linearly between
  ``hz`` and ``hz + drift_hz`` over ``drift_period_s``, then wraps.

Enable with the environment variable ``RIOTDUCK_FAKE_DEVICES`` set to
the number of fake devices (default off when unset). When set, the
``DeviceManager`` will discover ``fake-0001``..``fake-NNNN``.

Profiles
--------

The emitter population is read from a YAML file pointed to by
``RIOTDUCK_FAKE_PROFILE``. When unset, a built-in default profile is
used (a 433 MHz keyfob-shaped burst, a steady 915 MHz LoRa-shaped
carrier, and a 2.4 GHz wandering drone-video-shaped drifter).

The fake backend uses host wall-clock time to drive emitter
schedules, so re-observations and disappearances behave as they would
with real hardware (modulo noise statistics).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from loguru import logger

from riotduck.sdr.base import DeviceInfo, SDRBackend, SDRSession


EmitterKind = Literal["carrier", "burst", "drift"]


@dataclass
class Emitter:
    kind: EmitterKind
    hz: float
    amp: float = 0.5
    # burst-only
    period_s: float = 1.0
    duty: float = 0.05
    phase_s: float = 0.0
    # drift-only
    drift_hz: float = 0.0
    drift_period_s: float = 30.0

    def active_at(self, t: float) -> bool:
        if self.kind != "burst":
            return True
        phase = ((t - self.phase_s) % self.period_s) / self.period_s
        return phase < self.duty

    def freq_at(self, t: float) -> float:
        if self.kind != "drift":
            return self.hz
        # Triangle wander: 0..1..0 over drift_period_s.
        frac = (t % self.drift_period_s) / self.drift_period_s
        tri = 2.0 * frac if frac < 0.5 else 2.0 * (1.0 - frac)
        return self.hz + self.drift_hz * tri


DEFAULT_PROFILE: list[Emitter] = [
    # Keyfob-shaped 50 ms burst every ~3 s at 433.92 MHz, ASK style.
    Emitter(kind="burst", hz=433.920e6, amp=0.6, period_s=3.0, duty=0.02),
    # Steady LoRa-ish carrier at 915 MHz.
    Emitter(kind="carrier", hz=915.000e6, amp=0.3),
    # Wandering FPV video transmitter around 2440 MHz.
    Emitter(kind="drift", hz=2.440e9, amp=0.2, drift_hz=4e6, drift_period_s=20.0),
]


def load_profile(path: str | Path | None = None) -> list[Emitter]:
    """Load emitter list from YAML or return the built-in default."""
    if path is None:
        env = os.environ.get("RIOTDUCK_FAKE_PROFILE")
        path = Path(env) if env else None
    if path is None:
        return list(DEFAULT_PROFILE)
    p = Path(path)
    if not p.exists():
        logger.warning("fake profile {} not found; using defaults", p)
        return list(DEFAULT_PROFILE)
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    out: list[Emitter] = []
    for e in raw.get("emitters", []):
        out.append(Emitter(**e))
    return out


def _fake_device_count() -> int:
    raw = os.environ.get("RIOTDUCK_FAKE_DEVICES", "")
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


class _FakeSession(SDRSession):
    """One open fake-SDR session."""

    SUPPORTED_RATES: tuple[float, ...] = (
        250e3, 1.024e6, 2.048e6, 2.4e6, 8e6, 20e6,
    )
    TUNING_RANGE_HZ: tuple[float, float] = (1e6, 6e9)
    NOISE_FLOOR: float = 1e-4    # linear, ≈ -80 dBFS for unit-amplitude

    def __init__(self, info: DeviceInfo, emitters: list[Emitter]) -> None:
        self._info = info
        self._sr = 2.4e6
        self._center = 100e6
        self._gain: dict[str, float | int] = {}
        self._emitters = emitters
        self._rng = np.random.default_rng()
        # Wall-clock t0 so multiple sessions share an emitter schedule.
        self._t0 = time.monotonic()

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def set_center_hz(self, hz: float) -> None:
        self._center = float(hz)

    def set_samp_rate(self, sps: float) -> float:
        # Snap to the closest supported rate.
        nearest = min(self.SUPPORTED_RATES, key=lambda x: abs(x - sps))
        self._sr = float(nearest)
        return self._sr

    def set_gain(self, stages: dict[str, float | int]) -> None:
        self._gain = dict(stages)

    def read_iq(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            return np.empty(0, dtype=np.complex64)
        sr = self._sr
        t_start = time.monotonic() - self._t0
        t = t_start + np.arange(n_samples, dtype=np.float64) / sr

        out = np.zeros(n_samples, dtype=np.complex128)

        usable_half = sr * 0.45    # match scanner's usable-BW crop
        for em in self._emitters:
            # Active at midpoint of this read window — cheap gate that
            # correctly captures bursts straddling the read boundary.
            mid_t = float(t[n_samples // 2])
            if not em.active_at(mid_t):
                continue
            f = em.freq_at(mid_t)
            offset = f - self._center
            if abs(offset) > usable_half:
                continue
            # Modest tuner-gain coupling so set_gain has *some* effect.
            tuner_gain_db = float(self._gain.get("tuner", 28))
            amp_factor = 10.0 ** ((tuner_gain_db - 28.0) / 20.0)
            phase = 2j * np.pi * offset * t
            out += em.amp * amp_factor * np.exp(phase)

        # Complex Gaussian noise floor.
        sigma = np.sqrt(self.NOISE_FLOOR / 2.0)
        noise = (self._rng.standard_normal(n_samples) + 1j * self._rng.standard_normal(n_samples)) * sigma
        out += noise

        # Block until we'd have actually read these samples — keeps
        # sweep cadence realistic and lets the asyncio loop breathe.
        target_dur = n_samples / sr
        elapsed = time.monotonic() - self._t0 - t_start
        if elapsed < target_dur:
            time.sleep(target_dur - elapsed)

        return out.astype(np.complex64)

    def close(self) -> None:
        pass


class FakeBackend(SDRBackend):
    """SDR backend that materializes from env vars / config."""

    name = "fake"

    def __init__(self, n_devices: int | None = None,
                 emitters: list[Emitter] | None = None) -> None:
        self._n = _fake_device_count() if n_devices is None else n_devices
        self._emitters = emitters if emitters is not None else load_profile()

    def discover(self) -> list[DeviceInfo]:
        out: list[DeviceInfo] = []
        for i in range(self._n):
            serial = f"fake-{i+1:04d}"
            out.append(
                DeviceInfo(
                    serial=serial,
                    type="fake",
                    label=f"FakeSDR #{i+1}",
                    driver=self.name,
                    tuning_range_hz=_FakeSession.TUNING_RANGE_HZ,
                    samp_rates=_FakeSession.SUPPORTED_RATES,
                    gain_stages=("tuner",),
                )
            )
        return out

    def open(self, serial: str) -> SDRSession:
        info = DeviceInfo(
            serial=serial,
            type="fake",
            label=f"FakeSDR:{serial}",
            driver=self.name,
            tuning_range_hz=_FakeSession.TUNING_RANGE_HZ,
            samp_rates=_FakeSession.SUPPORTED_RATES,
            gain_stages=("tuner",),
        )
        return _FakeSession(info, self._emitters)


def fake_available() -> bool:
    return _fake_device_count() > 0
