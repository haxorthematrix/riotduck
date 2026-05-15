"""Sweep scanner.

Given an SDR session and a range, the scanner steps the tuner across
the range, accumulates FFT frames per dwell, decimates to the
configured bin width, and yields a single SweepFrame covering the
whole range.

Implementation notes:
- We work in `samp_rate * usable_fraction` chunks to avoid the spectral
  edges (anti-alias rolloff, DC spur, IQ image at the band edges).
- FFT size is chosen so the *native* bin width is <= the target bin
  width; we then decimate to the target. This keeps FFT size sane on
  wide ranges while still giving the user control over resolution.
- The scanner is sync. It is meant to be driven from an async agent
  that wraps blocking reads in `asyncio.to_thread`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from riotduck.config import RangeConfig
from riotduck.dsp import (
    average_frames,
    decimate_to_bin_width,
    fft_power_dbfs,
    make_window,
    usable_bw,
)
from riotduck.events import SweepFrame
from riotduck.sdr.base import SDRSession


@dataclass
class SweepPlan:
    range_cfg: RangeConfig
    samp_rate: float
    usable_bw_hz: float
    tune_points: list[float]
    fft_size: int
    frames_per_dwell: int
    window_kind: str

    @property
    def native_bin_hz(self) -> float:
        return self.samp_rate / self.fft_size


@dataclass
class TuneCapture:
    """Raw I/Q from one tune point of the most recent sweep.

    Retained by the Scanner so the agent can dump the *same samples
    that triggered a detection* on appearance, instead of retuning
    and re-reading 400 ms of post-burst noise. A burst that landed
    inside a sweep's dwell window is preserved here.
    """

    center_hz: float
    samp_rate: float
    usable_bw_hz: float
    iq: np.ndarray
    ts: float = field(default_factory=time.time)

    def covers(self, freq_hz: float) -> bool:
        return abs(freq_hz - self.center_hz) <= self.usable_bw_hz / 2.0

    @property
    def duration_s(self) -> float:
        if self.samp_rate <= 0:
            return 0.0
        return len(self.iq) / self.samp_rate


def plan_sweep(range_cfg: RangeConfig, supported_samp_rates: tuple[float, ...]) -> SweepPlan:
    """Build a SweepPlan for `range_cfg` given the device's supported rates."""
    # Pick the highest supported sample rate at or below the user request,
    # falling back to the lowest supported if the user asked for less.
    requested = range_cfg.samp_rate
    if requested is None:
        # default: max practical sample rate; pick the highest available
        sr = max(supported_samp_rates) if supported_samp_rates else 2.4e6
    else:
        sr = max((s for s in supported_samp_rates if s <= requested), default=min(supported_samp_rates))

    ub = usable_bw(sr, 0.75)

    # FFT size: smallest power of two whose native bin width is
    # <= configured bin_hz / 2. Decimation in dsp then collapses to bin_hz.
    target_native = range_cfg.bin_hz / 2.0
    fft_size = 256
    while sr / fft_size > target_native and fft_size < 65536:
        fft_size *= 2

    # Frames per dwell: enough samples for at least one full FFT plus
    # the user's dwell-time-worth of samples for averaging.
    n_samples = max(int(sr * range_cfg.dwell_ms / 1000.0), fft_size)
    frames_per_dwell = max(1, n_samples // fft_size)

    # Tune points: step across the range in usable_bw_hz strides.
    points: list[float] = []
    f = range_cfg.f_start + ub / 2.0
    while f - ub / 2.0 < range_cfg.f_end:
        points.append(f)
        f += ub
    if not points:
        points = [(range_cfg.f_start + range_cfg.f_end) / 2.0]

    return SweepPlan(
        range_cfg=range_cfg,
        samp_rate=sr,
        usable_bw_hz=ub,
        tune_points=points,
        fft_size=fft_size,
        frames_per_dwell=frames_per_dwell,
        window_kind=range_cfg.window,
    )


class Scanner:
    """One scanner is bound to one SDRSession.

    A scanner is not tied to a particular range; the same session can
    sweep multiple ranges if you call `sweep(range_cfg)` repeatedly.
    Sample rate / gain are re-applied per sweep so the session may be
    shared with another consumer between sweeps.
    """

    def __init__(self, session: SDRSession, retain_captures: bool = True) -> None:
        self.session = session
        self.retain_captures = retain_captures
        self.last_captures: list[TuneCapture] = []
        self._window_cache: dict[tuple[str, int], np.ndarray] = {}

    def find_capture_for_freq(self, freq_hz: float) -> TuneCapture | None:
        """Return the most-recent tune capture whose passband contains freq_hz."""
        candidates = [c for c in self.last_captures if c.covers(freq_hz)]
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(c.center_hz - freq_hz))

    def _get_window(self, kind: str, n: int) -> np.ndarray:
        key = (kind, n)
        w = self._window_cache.get(key)
        if w is None:
            w = make_window(kind, n)
            self._window_cache[key] = w
        return w

    def sweep(self, plan: SweepPlan) -> SweepFrame:
        """Perform one full sweep of the planned range. Blocking."""
        rng = plan.range_cfg
        sr = self.session.set_samp_rate(plan.samp_rate)
        if abs(sr - plan.samp_rate) > 1.0:
            logger.debug("samp_rate clamped to {} Hz (asked for {})", sr, plan.samp_rate)
        self.session.set_gain({k: v for k, v in rng.gain.model_dump().items() if v is not None})

        all_freqs: list[np.ndarray] = []
        all_power: list[np.ndarray] = []
        captures: list[TuneCapture] = []

        window = self._get_window(plan.window_kind, plan.fft_size)
        frame_samples = plan.fft_size

        for center in plan.tune_points:
            self.session.set_center_hz(center)
            need = frame_samples * plan.frames_per_dwell
            iq = self.session.read_iq(need)
            if len(iq) < frame_samples:
                logger.warning("short read at {} Hz: got {}/{}", center, len(iq), need)
                continue

            if self.retain_captures:
                # Retain a *copy* so the underlying read buffer can
                # be reused on the next iteration without aliasing.
                captures.append(
                    TuneCapture(
                        center_hz=float(center),
                        samp_rate=float(plan.samp_rate),
                        usable_bw_hz=float(plan.usable_bw_hz),
                        iq=np.asarray(iq, dtype=np.complex64).copy(),
                        ts=time.time(),
                    )
                )

            frames: list[np.ndarray] = []
            for k in range(plan.frames_per_dwell):
                chunk = iq[k * frame_samples : (k + 1) * frame_samples]
                if len(chunk) < frame_samples:
                    break
                frames.append(fft_power_dbfs(chunk, window))
            if not frames:
                continue
            psd = average_frames(frames)

            # FFT bin centers, then crop to the usable bandwidth.
            df = plan.samp_rate / plan.fft_size
            freqs = center + (np.arange(plan.fft_size) - plan.fft_size / 2) * df
            mask = np.abs(freqs - center) <= (plan.usable_bw_hz / 2.0)
            freqs = freqs[mask]
            psd = psd[mask]

            # Decimate to the user-requested bin width.
            freqs, psd = decimate_to_bin_width(freqs, psd, rng.bin_hz)
            all_freqs.append(freqs)
            all_power.append(psd)

        # Replace the prior sweep's retained buffers in one shot, so the
        # agent's lookups see a coherent set of tune captures.
        self.last_captures = captures

        if not all_freqs:
            return SweepFrame(
                range_name=rng.name,
                device_serial=self.session.info.serial,
                freqs_hz=np.empty(0),
                power_dbfs=np.empty(0),
                bin_hz=rng.bin_hz,
            )

        freqs = np.concatenate(all_freqs)
        power = np.concatenate(all_power)

        # Sort & de-overlap: adjacent tune points may overlap at the
        # crop boundary; sort then keep the first occurrence per bin
        # (preferring the bin closer to its tune-point center).
        order = np.argsort(freqs)
        freqs = freqs[order]
        power = power[order]
        # Collapse near-duplicate frequencies by binning to bin_hz grid.
        f0 = rng.f_start
        idx = np.round((freqs - f0) / rng.bin_hz).astype(int)
        # max-hold on duplicate bins is conservative for detection:
        # we'd rather see a transient than smooth it away.
        unique_idx, first = np.unique(idx, return_index=True)
        collapsed_power = np.full(unique_idx.shape, -200.0, dtype=np.float32)
        for j, ix in enumerate(idx):
            slot = np.searchsorted(unique_idx, ix)
            if power[j] > collapsed_power[slot]:
                collapsed_power[slot] = power[j]
        collapsed_freqs = f0 + unique_idx * rng.bin_hz

        return SweepFrame(
            range_name=rng.name,
            device_serial=self.session.info.serial,
            freqs_hz=collapsed_freqs,
            power_dbfs=collapsed_power,
            bin_hz=rng.bin_hz,
        )
