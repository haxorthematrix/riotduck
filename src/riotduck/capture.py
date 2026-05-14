"""Inline I/Q capture for a single detection.

Re-uses the SDR session already held by a scanner agent: stops sweeping
briefly, retunes to the detection center, sets a sample rate
appropriate for the detection, reads `capture_ms` of samples, and
writes them as `.cf32`. Caller is responsible for restoring the
prior tuning state if needed (the next sweep iteration does this on
its own).

Sample-rate selection:
- Prefer the range's configured `samp_rate` (already validated to be
  a device-supported rate during sweep planning) — this avoids
  triggering a HackRF filter reconfiguration mid-sweep.
- Fall back to the lowest supported rate that's at least the detection
  bandwidth * 4 (Nyquist + headroom for OOK/FSK preambles).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from riotduck.events import CaptureResult, Detection
from riotduck.sdr.base import SDRSession
from riotduck.storage.files import capture_path, write_iq_cf32


def choose_capture_samp_rate(
    detection: Detection,
    range_samp_rate: float | None,
    supported_rates: tuple[float, ...],
) -> float:
    """Pick a sample rate for the capture step."""
    if not supported_rates:
        return range_samp_rate or 2.4e6
    if range_samp_rate is not None:
        return range_samp_rate
    needed = max(detection.bw_hz * 4.0, 250e3)
    feasible = [s for s in supported_rates if s >= needed]
    if feasible:
        return min(feasible)
    return max(supported_rates)


def capture_for_detection(
    session: SDRSession,
    detection: Detection,
    captures_dir: Path | str,
    capture_ms: float,
    samp_rate: float,
    gain: dict[str, float | int] | None = None,
) -> CaptureResult | None:
    """Tune, read, and write a single I/Q capture. Returns CaptureResult.

    Returns None if the read produces no samples (e.g., USB stall).
    Exceptions propagate so the caller can decide whether to retry.
    """
    sr_actual = session.set_samp_rate(samp_rate)
    session.set_center_hz(detection.center_hz)
    if gain:
        session.set_gain({k: v for k, v in gain.items() if v is not None})

    n_samples = max(int(sr_actual * capture_ms / 1000.0), 1024)
    iq = session.read_iq(n_samples)
    if iq is None or len(iq) == 0:
        logger.warning("capture: empty read for detection {}", detection.id)
        return None

    path = capture_path(captures_dir, detection.id, detection.ts)
    n_written = write_iq_cf32(path, iq)
    duration_s = n_written / sr_actual if sr_actual > 0 else 0.0
    logger.info(
        "capture: {} samples ({:.3f} s @ {:.3f} MS/s) → {}",
        n_written,
        duration_s,
        sr_actual / 1e6,
        path,
    )
    return CaptureResult(
        detection_id=detection.id,
        path=str(path),
        samp_rate=float(sr_actual),
        center_hz=float(detection.center_hz),
        duration_s=duration_s,
    )
