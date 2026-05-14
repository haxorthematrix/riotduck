"""Filesystem layout for I/Q captures.

Captures are written as complex float-32 (`.cf32`) — rtl_433 detects
the format from the extension, and it's the native dtype for our
SoapySDR/pyrtlsdr backends so no conversion is needed.

Path scheme:

    <captures_dir>/YYYY-MM-DD/<detection_id>.cf32

Day partitioning keeps directory sizes sane and makes retention pruning
a simple per-directory operation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def capture_path(captures_dir: Path | str, detection_id: str, ts: float) -> Path:
    """Compute (and ensure) the path for an I/Q capture file."""
    base = Path(captures_dir)
    day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    day_dir = base / day
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{detection_id}.cf32"


def write_iq_cf32(path: Path | str, iq: np.ndarray) -> int:
    """Write I/Q samples as complex64. Returns number of samples written."""
    arr = np.asarray(iq, dtype=np.complex64)
    arr.tofile(str(path))
    return arr.size


def read_iq_cf32(path: Path | str) -> np.ndarray:
    """Read a complex64 I/Q file."""
    return np.fromfile(str(path), dtype=np.complex64)
