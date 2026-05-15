"""Filesystem layout for I/Q captures.

Captures are written as complex float-32 (`.cf32`) — rtl_433 detects
the format from the extension, and it's the native dtype for our
SoapySDR/pyrtlsdr backends so no conversion is needed.

A `.meta.json` sidecar lives next to each capture with metadata that
isn't recoverable from the raw I/Q bytes: sample rate, tune center
frequency, originating detection's SNR / bandwidth, etc. Offline
`riotduck analyze` and `library add --from-capture` read it.

Path scheme:

    <captures_dir>/YYYY-MM-DD/<detection_id>.cf32
    <captures_dir>/YYYY-MM-DD/<detection_id>.meta.json

Day partitioning keeps directory sizes sane and makes retention pruning
a simple per-directory operation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

META_SCHEMA_VERSION = 1


def capture_path(captures_dir: Path | str, detection_id: str, ts: float) -> Path:
    """Compute (and ensure) the path for an I/Q capture file."""
    base = Path(captures_dir)
    day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    day_dir = base / day
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{detection_id}.cf32"


def sidecar_path_for(iq_path: Path | str) -> Path:
    """Return the .meta.json path that should sit next to an .cf32 file."""
    p = Path(iq_path)
    return p.with_suffix(".meta.json")


def write_iq_cf32(path: Path | str, iq: np.ndarray) -> int:
    """Write I/Q samples as complex64. Returns number of samples written."""
    arr = np.asarray(iq, dtype=np.complex64)
    arr.tofile(str(path))
    return arr.size


def read_iq_cf32(path: Path | str) -> np.ndarray:
    """Read a complex64 I/Q file."""
    return np.fromfile(str(path), dtype=np.complex64)


def write_capture_meta(
    iq_path: Path | str,
    *,
    detection,
    samp_rate: float,
    capture_center_hz: float,
    duration_s: float,
) -> Path:
    """Write a .meta.json sidecar for a capture.

    `detection` is a Detection dataclass instance; its fields are
    selectively projected into the sidecar JSON so the recipient can
    reconstruct who / what / when / where for the capture without
    needing the riotduck Python types installed.
    """
    sidecar = sidecar_path_for(iq_path)

    det_block: dict[str, Any] = {}
    if detection is not None:
        # Best-effort dump that doesn't require detection to be a
        # specific class — works for our Detection dataclass and for
        # plain dicts produced in tests.
        if is_dataclass(detection):
            raw = asdict(detection)
        elif hasattr(detection, "__dict__"):
            raw = dict(detection.__dict__)
        else:
            raw = dict(detection)
        # Project a tight, documented subset; ignore the rest so
        # additions to Detection don't pollute the on-disk schema.
        for k in (
            "id", "type", "ts", "range_name", "device_serial",
            "center_hz", "bw_hz", "power_dbfs", "snr_db",
            "first_seen_ts", "last_seen_ts",
        ):
            if k in raw and raw[k] is not None:
                det_block[k] = raw[k]

    meta = {
        "schema_version": META_SCHEMA_VERSION,
        "samp_rate": float(samp_rate),
        "capture_center_hz": float(capture_center_hz),
        "duration_s": float(duration_s),
        "iq_path": str(Path(iq_path).name),
        "detection": det_block,
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    return sidecar


def read_capture_meta(iq_path: Path | str) -> dict | None:
    """Read the .meta.json sidecar for a capture, or None if absent."""
    p = sidecar_path_for(iq_path)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
