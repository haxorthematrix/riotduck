"""Persistent baselines.

Each (range, device) pair gets its own `<range>__<device>.npz` file
under `storage.baselines_dir`. On startup the runner loads it if it's
younger than `baseline_max_age_s`; otherwise it's ignored and the
engine warms up from scratch.

Layout:

    <baselines_dir>/<range_name>__<device_serial>.npz

The double-underscore separator is unambiguous since neither range
names nor SoapySDR device serials contain `__` in practice.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from riotduck.baseline import BaselineEngine


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(s: str) -> str:
    """Coerce a name to a filesystem-safe slug."""
    return _SAFE.sub("_", s) or "unnamed"


def baseline_path(
    baselines_dir: Path | str, range_name: str, device_serial: str
) -> Path:
    """Compute the on-disk path for a (range, device) baseline."""
    base = Path(baselines_dir)
    return base / f"{_safe(range_name)}__{_safe(device_serial)}.npz"


def save_baseline(
    path: Path | str, engine: "BaselineEngine"
) -> Path | None:
    """Persist an engine's snapshot to disk. Returns None if the engine
    has never ingested a frame (nothing useful to save).
    """
    if not engine._initialized:
        return None
    snap = engine.snapshot()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # `np.savez_compressed` writes atomically-ish via a temp file
    # internally on most platforms; for our purposes (one writer,
    # readable on next startup) that's fine.
    np.savez_compressed(str(p), **snap)
    return p


def load_baseline(
    path: Path | str, max_age_s: float | None = None
) -> dict[str, np.ndarray] | None:
    """Read a baseline snapshot. Returns None if the file is missing,
    corrupt, or older than `max_age_s` (when set).
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with np.load(str(p)) as data:
            snap = {k: np.array(data[k]) for k in data.files}
    except (OSError, ValueError, EOFError) as e:
        logger.warning("baseline file unreadable: {} ({})", p, e)
        return None

    if max_age_s is not None and max_age_s > 0:
        ts = snap.get("ts")
        if ts is None or ts.size == 0:
            logger.warning("baseline file has no timestamp; ignoring: {}", p)
            return None
        age = time.time() - float(ts.flatten()[0])
        if age > max_age_s:
            logger.info(
                "baseline file too old ({:.0f}s > {:.0f}s); skipping: {}",
                age, max_age_s, p,
            )
            return None
    return snap
