"""Event types carried over the bus.

All inter-agent messages are dataclasses defined here so consumers can
import them without circulars. Topics are strings: see EventBus docs.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DetectionKind = Literal["appearance", "disappearance"]


def _now() -> float:
    return time.time()


def _uuid() -> str:
    return uuid.uuid4().hex


@dataclass
class SweepFrame:
    """Power-spectral-density frame for a single range, single sweep.

    `power_dbfs` is a 1-D array of length `len(freqs_hz)`. Bin n is
    centered at `freqs_hz[n]`. Units are dBFS (relative to ADC
    full-scale); calibration to absolute dBm is out of scope here.
    """

    range_name: str
    device_serial: str
    ts: float = field(default_factory=_now)
    freqs_hz: np.ndarray = field(default_factory=lambda: np.empty(0))
    power_dbfs: np.ndarray = field(default_factory=lambda: np.empty(0))
    bin_hz: float = 0.0


@dataclass
class Detection:
    id: str
    type: DetectionKind
    ts: float
    range_name: str
    device_serial: str
    center_hz: float
    bw_hz: float
    power_dbfs: float
    snr_db: float
    bins: list[int]
    first_seen_ts: float
    last_seen_ts: float
    iq_path: str | None = None

    @classmethod
    def new(cls, **kw) -> "Detection":
        kw.setdefault("id", _uuid())
        kw.setdefault("ts", _now())
        return cls(**kw)


@dataclass
class CaptureRequest:
    detection_id: str
    range_name: str
    center_hz: float
    bw_hz: float
    capture_ms: float
    device_hint: str | None = None    # serial of a preferred device


@dataclass
class CaptureResult:
    detection_id: str
    path: str
    samp_rate: float
    center_hz: float
    duration_s: float
    ts: float = field(default_factory=_now)


@dataclass
class Identification:
    detection_id: str
    source: Literal["rtl_433", "urh", "manual"]
    device_class: str | None
    decoded: dict
    confidence: float
    ts: float = field(default_factory=_now)


@dataclass
class AnalysisReport:
    detection_id: str
    modulation: str | None
    symbol_rate_hz: float | None
    bw_3db_hz: float | None
    bw_6db_hz: float | None
    bw_20db_hz: float | None
    notes: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    ts: float = field(default_factory=_now)


@dataclass
class DeviceEvent:
    serial: str
    kind: Literal["online", "offline", "acquired", "released", "error"]
    detail: str = ""
    ts: float = field(default_factory=_now)


# Topic constants — keep names in one place.
class Topics:
    SWEEP_FRAME = "sweep.frame"
    DETECTION = "detection"
    DETECTION_APPEARANCE = "detection.appearance"
    DETECTION_DISAPPEARANCE = "detection.disappearance"
    CAPTURE_REQUEST = "capture.request"
    CAPTURE_RESULT = "capture.result"
    JOB_FINGERPRINT = "job.fingerprint"
    JOB_ANALYZE = "job.analyze"
    IDENTIFICATION = "identification"
    ANALYSIS_REPORT = "analysis.report"
    DEVICE = "device"
    CONTROL = "control"
    ERROR = "error"
