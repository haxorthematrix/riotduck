"""Abstract SDR backend interface.

Backends are thin adapters around device-specific I/O. They expose a
narrow surface: tune, set sample rate, set gain, stream I/Q. Sweep
strategy, FFT, baselining all live above this layer.

The interface is sync (blocking read_iq) but is consumed from an
asyncio-friendly wrapper that runs reads in a worker thread; trying to
push asyncio down into device drivers fights the underlying libraries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class DeviceInfo:
    serial: str
    type: str                  # "rtlsdr", "hackrf", ...
    label: str = ""
    driver: str = ""           # backend driver name (e.g. "soapy", "pyrtlsdr")
    tuning_range_hz: tuple[float, float] = (0.0, 0.0)
    samp_rates: tuple[float, ...] = ()
    gain_stages: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)


class SDRSession(ABC):
    """An open, tuned session against a single SDR.

    Sessions are obtained from a backend via `open()` and released by
    closing them. A session is single-consumer: do not share between
    threads/tasks.
    """

    @property
    @abstractmethod
    def info(self) -> DeviceInfo: ...

    @abstractmethod
    def set_center_hz(self, hz: float) -> None: ...

    @abstractmethod
    def set_samp_rate(self, sps: float) -> float:
        """Set sample rate; returns the rate actually programmed."""

    @abstractmethod
    def set_gain(self, stages: dict[str, float | int]) -> None: ...

    @abstractmethod
    def read_iq(self, n_samples: int) -> np.ndarray:
        """Block until n_samples of complex64 I/Q are available.

        Returns a complex64 ndarray of length n_samples. May return
        fewer on error/timeout (caller should validate).
        """

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "SDRSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SDRBackend(ABC):
    """Backend implementations enumerate and open devices."""

    name: str = "abstract"

    @abstractmethod
    def discover(self) -> list[DeviceInfo]: ...

    @abstractmethod
    def open(self, serial: str) -> SDRSession: ...

    @contextmanager
    def session(self, serial: str) -> Iterator[SDRSession]:
        s = self.open(serial)
        try:
            yield s
        finally:
            s.close()
