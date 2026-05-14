"""Device manager: discovery, reservation, role tracking.

The manager is the single source of truth for which physical SDR is
currently held by which agent. Acquiring a device returns an
SDRSession; releasing it relinquishes the reservation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterable

from loguru import logger

from riotduck.config import DeviceConfig
from riotduck.sdr.base import DeviceInfo, SDRBackend, SDRSession
from riotduck.sdr.fake import FakeBackend, fake_available
from riotduck.sdr.hackrf import HackRFBackend, hackrf_available
from riotduck.sdr.rtlsdr import RTLSDRBackend, rtlsdr_available


@dataclass
class DeviceRecord:
    info: DeviceInfo
    role: str = "auto"
    antenna: str | None = None
    held_by: str | None = None
    backend: SDRBackend | None = None


class DeviceManager:
    def __init__(self) -> None:
        self._records: dict[str, DeviceRecord] = {}
        self._lock = threading.Lock()
        self._backends: list[SDRBackend] = self._build_backends()

    @staticmethod
    def _build_backends() -> list[SDRBackend]:
        backends: list[SDRBackend] = []
        if fake_available():
            backends.append(FakeBackend())
            logger.info("Fake SDR backend enabled via RIOTDUCK_FAKE_DEVICES")
        if rtlsdr_available():
            try:
                backends.append(RTLSDRBackend())
            except Exception as e:
                logger.warning("RTL-SDR backend init failed: {}", e)
        else:
            logger.debug("No RTL-SDR backend available (no SoapySDR, no pyrtlsdr)")
        if hackrf_available():
            backends.append(HackRFBackend())
        else:
            logger.debug("No HackRF backend available (no SoapySDR)")
        return backends

    def discover(self) -> list[DeviceRecord]:
        """Enumerate devices via all available backends."""
        seen: dict[str, DeviceRecord] = {}
        for backend in self._backends:
            try:
                infos = backend.discover()
            except Exception as e:
                logger.warning("discover() on {} failed: {}", backend.name, e)
                continue
            for info in infos:
                if info.serial in seen:
                    continue   # first backend wins (Soapy preferred)
                rec = DeviceRecord(info=info, backend=backend)
                seen[info.serial] = rec
        with self._lock:
            self._records = seen
        return list(seen.values())

    def apply_device_configs(self, configs: Iterable[DeviceConfig]) -> None:
        """Apply user-supplied roles/antennas to discovered devices."""
        with self._lock:
            for cfg in configs:
                rec = self._records.get(cfg.serial)
                if rec is None:
                    logger.warning("configured device serial {} not found", cfg.serial)
                    continue
                rec.role = cfg.role
                rec.antenna = cfg.antenna

    def list(self) -> list[DeviceRecord]:
        with self._lock:
            return list(self._records.values())

    def acquire(self, serial: str, holder: str) -> SDRSession:
        with self._lock:
            rec = self._records.get(serial)
            if rec is None:
                raise KeyError(f"unknown device serial: {serial}")
            if rec.held_by is not None:
                raise RuntimeError(f"device {serial} already held by {rec.held_by}")
            assert rec.backend is not None
            session = rec.backend.open(serial)
            rec.held_by = holder
            return session

    def release(self, serial: str, session: SDRSession | None = None) -> None:
        with self._lock:
            rec = self._records.get(serial)
            if rec is None:
                return
            rec.held_by = None
        if session is not None:
            try:
                session.close()
            except Exception as e:
                logger.warning("session.close() for {} raised: {}", serial, e)

    def find_idle(self, role: str | None = None) -> DeviceRecord | None:
        with self._lock:
            for rec in self._records.values():
                if rec.held_by is not None:
                    continue
                if role is not None and rec.role not in (role, "auto"):
                    continue
                return rec
            return None
