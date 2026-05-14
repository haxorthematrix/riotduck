"""RTL-SDR backend.

Two implementations:

- `SoapyRTLBackend`: SoapySDR-based, preferred. Used when SoapySDR is
  importable and reports `driver=rtlsdr` devices.
- `PyRTLBackend`: fallback using pyrtlsdr directly. Used when SoapySDR
  is not present.

The module exposes a single `RTLSDRBackend()` factory that picks the
best available implementation at instantiation time.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from riotduck.sdr.base import DeviceInfo, SDRBackend, SDRSession

# ---------- SoapySDR-backed implementation ----------

try:
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX  # type: ignore

    _HAS_SOAPY = True
except Exception:
    _HAS_SOAPY = False


class _SoapyRTLSession(SDRSession):
    def __init__(self, info: DeviceInfo, dev, stream) -> None:
        self._info = info
        self._dev = dev
        self._stream = stream

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def set_center_hz(self, hz: float) -> None:
        self._dev.setFrequency(SOAPY_SDR_RX, 0, float(hz))

    def set_samp_rate(self, sps: float) -> float:
        self._dev.setSampleRate(SOAPY_SDR_RX, 0, float(sps))
        return float(self._dev.getSampleRate(SOAPY_SDR_RX, 0))

    def set_gain(self, stages: dict[str, float | int]) -> None:
        for stage, value in stages.items():
            if value is None:
                continue
            try:
                # RTL-SDR exposes a single "TUNER" stage in Soapy.
                if stage in ("tuner", "TUNER"):
                    self._dev.setGain(SOAPY_SDR_RX, 0, float(value))
                else:
                    self._dev.setGain(SOAPY_SDR_RX, 0, stage, float(value))
            except Exception as e:
                logger.warning("RTL-SDR set_gain({}={}) failed: {}", stage, value, e)

    def read_iq(self, n_samples: int) -> np.ndarray:
        buf = np.empty(n_samples, dtype=np.complex64)
        total = 0
        # Soapy returns whatever is currently available; loop until full.
        while total < n_samples:
            chunk = buf[total:]
            sr = self._dev.readStream(self._stream, [chunk], len(chunk), timeoutUs=int(1e6))
            if sr.ret <= 0:
                logger.debug("readStream returned {} flags={}", sr.ret, sr.flags)
                break
            total += sr.ret
        return buf[:total]

    def close(self) -> None:
        try:
            self._dev.deactivateStream(self._stream)
            self._dev.closeStream(self._stream)
        except Exception:
            pass


class SoapyRTLBackend(SDRBackend):
    name = "soapy-rtlsdr"

    def discover(self) -> list[DeviceInfo]:
        out: list[DeviceInfo] = []
        if not _HAS_SOAPY:
            return out
        for entry in SoapySDR.Device.enumerate({"driver": "rtlsdr"}):
            serial = entry.get("serial") or entry.get("label", "")
            label = entry.get("label", "")
            out.append(
                DeviceInfo(
                    serial=serial,
                    type="rtlsdr",
                    label=label,
                    driver=self.name,
                    tuning_range_hz=(24e6, 1.766e9),
                    samp_rates=(225e3, 1.024e6, 1.4e6, 1.8e6, 1.92e6, 2.048e6, 2.4e6, 2.56e6),
                    gain_stages=("tuner",),
                    extra={"soapy_args": dict(entry)},
                )
            )
        return out

    def open(self, serial: str) -> SDRSession:
        if not _HAS_SOAPY:
            raise RuntimeError("SoapySDR not available")
        dev = SoapySDR.Device({"driver": "rtlsdr", "serial": serial})
        stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        dev.activateStream(stream)
        info = DeviceInfo(
            serial=serial,
            type="rtlsdr",
            label=f"rtlsdr:{serial}",
            driver=self.name,
            tuning_range_hz=(24e6, 1.766e9),
            samp_rates=(225e3, 1.024e6, 1.4e6, 1.8e6, 1.92e6, 2.048e6, 2.4e6, 2.56e6),
            gain_stages=("tuner",),
        )
        return _SoapyRTLSession(info, dev, stream)


# ---------- pyrtlsdr-backed fallback ----------

try:
    from rtlsdr import RtlSdr  # type: ignore

    _HAS_PYRTL = True
except Exception:
    _HAS_PYRTL = False


class _PyRTLSession(SDRSession):
    def __init__(self, info: DeviceInfo, dev) -> None:
        self._info = info
        self._dev = dev

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def set_center_hz(self, hz: float) -> None:
        self._dev.center_freq = float(hz)

    def set_samp_rate(self, sps: float) -> float:
        self._dev.sample_rate = float(sps)
        return float(self._dev.sample_rate)

    def set_gain(self, stages: dict[str, float | int]) -> None:
        val = stages.get("tuner")
        if val is None:
            self._dev.gain = "auto"
        else:
            self._dev.gain = float(val)

    def read_iq(self, n_samples: int) -> np.ndarray:
        samples = self._dev.read_samples(n_samples)
        # pyrtlsdr returns complex128; downcast for memory.
        return np.asarray(samples, dtype=np.complex64)

    def close(self) -> None:
        try:
            self._dev.close()
        except Exception:
            pass


class PyRTLBackend(SDRBackend):
    name = "pyrtlsdr"

    def discover(self) -> list[DeviceInfo]:
        out: list[DeviceInfo] = []
        if not _HAS_PYRTL:
            return out
        # pyrtlsdr's enumeration is index-only; we report by index.
        try:
            count = RtlSdr.get_device_count()  # type: ignore[attr-defined]
        except Exception:
            count = 0
        for i in range(count):
            try:
                serial = RtlSdr.get_device_serial_addresses()[i]  # type: ignore[attr-defined]
            except Exception:
                serial = str(i)
            out.append(
                DeviceInfo(
                    serial=serial,
                    type="rtlsdr",
                    label=f"rtlsdr#{i}",
                    driver=self.name,
                    tuning_range_hz=(24e6, 1.766e9),
                    samp_rates=(225e3, 1.024e6, 1.4e6, 1.8e6, 1.92e6, 2.048e6, 2.4e6, 2.56e6),
                    gain_stages=("tuner",),
                    extra={"index": i},
                )
            )
        return out

    def open(self, serial: str) -> SDRSession:
        if not _HAS_PYRTL:
            raise RuntimeError("pyrtlsdr not available")
        # pyrtlsdr accepts either an index or a serial string.
        try:
            dev = RtlSdr(serial_number=serial)
        except TypeError:
            # older pyrtlsdr versions don't have serial_number kwarg
            dev = RtlSdr(int(serial)) if serial.isdigit() else RtlSdr()
        info = DeviceInfo(
            serial=serial,
            type="rtlsdr",
            label=f"rtlsdr:{serial}",
            driver=self.name,
            tuning_range_hz=(24e6, 1.766e9),
            samp_rates=(225e3, 1.024e6, 1.4e6, 1.8e6, 1.92e6, 2.048e6, 2.4e6, 2.56e6),
            gain_stages=("tuner",),
        )
        return _PyRTLSession(info, dev)


def RTLSDRBackend() -> SDRBackend:
    """Pick the best available RTL-SDR backend."""
    if _HAS_SOAPY:
        return SoapyRTLBackend()
    if _HAS_PYRTL:
        return PyRTLBackend()
    raise RuntimeError(
        "No RTL-SDR backend available. Install SoapySDR (preferred) or pyrtlsdr."
    )


def rtlsdr_available() -> bool:
    return _HAS_SOAPY or _HAS_PYRTL
