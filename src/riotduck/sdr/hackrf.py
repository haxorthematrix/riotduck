"""HackRF backend (SoapySDR-based).

This is a thin sibling of the SoapyRTLBackend. Phase 3 in the spec
fleshes out the `hackrf_sweep` fast path; for v1 we just expose the
streaming interface so wide ranges still work (just slower).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from riotduck.sdr.base import DeviceInfo, SDRBackend, SDRSession

try:
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX  # type: ignore

    _HAS_SOAPY = True
except Exception:
    _HAS_SOAPY = False


_HACKRF_GAIN_STAGES = ("LNA", "VGA", "AMP")


class _SoapyHackRFSession(SDRSession):
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
        # HackRF's filter follows sample rate; pin BW just below.
        try:
            self._dev.setBandwidth(SOAPY_SDR_RX, 0, float(sps) * 0.75)
        except Exception:
            pass
        return float(self._dev.getSampleRate(SOAPY_SDR_RX, 0))

    def set_gain(self, stages: dict[str, float | int]) -> None:
        # Map user-friendly names to HackRF Soapy stage names.
        aliases = {"lna": "LNA", "vga": "VGA", "amp": "AMP"}
        for k, v in stages.items():
            if v is None:
                continue
            soapy_stage = aliases.get(k.lower(), k.upper())
            if soapy_stage not in _HACKRF_GAIN_STAGES:
                continue
            try:
                self._dev.setGain(SOAPY_SDR_RX, 0, soapy_stage, float(v))
            except Exception as e:
                logger.warning("HackRF set_gain({}={}) failed: {}", soapy_stage, v, e)

    def read_iq(self, n_samples: int) -> np.ndarray:
        buf = np.empty(n_samples, dtype=np.complex64)
        total = 0
        while total < n_samples:
            chunk = buf[total:]
            sr = self._dev.readStream(self._stream, [chunk], len(chunk), timeoutUs=int(1e6))
            if sr.ret <= 0:
                logger.debug("HackRF readStream returned {} flags={}", sr.ret, sr.flags)
                break
            total += sr.ret
        return buf[:total]

    def close(self) -> None:
        try:
            self._dev.deactivateStream(self._stream)
            self._dev.closeStream(self._stream)
        except Exception:
            pass


class HackRFBackend(SDRBackend):
    name = "soapy-hackrf"

    def discover(self) -> list[DeviceInfo]:
        out: list[DeviceInfo] = []
        if not _HAS_SOAPY:
            return out
        for entry in SoapySDR.Device.enumerate({"driver": "hackrf"}):
            serial = entry.get("serial") or entry.get("label", "")
            label = entry.get("label", "")
            out.append(
                DeviceInfo(
                    serial=serial,
                    type="hackrf",
                    label=label,
                    driver=self.name,
                    tuning_range_hz=(1e6, 6e9),
                    samp_rates=(2e6, 4e6, 8e6, 10e6, 12.5e6, 16e6, 20e6),
                    gain_stages=_HACKRF_GAIN_STAGES,
                    extra={"soapy_args": dict(entry)},
                )
            )
        return out

    def open(self, serial: str) -> SDRSession:
        if not _HAS_SOAPY:
            raise RuntimeError("SoapySDR not available")
        dev = SoapySDR.Device({"driver": "hackrf", "serial": serial})
        stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        dev.activateStream(stream)
        info = DeviceInfo(
            serial=serial,
            type="hackrf",
            label=f"hackrf:{serial}",
            driver=self.name,
            tuning_range_hz=(1e6, 6e9),
            samp_rates=(2e6, 4e6, 8e6, 10e6, 12.5e6, 16e6, 20e6),
            gain_stages=_HACKRF_GAIN_STAGES,
        )
        return _SoapyHackRFSession(info, dev, stream)


def hackrf_available() -> bool:
    return _HAS_SOAPY
