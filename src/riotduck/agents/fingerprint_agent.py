"""Fingerprint agent: subscribes to capture.result, runs rtl_433.

Per CaptureResult:
  1. Run rtl_433 against the I/Q file (file mode, with samp_rate +
     center hint).
  2. For each decoded packet, emit one Identification event tagged
     with the original detection_id.
  3. On zero hits, emit a single "no match" Identification with
     source=rtl_433, device_class=None, confidence=0.0 so downstream
     analyzers know the fingerprinter ran but came up empty.

URH integration is gated on the rtl_433 result: when there are no
rtl_433 hits and `urh.enabled`, the agent will fall through to URH
(stubbed for Phase 3).
"""

from __future__ import annotations

import asyncio

from loguru import logger

from riotduck.agents.base import Agent
from riotduck.bus import EventBus
from riotduck.config import IdentificationConfig
from riotduck.events import CaptureResult, Identification, Topics
from riotduck.fingerprint.rtl_433 import run_on_file


class FingerprintAgent(Agent):
    name = "fingerprint"

    def __init__(self, bus: EventBus, id_cfg: IdentificationConfig) -> None:
        super().__init__(bus)
        self.id_cfg = id_cfg

    async def run(self) -> None:
        async with self.bus.subscribe(Topics.CAPTURE_RESULT) as sub:
            while not self.should_stop():
                try:
                    topic, payload = await sub.queue.get()
                except asyncio.CancelledError:
                    return
                if not isinstance(payload, CaptureResult):
                    continue
                try:
                    await self._handle(payload)
                except Exception as e:
                    logger.exception("fingerprint handler failed: {}", e)

    async def _handle(self, cap: CaptureResult) -> None:
        if self.id_cfg.rtl_433.enabled:
            await self._run_rtl_433(cap)
        else:
            logger.debug("rtl_433 disabled; skipping fingerprint for {}", cap.detection_id)

    async def _run_rtl_433(self, cap: CaptureResult) -> None:
        binary = self.id_cfg.rtl_433.binary or "rtl_433"
        result = await asyncio.to_thread(
            run_on_file,
            cap.path,
            cap.samp_rate,
            binary,
            cap.center_hz,
            tuple(self.id_cfg.rtl_433.extra_args),
        )
        # Capture metadata downstream consumers (analyzer agent) need
        # in order to load the I/Q without a sidecar file.
        cap_meta = {
            "iq_path": cap.path,
            "samp_rate": cap.samp_rate,
            "center_hz": cap.center_hz,
            "duration_s": cap.duration_s,
        }
        if result.hits:
            for hit in result.hits:
                ident = Identification(
                    detection_id=cap.detection_id,
                    source="rtl_433",
                    device_class=hit.model,
                    decoded={**hit.decoded, "_capture": cap_meta},
                    confidence=hit.confidence,
                )
                await self.bus.publish(Topics.IDENTIFICATION, ident)
        else:
            ident = Identification(
                detection_id=cap.detection_id,
                source="rtl_433",
                device_class=None,
                decoded={
                    "returncode": result.returncode,
                    "stderr_tail": result.stderr[-500:],
                    "_capture": cap_meta,
                },
                confidence=0.0,
            )
            await self.bus.publish(Topics.IDENTIFICATION, ident)
            if self.id_cfg.urh.enabled:
                logger.debug("rtl_433 miss for {}; URH fallback not yet implemented",
                             cap.detection_id)
