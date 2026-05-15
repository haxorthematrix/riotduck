"""Unknown-signal analyzer agent.

Subscribes to `identification` events. When rtl_433 reports a miss
(device_class is None, confidence == 0.0) AND the corresponding
capture file exists, loads the I/Q and runs the heuristic classifier.
Publishes an `AnalysisReport` on the bus.

The agent is independent of rtl_433 — when URH integration lands it
will gate on URH misses too. Today rtl_433 is the only thing that
produces identification events.
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from riotduck.agents.base import Agent
from riotduck.analysis.classifier import analyze
from riotduck.bus import EventBus
from riotduck.events import AnalysisReport, Identification, Topics
from riotduck.storage.files import read_iq_cf32


class AnalysisAgent(Agent):
    name = "analysis"

    def __init__(self, bus: EventBus) -> None:
        super().__init__(bus)
        # Track which detections we've already analyzed so an emitter
        # that re-fires within the dedup window doesn't get analyzed
        # repeatedly with the same data.
        self._seen: set[str] = set()

    async def run(self) -> None:
        async with self.bus.subscribe(Topics.IDENTIFICATION) as sub:
            while not self.should_stop():
                try:
                    topic, payload = await sub.queue.get()
                except asyncio.CancelledError:
                    return
                if not isinstance(payload, Identification):
                    continue
                # Only trigger on misses (sentinel events).
                if payload.device_class is not None or payload.confidence > 0:
                    continue
                if payload.detection_id in self._seen:
                    continue
                self._seen.add(payload.detection_id)
                try:
                    await self._handle(payload)
                except Exception as e:
                    logger.exception("analysis handler failed: {}", e)

    async def _handle(self, ident: Identification) -> None:
        cap_meta = (ident.decoded or {}).get("_capture") if isinstance(ident.decoded, dict) else None
        if not cap_meta:
            logger.debug("analysis: no capture metadata in {} ident", ident.detection_id[:8])
            return
        iq_path = cap_meta.get("iq_path")
        samp_rate = float(cap_meta.get("samp_rate") or 0.0)
        if not iq_path or samp_rate <= 0 or not os.path.exists(iq_path):
            logger.debug("analysis: capture file not found for {} at {}",
                         ident.detection_id[:8], iq_path)
            return

        result = await asyncio.to_thread(_load_and_analyze, iq_path, samp_rate)
        if result is None:
            return

        report = AnalysisReport(
            detection_id=ident.detection_id,
            artifacts={"iq_path": iq_path},
            **result.as_report_kwargs(),
        )
        await self.bus.publish(Topics.ANALYSIS_REPORT, report)
        logger.info(
            "analysis: det={} mod={}({:.2f}) bw3={} sr={} bursts={}",
            ident.detection_id[:8],
            result.modulation,
            result.modulation_confidence,
            f"{result.bw_3db_hz:.0f}Hz" if result.bw_3db_hz else "?",
            f"{result.symbol_rate_hz:.1f}Hz" if result.symbol_rate_hz else "?",
            len(result.bursts),
        )


def _load_and_analyze(iq_path: str, samp_rate: float):
    iq = read_iq_cf32(iq_path)
    if iq.size == 0:
        return None
    return analyze(iq, samp_rate)
