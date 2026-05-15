"""Unknown-signal analyzer agent.

Subscribes to `identification` events. When rtl_433 reports a miss
(device_class is None, confidence == 0.0) AND the corresponding
capture file exists, loads the I/Q and runs the heuristic classifier.
Publishes an `AnalysisReport` on the bus.

If a user-curated fingerprint Library is configured, the agent also
matches the analyzer's output against it. A hit becomes an
`Identification(source="library")` event — downstream consumers can
treat library hits and rtl_433 hits uniformly. A miss optionally
attaches a YAML candidate snippet to the report so the operator has
a one-step path to adding the signal to their library.
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from riotduck.agents.base import Agent
from riotduck.analysis.classifier import analyze
from riotduck.bus import EventBus
from riotduck.events import AnalysisReport, Identification, Topics
from riotduck.library import Library, suggest_yaml
from riotduck.storage.files import read_iq_cf32


class AnalysisAgent(Agent):
    name = "analysis"

    def __init__(
        self,
        bus: EventBus,
        library: Library | None = None,
        suggest_new: bool = True,
    ) -> None:
        super().__init__(bus)
        self._seen: set[str] = set()
        self._library = library or Library.empty()
        self._suggest_new = suggest_new

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
        center_hz = float(cap_meta.get("center_hz") or 0.0)
        if not iq_path or samp_rate <= 0 or not os.path.exists(iq_path):
            logger.debug("analysis: capture file not found for {} at {}",
                         ident.detection_id[:8], iq_path)
            return

        result = await asyncio.to_thread(_load_and_analyze, iq_path, samp_rate)
        if result is None:
            return

        # Apparent center of the signal in absolute Hz: tune center +
        # FFT peak offset within the captured passband.
        absolute_center_hz = center_hz + result.freq_offset_hz

        match = self._library.best_match(
            center_hz=absolute_center_hz,
            modulation=result.modulation,
            bw_3db_hz=result.bw_3db_hz,
            symbol_rate_hz=result.symbol_rate_hz,
        )

        notes_extra: list[str] = []
        if match is not None:
            notes_extra.append(
                f"library match: {match.entry.id} (conf={match.confidence:.2f})"
            )
        elif self._suggest_new and self._library_suggestion_makes_sense(result):
            yaml_snippet = suggest_yaml(
                center_hz=absolute_center_hz,
                modulation=result.modulation,
                bw_3db_hz=result.bw_3db_hz,
                symbol_rate_hz=result.symbol_rate_hz,
            )
            notes_extra.append(
                "no library match — candidate entry:\n" + yaml_snippet
            )

        report_kwargs = result.as_report_kwargs()
        if notes_extra:
            existing = report_kwargs.get("notes") or ""
            report_kwargs["notes"] = (
                (existing + " | " if existing else "") + " | ".join(notes_extra)
            )

        report = AnalysisReport(
            detection_id=ident.detection_id,
            artifacts={"iq_path": iq_path},
            **report_kwargs,
        )
        await self.bus.publish(Topics.ANALYSIS_REPORT, report)
        logger.info(
            "analysis: det={} mod={}({:.2f}) bw3={} sr={} bursts={} lib={}",
            ident.detection_id[:8],
            result.modulation,
            result.modulation_confidence,
            f"{result.bw_3db_hz:.0f}Hz" if result.bw_3db_hz else "?",
            f"{result.symbol_rate_hz:.1f}Hz" if result.symbol_rate_hz else "?",
            len(result.bursts),
            match.entry.id if match else "-",
        )

        # If we matched the library, ALSO publish an Identification
        # event so notifiers/aggregators see a "known" signal.
        if match is not None:
            lib_ident = Identification(
                detection_id=ident.detection_id,
                source="library",
                device_class=match.entry.name or match.entry.id,
                decoded={
                    "library_id": match.entry.id,
                    "tags": match.entry.tags,
                    "notes": match.entry.notes,
                    "distances": match.distances,
                },
                confidence=match.confidence,
            )
            await self.bus.publish(Topics.IDENTIFICATION, lib_ident)

    @staticmethod
    def _library_suggestion_makes_sense(result) -> bool:
        """Only suggest a library entry for signals that classified well."""
        if result.modulation in ("unknown", None):
            return False
        if result.modulation_confidence < 0.5:
            return False
        return True


def _load_and_analyze(iq_path: str, samp_rate: float):
    iq = read_iq_cf32(iq_path)
    if iq.size == 0:
        return None
    return analyze(iq, samp_rate)
