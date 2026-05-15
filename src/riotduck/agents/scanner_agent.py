"""Scanner agent: owns an SDR, sweeps a list of ranges, publishes frames.

When an appearance detection fires, the agent pauses sweeping, retunes
the same SDR to the detection's center frequency, records `capture_ms`
of I/Q to disk, sets `iq_path` on the detection event, and publishes
both `detection.appearance` and `capture.result`. The fingerprint
agent picks the capture result up from the bus.

This implements the spec §10 "single-SDR pause-for-analysis" behavior.
When the orchestrator/multi-SDR work lands (Phase 4), this inline
capture step will move into a dedicated CaptureAgent that the
orchestrator routes to a free SDR.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

from riotduck.agents.base import Agent
from riotduck.baseline import BaselineEngine
from riotduck.bus import EventBus
from riotduck.capture import (
    capture_for_detection,
    capture_from_buffer,
    choose_capture_samp_rate,
)
from riotduck.config import DetectionConfig, OrchestratorConfig, RangeConfig
from riotduck.dedup import EventTracker
from riotduck.events import Detection, Topics
from riotduck.scanner import Scanner, plan_sweep
from riotduck.sdr.manager import DeviceManager


class ScannerAgent(Agent):
    def __init__(
        self,
        bus: EventBus,
        manager: DeviceManager,
        device_serial: str,
        ranges: list[RangeConfig],
        detect_cfg: DetectionConfig,
        orch_cfg: OrchestratorConfig | None = None,
        captures_dir: str = "captures",
        capture_enabled: bool = True,
    ) -> None:
        super().__init__(bus)
        self.name = f"scanner[{device_serial}]"
        self.manager = manager
        self.device_serial = device_serial
        self.ranges = ranges
        self.detect_cfg = detect_cfg
        self.orch_cfg = orch_cfg or OrchestratorConfig()
        self.captures_dir = captures_dir
        self.capture_enabled = capture_enabled
        self._engines: dict[str, BaselineEngine] = {
            r.name: BaselineEngine(range_cfg=r, detect_cfg=detect_cfg) for r in ranges
        }
        self._range_by_name = {r.name: r for r in ranges}
        self._tracker = EventTracker(
            re_observation_s=self.orch_cfg.re_observation_s,
            min_freq_tolerance_hz=detect_cfg.min_freq_tolerance_hz,
        )
        self._scanner: Scanner | None = None

    async def run(self) -> None:
        session = self.manager.acquire(self.device_serial, holder=self.name)
        try:
            scanner = Scanner(session)
            self._scanner = scanner
            plans = [
                plan_sweep(r, session.info.samp_rates or (2.4e6,))
                for r in self.ranges
            ]
            logger.info(
                "{}: sweeping {} ranges, fft_size up to {}",
                self.name,
                len(plans),
                max((p.fft_size for p in plans), default=0),
            )

            while not self.should_stop():
                for plan in plans:
                    if self.should_stop():
                        break
                    started = time.monotonic()
                    try:
                        frame = await asyncio.to_thread(scanner.sweep, plan)
                    except Exception as e:
                        logger.exception("sweep failed for {}: {}", plan.range_cfg.name, e)
                        await asyncio.sleep(0.5)
                        continue

                    await self.bus.publish(Topics.SWEEP_FRAME, frame)

                    engine = self._engines[plan.range_cfg.name]
                    detections = engine.ingest(frame)
                    await self._dispatch_detections(session, detections)

                    elapsed = time.monotonic() - started
                    slack = plan.range_cfg.repeat_s - elapsed
                    if slack > 0:
                        await asyncio.sleep(slack)
                    else:
                        await asyncio.sleep(0)
        finally:
            self.manager.release(self.device_serial, session)

    async def _dispatch_detections(self, session, detections: list[Detection]) -> None:
        for d in detections:
            to_publish = self._tracker.observe(d)
            if to_publish is None:
                # Re-observation within the window — silently folded.
                continue

            if to_publish.type == "appearance" and self.capture_enabled:
                try:
                    await self._capture_inline(session, to_publish)
                except Exception as e:
                    logger.exception("inline capture failed for {}: {}", to_publish.id, e)

            topic = (
                Topics.DETECTION_APPEARANCE
                if to_publish.type == "appearance"
                else Topics.DETECTION_DISAPPEARANCE
            )
            await self.bus.publish(topic, to_publish)
            await self.bus.publish(Topics.DETECTION, to_publish)

    async def _capture_inline(self, session, detection: Detection) -> None:
        rng = self._range_by_name.get(detection.range_name)
        if rng is None:
            return

        # Preferred path: use the in-memory I/Q from the sweep that
        # produced this detection. The same samples that registered as
        # an FFT bin spike are guaranteed to contain the burst.
        if self._scanner is not None:
            tune = self._scanner.find_capture_for_freq(detection.center_hz)
            if tune is not None:
                cap = await asyncio.to_thread(
                    capture_from_buffer, tune, detection, self.captures_dir
                )
                if cap is not None:
                    detection.iq_path = cap.path
                    await self.bus.publish(Topics.CAPTURE_RESULT, cap)
                    return

        # Fallback: retune + re-read. Loses the burst on short
        # transmissions but is correct for steady emitters.
        samp_rate = choose_capture_samp_rate(
            detection,
            rng.samp_rate,
            session.info.samp_rates or (2.4e6,),
        )
        gain = {k: v for k, v in rng.gain.model_dump().items() if v is not None}
        cap = await asyncio.to_thread(
            capture_for_detection,
            session,
            detection,
            self.captures_dir,
            self.orch_cfg.capture_ms,
            samp_rate,
            gain,
        )
        if cap is None:
            return
        detection.iq_path = cap.path
        await self.bus.publish(Topics.CAPTURE_RESULT, cap)
