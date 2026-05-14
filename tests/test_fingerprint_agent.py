"""End-to-end test of the capture.result → identification flow."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from riotduck.agents.fingerprint_agent import FingerprintAgent
from riotduck.bus import EventBus
from riotduck.config import IdentificationConfig, IdToolConfig
from riotduck.events import CaptureResult, Identification, Topics
from riotduck.fingerprint.rtl_433 import Rtl433Hit, Rtl433Result


async def _collect_one(bus: EventBus, pattern: str, timeout: float = 2.0) -> Identification:
    async with bus.subscribe(pattern) as sub:
        topic, payload = await asyncio.wait_for(sub.queue.get(), timeout=timeout)
        return payload


async def _start_agent(bus: EventBus, id_cfg: IdentificationConfig) -> FingerprintAgent:
    agent = FingerprintAgent(bus=bus, id_cfg=id_cfg)
    await agent.start()
    # Yield once so the agent's subscription is registered before we publish.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return agent


@pytest.mark.asyncio
async def test_fingerprint_agent_publishes_identification_on_hit(tmp_path: Path):
    bus = EventBus()
    id_cfg = IdentificationConfig(
        rtl_433=IdToolConfig(enabled=True, binary="rtl_433"),
        urh=IdToolConfig(enabled=False),
    )
    iq = tmp_path / "x.cf32"
    iq.write_bytes(b"\x00" * 64)

    fake_result = Rtl433Result(
        returncode=0,
        hits=[Rtl433Hit(model="Acurite-Tower", decoded={"model": "Acurite-Tower", "id": 42},
                        confidence=1.0)],
    )

    agent = await _start_agent(bus, id_cfg)
    try:
        with patch("riotduck.agents.fingerprint_agent.run_on_file", return_value=fake_result):
            cap = CaptureResult(
                detection_id="det-1234",
                path=str(iq),
                samp_rate=2.4e6,
                center_hz=433.92e6,
                duration_s=0.5,
            )
            ident_task = asyncio.create_task(_collect_one(bus, Topics.IDENTIFICATION))
            # publish AFTER subscribing to avoid race
            await asyncio.sleep(0)
            await bus.publish(Topics.CAPTURE_RESULT, cap)
            ident = await ident_task
    finally:
        await agent.stop()

    assert isinstance(ident, Identification)
    assert ident.detection_id == "det-1234"
    assert ident.source == "rtl_433"
    assert ident.device_class == "Acurite-Tower"
    assert ident.confidence == 1.0


@pytest.mark.asyncio
async def test_fingerprint_agent_emits_no_match_sentinel(tmp_path: Path):
    bus = EventBus()
    id_cfg = IdentificationConfig(
        rtl_433=IdToolConfig(enabled=True, binary="rtl_433"),
        urh=IdToolConfig(enabled=False),
    )
    iq = tmp_path / "x.cf32"
    iq.write_bytes(b"\x00" * 64)

    no_hits = Rtl433Result(returncode=0, hits=[])

    agent = await _start_agent(bus, id_cfg)
    try:
        with patch("riotduck.agents.fingerprint_agent.run_on_file", return_value=no_hits):
            cap = CaptureResult(
                detection_id="det-empty",
                path=str(iq),
                samp_rate=2.4e6,
                center_hz=433.92e6,
                duration_s=0.5,
            )
            ident_task = asyncio.create_task(_collect_one(bus, Topics.IDENTIFICATION))
            await asyncio.sleep(0)
            await bus.publish(Topics.CAPTURE_RESULT, cap)
            ident = await ident_task
    finally:
        await agent.stop()

    assert ident.detection_id == "det-empty"
    assert ident.source == "rtl_433"
    assert ident.device_class is None
    assert ident.confidence == 0.0


@pytest.mark.asyncio
async def test_fingerprint_agent_skips_when_disabled(tmp_path: Path):
    bus = EventBus()
    id_cfg = IdentificationConfig(
        rtl_433=IdToolConfig(enabled=False),
        urh=IdToolConfig(enabled=False),
    )
    iq = tmp_path / "x.cf32"
    iq.write_bytes(b"\x00" * 64)

    agent = await _start_agent(bus, id_cfg)
    try:
        cap = CaptureResult(
            detection_id="det-skip",
            path=str(iq),
            samp_rate=2.4e6,
            center_hz=433.92e6,
            duration_s=0.5,
        )
        await bus.publish(Topics.CAPTURE_RESULT, cap)
        # No identification should arrive within a small window.
        async with bus.subscribe(Topics.IDENTIFICATION) as sub:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(sub.queue.get(), timeout=0.2)
    finally:
        await agent.stop()
