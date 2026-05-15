"""End-to-end AnalysisAgent test."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from riotduck.agents.analysis_agent import AnalysisAgent
from riotduck.bus import EventBus
from riotduck.events import AnalysisReport, Identification, Topics


@pytest.mark.asyncio
async def test_analysis_agent_publishes_report_on_miss(tmp_path: Path):
    sr = 1.024e6
    n = int(sr * 0.05)
    t = np.arange(n) / sr
    iq = (0.5 * np.exp(2j * np.pi * 100e3 * t)).astype(np.complex64)
    rng = np.random.default_rng(0)
    iq += ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-3).astype(np.complex64)
    iq_path = tmp_path / "test.cf32"
    iq.tofile(iq_path)

    bus = EventBus()
    agent = AnalysisAgent(bus=bus)
    await agent.start()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    received: list[AnalysisReport] = []

    async def listener():
        async with bus.subscribe(Topics.ANALYSIS_REPORT) as sub:
            while True:
                topic, payload = await sub.queue.get()
                received.append(payload)

    list_task = asyncio.create_task(listener())
    await asyncio.sleep(0)

    ident = Identification(
        detection_id="det-cw",
        source="rtl_433",
        device_class=None,
        decoded={
            "_capture": {
                "iq_path": str(iq_path),
                "samp_rate": sr,
                "center_hz": 433.92e6,
                "duration_s": 0.05,
            }
        },
        confidence=0.0,
    )
    await bus.publish(Topics.IDENTIFICATION, ident)

    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.02)

    await agent.stop()
    list_task.cancel()
    try:
        await list_task
    except (asyncio.CancelledError, Exception):
        pass

    assert len(received) == 1
    r = received[0]
    assert r.detection_id == "det-cw"
    assert r.modulation == "CW"
    assert r.bw_3db_hz is not None


@pytest.mark.asyncio
async def test_analysis_agent_ignores_successful_identifications():
    bus = EventBus()
    agent = AnalysisAgent(bus=bus)
    await agent.start()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    ident = Identification(
        detection_id="det-found",
        source="rtl_433",
        device_class="Acurite-Tower",
        decoded={"_capture": {"iq_path": "/nonexistent", "samp_rate": 2.4e6,
                              "center_hz": 0, "duration_s": 0}},
        confidence=1.0,
    )
    await bus.publish(Topics.IDENTIFICATION, ident)

    async with bus.subscribe(Topics.ANALYSIS_REPORT) as sub:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sub.queue.get(), timeout=0.2)
    await agent.stop()


@pytest.mark.asyncio
async def test_analysis_agent_dedups_by_detection_id(tmp_path: Path):
    sr = 1.024e6
    n = int(sr * 0.05)
    t = np.arange(n) / sr
    iq = (0.5 * np.exp(2j * np.pi * 100e3 * t)).astype(np.complex64)
    iq_path = tmp_path / "x.cf32"
    iq.tofile(iq_path)

    bus = EventBus()
    agent = AnalysisAgent(bus=bus)
    await agent.start()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    received = []
    async def listener():
        async with bus.subscribe(Topics.ANALYSIS_REPORT) as sub:
            while True:
                topic, payload = await sub.queue.get()
                received.append(payload)
    list_task = asyncio.create_task(listener())
    await asyncio.sleep(0)

    ident = Identification(
        detection_id="same-det",
        source="rtl_433",
        device_class=None,
        decoded={"_capture": {"iq_path": str(iq_path), "samp_rate": sr,
                              "center_hz": 0, "duration_s": 0.05}},
        confidence=0.0,
    )
    # Publish twice
    await bus.publish(Topics.IDENTIFICATION, ident)
    await asyncio.sleep(0.5)
    await bus.publish(Topics.IDENTIFICATION, ident)

    for _ in range(50):
        if len(received) >= 1:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.3)

    await agent.stop()
    list_task.cancel()
    try:
        await list_task
    except (asyncio.CancelledError, Exception):
        pass

    assert len(received) == 1
