"""AnalysisAgent + Library integration: library hits publish a second
Identification event; misses attach a YAML suggestion."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from riotduck.agents.analysis_agent import AnalysisAgent
from riotduck.bus import EventBus
from riotduck.events import AnalysisReport, Identification, Topics
from riotduck.library import Library, LibraryEntry, LibraryMatch


def _cw_capture(path: Path, sr: float = 1.024e6, freq_offset: float = 100e3) -> None:
    n = int(sr * 0.05)
    t = np.arange(n) / sr
    iq = (0.5 * np.exp(2j * np.pi * freq_offset * t)).astype(np.complex64)
    rng = np.random.default_rng(0)
    iq += ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-3).astype(np.complex64)
    iq.tofile(path)


def _make_ident(detection_id: str, iq_path: Path, samp_rate: float,
                center_hz: float) -> Identification:
    return Identification(
        detection_id=detection_id,
        source="rtl_433",
        device_class=None,
        decoded={"_capture": {
            "iq_path": str(iq_path),
            "samp_rate": samp_rate,
            "center_hz": center_hz,
            "duration_s": 0.05,
        }},
        confidence=0.0,
    )


@pytest.mark.asyncio
async def test_library_hit_publishes_second_identification(tmp_path: Path):
    sr = 1.024e6
    center = 433.92e6
    iq_path = tmp_path / "cw.cf32"
    _cw_capture(iq_path, sr=sr, freq_offset=100e3)

    # Library entry matched against a CW signal at 433.92 MHz + 100 kHz offset.
    lib = Library(entries=[
        LibraryEntry(
            id="my-cw-thing",
            name="My CW thing",
            match=LibraryMatch(
                center_hz=center + 100e3,
                center_tolerance_hz=20_000,
                modulation="CW",
            ),
        ),
    ])
    bus = EventBus()
    agent = AnalysisAgent(bus=bus, library=lib, suggest_new=False)
    await agent.start()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    received_ids: list[Identification] = []
    received_reports: list[AnalysisReport] = []

    async def listener():
        async with bus.subscribe(Topics.IDENTIFICATION) as id_sub, \
                   bus.subscribe(Topics.ANALYSIS_REPORT) as rep_sub:
            t1 = asyncio.create_task(id_sub.queue.get())
            t2 = asyncio.create_task(rep_sub.queue.get())
            tasks = [t1, t2]
            while True:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED, timeout=2.0
                )
                if not done:
                    break
                for d in done:
                    topic, payload = d.result()
                    if isinstance(payload, Identification):
                        received_ids.append(payload)
                        tasks[0] = asyncio.create_task(id_sub.queue.get())
                    else:
                        received_reports.append(payload)
                        tasks[1] = asyncio.create_task(rep_sub.queue.get())
            for t in tasks:
                t.cancel()

    listen_task = asyncio.create_task(listener())
    await asyncio.sleep(0)

    await bus.publish(Topics.IDENTIFICATION, _make_ident("det-lib", iq_path, sr, center))

    # Give the agent + listener time to process.
    for _ in range(60):
        if received_reports and received_ids:
            break
        await asyncio.sleep(0.02)

    await agent.stop()
    listen_task.cancel()
    try:
        await listen_task
    except (asyncio.CancelledError, Exception):
        pass

    assert len(received_reports) == 1
    # The agent publishes a second identification with source=library.
    lib_idents = [i for i in received_ids if i.source == "library"]
    assert len(lib_idents) == 1
    assert lib_idents[0].device_class == "My CW thing"
    assert lib_idents[0].confidence > 0.0
    assert lib_idents[0].decoded["library_id"] == "my-cw-thing"


@pytest.mark.asyncio
async def test_library_miss_includes_suggestion_in_notes(tmp_path: Path):
    sr = 1.024e6
    center = 433.92e6
    iq_path = tmp_path / "cw.cf32"
    _cw_capture(iq_path, sr=sr, freq_offset=100e3)

    bus = EventBus()
    # Empty library, suggestions enabled
    agent = AnalysisAgent(bus=bus, library=Library.empty(), suggest_new=True)
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
    await bus.publish(Topics.IDENTIFICATION, _make_ident("det-miss", iq_path, sr, center))

    for _ in range(60):
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
    notes = received[0].notes
    assert "no library match" in notes
    assert "center_hz" in notes        # suggestion block embedded


@pytest.mark.asyncio
async def test_library_miss_no_suggestion_when_disabled(tmp_path: Path):
    sr = 1.024e6
    center = 433.92e6
    iq_path = tmp_path / "cw.cf32"
    _cw_capture(iq_path, sr=sr, freq_offset=100e3)

    bus = EventBus()
    agent = AnalysisAgent(bus=bus, library=Library.empty(), suggest_new=False)
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
    await bus.publish(Topics.IDENTIFICATION, _make_ident("det-quiet", iq_path, sr, center))

    for _ in range(60):
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
    notes = received[0].notes or ""
    assert "no library match" not in notes
