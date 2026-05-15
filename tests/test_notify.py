"""NotificationSink regression tests.

Catches the orphaned-task bug where the sink's run() loop created
fresh queue.get() tasks each iteration but only awaited the first to
complete. Tasks left over from prior iterations stayed alive, still
consumed messages, and silently dropped them.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from riotduck.bus import EventBus
from riotduck.config import NotifySink
from riotduck.events import Detection, Identification, Topics
from riotduck.notify.base import NotificationSink


class _RecordingSink(NotificationSink):
    name = "recording"

    def __init__(self) -> None:
        super().__init__(NotifySink(sink="stdout"))
        self.received: list[tuple[str, Any]] = []

    async def deliver(self, topic: str, payload: Any) -> None:
        self.received.append((topic, payload))


def _det(type_: str, center_hz: float, id_: str) -> Detection:
    return Detection.new(
        type=type_,
        range_name="t",
        device_serial="x",
        center_hz=center_hz,
        bw_hz=5000.0,
        power_dbfs=-40.0,
        snr_db=30.0,
        bins=[0],
        first_seen_ts=0.0,
        last_seen_ts=0.0,
    )


@pytest.mark.asyncio
async def test_sink_delivers_interleaved_detections_and_identifications():
    bus = EventBus()
    sink = _RecordingSink()
    sink_task = asyncio.create_task(sink.run(bus))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    expected = []
    # Interleave detections and identifications so the sink loop has
    # to drain both subscriptions. If the orphan-task bug is back,
    # the identifications will be silently lost while detections leak
    # through.
    for i in range(10):
        d = _det("appearance", 100e6 + i * 1e6, f"det-{i:02d}")
        await bus.publish(Topics.DETECTION_APPEARANCE, d)
        expected.append(("detection.appearance", d.id))

        ident = Identification(
            detection_id=d.id,
            source="rtl_433",
            device_class=None,
            decoded={},
            confidence=0.0,
        )
        await bus.publish(Topics.IDENTIFICATION, ident)
        expected.append(("identification", d.id))

    # Give the sink a moment to drain.
    for _ in range(40):
        if len(sink.received) >= len(expected):
            break
        await asyncio.sleep(0.01)

    sink_task.cancel()
    try:
        await sink_task
    except (asyncio.CancelledError, Exception):
        pass

    got_topics = [topic for topic, _ in sink.received]
    assert got_topics.count("detection.appearance") == 10
    assert got_topics.count("identification") == 10, (
        f"identifications lost: got {got_topics.count('identification')} of 10. "
        f"Received: {got_topics}"
    )


@pytest.mark.asyncio
async def test_sink_delivers_back_to_back_identifications():
    """Burst of identifications with no detections interleaved."""
    bus = EventBus()
    sink = _RecordingSink()
    sink_task = asyncio.create_task(sink.run(bus))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    for i in range(20):
        ident = Identification(
            detection_id=f"det-{i:02d}",
            source="rtl_433",
            device_class=f"Model-{i}",
            decoded={"i": i},
            confidence=1.0,
        )
        await bus.publish(Topics.IDENTIFICATION, ident)

    for _ in range(40):
        if len(sink.received) >= 20:
            break
        await asyncio.sleep(0.01)

    sink_task.cancel()
    try:
        await sink_task
    except (asyncio.CancelledError, Exception):
        pass

    assert len(sink.received) == 20, f"lost identifications: {len(sink.received)} of 20"
