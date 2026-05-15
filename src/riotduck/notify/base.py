"""Notification sinks.

A sink subscribes to bus topics and converts payloads to JSON-able
dicts. Sinks are passive (they don't initiate work); the runner
attaches them to the bus and they emit to their backend.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from riotduck.bus import EventBus
from riotduck.config import NotifySink
from riotduck.events import Detection, Identification, Topics


def _to_jsonable(payload: Any) -> Any:
    if dataclasses.is_dataclass(payload):
        d = dataclasses.asdict(payload)
        # numpy arrays in dataclasses (SweepFrame) → omit by default,
        # they're huge and noisy for notifications.
        return {k: v for k, v in d.items() if not _is_array_like(v)}
    return payload


def _is_array_like(v: Any) -> bool:
    return hasattr(v, "shape") and hasattr(v, "dtype")


class NotificationSink(ABC):
    name: str = "abstract"

    def __init__(self, cfg: NotifySink) -> None:
        self.cfg = cfg
        self._task: asyncio.Task | None = None

    def _passes_filter(self, topic: str, payload: Any) -> bool:
        f = self.cfg.filter
        if f is None:
            return True
        if f.types:
            if isinstance(payload, Detection):
                if payload.type not in f.types:
                    return False
            elif not any(t in topic for t in f.types):
                return False
        if f.ranges and isinstance(payload, Detection):
            if payload.range_name not in f.ranges:
                return False
        if f.min_snr_db is not None and isinstance(payload, Detection):
            if payload.snr_db < f.min_snr_db:
                return False
        return True

    @abstractmethod
    async def deliver(self, topic: str, payload: Any) -> None: ...

    async def run(self, bus: EventBus) -> None:
        # One long-lived drain task per subscription. The previous
        # implementation spawned fresh queue.get() tasks each loop
        # iteration and only awaited the first to complete; tasks
        # from earlier iterations were orphaned but kept consuming
        # from the queue, silently dropping messages whose result was
        # never observed.
        async with bus.subscribe("detection.*") as detsub, \
                   bus.subscribe("identification") as idsub, \
                   bus.subscribe("analysis.report") as ansub:
            async def _drain(sub) -> None:
                while True:
                    topic, payload = await sub.queue.get()
                    if not self._passes_filter(topic, payload):
                        continue
                    try:
                        await self.deliver(topic, payload)
                    except Exception as e:
                        logger.exception("sink {} deliver failed: {}", self.name, e)

            tasks = [asyncio.create_task(_drain(s)) for s in (detsub, idsub, ansub)]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                raise
            finally:
                for t in tasks:
                    t.cancel()


class StdoutSink(NotificationSink):
    name = "stdout"

    async def deliver(self, topic: str, payload: Any) -> None:
        if isinstance(payload, Detection):
            arrow = "^" if payload.type == "appearance" else "v"
            iq = f" iq={payload.iq_path}" if payload.iq_path else ""
            print(
                f"[{topic}] {arrow} {payload.range_name} "
                f"@ {payload.center_hz/1e6:.4f} MHz "
                f"bw={payload.bw_hz/1e3:.1f} kHz "
                f"snr={payload.snr_db:.1f} dB "
                f"dev={payload.device_serial} id={payload.id[:8]}{iq}",
                flush=True,
            )
        elif isinstance(payload, Identification):
            if payload.device_class is None:
                print(
                    f"[{topic}] ?? no {payload.source} match "
                    f"det={payload.detection_id[:8]}",
                    flush=True,
                )
            else:
                print(
                    f"[{topic}] ID {payload.source}: {payload.device_class} "
                    f"conf={payload.confidence:.2f} det={payload.detection_id[:8]}",
                    flush=True,
                )
        else:
            print(f"[{topic}] {payload}", flush=True)


class JsonlSink(NotificationSink):
    name = "jsonl"

    def __init__(self, cfg: NotifySink) -> None:
        super().__init__(cfg)
        self.path = cfg.path or "events.jsonl"

    async def deliver(self, topic: str, payload: Any) -> None:
        obj = {"topic": topic, "payload": _to_jsonable(payload)}
        line = json.dumps(obj, default=str)
        # Tiny writes are fine here; rotation can be layered on later.
        await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with open(self.path, "a") as f:
            f.write(line + "\n")


class WebhookSink(NotificationSink):
    name = "webhook"

    def __init__(self, cfg: NotifySink) -> None:
        super().__init__(cfg)
        if not cfg.url:
            raise ValueError("webhook sink requires url")
        self.url = cfg.url

    async def deliver(self, topic: str, payload: Any) -> None:
        try:
            import aiohttp
        except ImportError:
            logger.error("webhook sink requires aiohttp")
            return
        body = {"topic": topic, "payload": _to_jsonable(payload)}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.url, json=body, timeout=10) as resp:
                    if resp.status >= 400:
                        logger.warning("webhook {} returned {}", self.url, resp.status)
            except Exception as e:
                logger.warning("webhook {} failed: {}", self.url, e)


_REGISTRY: dict[str, type[NotificationSink]] = {
    "stdout": StdoutSink,
    "jsonl": JsonlSink,
    "webhook": WebhookSink,
}


def build_sinks(configs: list[NotifySink]) -> list[NotificationSink]:
    out: list[NotificationSink] = []
    for c in configs:
        cls = _REGISTRY.get(c.sink)
        if cls is None:
            logger.warning("unknown notify sink: {}", c.sink)
            continue
        try:
            out.append(cls(c))
        except Exception as e:
            logger.warning("failed to build sink {}: {}", c.sink, e)
    return out
