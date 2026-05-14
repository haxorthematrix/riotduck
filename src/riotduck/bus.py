"""In-process asyncio pub/sub bus.

Topics are dot-separated strings. Subscribers may listen to an exact
topic or to a prefix (suffix "*") — e.g., "detection.*" matches both
"detection.appearance" and "detection.disappearance".

The bus is intentionally minimal. If we later need durability or
multi-host, swap this for a NATS/Redis-backed implementation behind
the same `publish/subscribe` surface.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger


class _Subscription:
    __slots__ = ("pattern", "queue", "maxsize")

    def __init__(self, pattern: str, maxsize: int = 256) -> None:
        self.pattern = pattern
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def matches(self, topic: str) -> bool:
        if self.pattern == "*":
            return True
        if self.pattern.endswith(".*"):
            return topic.startswith(self.pattern[:-1])
        return topic == self.pattern


class EventBus:
    def __init__(self) -> None:
        self._subs: list[_Subscription] = []
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, payload: Any) -> None:
        async with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if not sub.matches(topic):
                continue
            try:
                sub.queue.put_nowait((topic, payload))
            except asyncio.QueueFull:
                # Drop oldest to make room — backpressure on a real-time
                # signal stream would be worse than a dropped frame.
                try:
                    _ = sub.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                sub.queue.put_nowait((topic, payload))
                logger.warning("bus: queue full for pattern {!r}, dropped oldest", sub.pattern)

    @asynccontextmanager
    async def subscribe(self, pattern: str, maxsize: int = 256) -> AsyncIterator[_Subscription]:
        sub = _Subscription(pattern, maxsize=maxsize)
        async with self._lock:
            self._subs.append(sub)
        try:
            yield sub
        finally:
            async with self._lock:
                if sub in self._subs:
                    self._subs.remove(sub)

    async def stream(self, pattern: str, maxsize: int = 256) -> AsyncIterator[tuple[str, Any]]:
        """Convenience async generator over a subscription."""
        async with self.subscribe(pattern, maxsize=maxsize) as sub:
            while True:
                yield await sub.queue.get()


async def drain(sub: _Subscription, handler: Callable[[str, Any], Any]) -> None:
    """Run `handler(topic, payload)` for every message until cancelled."""
    while True:
        topic, payload = await sub.queue.get()
        try:
            r = handler(topic, payload)
            if asyncio.iscoroutine(r):
                await r
        except Exception as e:
            logger.exception("bus handler raised: {}", e)
