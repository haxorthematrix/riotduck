"""Agent base class.

An Agent is a long-lived asyncio task. Subclasses implement `run()`
and may subscribe to bus topics. The runner owns lifecycle and is
responsible for clean shutdown.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from loguru import logger

from riotduck.bus import EventBus


class Agent(ABC):
    name: str = "agent"

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @abstractmethod
    async def run(self) -> None:
        """Main loop. Should periodically check `self._stop.is_set()`."""

    async def start(self) -> None:
        if self._task is not None:
            return
        logger.info("starting agent {}", self.name)
        self._task = asyncio.create_task(self._wrap(), name=self.name)

    async def stop(self) -> None:
        if self._task is None:
            return
        logger.info("stopping agent {}", self.name)
        self._stop.set()
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None

    async def _wrap(self) -> None:
        try:
            await self.run()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("agent {} crashed: {}", self.name, e)
            raise

    def should_stop(self) -> bool:
        return self._stop.is_set()
