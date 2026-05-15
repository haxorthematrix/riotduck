"""Top-level runner that wires bus + manager + agents + sinks together.

In v1 the wiring policy is simple:

- For each discovered device with role in {scan, auto}, start a
  ScannerAgent assigned the configured ranges. (Multi-device scanning
  is naive in v1: each device sweeps every range. Smarter
  partitioning lands in phase 4.)
- Notification sinks subscribe to the bus.
- Ctrl-C / SIGTERM triggers a clean shutdown.
"""

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress

from loguru import logger

from riotduck.agents.analysis_agent import AnalysisAgent
from riotduck.agents.fingerprint_agent import FingerprintAgent
from riotduck.agents.scanner_agent import ScannerAgent
from riotduck.bus import EventBus
from riotduck.config import Config, RangeConfig, RangeRef
from riotduck.notify import build_sinks
from riotduck.sdr.manager import DeviceManager


def _materialize_ranges(items: list[RangeConfig | RangeRef]) -> list[RangeConfig]:
    out: list[RangeConfig] = []
    for it in items:
        if isinstance(it, RangeConfig):
            out.append(it)
        else:
            raise RuntimeError(
                f"unresolved range reference {it.use!r}; use load_config()"
            )
    return out


async def run_scan(config: Config) -> None:
    bus = EventBus()
    manager = DeviceManager()
    records = manager.discover()
    if not records:
        logger.error("no SDR devices discovered; aborting")
        return
    manager.apply_device_configs(config.devices)

    ranges = _materialize_ranges(config.ranges)
    if not ranges:
        logger.error("no ranges configured; nothing to scan")
        return

    # Pick devices for scanning.
    scan_devices = [r for r in manager.list() if r.role in ("scan", "auto")]
    if not scan_devices:
        logger.warning("no scan-role devices; using all discovered devices")
        scan_devices = manager.list()

    capture_enabled = (
        config.identification.rtl_433.enabled or config.identification.urh.enabled
    )

    scanner_agents: list[ScannerAgent] = [
        ScannerAgent(
            bus=bus,
            manager=manager,
            device_serial=rec.info.serial,
            ranges=ranges,
            detect_cfg=config.detection,
            orch_cfg=config.orchestrator,
            captures_dir=config.storage.captures_dir,
            capture_enabled=capture_enabled,
        )
        for rec in scan_devices
    ]

    fingerprint_agent: FingerprintAgent | None = None
    analysis_agent: AnalysisAgent | None = None
    if capture_enabled:
        fingerprint_agent = FingerprintAgent(bus=bus, id_cfg=config.identification)
        analysis_agent = AnalysisAgent(bus=bus)

    sinks = build_sinks(config.notify)
    sink_tasks = [asyncio.create_task(s.run(bus), name=f"sink:{s.name}") for s in sinks]

    agents: list = list(scanner_agents)
    if fingerprint_agent is not None:
        agents.append(fingerprint_agent)
    if analysis_agent is not None:
        agents.append(analysis_agent)
    for a in agents:
        await a.start()

    stop = asyncio.Event()

    def _signal_handler():
        logger.info("shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal_handler)

    logger.info(
        "scanning {} ranges across {} device(s); fingerprint={} analysis={}",
        len(ranges),
        len(scanner_agents),
        fingerprint_agent is not None,
        analysis_agent is not None,
    )
    try:
        await stop.wait()
    finally:
        for a in agents:
            await a.stop()
        for t in sink_tasks:
            t.cancel()
        for t in sink_tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
