"""rtl_433 fingerprint integration.

Spawns `rtl_433 -F json -r <iq_path>` against a captured I/Q file and
parses the resulting JSONL into structured hits.

File-mode replay: rtl_433 detects file format from the extension
(`.cf32` = complex float32, which is what `storage/files.py` writes).
We always pass `-s <samp_rate>` because rtl_433 cannot infer that
from the .cf32 container.

Live-mode (spawning rtl_433 against an SDR directly) is not used by
the agent — riotduck always captures to disk first so the same I/Q is
preserved for later URH / manual analysis.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from loguru import logger


@dataclass
class Rtl433Hit:
    model: str
    decoded: dict
    confidence: float = 1.0
    raw_line: str = ""


@dataclass
class Rtl433Result:
    returncode: int
    hits: list[Rtl433Hit] = field(default_factory=list)
    stderr: str = ""
    cmd: list[str] = field(default_factory=list)


def _parse_jsonl(text: str) -> list[Rtl433Hit]:
    hits: list[Rtl433Hit] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # rtl_433 occasionally emits status objects (e.g., {"time": ..., "msg": ...}).
        # We treat anything with a "model" field as a decoded packet.
        if "model" not in obj:
            continue
        hits.append(
            Rtl433Hit(
                model=str(obj.get("model", "unknown")),
                decoded=obj,
                confidence=1.0,
                raw_line=line,
            )
        )
    return hits


def run_on_file(
    iq_path: str | Path,
    samp_rate: float,
    binary: str = "rtl_433",
    center_hz: float | None = None,
    extra_args: Iterable[str] = (),
    timeout_s: float = 60.0,
) -> Rtl433Result:
    """Run rtl_433 against an I/Q file and return decoded hits.

    On binary-not-found or timeout, returns an empty Rtl433Result with
    a non-zero returncode and the stderr populated. Callers should
    not treat this as a hard failure — it just means no fingerprint
    is available.
    """
    cmd: list[str] = [binary, "-F", "json", "-Y", "autolevel", "-s", str(int(samp_rate))]
    if center_hz is not None:
        cmd += ["-f", str(int(center_hz))]
    cmd += list(extra_args)
    cmd += ["-r", str(iq_path)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("rtl_433 binary not found: {}", binary)
        return Rtl433Result(returncode=-1, stderr=f"binary not found: {binary}", cmd=cmd)
    except subprocess.TimeoutExpired as e:
        logger.warning("rtl_433 timed out after {}s on {}", timeout_s, iq_path)
        return Rtl433Result(returncode=-2, stderr=f"timeout: {e}", cmd=cmd)
    except OSError as e:
        logger.warning("rtl_433 OS error: {}", e)
        return Rtl433Result(returncode=-3, stderr=str(e), cmd=cmd)

    hits = _parse_jsonl(proc.stdout)
    return Rtl433Result(
        returncode=proc.returncode,
        hits=hits,
        stderr=proc.stderr,
        cmd=cmd,
    )
