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
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from loguru import logger


_VERSION_RE = re.compile(r"version\s+(\d+)\.(\d+)(?:[-.]\S*)?", re.IGNORECASE)

# Anything older than this is flagged as stale by `riotduck doctor`.
# rtl_433 versions are YY.MM: 23.x = early 2023, plenty of modern
# decoders. The 2021 Homebrew binary that shipped as 20.11 will trip.
RTL_433_MIN_RECOMMENDED: tuple[int, int] = (23, 0)


@dataclass
class Rtl433Info:
    """Snapshot of the rtl_433 install on PATH (and any stale shadows)."""

    path: str | None
    version: tuple[int, int] | None
    version_str: str
    shadows: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def installed(self) -> bool:
        return self.path is not None

    @property
    def is_stale(self) -> bool:
        return (
            self.version is not None
            and self.version < RTL_433_MIN_RECOMMENDED
        )


def _brew_rtl_433_path(timeout_s: float = 3.0) -> str | None:
    """Return the brew-installed rtl_433 binary if Homebrew knows about it.

    macOS-specific. Returns None when `brew` isn't on PATH, when the
    formula isn't installed, or when the binary doesn't exist where
    brew thinks it should.
    """
    try:
        proc = subprocess.run(
            ["brew", "--prefix", "rtl_433"],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    prefix = (proc.stdout or "").strip()
    if not prefix:
        return None
    candidate = os.path.join(prefix, "bin", "rtl_433")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _all_rtl_433_on_path(name: str = "rtl_433") -> list[str]:
    """Return every executable `name` on PATH, in PATH order, deduped by realpath."""
    found: list[str] = []
    seen: set[str] = set()
    raw_path = os.environ.get("PATH", "")
    for entry in raw_path.split(os.pathsep):
        if not entry:
            continue
        candidate = os.path.join(entry, name)
        try:
            if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
                continue
        except OSError:
            continue
        try:
            resolved = os.path.realpath(candidate)
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(candidate)
    return found


def _probe_version(binary: str, timeout_s: float = 5.0) -> tuple[tuple[int, int] | None, str, str | None]:
    """Run `<binary> -V` and parse the version. Returns (tuple, raw_str, error)."""
    try:
        proc = subprocess.run(
            [binary, "-V"],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except FileNotFoundError:
        return None, "", "binary not found at runtime"
    except subprocess.TimeoutExpired:
        return None, "", "timed out running -V"
    except OSError as e:
        return None, "", str(e)
    blob = "\n".join(p for p in (proc.stdout, proc.stderr) if p)
    m = _VERSION_RE.search(blob)
    if m is None:
        return None, "(unparseable)", "could not parse version"
    major, minor = int(m.group(1)), int(m.group(2))
    return (major, minor), m.group(0).split(maxsplit=1)[1], None


def get_rtl_433_info(binary: str | None = None, timeout_s: float = 5.0) -> Rtl433Info:
    """Inspect the rtl_433 install. Detects version + alternative installs.

    `binary` overrides the auto-discovered path on `$PATH`.

    `shadows` contains paths to *other* rtl_433 installs the user
    might not be using. Two sources are checked:

    - PATH walk: later `rtl_433` entries on PATH after the active one
      (would-be shadows if the first were removed).
    - Homebrew (`brew --prefix rtl_433`): the brew install may not be
      symlinked into PATH if an older copy already squats at
      /usr/local/bin/rtl_433. That's the case `riotduck doctor` is
      most likely to flag on macOS.
    """
    active = binary or shutil.which("rtl_433")

    # Collect all alternative installs we know about.
    candidates: list[str] = []
    for p in _all_rtl_433_on_path():
        if p not in candidates:
            candidates.append(p)
    brew_path = _brew_rtl_433_path()
    if brew_path and brew_path not in candidates:
        candidates.append(brew_path)

    shadows: list[str] = []
    if active:
        try:
            active_real = os.path.realpath(active)
        except OSError:
            active_real = active
        for p in candidates:
            try:
                p_real = os.path.realpath(p)
            except OSError:
                p_real = p
            if p_real != active_real:
                shadows.append(p)

    if not active:
        return Rtl433Info(path=None, version=None, version_str="", shadows=shadows)

    version, version_str, error = _probe_version(active, timeout_s)
    return Rtl433Info(
        path=active,
        version=version,
        version_str=version_str,
        shadows=shadows,
        error=error,
    )


def probe_rtl_433_version(binary: str, timeout_s: float = 5.0) -> tuple[int, int] | None:
    """Convenience wrapper for one-off version probes (used by doctor)."""
    version, _, _ = _probe_version(binary, timeout_s)
    return version


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
