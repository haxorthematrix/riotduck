"""URH (Universal Radio Hacker) integration — placeholder.

Phase 3. The spec assumes a `urh_cli` binary; if Larry's `URH-NG`
target exposes a different command, override `binary` in
identification.urh.binary in the config.

Roughed-in interface so the orchestrator can call it once filled in:

    run_on_file(iq_path, samp_rate) -> list[UrhHit]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UrhHit:
    modulation: str | None
    bit_stream: bytes | None
    notes: str
    confidence: float


def run_on_file(iq_path: str, samp_rate: float, binary: str = "urh_cli") -> list[UrhHit]:
    # Not implemented yet — Phase 3.
    return []
