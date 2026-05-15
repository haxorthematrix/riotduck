"""User-defined fingerprint library.

When rtl_433 has no decoder for a signal, the analyzer extracts a
small set of robust features — center frequency, modulation, -3 dB
bandwidth, symbol rate. The Library compares those features against
a user-curated YAML file. A hit becomes an `Identification` event
with `source="library"`, indistinguishable to downstream consumers
from an rtl_433 hit except by the source field.

Schema (kept deliberately small so it's easy to hand-edit):

```yaml
entries:
  - id: my-mystery-433
    name: "Mystery 433 MHz OOK device"
    notes: "First observed 2026-05-14 in the lab"
    tags: [lab, unknown]
    match:
      center_hz: 433.928e+6
      center_tolerance_hz: 50_000
      modulation: OOK              # optional; exact match
      bw_3db_hz: 2200              # optional
      bw_3db_tolerance_hz: 1500
      symbol_rate_hz: 5900         # optional
      symbol_rate_tolerance_hz: 300
```

Match scoring: every criterion the user supplied must pass its
tolerance check. Confidence is `1 - mean(d_i)` where `d_i` is the
distance from target normalized by tolerance (so a perfect hit
scores 1.0 and a just-barely-pass scores ~0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from riotduck.config import _coerce_numerics


ModulationLabel = Literal["CW", "OOK", "ASK", "FSK", "FM", "PSK", "AM"]


class LibraryMatch(BaseModel):
    """Criteria for matching an analysis report to a library entry.

    `center_hz` is mandatory. The others are optional — leave a field
    out to skip that criterion entirely. Tolerances default to
    reasonable values but can be overridden per-entry.
    """

    center_hz: float
    center_tolerance_hz: float = 50_000.0

    modulation: str | None = None
    bw_3db_hz: float | None = None
    bw_3db_tolerance_hz: float = 1500.0
    symbol_rate_hz: float | None = None
    symbol_rate_tolerance_hz: float = 300.0

    @field_validator("modulation")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v


class LibraryEntry(BaseModel):
    id: str
    name: str = ""
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    match: LibraryMatch


class LibraryFile(BaseModel):
    """Top-level YAML schema."""

    entries: list[LibraryEntry] = Field(default_factory=list)


@dataclass
class MatchResult:
    """Outcome of scoring an analysis report against one entry."""

    entry: LibraryEntry
    confidence: float       # 0..1; 1 = perfect hit, 0 = at-tolerance boundary
    distances: dict[str, float]  # per-criterion normalized distances


# ---------- scoring ----------

def score_entry(
    entry: LibraryEntry,
    *,
    center_hz: float,
    modulation: str | None,
    bw_3db_hz: float | None,
    symbol_rate_hz: float | None,
) -> MatchResult | None:
    """Return a MatchResult if every defined criterion passes, else None."""
    m = entry.match
    dists: dict[str, float] = {}

    # center frequency — always required
    d_center = abs(center_hz - m.center_hz)
    if d_center > m.center_tolerance_hz:
        return None
    dists["center_hz"] = d_center / m.center_tolerance_hz if m.center_tolerance_hz > 0 else 0.0

    if m.modulation:
        if modulation is None or modulation.upper() != m.modulation.upper():
            return None
        dists["modulation"] = 0.0

    if m.bw_3db_hz is not None:
        if bw_3db_hz is None:
            return None
        d = abs(bw_3db_hz - m.bw_3db_hz)
        if d > m.bw_3db_tolerance_hz:
            return None
        dists["bw_3db_hz"] = d / m.bw_3db_tolerance_hz if m.bw_3db_tolerance_hz > 0 else 0.0

    if m.symbol_rate_hz is not None:
        if symbol_rate_hz is None:
            return None
        d = abs(symbol_rate_hz - m.symbol_rate_hz)
        if d > m.symbol_rate_tolerance_hz:
            return None
        dists["symbol_rate_hz"] = (
            d / m.symbol_rate_tolerance_hz if m.symbol_rate_tolerance_hz > 0 else 0.0
        )

    confidence = 1.0 - (sum(dists.values()) / len(dists)) if dists else 1.0
    confidence = max(0.0, min(1.0, confidence))
    return MatchResult(entry=entry, confidence=confidence, distances=dists)


# ---------- library ----------

class Library:
    """Loaded fingerprint library. Holds entries in memory; matches in O(N)."""

    def __init__(self, entries: Iterable[LibraryEntry] = ()) -> None:
        self.entries: list[LibraryEntry] = list(entries)
        self._by_id: dict[str, LibraryEntry] = {e.id: e for e in self.entries}

    @classmethod
    def empty(cls) -> "Library":
        return cls()

    @classmethod
    def load(cls, path: Path | str) -> "Library":
        p = Path(path)
        if not p.exists():
            logger.debug("library file not found: {} (empty library)", p)
            return cls.empty()
        with p.open() as f:
            raw = yaml.safe_load(f) or {}
        normalized_entries: list[dict] = []
        for e in raw.get("entries", []) or []:
            e = dict(e)
            if "match" in e and isinstance(e["match"], dict):
                e["match"] = _coerce_numerics(e["match"])
            normalized_entries.append(e)
        raw["entries"] = normalized_entries
        parsed = LibraryFile.model_validate(raw)
        return cls(parsed.entries)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = LibraryFile(entries=self.entries).model_dump(mode="json")
        with p.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    def get(self, entry_id: str) -> LibraryEntry | None:
        return self._by_id.get(entry_id)

    def best_match(
        self,
        *,
        center_hz: float,
        modulation: str | None,
        bw_3db_hz: float | None,
        symbol_rate_hz: float | None,
    ) -> MatchResult | None:
        """Score every entry; return the highest-confidence match (if any)."""
        best: MatchResult | None = None
        for e in self.entries:
            r = score_entry(
                e,
                center_hz=center_hz,
                modulation=modulation,
                bw_3db_hz=bw_3db_hz,
                symbol_rate_hz=symbol_rate_hz,
            )
            if r is None:
                continue
            if best is None or r.confidence > best.confidence:
                best = r
        return best

    def __len__(self) -> int:
        return len(self.entries)


# ---------- helpers ----------

def suggest_yaml(
    *,
    center_hz: float,
    modulation: str | None,
    bw_3db_hz: float | None,
    symbol_rate_hz: float | None,
    placeholder_id: str | None = None,
) -> str:
    """Build a YAML snippet the user can paste into their library.

    Used when the analyzer produces a useful classification but no
    existing library entry matches — gives the operator a one-step
    way to remember this signal next time.
    """
    pid = placeholder_id or _placeholder_id(center_hz, modulation, symbol_rate_hz)
    lines = [
        f"- id: {pid}",
        f"  name: ''",
        f"  match:",
        f"    center_hz: {center_hz/1e6:.6f}e+6",
        f"    center_tolerance_hz: 50_000",
    ]
    if modulation:
        lines.append(f"    modulation: {modulation}")
    if bw_3db_hz:
        lines.append(f"    bw_3db_hz: {bw_3db_hz:.0f}")
        lines.append(f"    bw_3db_tolerance_hz: {max(500.0, bw_3db_hz*0.5):.0f}")
    if symbol_rate_hz:
        lines.append(f"    symbol_rate_hz: {symbol_rate_hz:.0f}")
        lines.append(f"    symbol_rate_tolerance_hz: {max(200.0, symbol_rate_hz*0.05):.0f}")
    return "\n".join(lines)


def _placeholder_id(
    center_hz: float, modulation: str | None, symbol_rate_hz: float | None
) -> str:
    parts = [f"unknown-{center_hz/1e6:.3f}M"]
    if modulation:
        parts.append(modulation.lower())
    if symbol_rate_hz:
        parts.append(f"{symbol_rate_hz:.0f}hz")
    return "-".join(parts).replace(".", "p")
