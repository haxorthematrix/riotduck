"""Re-observation / event-dedup tracker.

A `BaselineEngine` produces one Detection per state transition. For a
bursty emitter (key fob, TPMS) that fires twice a second, that means
two alerts per second — useless noise. The tracker folds repeat
detections at the same frequency into a single logical event:

- **Active emitter**, fresh detection arrives within re_observation_s:
  suppress the publish but renew last_seen_ts on the tracked event.
- **Silent emitter** (disappearance was published), appearance arrives
  within re_observation_s: publish the appearance again but reuse the
  original event id, so the operator sees "X came back" rather than
  "new emitter".
- **Outside the window**: a fresh appearance is treated as a brand-new
  event with a brand-new id.

Frequency matching tolerance is the larger of:
- a configurable floor (`min_freq_tolerance_hz`, default 5 kHz), and
- half the wider of the new vs. tracked detection's bandwidth.

This keeps two close-but-distinct narrow emitters separate while still
recognizing a drifting wideband emitter as one event.

The tracker is per-ScannerAgent in v1. If multiple SDRs sweep the same
range, each agent tracks independently and may publish duplicate
events — by design, since dedup across hosts/agents requires the
phase-4 orchestrator's shared event store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from riotduck.events import Detection

EventState = Literal["active", "silent"]


@dataclass
class TrackedEvent:
    id: str
    range_name: str
    center_hz: float
    bw_hz: float
    first_seen_ts: float
    last_seen_ts: float
    state: EventState


class EventTracker:
    def __init__(self, re_observation_s: float, min_freq_tolerance_hz: float = 5000.0) -> None:
        self.re_observation_s = float(re_observation_s)
        self.min_tol_hz = float(min_freq_tolerance_hz)
        self._events: list[TrackedEvent] = []

    # ---- public surface ----

    def observe(self, det: Detection) -> Detection | None:
        """Process a fresh detection.

        Returns the Detection to publish (with id possibly replaced by
        the original event's id), or None if the detection should be
        suppressed entirely as a re-observation.
        """
        self._expire(det.ts)
        match = self._find_match(det)

        if det.type == "appearance":
            return self._observe_appearance(det, match)
        return self._observe_disappearance(det, match)

    def active_events(self) -> list[TrackedEvent]:
        return [e for e in self._events if e.state == "active"]

    def all_events(self) -> list[TrackedEvent]:
        return list(self._events)

    # ---- internals ----

    def _observe_appearance(self, det: Detection, match: TrackedEvent | None) -> Detection | None:
        if match is None:
            self._events.append(_event_from(det, state="active"))
            return det
        if match.state == "active":
            # Pure re-observation — suppress, just refresh last_seen.
            match.last_seen_ts = det.ts
            return None
        # state == "silent": emitter coming back after a disappearance.
        # Re-emit appearance but reuse the original event id so the
        # operator sees continuity.
        det.id = match.id
        det.first_seen_ts = match.first_seen_ts
        match.state = "active"
        match.last_seen_ts = det.ts
        match.bw_hz = max(match.bw_hz, det.bw_hz)
        return det

    def _observe_disappearance(
        self, det: Detection, match: TrackedEvent | None
    ) -> Detection | None:
        if match is None:
            # Disappearance with no prior tracking — typical at startup
            # when a bin is warmup-promoted to PRESENT then goes silent.
            self._events.append(_event_from(det, state="silent"))
            return det
        if match.state == "silent":
            # Already published a disappearance for this one; suppress
            # the dupe.
            match.last_seen_ts = det.ts
            return None
        det.id = match.id
        det.first_seen_ts = match.first_seen_ts
        match.state = "silent"
        match.last_seen_ts = det.ts
        return det

    def _find_match(self, det: Detection) -> TrackedEvent | None:
        best: TrackedEvent | None = None
        best_dist = float("inf")
        for e in self._events:
            if e.range_name != det.range_name:
                continue
            tol = max(self.min_tol_hz, max(e.bw_hz, det.bw_hz) / 2.0)
            dist = abs(e.center_hz - det.center_hz)
            if dist <= tol and dist < best_dist:
                best = e
                best_dist = dist
        return best

    def _expire(self, now_ts: float) -> None:
        cutoff = now_ts - self.re_observation_s
        self._events = [e for e in self._events if e.last_seen_ts >= cutoff]


def _event_from(det: Detection, state: EventState) -> TrackedEvent:
    return TrackedEvent(
        id=det.id,
        range_name=det.range_name,
        center_hz=det.center_hz,
        bw_hz=det.bw_hz,
        first_seen_ts=det.first_seen_ts or det.ts,
        last_seen_ts=det.ts,
        state=state,
    )
