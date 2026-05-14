"""Per-bin rolling baseline and change detection.

State per bin lives in numpy arrays, parallel-indexed by bin number.
A `BaselineEngine` is instantiated per range; it consumes SweepFrames
and yields `Detection`s when state transitions fire.

Detection logic (see specification.md §6):

- A *global* per-frame noise floor is estimated as the 10th percentile
  of bin powers in the current sweep. A bin is "active" if its power
  exceeds `noise_floor + weak_signal_db`. This is robust against the
  case where a persistent emitter dominates its own per-bin window.

- Per-bin median and MAD characterize a bin's typical level. The
  appearance threshold T_hi = median + k_up * 1.4826 * MAD is a
  *bin-relative* trigger ("this bin is louder than usual").

- A bin transitions ABSENT -> PRESENT when it is BOTH active globally
  AND above T_hi for n_up consecutive sweeps. This combines a global
  signal/noise sanity check with a per-bin adaptive threshold.

- A bin transitions PRESENT -> ABSENT when it is *not* active globally
  for n_down consecutive sweeps. The per-bin median in PRESENT is
  pinned to the signal level, so we deliberately use the global
  activity criterion rather than a per-bin floor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from riotduck.config import DetectionConfig, RangeConfig
from riotduck.events import Detection, SweepFrame

# Bin state codes (uint8)
S_WARMUP = 0
S_ABSENT = 1
S_PRESENT = 3


@dataclass
class _BinTimes:
    first_seen: np.ndarray
    last_seen: np.ndarray


@dataclass
class BaselineEngine:
    range_cfg: RangeConfig
    detect_cfg: DetectionConfig

    # Lazily initialized on first frame, once we know the bin count.
    _initialized: bool = False
    _n_bins: int = 0
    _bin_centers_hz: np.ndarray = field(default_factory=lambda: np.empty(0))

    _ring: np.ndarray | None = None     # (window_size, n_bins) float32
    _ring_pos: int = 0
    _ring_filled: int = 0

    _state: np.ndarray | None = None    # (n_bins,) uint8
    _up_count: np.ndarray | None = None
    _down_count: np.ndarray | None = None
    _times: _BinTimes | None = None

    # Per-bin median/MAD recomputed each sweep on the active window.
    _median: np.ndarray | None = None
    _mad: np.ndarray | None = None

    def _init(self, freqs_hz: np.ndarray) -> None:
        self._n_bins = len(freqs_hz)
        self._bin_centers_hz = freqs_hz.copy()
        ws = self.detect_cfg.window_size
        self._ring = np.full((ws, self._n_bins), -200.0, dtype=np.float32)
        self._ring_pos = 0
        self._ring_filled = 0
        self._state = np.full(self._n_bins, S_WARMUP, dtype=np.uint8)
        self._up_count = np.zeros(self._n_bins, dtype=np.uint16)
        self._down_count = np.zeros(self._n_bins, dtype=np.uint16)
        self._times = _BinTimes(
            first_seen=np.zeros(self._n_bins, dtype=np.float64),
            last_seen=np.zeros(self._n_bins, dtype=np.float64),
        )
        self._median = np.full(self._n_bins, -120.0, dtype=np.float32)
        self._mad = np.full(self._n_bins, 1.0, dtype=np.float32)
        self._initialized = True

    def ingest(self, frame: SweepFrame) -> list[Detection]:
        """Push a sweep frame, return any detections triggered."""
        if not self._initialized:
            self._init(frame.freqs_hz)
        elif len(frame.freqs_hz) != self._n_bins:
            # Bin grid changed — reset rather than confuse the state.
            self._init(frame.freqs_hz)

        assert self._ring is not None
        assert self._state is not None
        assert self._up_count is not None
        assert self._down_count is not None
        assert self._times is not None

        ws = self.detect_cfg.window_size
        self._ring[self._ring_pos, :] = frame.power_dbfs
        self._ring_pos = (self._ring_pos + 1) % ws
        self._ring_filled = min(ws, self._ring_filled + 1)

        active = self._ring[: self._ring_filled, :]
        self._median = np.median(active, axis=0).astype(np.float32)
        self._mad = np.median(np.abs(active - self._median), axis=0).astype(np.float32)
        mad_eff = np.maximum(self._mad, 1.0)
        scale = 1.4826 * mad_eff

        pow_now = frame.power_dbfs

        # Global per-frame noise floor: P10 across bins. Robust to a
        # handful of strong emitters in the range.
        noise_floor = float(np.percentile(pow_now, 10))
        active_thresh = noise_floor + self.detect_cfg.weak_signal_db
        is_active = pow_now > active_thresh

        # Per-bin appearance trigger.
        t_hi = self._median + self.detect_cfg.k_up * scale
        above_hi = pow_now > t_hi

        warmup_done = self._ring_filled >= max(self.detect_cfg.warmup_min, 1)

        # Promote WARMUP bins to ABSENT or PRESENT once we have data.
        promoted_present = np.zeros(self._n_bins, dtype=bool)
        if warmup_done:
            still_warmup = self._state == S_WARMUP
            if np.any(still_warmup):
                # Static noise floor across the whole warmup window;
                # used only for initial classification.
                static_floor = float(np.percentile(active, 10))
                static_thresh = static_floor + self.detect_cfg.weak_signal_db
                presence_frac = np.mean(active > static_thresh, axis=0)
                init_present = still_warmup & (presence_frac >= self.detect_cfg.presence_threshold)
                init_absent = still_warmup & ~init_present
                self._state[init_present] = S_PRESENT
                self._state[init_absent] = S_ABSENT
                self._times.first_seen[init_present] = frame.ts
                self._times.last_seen[init_present] = frame.ts
                # Don't fire appearance detections for warmup-classified
                # PRESENT bins — they were already there at startup.
                promoted_present[init_present] = True

        appearances: list[Detection] = []
        disappearances: list[Detection] = []

        if warmup_done:
            # ABSENT -> PRESENT: bin must be globally active AND above
            # its own bin-relative T_hi.
            abs_mask = self._state == S_ABSENT
            cand_up = abs_mask & above_hi & is_active
            self._up_count[cand_up] += 1
            self._up_count[abs_mask & ~cand_up] = 0
            promote_up = abs_mask & (self._up_count >= self.detect_cfg.n_up)
            self._state[promote_up] = S_PRESENT
            self._times.first_seen[promote_up] = frame.ts
            self._times.last_seen[promote_up] = frame.ts
            self._up_count[promote_up] = 0

            # PRESENT -> ABSENT: bin no longer globally active.
            present_mask = self._state == S_PRESENT
            # Refresh last_seen for present-and-still-active bins.
            self._times.last_seen[present_mask & is_active] = frame.ts
            cand_down = present_mask & ~is_active
            self._down_count[cand_down] += 1
            self._down_count[present_mask & is_active] = 0
            demote = present_mask & (self._down_count >= self.detect_cfg.n_down)
            self._state[demote] = S_ABSENT
            self._down_count[demote] = 0

            if np.any(promote_up):
                appearances = _coalesce(
                    self._bin_centers_hz,
                    pow_now,
                    self._median,
                    promote_up,
                    "appearance",
                    self.range_cfg,
                    frame,
                    self._times,
                )
            if np.any(demote):
                disappearances = _coalesce(
                    self._bin_centers_hz,
                    pow_now,
                    self._median,
                    demote,
                    "disappearance",
                    self.range_cfg,
                    frame,
                    self._times,
                )

        return appearances + disappearances

    def snapshot(self) -> dict[str, np.ndarray]:
        """Snapshot suitable for npz persistence."""
        assert self._ring is not None
        assert self._state is not None
        return {
            "freqs_hz": self._bin_centers_hz,
            "median_dbfs": self._median if self._median is not None else np.empty(0),
            "mad_dbfs": self._mad if self._mad is not None else np.empty(0),
            "state": self._state,
            "ring": self._ring,
            "ring_pos": np.array([self._ring_pos], dtype=np.int64),
            "ring_filled": np.array([self._ring_filled], dtype=np.int64),
            "ts": np.array([time.time()]),
        }


def _coalesce(
    freqs: np.ndarray,
    power_now: np.ndarray,
    median: np.ndarray,
    mask: np.ndarray,
    kind: str,
    range_cfg: RangeConfig,
    frame: SweepFrame,
    times: _BinTimes,
) -> list[Detection]:
    """Group contiguous True bins into Detection events."""
    if not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    runs: list[tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))

    bin_hz = range_cfg.bin_hz
    out: list[Detection] = []
    for s, e in runs:
        cluster_power = power_now[s : e + 1]
        cluster_freqs = freqs[s : e + 1]
        peak_local = int(np.argmax(cluster_power))
        peak = float(cluster_power[peak_local])
        center = float(cluster_freqs[peak_local])
        in_6dB = cluster_power >= (peak - 6.0)
        bw_hz = max(float(np.sum(in_6dB)) * bin_hz, bin_hz)
        # Noise floor across the median; outside-of-cluster median is
        # roughly the noise level.
        noise = float(np.median(median))
        snr = peak - noise
        if bw_hz < range_cfg.min_bw_hz:
            continue
        bins = list(range(int(s), int(e) + 1))
        if kind == "appearance":
            first_seen = frame.ts
            last_seen = frame.ts
        else:
            first_seen_arr = times.first_seen[s : e + 1]
            last_seen_arr = times.last_seen[s : e + 1]
            nz_first = first_seen_arr[first_seen_arr > 0]
            nz_last = last_seen_arr[last_seen_arr > 0]
            first_seen = float(np.min(nz_first)) if nz_first.size else frame.ts
            last_seen = float(np.max(nz_last)) if nz_last.size else frame.ts
        out.append(
            Detection.new(
                type=kind,  # type: ignore[arg-type]
                range_name=range_cfg.name,
                device_serial=frame.device_serial,
                center_hz=center,
                bw_hz=bw_hz,
                power_dbfs=peak,
                snr_db=snr,
                bins=bins,
                first_seen_ts=first_seen,
                last_seen_ts=last_seen,
            )
        )
    return out
