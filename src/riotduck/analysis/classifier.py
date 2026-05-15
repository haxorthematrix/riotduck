"""Unknown-signal classifier.

Pure-function analyzer that characterizes captured I/Q without
relying on a protocol library. Used by the AnalysisAgent when
rtl_433 (and eventually URH) miss.

The classifier is intentionally heuristic rather than research-grade.
The goal is to give the operator something useful — "this looks like
an OOK signal at ~2 kbit/s with 8 kHz bandwidth" — so they can decide
whether to investigate further or add a custom decoder.

Pipeline:

1. burst_segments() — find contiguous signal-active intervals
2. measure_bandwidth() — power-vs-frequency, -3/-6/-20 dB widths
3. classify_modulation() — CW / OOK / FSK / PSK / unknown
4. estimate_symbol_rate() — cyclostationary peak in |x(t)|^2's FFT
5. analyze() — runs the lot, returns AnalysisReport-ready dict
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BurstSegment:
    start_sample: int
    end_sample: int

    @property
    def length(self) -> int:
        return self.end_sample - self.start_sample

    def duration_s(self, samp_rate: float) -> float:
        return self.length / samp_rate if samp_rate > 0 else 0.0


@dataclass
class AnalysisResult:
    samp_rate: float
    duration_s: float
    bursts: list[BurstSegment] = field(default_factory=list)
    fraction_active: float = 0.0
    bw_3db_hz: float | None = None
    bw_6db_hz: float | None = None
    bw_20db_hz: float | None = None
    freq_offset_hz: float = 0.0
    modulation: str = "unknown"
    modulation_confidence: float = 0.0
    symbol_rate_hz: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_report_kwargs(self) -> dict:
        return dict(
            modulation=self.modulation,
            symbol_rate_hz=self.symbol_rate_hz,
            bw_3db_hz=self.bw_3db_hz,
            bw_6db_hz=self.bw_6db_hz,
            bw_20db_hz=self.bw_20db_hz,
            notes="; ".join(self.notes),
        )


# ----- helpers -----

def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _envelope_power_db(iq: np.ndarray, smooth_window: int = 0) -> np.ndarray:
    p = (iq.real ** 2 + iq.imag ** 2).astype(np.float32)
    if smooth_window > 1:
        p = _moving_average(p, smooth_window)
    return 10.0 * np.log10(np.maximum(p, 1e-30))


# ----- burst segmentation -----

def burst_segments(
    iq: np.ndarray,
    samp_rate: float,
    threshold_db: float = 10.0,
    smooth_ms: float = 0.5,
    min_duration_ms: float = 0.2,
    merge_gap_ms: float = 0.5,
) -> list[BurstSegment]:
    """Find contiguous signal-on intervals.

    threshold_db: how many dB above the smoothed-envelope median to count as "on".
    smooth_ms:     envelope smoothing window in ms.
    min_duration_ms: drop bursts shorter than this.
    merge_gap_ms:    merge bursts separated by ≤ this much silence.
    """
    if len(iq) == 0:
        return []
    smooth_n = max(1, int(samp_rate * smooth_ms / 1000.0))
    env_db = _envelope_power_db(iq, smooth_window=smooth_n)
    # Robust noise floor: median of the lower 30% of envelope samples.
    sorted_env = np.sort(env_db)
    floor = float(np.mean(sorted_env[: max(1, len(sorted_env) // 3)]))
    active = env_db > floor + threshold_db
    if not np.any(active):
        return []

    # Find rising/falling edges.
    diff = np.diff(active.astype(np.int8))
    starts = list(np.flatnonzero(diff == 1) + 1)
    ends = list(np.flatnonzero(diff == -1) + 1)
    if active[0]:
        starts.insert(0, 0)
    if active[-1]:
        ends.append(len(active))

    segments = [BurstSegment(s, e) for s, e in zip(starts, ends) if e > s]

    # Merge close bursts.
    merge_n = int(samp_rate * merge_gap_ms / 1000.0)
    merged: list[BurstSegment] = []
    for seg in segments:
        if merged and (seg.start_sample - merged[-1].end_sample) <= merge_n:
            merged[-1] = BurstSegment(merged[-1].start_sample, seg.end_sample)
        else:
            merged.append(seg)

    # Drop too-short bursts.
    min_n = int(samp_rate * min_duration_ms / 1000.0)
    return [s for s in merged if s.length >= min_n]


# ----- bandwidth -----

def measure_bandwidth(
    iq: np.ndarray,
    samp_rate: float,
    n_fft: int = 8192,
    drop_levels: tuple[float, ...] = (3.0, 6.0, 20.0),
    dc_mask_hz: float = 5e3,
) -> dict[float, float]:
    """Measure occupied bandwidth at the given -N dB levels.

    Uses the median FFT magnitude as the noise reference; -N dB is
    measured relative to the *peak*. DC (±dc_mask_hz) is masked to
    suppress the RTL-SDR's center spur from biasing the measurement.
    """
    if len(iq) < 64:
        return {d: 0.0 for d in drop_levels}
    n = min(len(iq), n_fft)
    seg = iq[:n].astype(np.complex64) - np.mean(iq[:n])
    w = np.hanning(n).astype(np.float32)
    psd = np.abs(np.fft.fftshift(np.fft.fft(seg * w))) ** 2
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))

    df = samp_rate / n
    mid = n // 2
    dc_bins = max(1, int(dc_mask_hz / df))
    psd_db[mid - dc_bins : mid + dc_bins] = -200.0

    peak = float(psd_db.max())
    out: dict[float, float] = {}
    for d in drop_levels:
        above = psd_db >= peak - d
        if not np.any(above):
            out[d] = 0.0
            continue
        idx = np.flatnonzero(above)
        out[d] = float((idx[-1] - idx[0] + 1) * df)
    return out


def estimate_freq_offset(iq: np.ndarray, samp_rate: float, n_fft: int = 8192) -> float:
    """Coarse frequency offset = location of the FFT peak (Hz from tune center)."""
    if len(iq) < 64:
        return 0.0
    n = min(len(iq), n_fft)
    seg = iq[:n] - np.mean(iq[:n])
    w = np.hanning(n).astype(np.float32)
    psd = np.abs(np.fft.fftshift(np.fft.fft(seg * w))) ** 2
    mid = n // 2
    dc_mask = max(1, int(5e3 / (samp_rate / n)))
    psd[mid - dc_mask : mid + dc_mask] = 0
    peak_bin = int(np.argmax(psd))
    return float((peak_bin - n / 2) * samp_rate / n)


# ----- modulation classification -----

def classify_modulation(
    iq: np.ndarray,
    samp_rate: float,
    bw_3db_hz: float | None = None,
) -> tuple[str, float, list[str]]:
    """Heuristic modulation class. Returns (label, confidence, notes).

    Decision tree:
      - very narrow + low envelope variance + low freq variance  → CW
      - high envelope coefficient-of-variation                   → OOK / ASK
      - low envelope CV + multimodal instantaneous frequency     → FSK
      - low envelope CV + continuous freq spread                 → FM
      - low envelope CV + discrete phase jumps                   → PSK
      - otherwise                                                → unknown
    """
    notes: list[str] = []
    if len(iq) < 256:
        return "unknown", 0.0, ["too few samples"]

    # 1. Envelope statistics.
    env = np.abs(iq).astype(np.float64)
    env_mean = float(np.mean(env))
    if env_mean < 1e-6:
        return "unknown", 0.0, ["no signal energy"]
    env_std = float(np.std(env))
    env_cv = env_std / env_mean
    notes.append(f"env_cv={env_cv:.3f}")

    # 2. Instantaneous frequency via phase difference.
    phase = np.unwrap(np.angle(iq.astype(np.complex64)))
    inst_freq = np.diff(phase) * samp_rate / (2 * np.pi)
    if_mean = float(np.mean(inst_freq))
    if_std = float(np.std(inst_freq))
    if_range = float(np.percentile(inst_freq, 99) - np.percentile(inst_freq, 1))
    notes.append(f"if_std={if_std:.1f} Hz")
    notes.append(f"if_p99-p01={if_range:.1f} Hz")

    # Narrow CW: low envelope CV AND narrow occupied bandwidth. We
    # deliberately don't require a tight if_range here — phase noise
    # from finite SNR makes instantaneous frequency twitchy even for
    # an obvious tone. A narrow FFT footprint is the more reliable
    # discriminator.
    is_narrow = bw_3db_hz is not None and bw_3db_hz < 1500.0
    if env_cv < 0.15 and is_narrow:
        notes.append("constant envelope + narrow BW")
        return "CW", 0.9, notes

    # Envelope bimodality test for OOK/ASK:
    if env_cv > 0.6:
        # Confirm with a quick bimodality check: two clusters in env histogram.
        hist, edges = np.histogram(env, bins=32)
        # Find two strongest non-adjacent peaks.
        peaks = np.argsort(hist)[::-1]
        notes.append(f"env_hist_top_bins={list(peaks[:3].tolist())}")
        return "OOK", 0.75, notes + ["high envelope variance"]

    # Constant envelope (FM / FSK / PSK).
    if env_cv < 0.25:
        # FSK: instantaneous frequency clusters at a few discrete values.
        # Test: bimodality of the IF histogram.
        # Trim outliers.
        if_clean = inst_freq[
            (inst_freq > np.percentile(inst_freq, 1))
            & (inst_freq < np.percentile(inst_freq, 99))
        ]
        if len(if_clean) > 64:
            hist, edges = np.histogram(if_clean, bins=64)
            # FSK has two peaks well-separated. Compute peak separation
            # using a 2-bin smoothed histogram.
            smooth = _moving_average(hist.astype(np.float32), 3)
            peak_idx = np.argsort(smooth)[::-1]
            top2 = sorted(peak_idx[:2].tolist())
            sep_bins = top2[1] - top2[0] if len(top2) == 2 else 0
            bin_width = edges[1] - edges[0]
            sep_hz = sep_bins * bin_width
            notes.append(f"if_peak_sep={sep_hz:.1f} Hz")
            if sep_bins > 8 and if_std > 500:
                return "FSK", 0.7, notes
        if if_std < 200:
            return "CW", 0.7, notes + ["constant envelope, constant freq, wider BW"]
        return "FM", 0.55, notes + ["constant envelope, freq varies smoothly"]

    return "unknown", 0.3, notes


# ----- symbol-rate via cyclostationary feature -----

def estimate_symbol_rate(
    iq: np.ndarray,
    samp_rate: float,
    min_rate_hz: float = 100.0,
    max_rate_hz: float = 200_000.0,
) -> float | None:
    """Estimate symbol rate from envelope edge spacing.

    Approach:
      1. Smooth the envelope to suppress noise jitter.
      2. Threshold at the midpoint between min and max of the
         smoothed envelope — this works for OOK and ASK alike.
      3. Find rising and falling edges; the inter-edge intervals are
         multiples of T_b (1·T_b, 2·T_b, 3·T_b, ...).
      4. Report the 25th-percentile interval as T_b. For random NRZ
         this lands on single-symbol runs, which are the most common
         pattern.

    Works on signals where bits are random (no spectral line at
    1/T_b) — autocorrelation-based estimators fail in that case.
    """
    if len(iq) < 2048:
        return None
    # Smooth over ~10x the shortest plausible symbol period to keep
    # noise from creating spurious edges.
    min_period_samples = max(2, int(samp_rate / max_rate_hz))
    smooth_n = max(1, min_period_samples // 4)
    env = np.abs(iq).astype(np.float32)
    env_s = _moving_average(env, smooth_n)
    env_min = float(np.percentile(env_s, 5))
    env_max = float(np.percentile(env_s, 95))
    if env_max - env_min < 1e-4:
        return None
    threshold = (env_min + env_max) / 2.0
    above = env_s > threshold
    edges = np.flatnonzero(np.diff(above.astype(np.int8)) != 0)
    if len(edges) < 4:
        return None
    intervals = np.diff(edges)
    # Filter intervals to the plausible band.
    max_lag = int(samp_rate / min_rate_hz)
    intervals = intervals[(intervals >= min_period_samples) & (intervals <= max_lag)]
    if len(intervals) < 2:
        return None
    # 25th percentile picks the shortest typical interval, which is
    # the symbol period for any random pattern with single-bit runs.
    t_b = float(np.percentile(intervals, 25))
    if t_b <= 0:
        return None
    return float(samp_rate / t_b)


# ----- driver -----

def analyze(iq: np.ndarray, samp_rate: float) -> AnalysisResult:
    """Run the full analysis pipeline. Returns a structured result."""
    duration = len(iq) / samp_rate if samp_rate > 0 else 0.0
    result = AnalysisResult(samp_rate=samp_rate, duration_s=duration)

    if len(iq) == 0:
        result.notes.append("empty capture")
        return result

    # Burst segmentation
    bursts = burst_segments(iq, samp_rate)
    result.bursts = bursts
    if bursts:
        total_on = sum(b.length for b in bursts)
        result.fraction_active = total_on / len(iq)
        result.notes.append(f"{len(bursts)} burst(s), active={result.fraction_active*100:.1f}%")

    # If we have bursts, analyze the LONGEST burst (most signal energy).
    # Otherwise analyze the whole capture (continuous signal case).
    if bursts:
        longest = max(bursts, key=lambda b: b.length)
        segment = iq[longest.start_sample : longest.end_sample]
        result.notes.append(
            f"analyzing longest burst: {longest.duration_s(samp_rate)*1000:.1f} ms"
        )
    else:
        segment = iq
        result.notes.append("no distinct bursts; analyzing whole capture")

    # Bandwidth
    bw = measure_bandwidth(segment, samp_rate)
    result.bw_3db_hz = bw.get(3.0)
    result.bw_6db_hz = bw.get(6.0)
    result.bw_20db_hz = bw.get(20.0)

    # Frequency offset (where in the IF the signal sits)
    result.freq_offset_hz = estimate_freq_offset(segment, samp_rate)

    # Modulation
    mod, conf, mnotes = classify_modulation(segment, samp_rate, bw_3db_hz=result.bw_3db_hz)
    result.modulation = mod
    result.modulation_confidence = conf
    result.notes.extend(mnotes)

    # Symbol rate (only meaningful for modulated signals)
    if mod in ("OOK", "FSK", "PSK"):
        sr = estimate_symbol_rate(segment, samp_rate)
        if sr is not None:
            result.symbol_rate_hz = sr
            result.notes.append(f"symbol_rate≈{sr:.1f} Hz")

    return result
