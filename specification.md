# riotduck — RF Anomaly Detection System

**Status:** v0.1 — Phases 1, 2, and the start of Phase 4 in. Validated on
real RTL-SDR hardware end-to-end.
**Owner:** Larry
**Date:** 2026-05-14

### At a glance

What works today:

- Sweep-scan over user-defined or predefined ranges. **RTL-SDR
  validated on hardware** via pyrtlsdr (Realtek RTL2832U + R820T
  tuner); HackRF streaming path is in via SoapySDR but not yet
  hardware-tested.
- Per-bin rolling baseline (median/MAD) with global P10 noise floor.
- Appearance + disappearance detection with hysteresis, coalescing,
  and a re-observation/dedup tracker that folds repeat bursts into
  one logical event.
- **Pre-detection ring buffer**: the I/Q that produced the FFT spike
  is preserved in memory and written to disk on detection, instead of
  retuning and re-reading. Captures contain the actual burst, not
  post-burst noise.
- I/Q captures written as `.cf32`, day-partitioned under
  `captures/YYYY-MM-DD/<detection_id>.cf32`.
- rtl_433 fingerprint pipeline: subprocess wrapper, async
  FingerprintAgent, identification events on the bus.
- **Unknown-signal analyzer agent**: when rtl_433 returns "no
  match", the analyzer characterizes the signal — burst
  segmentation, bandwidth at -3/-6/-20 dB, modulation classification
  (CW / OOK / FSK / FM), symbol-rate via edge-spacing. Publishes
  `AnalysisReport` on the bus.
- Notification sinks: stdout, JSONL, webhook. Single-drain-per-
  subscription pattern (a bug where `identification` events were
  being silently dropped was caught + fixed).
- Synthetic SDR backend with configurable emitters — the full
  pipeline runs end-to-end without hardware.
- CLI: `scan`, `devices`, `ranges`, `doctor`, `capture`, `analyze`.
- 65 unit tests, all passing, no hardware required.

What's deferred:

- HackRF `hackrf_sweep` fast path and hardware-tested HackRF
  streaming (Phase 3).
- URH integration — wired through config but the demod call is a
  stub (Phase 3).
- Multi-SDR orchestration with dedicated CaptureAgent and dynamic
  device reassignment (Phase 4). Today, one ScannerAgent per device
  sweeps every range and captures inline; dedup is per-agent.
- SQLite event store (JSONL sink covers durable logging).
- Persistent baselines across restarts.
- Web dashboard (Phase 5).

### Verified against a real unknown emitter

A continuous 433.92 MHz transmitter that rtl_433 (both the 2021
build and 2025.12) cannot decode produced this pipeline output:

```
[detection.appearance]   ^ ism_433 @ 433.9080 MHz bw=8.0 kHz snr=31.1 dB id=3f0094b2 ...
[identification]         ?? no rtl_433 match det=3f0094b2
[analysis.report]        >> analysis: mod=OOK bw_3dB=937 Hz sym_rate=6497 Hz det=3f0094b2
```

The analyzer correctly identifies it as OOK modulation, ~6.5 kbit/s
symbol rate, ~940 Hz carrier bandwidth, ~50 kHz offset from the
sweep's tune center. That's enough to feed straight into URH or a
custom decoder.

## 1. Overview

`riotduck` is a software-defined-radio (SDR) based RF anomaly detection
system. It continuously sweeps one or more user-defined frequency ranges,
builds a statistical baseline of present RF energy per frequency bin, and
raises alerts on two classes of change:

- **Appearance** — a new emitter shows up where the baseline was quiet.
  This is the classic "rogue transmitter" / "new threat emitter" case.
- **Disappearance** — an emitter that was reliably present in the
  baseline goes silent. This catches situations like a previously-placed
  covert transmitter (e.g., a VaporTrail-style implant) being removed,
  jammed, or having lost power.

When a change is detected, the system can hand the suspect frequency off
to a second pipeline that captures I/Q, attempts known-signal
fingerprinting (rtl_433, URH-style demod), and if that fails, performs
deeper unknown-signal analysis (modulation classification, symbol
recovery, recording).

Hardware support starts with **RTL-SDR** (RTL2832U dongles), with first-
class support planned for **HackRF One**. Multi-device operation is a
core design goal: with two or more SDRs, one device keeps sweeping while
another is reassigned to capture/analyze candidate signals. The device
abstraction is built on **SoapySDR** so additional radios (Airspy, USRP,
LimeSDR, SDRplay) can be added with minimal new code.

## 2. Goals & Non-Goals

### Goals

- Drop-in baseline scanner that produces heatmap-grade spectral data
  comparable to `rtl_power` / keenerd's `heatmap.py`.
- Statistically robust appearance/disappearance detection per bin with
  hysteresis (no flapping on borderline signals).
- Library of pre-defined ISM / common-allocation ranges, plus
  user-defined ranges in YAML.
- Pluggable identification pipeline: rtl_433 first, URH demod second,
  unknown-signal analysis third.
- Multi-SDR scheduling: scan + analyze concurrently when ≥2 devices.
- Pluggable notification sinks (stdout/log, file, webhook, syslog,
  optional MQTT).
- Recorded I/Q artifacts retained for post-hoc analysis.

### Non-Goals (initially)

- No transmit. riotduck is RX-only; HackRF TX paths are explicitly off.
- No protocol decoding beyond what rtl_433/URH already provide. We
  defer decoding of arbitrary new protocols to URH/manual analysis.
- No GUI in v1. CLI + structured logs + optional web dashboard later.
- No direction-finding (TDoA, Doppler). Possible long-term phase.
- No cellular/Wi-Fi/Bluetooth dissection. We *detect* their presence
  like any other emitter but don't decode them.
- Not a regulatory-compliance scanner — it observes whatever the
  attached antenna sees.

## 3. Hardware

### 3.1 Supported devices (v1)

| Device          | Status   | Tuning range     | Max practical BW | Notes                              |
|-----------------|----------|------------------|------------------|------------------------------------|
| RTL-SDR (R820T2)| Required | ~24 MHz – 1.7 GHz| ~2.4 MS/s        | Reference device. Cheap, ubiquitous|
| HackRF One      | Required | 1 MHz – 6 GHz    | ~20 MS/s         | Use `hackrf_sweep` fast path too   |

### 3.2 Planned (v2+)

Airspy Mini/R2, SDRplay RSP1A/RSPdx, USRP B-series, LimeSDR Mini.

### 3.3 Discovery & identification

- All devices are addressed through **SoapySDR** (`SoapySDRDevice`).
- Devices are uniquely identified by **serial number** (RTL-SDR EEPROM
  serial; HackRF board serial). If multiple identical RTL-SDRs are
  plugged in without unique serials set, riotduck refuses to run until
  the user re-serializes them with `rtl_eeprom`. (Soft warning, hard
  error if config references them by serial.)
- Each device has a configurable **role**:
  - `scan` — assigned to sweep work
  - `analyze` — assigned to capture/identify work
  - `auto` — orchestrator decides per-task
- A device may also have an **antenna profile** (e.g., "discone",
  "log-periodic 400-1000 MHz") which affects per-band gain/calibration
  but is otherwise opaque metadata.

### 3.4 USB / host constraints

- USB 2.0 bus saturation is real. Two HackRFs at 20 MS/s on one
  controller will not work; riotduck reports the assignment but does
  not enforce host topology — operator's responsibility.
- riotduck emits a warning if the configured aggregate sample rate
  per USB controller exceeds a configurable budget.

## 4. System Architecture

```
                ┌────────────────────────────────────────────┐
                │                 Orchestrator                │
                │  (job queue, device allocation, policies)   │
                └─────┬───────────────┬──────────────────┬────┘
                      │               │                  │
                ┌─────▼──────┐  ┌─────▼──────┐    ┌──────▼──────┐
                │  Scanner   │  │ Fingerprint│    │  Unknown    │
                │   Agent    │  │   Agent    │    │  Signal     │
                │            │  │ (rtl_433,  │    │  Analyzer   │
                │            │  │  URH)      │    │  Agent      │
                └─────┬──────┘  └─────┬──────┘    └──────┬──────┘
                      │               │                  │
                      ▼               ▼                  ▼
                ┌─────────────────────────────────────────────┐
                │              Event Bus (asyncio)            │
                └──────────────┬────────────────────┬─────────┘
                               │                    │
                       ┌───────▼──────┐    ┌────────▼───────┐
                       │  Baseline /  │    │  Notification  │
                       │  Detector    │    │     Sinks      │
                       └───────┬──────┘    └────────────────┘
                               │
                       ┌───────▼──────┐
                       │   Storage    │
                       │ (SQLite +    │
                       │  IQ files)   │
                       └──────────────┘
```

Components:

- **Device Manager** — enumerates SoapySDR devices, manages
  reservations (`acquire(serial) / release(serial)`), surfaces
  capabilities (tuning range, sample rates, gain stages).
- **Scanner Agent** — owns one or more SDRs in `scan` role; performs
  swept-tune FFT sweeps over configured ranges and emits
  `SweepFrame` events.
- **Baseline / Detector** — maintains per-bin rolling statistics, runs
  appearance/disappearance state machines, emits `Detection` events.
- **Orchestrator** — subscribes to detections, decides which agent (and
  which SDR) should handle each detection, dispatches `AnalysisJob`s.
- **Fingerprint Agent** — runs rtl_433 against an I/Q capture or live
  tune of the suspect frequency; on miss, falls back to URH-style
  modulation/symbol recovery.
- **Unknown Signal Analyzer** — performs modulation classification
  (AM/FM/OOK/FSK/PSK), symbol-rate estimation, persistent recording,
  and writes an "analysis report" event.
- **Notification Sinks** — stdout, file, JSON-lines, webhook, syslog,
  MQTT (later). Multiple sinks active in parallel.
- **Storage** — SQLite for events/baselines/recordings index;
  filesystem for raw I/Q and waterfall PNGs.
- **Event Bus** — in-process asyncio pub/sub. Designed so a later
  refactor to NATS/Redis is a swap of the bus implementation.

## 5. Scanning & Baseline

### 5.1 Sweep strategy

Two backends, chosen per device:

1. **Native sweep** — riotduck steps the SDR through tuning points,
   collects N FFT frames per dwell, windows + averages, decimates to
   bin width, advances. Used for RTL-SDR and as fallback for HackRF.
2. **`hackrf_sweep` fast path** — for HackRF only, spawn `hackrf_sweep`
   and parse its stdout. Vastly faster across multi-GHz ranges. Used
   automatically when a range exceeds a configurable width
   (default: 100 MHz).

Per range, the user specifies:

```yaml
ranges:
  - name: 433_ism
    f_start: 433.0e6
    f_end:   434.8e6
    bin_hz:  4000        # FFT bin width in Hz
    dwell_ms: 50         # time per tune point
    window:  hann
    samp_rate: 2.4e6     # advisory; clamped to device capability
    gain:                # device-specific; auto if omitted
      lna: 28
      vga: 16
      mixer: 12
    repeat_s: 1.0        # nominal sweep cadence (best effort)
```

Defaults: `bin_hz=4000`, `dwell_ms=50`, `window=hann`. Sweep cadence is
best-effort — if the configured range is too wide to complete in
`repeat_s`, riotduck logs a warning and runs as fast as it can.

### 5.2 Pre-defined ranges

Ships with `config/ism_bands.yaml` containing common allocations the
user can reference by name. Initial set:

- `ism_433` — 433.05–434.79 MHz (ITU Region 1 ISM, US Part 15)
- `ism_315` — 314–316 MHz (common in NA OEM remotes)
- `ism_868` — 863–870 MHz (ITU Region 1 SRD; LoRa EU)
- `ism_902_928` — 902–928 MHz (US ISM; LoRa US, Z-Wave US)
- `ism_2400` — 2400–2483.5 MHz (BT, Wi-Fi 2.4, Zigbee, drones)
- `ism_5800` — 5725–5875 MHz (Wi-Fi 5G upper, drone video)
- `pmr446` — 446.0–446.2 MHz
- `frs_gmrs` — 462.5625–467.7250 MHz
- `marine_vhf` — 156.0–162.025 MHz
- `aviation_air` — 118.0–137.0 MHz (AM)
- `airband_acars` — 131.5–131.9 MHz
- `keyfobs_315` — 315 MHz ± 250 kHz (NA fobs)
- `keyfobs_433` — 433.92 MHz ± 250 kHz (EU/aftermarket fobs)
- `tpms_315` — 315 MHz ± 250 kHz (NA TPMS)
- `tpms_433` — 433.92 MHz ± 250 kHz (EU TPMS)
- `weather_sondes` — 400.15–406 MHz
- `pagers_pocsag_lo` — 138–174 MHz (subset)
- `pagers_pocsag_hi` — 929–932 MHz

User-defined ranges live in the user config and merge with this set.

### 5.3 Baseline statistics

Per range, per bin, riotduck maintains:

- **Rolling window** of the last N power values (default N=300, ≈5
  minutes at 1 s repeat). Implemented as a fixed-size ring buffer.
- **Median** and **MAD** (median absolute deviation). MAD is preferred
  over stddev because transient emitters are exactly the outliers we
  don't want polluting the noise floor estimate.
- **Quantiles** Q05, Q50, Q95 retained for reporting.
- **Presence histogram**: fraction of the window above a "weak signal"
  threshold (default: noise_floor + 6 dB), used by the disappearance
  detector.

Baselines are **per-bin**, not per-range. Adjacent strong bins
(carrier + sidebands) are not coalesced at baseline time; coalescing
happens at detection time.

### 5.4 Warmup

A bin's state machine is `WARMUP` until the rolling window has at
least `warmup_min` samples (default: 60). During warmup no detections
fire. This prevents alert storms at startup.

### 5.5 Persistence

The rolling baseline can be persisted to SQLite on shutdown and
re-loaded on startup with a configurable max-age (default: 1 hour). If
the persisted baseline is older than max-age, it is discarded and
warmup re-runs.

## 6. Detection

### 6.1 Per-bin state machine

```
        ┌──────────┐
        │  WARMUP  │
        └────┬─────┘
             │ window full → classify by presence_frac:
             │   frac ≥ 0.8 → PRESENT
             │   frac < 0.8 → ABSENT
             ▼
        ┌──────────┐    above_hi  ┌──────────┐
        │  ABSENT  ├───AND active─►│  PRESENT │
        │          │  (n_up consec)│          │
        └────▲─────┘               └────┬─────┘
             │                          │
             │   NOT active             │
             └──(n_down consec)─────────┘
```

We use a **two-criterion** trigger to combine a stable global signal/
noise sense with per-bin adaptive sensitivity:

- **Global activity** — `is_active = pow_now > (noise_floor +
  weak_signal_db)`, where `noise_floor` is the 10th percentile of bin
  powers in the current sweep. P10 is robust against a few strong
  emitters in the range polluting the floor estimate.
- **Per-bin appearance threshold** — `T_hi = median_bin + k_up *
  1.4826 * MAD_bin`. The bin's own rolling median/MAD track slow drift
  and per-bin spurs.

Transitions:

- `ABSENT → PRESENT` when a bin is BOTH `above_hi` AND `is_active`
  for `n_up` consecutive sweeps. Fires an **appearance** detection.
- `PRESENT → ABSENT` when a bin is `NOT is_active` for `n_down`
  consecutive sweeps. Fires a **disappearance** detection.

Why not symmetric per-bin thresholds? When a bin is `PRESENT` due to
a long-duty-cycle emitter, its rolling median tracks the *signal*
level, not the noise floor. A per-bin "below T_lo" trigger then has
no useful reference — `median - k*MAD` is still well above noise, so
the bin would fire disappearance even while transmitting. Using the
global noise floor for disappearance avoids that pathology.

Defaults: `k_up=8`, `n_up=3`, `n_down=5`, `weak_signal_db=6`,
`window_size=300`, `warmup_min=60`.

(`k_down` is preserved in the config schema for a future per-bin
floor-based disappearance criterion; it is currently unused.)

Bins promoted directly to `PRESENT` at the end of warmup
(via `presence_frac ≥ presence_threshold`) do **not** emit an
appearance event — they were already there at startup. They will
emit a disappearance if they later go silent.

### 6.2 Coalescing

After per-bin decisions are made each sweep, a coalescing pass groups
adjacent active bins into a single **Detection** with:

- `center_hz`, `bw_hz` (estimated from the cluster's -6 dB extent)
- `power_dbfs` (peak)
- `snr_db` (peak − median noise outside cluster)
- `bins[]` (the contributing bin indices)
- `first_seen_ts`, `last_seen_ts`
- `class`: `APPEARED` or `DISAPPEARED`

A detection is suppressed if its `bw_hz` is below a per-range
`min_bw_hz` (default 0 — no suppression). Useful for ignoring narrow
spurs from the RTL-SDR's reference oscillator drift.

### 6.3a Re-observation / event dedup

`BaselineEngine` fires one Detection per state transition. For a
bursty emitter (key fob, TPMS) appearing at ~1 Hz cadence, this
produces an alert per burst — useless noise. The `EventTracker`
(`src/riotduck/dedup.py`) sits between the engine and the bus and
folds repeats:

- An active emitter's repeat appearance within `re_observation_s` is
  silently suppressed; only `last_seen_ts` is refreshed.
- A disappearance for a tracked active emitter is published with the
  original event id and marks the tracker silent.
- An appearance that arrives while the tracker is silent (within the
  window) is published — but with the *original* event id, so the
  operator sees "X came back" with continuity to the prior alert.
- Outside the window, a new appearance is a brand-new event with a
  fresh id.

Frequency matching tolerance is `max(min_freq_tolerance_hz,
max(tracked_bw, new_bw) / 2)` — wider for wideband signals, capped
at the configured floor for narrow ones.

### 6.3 Short-transmission handling

Many threat emitters (key fobs, TPMS, garage doors) are sub-100 ms
bursts. Their probability of capture during sweep dwell is low. For
ranges flagged `short_burst: true`, riotduck switches that range's
scanner to **stare mode** during its sweep slot — narrow the tune to
the range, increase sample rate, and write a continuous power-vs-time
record for the burst-detection window. The trade-off: that sweep slot
no longer covers anything but the flagged range during its turn.

When a short burst is detected:

- `capture_ms` of I/Q (default: 500 ms, centered on detection) is
  written to disk.
- If only one SDR is available and the capture is non-trivial, the
  orchestrator may **pause sweeping** for `capture_ms + analysis_budget_ms`
  (configurable, default 5 s) to record. This is the "stop scanning,
  capture and analyze" behavior the user specified.
- A **re-observation timeout** is started (default 30 s): if the burst
  reappears within the timeout it is treated as the same emitter (same
  event id, updated `last_seen_ts`). Outside the timeout it is a new
  event.

### 6.4 Detection event schema

```json
{
  "id": "uuid",
  "type": "appearance | disappearance",
  "ts": 1715625600.123,
  "range": "ism_433",
  "device_serial": "00000001",
  "center_hz": 433920000,
  "bw_hz": 12000,
  "power_dbfs": -38.2,
  "snr_db": 27.4,
  "bins": [482, 483, 484],
  "iq_path": "captures/2026-05-13/uuid.cf32",
  "first_seen_ts": 1715625600.001,
  "last_seen_ts": 1715625600.123
}
```

## 7. Identification Pipeline

Order of attempts on a new appearance:

1. **rtl_433 fingerprinting** — if the detection's center frequency is
   within a band rtl_433 supports (low-VHF/UHF/433/868/915/etc), spawn
   `rtl_433 -r <iq_path>` or, with a live SDR, `rtl_433 -f <center>`.
   Parse JSON output. On a hit, emit an `Identification` event with
   the decoded device class (e.g., `Acurite-Tower`, `LaCrosse-TX141`).
2. **URH-style demod** — call out to `urh_cli` (or, in v2,
   GNU-Radio-based custom flowgraph) to attempt automatic demodulation.
   If a stable bit stream is recovered and matches a known
   protocol-signature library (preamble + sync word + framing), emit
   an `Identification` with `source: urh`.
3. **Unknown-signal analysis** — see §8.

Steps (1) and (2) run from the I/Q capture in §6.3. Step (1) is
preferred because rtl_433 is fast and high-precision for its supported
device set. If a second SDR is free, step (2) can also be done live
against the live emitter for additional captures.

Identifications include a confidence score (0..1). Sub-threshold
matches (default < 0.7) are kept but flagged `low_confidence` and the
pipeline still advances to deeper analysis.

## 8. Unknown Signal Analysis

When fingerprinting fails (rtl_433 produces no match), the
AnalysisAgent picks up the capture and runs a heuristic pipeline in
`analysis/classifier.py`:

- **Burst segmentation** — smoothed-envelope thresholding. Bursts
  shorter than `min_duration_ms` are dropped; bursts separated by
  ≤ `merge_gap_ms` of silence are merged. Continuous emitters
  return a single full-capture "burst".
- **Bandwidth refinement** — windowed FFT of the longest burst,
  power-vs-frequency profile, report `bw_3db_hz`, `bw_6db_hz`,
  `bw_20db_hz`. DC ±5 kHz is masked to suppress the RTL-SDR's
  center spur from biasing the measurement.
- **Frequency offset** — coarse offset = location of the FFT peak
  from tune center, useful for FSK and for confirming the emitter
  isn't a spur sitting near the tune.
- **Modulation classification** — heuristic:
  - Narrow occupied bandwidth + low envelope coefficient-of-
    variation → **CW** (carrier-only).
  - High envelope CV (≥ 0.6) → **OOK / ASK** (bimodal envelope).
  - Low envelope CV + multimodal instantaneous-frequency histogram
    → **FSK**.
  - Low envelope CV + smooth IF spread → **FM**.
  - Otherwise → **unknown**.
  Higher-order-cumulant PSK distinguishing is on the roadmap but
  not in v0.1.
- **Symbol rate estimation** — envelope edge-spacing 25th percentile.
  Smooth the envelope, threshold at midpoint, detect rising/falling
  edges, take the 25th percentile of inter-edge intervals as the
  symbol period. This works on random NRZ patterns where the
  spectral-line / cyclostationary approach finds no clear peak.
- **Re-observation request** — not yet wired. (Future: ask the
  scanner to extend the next capture for an ambiguous emitter.)
- **Artifacts** — currently the raw `.cf32` capture is the only
  artifact preserved. Spectrogram PNG, demodulated bit-stream, and
  a JSON classification report are future work.

Output: `AnalysisReport` event referencing the originating detection
id, picked up by all notification sinks.

## 9. Agent Model

Agents are concurrent asyncio tasks, each owning its own state and
communicating exclusively via the event bus. In v1 the bus is in-
process; the abstraction is preserved so v2 can move to NATS or Redis
Streams.

| Agent              | Subscribes to            | Publishes              | SDR? |
|--------------------|--------------------------|------------------------|------|
| Scanner            | `control.*`              | `sweep.frame`          | yes  |
| Baseline/Detector  | `sweep.frame`            | `detection.*`          | no   |
| Orchestrator       | `detection.*`, `device.*`| `job.*`, `control.*`   | no   |
| Fingerprint        | `job.fingerprint`        | `identification.*`     | yes\*|
| Unknown Analyzer   | `job.analyze`            | `analysis.report`      | yes\*|
| Notifier           | everything (configurable)| (egress)               | no   |

\* Live tune optional — capture-only path doesn't need an SDR.

Each agent type can be instantiated multiple times. With two HackRFs
the operator might run one `Scanner` per device or run one `Scanner`
and one `UnknownAnalyzer` simultaneously — the orchestrator binds
devices to agents through the Device Manager at startup or
dynamically as roles allow.

## 10. Device Allocation Policy

Orchestrator policy:

1. **Steady state**: all SDRs in `scan` or `auto` role are scanning.
2. **On detection**:
   - If a device is in `analyze` role and idle → assign analysis job.
   - Else if an `auto` device exists and at least one other device is
     still scanning the affected range → reassign the `auto` device.
   - Else (only one SDR total, or losing the device would blackout the
     range) → optionally pause scanning per `pause_for_analysis: true`
     (default true for single-SDR setups, false otherwise).
3. **After analysis**: device returns to its prior role.

If multiple detections fire concurrently, jobs queue with priority:
`appearance > disappearance` by default, configurable.

## 11. Notifications

Sinks implemented in v1:

- **stdout** — human-readable lines
- **jsonl** — newline-delimited JSON to a file (rotated daily)
- **webhook** — POST JSON to a URL, with retry + dead-letter
- **syslog** — RFC5424 over UDP/TCP

Each sink has a **filter** (severity, event types, frequency ranges).
A sink may subscribe to: `detection.appearance`, `detection.disappearance`,
`identification.*`, `analysis.report`, `device.*`, `error.*`.

## 12. Storage

- **SQLite** (`riotduck.db`) with tables:
  - `devices` — discovered devices, last seen
  - `ranges` — active range configurations
  - `baselines` — periodic snapshots
  - `detections` — every detection event
  - `identifications` — fingerprint hits
  - `analyses` — analyzer reports
  - `captures` — paths + metadata to I/Q files
- **Filesystem**:
  - `captures/YYYY-MM-DD/<uuid>.cf32` — raw I/Q (complex float32)
  - `captures/YYYY-MM-DD/<uuid>.png` — spectrogram
  - `baselines/<range>/<ts>.npz` — full baseline snapshot
- Retention policy is configurable (default: 14 days for I/Q, 90 days
  for SQLite events).

## 13. Configuration

A single YAML file (default: `~/.config/riotduck/config.yaml`),
override-able with `--config`. Skeleton:

```yaml
devices:
  - serial: "00000001"
    type:   rtlsdr
    role:   scan
    antenna: discone
  - serial: "457863dc2eXXXX"
    type:   hackrf
    role:   auto

ranges:
  - use: ism_433        # reference predefined
    overrides:
      bin_hz: 2000
      short_burst: true
  - name: my_local_unlicensed
    f_start: 902.0e6
    f_end:   928.0e6
    bin_hz:  10000
    dwell_ms: 30

detection:
  k_up: 8
  k_down: 3
  n_up: 3
  n_down: 5
  warmup_min: 60

orchestrator:
  pause_for_analysis: auto    # auto|true|false; auto = true iff 1 SDR
  re_observation_s: 30
  capture_ms: 500

identification:
  rtl_433:
    enabled: true
    binary: rtl_433
  urh:
    enabled: true
    binary: urh_cli

notify:
  - sink: stdout
  - sink: jsonl
    path: /var/log/riotduck/events.jsonl
  - sink: webhook
    url: https://hooks.example.com/riotduck
    filter:
      types: [appearance, disappearance, identification]
```

## 14. CLI

```
riotduck scan         # run continuous scan+detect+notify (the main loop)
riotduck devices      # list discovered SDRs, capabilities, current role
riotduck ranges       # list configured + predefined ranges
riotduck baseline     # one-shot: build a baseline for N seconds and dump
riotduck capture      # one-shot: capture I/Q at f, bw, dur to file
riotduck analyze      # offline: run id/analysis pipeline on an .cf32 file
riotduck replay       # offline: replay a baseline.npz through detection
riotduck doctor       # environment check: SoapySDR, rtl_433, urh_cli, perms
```

Global flags: `--config`, `--log-level`, `--db`, `--captures-dir`.

## 15. Operational Concerns

- **Calibration** — RTL-SDR PPM offset, HackRF DC bias. `riotduck doctor`
  exposes a `--calibrate` mode that uses a known reference (FM bcast or
  user-supplied frequency) to nudge PPM.
- **Antenna selection** — out of scope; antenna is metadata only. We
  do not switch antennas. Operator is responsible for picking sensible
  ranges for the attached antenna.
- **Gain** — per-range gain overrides; default to AGC on RTL-SDR if no
  override. Document the trap of AGC inflating baselines on quiet bands.
- **Time** — all timestamps are UTC unix seconds, float, host clock.
  NTP is the operator's responsibility.
- **Permissions** — under Linux, ship a udev rules sample for both
  devices so users don't run as root. `doctor` reports if rules are
  missing.
- **Logging** — structured (loguru or stdlib `logging` w/ JSON
  formatter). Component name as a contextual field.

## 16. Implementation Plan / Phases

### Phase 1 — Baseline (MVP) ✅ complete
- Project skeleton, config loader, device discovery via SoapySDR
  (with pyrtlsdr fallback). ✅
- RTL-SDR + HackRF SoapySDR streaming backends. ✅
- Synthetic SDR backend for hardware-free testing. ✅
- Native sweep planner + executor + rolling baseline + per-bin
  detector with hysteresis. ✅
- Re-observation / dedup tracker. ✅
- Notification sinks: stdout, JSONL, webhook. ✅
- CLI: `scan`, `devices`, `ranges`, `doctor`, `capture`. ✅

### Phase 2 — Identification ✅ complete
- Inline I/Q capture on appearance (cf32, day-partitioned). ✅
- Pre-detection ring buffer: capture from in-memory sweep I/Q so
  the burst is actually in the file. ✅
- rtl_433 subprocess integration + JSON parsing. ✅
- FingerprintAgent on the bus emitting Identification events. ✅
- Offline `riotduck analyze <file>` against the same pipeline. ✅
- Validated end-to-end on a real RTL-SDR + a 433.92 MHz transmitter. ✅
- SQLite event tables: deferred — JSONL is the durable log for now.

### Phase 4 (early) — Unknown-signal analysis ✅ in
- AnalysisAgent subscribes to rtl_433 misses. ✅
- Burst segmentation (envelope thresholding, smoothing, merging). ✅
- Bandwidth measurement at -3/-6/-20 dB with DC mask. ✅
- Modulation classification: CW / OOK / FSK / FM / unknown. ✅
- Symbol-rate estimation via envelope edge-spacing (works on
  random NRZ where cyclostationary approaches fail). ✅
- AnalysisReport published on the bus, picked up by all
  notification sinks. ✅
- Validated against a real transmitter whose protocol rtl_433
  doesn't decode: correctly reported OOK, ~6.5 kbit/s, ~940 Hz
  carrier BW, ~50 kHz freq offset. ✅

### Phase 3 — HackRF fast path + URH
- HackRF SoapySDR streaming validated on real hardware (path
  exists; no rig tested yet).
- `hackrf_sweep` subprocess parser as the fast path for ranges
  > 100 MHz wide.
- URH demod integration (confirm `urh_cli` vs URH-NG surface first).
- Live-mode rtl_433 as an alternative to file-mode replay.

### Phase 4 (rest) — Multi-SDR orchestration
- Dedicated CaptureAgent under Orchestrator with device role
  assignment policy (§10) and dynamic reassignment on detection.
- Cross-agent dedup (currently per-agent; multi-SDR with
  overlapping ranges can publish duplicates).
- PSK distinguishing in the classifier (currently coarsely lumped
  under "unknown" or "FSK").
- Persistent baselines across restarts.

### Phase 5 — Polish
- SQLite event store with retention policies.
- Web dashboard (FastAPI + websockets).
- Additional sinks: MQTT, Slack, email, syslog.
- `riotduck replay` for offline detection tuning against a recorded
  baseline.
- Sidecar `.meta.json` next to each `.cf32` so offline `analyze`
  / `replay` doesn't need to guess sample rate or center.

## 16a. Real-hardware findings + next steps

Things uncovered by the first RTL-SDR runs that informed v0.1.1:

1. **pyrtlsdr 0.4 / librtlsdr 2.x mismatch.** Stock Osmocom
   librtlsdr 2.0.2 doesn't export `rtlsdr_set_dithering` which
   pyrtlsdr 0.4.0 imports at module load. Pinned `pyrtlsdr>=0.3,<0.4`
   in the `rtlsdr` extra.
2. **setuptools 81 drops `pkg_resources`.** pyrtlsdr 0.3.x still
   uses it. Pinned `setuptools<81` in the `rtlsdr` extra.
3. **LIBUSB_ERROR_OVERFLOW on unaligned trailing reads.** pyrtlsdr's
   `read_samples` does no internal chunking. The session wrapper now
   always reads aligned `131_072`-sample chunks and trims.
4. **NotificationSink was dropping identification events.** The old
   loop created fresh queue waiter tasks each iteration and only
   awaited the first to complete; orphaned waiters consumed events
   that were never observed. Now one drain task per subscription.
5. **Post-detection capture was missing the burst.** A short burst
   ends before the SDR finishes retuning. The Scanner now retains
   the sweep's tune I/Q so the agent dumps the same samples that
   produced the FFT spike — burst contents preserved.
6. **rtl_433 from April 2021 was on `$PATH`** (shadowing brew's
   2025.12). Spec assumption that whatever's on PATH is current is
   wrong; `riotduck doctor` should grow a version check.

What I'd tackle next, in rough order:

1. **`riotduck doctor` rtl_433 version check.** Parse `-V`, warn on
   anything older than 22.x (when many decoders were added) and on
   binaries shadowed by older copies earlier in PATH.
2. **Capture sidecar metadata.** Write `<id>.meta.json` next to each
   `.cf32` with samp_rate / center_hz / device / antenna / config
   snapshot. Today the AnalysisAgent receives this via the
   Identification's `_capture` dict; offline `analyze` and `replay`
   need it on disk.
3. **HackRF on real hardware.** Validate streaming, implement the
   `hackrf_sweep` subprocess parser for wide ranges.
4. **PSK distinguishing in the classifier.** Add a phase-jump
   detector for BPSK/QPSK so we don't lump them under "unknown" or
   "FSK".
5. **URH demod once the URH-NG CLI surface is confirmed.**
6. **Persistent baselines.** `BaselineEngine.snapshot()` already
   produces the npz blob; need save/load + max-age policy in the
   runner.
7. **Orchestrator + CaptureAgent split for multi-SDR.**

## 17. Open Questions

- **URH-NG**: confirm exact binary/CLI surface the user has in mind.
  Spec assumes `urh_cli`; may need a thin GNU-Radio flowgraph instead.
- **Baseline window size**: 300 samples is a guess. Should be tuned
  against real-world false-positive rate on a typical urban spectrum.
- **Bin width vs. emitter bandwidth**: detection currently treats each
  bin independently before coalescing. For very narrow emitters
  (key fobs at 2 kHz deviation) we may want a matched-bandwidth
  detector instead. TBD after Phase 1 measurements.
- **Multi-host federation**: out of scope for v1; mentioned because the
  bus abstraction admits it.
- **Privacy / scope-of-collection**: by default the system records
  I/Q broadly. We should make the retention defaults conservative and
  call out the legal/ethical implications in the README.

## 18. Glossary

- **MAD** — Median Absolute Deviation; robust estimator of scale.
- **CFAR** — Constant False Alarm Rate detection.
- **ISM** — Industrial, Scientific, Medical (unlicensed bands).
- **SRD** — Short Range Devices (European equivalent to US Part 15).
- **Stare mode** — fixed-tune dwell on a narrow range vs. sweeping.
- **Re-observation timeout** — window after a detection during which a
  recurrence is treated as the same event.
