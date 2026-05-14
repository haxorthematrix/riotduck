# riotduck

SDR-based RF baseline and anomaly detection.

`riotduck` sweeps user-defined frequency ranges with an RTL-SDR (or
HackRF, or both), learns the steady-state spectral baseline, and
alerts when an emitter **appears** (a new transmitter shows up where
the band was quiet) or **disappears** (a previously-reliable emitter
goes silent — e.g., a covert transmitter being removed or jammed).

On a new appearance, riotduck captures I/Q to disk and hands the
suspect frequency off to an identification pipeline (`rtl_433`,
later URH and an unknown-signal analyzer). Repeat detections at the
same frequency are folded into one logical event with a
re-observation / dedup tracker, so a chatty key fob doesn't generate
an alert storm.

See [`specification.md`](specification.md) for the full design and
current status.

## Status

v0.1 — Phases 1 and 2 are complete and unit-tested against
synthesized I/Q. Real-hardware validation is the next milestone.
46 unit tests; `pytest -q` runs in ~2 s. Try it now without an SDR:

```bash
pip install -e .
riotduck scan --fake 1 --config config/default.yaml
```

## Install

### System dependencies

```bash
# --- Linux (Debian/Ubuntu) -----------------------------------------
sudo apt install rtl-sdr librtlsdr-dev \
                 hackrf libhackrf-dev \
                 soapysdr-tools \
                 soapysdr-module-rtlsdr \
                 soapysdr-module-hackrf \
                 python3-soapysdr \
                 rtl-433

# --- macOS (Homebrew) ----------------------------------------------
brew install librtlsdr hackrf soapysdr soapyrtlsdr soapyhackrf rtl_433
```

Everything except SoapySDR is technically optional — riotduck falls
back to `pyrtlsdr` if SoapySDR isn't present, and `rtl_433` is only
required when you want fingerprint identification. SoapySDR is
strongly recommended because it's the only path that supports both
RTL-SDR and HackRF behind one abstraction.

### Python package

```bash
git clone https://github.com/<your-handle>/riotduck.git
cd riotduck
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # dev extras include pytest/ruff/mypy
```

Minimum Python: 3.11. The `dev` extra is only needed if you want to
run the tests; runtime use just needs `pip install -e .`.

### udev (Linux only)

```bash
sudo cp etc/udev/rules.d/99-riotduck.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo usermod -aG plugdev $USER     # then log out + back in
```

This gives the `plugdev` group access to RTL-SDR and HackRF dongles
so you don't need to run riotduck as root.

## Quick start

```bash
# 1. Confirm the environment sees what it needs.
riotduck doctor

# 2. See discovered SDRs.
riotduck devices

# 3. Browse the predefined ISM / common-allocation range library.
riotduck ranges --predefined

# 4. Run the main detection loop. Ctrl-C to stop.
riotduck scan --config config/default.yaml
```

`config/default.yaml` watches the 433 MHz ISM band and US 902-928 MHz.
Copy it to `~/.config/riotduck/config.yaml` and edit, or keep an
explicit `--config` flag.

## No SDR yet? Use the synthetic backend

```bash
# 1 fake SDR with the built-in emitter profile (keyfob-style 433.92
# burst, steady 915 MHz carrier, drifting 2.44 GHz transmitter).
riotduck scan --fake 1 --config config/default.yaml

# Custom emitter profile:
riotduck scan --fake 1 \
              --fake-profile config/fake_profile.yaml \
              --config config/default.yaml
```

The synthetic backend produces real `complex64` I/Q that flows through
the same scanner → baseline → dedup → capture → fingerprint pipeline,
so detections, captures, and rtl_433 invocations all happen normally.
You'll see something like:

```
[detection.appearance]   ^ ism_433 @ 433.9200 MHz bw=10.0 kHz snr=68.5 dB ...
[detection.disappearance] v ism_433 @ 433.9200 MHz bw=70.0 kHz snr= 0.3 dB ...
[detection.appearance]   ^ ism_433 @ 433.9200 MHz bw=10.0 kHz snr=68.5 dB ...
```

Repeat bursts at the same frequency share an event id — that's the
dedup tracker working. Set `RIOTDUCK_FAKE_DEVICES=2` to exercise
multi-device discovery.

## Offline analysis

`riotduck analyze` runs the identification pipeline against any I/Q
file. The file doesn't need to come from riotduck; rtl_433's test
corpus is a useful regression source.

```bash
riotduck analyze capture.cf32 -s 2400000 -f 433920000
riotduck analyze capture.cf32 -s 2400000 --json    # one JSON per hit
```

Exit code is 2 if rtl_433 isn't on `$PATH` (override with `--binary`).

## Configuration

YAML, schema documented in `specification.md` §13. Common knobs:

```yaml
ranges:
  - use: ism_433              # reference the predefined library
  - name: my_local_lpd        # or define a custom range
    f_start: 433.075e+6
    f_end:   434.775e+6
    bin_hz:  2000
    short_burst: true

detection:
  k_up: 8              # std-devs above bin median to trigger appearance
  n_up: 3              # consecutive sweeps required
  n_down: 5            # consecutive sweeps below noise to declare gone
  warmup_min: 60       # sweeps before any alerts fire
  window_size: 300     # rolling baseline depth (per bin)
  min_freq_tolerance_hz: 5000   # dedup floor (wider for wideband signals)

orchestrator:
  re_observation_s: 30          # repeat-detection fold window
  capture_ms: 500               # I/Q captured per appearance

identification:
  rtl_433:
    enabled: true
    binary: rtl_433
  urh:
    enabled: false              # stubbed for Phase 3

notify:
  - sink: stdout
  - sink: jsonl
    path: events.jsonl
  - sink: webhook
    url: https://hooks.example.com/riotduck
```

## CLI

```
riotduck scan       # main detect+capture+identify loop
riotduck devices    # list discovered SDRs
riotduck ranges     # list configured + predefined ranges
riotduck doctor     # environment / dependency check
riotduck capture    # one-shot I/Q dump to .cf32
riotduck analyze    # offline rtl_433 against a .cf32 file
```

Flags on `scan` and `devices`:
- `--fake N` — bring up N synthetic SDRs
- `--fake-profile <yaml>` — emitter profile for the synthetic SDR
- `--config <path>` — override the user config

## Development

```bash
pip install -e ".[dev]"
pytest -q                                  # full test suite (~2 s)
pytest tests/test_dedup.py -v              # specific module
ruff check src/                            # lint
```

The tests cover config loading, FFT/DSP correctness, the detection
state machine, the dedup tracker, the rtl_433 wrapper (mocked
subprocess), the analyze CLI, the fingerprint agent, the inline
capture step, and the synthetic SDR backend. They do not require any
SDR hardware or the `rtl_433` binary.

## Layout

```
src/riotduck/
  config.py            # pydantic models + YAML loader
  events.py            # bus payload dataclasses
  bus.py               # in-process asyncio pub/sub
  dsp.py               # FFT, windowing, decimation, dBFS helpers
  scanner.py           # sweep planning + execution
  baseline.py          # per-bin rolling stats + state machine
  dedup.py             # re-observation / event tracker
  capture.py           # inline I/Q capture on detection
  runner.py            # top-level wiring (bus + agents + sinks)
  cli.py               # click entry points
  sdr/
    base.py            # SDRBackend / SDRSession interfaces
    rtlsdr.py          # SoapySDR + pyrtlsdr backends
    hackrf.py          # SoapySDR HackRF backend
    fake.py            # synthetic emitter SDR (no hardware)
    manager.py         # discovery + reservation
  agents/
    base.py            # async Agent lifecycle
    scanner_agent.py   # one per scanning SDR
    fingerprint_agent.py
  notify/              # stdout / jsonl / webhook sinks
  fingerprint/         # rtl_433 wrapper, URH stub
  storage/             # cf32 capture file layout
  analysis/            # unknown-signal classifier (Phase 4 stub)
config/
  default.yaml         # default user config
  ism_bands.yaml       # predefined range library (18 bands)
  fake_profile.yaml    # sample synthetic emitter profile
etc/udev/rules.d/
  99-riotduck.rules    # Linux device permissions
tests/                 # 46 unit tests, no hardware required
```

## Legal note

riotduck *records* whatever the attached antenna picks up. Recording
broadcast or licensed traffic for analysis may be regulated in your
jurisdiction; the operator is responsible for compliance. Default
I/Q retention is 14 days (`storage.retain_iq_days`); tune that and
the configured ranges to match your operational and legal constraints.

## License

MIT.
