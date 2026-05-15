"""Tests for `riotduck library list/show/add/remove`."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import numpy as np
from click.testing import CliRunner

from riotduck.cli import main
from riotduck.library import Library
from riotduck.storage.files import sidecar_path_for


SAMPLE = dedent(
    """
    entries:
      - id: alpha
        name: "Alpha device"
        notes: "first test entry"
        tags: [test, ook]
        match:
          center_hz: 433.92e+6
          modulation: OOK
          bw_3db_hz: 2200
          symbol_rate_hz: 5900
      - id: beta
        match:
          center_hz: 915e+6
          modulation: FSK
    """
)


def test_library_list_empty(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("entries: []\n")
    r = CliRunner().invoke(main, ["library", "list", "--path", str(p)])
    assert r.exit_code == 0
    assert "empty" in r.output


def test_library_list_renders_entries(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    p.write_text(SAMPLE)
    r = CliRunner().invoke(main, ["library", "list", "--path", str(p)])
    assert r.exit_code == 0
    assert "alpha" in r.output
    assert "Alpha device" in r.output
    assert "OOK" in r.output
    assert "beta" in r.output
    assert "FSK" in r.output


def test_library_show_existing(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    p.write_text(SAMPLE)
    r = CliRunner().invoke(main, ["library", "show", "alpha", "--path", str(p)])
    assert r.exit_code == 0
    assert "alpha" in r.output
    assert "Alpha device" in r.output
    assert "OOK" in r.output
    assert "5900" in r.output
    assert "first test entry" in r.output


def test_library_show_missing(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    p.write_text(SAMPLE)
    r = CliRunner().invoke(main, ["library", "show", "no_such", "--path", str(p)])
    assert r.exit_code == 1
    assert "no entry" in r.output


# ---------- add ----------

def test_library_add_explicit_fields(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    r = CliRunner().invoke(main, [
        "library", "add",
        "--id", "remote-1",
        "--name", "Garage Remote",
        "--center", "433920000",
        "--modulation", "ook",
        "--bw-hz", "8000",
        "--symbol-rate-hz", "5900",
        "--tag", "lab", "--tag", "ook",
        "--path", str(p),
    ])
    assert r.exit_code == 0, r.output
    assert "added" in r.output
    lib = Library.load(p)
    assert len(lib) == 1
    e = lib.get("remote-1")
    assert e is not None
    assert e.name == "Garage Remote"
    assert e.tags == ["lab", "ook"]
    assert e.match.center_hz == 433_920_000
    assert e.match.modulation == "OOK"      # normalized upper
    assert e.match.bw_3db_hz == 8000
    assert e.match.symbol_rate_hz == 5900


def test_library_add_requires_center(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    r = CliRunner().invoke(main, [
        "library", "add", "--id", "x", "--path", str(p),
    ])
    assert r.exit_code == 2
    assert "center frequency required" in r.output


def test_library_add_rejects_duplicate(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    CliRunner().invoke(main, [
        "library", "add", "--id", "dup", "--center", "433920000",
        "--path", str(p),
    ])
    r = CliRunner().invoke(main, [
        "library", "add", "--id", "dup", "--center", "915000000",
        "--path", str(p),
    ])
    assert r.exit_code == 1
    assert "already exists" in r.output
    # Untouched.
    assert Library.load(p).get("dup").match.center_hz == 433_920_000


def test_library_add_replace_overwrites(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    CliRunner().invoke(main, [
        "library", "add", "--id", "dup", "--center", "433920000",
        "--path", str(p),
    ])
    r = CliRunner().invoke(main, [
        "library", "add", "--id", "dup", "--center", "915000000",
        "--replace", "--path", str(p),
    ])
    assert r.exit_code == 0
    lib = Library.load(p)
    assert len(lib) == 1
    assert lib.get("dup").match.center_hz == 915_000_000


def _write_ook_capture(tmp_path: Path) -> Path:
    """Synthesize a deterministic OOK-ish burst capture + sidecar."""
    samp_rate = 1_024_000.0
    n = 32768
    rng = np.random.default_rng(0)
    iq = (rng.normal(scale=0.005, size=n)
          + 1j * rng.normal(scale=0.005, size=n)).astype(np.complex64)
    # A few "on" bursts in the middle.
    for start in (4000, 9000, 14000, 19000, 24000):
        iq[start:start + 2000] += np.complex64(1.0)
    cf32 = tmp_path / "burst.cf32"
    iq.tofile(str(cf32))
    sidecar_path_for(cf32).write_text(json.dumps({
        "schema_version": 1,
        "samp_rate": samp_rate,
        "capture_center_hz": 433_920_000.0,
        "duration_s": n / samp_rate,
        "iq_path": cf32.name,
        "detection": {},
    }))
    return cf32


def test_library_add_from_capture_pulls_sidecar(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    cf32 = _write_ook_capture(tmp_path)
    r = CliRunner().invoke(main, [
        "library", "add", "--id", "burst-1",
        "--from-capture", str(cf32),
        "--path", str(p),
    ])
    assert r.exit_code == 0, r.output
    assert "analyzer:" in r.output
    e = Library.load(p).get("burst-1")
    assert e is not None
    # Center pulled from sidecar.
    assert e.match.center_hz == 433_920_000.0


def test_library_add_from_capture_cli_overrides_analyzer(tmp_path: Path):
    """Explicit --bw-hz / --modulation override whatever the analyzer guessed."""
    p = tmp_path / "lib.yaml"
    cf32 = _write_ook_capture(tmp_path)
    r = CliRunner().invoke(main, [
        "library", "add", "--id", "ovr",
        "--from-capture", str(cf32),
        "--modulation", "FSK",
        "--bw-hz", "12345",
        "--path", str(p),
    ])
    assert r.exit_code == 0, r.output
    e = Library.load(p).get("ovr")
    assert e.match.modulation == "FSK"
    assert e.match.bw_3db_hz == 12345


# ---------- remove ----------

def test_library_remove_existing(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    p.write_text(SAMPLE)
    r = CliRunner().invoke(main, ["library", "remove", "alpha", "--path", str(p)])
    assert r.exit_code == 0
    assert "removed" in r.output
    lib = Library.load(p)
    assert lib.get("alpha") is None
    assert lib.get("beta") is not None


def test_library_remove_missing(tmp_path: Path):
    p = tmp_path / "lib.yaml"
    p.write_text(SAMPLE)
    r = CliRunner().invoke(main, ["library", "remove", "ghost", "--path", str(p)])
    assert r.exit_code == 1
    assert "no entry" in r.output
