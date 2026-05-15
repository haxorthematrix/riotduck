"""Tests for the `riotduck analyze` CLI subcommand."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from riotduck.cli import main
from riotduck.fingerprint.rtl_433 import Rtl433Hit, Rtl433Result
from riotduck.storage.files import sidecar_path_for


@pytest.fixture
def dummy_cf32(tmp_path: Path) -> Path:
    p = tmp_path / "x.cf32"
    np.full(1024, 0, dtype=np.complex64).tofile(str(p))
    return p


def test_analyze_missing_binary(dummy_cf32: Path):
    runner = CliRunner()
    with patch("riotduck.cli.shutil.which", return_value=None):
        r = runner.invoke(
            main,
            ["analyze", str(dummy_cf32), "-s", "250000", "--binary", "no_such_rtl_433"],
        )
    assert r.exit_code == 2
    assert "not found" in r.output


def test_analyze_no_hits_table(dummy_cf32: Path):
    runner = CliRunner()
    fake = Rtl433Result(returncode=0, hits=[])
    with patch("riotduck.cli.shutil.which", return_value="/fake/rtl_433"), \
         patch("riotduck.fingerprint.rtl_433.run_on_file", return_value=fake):
        r = runner.invoke(main, ["analyze", str(dummy_cf32), "-s", "2400000"])
    assert r.exit_code == 0
    assert "no rtl_433 hits" in r.output


def test_analyze_renders_hits(dummy_cf32: Path):
    runner = CliRunner()
    fake = Rtl433Result(
        returncode=0,
        hits=[
            Rtl433Hit(model="Acurite-Tower", decoded={"model": "Acurite-Tower",
                                                       "id": 7, "temperature_C": 22.4}),
        ],
    )
    with patch("riotduck.cli.shutil.which", return_value="/fake/rtl_433"), \
         patch("riotduck.fingerprint.rtl_433.run_on_file", return_value=fake):
        r = runner.invoke(
            main,
            ["analyze", str(dummy_cf32), "-s", "2400000", "-f", "433920000"],
        )
    assert r.exit_code == 0
    assert "Acurite-Tower" in r.output
    assert "temperature_C=22.4" in r.output


def test_analyze_reads_sidecar_when_flags_omitted(dummy_cf32: Path):
    """No -s / -f flags but a sidecar exists → values come from sidecar."""
    sidecar_path_for(dummy_cf32).write_text(json.dumps({
        "schema_version": 1,
        "samp_rate": 1_024_000.0,
        "capture_center_hz": 433_920_000.0,
        "duration_s": 0.01,
        "iq_path": dummy_cf32.name,
        "detection": {},
    }))
    runner = CliRunner()
    fake = Rtl433Result(returncode=0, hits=[])
    with patch("riotduck.cli.shutil.which", return_value="/fake/rtl_433"), \
         patch("riotduck.fingerprint.rtl_433.run_on_file", return_value=fake) as ran:
        r = runner.invoke(main, ["analyze", str(dummy_cf32)])
    assert r.exit_code == 0, r.output
    assert "loaded from sidecar" in r.output
    # Verify the values flowed through to run_on_file.
    kwargs = ran.call_args.kwargs
    assert kwargs["samp_rate"] == 1_024_000.0
    assert kwargs["center_hz"] == 433_920_000.0


def test_analyze_explicit_flags_override_sidecar(dummy_cf32: Path):
    sidecar_path_for(dummy_cf32).write_text(json.dumps({
        "schema_version": 1,
        "samp_rate": 1_024_000.0,
        "capture_center_hz": 433_920_000.0,
        "duration_s": 0.01,
        "iq_path": dummy_cf32.name,
        "detection": {},
    }))
    runner = CliRunner()
    fake = Rtl433Result(returncode=0, hits=[])
    with patch("riotduck.cli.shutil.which", return_value="/fake/rtl_433"), \
         patch("riotduck.fingerprint.rtl_433.run_on_file", return_value=fake) as ran:
        r = runner.invoke(
            main,
            ["analyze", str(dummy_cf32), "-s", "2400000", "-f", "915000000"],
        )
    assert r.exit_code == 0, r.output
    kwargs = ran.call_args.kwargs
    assert kwargs["samp_rate"] == 2_400_000.0
    assert kwargs["center_hz"] == 915_000_000.0


def test_analyze_errors_without_samp_rate_or_sidecar(dummy_cf32: Path):
    """No sidecar and no -s flag → clean exit-code-2 error."""
    runner = CliRunner()
    r = runner.invoke(main, ["analyze", str(dummy_cf32)])
    assert r.exit_code == 2
    assert "no sample rate" in r.output.lower()


def test_analyze_json_mode(dummy_cf32: Path):
    runner = CliRunner()
    fake = Rtl433Result(
        returncode=0,
        hits=[Rtl433Hit(model="LaCrosse", decoded={"model": "LaCrosse", "id": 12},
                       confidence=1.0)],
    )
    with patch("riotduck.cli.shutil.which", return_value="/fake/rtl_433"), \
         patch("riotduck.fingerprint.rtl_433.run_on_file", return_value=fake):
        r = runner.invoke(
            main,
            ["analyze", str(dummy_cf32), "-s", "2400000", "--json"],
        )
    assert r.exit_code == 0
    # Find the JSON line in the output.
    json_lines = [
        l for l in r.output.splitlines()
        if l.startswith("{") and "model" in l
    ]
    assert len(json_lines) == 1
    parsed = json.loads(json_lines[0])
    assert parsed["model"] == "LaCrosse"
    assert parsed["confidence"] == 1.0
