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
