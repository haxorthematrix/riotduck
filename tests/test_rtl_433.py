"""Tests for the rtl_433 wrapper. Subprocess is mocked."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from riotduck.fingerprint.rtl_433 import _parse_jsonl, run_on_file


def test_parse_jsonl_keeps_model_entries():
    stdout = (
        '{"time": "2026-05-13 12:34:56", "model": "Acurite-Tower", "id": 4123, '
        '"channel": "A", "temperature_C": 22.4}\n'
        '{"msg": "rtl_433 startup, no model field"}\n'
        'not json at all\n'
        '{"time": "2026-05-13 12:34:57", "model": "LaCrosse-TX141", "id": 7,'
        '"temperature_C": 19.1}\n'
    )
    hits = _parse_jsonl(stdout)
    assert [h.model for h in hits] == ["Acurite-Tower", "LaCrosse-TX141"]
    assert hits[0].decoded["temperature_C"] == 22.4
    assert hits[0].confidence == 1.0


def test_run_on_file_invokes_with_correct_args(tmp_path: Path):
    iq = tmp_path / "x.cf32"
    iq.write_bytes(b"\x00" * 16)
    fake = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"model": "FakeDevice", "id": 1}\n',
        stderr="",
    )
    with patch("riotduck.fingerprint.rtl_433.subprocess.run", return_value=fake) as r:
        result = run_on_file(iq, samp_rate=2.4e6, center_hz=433.92e6)
    call_args = r.call_args.args[0]
    assert call_args[0] == "rtl_433"
    assert "-F" in call_args and "json" in call_args
    assert "-s" in call_args and "2400000" in call_args
    assert "-f" in call_args and "433920000" in call_args
    assert "-r" in call_args and str(iq) in call_args
    assert result.returncode == 0
    assert len(result.hits) == 1
    assert result.hits[0].model == "FakeDevice"


def test_run_on_file_binary_missing(tmp_path: Path):
    iq = tmp_path / "x.cf32"
    iq.write_bytes(b"\x00" * 16)
    with patch("riotduck.fingerprint.rtl_433.subprocess.run", side_effect=FileNotFoundError()):
        result = run_on_file(iq, samp_rate=2.4e6, binary="rtl_433_missing")
    assert result.returncode == -1
    assert result.hits == []
    assert "not found" in result.stderr.lower()


def test_run_on_file_timeout(tmp_path: Path):
    iq = tmp_path / "x.cf32"
    iq.write_bytes(b"\x00" * 16)
    with patch(
        "riotduck.fingerprint.rtl_433.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="rtl_433", timeout=1.0),
    ):
        result = run_on_file(iq, samp_rate=2.4e6, timeout_s=1.0)
    assert result.returncode == -2
    assert result.hits == []


def test_run_on_file_no_hits_on_garbage(tmp_path: Path):
    iq = tmp_path / "x.cf32"
    iq.write_bytes(b"\x00" * 16)
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="garbage\nmore garbage\n", stderr=""
    )
    with patch("riotduck.fingerprint.rtl_433.subprocess.run", return_value=fake):
        result = run_on_file(iq, samp_rate=2.4e6)
    assert result.returncode == 0
    assert result.hits == []
