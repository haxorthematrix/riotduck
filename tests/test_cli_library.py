"""Tests for `riotduck library list/show`."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from riotduck.cli import main


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
