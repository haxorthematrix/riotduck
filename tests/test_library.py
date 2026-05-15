"""Tests for the fingerprint library: scoring, YAML I/O, suggestions."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from riotduck.library import (
    Library,
    LibraryEntry,
    LibraryMatch,
    score_entry,
    suggest_yaml,
)


def _entry(**kw) -> LibraryEntry:
    """Build a LibraryEntry with sensible defaults for tests."""
    defaults = dict(
        id="x",
        name="X",
        match=LibraryMatch(
            center_hz=433.92e6,
            center_tolerance_hz=50_000,
            modulation="OOK",
            bw_3db_hz=2200.0,
            bw_3db_tolerance_hz=1500.0,
            symbol_rate_hz=5900.0,
            symbol_rate_tolerance_hz=300.0,
        ),
    )
    defaults.update(kw)
    return LibraryEntry(**defaults)


# ---------- score_entry ----------

def test_score_perfect_hit():
    r = score_entry(
        _entry(),
        center_hz=433.92e6, modulation="OOK", bw_3db_hz=2200.0, symbol_rate_hz=5900.0,
    )
    assert r is not None
    assert r.confidence == pytest.approx(1.0, abs=1e-6)


def test_score_center_outside_tolerance_rejects():
    r = score_entry(
        _entry(),
        center_hz=434.0e6,    # 80 kHz off; tolerance is 50 kHz
        modulation="OOK", bw_3db_hz=2200.0, symbol_rate_hz=5900.0,
    )
    assert r is None


def test_score_modulation_mismatch_rejects():
    r = score_entry(
        _entry(),
        center_hz=433.92e6, modulation="FSK",
        bw_3db_hz=2200.0, symbol_rate_hz=5900.0,
    )
    assert r is None


def test_score_modulation_case_insensitive():
    r = score_entry(
        _entry(),
        center_hz=433.92e6, modulation="ook",       # lower-case
        bw_3db_hz=2200.0, symbol_rate_hz=5900.0,
    )
    assert r is not None


def test_score_bw_outside_tolerance_rejects():
    r = score_entry(
        _entry(),
        center_hz=433.92e6, modulation="OOK",
        bw_3db_hz=5000.0,    # 2800 Hz off; tolerance is 1500
        symbol_rate_hz=5900.0,
    )
    assert r is None


def test_score_symbol_rate_outside_tolerance_rejects():
    r = score_entry(
        _entry(),
        center_hz=433.92e6, modulation="OOK",
        bw_3db_hz=2200.0, symbol_rate_hz=7000.0,    # 1100 off; tol 300
    )
    assert r is None


def test_score_partial_criteria_only_require_what_is_set():
    """Entry without bw_3db / symbol_rate criteria matches without
    requiring those values in the report."""
    e = _entry(
        match=LibraryMatch(
            center_hz=433.92e6, center_tolerance_hz=50_000, modulation="OOK",
        ),
    )
    r = score_entry(
        e,
        center_hz=433.93e6, modulation="OOK",
        bw_3db_hz=None, symbol_rate_hz=None,
    )
    assert r is not None
    assert r.confidence > 0.0


def test_score_partial_criteria_still_rejects_required_modulation():
    e = _entry(
        match=LibraryMatch(
            center_hz=433.92e6, center_tolerance_hz=50_000, modulation="OOK",
        ),
    )
    # Report has no modulation; entry requires OOK → reject
    r = score_entry(e, center_hz=433.92e6,
                    modulation=None, bw_3db_hz=None, symbol_rate_hz=None)
    assert r is None


def test_score_at_tolerance_edge_scores_near_zero():
    """A match that just barely passes its tolerance scores ~0."""
    r = score_entry(
        _entry(),
        center_hz=433.92e6 + 50_000,    # exactly at center tolerance
        modulation="OOK", bw_3db_hz=2200.0, symbol_rate_hz=5900.0,
    )
    assert r is not None
    # one criterion at distance 1.0, others at 0.0; mean(1, 0, 0, 0) = 0.25
    # confidence = 1 - 0.25 = 0.75
    assert r.confidence == pytest.approx(0.75, abs=1e-3)


# ---------- Library.best_match ----------

def test_best_match_picks_highest_confidence():
    lib = Library(entries=[
        _entry(id="far",     match=LibraryMatch(center_hz=433.92e6 + 40_000,
                                                modulation="OOK")),
        _entry(id="close",   match=LibraryMatch(center_hz=433.92e6 + 2_000,
                                                modulation="OOK")),
        _entry(id="rejected", match=LibraryMatch(center_hz=433.92e6,
                                                 modulation="FSK")),
    ])
    r = lib.best_match(center_hz=433.92e6, modulation="OOK",
                       bw_3db_hz=None, symbol_rate_hz=None)
    assert r is not None
    assert r.entry.id == "close"


def test_best_match_returns_none_when_nothing_matches():
    lib = Library(entries=[_entry()])
    r = lib.best_match(center_hz=900e6, modulation="OOK",
                       bw_3db_hz=2200.0, symbol_rate_hz=5900.0)
    assert r is None


# ---------- YAML I/O ----------

def test_load_yaml_round_trip(tmp_path: Path):
    src = tmp_path / "lib.yaml"
    src.write_text(dedent(
        """
        entries:
          - id: thing-a
            name: "Thing A"
            notes: "fancy notes"
            tags: [lab, ook]
            match:
              center_hz: 433.928e+6
              center_tolerance_hz: 30_000
              modulation: OOK
              bw_3db_hz: 2200
              bw_3db_tolerance_hz: 1500
              symbol_rate_hz: 5900
              symbol_rate_tolerance_hz: 300
          - id: thing-b
            name: "Thing B"
            match:
              center_hz: 915e+6
              modulation: FSK
        """
    ))
    lib = Library.load(src)
    assert len(lib) == 2
    a = lib.get("thing-a")
    b = lib.get("thing-b")
    assert a is not None and b is not None
    assert a.match.center_hz == 433.928e6
    assert a.match.modulation == "OOK"
    assert a.tags == ["lab", "ook"]
    assert b.match.center_hz == 915e6
    assert b.match.bw_3db_hz is None

    dst = tmp_path / "out.yaml"
    lib.save(dst)
    lib2 = Library.load(dst)
    assert len(lib2) == 2
    assert lib2.get("thing-a").match.center_hz == 433.928e6


def test_load_missing_file_is_empty(tmp_path: Path):
    lib = Library.load(tmp_path / "does_not_exist.yaml")
    assert len(lib) == 0


def test_load_accepts_pyyaml_unsigned_exponent_form(tmp_path: Path):
    """PyYAML 1.1 requires `e+6` but users will write `e6`; numeric
    coercion runs on every entry's match dict."""
    p = tmp_path / "lib.yaml"
    p.write_text("entries:\n  - id: x\n    match:\n      center_hz: 433.928e6\n")
    lib = Library.load(p)
    assert lib.get("x").match.center_hz == 433.928e6


# ---------- suggest_yaml ----------

def test_suggest_yaml_uses_observed_values():
    snippet = suggest_yaml(
        center_hz=433.928e6,
        modulation="OOK",
        bw_3db_hz=2200.0,
        symbol_rate_hz=5900.0,
    )
    assert "center_hz: 433.928000e+6" in snippet
    assert "modulation: OOK" in snippet
    assert "bw_3db_hz: 2200" in snippet
    assert "symbol_rate_hz: 5900" in snippet


def test_suggest_yaml_omits_missing_fields():
    snippet = suggest_yaml(
        center_hz=915e6, modulation="CW",
        bw_3db_hz=None, symbol_rate_hz=None,
    )
    assert "modulation: CW" in snippet
    assert "bw_3db_hz" not in snippet
    assert "symbol_rate_hz" not in snippet
