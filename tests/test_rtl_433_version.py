"""Tests for the rtl_433 version probe and alternative-install detection."""
from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

from riotduck.fingerprint.rtl_433 import (
    RTL_433_MIN_RECOMMENDED,
    Rtl433Info,
    _VERSION_RE,
    _all_rtl_433_on_path,
    get_rtl_433_info,
    probe_rtl_433_version,
)


def _completed(stdout: str = "", stderr: str = "", rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


# ---------- regex ----------

def test_version_re_modern():
    m = _VERSION_RE.search("rtl_433 version 25.12 (2025-12-12) inputs file rtl_tcp")
    assert m and m.group(1) == "25" and m.group(2) == "12"


def test_version_re_git_revision():
    m = _VERSION_RE.search(
        "rtl_433 version 20.11-118-g836bf756 branch master at 202104272025"
    )
    assert m and m.group(1) == "20" and m.group(2) == "11"


def test_version_re_no_match():
    assert _VERSION_RE.search("hello there, no version string here") is None


# ---------- probe_rtl_433_version ----------

def test_probe_returns_tuple():
    with patch(
        "riotduck.fingerprint.rtl_433.subprocess.run",
        return_value=_completed(stderr="rtl_433 version 25.12 (2025-12-12)"),
    ):
        assert probe_rtl_433_version("rtl_433") == (25, 12)


def test_probe_returns_none_when_unparseable():
    with patch(
        "riotduck.fingerprint.rtl_433.subprocess.run",
        return_value=_completed(stdout="hello\nworld\n"),
    ):
        assert probe_rtl_433_version("rtl_433") is None


def test_probe_returns_none_when_missing():
    with patch(
        "riotduck.fingerprint.rtl_433.subprocess.run",
        side_effect=FileNotFoundError(),
    ):
        assert probe_rtl_433_version("not_real") is None


def test_probe_returns_none_on_timeout():
    with patch(
        "riotduck.fingerprint.rtl_433.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="rtl_433", timeout=1.0),
    ):
        assert probe_rtl_433_version("rtl_433") is None


# ---------- get_rtl_433_info ----------

def test_get_info_missing_binary():
    with patch("riotduck.fingerprint.rtl_433.shutil.which", return_value=None), \
         patch("riotduck.fingerprint.rtl_433._all_rtl_433_on_path", return_value=[]), \
         patch("riotduck.fingerprint.rtl_433._brew_rtl_433_path", return_value=None):
        info = get_rtl_433_info()
    assert not info.installed
    assert info.version is None
    assert info.shadows == []


def test_get_info_modern_version_passes():
    with patch("riotduck.fingerprint.rtl_433.shutil.which", return_value="/usr/bin/rtl_433"), \
         patch("riotduck.fingerprint.rtl_433._all_rtl_433_on_path", return_value=["/usr/bin/rtl_433"]), \
         patch("riotduck.fingerprint.rtl_433._brew_rtl_433_path", return_value=None), \
         patch("riotduck.fingerprint.rtl_433.subprocess.run",
               return_value=_completed(stderr="rtl_433 version 25.12 (2025-12-12)")):
        info = get_rtl_433_info()
    assert info.installed
    assert info.version == (25, 12)
    assert info.is_stale is False
    assert info.shadows == []


def test_get_info_stale_version_flagged():
    """The 2021 binary triggers the stale flag."""
    with patch("riotduck.fingerprint.rtl_433.shutil.which",
               return_value="/usr/local/bin/rtl_433"), \
         patch("riotduck.fingerprint.rtl_433._all_rtl_433_on_path",
               return_value=["/usr/local/bin/rtl_433"]), \
         patch("riotduck.fingerprint.rtl_433._brew_rtl_433_path", return_value=None), \
         patch(
             "riotduck.fingerprint.rtl_433.subprocess.run",
             return_value=_completed(
                 stderr="rtl_433 version 20.11-118-g836bf756 branch master at 202104272025"
             ),
         ):
        info = get_rtl_433_info()
    assert info.version == (20, 11)
    assert info.is_stale
    assert info.version < RTL_433_MIN_RECOMMENDED


def test_get_info_detects_brew_shadow(tmp_path):
    """When the brew binary exists but isn't on PATH, it's a shadow."""
    # Use a real path so realpath comparisons work.
    active = tmp_path / "stale_rtl_433"
    active.write_text("")
    active.chmod(0o755)
    brew_bin = tmp_path / "brew" / "bin" / "rtl_433"
    brew_bin.parent.mkdir(parents=True)
    brew_bin.write_text("")
    brew_bin.chmod(0o755)

    with patch("riotduck.fingerprint.rtl_433.shutil.which", return_value=str(active)), \
         patch("riotduck.fingerprint.rtl_433._all_rtl_433_on_path",
               return_value=[str(active)]), \
         patch("riotduck.fingerprint.rtl_433._brew_rtl_433_path",
               return_value=str(brew_bin)), \
         patch(
             "riotduck.fingerprint.rtl_433.subprocess.run",
             return_value=_completed(stderr="rtl_433 version 20.11"),
         ):
        info = get_rtl_433_info()
    assert info.is_stale
    assert str(brew_bin) in info.shadows


# ---------- _all_rtl_433_on_path ----------

def test_all_paths_walks_PATH_in_order(tmp_path, monkeypatch):
    """Path order is preserved; duplicates by realpath are removed."""
    dir_a = tmp_path / "a" / "bin"
    dir_b = tmp_path / "b" / "bin"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    bin_a = dir_a / "rtl_433"
    bin_b = dir_b / "rtl_433"
    bin_a.write_text("")
    bin_b.write_text("")
    bin_a.chmod(0o755)
    bin_b.chmod(0o755)

    monkeypatch.setenv("PATH", os.pathsep.join([str(dir_a), str(dir_b)]))
    found = _all_rtl_433_on_path()
    assert found == [str(bin_a), str(bin_b)]


def test_all_paths_dedups_symlinks(tmp_path, monkeypatch):
    """A symlink to the same realpath isn't counted twice."""
    real_dir = tmp_path / "real" / "bin"
    link_dir = tmp_path / "link" / "bin"
    real_dir.mkdir(parents=True)
    link_dir.mkdir(parents=True)
    real_bin = real_dir / "rtl_433"
    real_bin.write_text("")
    real_bin.chmod(0o755)
    link_bin = link_dir / "rtl_433"
    link_bin.symlink_to(real_bin)

    monkeypatch.setenv("PATH", os.pathsep.join([str(real_dir), str(link_dir)]))
    found = _all_rtl_433_on_path()
    # Whichever appears first is kept; the symlink to the same realpath is dropped.
    assert len(found) == 1
