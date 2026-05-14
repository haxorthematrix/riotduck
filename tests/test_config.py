from pathlib import Path
from textwrap import dedent

import pytest

from riotduck.config import RangeConfig, load_config, load_predefined_ranges


def test_predefined_library_loads():
    lib = load_predefined_ranges()
    assert "ism_433" in lib
    assert lib["ism_433"]["f_start"] < lib["ism_433"]["f_end"]


def test_load_config_resolves_use(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        dedent(
            """
            ranges:
              - use: ism_433
                overrides:
                  bin_hz: 2000
                  short_burst: true
              - name: custom
                f_start: 100.0e6
                f_end: 100.5e6
                bin_hz: 1000
            """
        )
    )
    cfg = load_config(cfg_path)
    assert len(cfg.ranges) == 2
    ism, custom = cfg.ranges
    assert isinstance(ism, RangeConfig)
    assert ism.name == "ism_433"
    assert ism.bin_hz == 2000
    assert ism.short_burst is True
    assert custom.name == "custom"


def test_load_config_unknown_use(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("ranges:\n  - use: not_a_band\n")
    with pytest.raises(ValueError, match="unknown predefined band"):
        load_config(cfg_path)


def test_range_validation():
    with pytest.raises(ValueError):
        RangeConfig(name="x", f_start=200e6, f_end=100e6)
    with pytest.raises(ValueError):
        RangeConfig(name="x", f_start=100e6, f_end=200e6, bin_hz=0)
