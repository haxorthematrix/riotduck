"""Configuration models for riotduck.

YAML config is loaded into Pydantic models. Predefined ranges in
config/ism_bands.yaml are merged with user-defined ranges; user
entries may either define a new range or reference a predefined one
by name with `use:` and optional `overrides:`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

DeviceType = Literal["rtlsdr", "hackrf", "auto"]
DeviceRole = Literal["scan", "analyze", "auto"]
WindowKind = Literal["hann", "hamming", "blackman", "blackmanharris", "rect"]


class DeviceConfig(BaseModel):
    serial: str
    type: DeviceType = "auto"
    role: DeviceRole = "auto"
    antenna: str | None = None


class GainConfig(BaseModel):
    """Free-form gain overrides; per-device stages.

    Common keys: lna, vga, mixer, if, tuner. Validated lazily against
    device capabilities at acquisition time.
    """

    lna: int | None = None
    vga: int | None = None
    mixer: int | None = None
    tuner: float | None = None
    if_gain: int | None = Field(default=None, alias="if")

    model_config = {"populate_by_name": True, "extra": "allow"}


class RangeConfig(BaseModel):
    name: str
    f_start: float
    f_end: float
    bin_hz: float = 4000.0
    dwell_ms: float = 50.0
    window: WindowKind = "hann"
    samp_rate: float | None = None     # None → device default / auto
    gain: GainConfig = Field(default_factory=GainConfig)
    repeat_s: float = 1.0
    short_burst: bool = False
    min_bw_hz: float = 0.0
    description: str | None = None

    @field_validator("f_end")
    @classmethod
    def _f_end_after_start(cls, v: float, info: Any) -> float:
        start = info.data.get("f_start")
        if start is not None and v <= start:
            raise ValueError("f_end must be greater than f_start")
        return v

    @field_validator("bin_hz")
    @classmethod
    def _bin_hz_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("bin_hz must be positive")
        return v


class RangeRef(BaseModel):
    """User config entry that references a predefined range by name.

    Optional `overrides` are deep-merged onto the predefined config.
    """

    use: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class DetectionConfig(BaseModel):
    k_up: float = 8.0
    k_down: float = 3.0
    n_up: int = 3
    n_down: int = 5
    warmup_min: int = 60
    window_size: int = 300
    weak_signal_db: float = 6.0           # for presence histogram
    presence_threshold: float = 0.8       # PRESENT-at-warmup fraction
    min_freq_tolerance_hz: float = 5000.0  # floor on dedup freq tolerance

    # ---- bin-cluster (sidelobe) suppression ----
    #
    # A strong appearance often triggers weaker detections in
    # adjacent bins from FFT sidelobes, IQ imbalance images, tuner
    # spurs, and OOK-sideband artifacts. When enabled, the engine
    # sorts each sweep's detections by SNR (descending) and drops
    # any detection that lands within an earlier (stronger)
    # detection's "shadow zone".
    #
    # Shadow radius for a detection D = max(base_hz, D.snr_db *
    # per_db_hz). Detections with SNR below the threshold cast no
    # shadow, so two adjacent moderate-strength emitters are NOT
    # cross-suppressed.
    cluster_suppression: bool = True
    cluster_shadow_min_snr_db: float = 25.0
    cluster_shadow_base_hz: float = 20_000.0
    cluster_shadow_per_db_hz: float = 4_000.0


class OrchestratorConfig(BaseModel):
    pause_for_analysis: Literal["auto", True, False] = "auto"
    re_observation_s: float = 30.0
    capture_ms: float = 500.0
    analysis_budget_ms: float = 5000.0
    priority: list[str] = Field(
        default_factory=lambda: ["appearance", "disappearance"]
    )


class IdToolConfig(BaseModel):
    enabled: bool = True
    binary: str | None = None
    extra_args: list[str] = Field(default_factory=list)


class IdentificationConfig(BaseModel):
    rtl_433: IdToolConfig = Field(default_factory=lambda: IdToolConfig(binary="rtl_433"))
    urh: IdToolConfig = Field(default_factory=lambda: IdToolConfig(binary="urh_cli"))
    min_confidence: float = 0.7


class LibraryConfig(BaseModel):
    """User-curated fingerprint library (see riotduck/library.py)."""

    enabled: bool = True
    path: str = "library.yaml"
    suggest_new: bool = True       # emit a YAML snippet on miss


class NotifyFilter(BaseModel):
    types: list[str] | None = None
    ranges: list[str] | None = None
    min_snr_db: float | None = None


class NotifySink(BaseModel):
    sink: Literal["stdout", "jsonl", "webhook", "syslog"]
    path: str | None = None
    url: str | None = None
    host: str | None = None
    port: int | None = None
    filter: NotifyFilter | None = None


class StorageConfig(BaseModel):
    db_path: str = "riotduck.db"
    captures_dir: str = "captures"
    baselines_dir: str = "baselines"
    retain_iq_days: int = 14
    retain_events_days: int = 90


class Config(BaseModel):
    devices: list[DeviceConfig] = Field(default_factory=list)
    ranges: list[RangeConfig | RangeRef] = Field(default_factory=list)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    identification: IdentificationConfig = Field(default_factory=IdentificationConfig)
    library: LibraryConfig = Field(default_factory=LibraryConfig)
    notify: list[NotifySink] = Field(
        default_factory=lambda: [NotifySink(sink="stdout")]
    )
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def _resolve_ranges(self) -> "Config":
        # Replace RangeRef entries with concrete RangeConfig from the
        # predefined library. Must be called after the library is loaded.
        # We defer actual resolution to load_config() which has access
        # to the predefined set.
        return self


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_NUMERIC_FIELDS = {"f_start", "f_end", "bin_hz", "dwell_ms", "samp_rate",
                   "repeat_s", "min_bw_hz"}


def _coerce_numerics(entry: dict[str, Any]) -> dict[str, Any]:
    """Coerce YAML-parsed strings like '433.05e6' to floats.

    PyYAML 1.1 requires an explicit sign in the exponent ('e+6') to
    recognize a value as a float; we accept either form by trying a
    float conversion on known numeric fields.
    """
    out = dict(entry)
    for k in _NUMERIC_FIELDS:
        v = out.get(k)
        if isinstance(v, str):
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def load_predefined_ranges(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the predefined ISM/common-allocation range library."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "config" / "ism_bands.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    out = {}
    for entry in data.get("ranges", []):
        entry = _coerce_numerics(entry)
        name = entry["name"]
        out[name] = entry
    return out


def load_config(
    path: Path,
    predefined_path: Path | None = None,
) -> Config:
    """Load a user YAML config and resolve any `use:` range references."""
    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    predefined = load_predefined_ranges(predefined_path)
    resolved_ranges: list[dict[str, Any]] = []
    for entry in raw.get("ranges", []) or []:
        entry = _coerce_numerics(entry)
        if "use" in entry:
            base_name = entry["use"]
            if base_name not in predefined:
                raise ValueError(
                    f"range references unknown predefined band: {base_name!r}"
                )
            base = predefined[base_name]
            merged = _deep_merge(base, entry.get("overrides", {}))
            merged.pop("use", None)
            merged.pop("overrides", None)
            resolved_ranges.append(merged)
        else:
            resolved_ranges.append(entry)
    raw["ranges"] = resolved_ranges
    return Config.model_validate(raw)


def predefined_range_names(predefined_path: Path | None = None) -> list[str]:
    return sorted(load_predefined_ranges(predefined_path).keys())
