"""riotduck CLI."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from riotduck.config import Config, load_config, load_predefined_ranges
from riotduck.runner import run_scan
from riotduck.sdr.manager import DeviceManager

console = Console()


def _apply_fake_env(n: int | None, profile: str | None) -> None:
    """Apply --fake / --fake-profile to the env vars the backend reads."""
    if n is not None:
        os.environ["RIOTDUCK_FAKE_DEVICES"] = str(n)
    if profile:
        os.environ["RIOTDUCK_FAKE_PROFILE"] = profile


def _resolve_config(path: str | None) -> Config:
    if path:
        p = Path(path)
    else:
        # Prefer user config; fall back to bundled default.
        user_path = Path.home() / ".config" / "riotduck" / "config.yaml"
        bundled = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        p = user_path if user_path.exists() else bundled
    logger.info("loading config: {}", p)
    return load_config(p)


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.pass_context
def main(ctx: click.Context, log_level: str, config_path: str | None) -> None:
    """riotduck — RF baseline + anomaly detection."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@main.command()
@click.option("--fake", "fake_n", type=int, default=None,
              help="Use N synthetic SDRs (no hardware needed).")
@click.option("--fake-profile", type=click.Path(exists=True), default=None,
              help="YAML profile of emitters for the fake SDR.")
@click.pass_context
def scan(ctx: click.Context, fake_n: int | None, fake_profile: str | None) -> None:
    """Run the continuous scan + detect + notify loop."""
    _apply_fake_env(fake_n, fake_profile)
    cfg = _resolve_config(ctx.obj.get("config_path"))
    try:
        asyncio.run(run_scan(cfg))
    except KeyboardInterrupt:
        pass


@main.command()
@click.option("--fake", "fake_n", type=int, default=None,
              help="Include N synthetic SDRs in the listing.")
def devices(fake_n: int | None) -> None:
    """List discovered SDRs."""
    _apply_fake_env(fake_n, None)
    mgr = DeviceManager()
    recs = mgr.discover()
    if not recs:
        console.print("[red]No SDR devices found.[/red]")
        return
    t = Table(title="SDR devices")
    t.add_column("serial")
    t.add_column("type")
    t.add_column("driver")
    t.add_column("tuning (MHz)")
    t.add_column("sample rates (MS/s)")
    t.add_column("gain stages")
    for r in recs:
        lo, hi = r.info.tuning_range_hz
        t.add_row(
            r.info.serial or "(unknown)",
            r.info.type,
            r.info.driver,
            f"{lo/1e6:.1f}–{hi/1e6:.1f}",
            ", ".join(f"{s/1e6:g}" for s in r.info.samp_rates),
            ", ".join(r.info.gain_stages) or "-",
        )
    console.print(t)


@main.command()
@click.option("--predefined", is_flag=True, help="Show only predefined library")
@click.pass_context
def ranges(ctx: click.Context, predefined: bool) -> None:
    """List configured + predefined ranges."""
    lib = load_predefined_ranges()
    t = Table(title="Predefined ranges")
    t.add_column("name")
    t.add_column("start (MHz)")
    t.add_column("end (MHz)")
    t.add_column("bin (kHz)")
    t.add_column("short_burst")
    t.add_column("description")
    for name in sorted(lib.keys()):
        e = lib[name]
        t.add_row(
            name,
            f"{e['f_start']/1e6:.3f}",
            f"{e['f_end']/1e6:.3f}",
            f"{e.get('bin_hz', 4000)/1e3:g}",
            str(bool(e.get("short_burst", False))),
            e.get("description", ""),
        )
    console.print(t)

    if predefined:
        return

    try:
        cfg = _resolve_config(ctx.obj.get("config_path"))
    except Exception as e:
        console.print(f"[yellow]Could not load user config: {e}[/yellow]")
        return
    t2 = Table(title="Configured ranges")
    t2.add_column("name")
    t2.add_column("start (MHz)")
    t2.add_column("end (MHz)")
    t2.add_column("bin (kHz)")
    t2.add_column("short_burst")
    for r in cfg.ranges:
        # After load_config, refs are resolved to RangeConfig already.
        rc = r  # type: ignore[assignment]
        t2.add_row(
            rc.name,
            f"{rc.f_start/1e6:.3f}",
            f"{rc.f_end/1e6:.3f}",
            f"{rc.bin_hz/1e3:g}",
            str(rc.short_burst),
        )
    console.print(t2)


@main.command()
def doctor() -> None:
    """Check the environment: SDR backends, rtl_433, urh_cli, perms."""
    from riotduck.fingerprint.rtl_433 import (
        RTL_433_MIN_RECOMMENDED,
        get_rtl_433_info,
    )
    from riotduck.sdr.hackrf import hackrf_available
    from riotduck.sdr.rtlsdr import rtlsdr_available

    OK = "[green]ok[/green]"
    WARN = "[yellow]warn[/yellow]"
    MISSING = "[red]missing[/red]"

    t = Table(title="riotduck doctor")
    t.add_column("check")
    t.add_column("status")
    t.add_column("notes")

    t.add_row("RTL-SDR backend",
              OK if rtlsdr_available() else MISSING,
              "SoapySDR or pyrtlsdr")
    t.add_row("HackRF backend",
              OK if hackrf_available() else MISSING,
              "SoapySDR with hackrf driver")

    # rtl_433 — version + alternative-install detection.
    info = get_rtl_433_info()
    if not info.installed:
        t.add_row("rtl_433 binary", MISSING, "needed for fingerprinting")
    else:
        notes: list[str] = []
        status = OK
        if info.version is None:
            status = WARN
            notes.append(info.error or "version unparseable")
        else:
            notes.append(f"v{info.version_str}")
            if info.is_stale:
                status = WARN
                min_v = ".".join(str(x) for x in RTL_433_MIN_RECOMMENDED)
                notes.append(f"[yellow]old (< {min_v} recommended)[/yellow]")
        notes.append(info.path or "")
        t.add_row("rtl_433 binary", status, " · ".join(notes))
        for shadow in info.shadows:
            from riotduck.fingerprint.rtl_433 import probe_rtl_433_version
            sv = probe_rtl_433_version(shadow)
            sv_str = f"v{sv[0]}.{sv[1]:02d}" if sv else "?"
            # If the alternative is *newer* than the active binary,
            # warn — the user is probably running the wrong one.
            newer = sv is not None and info.version is not None and sv > info.version
            t.add_row(
                "rtl_433 alt install",
                WARN if newer else "[dim]info[/dim]",
                f"{sv_str} at {shadow}"
                + (" [yellow](newer than active)[/yellow]" if newer else ""),
            )

    t.add_row("urh_cli binary",
              OK if shutil.which("urh_cli") is not None else MISSING,
              "optional, URH-based demod")

    mgr = DeviceManager()
    recs = mgr.discover()
    t.add_row("Devices discovered",
              OK if recs else MISSING,
              f"{len(recs)} device(s)")

    console.print(t)


@main.command()
@click.option("--center", "-f", "center", type=float, required=True, help="Center freq Hz")
@click.option("--samp-rate", "-s", "samp_rate", type=float, default=2.4e6, show_default=True)
@click.option("--duration", "-t", "duration", type=float, default=1.0, show_default=True, help="Seconds")
@click.option("--serial", default=None, help="Device serial; default first found")
@click.option("--out", "-o", "out_path", type=click.Path(), required=True, help="Output .cf32 file")
def capture(center: float, samp_rate: float, duration: float, serial: str | None, out_path: str) -> None:
    """One-shot I/Q capture to a complex64 file."""
    import numpy as np

    mgr = DeviceManager()
    recs = mgr.discover()
    if not recs:
        console.print("[red]No SDR found.[/red]")
        sys.exit(1)
    target_serial = serial or recs[0].info.serial
    sess = mgr.acquire(target_serial, holder="capture-cli")
    try:
        actual = sess.set_samp_rate(samp_rate)
        sess.set_center_hz(center)
        sess.set_gain({"tuner": 28.0})
        n = int(actual * duration)
        console.print(f"capturing {n} samples @ {actual/1e6:.3f} MS/s, {center/1e6:.4f} MHz")
        iq = sess.read_iq(n)
        np.asarray(iq, dtype=np.complex64).tofile(out_path)
        console.print(f"wrote {len(iq)} samples to {out_path}")
    finally:
        mgr.release(target_serial, sess)


@main.command()
@click.argument("iq_path", type=click.Path(exists=True))
@click.option("--samp-rate", "-s", "samp_rate", type=float, required=True,
              help="Sample rate the capture was recorded at, in Hz.")
@click.option("--center", "-f", "center", type=float, default=None,
              help="Center frequency the capture was tuned to, in Hz (optional hint).")
@click.option("--binary", default=None,
              help="Override rtl_433 binary path; defaults to first on PATH.")
@click.option("--timeout", "timeout_s", type=float, default=60.0, show_default=True,
              help="Per-tool timeout in seconds.")
@click.option("--json", "json_out", is_flag=True,
              help="Print one JSON object per identification on stdout.")
@click.pass_context
def analyze(ctx: click.Context, iq_path: str, samp_rate: float,
            center: float | None, binary: str | None,
            timeout_s: float, json_out: bool) -> None:
    """Offline: run the identification pipeline against an I/Q file.

    Currently runs rtl_433 in file-mode replay. URH and unknown-signal
    analysis hooks will be added under the same command as they land.

    Example:

        riotduck analyze capture.cf32 -s 2400000 -f 433920000
    """
    import json as _json

    from riotduck.config import IdToolConfig
    from riotduck.fingerprint.rtl_433 import run_on_file

    cfg = ctx.obj.get("config_path")
    try:
        loaded = _resolve_config(cfg) if cfg else None
    except Exception:
        loaded = None

    bin_name = (
        binary
        or (loaded.identification.rtl_433.binary if loaded else None)
        or "rtl_433"
    )
    extra = list(
        loaded.identification.rtl_433.extra_args
        if loaded
        else IdToolConfig().extra_args
    )

    if shutil.which(bin_name) is None:
        console.print(
            f"[red]rtl_433 binary not found:[/red] {bin_name}. "
            "Install it (apt install rtl-433 / brew install rtl_433) or "
            "pass --binary."
        )
        sys.exit(2)

    console.print(
        f"[cyan]rtl_433[/cyan] on {iq_path} "
        f"(sr={samp_rate/1e6:.3f} MS/s"
        + (f", center={center/1e6:.4f} MHz" if center else "")
        + ")"
    )
    result = run_on_file(
        iq_path,
        samp_rate=samp_rate,
        binary=bin_name,
        center_hz=center,
        extra_args=extra,
        timeout_s=timeout_s,
    )

    if result.returncode != 0 and result.returncode > -10:
        console.print(f"[yellow]rtl_433 exited {result.returncode}[/yellow]")
    if result.stderr.strip() and not json_out:
        console.print(
            "[dim]stderr (tail):[/dim] "
            + result.stderr.strip().splitlines()[-1]
        )

    if not result.hits:
        if json_out:
            print(_json.dumps({"hits": [], "returncode": result.returncode}))
        else:
            console.print("[yellow]no rtl_433 hits[/yellow]")
        return

    if json_out:
        for h in result.hits:
            print(_json.dumps({"model": h.model, "decoded": h.decoded,
                               "confidence": h.confidence}))
        return

    t = Table(title=f"rtl_433 hits ({len(result.hits)})")
    t.add_column("model")
    t.add_column("fields")
    for h in result.hits:
        # Drop fields that are mostly noise for human reading.
        skip = {"time", "model", "mod", "freq", "freq1", "freq2", "rssi",
                "snr", "noise"}
        kv = ", ".join(f"{k}={v}" for k, v in h.decoded.items() if k not in skip)
        t.add_row(h.model, kv)
    console.print(t)


@main.group()
def library() -> None:
    """Inspect the user-curated fingerprint library."""


def _resolve_library_path(ctx: click.Context, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    cfg_path = ctx.obj.get("config_path") if ctx.obj else None
    try:
        cfg = _resolve_config(cfg_path) if cfg_path else None
    except Exception:
        cfg = None
    if cfg is not None:
        return Path(cfg.library.path)
    return Path("library.yaml")


@library.command("list")
@click.option("--path", default=None, help="Library YAML; default from config.")
@click.pass_context
def library_list(ctx: click.Context, path: str | None) -> None:
    """List entries with a one-line summary each."""
    from riotduck.library import Library
    p = _resolve_library_path(ctx, path)
    lib = Library.load(p)
    if len(lib) == 0:
        console.print(f"[yellow]library is empty[/yellow] ({p})")
        return
    t = Table(title=f"fingerprint library — {p}")
    t.add_column("id")
    t.add_column("name")
    t.add_column("center (MHz)")
    t.add_column("mod")
    t.add_column("BW (kHz)")
    t.add_column("sym (Hz)")
    t.add_column("tags")
    for e in lib.entries:
        m = e.match
        t.add_row(
            e.id,
            e.name or "[dim]—[/dim]",
            f"{m.center_hz/1e6:.4f}",
            m.modulation or "-",
            f"{m.bw_3db_hz/1e3:.2f}" if m.bw_3db_hz else "-",
            f"{m.symbol_rate_hz:.0f}" if m.symbol_rate_hz else "-",
            ", ".join(e.tags) or "-",
        )
    console.print(t)


@library.command("show")
@click.argument("entry_id")
@click.option("--path", default=None, help="Library YAML; default from config.")
@click.pass_context
def library_show(ctx: click.Context, entry_id: str, path: str | None) -> None:
    """Show full detail for one entry."""
    from riotduck.library import Library
    p = _resolve_library_path(ctx, path)
    lib = Library.load(p)
    entry = lib.get(entry_id)
    if entry is None:
        console.print(f"[red]no entry with id {entry_id!r}[/red] in {p}")
        sys.exit(1)
    m = entry.match
    console.print(f"[bold]{entry.id}[/bold]" + (f"  — {entry.name}" if entry.name else ""))
    if entry.notes:
        console.print(f"  notes: {entry.notes}")
    if entry.tags:
        console.print(f"  tags:  {', '.join(entry.tags)}")
    console.print(f"  center_hz:   {m.center_hz/1e6:.6f} MHz  (± {m.center_tolerance_hz/1e3:.1f} kHz)")
    if m.modulation:
        console.print(f"  modulation:  {m.modulation}")
    if m.bw_3db_hz is not None:
        console.print(f"  bw_3db:      {m.bw_3db_hz/1e3:.2f} kHz  (± {m.bw_3db_tolerance_hz/1e3:.2f} kHz)")
    if m.symbol_rate_hz is not None:
        console.print(f"  sym_rate:    {m.symbol_rate_hz:.0f} Hz  (± {m.symbol_rate_tolerance_hz:.0f} Hz)")


if __name__ == "__main__":
    main()
