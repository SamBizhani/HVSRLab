"""Getting results out: text curves, tables, ProTO projects, and reports.

Every exporter writes into the project's ``exports`` directory unless told
otherwise, and every one of them records the processing parameters alongside
the numbers. A curve without the window length, the smoothing constant and the
horizontal-combination rule that produced it cannot be reproduced or compared,
and an H/V amplitude in particular is meaningless without the last of those.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import io
from pathlib import Path

import numpy as np

from .core import bedrock, picking, sesame
from .io import curves as curve_io
from .project import HVSR_STRATEGY_LABELS, Project

CSV_COLUMNS = [
    "site", "latitude", "longitude", "elevation", "easting_m", "northing_m",
    "f0_Hz", "A0", "sigma_f0_Hz", "f0_source", "depth_m",
    "sesame", "reliable", "clear", "windows_kept", "windows_total",
    "window_start_utc", "window_end_utc", "window_hours", "status", "computed",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _crs_label(project) -> str:
    """The projection every x/y in an export is expressed in."""
    from .io import crs as crs_io

    zone = crs_io.UTM.from_dict(getattr(project, "crs", None))
    return zone.label + ", metres" if zone else "not set"


def parameter_block(project: Project) -> list[str]:
    """The processing settings, as comment lines."""
    p = project.params
    return [
        f"HVSRLab export — {_stamp()}",
        f"project: {project.name}",
        f"windows: {p.window_width_s:g} s, {p.window_overlap_pc:g} % overlap, "
        f"{p.taper_pc:g} % taper",
        f"anti-trigger: " + (f"STA {p.sta_s:g} s / LTA {p.lta_s:g} s, "
                             f"reject above {p.sta_lta_ratio:g}"
                             if p.antitrigger else "off"),
        f"band: {p.freq_min:g}–{p.freq_max:g} Hz on a {p.freq_grid} axis",
        f"smoothing: {p.smoothing_kind} (b = {p.smoothing_b:g})",
        f"horizontals: {HVSR_STRATEGY_LABELS.get(p.hvsr_strategy, p.hvsr_strategy)}",
        f"statistics: {p.statistics}",
        f"pre-filter: {p.filter_kind}"
        + (f" order {p.filter_order}, {p.filter_fmin:g}–{p.filter_fmax:g} Hz"
           if p.filter_kind != "off" else ""),
        f"plan grid: {_crs_label(project)}",
        f"bedrock law: H = {project.regression.a:.6g} · f0^{project.regression.b:.6g}"
        + (f" (fitted to {project.regression.n_points} wells)"
           if project.regression.fitted else f" ({project.regression.name})"),
    ]


# ---------------------------------------------------------------------------
# Curves and tables
# ---------------------------------------------------------------------------

def hvsr_curves(project: Project, results: dict, *, directory: Path | None = None
                ) -> Path:
    """One three-column text file per site: frequency, H/V, standard deviation."""
    directory = Path(directory or project.exports_dir / "curves")
    directory.mkdir(parents=True, exist_ok=True)
    header = parameter_block(project)

    for site in project.sites:
        result = results.get(site.sid)
        if result is None or result.freq.size == 0:
            continue
        lines = header + [
            f"site: {site.label()}",
            f"lat {site.lat:.6f}  lon {site.lon:.6f}  elev {site.elev:.2f}",
            f"f0 = {site.f0:.6g} Hz   A0 = {site.a0:.4g}",
            f"windows: {result.n_ok} of {result.n_windows}",
            "std is of log(H/V)" if result.statistics == "lognormal"
            else "std is linear",
        ]
        curve_io.write_curve(directory / f"{_safe(site.label())}.txt",
                             result.freq, result.hv, result.hv_std,
                             header="\n".join(lines))
    return directory


def summary_csv(project: Project, path: Path | None = None) -> Path:
    """One row per site — the table that goes into a report or a GIS."""
    path = Path(path or project.exports_dir / f"{_safe(project.name)}_summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    law = project.regression

    with open(path, "w", encoding="utf-8", newline="") as fh:
        for line in parameter_block(project):
            fh.write(f"# {line}\n")
        fh.write(",".join(CSV_COLUMNS) + "\n")
        for site in project.sites:
            reliable = clear = ""
            if site.sesame_score:
                try:
                    r, c = site.sesame_score.split("·")
                    reliable = str(int(r.strip().split("/")[0]) == 3)
                    clear = str(int(c.strip().split("/")[0]) >= 5)
                except (ValueError, IndexError):
                    pass
            row = [
                site.label(), _n(site.lat, 6), _n(site.lon, 6), _n(site.elev, 2),
                _n(site.x, 2), _n(site.y, 2),
                _n(site.f0, 6), _n(site.a0, 4), _n(site.f0_std, 6),
                site.f0_source,
                _n(bedrock.depth_from_f0(site.f0, law.a, law.b), 2),
                site.sesame_score, reliable, clear,
                str(site.n_windows_ok), str(site.n_windows),
                site.time.start, site.time.end, _n(site.time.hours, 2),
                site.status, site.computed,
            ]
            fh.write(",".join(_csv(v) for v in row) + "\n")
    return path


def full_output(project: Project, results: dict, *,
                directory: Path | None = None) -> Path:
    """Curves, the summary table, per-site SESAME reports and the picks."""
    directory = Path(directory or project.exports_dir /
                     f"{_safe(project.name)}_full")
    directory.mkdir(parents=True, exist_ok=True)

    hvsr_curves(project, results, directory=directory / "curves")
    summary_csv(project, directory / "summary.csv")

    with open(directory / "sesame.txt", "w", encoding="utf-8") as fh:
        for line in parameter_block(project):
            fh.write(f"# {line}\n")
        fh.write("\n")
        for site in project.sites:
            result = results.get(site.sid)
            if result is None:
                continue
            report = sesame.evaluate(result, site.f0)
            fh.write(f"== {site.label()} ==\n")
            fh.write(f"f0 = {report.f0:.6g} Hz   A0 = {report.a0:.4g}   "
                     f"{report.summary}   {report.verdict}\n")
            for c in report.all_criteria():
                fh.write(f"  [{c.mark}] {c.key} {c.text}: "
                         f"{c.value:.6g} vs {c.threshold:.6g}"
                         + (f"   {c.detail}" if c.detail else "") + "\n")
            for note in report.notes:
                fh.write(f"  note: {note}\n")
            fh.write("\n")

    peaks = directory / "peaks.txt"
    with open(peaks, "w", encoding="utf-8") as fh:
        fh.write("# site  x  y  z  f0  A0  [additional peaks: f A ...]\n")
        for site in project.sites:
            if not np.isfinite(site.f0):
                continue
            extras = " ".join(f"{f:.6g} {a:.4g}" for f, a in site.extra_peaks)
            fh.write(f"{_safe(site.label())} {_n(site.x, 3)} {_n(site.y, 3)} "
                     f"{_n(site.z, 3)} {site.f0:.6g} {site.a0:.4g} {extras}\n")

    with open(directory / "parameters.txt", "w", encoding="utf-8") as fh:
        for line in parameter_block(project):
            fh.write(line + "\n")
        fh.write("\nfull parameter set:\n")
        for key, value in asdict(project.params).items():
            fh.write(f"  {key} = {value}\n")
    return directory


def openhvsr_project(project: Project, results: dict, *,
                     directory: Path | None = None) -> Path:
    """Write a project OpenHVSR-ProTO and OpenHVSR-Inversion can open.

    The curves go out as three-column text and the ``.m`` file lists them with
    their coordinates, in the ``SURVEYS{n,1..3}`` layout ProTO expects. Plan
    coordinates are used rather than latitude and longitude: ProTO treats the
    location as Cartesian metres, and feeding it degrees makes every distance
    and profile in that program wrong.
    """
    directory = Path(directory or project.exports_dir /
                     f"{_safe(project.name)}_openhvsr")
    directory.mkdir(parents=True, exist_ok=True)

    exported = []
    for site in project.sites:
        result = results.get(site.sid)
        if result is None or result.freq.size == 0 or site.status == "excluded":
            continue
        name = f"{_safe(site.label())}.txt"
        curve_io.write_curve(directory / name, result.freq, result.hv,
                             result.hv_std,
                             header=f"{site.label()} — HVSRLab {_stamp()}")
        exported.append((site, name))

    path = directory / "OpenHVSR_ProTO_project.m"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% OpenHVSR-ProTO project written by HVSRLab\n")
        fh.write(f"% {_stamp()}\n%\n")
        for line in parameter_block(project):
            fh.write(f"% {line}\n")
        fh.write("%\n")
        fh.write("% Curves are already computed: columns are "
                 "[frequency, H/V, std].\n")
        fh.write(f"% Locations are {_crs_label(project)} -- ProTO treats them "
                 "as Cartesian metres.\n")
        fh.write("datafile_separator = 'none';\n")
        fh.write("datafile_columns   = [1 2 3];% [frequency, H/V, std]\n%\n")
        for i, (site, name) in enumerate(exported, start=1):
            fh.write(f"SURVEYS{{{i},1}} = [{site.x:.2f},{site.y:.2f},"
                     f"{site.z:.2f}]; SURVEYS{{{i},2}} = '{name}';\n")
        if project.topography_file:
            fh.write(f"\nTOPOGRAPHY_file_name = "
                     f"'{Path(project.topography_file).name}';\n")
    return path


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def html_report(project: Project, results: dict, *, path: Path | None = None,
                include_sites: bool = True, progress=None) -> Path:
    """A single self-contained HTML file: survey summary and per-site pages.

    Figures are embedded as base64 PNG, so the file can be emailed and will
    still render years from now with no dependency on this program.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    path = Path(path or project.exports_dir / f"{_safe(project.name)}_report.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    law = project.regression

    sites = [s for s in project.sites if s.status != "excluded"]
    computed = [s for s in sites if np.isfinite(s.f0)]

    parts: list[str] = [_REPORT_HEAD.format(
        title=escape(project.name), stamp=_stamp())]

    parts.append("<section class='card'><h2>Survey</h2>")
    parts.append("<div class='tiles'>")
    for label, value in (
        ("sites", str(len(sites))),
        ("computed", str(len(computed))),
        ("median f₀", _fmt(np.nanmedian([s.f0 for s in computed]) if computed
                           else np.nan, 3) + " Hz"),
        ("median A₀", _fmt(np.nanmedian([s.a0 for s in computed]) if computed
                           else np.nan, 2)),
        ("reliable & clear", str(sum(1 for s in computed
                                     if _is_good(s.sesame_score)))),
    ):
        parts.append(f"<div class='tile'><b>{escape(value)}</b><span>"
                     f"{escape(label)}</span></div>")
    parts.append("</div>")
    parts.append("<h3>Processing</h3><pre>"
                 + escape("\n".join(parameter_block(project))) + "</pre>")
    parts.append("</section>")

    if computed:
        fig = Figure(figsize=(11, 3.2), dpi=110)
        axes = fig.subplots(1, 3)
        f0 = np.array([s.f0 for s in computed])
        a0 = np.array([s.a0 for s in computed])
        axes[0].hist(f0, bins=min(24, max(5, len(f0) // 3)), color="#2b7bba")
        axes[0].set_xlabel("f₀ (Hz)")
        axes[0].set_ylabel("sites")
        axes[1].scatter(f0, a0, s=18, color="#c78a1e")
        axes[1].axhline(2.0, color="#888", linestyle="--", linewidth=0.9)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("f₀ (Hz)")
        axes[1].set_ylabel("A₀")
        x = np.array([s.x for s in computed])
        y = np.array([s.y for s in computed])
        if np.isfinite(x).any():
            sc = axes[2].scatter(x, y, c=f0, s=26, cmap="viridis")
            fig.colorbar(sc, ax=axes[2], label="f₀ (Hz)")
            axes[2].set_aspect("equal", adjustable="datalim")
        axes[2].set_xlabel("easting (m)")
        axes[2].set_ylabel("northing (m)")
        fig.tight_layout()
        parts.append("<section class='card'><h2>Distributions</h2>"
                     + _embed(fig) + "</section>")

    parts.append("<section class='card'><h2>Sites</h2><table>")
    parts.append("<tr><th>Site</th><th>f₀ (Hz)</th><th>A₀</th>"
                 "<th>σ f₀</th><th>depth (m)</th><th>SESAME</th>"
                 "<th>windows</th><th>window (UTC)</th></tr>")
    for site in sites:
        depth = bedrock.depth_from_f0(site.f0, law.a, law.b)
        css = "good" if _is_good(site.sesame_score) else (
            "warn" if site.sesame_score else "")
        parts.append(
            f"<tr><td>{escape(site.label())}</td><td>{_fmt(site.f0, 3)}</td>"
            f"<td>{_fmt(site.a0, 2)}</td><td>{_fmt(site.f0_std, 3)}</td>"
            f"<td>{_fmt(depth, 1)}</td>"
            f"<td class='{css}'>{escape(site.sesame_score or '—')}</td>"
            f"<td>{site.n_windows_ok}/{site.n_windows}</td>"
            f"<td>{escape(site.time.start)}</td></tr>")
    parts.append("</table></section>")

    if include_sites:
        total = len(computed)
        for i, site in enumerate(computed, start=1):
            if progress is not None:
                progress(i / max(1, total), f"rendering {site.label()}")
            result = results.get(site.sid)
            if result is None:
                continue
            parts.append(_site_section(project, site, result, law))

    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _site_section(project, site, result, law) -> str:
    from matplotlib.figure import Figure

    report = sesame.evaluate(result, site.f0)
    fig = Figure(figsize=(11, 3.4), dpi=110)
    ax, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})

    if result.hv_windows.size:
        kept = result.kept()
        step = max(1, kept.shape[1] // 200)
        ax.plot(result.freq, kept[:, ::step], color="#bbb", linewidth=0.3,
                alpha=0.5)
    ax.fill_between(result.freq, result.hv_lo, result.hv_hi, color="#2b7bba",
                    alpha=0.2, linewidth=0)
    ax.plot(result.freq, result.hv, color="#1f5f8b", linewidth=2)
    if np.isfinite(site.f0):
        ax.axvline(site.f0, color="#c78a1e", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("H/V")
    ax.set_ylim(0, max(3.0, float(np.nanpercentile(result.hv_hi, 99)) * 1.1))
    ax.grid(alpha=0.3)

    keep = result.ok & np.isfinite(result.window_f0)
    if keep.any():
        ax2.plot(np.arange(keep.sum()), result.window_f0[keep], ".",
                 markersize=2, color="#2b7bba")
        if np.isfinite(site.f0):
            ax2.axhline(site.f0, color="#c78a1e", linestyle="--")
    ax2.set_yscale("log")
    ax2.set_xlabel("window")
    ax2.set_ylabel("window f₀ (Hz)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    rows = "".join(
        f"<tr><td class='{'good' if c.passed else 'bad'}'>{c.mark}</td>"
        f"<td>{escape(c.key)} {escape(c.text)}</td>"
        f"<td>{_fmt(c.value, 3)}</td><td>{_fmt(c.threshold, 3)}</td></tr>"
        for c in report.all_criteria())

    depth = bedrock.depth_from_f0(site.f0, law.a, law.b)
    return (
        f"<section class='card'><h2>{escape(site.label())}</h2>"
        f"<p>f₀ = <b>{_fmt(site.f0, 3)} Hz</b>, A₀ = <b>{_fmt(site.a0, 2)}</b>, "
        f"depth ≈ <b>{_fmt(depth, 1)} m</b> · {escape(report.summary)} — "
        f"{escape(report.verdict)} · {result.n_ok}/{result.n_windows} windows · "
        f"{escape(site.time.start)} → {escape(site.time.end)} UTC</p>"
        + _embed(fig)
        + "<table><tr><th></th><th>Criterion</th><th>Measured</th>"
          "<th>Threshold</th></tr>" + rows + "</table></section>")


def _embed(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"<img src='data:image/png;base64,{encoded}'/>"


_REPORT_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title} — H/V report</title>
<style>
 body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0;
         background: #f4f6f8; color: #1d2430; }}
 header {{ background: #1f2a37; color: #fff; padding: 22px 28px; }}
 header h1 {{ margin: 0; font-size: 21px; }}
 header p {{ margin: 4px 0 0; opacity: .75; font-size: 13px; }}
 main {{ max-width: 1180px; margin: 0 auto; padding: 18px; }}
 .card {{ background: #fff; border: 1px solid #dde3ea; border-radius: 9px;
          padding: 16px 20px; margin: 14px 0;
          box-shadow: 0 1px 2px rgba(20,30,50,.05); }}
 h2 {{ font-size: 16px; margin: 0 0 10px; }}
 h3 {{ font-size: 13px; margin: 14px 0 6px; color: #5a6675; }}
 img {{ max-width: 100%; height: auto; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 12.5px;
          margin-top: 8px; }}
 th, td {{ border-bottom: 1px solid #e6eaef; padding: 5px 8px;
           text-align: left; }}
 th {{ background: #f7f9fb; font-weight: 600; color: #5a6675; }}
 pre {{ background: #f7f9fb; border: 1px solid #e6eaef; border-radius: 6px;
        padding: 10px; font-size: 12px; overflow-x: auto; }}
 .tiles {{ display: flex; gap: 10px; flex-wrap: wrap; }}
 .tile {{ background: #f7f9fb; border: 1px solid #e6eaef; border-radius: 7px;
          padding: 9px 14px; min-width: 108px; }}
 .tile b {{ display: block; font-size: 19px; }}
 .tile span {{ font-size: 10px; text-transform: uppercase; color: #7b8794; }}
 .good {{ color: #1a7f47; font-weight: 600; }}
 .warn {{ color: #a76a00; font-weight: 600; }}
 .bad  {{ color: #b3261e; font-weight: 600; }}
</style></head><body>
<header><h1>{title} — horizontal-to-vertical spectral ratio</h1>
<p>HVSRLab · {stamp}</p></header><main>
"""


def _is_good(score: str) -> bool:
    try:
        r, c = score.split("·")
        return int(r.strip().split("/")[0]) == 3 and int(c.strip().split("/")[0]) >= 5
    except (ValueError, IndexError, AttributeError):
        return False


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def _n(value, digits: int) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(value) else f"{value:.{digits}f}"


def _csv(value: str) -> str:
    text = "" if value is None else str(value)
    return f'"{text}"' if ("," in text or '"' in text) else text


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))
