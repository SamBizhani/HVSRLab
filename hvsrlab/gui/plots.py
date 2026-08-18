"""Embedded matplotlib figures.

Each class draws one kind of output and is responsible for degrading to a
readable message rather than an exception when the data it needs are not there
yet — pages are built before anything has been computed.

Interaction is kept in the canvas that owns the axes: clicking a peak, toggling
a window, dragging a profile. Each emits a Qt signal and lets the page decide
what it means.
"""

from __future__ import annotations

from datetime import datetime, timezone
import warnings

import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from .theme import C, COMPONENT_COLORS


class Canvas(QWidget):
    """A figure, optionally with a navigation toolbar."""

    def __init__(self, parent: QWidget | None = None, *, height: float = 3.4,
                 toolbar: bool = True, polar: bool = False) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, height), facecolor=C["surface"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if toolbar:
            bar = NavigationToolbar2QT(self.canvas, self)
            bar.setStyleSheet(
                f"QToolBar {{ border: none; background: transparent; }}"
                f"QToolButton {{ color: {C['text_dim']}; }}")
            bar.setIconSize(bar.iconSize() * 0.8)
            layout.addWidget(bar)
        layout.addWidget(self.canvas, 1)
        self._polar = polar

    # -- helpers -----------------------------------------------------------
    def clear(self):
        self.figure.clear()
        return self.figure

    def axes(self, *args, **kwargs):
        self.figure.clear()
        ax = self.figure.add_subplot(*args or (111,), **kwargs)
        return ax

    def message(self, text: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
                transform=ax.transAxes, color=C["text_faint"], fontsize=9)
        ax.set_axis_off()
        self.draw()

    def navigating(self) -> bool:
        """True while the toolbar's pan or zoom tool is armed.

        Every canvas here also listens for clicks — to pick a peak, to keep or
        drop a window, to place a profile. Without this check a zoom drag would
        register as one of those and the redraw would throw the zoom away.
        """
        mode = getattr(self.canvas.toolbar, "mode", "")
        return bool(str(mode))

    def draw(self) -> None:
        # tight_layout warns rather than fails on 3D axes, polar axes and
        # colorbars, all of which appear here. The layout it produces is still
        # the one we want, so the warning is noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.figure.tight_layout()
            except Exception:
                pass
        self.canvas.draw_idle()

    def save(self, path: str, dpi: int = 200) -> str:
        self.figure.savefig(path, dpi=dpi, facecolor=self.figure.get_facecolor())
        return path


# ---------------------------------------------------------------------------
# H/V curve
# ---------------------------------------------------------------------------

class HVSRCurve(Canvas):
    """The mean H/V curve with its scatter, its windows, and the pick."""

    picked = pyqtSignal(float)          # left click: the main peak
    extraPicked = pyqtSignal(float)     # right click: an additional peak

    def __init__(self, parent: QWidget | None = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.show_windows = True
        self.show_components = False
        self._ax = None

    def _on_click(self, event) -> None:
        if event.inaxes is None or not event.xdata or self.navigating():
            return
        if event.button == 1:
            self.picked.emit(float(event.xdata))
        elif event.button == 3:
            self.extraPicked.emit(float(event.xdata))

    def plot(self, result, *, f0: float = float("nan"),
             extra_peaks=(), sesame_report=None, title: str = "") -> None:
        if result is None or result.freq.size == 0:
            self.message("No H/V computed for this site yet.")
            return

        ax = self.axes(111)
        self._ax = ax
        freq = result.freq

        if self.show_windows and result.hv_windows.size:
            kept = result.kept()
            step = max(1, kept.shape[1] // 300)      # keep the redraw responsive
            ax.plot(freq, kept[:, ::step], color=C["muted"], linewidth=0.35,
                    alpha=0.35, zorder=1)

        ax.fill_between(freq, result.hv_lo, result.hv_hi, color=C["accent"],
                        alpha=0.16, linewidth=0, zorder=2,
                        label="±1σ over windows")
        ax.plot(freq, result.hv, color=C["accent"], linewidth=2.0, zorder=4,
                label="H/V")

        if self.show_components and result.ev.size and result.nv.size:
            ax.plot(freq, result.ev, color=COMPONENT_COLORS["E"], linewidth=1.0,
                    alpha=0.9, zorder=3, label="E/V")
            ax.plot(freq, result.nv, color=COMPONENT_COLORS["N"], linewidth=1.0,
                    alpha=0.9, zorder=3, label="N/V")

        if np.isfinite(f0):
            i = int(np.argmin(np.abs(freq - f0)))
            a0 = float(result.hv[i])
            ax.axvline(f0, color=C["pick"], linewidth=1.3, linestyle="--", zorder=5)
            ax.plot([f0], [a0], "o", color=C["pick"], markersize=7, zorder=6)
            ax.annotate(f"f₀ = {f0:.3g} Hz\nA₀ = {a0:.2f}", xy=(f0, a0),
                        xytext=(8, 8), textcoords="offset points",
                        color=C["pick"], fontsize=9, fontweight="bold")
            ax.axhline(a0 / 2.0, color=C["pick"], linewidth=0.7, alpha=0.4,
                       linestyle=":", zorder=3)

        for f, a in extra_peaks or ():
            ax.axvline(f, color=C["text_faint"], linewidth=0.9, linestyle=":")
            ax.plot([f], [a], "v", color=C["text_faint"], markersize=5)

        if sesame_report is not None and np.isfinite(sesame_report.f0):
            f0r = sesame_report.f0
            ax.axvspan(f0r / 4.0, f0r * 4.0, color=C["accent"], alpha=0.04,
                       zorder=0)

        ax.set_xscale("log")
        ax.set_xlim(freq[0], freq[-1])
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("H/V amplitude")
        top = np.nanpercentile(result.hv_hi, 99.5) if result.hv_hi.size else 1
        ax.set_ylim(0, max(3.0, float(top) * 1.15))
        _legend(ax, loc="upper right")
        if title:
            ax.set_title(title, loc="left")
        self.draw()


class ComponentSpectra(Canvas):
    """Mean Fourier amplitude of each component — the H/V numerator and denominator."""

    def plot(self, result) -> None:
        if result is None or not result.amp:
            self.message("No spectra yet.")
            return
        ax = self.axes(111)
        for comp in ("Z", "N", "E"):
            if comp in result.amp:
                ax.plot(result.freq, result.amp[comp],
                        color=COMPONENT_COLORS[comp], label=comp)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("amplitude")
        ax.set_xlim(result.freq[0], result.freq[-1])
        _legend(ax, loc="best", ncol=3)
        self.draw()


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

class WindowView(Canvas):
    """Traces with the window boxes drawn on them; click a window to toggle it."""

    toggled = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._windows = None
        self._axes: list = []
        self._signature = None

    def _on_click(self, event) -> None:
        if self._windows is None or event.xdata is None or event.inaxes is None:
            return
        if self.navigating():
            return
        if event.inaxes not in self._axes:
            return
        t = float(event.xdata)
        ws = self._windows
        centres = ws.centre_times()
        if centres.size == 0:
            return
        w = int(np.argmin(np.abs(centres - t)))
        half = 0.5 * ws.width_samples / ws.fs
        if abs(centres[w] - t) <= half:
            self.toggled.emit(w)

    def plot(self, segment_data: dict, fs: float, ws, *,
             decimate_to: int = 6000) -> None:
        if ws is None or ws.n == 0:
            self.message("Window the recording to see it here.")
            return

        # Toggling one window redraws everything, so remember where the user
        # had zoomed to and put them back there — as long as it is still the
        # same recording.
        signature = (round(fs, 6), len(next(iter(segment_data.values()))), ws.n)
        previous = (self._axes[0].get_xlim()
                    if self._axes and signature == self._signature else None)
        self._signature = signature

        self.figure.clear()
        comps = [c for c in ("Z", "N", "E") if c in segment_data]
        gs = self.figure.add_gridspec(len(comps) + 1, 1,
                                      height_ratios=[3] * len(comps) + [1.4],
                                      hspace=0.12)
        self._axes = []
        self._windows = ws
        share = None

        for row, comp in enumerate(comps):
            ax = self.figure.add_subplot(gs[row], sharex=share)
            share = share or ax
            x = np.asarray(segment_data[comp], dtype=float)
            step = max(1, x.size // decimate_to)
            t = np.arange(0, x.size, step) / fs
            ax.plot(t, x[::step], color=COMPONENT_COLORS[comp], linewidth=0.5)
            ax.set_ylabel(comp, color=COMPONENT_COLORS[comp])
            ax.tick_params(labelbottom=False)
            ax.margins(x=0)
            self._axes.append(ax)

            for w in range(ws.n):
                if ws.ok[w]:
                    continue
                a = ws.idx[w, 0] / fs
                b = ws.idx[w, 1] / fs
                ax.axvspan(a, b, color=C["bad"], alpha=0.16, linewidth=0)

        ax = self.figure.add_subplot(gs[-1], sharex=share)
        self._axes.append(ax)
        centres = ws.centre_times()
        colours = [C["good"] if ok else C["bad"] for ok in ws.ok]
        if ws.ratio.size == ws.n:
            ax.bar(centres, ws.ratio, width=ws.width_samples / ws.fs * 0.9,
                   color=colours, linewidth=0)
            threshold = float(self.property("threshold") or 0)
            if threshold:
                ax.axhline(threshold, color=C["pick"], linewidth=1.0,
                           linestyle="--")
            ax.set_ylabel("STA/LTA")
        else:
            ax.bar(centres, np.ones(ws.n), color=colours, linewidth=0)
            ax.set_ylabel("kept")
        ax.set_xlabel("time from start of segment (s)")
        ax.margins(x=0)

        if previous is not None:
            ax.set_xlim(previous)

        self.figure.suptitle(
            f"{ws.n_ok} of {ws.n} windows kept", color=C["text_dim"],
            fontsize=9, x=0.01, ha="left")
        self.draw()


# ---------------------------------------------------------------------------
# Azimuth and stability
# ---------------------------------------------------------------------------

class AzimuthView(Canvas):
    """H/V against azimuth: a heat map, and a polar slice at f0."""

    def plot(self, result, f0: float = float("nan")) -> None:
        if result is None or result.hv_azimuth.size == 0:
            self.message("Azimuthal analysis is off.\n"
                         "Set an angular step in the parameters and recompute.")
            return

        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2, width_ratios=[2.0, 1.1], wspace=0.28)
        ax = self.figure.add_subplot(gs[0])

        freq = result.freq
        az = result.azimuths
        data = result.hv_azimuth
        mesh = ax.pcolormesh(az, freq, data, cmap="magma", shading="auto")
        ax.set_yscale("log")
        ax.set_xlabel("azimuth (° clockwise from north)")
        ax.set_ylabel("frequency (Hz)")
        ax.set_xticks([0, 45, 90, 135, 180])
        cb = self.figure.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("H/V", color=C["text_dim"], fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=C["text_dim"])

        if np.isfinite(f0):
            ax.axhline(f0, color=C["pick"], linewidth=1.1, linestyle="--")

        polar = self.figure.add_subplot(gs[1], projection="polar")
        i = (int(np.argmin(np.abs(freq - f0))) if np.isfinite(f0)
             else int(np.nanargmax(result.hv)))
        line = data[i]
        # Mirror 0–180° onto the full circle: the amplitude spectrum cannot
        # tell θ from θ+180°, and showing half a rose invites misreading.
        theta = np.radians(np.concatenate([az, az + 180.0, [az[0]]]))
        radius = np.concatenate([line, line, [line[0]]])
        polar.plot(theta, radius, color=C["accent"], linewidth=1.6)
        polar.fill(theta, radius, color=C["accent"], alpha=0.18)
        polar.set_theta_zero_location("N")
        polar.set_theta_direction(-1)
        polar.set_title(f"{freq[i]:.3g} Hz", fontsize=9, color=C["text_dim"])
        polar.tick_params(labelsize=7)
        polar.grid(color=C["border_soft"])
        self.draw()


class StabilityView(Canvas):
    """How the curve and its peak behave through the recording."""

    def plot(self, result, f0: float = float("nan"), *, utc_offset: float = 0.0
             ) -> None:
        if result is None or result.hv_windows.size == 0:
            self.message("No windows to show.")
            return

        self.figure.clear()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.08)

        times = result.window_times
        if times.size and np.isfinite(times[0]) and times[0] > 1e8:
            t_hours = (times - times[0]) / 3600.0
            start_label = _iso_local(times[0], utc_offset)
        else:
            t_hours = np.arange(result.n_windows) * 1.0
            start_label = "start of segment"

        ax = self.figure.add_subplot(gs[0])
        data = np.where(np.isfinite(result.hv_windows), result.hv_windows, np.nan)
        mesh = ax.pcolormesh(t_hours, result.freq, data, cmap="viridis",
                             shading="auto",
                             vmax=np.nanpercentile(data, 99) if data.size else None)
        ax.set_yscale("log")
        ax.set_ylabel("frequency (Hz)")
        ax.tick_params(labelbottom=False)
        if np.isfinite(f0):
            ax.axhline(f0, color=C["pick"], linewidth=1.0, linestyle="--")
        cb = self.figure.colorbar(mesh, ax=ax, pad=0.01)
        cb.set_label("H/V", color=C["text_dim"], fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=C["text_dim"])

        # Rejected windows are drawn as gaps, not silently averaged away.
        for w in np.where(~result.ok)[0]:
            if w < t_hours.size:
                ax.axvline(t_hours[w], color=C["bad"], alpha=0.12, linewidth=1.2)

        ax2 = self.figure.add_subplot(gs[1], sharex=ax)
        keep = result.ok & np.isfinite(result.window_f0)
        ax2.plot(t_hours[keep], result.window_f0[keep], ".", color=C["accent"],
                 markersize=2.5, alpha=0.7)
        if np.isfinite(f0):
            ax2.axhline(f0, color=C["pick"], linewidth=1.0, linestyle="--")
        ax2.set_yscale("log")
        ax2.set_ylabel("window f₀")
        ax2.set_xlabel(f"hours from {start_label}")
        self.draw()


# ---------------------------------------------------------------------------
# Time selection
# ---------------------------------------------------------------------------

class NoiseScanView(Canvas):
    """The reconnaissance scan: amplitude through the deployment, and by hour."""

    selected = pyqtSignal(float)        # epoch time clicked

    def __init__(self, parent: QWidget | None = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._t0 = 0.0

    def _on_click(self, event) -> None:
        if self.navigating():
            return
        if event.inaxes is not None and event.xdata is not None and event.button == 1:
            self.selected.emit(self._t0 + float(event.xdata) * 86400.0)

    def plot(self, coarse, fine=None, block=None, *, utc_offset: float = 0.0) -> None:
        if coarse is None or coarse.n == 0:
            self.message("Scan the recording to choose a window.")
            return

        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.24)
        ax = self.figure.add_subplot(gs[0])

        self._t0 = float(coarse.times[0])
        days = (coarse.times - self._t0) / 86400.0
        level = coarse.level()
        ax.semilogy(days, level, "-", color=C["muted"], linewidth=0.9,
                    marker=".", markersize=3, label="reconnaissance")

        if fine is not None and fine.n:
            fdays = (fine.times - self._t0) / 86400.0
            ax.semilogy(fdays, fine.level(), "-", color=C["accent"],
                        linewidth=1.2, marker=".", markersize=3, label="refined")

        if block is not None:
            a = (block.start - self._t0) / 86400.0
            b = (block.end - self._t0) / 86400.0
            ax.axvspan(a, b, color=C["pick"], alpha=0.22, linewidth=0,
                       label="selected")
        ax.set_xlabel(f"days from {_iso_local(self._t0, utc_offset)}")
        ax.set_ylabel("ground motion RMS (counts)")
        _legend(ax, loc="upper right", ncol=3)

        ax2 = self.figure.add_subplot(gs[1])
        hours, median = coarse.diurnal(24)
        shifted = (hours + utc_offset) % 24.0
        order = np.argsort(shifted)
        ax2.bar(shifted[order], median[order], width=0.85, color=C["accent"],
                alpha=0.85)
        ax2.set_xlabel("hour of day" + (" (local)" if utc_offset else " (UTC)"))
        ax2.set_ylabel("median RMS")
        ax2.set_yscale("log")
        ax2.set_xticks([0, 6, 12, 18, 24])
        self.draw()


# ---------------------------------------------------------------------------
# Maps, sections, volumes
# ---------------------------------------------------------------------------

class MapView(Canvas):
    """Site positions, an interpolated surface, and any profiles drawn on it."""

    clicked = pyqtSignal(float, float, int)     # x, y, button

    def __init__(self, parent: QWidget | None = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        #: Modifier held during the last click ("control", "shift", …). Read by
        #: the page, so a modifier can change what a click means without every
        #: interaction needing its own signal.
        self.modifier = ""

    def _on_click(self, event) -> None:
        if self.navigating():
            return
        if event.inaxes is not None and event.xdata is not None:
            self.modifier = str(event.key or "")
            self.clicked.emit(float(event.xdata), float(event.ydata),
                              int(event.button))

    def plot(self, x, y, values=None, *, labels=None, grid=None,
             contours: int = 12, style: str = "filled", cmap: str = "viridis",
             title: str = "", unit: str = "", profiles=(), highlight: int = -1,
             show_labels: bool = False, wells=None, members=(),
             well_labels: bool = True) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0:
            self.message("No site coordinates.")
            return

        ax = self.axes(111)

        if grid is not None and np.isfinite(grid.z).any():
            if style == "lines":
                cs = ax.contour(grid.x, grid.y, grid.z, contours, cmap=cmap,
                                linewidths=1.0)
                ax.clabel(cs, inline=True, fontsize=7)
                mappable = cs
            else:
                mappable = ax.contourf(grid.x, grid.y, grid.z, contours,
                                       cmap=cmap, alpha=0.92)
            cb = self.figure.colorbar(mappable, ax=ax, pad=0.02)
            cb.set_label(unit or title, color=C["text_dim"], fontsize=8)
            cb.ax.tick_params(labelsize=7, colors=C["text_dim"])

        if values is not None and np.isfinite(np.asarray(values, float)).any():
            v = np.asarray(values, dtype=float)
            sc = ax.scatter(x, y, c=v, cmap=cmap, s=46, edgecolor=C["bg"],
                            linewidth=0.8, zorder=5)
            if grid is None:
                cb = self.figure.colorbar(sc, ax=ax, pad=0.02)
                cb.set_label(unit or title, color=C["text_dim"], fontsize=8)
                cb.ax.tick_params(labelsize=7, colors=C["text_dim"])
        else:
            ax.scatter(x, y, s=30, color=C["accent"], edgecolor=C["bg"],
                       linewidth=0.7, zorder=5)

        if show_labels and labels is not None:
            for xi, yi, text in zip(x, y, labels):
                ax.annotate(str(text), (xi, yi), xytext=(4, 4),
                            textcoords="offset points", fontsize=6.5,
                            color=C["text_dim"])

        if 0 <= highlight < x.size:
            ax.plot([x[highlight]], [y[highlight]], "o", markersize=15,
                    markerfacecolor="none", markeredgecolor=C["pick"],
                    markeredgewidth=2.0, zorder=6)

        for i, p in enumerate(profiles or ()):
            ax.plot([p.x1, p.x2], [p.y1, p.y2], "-", color=C["pick"],
                    linewidth=1.6, zorder=7)
            ax.annotate(p.name or f"P{i + 1}", ((p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2),
                        color=C["pick"], fontsize=8, fontweight="bold")

        if members is not None and len(members):
            member = np.asarray(list(members), dtype=int)
            member = member[(member >= 0) & (member < x.size)]
            if member.size:
                ax.plot(x[member], y[member], "o", markersize=13,
                        markerfacecolor="none", markeredgecolor=C["good"],
                        markeredgewidth=2.0, zorder=7, linestyle="none",
                        label="on this profile")
                outside = np.setdiff1d(np.arange(x.size), member)
                if outside.size:
                    ax.plot(x[outside], y[outside], "x", markersize=7,
                            color=C["bad"], markeredgewidth=1.3, zorder=7,
                            linestyle="none", label="excluded")
                _legend(ax, loc="lower left", fontsize=7)

        if wells:
            wx = [w.x for w in wells if np.isfinite(w.x)]
            wy = [w.y for w in wells if np.isfinite(w.y)]
            if wx:
                ax.scatter(wx, wy, marker="^", s=80, facecolor=C["good"],
                           edgecolor=C["bg"], linewidth=1.0, zorder=8)
                if well_labels:
                    for well in wells:
                        if not (np.isfinite(well.x) and np.isfinite(well.y)):
                            continue
                        depth = (f"\n{well.bedrock_depth:.0f} m"
                                 if np.isfinite(well.bedrock_depth) else "")
                        ax.annotate(f"{well.name}{depth}", (well.x, well.y),
                                    xytext=(6, -10), textcoords="offset points",
                                    fontsize=6.5, color=C["good"], zorder=8)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("easting (m)")
        ax.set_ylabel("northing (m)")
        if title:
            ax.set_title(title, loc="left")
        self.draw()


class SectionView(Canvas):
    """A pseudo-section along a profile."""

    def plot(self, distance, vertical, section, *, sites=(), depth: bool = True,
             cmap: str = "viridis", title: str = "", clim=None,
             bedrock=None) -> None:
        if section is None or not np.isfinite(section).any():
            self.message("Define a profile with at least two computed sites.")
            return

        ax = self.axes(111)
        kwargs = {}
        if clim:
            kwargs["vmin"], kwargs["vmax"] = clim
        mesh = ax.pcolormesh(distance, vertical, section, cmap=cmap,
                             shading="auto", **kwargs)
        cb = self.figure.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("H/V", color=C["text_dim"], fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=C["text_dim"])

        for d, label in sites:
            ax.axvline(d, color=C["text_faint"], linewidth=0.5, alpha=0.5)
            ax.annotate(label, (d, vertical[0]), xytext=(0, 4),
                        textcoords="offset points", rotation=90, fontsize=6.5,
                        color=C["text_dim"], ha="center")

        if bedrock is not None:
            bd, bz = bedrock
            ax.plot(bd, bz, "-", color=C["pick"], linewidth=1.8,
                    label="bedrock from f₀")
            _legend(ax, loc="lower right")

        ax.set_xlabel("distance along profile (m)")
        if depth:
            ax.set_ylabel("pseudo-depth (m)")
            ax.invert_yaxis()
        else:
            ax.set_ylabel("frequency (Hz)")
            ax.set_yscale("log")
        if title:
            ax.set_title(title, loc="left")
        self.draw()


class Volume3D(Canvas):
    """3D views: site geometry, a draped surface, or H/V slices in the block."""

    def plot_surface(self, grid, *, x=None, y=None, z=None, labels=None,
                     cmap: str = "viridis", title: str = "", unit: str = "",
                     elev: float = 32.0, azim: float = -125.0,
                     zscale: float = 1.0) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.set_facecolor(C["surface"])

        if grid is not None and np.isfinite(grid.z).any():
            with warnings.catch_warnings():
                # The mask is the point: outside the sites there is no surface.
                warnings.simplefilter("ignore", UserWarning)
                surf = ax.plot_surface(grid.x, grid.y, grid.z, cmap=cmap,
                                       linewidth=0, antialiased=True,
                                       alpha=0.92)
            cb = self.figure.colorbar(surf, ax=ax, pad=0.08, shrink=0.65)
            cb.set_label(unit or title, color=C["text_dim"], fontsize=8)
            cb.ax.tick_params(labelsize=7, colors=C["text_dim"])

        if x is not None and z is not None:
            ax.scatter(x, y, z, c=C["pick"], s=18, depthshade=False)
            if labels is not None:
                for xi, yi, zi, text in zip(x, y, z, labels):
                    ax.text(xi, yi, zi, str(text), fontsize=6,
                            color=C["text_dim"])

        _style_3d(ax, zscale)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("easting (m)", fontsize=8)
        ax.set_ylabel("northing (m)", fontsize=8)
        ax.set_zlabel(unit or "value", fontsize=8)
        if title:
            ax.set_title(title, loc="left")
        self.draw()

    def plot_tiles(self, columns, *, cmap: str = "viridis", title: str = "",
                   elev: float = 22.0, azim: float = -125.0) -> None:
        """Each site's H/V curve as a vertical ribbon at its map position.

        ProTO calls these "HVSR tiles". They are the most honest 3D view of an
        H/V survey: no interpolation between sites, so what you see is measured.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.set_facecolor(C["surface"])
        if not columns:
            self.message("No computed sites to draw.")
            return

        vmax = max(float(np.nanpercentile(c["values"], 98)) for c in columns)
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(0, max(vmax, 1e-6))
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        for col in columns:
            depth = np.asarray(col["depth"], dtype=float)
            values = np.asarray(col["values"], dtype=float)
            good = np.isfinite(depth) & np.isfinite(values)
            if good.sum() < 2:
                continue
            xs = np.full(good.sum(), col["x"])
            ys = np.full(good.sum(), col["y"])
            zs = col.get("z", 0.0) - depth[good]
            ax.scatter(xs, ys, zs, c=mapper.to_rgba(values[good]), s=7,
                       marker="s", depthshade=False, linewidth=0)

        cb = self.figure.colorbar(mapper, ax=ax, pad=0.08, shrink=0.65)
        cb.set_label("H/V", color=C["text_dim"], fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=C["text_dim"])
        _style_3d(ax, 1.0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("easting (m)", fontsize=8)
        ax.set_ylabel("northing (m)", fontsize=8)
        ax.set_zlabel("elevation − depth (m)", fontsize=8)
        if title:
            ax.set_title(title, loc="left")
        self.draw()


def _legend(ax, **kwargs) -> None:
    """Draw a legend only when something asked to be in it."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _style_3d(ax, zscale: float) -> None:
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0, 0, 0, 0))
        pane._axinfo["grid"]["color"] = C["border_soft"]
        pane.set_tick_params(labelsize=7, colors=C["text_dim"])
    try:
        ax.set_box_aspect((1, 1, max(0.2, zscale)))
    except Exception:                      # matplotlib < 3.3
        pass


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

class RegressionView(Canvas):
    """Depth against f0: the control points, the fitted law, published laws."""

    def plot(self, f0=None, depth=None, *, fit=None, laws=(), active=None,
             site_f0=None) -> None:
        ax = self.axes(111)
        f_axis = np.logspace(-1, 1.4, 200)

        for law in laws:
            ax.plot(f_axis, law.depth(f_axis), color=C["text_faint"],
                    linewidth=0.8, alpha=0.7)
            ax.annotate(law.name.split(" (")[0], (f_axis[-1], law.depth(f_axis[-1])),
                        fontsize=6.5, color=C["text_faint"], va="center")

        if active is not None:
            ax.plot(f_axis, active.depth(f_axis), color=C["accent"],
                    linewidth=2.0,
                    label=f"active: H = {active.a:.4g}·f^{active.b:.3f}")

        if fit is not None:
            ax.plot(f_axis, fit.depth(f_axis), color=C["pick"], linewidth=1.8,
                    linestyle="--",
                    label=f"fit: H = {fit.a:.4g}·f^{fit.b:.3f}  "
                          f"(n={fit.n}, RMS {fit.rms:.1f} m)")

        if f0 is not None and depth is not None and len(f0):
            ax.plot(f0, depth, "o", color=C["good"], markersize=7,
                    label="borehole control")

        if site_f0 is not None and len(site_f0):
            law = active or fit
            if law is not None:
                ax.plot(site_f0, law.depth(np.asarray(site_f0, float)), ".",
                        color=C["muted"], markersize=4, label="sites")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f₀ (Hz)")
        ax.set_ylabel("depth to bedrock (m)")
        _legend(ax, loc="upper right", fontsize=7)
        self.draw()


class ModelView(Canvas):
    """A Vs model, and the transfer function it predicts against the observed H/V."""

    def plot(self, model=None, *, freq=None, observed=None, predicted=None,
             f0_obs: float = float("nan"), f0_model: float = float("nan")) -> None:
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.28)

        ax = self.figure.add_subplot(gs[0])
        if model is not None and model.n:
            depth = [0.0]
            vs = []
            for layer in model.layers[:-1]:
                vs.append(layer.vs)
                depth.append(depth[-1] + layer.thickness)
            vs.append(model.layers[-1].vs)
            bottom = depth[-1] * 1.35 if depth[-1] > 0 else 50.0
            depth.append(bottom)
            steps_z, steps_v = [], []
            for i, v in enumerate(vs):
                steps_z += [depth[i], depth[i + 1]]
                steps_v += [v, v]
            ax.plot(steps_v, steps_z, color=C["accent"], linewidth=1.8)
            ax.fill_betweenx(steps_z, 0, steps_v, color=C["accent"], alpha=0.12)
            ax.invert_yaxis()
            ax.set_xlabel("Vs (m/s)")
            ax.set_ylabel("depth (m)")
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no model", ha="center", va="center",
                    color=C["text_faint"], transform=ax.transAxes)

        ax2 = self.figure.add_subplot(gs[1])
        if freq is not None and observed is not None:
            ax2.plot(freq, observed, color=C["accent"], linewidth=1.8,
                     label="observed H/V")
        if freq is not None and predicted is not None:
            ax2.plot(freq, predicted, color=C["pick"], linewidth=1.5,
                     linestyle="--", label="SH transfer function")
        for value, colour, label in ((f0_obs, C["accent"], "f₀ observed"),
                                     (f0_model, C["pick"], "f₀ model")):
            if np.isfinite(value):
                ax2.axvline(value, color=colour, linewidth=0.9, linestyle=":")
        ax2.set_xscale("log")
        ax2.set_xlabel("frequency (Hz)")
        ax2.set_ylabel("amplitude")
        _legend(ax2, loc="upper right", fontsize=8)
        ax2.annotate(
            "amplitudes are not comparable — the transfer function is the SH\n"
            "response of the column, H/V is a wavefield property",
            xy=(0.01, 0.02), xycoords="axes fraction", fontsize=6.5,
            color=C["text_faint"])
        self.draw()


class CurveGallery(Canvas):
    """Every site's H/V curve as a small multiple, on one scrolling sheet.

    The point is comparison. One site at a time tells you what that site does;
    a wall of them tells you where the basin deepens, which sites disagree with
    their neighbours, and which ones are simply bad — in about two seconds of
    looking. Each panel is framed in its SESAME colour, so failures stand out
    without reading a single number.
    """

    chosen = pyqtSignal(str)            # sid of the panel that was clicked

    COLUMNS = 5
    PANEL_HEIGHT = 1.35                 # inches

    def __init__(self, parent: QWidget | None = None, **kwargs) -> None:
        kwargs.setdefault("toolbar", False)
        super().__init__(parent, **kwargs)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._panels: list = []

    def _on_click(self, event) -> None:
        if event.inaxes is None or self.navigating():
            return
        for ax, sid in self._panels:
            if ax is event.inaxes:
                self.chosen.emit(sid)
                return

    def plot(self, entries: list[dict], *, current: str = "",
             share_axis: bool = True) -> None:
        """*entries* carry ``sid, label, freq, hv, f0, a0, tone``."""
        self._panels = []
        if not entries:
            self.message("No computed sites yet.\n"
                         "Press “Compute all sites” to fill this in.")
            return

        columns = min(self.COLUMNS, len(entries))
        rows = int(np.ceil(len(entries) / columns))
        self.figure.clear()
        self.figure.set_size_inches(9, max(1.6, rows * self.PANEL_HEIGHT))
        self.canvas.setMinimumHeight(int(rows * self.PANEL_HEIGHT * 96) + 20)

        # One amplitude scale across the whole sheet, or the eye reads panel
        # height as amplitude when it is only autoscaling.
        top = 3.0
        if share_axis:
            peaks = [e["a0"] for e in entries if np.isfinite(e.get("a0", np.nan))]
            if peaks:
                top = max(3.0, float(np.percentile(peaks, 95)) * 1.25)

        for i, entry in enumerate(entries):
            ax = self.figure.add_subplot(rows, columns, i + 1)
            freq = np.asarray(entry["freq"], dtype=float)
            hv = np.asarray(entry["hv"], dtype=float)
            colour = {"good": C["good"], "warn": C["warn"],
                      "bad": C["bad"]}.get(entry.get("tone"), C["muted"])

            ax.plot(freq, hv, color=C["accent"], linewidth=1.1)
            f0 = entry.get("f0", np.nan)
            if np.isfinite(f0):
                ax.axvline(f0, color=C["pick"], linewidth=0.9, linestyle="--")
            ax.set_xscale("log")
            if freq.size:
                ax.set_xlim(freq[0], freq[-1])
            if share_axis:
                ax.set_ylim(0, top)
            ax.tick_params(labelsize=5.5, length=2)
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.set_title(f"{entry['label']}   {f0:.2f} Hz" if np.isfinite(f0)
                         else entry["label"], fontsize=6.5, color=C["text_dim"],
                         pad=2)
            for spine in ax.spines.values():
                spine.set_color(colour)
                spine.set_linewidth(1.8 if entry["sid"] == current else 0.9)
            if entry["sid"] == current:
                ax.set_facecolor(C["surface_alt"])
            self._panels.append((ax, entry["sid"]))

        self.figure.subplots_adjust(left=0.04, right=0.99, top=0.94,
                                    bottom=0.05, hspace=0.75, wspace=0.25)
        self.canvas.draw_idle()

    def draw(self) -> None:
        # The gallery sets its own geometry; tight_layout would fight it.
        self.canvas.draw_idle()


class SummaryView(Canvas):
    """Survey-wide distributions: f0, amplitude, and the SESAME outcome."""

    def plot(self, f0, a0, sesame_scores=None) -> None:
        f0 = np.asarray([v for v in f0 if np.isfinite(v)], dtype=float)
        if f0.size == 0:
            self.message("Compute some sites to see the survey summary.")
            return

        self.figure.clear()
        gs = self.figure.add_gridspec(1, 3, wspace=0.3)

        ax = self.figure.add_subplot(gs[0])
        ax.hist(f0, bins=min(24, max(6, f0.size // 3)), color=C["accent"],
                alpha=0.85)
        ax.set_xlabel("f₀ (Hz)")
        ax.set_ylabel("sites")

        ax2 = self.figure.add_subplot(gs[1])
        a0 = np.asarray([v for v in a0 if np.isfinite(v)], dtype=float)
        if a0.size:
            ax2.plot(f0[:a0.size], a0, "o", color=C["pick"], markersize=5,
                     alpha=0.8)
            ax2.axhline(2.0, color=C["muted"], linestyle="--", linewidth=0.9)
            ax2.annotate("SESAME A₀ > 2", (0.03, 0.9), xycoords="axes fraction",
                         fontsize=7, color=C["text_faint"])
        ax2.set_xscale("log")
        ax2.set_xlabel("f₀ (Hz)")
        ax2.set_ylabel("A₀")

        ax3 = self.figure.add_subplot(gs[2])
        if sesame_scores:
            categories = ["reliable\n& clear", "reliable\nonly", "neither"]
            counts = [0, 0, 0]
            for reliable, clear in sesame_scores:
                counts[0 if (reliable and clear) else (1 if reliable else 2)] += 1
            ax3.bar(categories, counts,
                    color=[C["good"], C["warn"], C["bad"]], alpha=0.9)
            ax3.set_ylabel("sites")
        else:
            ax3.set_axis_off()
        self.draw()


def _iso_local(epoch: float, utc_offset: float = 0.0) -> str:
    try:
        t = datetime.fromtimestamp(epoch + utc_offset * 3600.0, timezone.utc)
        return t.strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "start"
