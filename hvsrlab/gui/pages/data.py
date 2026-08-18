"""Data & Windows — which hours to analyse, and which windows inside them."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QSplitter,
    QVBoxLayout, QWidget,
)

from ...core import timeselect, windows as win_mod
from ...io import mseed
from ..plots import NoiseScanView, WindowView
from ..widgets import (
    Badge, Card, ParamForm, StatTile, button, hint, scroll_column,
    section_label)
from .base import Page

HOUR_CHOICES = [("4", 4.0), ("8", 8.0), ("12", 12.0), ("24", 24.0)]


class DataPage(Page):
    title = "Data & Windows"
    subtitle = ("Pick the quietest hours of the deployment, then cut them into "
                "windows and drop the ones with transients in them.")

    def build(self) -> None:
        self.site_label = QLabel("no site selected")
        self.site_label.setStyleSheet("font-weight: 600;")
        self.header.add_action(self.site_label)

        split = QSplitter(Qt.Vertical)

        # ================= time selection =================================
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        controls = Card("Recording window")

        grid = QGridLayout()
        grid.setSpacing(6)
        grid.addWidget(QLabel("Duration"), 0, 0)
        self.hours_combo = QComboBox()
        for label, value in HOUR_CHOICES:
            self.hours_combo.addItem(f"{label} hours", value)
        self.hours_combo.setCurrentIndex(1)
        grid.addWidget(self.hours_combo, 0, 1)

        grid.addWidget(QLabel("Start (UTC)"), 1, 0)
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        grid.addWidget(self.start_edit, 1, 1)

        grid.addWidget(QLabel("End (UTC)"), 2, 0)
        self.end_edit = QLineEdit()
        self.end_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        grid.addWidget(self.end_edit, 2, 1)
        controls.add_layout(grid)

        self.scan_button = button("Find the quietest window", self._scan,
                                  primary=True)
        controls.add(self.scan_button)
        row = QHBoxLayout()
        row.addWidget(button("Use these times", self._apply_manual))
        self.load_button = button("Load", self._load_segment,
                                  tooltip="Read this window off disk. "
                                          "“Apply” below does it for you.")
        row.addWidget(self.load_button)
        controls.add_layout(row)
        controls.add(button("Apply this window to every site",
                            self._apply_to_all, ghost=True))

        controls.add(section_label("what the scan does"))
        controls.add(hint(
            "Short probes are read across the whole deployment to build its "
            "amplitude history, then a second finer pass ranks candidate "
            "blocks by how quiet and how steady they are. Steadiness counts: a "
            "block that is quiet but lurching gives a worse H/V than one "
            "slightly louder and stationary."))

        controls.add(section_label("coverage"))
        self.coverage_label = QLabel("—")
        self.coverage_label.setObjectName("Hint")
        self.coverage_label.setWordWrap(True)
        controls.add(self.coverage_label)
        controls.add_stretch()
        top_layout.addWidget(scroll_column(controls, 336))

        scan_card = Card("Noise through the deployment", dense=True)
        self.scan_plot = NoiseScanView(height=3.0)
        self.scan_plot.selected.connect(self._scan_clicked)
        scan_card.add(self.scan_plot, 1)
        scan_card.add(hint("Click the trace to move the window there."))
        top_layout.addWidget(scan_card, 1)
        split.addWidget(top)

        # ================= windowing ======================================
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        win_card = Card("Windowing")
        self.form = ParamForm()
        self.form.number("window_width_s", "Window width", minimum=1.0,
                         maximum=3600.0, step=5.0, decimals=1, suffix="s",
                         tooltip="Longer windows resolve lower frequencies: the "
                                 "SESAME reliability test needs f₀ > 10 / width.")
        self.form.number("window_overlap_pc", "Overlap", minimum=0.0,
                         maximum=95.0, step=5.0, decimals=0, suffix="%")
        self.form.number("taper_pc", "Taper each end", minimum=0.0, maximum=50.0,
                         step=1.0, decimals=1, suffix="%",
                         tooltip="Cosine taper, suppressing spectral leakage "
                                 "from the window edges.")
        self.form.flag("antitrigger", "STA/LTA anti-trigger",
                       tooltip="Reject windows containing transients.")
        self.form.number("sta_s", "STA window", minimum=0.05, maximum=60.0,
                         step=0.5, decimals=2, suffix="s")
        self.form.number("lta_s", "LTA window", minimum=1.0, maximum=600.0,
                         step=5.0, decimals=1, suffix="s")
        self.form.number("sta_lta_ratio", "Reject above", minimum=1.0,
                         maximum=50.0, step=0.5, decimals=2,
                         tooltip="A window is dropped when its loudest short-term "
                                 "average exceeds the long-term average by more "
                                 "than this.")
        win_card.add(self.form)

        row = QHBoxLayout()
        self.apply_button = button(
            "Apply", self._apply_windowing, primary=True,
            tooltip="Cut the recording into windows. Reads it off disk first "
                    "if it is not loaded yet.")
        row.addWidget(self.apply_button)
        row.addWidget(button("Keep all", lambda: self._set_all(True)))
        row.addWidget(button("Drop all", lambda: self._set_all(False)))
        win_card.add_layout(row)
        win_card.add(hint("Click a window in the plot to keep or drop it by "
                          "hand; manual choices survive re-applying STA/LTA. "
                          "The toolbar's zoom and pan work here — clicks only "
                          "toggle a window when neither tool is active."))

        stats = QHBoxLayout()
        stats.setSpacing(6)
        self.tile_windows = StatTile("windows")
        self.tile_kept = StatTile("kept")
        self.tile_duration = StatTile("usable")
        for tile in (self.tile_windows, self.tile_kept, self.tile_duration):
            stats.addWidget(tile)
        win_card.add_layout(stats)
        win_card.add_stretch()
        bottom_layout.addWidget(scroll_column(win_card, 336))

        trace_card = Card("Recording and windows", dense=True)
        self.badge = Badge("not loaded", "muted")
        trace_card.body().insertWidget(1, self.badge)
        self.window_plot = WindowView(height=3.6)
        self.window_plot.toggled.connect(self._toggle_window)
        trace_card.add(self.window_plot, 1)
        bottom_layout.addWidget(trace_card, 1)
        split.addWidget(bottom)

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        self.add(split, 1)

        self._windows: win_mod.WindowSet | None = None

    # -- wiring ------------------------------------------------------------
    def connect_signals(self) -> None:
        self.ws.currentChanged.connect(lambda _: self.refresh())
        self.ws.projectChanged.connect(self.refresh)
        self.ws.segmentChanged.connect(lambda _: self.refresh())

    def refresh(self) -> None:
        site = self.site
        if site is None:
            self.site_label.setText("no site selected")
            self.scan_plot.message("Select a site on the Sites page.")
            self.window_plot.message("")
            return

        self.site_label.setText(site.label())
        self.form.load(self.ws.params_for(site))
        self.start_edit.setText(site.time.start)
        self.end_edit.setText(site.time.end)
        index = self.hours_combo.findData(site.time.hours)
        if index >= 0:
            self.hours_combo.setCurrentIndex(index)

        rec = self.ws.recording(site.sid)
        if rec is None:
            self.coverage_label.setText(
                "No catalogue for this site — scan the raw data on the Sites page.")
        else:
            t0, t1 = rec.common_span()
            self.coverage_label.setText(
                f"{'/'.join(rec.components)} at {rec.fs:g} Hz\n"
                f"{mseed.iso(t0)}\n→ {mseed.iso(t1)}\n"
                f"{rec.duration_days:.1f} days, {rec.n_files} files")

        cached = self.ws.scans.get(site.sid)
        block = self._current_block(site)
        if cached:
            self.scan_plot.plot(cached[0], cached[1], block,
                                utc_offset=self.ws.utc_offset)
        else:
            self.scan_plot.message(
                "Press “Find the quietest window” to scan this recording,\n"
                "or type the times directly.")

        self._draw_windows()

    # -- time selection ----------------------------------------------------
    def _current_block(self, site):
        if not site.time.is_set():
            return None
        try:
            return timeselect.Block(start=mseed.to_epoch(site.time.start),
                                    end=mseed.to_epoch(site.time.end))
        except ValueError:
            return None

    def _scan(self) -> None:
        site = self.site
        rec = self.ws.recording(site.sid) if site else None
        if rec is None:
            self.warn("Scan the raw data directory on the Sites page first.")
            return
        hours = float(self.hours_combo.currentData())
        self._set_busy(True, "scanning the recording…")

        def work(job):
            job.log_line(f"{site.label()}: scanning {rec.duration_days:.1f} days")
            block, coarse, fine = timeselect.find_window(
                rec, hours=hours, budget=160,
                progress=lambda i, n, msg: job.counted(i, n, msg))
            job.log_line(f"chose {block.label()}  (level {block.level:.0f}, "
                         f"steadiness {block.steadiness:.2f})")
            return block, coarse, fine

        self.ws.submit(f"Scan {site.label()}", work, on_done=self._scan_done)

    def _scan_done(self, job) -> None:
        self._set_busy(False)
        site = self.site
        if job.state.value != "succeeded" or site is None:
            self.fail(f"Scan failed: {job.error}")
            return
        block, coarse, fine = job.result
        self.ws.scans[site.sid] = (coarse, fine, block)
        site.time.start = mseed.iso(block.start)
        site.time.end = mseed.iso(block.end)
        site.time.hours = block.hours
        site.time.mode = "auto"
        site.time.score = block.score
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self.refresh()
        self.ok(f"Window set to {block.label()}")

    def _scan_clicked(self, epoch: float) -> None:
        site = self.site
        if site is None:
            return
        hours = float(self.hours_combo.currentData())
        site.time.start = mseed.iso(epoch)
        site.time.end = mseed.iso(epoch + hours * 3600.0)
        site.time.hours = hours
        site.time.mode = "manual"
        self.ws.touch()
        self.refresh()

    def _apply_manual(self) -> None:
        site = self.site
        if site is None:
            return
        try:
            t0 = mseed.to_epoch(self.start_edit.text())
            t1 = mseed.to_epoch(self.end_edit.text())
        except ValueError as exc:
            self.fail(str(exc))
            return
        if t1 <= t0:
            self.fail("The end time must be after the start time.")
            return
        site.time.start = mseed.iso(t0)
        site.time.end = mseed.iso(t1)
        site.time.hours = (t1 - t0) / 3600.0
        site.time.mode = "manual"
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self.refresh()

    def _apply_to_all(self) -> None:
        """Give every site the same hour-of-day window as this one.

        Sites are deployed and recovered at different times, so the absolute
        window is shifted to each site's own coverage — what is copied is the
        time of day and the duration, which is what makes the sites comparable.
        """
        site = self.site
        if site is None or not site.time.is_set():
            self.warn("Choose a window for this site first.")
            return

        t0 = mseed.to_epoch(site.time.start)
        hours = site.time.hours
        applied = skipped = 0
        for other in self.project.sites:
            if other.sid == site.sid or other.status == "excluded":
                continue
            rec = self.ws.recording(other.sid)
            if rec is None:
                skipped += 1
                continue
            block = timeselect.clip_to_coverage(
                rec, timeselect.Block(start=_same_time_of_day(t0, rec),
                                      end=_same_time_of_day(t0, rec) + hours * 3600))
            other.time.start = mseed.iso(block.start)
            other.time.end = mseed.iso(block.end)
            other.time.hours = hours
            other.time.mode = "manual"
            applied += 1

        self.ws.touch()
        self.ws.sitesChanged.emit()
        message = f"Applied the same {hours:g}-hour window to {applied} site(s)."
        if skipped:
            message += f" {skipped} skipped — not catalogued."
        self.ok(message)

    def _load_segment(self) -> None:
        site = self.site
        if site is None:
            return
        if not site.time.is_set():
            self.warn("Choose a window first.")
            return

        self._set_busy(True, "reading the recording…")

        def work(job):
            job.progress_to(0.1, "reading")
            segment = self.ws.load_segment(site, job)
            job.log_line(f"{site.label()}: {segment.npts} samples at "
                         f"{segment.fs:g} Hz ({segment.duration / 3600:.2f} h)"
                         + (f", {segment.gaps} gap(s) filled" if segment.gaps else ""))
            job.progress_to(1.0, "loaded")
            return segment

        self.ws.submit(f"Load {site.label()}", work, on_done=self._load_done)

    def _load_done(self, job) -> None:
        self._set_busy(False)
        if job.state.value != "succeeded":
            self.fail(f"Load failed: {job.error}")
            self.badge.set("load failed", "bad")
            return
        self._windows = None
        self._apply_windowing()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        """Say plainly that something is happening — reading eight hours off
        the survey drive takes long enough that a silent interface reads as a
        broken one."""
        for widget in (self.load_button, self.apply_button, self.scan_button):
            widget.setEnabled(not busy)
        if busy:
            self.badge.set(message or "working…", "info")

    # -- windowing ---------------------------------------------------------
    def _apply_windowing(self) -> None:
        site = self.site
        if site is None:
            return
        segment = self.ws.segment(site.sid)
        if segment is None:
            # Nothing loaded yet. Rather than telling the user to press the
            # other button first, do it — the load ends in this same method.
            self._load_segment()
            return

        params = self.ws.params_for(site)
        self.form.apply(params)
        _store_overrides(site, self.project.params, params)
        self.ws.touch()

        try:
            ws = win_mod.make_windows(segment.npts, segment.fs,
                                      params.window_width_s,
                                      params.window_overlap_pc, segment.start)
            if params.antitrigger:
                win_mod.sta_lta_mask(segment.data, ws, sta_s=params.sta_s,
                                     lta_s=params.lta_s,
                                     threshold=params.sta_lta_ratio)
        except ValueError as exc:
            self.fail(str(exc))
            return

        self._windows = ws
        self._draw_windows()

    def _set_all(self, keep: bool) -> None:
        if self._windows is None:
            return
        self._windows.set_all(keep)
        self._draw_windows()

    def _toggle_window(self, index: int) -> None:
        if self._windows is None:
            return
        self._windows.toggle(index)
        self._draw_windows()

    def _draw_windows(self) -> None:
        site = self.site
        segment = self.ws.segment(site.sid) if site else None
        if segment is None:
            self.badge.set("not loaded", "muted")
            self.window_plot.message(
                "Choose a window above, then press “Apply”.")
            self.tile_windows.set("—")
            self.tile_kept.set("—")
            self.tile_duration.set("—")
            return

        self.badge.set(f"{segment.npts:,} samples at {segment.fs:g} Hz", "info")
        ws = self._windows
        if ws is None:
            self.window_plot.message("Press “Apply” to window this recording.")
            return

        params = self.ws.params_for(site)
        self.window_plot.setProperty("threshold", params.sta_lta_ratio)
        self.window_plot.plot(segment.data, segment.fs, ws)

        self.tile_windows.set(str(ws.n))
        self.tile_kept.set(str(ws.n_ok),
                           "#34d399" if ws.n_ok >= 10 else "#fbbf24")
        self.tile_duration.set(_duration(_covered_seconds(ws)))

    def current_windows(self) -> win_mod.WindowSet | None:
        """The window set the analysis page should reuse, if the user edited one."""
        return self._windows


def _covered_seconds(ws) -> float:
    """How much of the recording the kept windows actually span.

    Not the sum of their lengths: at 50 % overlap that counts most of the
    record twice and would report 15 hours of "usable" data inside an
    8-hour segment. Overlapping windows are legitimate for averaging, but
    they do not add independent recording.
    """
    if not ws.n_ok:
        return 0.0
    covered = np.zeros(int(ws.idx[-1, 1]), dtype=bool)
    for a, b in ws.idx[ws.ok]:
        covered[a:b] = True
    return float(covered.sum()) / ws.fs


def _duration(seconds: float) -> str:
    if seconds >= 5400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 60:.0f} min"


def _same_time_of_day(reference: float, rec) -> float:
    """The first instant in *rec* at the same UTC time of day as *reference*."""
    t0, t1 = rec.common_span()
    day = np.floor(t0 / 86400.0) * 86400.0
    offset = reference % 86400.0
    candidate = day + offset
    while candidate < t0:
        candidate += 86400.0
    if candidate > t1:
        candidate = t0
    return float(candidate)


def _store_overrides(site, defaults, params) -> None:
    """Record only the fields where this site differs from the project defaults."""
    from dataclasses import asdict

    base = asdict(defaults)
    site.params = {k: v for k, v in asdict(params).items() if base.get(k) != v}
