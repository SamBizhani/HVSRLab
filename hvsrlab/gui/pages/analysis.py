"""H/V Analysis — the curve, the pick, and whether either can be trusted."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QScrollArea, QSplitter, QTabWidget, QTableWidget, QVBoxLayout,
    QWidget,
)

from ... import batch as batch_mod
from ...core import hvsr as hvsr_core
from ...core import picking, sesame
from ...project import HVSR_STRATEGY_LABELS
from ..plots import (
    AzimuthView, ComponentSpectra, CurveGallery, HVSRCurve, StabilityView)
from ..widgets import (
    Badge, Card, ParamForm, StatTile, button, hint, scroll_column,
    section_label, table_item)
from .base import Page


class AnalysisPage(Page):
    title = "H/V Analysis"
    subtitle = ("Compute the spectral ratio, pick the resonance, and check it "
                "against the SESAME criteria.")

    def build(self) -> None:
        self.site_label = QLabel("no site selected")
        self.site_label.setStyleSheet("font-weight: 600;")
        self.header.add_action(self.site_label)
        self.verdict = Badge("not computed", "muted")
        self.header.add_action(self.verdict)

        split = QSplitter(Qt.Horizontal)

        # ================= controls =======================================
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        params_card = Card("Processing")
        self.form = ParamForm()
        self.form.number("freq_min", "Frequency from", minimum=0.01,
                         maximum=100.0, step=0.1, decimals=2, suffix="Hz")
        self.form.number("freq_max", "to", minimum=0.1, maximum=500.0,
                         step=1.0, decimals=2, suffix="Hz")
        self.form.number("target_fs", "Decimate to", minimum=0.0,
                         maximum=1000.0, step=10.0, decimals=1, suffix="Hz",
                         tooltip="0 keeps the native rate. Decimating to about "
                                 "2.5× the highest frequency of interest costs "
                                 "nothing and saves a lot of time.")
        self.form.choice("hvsr_strategy", "Horizontals",
                         [(k, v) for k, v in HVSR_STRATEGY_LABELS.items()],
                         tooltip="How E and N are combined into H. Total Energy "
                                 "is √2 larger than Average Squared, which "
                                 "matters for the A₀ > 2 criterion.")
        self.form.choice("smoothing_kind", "Smoothing",
                         [("konno_ohmachi", "Konno-Ohmachi"),
                          ("moving_average", "Moving average"),
                          ("none", "None")])
        self.form.number("smoothing_b", "Bandwidth b", minimum=1.0,
                         maximum=200.0, step=5.0, decimals=1,
                         tooltip="Konno-Ohmachi bandwidth. Small values smooth "
                                 "hard; 40 is the usual choice.")
        self.form.choice("statistics", "Statistics",
                         [("lognormal", "Log-normal (geometric mean)"),
                          ("linear", "Linear (ProTO)")],
                         tooltip="H/V is a ratio of positive quantities, so the "
                                 "log-normal form is the better-behaved one and "
                                 "is what SESAME's σ thresholds assume.")
        self.form.choice("freq_grid", "Frequency axis",
                         [("log", "Logarithmic"), ("linear", "Linear (ProTO)")])
        self.form.integer("n_freq", "Points", minimum=64, maximum=4096, step=64)
        self.form.number("azimuth_step_deg", "Azimuth step", minimum=0.0,
                         maximum=90.0, step=5.0, decimals=0, suffix="°",
                         tooltip="0 turns the azimuthal analysis off. 10° is a "
                                 "good compromise between detail and time.")
        params_card.add(self.form)

        filter_card = Card("Pre-filter")
        self.filter_form = ParamForm()
        self.filter_form.choice("filter_kind", "Type",
                                [("off", "Off"), ("bandpass", "Band-pass"),
                                 ("lowpass", "Low-pass"), ("highpass", "High-pass")])
        self.filter_form.integer("filter_order", "Order", minimum=1, maximum=10)
        self.filter_form.number("filter_fmin", "Corner low", minimum=0.001,
                                maximum=500.0, step=0.1, decimals=3, suffix="Hz")
        self.filter_form.number("filter_fmax", "Corner high", minimum=0.01,
                                maximum=1000.0, step=1.0, decimals=3, suffix="Hz")
        self.filter_form.choice("filter_target", "Applies to",
                                [("hvsr", "The H/V itself"),
                                 ("antitrigger_only", "Window selection only")])
        filter_card.add(self.filter_form)
        filter_card.add(hint(
            "Zero-phase Butterworth. Filtering before H/V removes energy from "
            "both numerator and denominator, so it changes the curve very "
            "little — its real use is keeping a swamping low-frequency drift "
            "out of the STA/LTA decision."))

        actions = Card("Run")
        row = QHBoxLayout()
        self.compute_button = button("Compute this site", self._compute,
                                     primary=True)
        row.addWidget(self.compute_button)
        row.addWidget(button("Re-pick automatically", self._auto_pick))
        actions.add_layout(row)

        row = QHBoxLayout()
        self.all_button = button(
            "Compute all sites", self._compute_all,
            tooltip="Apply these settings to every active site, in parallel. "
                    "Results appear in the All sites tab as they land.")
        row.addWidget(self.all_button)
        self.stop_button = button("Stop", self._stop_all)
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)
        actions.add_layout(row)
        actions.add(hint("Get one site right here first — “Compute all "
                         "sites” uses exactly these settings for the whole "
                         "survey."))
        actions.add(hint("Left-click the curve to set f₀ by hand; right-click "
                         "to add a secondary peak."))

        tiles = QHBoxLayout()
        tiles.setSpacing(6)
        self.tile_f0 = StatTile("f₀ (Hz)")
        self.tile_a0 = StatTile("A₀")
        self.tile_depth = StatTile("depth (m)")
        for tile in (self.tile_f0, self.tile_a0, self.tile_depth):
            tiles.addWidget(tile)
        actions.add_layout(tiles)

        tiles2 = QHBoxLayout()
        tiles2.setSpacing(6)
        self.tile_windows = StatTile("windows kept")
        self.tile_sigma = StatTile("σ f₀ (Hz)")
        for tile in (self.tile_windows, self.tile_sigma):
            tiles2.addWidget(tile)
        actions.add_layout(tiles2)

        self.pick_note = QLabel("")
        self.pick_note.setObjectName("Hint")
        self.pick_note.setWordWrap(True)
        actions.add(self.pick_note)
        actions.add(button("Clear secondary peaks", self._clear_extra, ghost=True))

        left_layout.addWidget(params_card)
        left_layout.addWidget(filter_card)
        left_layout.addWidget(actions)
        left_layout.addStretch(1)
        split.addWidget(scroll_column(left, 348))

        # ================= views ==========================================
        self.tabs = QTabWidget()

        curve_page = QWidget()
        curve_layout = QVBoxLayout(curve_page)
        curve_layout.setContentsMargins(6, 6, 6, 6)
        curve_layout.setSpacing(6)

        toggles = QHBoxLayout()
        self.show_windows = QCheckBox("show individual windows")
        self.show_windows.setChecked(True)
        self.show_windows.stateChanged.connect(self._redraw)
        toggles.addWidget(self.show_windows)
        self.show_components = QCheckBox("show E/V and N/V")
        self.show_components.stateChanged.connect(self._redraw)
        toggles.addWidget(self.show_components)
        toggles.addStretch(1)
        curve_layout.addLayout(toggles)

        self.curve = HVSRCurve(height=3.6)
        self.curve.picked.connect(self._pick_f0)
        self.curve.extraPicked.connect(self._pick_extra)
        curve_layout.addWidget(self.curve, 3)

        self.spectra = ComponentSpectra(height=2.0, toolbar=False)
        curve_layout.addWidget(self.spectra, 2)
        self.tabs.addTab(curve_page, "Curve")

        sesame_page = QWidget()
        sesame_layout = QVBoxLayout(sesame_page)
        sesame_layout.setContentsMargins(8, 8, 8, 8)
        self.sesame_summary = QLabel("")
        self.sesame_summary.setWordWrap(True)
        sesame_layout.addWidget(self.sesame_summary)
        self.criteria = QTableWidget(0, 5)
        self.criteria.setHorizontalHeaderLabels(
            ["", "Criterion", "Measured", "Threshold", "Note"])
        self.criteria.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.criteria.verticalHeader().setVisible(False)
        self.criteria.setAlternatingRowColors(True)
        head = self.criteria.horizontalHeader()
        head.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in (0, 2, 3, 4):
            head.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        sesame_layout.addWidget(self.criteria, 1)
        sesame_layout.addWidget(hint(
            "SESAME (2004), Annex A. All three reliability conditions and at "
            "least five of the six clarity criteria are required. A peak that "
            "fails is not necessarily wrong — but it is not evidence."))
        self.tabs.addTab(sesame_page, "SESAME")

        self.azimuth = AzimuthView(height=3.6)
        self.tabs.addTab(self.azimuth, "Azimuth")

        self.stability = StabilityView(height=3.6)
        self.tabs.addTab(self.stability, "Stability")

        gallery_page = QWidget()
        gallery_layout = QVBoxLayout(gallery_page)
        gallery_layout.setContentsMargins(6, 6, 6, 6)
        gallery_layout.setSpacing(6)

        controls = QHBoxLayout()
        self.gallery_note = QLabel("")
        self.gallery_note.setObjectName("Hint")
        controls.addWidget(self.gallery_note, 1)
        self.share_axis = QCheckBox("same amplitude scale")
        self.share_axis.setChecked(True)
        self.share_axis.stateChanged.connect(self._draw_gallery)
        controls.addWidget(self.share_axis)
        controls.addWidget(button("Refresh", self._draw_gallery))
        gallery_layout.addLayout(controls)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self.gallery = CurveGallery()
        self.gallery.chosen.connect(self._gallery_chosen)
        scroll.setWidget(self.gallery)
        gallery_layout.addWidget(scroll, 1)
        gallery_layout.addWidget(hint(
            "One panel per computed site, framed in its SESAME colour: green "
            "reliable and clear, amber reliable but the peak is not clean, red "
            "neither. Click a panel to open that site."))
        self.tabs.addTab(gallery_page, "All sites")
        self.tabs.currentChanged.connect(self._tab_changed)

        split.addWidget(self.tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        self.add(split, 1)

    def _tab_changed(self, index: int) -> None:
        if index == self.tabs.count() - 1:
            self._draw_gallery()

    # -- wiring ------------------------------------------------------------
    def connect_signals(self) -> None:
        self.ws.currentChanged.connect(lambda _: self.refresh())
        self.ws.resultChanged.connect(lambda _: self.refresh())
        self.ws.projectChanged.connect(self.refresh)

    def refresh(self) -> None:
        if self.tabs.currentIndex() == self.tabs.count() - 1:
            self._draw_gallery()
        site = self.site
        if site is None:
            self.site_label.setText("no site selected")
            self.curve.message("Select a site on the Sites page.")
            return

        self.site_label.setText(site.label())
        params = self.ws.params_for(site)
        self.form.load(params)
        self.filter_form.load(params)
        self._redraw()

    # -- computing ---------------------------------------------------------
    def _collect_params(self):
        site = self.site
        params = self.ws.params_for(site)
        self.form.apply(params)
        self.filter_form.apply(params)
        from dataclasses import asdict
        base = asdict(self.project.params)
        site.params = {k: v for k, v in asdict(params).items() if base.get(k) != v}
        self.ws.touch()
        return params

    def _compute(self) -> None:
        site = self.site
        if site is None:
            return
        params = self._collect_params()
        if params.freq_max <= params.freq_min:
            self.fail("The upper frequency must exceed the lower one.")
            return

        reuse = self._edited_windows()
        self.compute_button.setEnabled(False)

        def work(job):
            segment = self.ws.segment(site.sid)
            if segment is None or (params.target_fs and
                                   abs(segment.fs - params.target_fs) > 1e-6):
                job.progress_to(0.05, "reading the recording")
                segment = self.ws.load_segment(site, job)
            job.log_line(f"{site.label()}: {segment.duration / 3600:.2f} h at "
                         f"{segment.fs:g} Hz")
            result = hvsr_core.compute(
                segment.data, segment.fs, params, sid=site.sid,
                start=segment.start, windows=reuse,
                progress=lambda f, m: job.progress_to(0.1 + 0.9 * f, m))
            job.log_line(f"{result.n_ok}/{result.n_windows} windows, "
                         f"{result.freq.size} frequencies")
            return result

        self.ws.submit(f"Compute {site.label()}", work, on_done=self._compute_done)

    def _edited_windows(self):
        """Reuse the Data page's window set when the user hand-edited it."""
        window = self.window()
        page = getattr(window, "pages", {}).get("Data & Windows")
        if page is None:
            return None
        ws = page.current_windows()
        site = self.site
        segment = self.ws.segment(site.sid) if site else None
        if ws is None or segment is None:
            return None
        if ws.manual.size and ws.manual.any() and \
                abs(ws.fs - segment.fs) < 1e-9 and \
                ws.idx[-1, 1] <= segment.npts:
            return ws
        return None

    def _compute_done(self, job) -> None:
        self.compute_button.setEnabled(True)
        if job.state.value != "succeeded":
            self.fail(f"Computation failed: {job.error}")
            return
        site = self.site
        if site is None:
            return
        site.f0_source = ""            # a fresh run re-picks automatically
        self.ws.set_result(site.sid, job.result)
        self.ws.sitesChanged.emit()
        self.ok(f"{site.label()}: f₀ = {site.f0:.3g} Hz, A₀ = {site.a0:.2f}")

    # -- the whole survey ---------------------------------------------------
    def _compute_all(self) -> None:
        params = self._collect_params()
        targets = [s for s in self.project.sites if s.is_active]
        if not targets:
            self.warn("No active sites.")
            return
        if all(self.ws.recording(s.sid) is None for s in targets):
            self.warn("No MiniSEED catalogue \u2014 scan the raw data on the "
                      "Sites page first.")
            return

        answer = QMessageBox.question(
            self, "Compute the whole survey",
            f"Compute {len(targets)} site(s) with the settings on this page?"
            f"\n\nBand {params.freq_min:g}\u2013{params.freq_max:g} Hz \u00b7 "
            f"{params.window_width_s:g} s windows \u00b7 "
            + ("azimuthal analysis on" if params.azimuth_step_deg
               else "no azimuthal analysis")
            + "\n\nThese become the project defaults, replacing any per-site "
              "overrides. Sites with no chosen window will use the first night "
              "in their recording.\n\nProgress appears in the Activity panel.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes)
        if answer != QMessageBox.Yes:
            return

        # One parameter set for the survey: results computed with different
        # settings per site are not comparable, which is the whole point of
        # mapping them together.
        self.project.params = params.copy()
        for site in targets:
            site.params = {}

        self.all_button.setEnabled(False)
        self.compute_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)

        project = self.project
        recordings = dict(self.ws.recordings)

        def work(job):
            job.log_line(f"{len(targets)} site(s), 4 in parallel")
            return batch_mod.run(project, targets, recordings, workers=4,
                                 choose_window=False, job=job)

        self.ws.submit("Compute all sites", work, on_done=self._compute_all_done)

    def _stop_all(self) -> None:
        self.ws.queue.cancel_current()
        self.warn("Stopping after the sites already in flight finish.")

    def _compute_all_done(self, job) -> None:
        self.all_button.setEnabled(True)
        self.compute_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        report = job.result
        if report is None:
            self.fail(f"Survey run failed: {job.error}")
            return
        for outcome in report.failed:
            self.ws.log(f"{outcome.sid}: {outcome.error}", "error")
        self.ws.touch()
        self.ws.save()
        self.ws.sitesChanged.emit()
        self._draw_gallery()
        self.ws.notify.emit(report.summary(),
                            "good" if not report.failed else "warn")

    def _draw_gallery(self) -> None:
        entries = []
        for site in self.project.sites:
            if site.status == "excluded":
                continue
            path = self.project.result_path(site.sid)
            if not path.exists():
                continue
            try:
                freq, hv, _ = hvsr_core.load_curve(path)
            except Exception as exc:                   # noqa: BLE001
                self.ws.log(f"{site.label()}: cannot read {path.name} \u2014 "
                            f"{exc}", "error")
                continue
            entries.append({
                "sid": site.sid, "label": site.label(), "freq": freq, "hv": hv,
                "f0": site.f0, "a0": site.a0, "tone": _tone(site.sesame_score),
            })

        self.gallery.plot(entries, current=self.ws.current_sid,
                          share_axis=self.share_axis.isChecked())
        good = sum(1 for e in entries if e["tone"] == "good")
        self.gallery_note.setText(
            f"{len(entries)} computed \u00b7 {good} reliable and clear"
            if entries else "nothing computed yet")

    def _gallery_chosen(self, sid: str) -> None:
        self.ws.set_current(sid)
        self.tabs.setCurrentIndex(0)

    # -- picking -----------------------------------------------------------
    def _pick_f0(self, frequency: float) -> None:
        site = self.site
        result = self.ws.result(site.sid) if site else None
        if result is None:
            return
        peak = picking.pick_nearest(result.freq, result.hv, frequency)
        site.f0, site.a0 = peak.frequency, peak.amplitude
        site.f0_source = "user"
        report = sesame.evaluate(result, peak.frequency)
        site.sesame_score = report.summary
        result.meta["f0"] = peak.frequency
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self._redraw()

    def _pick_extra(self, frequency: float) -> None:
        site = self.site
        result = self.ws.result(site.sid) if site else None
        if result is None:
            return
        peak = picking.pick_nearest(result.freq, result.hv, frequency)
        site.extra_peaks.append([peak.frequency, peak.amplitude])
        self.ws.touch()
        self._redraw()

    def _clear_extra(self) -> None:
        if self.site is not None:
            self.site.extra_peaks = []
            self.ws.touch()
            self._redraw()

    def _auto_pick(self) -> None:
        site = self.site
        result = self.ws.result(site.sid) if site else None
        if result is None:
            return
        site.f0_source = ""
        self.ws._sync_site_from_result(site.sid, result)
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self._redraw()

    # -- drawing -----------------------------------------------------------
    def _redraw(self) -> None:
        site = self.site
        if site is None:
            return
        result = self.ws.result(site.sid)
        if result is None:
            self.verdict.set("not computed", "muted")
            self.curve.message(
                f"{site.label()} has no H/V yet.\nPress “Compute this site”.")
            self.spectra.message("")
            self.azimuth.message("")
            self.stability.message("")
            self._clear_tiles()
            self.criteria.setRowCount(0)
            self.sesame_summary.setText("")
            return

        f0 = site.f0 if np.isfinite(site.f0) else float("nan")
        report = sesame.evaluate(result, f0)

        self.curve.show_windows = self.show_windows.isChecked()
        self.curve.show_components = self.show_components.isChecked()
        self.curve.plot(result, f0=f0, extra_peaks=site.extra_peaks,
                        sesame_report=report, title=site.label())
        self.spectra.plot(result)
        self.azimuth.plot(result, f0)
        self.stability.plot(result, f0, utc_offset=self.ws.utc_offset)
        self._fill_criteria(report)
        self._fill_tiles(site, result, report)

    def _fill_criteria(self, report) -> None:
        criteria = report.all_criteria()
        self.criteria.setRowCount(len(criteria))
        for row, c in enumerate(criteria):
            colour = "#34d399" if c.passed else "#f87171"
            self.criteria.setItem(row, 0, table_item(
                "PASS" if c.passed else "FAIL", colour=colour, bold=True))
            self.criteria.setItem(row, 1, table_item(f"{c.key}  {c.text}"))
            self.criteria.setItem(row, 2, table_item(
                _fmt(c.value), align_right=True))
            self.criteria.setItem(row, 3, table_item(
                _fmt(c.threshold), align_right=True))
            self.criteria.setItem(row, 4, table_item(c.detail))

        tone = "good" if (report.reliable and report.clear) else (
            "warn" if report.reliable else "bad")
        self.verdict.set(f"{report.summary} · {report.verdict}", tone)
        note = (f"Reliability {report.n_reliability}/3 (all three required) · "
                f"Clarity {report.n_clarity}/6 (five required) — "
                f"{report.verdict}.")
        if report.notes:
            note += "  " + "  ".join(report.notes)
        self.sesame_summary.setText(note)

    def _fill_tiles(self, site, result, report) -> None:
        from ...core import bedrock

        self.tile_f0.set(_fmt(site.f0, 3), "#38bdf8")
        self.tile_a0.set(_fmt(site.a0, 2),
                         "#34d399" if site.a0 > 2 else "#fbbf24")
        law = self.project.regression
        depth = bedrock.depth_from_f0(site.f0, law.a, law.b)
        self.tile_depth.set(_fmt(depth, 1))
        self.tile_windows.set(f"{result.n_ok}/{result.n_windows}")
        self.tile_sigma.set(_fmt(site.f0_std, 3))

        source = "picked by hand" if site.f0_source == "user" else "automatic"
        extra = (f"; {len(site.extra_peaks)} secondary peak(s)"
                 if site.extra_peaks else "")
        stats = picking.window_peak_statistics(result.window_f0, result.ok)
        self.pick_note.setText(
            f"f₀ {source}{extra}. Window-to-window peak scatter: "
            f"σ = {_fmt(stats.get('std'), 3)} Hz, "
            f"log-normal σ = {_fmt(stats.get('sigma_log'), 3)} "
            f"over {int(stats.get('n', 0))} windows.")

    def _clear_tiles(self) -> None:
        for tile in (self.tile_f0, self.tile_a0, self.tile_depth,
                     self.tile_windows, self.tile_sigma):
            tile.set("—")
        self.pick_note.setText("")


def _tone(score: str) -> str:
    """A SESAME summary as a colour name: green, amber or red."""
    try:
        reliability, clarity = score.split("·")
        r = int(reliability.strip().split("/")[0])
        c = int(clarity.strip().split("/")[0])
    except (ValueError, IndexError, AttributeError):
        return "muted"
    if r == 3 and c >= 5:
        return "good"
    return "warn" if r == 3 else "bad"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "—"
    if abs(value) >= 1e4 or (value and abs(value) < 1e-3):
        return f"{value:.3g}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")
