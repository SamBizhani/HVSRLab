"""Batch & Export — run the whole survey, then get the results out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QSplitter, QTableWidget, QVBoxLayout, QWidget,
)

from ... import batch as batch_mod
from ... import export as export_mod
from ..widgets import (
    Card, LogConsole, ParamForm, StatTile, button, hint, section_label,
    table_item)
from .base import Page

SCOPES = [
    ("pending", "Sites not computed yet"),
    ("active", "All active sites"),
    ("all", "Every site except excluded"),
]


class BatchPage(Page):
    title = "Batch & Export"
    subtitle = ("Compute every site with the current settings, then write "
                "curves, tables, reports and a ProTO project.")

    def build(self) -> None:
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left.setMaximumWidth(370)
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        run_card = Card("Run")
        self.scope = QComboBox()
        for key, label in SCOPES:
            self.scope.addItem(label, key)
        self.scope.currentIndexChanged.connect(self._update_counts)
        run_card.add(self.scope)

        self.form = ParamForm()
        self.form.flag("choose_window", "Scan each recording for its quietest "
                                        "window",
                       tooltip="Off reuses each site's chosen window, or the "
                               "first night in the record where none is set.")
        self.form.number("hours", "Window length", minimum=0.5, maximum=48.0,
                         step=1.0, decimals=1, suffix="h")
        self.form.integer("workers", "Parallel sites", minimum=1, maximum=16,
                          tooltip="Past about four the survey disk, not the "
                                  "processor, sets the pace.")
        self.form.integer("scan_budget", "Probes per scan", minimum=40,
                          maximum=400, step=20)
        run_card.add(self.form)

        row = QHBoxLayout()
        self.run_button = button("Run batch", self._run, primary=True)
        row.addWidget(self.run_button)
        self.stop_button = button("Stop", self._stop)
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)
        run_card.add_layout(row)

        tiles = QHBoxLayout()
        tiles.setSpacing(6)
        self.tile_queued = StatTile("to compute")
        self.tile_done = StatTile("computed")
        self.tile_failed = StatTile("failed")
        for tile in (self.tile_queued, self.tile_done, self.tile_failed):
            tiles.addWidget(tile)
        run_card.add_layout(tiles)
        run_card.add(hint(
            "Each site is written to disk as it finishes, so a run that is "
            "stopped or interrupted keeps everything already done."))

        export_card = Card("Export")
        export_card.add(button("H/V curves (one file per site)",
                               self._export_curves))
        export_card.add(button("Summary table (CSV)", self._export_csv))
        export_card.add(button("Full output set", self._export_full))
        export_card.add(button("OpenHVSR-ProTO project", self._export_proto))
        export_card.add(section_label("report"))
        self.report_form = ParamForm()
        self.report_form.flag("include_sites", "Include a page per site")
        export_card.add(self.report_form)
        export_card.add(button("HTML report", self._export_report, primary=True))
        self.export_note = QLabel("")
        self.export_note.setObjectName("Hint")
        self.export_note.setWordWrap(True)
        export_card.add(self.export_note)

        layout.addWidget(run_card)
        layout.addWidget(export_card)
        layout.addStretch(1)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.console = LogConsole()
        console_card = Card("Progress", dense=True)
        console_card.add(self.console, 1)
        right_layout.addWidget(console_card, 2)

        table_card = Card("Last run", dense=True)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Site", "f₀ (Hz)", "A₀", "SESAME", "Note"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch)
        table_card.add(self.table, 1)
        right_layout.addWidget(table_card, 3)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.add(split, 1)

        self._settings = _Settings()
        self.form.load(self._settings)
        self.report_form.load(_ReportSettings())

    # -- wiring ------------------------------------------------------------
    def connect_signals(self) -> None:
        self.ws.bridge.line.connect(self.console.append)
        self.ws.bridge.progress.connect(self.console.set_progress)
        self.ws.bridge.state.connect(self.console.on_state)
        self.ws.sitesChanged.connect(self._update_counts)
        self.ws.projectChanged.connect(self.refresh)

    def refresh(self) -> None:
        self._update_counts()

    def _targets(self):
        key = self.scope.currentData()
        sites = [s for s in self.project.sites if s.status != "excluded"]
        if key == "active":
            return [s for s in sites if s.is_active]
        if key == "pending":
            return [s for s in sites if s.is_active and not np.isfinite(s.f0)]
        return sites

    def _update_counts(self) -> None:
        self.tile_queued.set(str(len(self._targets())))
        self.tile_done.set(str(self.ws.computed_count()))

    # -- running -----------------------------------------------------------
    def _run(self) -> None:
        targets = self._targets()
        if not targets:
            self.warn("Nothing to compute with the current selection.")
            return
        missing = [s for s in targets if self.ws.recording(s.sid) is None]
        if len(missing) == len(targets):
            self.warn("No MiniSEED catalogue — scan the raw data on the Sites "
                      "page first.")
            return

        self.form.apply(self._settings)
        s = self._settings
        for site in targets:
            if not site.time.hours:
                site.time.hours = s.hours
            elif s.choose_window:
                site.time.hours = s.hours

        self.console.clear()
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        project = self.project
        recordings = dict(self.ws.recordings)

        def work(job):
            job.log_line(f"{len(targets)} site(s), {s.workers} in parallel"
                         + (", scanning each recording for its quietest window"
                            if s.choose_window else ""))
            return batch_mod.run(project, targets, recordings,
                                 workers=s.workers,
                                 choose_window=s.choose_window,
                                 scan_budget=s.scan_budget, job=job)

        self.ws.submit("Batch", work, on_done=self._run_done)

    def _stop(self) -> None:
        self.ws.queue.cancel_current()
        self.warn("Stopping after the sites already in flight finish.")

    def _run_done(self, job) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        report = job.result
        if report is None:
            self.fail(f"Batch failed: {job.error}")
            return

        self.table.setRowCount(len(report.outcomes))
        for row, outcome in enumerate(sorted(report.outcomes,
                                             key=lambda o: (o.ok, o.sid))):
            site = self.project.site(outcome.sid)
            colour = None if outcome.ok else "#f87171"
            self.table.setItem(row, 0, table_item(
                site.label() if site else outcome.sid, colour=colour))
            self.table.setItem(row, 1, table_item(
                _fmt(outcome.f0, 3), align_right=True))
            self.table.setItem(row, 2, table_item(
                _fmt(outcome.a0, 2), align_right=True))
            self.table.setItem(row, 3, table_item(outcome.sesame or "—"))
            self.table.setItem(row, 4, table_item(
                outcome.error or f"{outcome.n_ok} windows, "
                                 f"{outcome.seconds:.1f} s", colour=colour))

        self.tile_failed.set(str(len(report.failed)),
                             "#f87171" if report.failed else None)
        self.ws.touch()
        self.ws.save()
        self.ws.sitesChanged.emit()
        self._update_counts()
        tone = "good" if not report.failed else "warn"
        self.ws.notify.emit(report.summary(), tone)

    # -- exporting ---------------------------------------------------------
    def _results(self) -> dict:
        """Load every stored result. Deliberately not cached — exports are rare."""
        out = {}
        for site in self.project.sites:
            path = self.project.result_path(site.sid)
            if not path.exists():
                continue
            result = self.ws.result(site.sid)
            if result is not None:
                out[site.sid] = result
        return out

    def _export(self, name: str, function) -> None:
        def work(job):
            job.progress_to(0.05, "collecting results")
            results = self._results()
            job.log_line(f"{len(results)} result file(s)")
            job.progress_to(0.3, name)
            path = function(results, job)
            job.log_line(f"wrote {path}")
            job.progress_to(1.0, "done")
            return path

        self.ws.submit(name, work, on_done=self._export_done)

    def _export_done(self, job) -> None:
        if job.state.value != "succeeded":
            self.fail(f"Export failed: {job.error}")
            return
        self.export_note.setText(f"Wrote {job.result}")
        self.ok(f"Exported to {Path(job.result).name}")

    def _export_curves(self) -> None:
        self._export("Export curves",
                     lambda results, job: export_mod.hvsr_curves(
                         self.project, results))

    def _export_csv(self) -> None:
        self._export("Export summary",
                     lambda results, job: export_mod.summary_csv(self.project))

    def _export_full(self) -> None:
        self._export("Export full set",
                     lambda results, job: export_mod.full_output(
                         self.project, results))

    def _export_proto(self) -> None:
        self._export("Export ProTO project",
                     lambda results, job: export_mod.openhvsr_project(
                         self.project, results))

    def _export_report(self) -> None:
        settings = _ReportSettings()
        self.report_form.apply(settings)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the report",
            str(self.project.exports_dir / f"{self.project.name}_report.html"),
            "HTML (*.html)")
        if not path:
            return
        self._export("Build report",
                     lambda results, job: export_mod.html_report(
                         self.project, results, path=Path(path),
                         include_sites=settings.include_sites,
                         progress=lambda f, m: job.progress_to(0.3 + 0.7 * f, m)))


class _Settings:
    choose_window = False
    hours = 8.0
    workers = 4
    scan_budget = 120


class _ReportSettings:
    include_sites = True


def _fmt(value, digits: int) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"
