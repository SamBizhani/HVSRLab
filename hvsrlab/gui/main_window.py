"""The main window: navigation, pages, the job console and the menus."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAction, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QStackedWidget, QVBoxLayout, QWidget,
)

from .. import diagnostics, paths
from ..project import Project
from .activity import ActivityLog, install_handlers
from .pages import PAGES
from .state import Workspace
from .theme import C
from .widgets import Badge, button


class MainWindow(QMainWindow):
    def __init__(self, project: Project | None = None) -> None:
        super().__init__()
        self.ws = Workspace(project)

        self.setWindowTitle("HVSRLab")
        self.resize(1560, 980)

        self._build_central()
        self._build_console_dock()
        self._build_menu()
        self._build_status()

        self.ws.notify.connect(self._notify)
        self.ws.projectChanged.connect(self._retitle)
        self.ws.sitesChanged.connect(self._update_status)
        self.ws.bridge.state.connect(self._job_state)
        self.ws.bridge.progress.connect(self._job_progress)
        self.ws.projectChanged.connect(
            lambda: self.console.set_log_file(
                self.ws.project.root / "logs" / "session.log"))

        self._retitle()
        self._select(0)

    # -- construction ------------------------------------------------------
    def _build_central(self) -> None:
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        side = QWidget()
        side.setFixedWidth(186)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)

        brand = QLabel("HVSR<b>Lab</b>")
        brand.setStyleSheet(
            f"color: {C['text']}; font-size: 15pt; padding: 14px 14px 8px 14px;"
            f"background: {C['surface']};")
        side_layout.addWidget(brand)

        self.nav = QListWidget()
        self.nav.setObjectName("Nav")
        for title, _ in PAGES:
            self.nav.addItem(QListWidgetItem(title))
        self.nav.currentRowChanged.connect(self._select)
        side_layout.addWidget(self.nav, 1)

        self.project_label = QLabel("")
        self.project_label.setWordWrap(True)
        self.project_label.setObjectName("Hint")
        self.project_label.setStyleSheet(
            f"padding: 10px 12px; background: {C['surface']};"
            f"border-top: 1px solid {C['border_soft']};")
        side_layout.addWidget(self.project_label)
        outer.addWidget(side)

        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {}
        for title, factory in PAGES:
            page = factory(self.ws)
            self.pages[title] = page
            self.stack.addWidget(page)
        outer.addWidget(self.stack, 1)

        self.setCentralWidget(central)

    def _build_console_dock(self) -> None:
        self.console = ActivityLog()
        self.console.set_environment_provider(
            lambda: diagnostics.environment_report(self.ws.project))

        dock = QDockWidget("Activity", self)
        dock.setObjectName("ActivityDock")
        dock.setWidget(self.console)
        dock.setFeatures(QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable |
                         QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        dock.setMinimumHeight(150)
        self.resizeDocks([dock], [190], Qt.Vertical)
        self.console_dock = dock

        # Anything that goes wrong anywhere in the process lands here, not in
        # a stderr nobody is watching.
        self._log_bridge = install_handlers(self.console)

        self.ws.bridge.line.connect(lambda text: self.console.append(text, "info"))
        self.ws.bridge.state.connect(self._log_job_state)
        self.ws.logged.connect(self.console.append)

        self.console.set_log_file(self.ws.project.root / "logs" / "session.log")
        self.console.append(diagnostics.environment_report(self.ws.project),
                            "debug")
        for note in diagnostics.compatibility_warnings():
            self.console.append(note, "warn")

    def _log_job_state(self, job) -> None:
        if job.state.value == "running":
            self.console.append(f"▶ {job.name}", "info")
        elif job.state.finished:
            level = {"succeeded": "good", "failed": "error"}.get(
                job.state.value, "warn")
            self.console.append(
                f"■ {job.name} — {job.state.value} in {job.elapsed:.1f} s", level)
            if job.traceback:
                self.console.append(job.traceback, "error")
            elif job.error:
                self.console.append(job.error, "error")

    def _build_menu(self) -> None:
        menu = self.menuBar()

        files = menu.addMenu("&Project")
        files.addAction(_action(self, "New…", self._new_project, "Ctrl+N"))
        files.addAction(_action(self, "Open…", self._open_project, "Ctrl+O"))
        files.addAction(_action(self, "Save", lambda: self.ws.save(), "Ctrl+S"))
        files.addAction(_action(self, "Save as…", self._save_as))
        files.addSeparator()
        files.addAction(_action(self, "Open the project folder",
                                self._open_folder))
        files.addSeparator()
        files.addAction(_action(self, "Quit", self.close, "Ctrl+Q"))

        view = menu.addMenu("&View")
        view.addAction(_action(self, "Activity log", self._toggle_console,
                               "Ctrl+L"))
        view.addSeparator()
        for i, (title, _) in enumerate(PAGES):
            view.addAction(_action(self, title, lambda _=False, n=i:
                                   self.nav.setCurrentRow(n),
                                   f"Ctrl+{i + 1}" if i < 9 else ""))

        run = menu.addMenu("&Run")
        run.addAction(_action(self, "Compute the current site",
                              lambda: self._page_action("H/V Analysis",
                                                        "_compute"), "F5"))
        run.addAction(_action(self, "Cancel the running job",
                              self.ws.queue.cancel_current, "Esc"))

        help_menu = menu.addMenu("&Help")
        help_menu.addAction(_action(self, "Copy diagnostics to the clipboard",
                                    self._copy_diagnostics))
        help_menu.addAction(_action(self, "Save the activity log…",
                                    lambda: self.console.save_as()))
        help_menu.addSeparator()
        help_menu.addAction(_action(self, "About HVSRLab", self._about))
        help_menu.addAction(_action(self, "Method notes", self._method_notes))

    def _build_status(self) -> None:
        bar = self.statusBar()
        self.status_message = QLabel("")
        bar.addWidget(self.status_message, 1)

        self.status_badge = Badge("idle", "muted")
        bar.addPermanentWidget(self.status_badge)

        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 1000)
        self.status_progress.setFixedWidth(150)
        self.status_progress.setValue(0)
        bar.addPermanentWidget(self.status_progress)

        self.status_counts = QLabel("")
        bar.addPermanentWidget(self.status_counts)

        self.log_button = button("Activity", self._toggle_console, ghost=True,
                                 tooltip="Show or hide the activity log "
                                         "(Ctrl+L)")
        bar.addPermanentWidget(self.log_button)

    # -- navigation --------------------------------------------------------
    def _select(self, row: int) -> None:
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)
            page = self.stack.widget(row)
            if hasattr(page, "refresh"):
                page.refresh()
            self.nav.setCurrentRow(row)

    def go_to_page(self, title: str) -> None:
        for i, (name, _) in enumerate(PAGES):
            if name == title:
                self.nav.setCurrentRow(i)
                return

    def _page_action(self, title: str, method: str) -> None:
        page = self.pages.get(title)
        if page is None:
            return
        self.go_to_page(title)
        getattr(page, method, lambda: None)()

    # -- project -----------------------------------------------------------
    def _new_project(self) -> None:
        name, ok = QInputDialog.getText(self, "New project", "Project name:")
        if not ok or not name.strip():
            return
        root = paths.unique_project_dir(name.strip())
        project = Project(root, name.strip())
        project.ensure_tree()
        project.save()
        self.ws.set_project(project)
        self._notify(f"Created {project.name} in {root}", "good")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a project", str(paths.projects_dir()),
            "HVSRLab project (project.json);;All files (*)")
        if not path:
            return
        try:
            project = Project.load(path)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Could not open the project", str(exc))
            return
        self.ws.set_project(project)
        self._notify(f"Opened {project.name}", "good")
        self._rescan_prompt()

    def _rescan_prompt(self) -> None:
        """Offer to rebuild the MiniSEED catalogue, which is not saved.

        The catalogue is thousands of file paths and is cheap to rebuild, so it
        is left out of ``project.json`` rather than going stale in it.
        """
        project = self.ws.project
        if not project.raw_dir or not project.sites:
            return
        if not Path(project.raw_dir).exists():
            self._notify(f"Raw data folder {project.raw_dir} is not reachable — "
                         "results already computed will still open.", "warn")
            return
        answer = QMessageBox.question(
            self, "Re-catalogue the raw data?",
            f"{project.name} has {len(project.sites)} sites.\n\n"
            "Re-read the MiniSEED catalogue from\n"
            f"{project.raw_dir}?\n\n"
            "It is needed to compute or re-window sites; stored results open "
            "without it.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self.go_to_page("Sites")
            QTimer.singleShot(120, self.pages["Sites"]._scan)

    def _save_as(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the project", str(paths.projects_dir()))
        if not directory:
            return
        project = self.ws.project
        project.root = Path(directory)
        project.name = Path(directory).name
        project.ensure_tree()
        self.ws.save()
        self._retitle()

    def _open_folder(self) -> None:
        import os
        import subprocess

        root = self.ws.project.root
        root.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(root))                    # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(root)])  # noqa: S607

    # -- feedback ----------------------------------------------------------
    def _notify(self, message: str, tone: str = "info") -> None:
        colour = {"good": C["good"], "warn": C["warn"], "bad": C["bad"]}.get(
            tone, C["text_dim"])
        self.status_message.setText(message)
        self.status_message.setStyleSheet(f"color: {colour};")
        self.console.append(message,
                            {"bad": "error", "warn": "warn",
                             "good": "good"}.get(tone, "info"))
        if tone in ("warn", "bad"):
            self.console_dock.show()

    def _job_state(self, job) -> None:
        if job.state.value == "running":
            self.status_badge.set(job.name, "info")
        elif job.state.finished:
            tone = {"succeeded": "good", "failed": "bad"}.get(
                job.state.value, "warn")
            self.status_badge.set(f"{job.name}: {job.state.value}", tone)
            self.status_progress.setValue(
                1000 if job.state.value == "succeeded" else 0)
            if job.state.value == "failed":
                self.console_dock.show()

    def _job_progress(self, fraction: float, stage: str) -> None:
        self.status_progress.setValue(int(fraction * 1000))
        if stage:
            self.status_message.setText(stage)
            self.status_message.setStyleSheet(f"color: {C['text_dim']};")

    def _update_status(self) -> None:
        project = self.ws.project
        self.status_counts.setText(
            f"{self.ws.computed_count()}/{len(project.sites)} computed")
        self.project_label.setText(
            f"<b>{project.name}</b><br>{len(project.sites)} sites"
            + ("<br><span style='color:#fbbf24'>unsaved changes</span>"
               if self.ws.dirty else ""))

    def _retitle(self) -> None:
        project = self.ws.project
        self.setWindowTitle(f"HVSRLab — {project.name}")
        self._update_status()

    def _copy_diagnostics(self) -> None:
        self.console_dock.show()
        self.console.copy_diagnostics()

    def _toggle_console(self) -> None:
        self.console_dock.setVisible(not self.console_dock.isVisible())

    def _about(self) -> None:
        from .. import __version__

        QMessageBox.about(
            self, "About HVSRLab",
            f"<b>HVSRLab {__version__}</b><br><br>"
            "Horizontal-to-vertical spectral ratio processing for passive "
            "seismic surveys.<br><br>"
            "A Python re-implementation of Samuel Bignardi's "
            "OpenHVSR&nbsp;Processing&nbsp;Toolkit (GPL-3), with native "
            "MiniSEED ingestion, SESAME 2004 quality criteria, azimuthal and "
            "temporal stability analysis, and 1D forward modelling.")

    def _method_notes(self) -> None:
        QMessageBox.information(
            self, "Method notes",
            "<b>Processing chain</b><br>"
            "filter → window → STA/LTA → demean → taper → FFT → "
            "Konno-Ohmachi → combine horizontals → divide by vertical → "
            "statistics over windows.<br><br>"
            "<b>Two deliberate departures from OpenHVSR-ProTO</b><br>"
            "• The cosine taper is applied. ProTO calls "
            "<tt>cosine_taper</tt> without keeping its return value, so the "
            "taper never reaches the FFT.<br>"
            "• Smoothing is one pre-computed matrix product rather than a "
            "loop rebuilt per window.<br><br>"
            "<b>What H/V does and does not tell you</b><br>"
            "The peak frequency is robust and is what depth interpretation "
            "rests on. The amplitude is not a site amplification factor: it "
            "depends on the wavefield and on which of the three "
            "horizontal-combination rules produced it. The 1D model fits the "
            "frequency; its amplitude is indicative only.")

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event) -> None:                # noqa: N802 (Qt naming)
        if self.ws.dirty:
            answer = QMessageBox.question(
                self, "Unsaved changes",
                f"Save {self.ws.project.name} before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes)
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Yes:
                self.ws.save()
        self.ws.queue.shutdown()
        super().closeEvent(event)


def _action(parent, text: str, slot, shortcut: str = "") -> QAction:
    action = QAction(text, parent)
    action.triggered.connect(lambda _=False: slot())
    if shortcut:
        action.setShortcut(shortcut)
    return action
