"""The Activity panel: everything the application has to say, in one place.

Three things feed it — job output, the messages pages raise, and anything that
goes wrong anywhere in the process. That last one matters most. Without it, an
exception inside a Qt slot goes to a stderr nobody is reading and the interface
just appears to do nothing, which is the least debuggable failure there is.

:func:`install_handlers` redirects the standard library's ``logging``, Python
warnings, unhandled exceptions and Qt's own message stream into the panel, so
"nothing happened" always leaves a trace.

Everything is also mirrored to ``<project>/logs/session.log``, and the
**Copy diagnostics** button puts the environment report and the recent log on
the clipboard as one block, ready to paste into a bug report.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from html import escape
from pathlib import Path
import logging
import sys
import traceback
import warnings

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit,
    QVBoxLayout, QWidget,
)

from .. import diagnostics
from .theme import C, mono_font
from .widgets import button

#: Ordered by severity; the filter shows a level and everything above it.
LEVELS = ("debug", "info", "good", "warn", "error")

LEVEL_COLOURS = {
    "debug": C["text_faint"],
    "info": C["text_dim"],
    "good": C["good"],
    "warn": C["warn"],
    "error": C["bad"],
}

#: Lines kept in memory. The file on disk keeps everything.
CAPACITY = 5000

#: Qt complains about things that are true but not actionable — a missing font
#: directory in the Anaconda tree, a platform plugin that will not resize a
#: dock. Left at warning level they bury the messages that matter, so they are
#: demoted rather than dropped: still in the log, not shouting.
_QT_NOISE = (
    "QFontDatabase: Cannot find font directory",
    "propagateSizeHints",
    "QWindowsWindow::setGeometry",
    "Unable to set geometry",
)


class ActivityLog(QWidget):
    """A filterable, copyable transcript of the session."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: deque = deque(maxlen=CAPACITY)
        self._minimum = 0                       # index into LEVELS
        self._file: Path | None = None
        self._counts = {level: 0 for level in LEVELS}
        self._last_line = ""
        self._last_level = "info"
        self._repeats = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)

        self.filter = QComboBox()
        self.filter.addItem("Everything", 0)
        self.filter.addItem("Info and above", 1)
        self.filter.addItem("Warnings and errors", 3)
        self.filter.addItem("Errors only", 4)
        self.filter.setCurrentIndex(1)
        self.filter.currentIndexChanged.connect(self._filter_changed)
        row.addWidget(QLabel("Show"))
        row.addWidget(self.filter)

        self.summary = QLabel("")
        self.summary.setObjectName("Hint")
        row.addWidget(self.summary, 1)

        row.addWidget(button("Copy diagnostics", self.copy_diagnostics,
                             primary=True,
                             tooltip="Environment report plus this log, on the "
                                     "clipboard — paste it straight into a "
                                     "bug report."))
        row.addWidget(button("Copy log", self.copy_log))
        row.addWidget(button("Save…", self.save_as))
        row.addWidget(button("Clear", self.clear, ghost=True))
        layout.addLayout(row)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setFont(mono_font(9))
        self.view.setMaximumBlockCount(CAPACITY)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.view.setStyleSheet(
            f"background: {C['bg']}; border: 1px solid {C['border_soft']};"
            f"border-radius: 6px; color: {C['text_dim']};")
        layout.addWidget(self.view, 1)

        self._provider = None            # callable returning the env report

    # -- input -------------------------------------------------------------
    def append(self, text: str, level: str = "info") -> None:
        if level not in LEVEL_COLOURS:
            level = "info"
        stamp = datetime.now().strftime("%H:%M:%S")
        for line in str(text).rstrip().splitlines() or [""]:
            # A message repeated fifty times is one fact, not fifty. Collapse
            # consecutive duplicates the way a syslog does.
            if line == self._last_line and level == self._last_level:
                self._repeats += 1
                continue
            self._flush_repeats()
            self._last_line, self._last_level = line, level

            self._records.append((stamp, level, line))
            self._counts[level] = self._counts.get(level, 0) + 1
            if LEVELS.index(level) >= self._minimum:
                self._emit(stamp, level, line)
            self._write_file(stamp, level, line)
        self._update_summary()

    def _flush_repeats(self) -> None:
        if self._repeats <= 0:
            return
        count, self._repeats = self._repeats, 0
        stamp = datetime.now().strftime("%H:%M:%S")
        note = f"    … previous line repeated {count} more time(s)"
        self._records.append((stamp, self._last_level, note))
        if LEVELS.index(self._last_level) >= self._minimum:
            self._emit(stamp, self._last_level, note)
        self._write_file(stamp, self._last_level, note)

    def _emit(self, stamp: str, level: str, line: str) -> None:
        colour = LEVEL_COLOURS[level]
        tag = level.upper()[:5].ljust(5)
        self.view.appendHtml(
            f"<span style='color:{C['text_faint']}'>{stamp}</span> "
            f"<span style='color:{colour}'>{tag}</span> "
            f"<span style='color:{colour if level in ('warn', 'error') else C['text_dim']}'>"
            f"{escape(line)}</span>")

    def _update_summary(self) -> None:
        errors = self._counts.get("error", 0)
        warnings_ = self._counts.get("warn", 0)
        parts = [f"{len(self._records)} lines"]
        if warnings_:
            parts.append(f"{warnings_} warnings")
        if errors:
            parts.append(f"{errors} errors")
        self.summary.setText(" · ".join(parts)
                             + (f"  →  {self._file}" if self._file else ""))

    # -- file mirror -------------------------------------------------------
    def set_log_file(self, path: Path | None) -> None:
        """Mirror everything to *path* from here on."""
        self._file = Path(path) if path else None
        if self._file is not None:
            try:
                self._file.parent.mkdir(parents=True, exist_ok=True)
                with open(self._file, "a", encoding="utf-8") as fh:
                    fh.write(f"\n{'=' * 72}\n"
                             f"{datetime.now():%Y-%m-%d %H:%M:%S} session start\n")
            except OSError:
                self._file = None
        self._update_summary()

    def _write_file(self, stamp: str, level: str, text: str) -> None:
        if self._file is None:
            return
        try:
            with open(self._file, "a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {level.upper():<5} {text}\n")
        except OSError:
            self._file = None            # a full or read-only disk is not fatal

    # -- filter ------------------------------------------------------------
    def _filter_changed(self) -> None:
        self._minimum = int(self.filter.currentData())
        self.view.clear()
        for stamp, level, line in self._records:
            if LEVELS.index(level) >= self._minimum:
                self._emit(stamp, level, line)

    # -- output ------------------------------------------------------------
    def set_environment_provider(self, provider) -> None:
        self._provider = provider

    def log_text(self) -> str:
        return "\n".join(f"{s} {lv.upper():<5} {ln}"
                         for s, lv, ln in self._records)

    def diagnostics_text(self) -> str:
        header = (self._provider() if self._provider
                  else diagnostics.environment_report())
        return (f"{header}\n\n{'-' * 72}\nActivity log "
                f"({len(self._records)} lines)\n{'-' * 72}\n{self.log_text()}")

    def copy_log(self) -> None:
        QApplication.clipboard().setText(self.log_text())
        self.append("Log copied to the clipboard.", "good")

    def copy_diagnostics(self) -> None:
        QApplication.clipboard().setText(self.diagnostics_text())
        self.append("Diagnostics copied to the clipboard — paste them into "
                    "your bug report.", "good")

    def save_as(self) -> None:
        default = str(Path.home() / f"hvsrlab_{datetime.now():%Y%m%d_%H%M}.log")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the activity log", default, "Log files (*.log *.txt)")
        if not path:
            return
        try:
            Path(path).write_text(self.diagnostics_text(), encoding="utf-8")
        except OSError as exc:
            self.append(f"Could not write {path}: {exc}", "error")
            return
        self.append(f"Saved to {path}", "good")

    def clear(self) -> None:
        self._records.clear()
        self._counts = {level: 0 for level in LEVELS}
        self._last_line, self._repeats = "", 0
        self.view.clear()
        self._update_summary()


class _Bridge(QObject):
    """Carries lines from any thread onto the Qt thread."""

    line = pyqtSignal(str, str)


class _LoggingHandler(logging.Handler):
    """Feeds the standard library's logging into the panel."""

    _MAP = {logging.DEBUG: "debug", logging.INFO: "debug",
            logging.WARNING: "warn", logging.ERROR: "error",
            logging.CRITICAL: "error"}

    def __init__(self, emit_line) -> None:
        super().__init__()
        self._emit_line = emit_line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = self._MAP.get(record.levelno, "info")
            self._emit_line(f"[{record.name}] {record.getMessage()}", level)
        except Exception:                              # noqa: BLE001
            pass                                        # logging must never raise


def install_handlers(log: ActivityLog) -> _Bridge:
    """Route logging, warnings, exceptions and Qt messages into *log*.

    Returns the bridge that owns the cross-thread signal; keep a reference to
    it or the connection is garbage-collected.
    """
    bridge = _Bridge()
    bridge.line.connect(log.append)

    def emit_line(text: str, level: str = "info") -> None:
        bridge.line.emit(text, level)

    handler = _LoggingHandler(emit_line)
    handler.setLevel(logging.WARNING)         # INFO from obspy is very chatty
    logging.getLogger().addHandler(handler)

    previous_showwarning = warnings.showwarning

    def show_warning(message, category, filename, lineno, file=None, line=None):
        emit_line(f"{category.__name__}: {message}  "
                  f"({Path(filename).name}:{lineno})", "warn")
        return previous_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = show_warning

    previous_hook = sys.excepthook

    def excepthook(kind, value, tb) -> None:
        emit_line("Unhandled error — please copy the diagnostics and report "
                  "this:", "error")
        emit_line("".join(traceback.format_exception(kind, value, tb)), "error")
        previous_hook(kind, value, tb)

    sys.excepthook = excepthook

    try:
        from PyQt5.QtCore import (
            QtCriticalMsg, QtFatalMsg, QtWarningMsg, qInstallMessageHandler)

        def qt_message(mode, context, message) -> None:
            level = "error" if mode in (QtCriticalMsg, QtFatalMsg) else (
                "warn" if mode == QtWarningMsg else "debug")
            if any(noise in str(message) for noise in _QT_NOISE):
                level = "debug"
            emit_line(f"[Qt] {message}", level)

        qInstallMessageHandler(qt_message)
    except ImportError:
        pass

    return bridge
