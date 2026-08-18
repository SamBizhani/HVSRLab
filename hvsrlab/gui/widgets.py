"""Reusable interface pieces: cards, stat tiles, badges, the log console,
and the bridge that turns job callbacks into Qt signals.
"""

from __future__ import annotations

from typing import Callable, Iterable

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..jobs import Job, JobState
from .theme import C, mono_font


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------

class Card(QFrame):
    """A titled panel. The unit every page is assembled from."""

    def __init__(self, title: str = "", parent: QWidget | None = None,
                 dense: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        margin = 8 if dense else 12
        self._layout.setContentsMargins(margin, margin, margin, margin)
        self._layout.setSpacing(6 if dense else 8)
        self.title_label: QLabel | None = None
        if title:
            self.title_label = QLabel(title)
            self.title_label.setObjectName("CardTitle")
            self._layout.addWidget(self.title_label)

    def set_title(self, text: str) -> None:
        if self.title_label is not None:
            self.title_label.setText(text)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def add_stretch(self, stretch: int = 1) -> None:
        self._layout.addStretch(stretch)

    def body(self) -> QVBoxLayout:
        return self._layout


class PageHeader(QWidget):
    """Title, one line of explanation, and a slot for page-level actions."""

    def __init__(self, title: str, subtitle: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        text = QVBoxLayout()
        text.setSpacing(1)
        self.title = QLabel(title)
        self.title.setObjectName("PageTitle")
        text.addWidget(self.title)
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("PageSubtitle")
        self.subtitle.setWordWrap(True)
        text.addWidget(self.subtitle)
        row.addLayout(text, 1)

        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        row.addLayout(self.actions)

    def add_action(self, widget: QWidget) -> QWidget:
        self.actions.addWidget(widget)
        return widget

    def set_subtitle(self, text: str) -> None:
        self.subtitle.setText(text)


class StatTile(QFrame):
    """A single number with a label — the survey-level summary vocabulary."""

    def __init__(self, label: str, value: str = "—", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 8, 11, 8)
        layout.setSpacing(0)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("StatValue")
        layout.addWidget(self.value_label)

        self.label_label = QLabel(label.upper())
        self.label_label.setObjectName("StatLabel")
        layout.addWidget(self.label_label)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def set(self, value: str, colour: str | None = None) -> None:
        self.value_label.setText(str(value))
        self.value_label.setStyleSheet(f"color: {colour};" if colour else "")


class Badge(QLabel):
    """A small coloured status chip."""

    TONES = {"good": C["good"], "warn": C["warn"], "bad": C["bad"],
             "info": C["accent"], "muted": C["muted"]}

    def __init__(self, text: str = "", tone: str = "muted",
                 parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        colour = self.TONES.get(tone, C["muted"])
        self.setStyleSheet(
            f"color: {colour}; border: 1px solid {colour}; border-radius: 8px;"
            f"padding: 1px 8px; font-size: 8pt; font-weight: 600;")

    def set(self, text: str, tone: str = "muted") -> None:
        self.setText(text)
        self.set_tone(tone)


def scroll_column(widget: QWidget, width: int = 0) -> QWidget:
    """Put a control column inside a scroll area.

    A page whose controls are taller than the screen forces the whole window
    to a minimum height it cannot have, and Qt then clips it — silently
    hiding whatever is at the bottom. Scrolling the column instead keeps the
    window resizable down to a laptop screen and keeps every control
    reachable.
    """
    from PyQt5.QtWidgets import QScrollArea

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
    area.setWidget(widget)
    if width:
        area.setMaximumWidth(width)
        widget.setMaximumWidth(width - 12)      # leave room for the scrollbar
    return area


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {C['border_soft']}; background: {C['border_soft']};"
                       "max-height: 1px;")
    return line


def hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    return label


def button(text: str, slot: Callable | None = None, *, primary: bool = False,
           ghost: bool = False, tooltip: str = "") -> QPushButton:
    btn = QPushButton(text)
    if primary:
        btn.setObjectName("Primary")
    elif ghost:
        btn.setObjectName("Ghost")
    if slot is not None:
        btn.clicked.connect(slot)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


# ---------------------------------------------------------------------------
# Parameter editing
# ---------------------------------------------------------------------------

class ParamForm(QWidget):
    """A form of named controls bound to attributes of a dataclass.

    Values flow one way on demand — :meth:`load` fills the widgets, :meth:`apply`
    writes them back — so editing several fields does not trigger a recompute
    per keystroke.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(6)
        self.form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._fields: dict[str, tuple[QWidget, str]] = {}

    def _add(self, name: str, label: str, widget: QWidget, kind: str,
             tooltip: str) -> QWidget:
        if tooltip:
            widget.setToolTip(tooltip)
        caption = QLabel(label)
        if tooltip:
            caption.setToolTip(tooltip)
        self.form.addRow(caption, widget)
        self._fields[name] = (widget, kind)
        return widget

    def number(self, name: str, label: str, *, minimum: float = 0.0,
               maximum: float = 1e9, step: float = 1.0, decimals: int = 3,
               suffix: str = "", tooltip: str = "") -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(minimum, maximum)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        if suffix:
            w.setSuffix(f" {suffix}")
        w.valueChanged.connect(lambda _: self.changed.emit())
        return self._add(name, label, w, "float", tooltip)

    def integer(self, name: str, label: str, *, minimum: int = 0,
                maximum: int = 10 ** 9, step: int = 1, suffix: str = "",
                tooltip: str = "") -> QSpinBox:
        w = QSpinBox()
        w.setRange(minimum, maximum)
        w.setSingleStep(step)
        if suffix:
            w.setSuffix(f" {suffix}")
        w.valueChanged.connect(lambda _: self.changed.emit())
        return self._add(name, label, w, "int", tooltip)

    def choice(self, name: str, label: str, options: Iterable[tuple[str, str]],
               tooltip: str = "") -> QComboBox:
        w = QComboBox()
        for value, caption in options:
            w.addItem(caption, value)
        w.currentIndexChanged.connect(lambda _: self.changed.emit())
        return self._add(name, label, w, "choice", tooltip)

    def flag(self, name: str, label: str, tooltip: str = "") -> QCheckBox:
        w = QCheckBox()
        w.stateChanged.connect(lambda _: self.changed.emit())
        return self._add(name, label, w, "bool", tooltip)

    def widget(self, name: str) -> QWidget | None:
        entry = self._fields.get(name)
        return entry[0] if entry else None

    def load(self, obj) -> None:
        """Fill every control from the matching attribute of *obj*."""
        blocked = [(w, w.blockSignals(True)) for w, _ in self._fields.values()]
        try:
            for name, (w, kind) in self._fields.items():
                if not hasattr(obj, name):
                    continue
                value = getattr(obj, name)
                if kind == "float":
                    w.setValue(float(value))
                elif kind == "int":
                    w.setValue(int(value))
                elif kind == "bool":
                    w.setChecked(bool(value))
                elif kind == "choice":
                    index = w.findData(value)
                    if index < 0:
                        index = w.findText(str(value))
                    w.setCurrentIndex(max(0, index))
        finally:
            for w, previous in blocked:
                w.blockSignals(previous)

    def apply(self, obj) -> bool:
        """Write the controls back to *obj*; True if anything actually changed."""
        changed = False
        for name, (w, kind) in self._fields.items():
            if not hasattr(obj, name):
                continue
            if kind == "float":
                value = float(w.value())
            elif kind == "int":
                value = int(w.value())
            elif kind == "bool":
                value = bool(w.isChecked())
            elif kind == "choice":
                value = w.currentData()
            else:
                continue
            if getattr(obj, name) != value:
                setattr(obj, name, value)
                changed = True
        return changed


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobBridge(QObject):
    """Marshals job callbacks onto the Qt thread.

    Jobs run on a worker thread and must not touch widgets. Emitting signals
    from that thread is safe — Qt queues them across the connection — so every
    callback goes through here.
    """

    line = pyqtSignal(str)
    progress = pyqtSignal(float, str)
    state = pyqtSignal(object)

    def attach(self, job: Job) -> Job:
        job.on_line(self.line.emit)
        job.on_progress(lambda f, s: self.progress.emit(f, s))
        job.on_state(self.state.emit)
        return job


class LogConsole(QWidget):
    """Scrolling output with a progress bar and a stage caption."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.stage = QLabel("idle")
        self.stage.setObjectName("Hint")
        row.addWidget(self.stage, 1)
        self.elapsed = QLabel("")
        self.elapsed.setObjectName("Hint")
        row.addWidget(self.elapsed)
        layout.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setFont(mono_font(9))
        self.view.setMaximumBlockCount(4000)
        self.view.setStyleSheet(
            f"background: {C['bg']}; border: 1px solid {C['border_soft']};"
            f"border-radius: 6px; color: {C['text_dim']};")
        layout.addWidget(self.view, 1)

    def append(self, text: str) -> None:
        self.view.appendPlainText(text.rstrip())

    def set_progress(self, fraction: float, stage: str = "") -> None:
        self.bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        if stage:
            self.stage.setText(stage)

    def on_state(self, job: Job) -> None:
        if job.state == JobState.RUNNING:
            self.stage.setText(f"{job.name} — running")
        elif job.state.finished:
            self.bar.setValue(1000 if job.state == JobState.SUCCEEDED else 0)
            self.stage.setText(f"{job.name} — {job.state.value}")
            self.elapsed.setText(f"{job.elapsed:.1f} s")
            if job.error:
                self.append(f"! {job.error}")

    def clear(self) -> None:
        self.view.clear()
        self.bar.setValue(0)
        self.stage.setText("idle")
        self.elapsed.setText("")


def table_item(text: str, *, colour: str | None = None, align_right: bool = False,
               bold: bool = False, tooltip: str = ""):
    """A configured QTableWidgetItem — read-only unless the caller says otherwise."""
    from PyQt5.QtWidgets import QTableWidgetItem

    item = QTableWidgetItem(str(text))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    if colour:
        item.setForeground(QColor(colour))
    if align_right:
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if bold:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    if tooltip:
        item.setToolTip(tooltip)
    return item


def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setStyleSheet(
        f"color: {C['text_faint']}; font-size: 8pt; font-weight: 700;"
        "letter-spacing: 0.6px; padding-top: 4px;")
    return label


def elide(text: str, limit: int = 48) -> str:
    text = str(text)
    return text if len(text) <= limit else "…" + text[-(limit - 1):]


def monospace(text: str) -> QLabel:
    label = QLabel(text)
    label.setFont(mono_font(9))
    return label


def big_number_font(pt: int) -> QFont:
    font = QFont()
    font.setPointSize(pt)
    font.setBold(True)
    return font
