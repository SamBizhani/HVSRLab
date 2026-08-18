"""Common scaffolding for pages."""

from __future__ import annotations

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ..state import Workspace
from ..widgets import PageHeader


class Page(QWidget):
    """A navigable page.

    Subclasses build their content in :meth:`build` and refresh it in
    :meth:`refresh`. Refresh is called whenever the page becomes visible and
    whenever the workspace says something it depends on has changed, so it must
    be cheap and must tolerate being called before anything has been computed.
    """

    title = "Page"
    subtitle = ""

    def __init__(self, workspace: Workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ws = workspace
        self._built = False

        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(14, 12, 14, 12)
        self.layout_.setSpacing(10)

        self.header = PageHeader(self.title, self.subtitle)
        self.layout_.addWidget(self.header)

        self.build()
        self._built = True
        self.connect_signals()

    # -- to override -------------------------------------------------------
    def build(self) -> None:
        """Create the page's widgets."""

    def connect_signals(self) -> None:
        """Subscribe to the workspace signals this page cares about."""

    def refresh(self) -> None:
        """Redraw from current state."""

    # -- helpers -----------------------------------------------------------
    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.layout_.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self.layout_.addLayout(layout)

    def showEvent(self, event) -> None:            # noqa: N802 (Qt naming)
        super().showEvent(event)
        if self._built:
            self.refresh()

    @property
    def project(self):
        return self.ws.project

    @property
    def site(self):
        return self.ws.site

    def busy(self) -> bool:
        return self.ws.queue.busy

    def warn(self, text: str) -> None:
        self.ws.notify.emit(text, "warn")

    def fail(self, text: str) -> None:
        self.ws.notify.emit(text, "bad")

    def ok(self, text: str) -> None:
        self.ws.notify.emit(text, "good")
