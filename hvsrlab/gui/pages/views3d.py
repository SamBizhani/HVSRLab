"""3D Views — surfaces over the survey, and H/V columns in place."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget, QSplitter)

from ...core import bedrock, grids
from ..plots import Volume3D
from ..widgets import Card, button, hint, scroll_column, section_label
from .base import Page

SURFACES = [
    ("depth", "Bedrock depth below surface"),
    ("bedrock_elev", "Bedrock elevation"),
    ("f0", "f₀"),
    ("a0", "A₀"),
    ("topography", "Topography"),
]


class Views3DPage(Page):
    title = "3D Views"
    subtitle = ("The bedrock surface implied by f₀, and each site's H/V curve "
                "standing at its own position.")

    def build(self) -> None:
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        card = Card("Scene")
        self.mode = QComboBox()
        self.mode.addItem("Interpolated surface", "surface")
        self.mode.addItem("H/V columns (tiles)", "tiles")
        self.mode.currentIndexChanged.connect(self._draw)
        card.add(self.mode)

        self.surface = QComboBox()
        for key, label in SURFACES:
            self.surface.addItem(label, key)
        self.surface.currentIndexChanged.connect(self._draw)
        card.add(self.surface)

        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "magma", "terrain", "cividis",
                            "coolwarm", "turbo"])
        self.cmap.currentIndexChanged.connect(self._draw)
        card.add(self.cmap)

        card.add(section_label("view"))
        self.elev = _slider(0, 89, 32, self._draw)
        card.add(_labelled("Elevation", self.elev))
        self.azim = _slider(-180, 180, -125, self._draw)
        card.add(_labelled("Rotation", self.azim))
        self.zscale = _slider(10, 200, 60, self._draw)
        card.add(_labelled("Vertical scale", self.zscale))

        card.add(section_label("mesh"))
        self.density = _slider(12, 90, 40, self._draw)
        card.add(_labelled("Grid density", self.density))
        card.add(button("Redraw", self._draw, primary=True))
        card.add(hint(
            "Tiles show one column per site with no interpolation between "
            "them: everything you see was measured. The surface is smoother "
            "and easier to read, but every point between sites is an "
            "assumption about how the basin behaves in between."))

        layout.addWidget(card)
        layout.addStretch(1)
        split.addWidget(scroll_column(left, 316))

        self.view = Volume3D(height=5.2)
        split.addWidget(self.view)
        split.setStretchFactor(1, 1)
        self.add(split, 1)

    def connect_signals(self) -> None:
        self.ws.sitesChanged.connect(self._draw_if_visible)
        self.ws.projectChanged.connect(self._draw_if_visible)

    def refresh(self) -> None:
        self._draw()

    def _draw_if_visible(self) -> None:
        if self.isVisible():
            self._draw()

    # -- drawing -----------------------------------------------------------
    def _sites(self):
        return [s for s in self.project.sites
                if s.status != "excluded" and np.isfinite(s.x)
                and np.isfinite(s.f0)]

    def _draw(self) -> None:
        sites = self._sites()
        if len(sites) < 3:
            self.view.message(
                "At least three computed sites with coordinates are needed.")
            return

        if self.mode.currentData() == "tiles":
            self._draw_tiles(sites)
        else:
            self._draw_surface(sites)

    def _draw_surface(self, sites) -> None:
        law = self.project.regression
        x = np.array([s.x for s in sites])
        y = np.array([s.y for s in sites])
        key = self.surface.currentData()

        if key == "depth":
            values = bedrock.depth_from_f0(np.array([s.f0 for s in sites]),
                                           law.a, law.b)
            unit, title = "m", "depth to bedrock"
        elif key == "bedrock_elev":
            depth = bedrock.depth_from_f0(np.array([s.f0 for s in sites]),
                                          law.a, law.b)
            values = np.array([s.z for s in sites]) - depth
            unit, title = "m", "bedrock elevation"
        elif key == "f0":
            values = np.array([s.f0 for s in sites])
            unit, title = "Hz", "f₀"
        elif key == "a0":
            values = np.array([s.a0 for s in sites])
            unit, title = "", "A₀"
        else:
            values = np.array([s.z for s in sites])
            unit, title = "m", "topography"

        n = int(self.density.value())
        try:
            grid = grids.interpolate(x, y, values, nx=n, ny=n, method="linear",
                                     mask="hull")
        except Exception as exc:                       # noqa: BLE001
            self.view.message(f"Could not interpolate: {exc}")
            return

        self.view.plot_surface(
            grid, x=x, y=y, z=values, cmap=self.cmap.currentText(),
            title=title, unit=unit, elev=float(self.elev.value()),
            azim=float(self.azim.value()),
            zscale=self.zscale.value() / 100.0)

    def _draw_tiles(self, sites) -> None:
        law = self.project.regression
        columns = []
        for site in sites:
            result = self.ws.result(site.sid)
            if result is None or result.freq.size == 0:
                continue
            depth = grids.frequency_to_depth(result.freq, law.a, law.b)
            columns.append({"x": site.x, "y": site.y, "z": site.z,
                            "depth": depth, "values": result.hv})
        if not columns:
            self.view.message(
                "No stored H/V results to draw.\n"
                "Compute the sites, or run a batch on the Batch & Export page.")
            return
        self.view.plot_tiles(columns, cmap=self.cmap.currentText(),
                             title=f"H/V columns — {len(columns)} sites",
                             elev=float(self.elev.value()),
                             azim=float(self.azim.value()))


def _slider(minimum: int, maximum: int, value: int, slot) -> QSlider:
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    slider.sliderReleased.connect(slot)
    return slider


def _labelled(text: str, widget: QWidget) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    caption = QLabel(text)
    caption.setMinimumWidth(96)
    row.addWidget(caption)
    row.addWidget(widget, 1)
    return holder
