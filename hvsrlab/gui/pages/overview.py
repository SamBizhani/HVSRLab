"""Overview — where the survey stands, at a glance."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from ...core import bedrock
from ..plots import MapView, SummaryView
from ..widgets import Card, StatTile, button, hint
from .base import Page


class OverviewPage(Page):
    title = "Overview"
    subtitle = "Project status, survey layout and the distribution of results."

    def build(self) -> None:
        self.header.add_action(button("Save project", lambda: self.ws.save()))

        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {
            "sites": StatTile("sites"),
            "computed": StatTile("computed"),
            "good": StatTile("reliable & clear"),
            "f0": StatTile("median f₀ (Hz)"),
            "a0": StatTile("median A₀"),
            "depth": StatTile("median depth (m)"),
        }
        for tile in self.tiles.values():
            tiles.addWidget(tile)
        tiles.addStretch(1)
        self.add_layout(tiles)

        split = QSplitter(Qt.Horizontal)

        left = Card("Project", dense=True)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left.add(self.info)
        left.add(hint(
            "The usual order of work: catalogue the raw data on Sites, choose "
            "a window on Data & Windows, get one site right on H/V Analysis, "
            "then apply the same settings to the survey on Batch & Export."))
        left.add_stretch()
        left.setMaximumWidth(330)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        map_card = Card("Layout", dense=True)
        self.map = MapView(height=3.2, toolbar=False)
        map_card.add(self.map, 1)
        right_layout.addWidget(map_card, 3)

        summary_card = Card("Results", dense=True)
        self.summary = SummaryView(height=2.4, toolbar=False)
        summary_card.add(self.summary, 1)
        right_layout.addWidget(summary_card, 2)

        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.add(split, 1)

    def connect_signals(self) -> None:
        self.ws.projectChanged.connect(self.refresh)
        self.ws.sitesChanged.connect(self.refresh)

    def refresh(self) -> None:
        project = self.project
        sites = [s for s in project.sites if s.status != "excluded"]
        computed = [s for s in sites if np.isfinite(s.f0)]
        law = project.regression

        f0 = np.array([s.f0 for s in computed]) if computed else np.zeros(0)
        a0 = np.array([s.a0 for s in computed]) if computed else np.zeros(0)
        depth = bedrock.depth_from_f0(f0, law.a, law.b) if f0.size else f0

        self.tiles["sites"].set(str(len(project.sites)))
        self.tiles["computed"].set(str(len(computed)),
                                   "#34d399" if computed else None)
        good = sum(1 for s in computed if _is_good(s.sesame_score))
        self.tiles["good"].set(str(good), "#34d399" if good else "#fbbf24")
        self.tiles["f0"].set(_fmt(np.nanmedian(f0) if f0.size else np.nan, 3))
        self.tiles["a0"].set(_fmt(np.nanmedian(a0) if a0.size else np.nan, 2))
        self.tiles["depth"].set(
            _fmt(np.nanmedian(depth) if depth.size else np.nan, 1))

        self.info.setText(
            f"<b>{project.name}</b><br>"
            f"<span style='color:#8b98a8'>{project.root}</span><br><br>"
            f"Raw data: {project.raw_dir or '—'}<br>"
            f"Stations: {project.station_file or '—'}<br>"
            f"Catalogued: {len(self.ws.recordings)} recording(s)<br>"
            f"Bedrock law: H = {law.a:.4g}·f₀^{law.b:.3f}"
            + (f" (fitted, {law.n_points} wells)" if law.fitted else "")
            + f"<br>Display time zone: UTC{self.ws.utc_offset:+g}<br>"
            f"Modified: {project.modified}")

        x, y, labels = self.ws.coordinates()
        if x.size and np.isfinite(x).any():
            values = np.array([s.f0 for s in sites])
            self.map.plot(x, y, values if np.isfinite(values).any() else None,
                          labels=labels, title="f₀ (Hz)", unit="f₀ (Hz)",
                          show_labels=x.size <= 30,
                          profiles=project.profiles)
        else:
            self.map.message("No coordinates yet.")

        self.summary.plot([s.f0 for s in computed], [s.a0 for s in computed],
                          [_split(s.sesame_score) for s in computed])


def _split(score: str) -> tuple[bool, bool]:
    try:
        r, c = score.split("·")
        return (int(r.strip().split("/")[0]) == 3,
                int(c.strip().split("/")[0]) >= 5)
    except (ValueError, IndexError, AttributeError):
        return (False, False)


def _is_good(score: str) -> bool:
    reliable, clear = _split(score)
    return reliable and clear


def _fmt(value, digits: int) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"
