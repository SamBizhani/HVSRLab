"""Bedrock — turning f₀ into depth, and calibrating the law that does it."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from ...core import bedrock, grids
from ...io import wells as well_io
from ...project import Well

from ..plots import MapView, RegressionView
from ..widgets import (
    Card, ParamForm, StatTile, button, hint, scroll_column, section_label,
    table_item)
from .base import Page

WELL_COLUMNS = ["Borehole", "Latitude", "Longitude", "Depth (m)",
                "Linked site", "Distance (m)", "site f₀ (Hz)"]

#: Beyond this a borehole-to-station link is worth a second look. Boreholes are
#: drilled near stations, not on them, so a few hundred metres is normal; a
#: kilometre means the calibration is assuming the basin does not change over
#: that kilometre, which it may well do.
FAR_LINK_M = 500.0


class BedrockPage(Page):
    title = "Bedrock"
    subtitle = ("Convert every site's f₀ into a depth to bedrock, so the "
                "survey becomes a basin map instead of a frequency map.")

    def build(self) -> None:
        purpose = Card("What this page is for")
        purpose.add(hint(
            "H/V measures a <b>frequency</b>. What you want from a survey is "
            "<b>depth</b>. A soft layer over stiff basement resonates at "
            "f₀ ≈ Vs / 4H, so a lower f₀ means a deeper basin — but turning "
            "that proportionality into metres needs the velocity, which H/V "
            "does not measure.<br><br>"
            "Ibs-von Seht &amp; Wohlenberg (1999) got around that empirically: "
            "over a basin where Vs increases with depth in a consistent way, "
            "<b>H = a·f₀^b</b> fits borehole depths well. That is what this "
            "page sets. Once a and b are chosen, every f₀ on the survey becomes "
            "a depth, and the Maps, 3D and Section pages all switch from "
            "frequency to metres.<br><br>"
            "<b>The honest version:</b> a and b encode one basin's velocity "
            "structure. Using another basin's published pair gives you the "
            "right <i>pattern</i> — where it is deep and where it is shallow — "
            "with an uncertain <i>scale</i>. Three or more local boreholes fix "
            "the scale. With none, quote the pattern and say which law you "
            "used. The 1D Model page is the alternative when you have an "
            "independent Vs, from tomography or a borehole log: it gets depth "
            "from physics instead of from a regression."))
        self.add(purpose)

        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        law_card = Card("Active law")
        self.law_combo = QComboBox()
        self.law_combo.addItem("— custom —", "")
        for law in bedrock.LAWS:
            self.law_combo.addItem(law.name, law.name)
        self.law_combo.currentIndexChanged.connect(self._law_chosen)
        law_card.add(self.law_combo)

        self.form = ParamForm()
        self.form.number("a", "a", minimum=0.001, maximum=100000.0, step=1.0,
                         decimals=4)
        self.form.number("b", "b", minimum=-5.0, maximum=5.0, step=0.01,
                         decimals=4,
                         tooltip="Physically this must be negative: deeper "
                                 "basin, lower resonance. A positive value "
                                 "means the fit found the opposite.")
        law_card.add(self.form)
        row = QHBoxLayout()
        row.addWidget(button("Apply", self._apply_law, primary=True))
        row.addWidget(button("Reset to Ibs-von Seht", self._reset_law))
        law_card.add_layout(row)

        self.law_note = QLabel("")
        self.law_note.setObjectName("Hint")
        self.law_note.setWordWrap(True)
        law_card.add(self.law_note)
        law_card.add(hint(
            "A power law fitted in one basin carries that basin's velocity "
            "structure. Published coefficients are starting points; where you "
            "have three or more boreholes, fit your own."))

        wells_card = Card("Borehole control")
        row = QHBoxLayout()
        row.addWidget(button("Add", self._add_well, primary=True,
                             tooltip="Type a borehole in by hand"))
        row.addWidget(button("Load table…", self._load_table))
        row.addWidget(button("Load log…", self._load_log))
        row.addWidget(button("Remove", self._remove_well))
        row.addWidget(button("Clear", self._clear_wells, ghost=True))
        wells_card.add_layout(row)

        self.well_table = QTableWidget(0, len(WELL_COLUMNS))
        self.well_table.setHorizontalHeaderLabels(WELL_COLUMNS)
        self.well_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.well_table.verticalHeader().setVisible(False)
        self.well_table.setAlternatingRowColors(True)
        head = self.well_table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(WELL_COLUMNS)):
            head.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.well_table.itemChanged.connect(self._well_edited)
        wells_card.add(self.well_table, 1)
        wells_card.add(hint(
            "Name, latitude, longitude and depth are editable — type them in "
            "directly, or load a table. Depth is to the top of bedrock, in "
            "metres below the surface.\n\n"
            "Each hole links itself to the nearest site and the separation is "
            "shown; drilling is rarely co-located with a station. A distance "
            "in amber is over 500 m, where the assumption that the basin is "
            "the same at both starts to matter. Pick a site from the dropdown "
            "to pin a link by hand."))

        wells_card.add(section_label("fit"))
        self.fit_form = ParamForm()
        self.fit_form.choice("method", "Method",
                             [("log-linear", "Log-linear (recommended)"),
                              ("nonlinear", "Non-linear on depth (ProTO)")])
        wells_card.add(self.fit_form)
        wells_card.add(button("Fit a and b to these wells", self._run_fit,
                              primary=True))

        tiles = QHBoxLayout()
        tiles.setSpacing(6)
        self.tile_n = StatTile("control points")
        self.tile_rms = StatTile("RMS (m)")
        self.tile_r2 = StatTile("R² (log)")
        for tile in (self.tile_n, self.tile_rms, self.tile_r2):
            tiles.addWidget(tile)
        wells_card.add_layout(tiles)
        self.fit_note = QLabel("")
        self.fit_note.setObjectName("Hint")
        self.fit_note.setWordWrap(True)
        wells_card.add(self.fit_note)

        layout.addWidget(law_card)
        layout.addWidget(wells_card, 1)
        split.addWidget(scroll_column(left, 380))

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.tabs = QTabWidget()

        calibration = QWidget()
        cal_layout = QVBoxLayout(calibration)
        cal_layout.setContentsMargins(4, 4, 4, 4)
        self.plot = RegressionView(height=4.4)
        cal_layout.addWidget(self.plot, 1)
        cal_layout.addWidget(hint(
            "Each published law is a grey line; the active one is blue. Green "
            "dots are your boreholes — where they sit relative to a line is "
            "how wrong that law is for this basin."))
        self.tabs.addTab(calibration, "Calibration")

        map_page = QWidget()
        map_layout = QVBoxLayout(map_page)
        map_layout.setContentsMargins(4, 4, 4, 4)
        map_layout.setSpacing(6)

        controls = QHBoxLayout()
        self.map_cmap = QComboBox()
        self.map_cmap.addItems(["terrain_r", "viridis_r", "magma_r",
                                "cividis_r", "Blues", "turbo"])
        self.map_cmap.currentIndexChanged.connect(self._draw_map)
        controls.addWidget(QLabel("Colour"))
        controls.addWidget(self.map_cmap)
        self.map_note = QLabel("")
        self.map_note.setObjectName("Hint")
        controls.addWidget(self.map_note, 1)
        map_layout.addLayout(controls)

        self.depth_map = MapView(height=4.2)
        map_layout.addWidget(self.depth_map, 1)
        self.residual_note = QLabel("")
        self.residual_note.setObjectName("Hint")
        self.residual_note.setWordWrap(True)
        map_layout.addWidget(self.residual_note)
        self.tabs.addTab(map_page, "Depth map")

        right_layout.addWidget(self.tabs, 1)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.tile_min = StatTile("shallowest (m)")
        self.tile_median = StatTile("median (m)")
        self.tile_max = StatTile("deepest (m)")
        for tile in (self.tile_min, self.tile_median, self.tile_max):
            stats.addWidget(tile)
        stats.addStretch(1)
        right_layout.addLayout(stats)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.add(split, 1)

        self._fit_result = None
        self._filling = False

    # -- wiring ------------------------------------------------------------
    def connect_signals(self) -> None:
        self.ws.projectChanged.connect(self.refresh)
        self.ws.sitesChanged.connect(self.refresh)

    def refresh(self) -> None:
        law = self.project.regression
        self.form.load(law)
        index = self.law_combo.findData(law.name if not law.fitted else "")
        self.law_combo.blockSignals(True)
        self.law_combo.setCurrentIndex(max(0, index))
        self.law_combo.blockSignals(False)
        self.law_note.setText(
            f"H = {law.a:.4g} · f₀^{law.b:.4f}"
            + (f"  — fitted to {law.n_points} wells, RMS {law.rms:.1f} m"
               if law.fitted else ""))
        self._fill_wells()
        self._draw()
        self._draw_map()

    # -- law ---------------------------------------------------------------
    def _law_chosen(self) -> None:
        name = self.law_combo.currentData()
        law = bedrock.get_law(name) if name else None
        if law is None:
            return
        self.project.regression.name = law.name
        self.project.regression.a = law.a
        self.project.regression.b = law.b
        self.project.regression.fitted = False
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self.refresh()

    def _apply_law(self) -> None:
        self.form.apply(self.project.regression)
        self.project.regression.fitted = False
        self.project.regression.name = "custom"
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self.refresh()

    def _reset_law(self) -> None:
        law = bedrock.LAWS[0]
        self.project.regression.name = law.name
        self.project.regression.a, self.project.regression.b = law.a, law.b
        self.project.regression.fitted = False
        self.ws.touch()
        self.refresh()

    # -- wells -------------------------------------------------------------
    def _load_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Well table", self.project.raw_dir or "",
            "Text files (*.txt *.csv *.dat);;All files (*)")
        if not path:
            return
        try:
            wells = well_io.read_table(path)
        except Exception as exc:                       # noqa: BLE001
            self.fail(f"Could not read the table: {exc}")
            return
        self.project.wells.extend(wells)
        self._locate_wells()
        self.ws.touch()
        self.refresh()
        self.ok(f"Loaded {len(wells)} well(s).")

    def _load_log(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Well logs", self.project.raw_dir or "",
            "Text files (*.txt *.dat);;All files (*)")
        for path in paths:
            try:
                self.project.wells.append(well_io.read_log(path))
            except Exception as exc:                   # noqa: BLE001
                self.fail(f"{path}: {exc}")
        self._locate_wells()
        self.ws.touch()
        self.refresh()

    def _locate_wells(self) -> None:
        """Give wells local plan coordinates on the project's origin."""
        from ...io import stations as station_io

        sites = self.project.sites
        lats = np.array([s.lat for s in sites], dtype=float)
        lons = np.array([s.lon for s in sites], dtype=float)
        finite = np.isfinite(lats) & np.isfinite(lons)
        if not finite.any():
            return
        origin = (float(np.mean(lats[finite])), float(np.mean(lons[finite])))
        for well in self.project.wells:
            if np.isfinite(well.lat) and np.isfinite(well.lon):
                x, y, _ = station_io.to_local_xy([well.lat], [well.lon], origin)
                well.x, well.y = float(x[0]), float(y[0])

    def _add_well(self) -> None:
        """Start a blank row, seeded at the current site.

        The seed is only a starting point: type the hole's real coordinates in
        and it re-links itself to whichever station turns out to be nearest.
        """
        well = Well(name=f"BH{len(self.project.wells) + 1}")
        site = self.ws.site
        if site is not None and np.isfinite(site.lat):
            well.lat, well.lon = site.lat, site.lon
            well.x, well.y = site.x, site.y
        self.project.wells.append(well)
        self.ws.locate_wells()               # places it and links it
        self.ws.touch()
        self._fill_wells()
        row = len(self.project.wells) - 1
        self.well_table.setCurrentCell(row, 0)
        self.well_table.editItem(self.well_table.item(row, 3))
        self.ws.notify.emit(
            "Added a borehole. Enter its depth to bedrock and its own "
            "coordinates — it links itself to the nearest site as you type.",
            "info")

    def _well_edited(self, item) -> None:
        """Write one edited cell back to its well, rejecting nonsense."""
        if self._filling or item is None:
            return
        row, column = item.row(), item.column()
        if not (0 <= row < len(self.project.wells)):
            return
        well = self.project.wells[row]
        text = item.text().strip()

        if column == 0:
            well.name = text or well.name
        else:
            try:
                value = float(text) if text and text != "—" else float("nan")
            except ValueError:
                self.warn(f"{text!r} is not a number.")
                self._fill_wells()
                return
            if column == 1:
                if np.isfinite(value) and abs(value) > 90:
                    self.warn("Latitude must be between −90 and 90.")
                    self._fill_wells()
                    return
                well.lat = value
            elif column == 2:
                if np.isfinite(value) and abs(value) > 180:
                    self.warn("Longitude must be between −180 and 180.")
                    self._fill_wells()
                    return
                well.lon = value
            elif column == 3:
                if np.isfinite(value) and value <= 0:
                    self.warn("Depth to bedrock must be positive.")
                    self._fill_wells()
                    return
                well.bedrock_depth = value

        # Moving a hole may put it nearer a different station, so the link
        # is re-derived rather than left pointing at the old one.
        self.ws.locate_wells()
        self.ws.touch()
        self._fill_wells()
        self._draw()

    def _link_changed(self, row: int, sid: str) -> None:
        """Pin a borehole to a chosen site, or hand it back to the auto-link."""
        if self._filling or not (0 <= row < len(self.project.wells)):
            return
        well = self.project.wells[row]
        if not sid:
            well.link_mode = "auto"
            self.ws.link_wells()
        else:
            well.link_mode = "user"
            well.site = sid
            site = self.project.site(sid)
            if site is not None:
                # A hole with no coordinates of its own inherits the site's,
                # which is what "this borehole is at that station" means.
                if not np.isfinite(well.lat):
                    well.lat, well.lon = site.lat, site.lon
                    well.x, well.y = site.x, site.y
                well.link_distance = (
                    float(np.hypot(site.x - well.x, site.y - well.y))
                    if np.isfinite(site.x) and np.isfinite(well.x)
                    else float("nan"))
        self.ws.touch()
        self._fill_wells()
        self._draw()

    def _clear_wells(self) -> None:
        if not self.project.wells:
            return
        from PyQt5.QtWidgets import QMessageBox

        if QMessageBox.question(
                self, "Clear boreholes",
                f"Remove all {len(self.project.wells)} borehole(s)?\n"
                "The fitted regression is kept until you change it.",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel) != QMessageBox.Yes:
            return
        self.project.wells = []
        self.ws.touch()
        self.refresh()

    def _remove_well(self) -> None:
        rows = sorted({i.row() for i in self.well_table.selectedIndexes()},
                      reverse=True)
        for row in rows:
            if 0 <= row < len(self.project.wells):
                del self.project.wells[row]
        self.ws.touch()
        self.refresh()

    def _fill_wells(self) -> None:
        wells = self.project.wells
        self._filling = True
        try:
            self.well_table.setRowCount(len(wells))
            for row, well in enumerate(wells):
                site = self.project.site(well.site) if well.site else None
                f0 = site.f0 if site is not None else float("nan")

                for column, value in ((0, well.name),
                                      (1, _fmt(well.lat, 6)),
                                      (2, _fmt(well.lon, 6)),
                                      (3, _fmt(well.bedrock_depth, 1))):
                    cell = QTableWidgetItem(str(value))
                    cell.setFlags(cell.flags() | Qt.ItemIsEditable)
                    if column:
                        cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.well_table.setItem(row, column, cell)

                combo = QComboBox()
                auto = f"nearest: {site.label()}" if (
                    well.link_mode == "auto" and site) else "nearest site (auto)"
                combo.addItem(auto, "")
                for candidate in self.project.sites:
                    if candidate.status != "excluded":
                        combo.addItem(candidate.label(), candidate.sid)
                combo.setCurrentIndex(
                    max(0, combo.findData(well.site))
                    if well.link_mode == "user" else 0)
                combo.currentIndexChanged.connect(
                    lambda _, r=row, c=combo:
                    self._link_changed(r, c.currentData()))
                self.well_table.setCellWidget(row, 4, combo)

                distance = well.link_distance
                far = np.isfinite(distance) and distance > FAR_LINK_M
                self.well_table.setItem(row, 5, table_item(
                    f"{distance:,.0f}" if np.isfinite(distance) else "—",
                    align_right=True,
                    colour="#fbbf24" if far else None,
                    tooltip=("This borehole is %.0f m from the station whose f₀ "
                             "it calibrates. The fit assumes the basin is the "
                             "same at both." % distance
                             if np.isfinite(distance) else
                             "No coordinates, so the separation is unknown")))

                usable = np.isfinite(f0) and np.isfinite(well.bedrock_depth)
                self.well_table.setItem(row, 6, table_item(
                    _fmt(f0, 3), align_right=True,
                    colour="#34d399" if usable else "#64748b",
                    tooltip="usable in the fit" if usable
                    else "needs both a depth and a computed f₀"))
        finally:
            self._filling = False

    # -- fitting -----------------------------------------------------------
    def _run_fit(self) -> None:
        settings = _FitSettings()
        self.fit_form.apply(settings)

        sites = [s for s in self.project.sites if s.status != "excluded"]
        f0, depth, notes = bedrock.pairs_from_wells(
            sites, self.project.wells, max_distance=settings.max_distance)
        if f0.size < 3:
            self.fail(f"Only {f0.size} usable pair(s); three are needed. "
                      + ("  ".join(notes[:3]) if notes else ""))
            self.fit_note.setText(
                "  ".join(notes) if notes else
                "Give each borehole a depth, and make sure the site it sits on "
                "has been computed.")
            return

        try:
            fit = bedrock.fit_regression(f0, depth, method=settings.method)
        except Exception as exc:                       # noqa: BLE001
            self.fail(str(exc))
            return

        linked = [w for w in self.project.wells
                  if w.site and np.isfinite(w.bedrock_depth)]
        separations = np.array([w.link_distance for w in linked
                                if np.isfinite(w.link_distance)])
        if separations.size and separations.max() > FAR_LINK_M:
            self.ws.log(
                f"Fit used boreholes up to {separations.max():,.0f} m from "
                f"their stations (median {np.median(separations):,.0f} m).",
                "warn")

        self._fit_result = fit
        self.tile_n.set(str(fit.n))
        self.tile_rms.set(f"{fit.rms:.1f}")
        self.tile_r2.set(f"{fit.r2:.3f}", "#34d399" if fit.r2 > 0.5 else "#fbbf24")

        # A positive exponent says the basin gets deeper as the resonance gets
        # higher, which is the wrong way round. It means the control points do
        # not constrain a power law -- almost always too few of them, spanning
        # too little of the frequency range.
        message = f"H = {fit.a:.4g}·f₀^{fit.b:.4f}.  " + "  ".join(notes[:2])
        if fit.b >= 0:
            message = ("This fit is physically implausible: b is positive, so "
                       "it has depth increasing with frequency. Treat it as a "
                       "warning that the control points do not constrain a "
                       "power law — check the depths, and prefer a published "
                       "law until you have more holes.  " + message)
            self.warn("Fitted exponent is positive — see the note below the fit.")
        elif fit.n < 5:
            message += ("  Only %d control points: the exponent is weakly "
                        "determined." % fit.n)
        self.fit_note.setText(message)

        self.project.regression.a = fit.a
        self.project.regression.b = fit.b
        self.project.regression.fitted = True
        self.project.regression.n_points = fit.n
        self.project.regression.rms = fit.rms
        self.project.regression.name = f"fitted to {fit.n} wells"
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self.refresh()
        if fit.b < 0:
            self.ok(f"Fitted a = {fit.a:.4g}, b = {fit.b:.4f} "
                    f"(RMS {fit.rms:.1f} m over {fit.n} wells).")

    # -- drawing -----------------------------------------------------------
    def _draw_map(self) -> None:
        """The basin as the current law sees it — the point of the whole page.

        Redrawn on every change to a and b, so choosing a different law or
        refitting shows immediately what it does to the depths. The boreholes
        are drawn on top with their residuals, because a depth map that
        disagrees with a hole you drilled is telling you something.
        """
        law = self.project.regression
        sites = [s for s in self.project.sites
                 if s.status != "excluded" and np.isfinite(s.x)
                 and np.isfinite(s.f0)]
        if len(sites) < 3:
            self.depth_map.message(
                "At least three computed sites with coordinates are needed "
                "for a depth map.")
            self.map_note.setText("")
            self.residual_note.setText("")
            return

        x = np.array([s.x for s in sites])
        y = np.array([s.y for s in sites])
        depth = bedrock.depth_from_f0(np.array([s.f0 for s in sites]),
                                      law.a, law.b)

        grid = None
        if np.isfinite(depth).sum() >= 3:
            try:
                grid = grids.interpolate(x, y, depth, method="linear",
                                         mask="hull")
            except Exception as exc:                   # noqa: BLE001
                self.ws.log(f"depth map: {exc}", "warn")

        self.depth_map.plot(
            x, y, depth, labels=[s.label() for s in sites], grid=grid,
            cmap=self.map_cmap.currentText(),
            title=f"depth to bedrock — H = {law.a:.4g}·f₀^{law.b:.3f}",
            unit="depth (m)", wells=self.project.wells,
            show_labels=len(sites) <= 30)
        self.map_note.setText(
            f"{len(sites)} sites · {law.name}"
            + (f", RMS {law.rms:.1f} m" if law.fitted
               and np.isfinite(law.rms) else ""))
        self._show_residuals()

    def _show_residuals(self) -> None:
        """How far the current law misses each borehole it can be checked against."""
        law = self.project.regression
        lines = []
        worst = None
        for well in self.project.wells:
            site = self.project.site(well.site) if well.site else None
            if site is None or not (np.isfinite(site.f0)
                                    and np.isfinite(well.bedrock_depth)):
                continue
            predicted = bedrock.depth_from_f0(site.f0, law.a, law.b)
            if not np.isfinite(predicted):
                continue
            ratio = predicted / well.bedrock_depth
            if worst is None or abs(np.log(ratio)) > abs(np.log(worst)):
                worst = ratio
            lines.append(
                f"{well.name}: drilled {well.bedrock_depth:.1f} m, law says "
                f"{predicted:.1f} m ({ratio:.2f}×)"
                + (f", {well.link_distance:,.0f} m from {site.label()}"
                   if np.isfinite(well.link_distance) else ""))

        if not lines:
            self.residual_note.setText(
                "No borehole can be checked against this law yet — add one "
                "with a depth, linked to a computed site.")
            return

        text = "Check against drilling:  " + "   ·   ".join(lines)
        if worst is not None and (worst < 0.7 or worst > 1.4):
            text += (f"   Off by {worst:.2f}× at worst — this law does not "
                     f"describe this basin. Fit your own, or scale it to the "
                     f"holes you have.")
        self.residual_note.setText(text)

    def _draw(self) -> None:
        law = self.project.regression
        active = bedrock.Law("active", law.a, law.b)
        sites = [s for s in self.project.sites
                 if s.status != "excluded" and np.isfinite(s.f0)]
        site_f0 = np.array([s.f0 for s in sites])

        f0 = depth = None
        if self.project.wells:
            f0, depth, _ = bedrock.pairs_from_wells(sites, self.project.wells)
            if f0.size == 0:
                f0 = depth = None

        self.plot.plot(f0, depth, fit=self._fit_result, laws=bedrock.LAWS,
                       active=active, site_f0=site_f0)

        if site_f0.size:
            depths = bedrock.depth_from_f0(site_f0, law.a, law.b)
            good = np.isfinite(depths)
            if good.any():
                self.tile_min.set(f"{np.nanmin(depths[good]):.1f}")
                self.tile_median.set(f"{np.nanmedian(depths[good]):.1f}")
                self.tile_max.set(f"{np.nanmax(depths[good]):.1f}")


class _FitSettings:
    method = "log-linear"
    #: Only reached by a well that somehow has no link at all; the table's own
    #: auto-linking has no distance cut, so this is a backstop, not a policy.
    max_distance = 5000.0


def _fmt(value, digits: int) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"
