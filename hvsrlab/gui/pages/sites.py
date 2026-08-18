"""Sites — where the survey is defined: raw data in, measurement points out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox, QSplitter,
    QTableWidget, QVBoxLayout, QWidget,
)

from ...io import crs as crs_io
from ...io import curves as curve_io
from ...io import mseed
from ...io import stations as station_io
from ...io import wells as well_io
from ...project import Site, Well
from ..plots import MapView
from ..widgets import Card, StatTile, button, hint, table_item
from .base import Page

#: A borehole further than this from its station gets flagged. Not a limit —
#: a judgement call the operator should make knowingly.
FAR_LINK_M = 500.0

COLUMNS = ["Site", "Latitude", "Longitude", "Elev", "Comp", "Record",
           "Window", "f₀ (Hz)", "A₀", "SESAME", "Status"]


class SitesPage(Page):
    title = "Sites"
    subtitle = ("Point the project at the raw recordings, match the station "
                "coordinates, and choose which sites take part.")

    def build(self) -> None:
        self.scan_button = self.header.add_action(
            button("Scan raw data", self._scan, primary=True))
        self.header.add_action(button("Import H/V curves", self._import_curves))
        self.header.add_action(button("Load boreholes", self._load_boreholes))

        # -- source card ---------------------------------------------------
        source = Card("Data source")
        grid = QVBoxLayout()
        grid.setSpacing(6)

        self.raw_edit = QLineEdit()
        self.raw_edit.setPlaceholderText(
            r"folder of MiniSEED — one subfolder per site, or all files together")
        grid.addLayout(_path_row("Raw MiniSEED", self.raw_edit,
                                 lambda: self._pick_dir(self.raw_edit)))

        self.station_edit = QLineEdit()
        self.station_edit.setPlaceholderText(
            "station list with coordinates (optional but needed for maps)")
        grid.addLayout(_path_row("Station file", self.station_edit,
                                 lambda: self._pick_file(
                                     self.station_edit,
                                     "Station list (*.txt *.csv *.dat);;All files (*)")))
        source.add_layout(grid)

        row = QHBoxLayout()
        row.addWidget(button("Match coordinates", self._match_coordinates,
                             tooltip="Attach coordinates from the station file "
                                     "to the scanned sites"))
        row.addWidget(button("Set time zone…", self._set_utc_offset,
                             tooltip="Display only — data stays in UTC"))
        row.addWidget(button("Grid…", self._set_zone,
                             tooltip="The UTM zone every plan coordinate, "
                                     "distance and profile length is measured "
                                     "in"))
        self.wells_check = QCheckBox("show boreholes")
        self.wells_check.setChecked(True)
        self.wells_check.stateChanged.connect(self._draw_map)
        row.addWidget(self.wells_check)
        row.addStretch(1)
        self.source_note = QLabel("")
        self.source_note.setObjectName("Hint")
        row.addWidget(self.source_note)
        source.add_layout(row)

        self.crs_label = QLabel("")
        self.crs_label.setObjectName("Hint")
        self.crs_label.setWordWrap(True)
        source.add(self.crs_label)
        self.add(source)

        # -- statistics ----------------------------------------------------
        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.tiles = {
            "sites": StatTile("sites"),
            "active": StatTile("active"),
            "computed": StatTile("computed"),
            "coords": StatTile("with coordinates"),
            "data": StatTile("raw volume"),
            "wells": StatTile("boreholes"),
        }
        for tile in self.tiles.values():
            stats.addWidget(tile)
        stats.addStretch(1)
        self.add_layout(stats)

        # -- table + map ---------------------------------------------------
        split = QSplitter(Qt.Horizontal)

        table_card = Card("Measurement points", dense=True)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("filter by name…")
        self.filter_edit.textChanged.connect(self._fill_table)
        table_card.add(self.filter_edit)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(COLUMNS)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        table_card.add(self.table, 1)
        table_card.add(hint("Right-click for status, window and coordinate "
                            "actions. Double-click a row to open it in H/V "
                            "Analysis."))
        self.table.itemDoubleClicked.connect(lambda *_: self._open_analysis())
        split.addWidget(table_card)

        map_card = Card("Survey layout", dense=True)
        self.map = MapView(height=4.2)
        self.map.clicked.connect(self._map_clicked)
        map_card.add(self.map, 1)
        split.addWidget(map_card)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.add(split, 1)

        self._rows: list[str] = []

    # -- wiring ------------------------------------------------------------
    def connect_signals(self) -> None:
        self.ws.projectChanged.connect(self.refresh)
        self.ws.sitesChanged.connect(self.refresh)
        self.ws.resultChanged.connect(lambda _: self.refresh())

    def refresh(self) -> None:
        project = self.project
        self.raw_edit.setText(project.raw_dir)
        self.station_edit.setText(project.station_file)
        self._show_crs()
        self._fill_table()
        self._fill_stats()
        self._draw_map()

    # -- actions -----------------------------------------------------------
    def _pick_dir(self, edit: QLineEdit) -> None:
        start = edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Choose a folder", start)
        if chosen:
            edit.setText(chosen)
            self.project.raw_dir = chosen
            self.ws.touch()

    def _pick_file(self, edit: QLineEdit, filters: str) -> None:
        start = edit.text() or self.project.raw_dir or str(Path.home())
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose a file", start, filters)
        if chosen:
            edit.setText(chosen)
            self.project.station_file = chosen
            self.ws.touch()

    def _set_utc_offset(self) -> None:
        from PyQt5.QtWidgets import QInputDialog

        value, ok = QInputDialog.getDouble(
            self, "Time zone",
            "Hours to add to UTC for display.\n"
            "Recording timestamps stay in UTC; this only changes how the\n"
            "hour-of-day plots and window labels read.",
            self.ws.utc_offset, -14.0, 14.0, 1)
        if ok:
            self.project.utc_offset = float(value)
            self.ws.touch()
            self.ok(f"Display offset set to UTC{value:+g}")

    def _show_crs(self) -> None:
        zone = self.ws.zone
        if zone is None:
            self.crs_label.setText(
                "Plan grid: not set yet — it is chosen from the survey centroid "
                "the first time coordinates are matched.")
            return
        warning = self.ws.zone_warning()
        text = (f"Plan grid: <b>{zone.label}</b>, metres. "
                f"Latitude and longitude are converted automatically "
                f"({crs_io.available()}).")
        if warning:
            text += f"<br><span style='color:#fbbf24'>{warning}</span>"
        self.crs_label.setText(text)

    def _set_zone(self) -> None:
        """Override the automatic zone.

        Worth doing when a survey straddles a boundary and the neighbouring
        sheet is the one the rest of the project's mapping uses; otherwise the
        automatic choice is right.
        """
        from PyQt5.QtWidgets import QInputDialog

        zone = self.ws.zone
        current = zone.zone if zone else 1
        value, ok = QInputDialog.getInt(
            self, "Plan grid",
            "UTM zone (1-60).\n\n"
            "Everything in metres — site positions, profile lengths, "
            "interpolation grids, the ProTO export — is measured on this grid.\n"
            "Changing it re-projects every site and borehole.",
            current, 1, 60)
        if not ok:
            return
        south, _ = QInputDialog.getItem(
            self, "Plan grid", "Hemisphere:", ["South", "North"],
            0 if (zone is None or zone.south) else 1, False)
        self.ws.set_zone(crs_io.UTM(zone=value, south=(south == "South")))
        self.ws.sitesChanged.emit()
        self.ok(f"Plan grid set to {self.ws.zone.label}.")

    def _scan(self) -> None:
        raw = self.raw_edit.text().strip()
        if not raw:
            self.warn("Choose the folder holding the MiniSEED files first.")
            return
        if not Path(raw).exists():
            self.fail(f"{raw} does not exist.")
            return

        self.project.raw_dir = raw
        self.project.station_file = self.station_edit.text().strip()
        self.project.source_kind = "mseed"
        self.scan_button.setEnabled(False)

        def work(job):
            job.log_line(f"Scanning {raw}")
            result = mseed.scan(
                raw, progress=lambda i, n, name: job.counted(i, n, "cataloguing"))
            job.log_line(f"{len(result.sites)} site(s), {result.n_files} files, "
                         f"{result.total_bytes / 1e9:.1f} GB")
            for warning in result.warnings:
                job.log_line(f"  ! {warning}")
            return result

        self.ws.submit("Scan raw data", work, on_done=self._scan_done)

    def _scan_done(self, job) -> None:
        self.scan_button.setEnabled(True)
        if job.state.value != "succeeded" or job.result is None:
            self.fail(f"Scan failed: {job.error}")
            return

        result = job.result
        self.ws.set_recordings(result.sites)
        added = self.project.add_sites(
            Site(sid=rec.sid, name=rec.sid, source="mseed", fs=rec.fs,
                 files={c: rec.files[c][0].path for c in rec.components})
            for rec in result.sites)

        for site in self.project.sites:
            rec = self.ws.recording(site.sid)
            if rec is not None and not site.fs:
                site.fs = rec.fs

        self._match_coordinates(silent=True)
        self._suggest_sampling_rate()
        self.ws.touch()
        self.ws.sitesChanged.emit()
        if not self.ws.current_sid and self.project.sites:
            self.ws.set_current(self.project.sites[0].sid)
        self.ok(f"Catalogued {len(result.sites)} site(s); {added} new.")

    def _suggest_sampling_rate(self) -> None:
        """Pick a decimation target the first time a survey is scanned.

        H/V above 25 Hz is rarely meaningful and 250 Hz costs five times the
        memory and time of 50 Hz for nothing. Only ever set when the user has
        not chosen one.
        """
        params = self.project.params
        if params.target_fs:
            return
        rates = {s.fs for s in self.project.sites if s.fs}
        if not rates:
            return
        native = max(rates)
        needed = 2.5 * params.freq_max
        if native > needed:
            for candidate in (50.0, 100.0, 125.0, 200.0):
                if candidate >= needed and native % candidate == 0:
                    params.target_fs = candidate
                    self.ws.notify.emit(
                        f"Native rate is {native:g} Hz; decimating to "
                        f"{candidate:g} Hz for analysis to "
                        f"{params.freq_max:g} Hz. Change it in H/V Analysis.",
                        "info")
                    return

    def _match_coordinates(self, silent: bool = False) -> None:
        path = self.station_edit.text().strip()
        if not path:
            if not silent:
                self.warn("No station file chosen.")
            return
        try:
            stations = station_io.read(path)
        except Exception as exc:                       # noqa: BLE001
            self.fail(f"Could not read the station file: {exc}")
            return

        self.project.station_file = path
        matched = projected = 0
        for site in self.project.sites:
            station = station_io.match(stations, site.sid) or \
                station_io.match(stations, site.name)
            if station is None:
                continue
            if station.projected:
                # The file was already in metres. Take it at face value and
                # let the project's zone turn it back into degrees.
                site.x, site.y = station.easting, station.northing
                projected += 1
            else:
                site.lat, site.lon = station.lat, station.lon
            site.elev = site.z = station.elev
            matched += 1

        self.ws.assign_plan_coordinates()
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self._show_crs()
        if not silent or matched:
            tone = "good" if matched == len(self.project.sites) else "warn"
            note = (f" {projected} were already in metres and were read as "
                    f"{self.ws.zone.name if self.ws.zone else 'grid'} "
                    f"coordinates." if projected else "")
            self.ws.notify.emit(
                f"Matched {matched} of {len(self.project.sites)} sites to "
                f"{len(stations)} station records.{note}", tone)

    def _import_curves(self) -> None:
        """Add sites from pre-computed H/V curve files (SAF, Geopsy, ProTO)."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose H/V curve files", self.project.raw_dir or str(Path.home()),
            "Curve files (*.txt *.dat *.csv *.hv *.SAF *.saf);;All files (*)")
        if not files:
            return

        added = 0
        failures: list[str] = []
        for path in files:
            try:
                freq, hv, std = curve_io.read_curve(path)
            except Exception as exc:                   # noqa: BLE001
                failures.append(f"{Path(path).name}: {exc}")
                continue
            sid = Path(path).stem
            if self.project.site(sid) is not None:
                continue
            site = Site(sid=sid, name=sid, source="curve", curve_file=path)
            i = int(np.nanargmax(hv)) if np.isfinite(hv).any() else 0
            site.f0, site.a0 = float(freq[i]), float(hv[i])
            site.f0_source = "auto"
            self.project.sites.append(site)
            added += 1

        self.ws.touch()
        self.ws.sitesChanged.emit()
        if failures:
            self.fail(f"Imported {added}; {len(failures)} failed — "
                      + "; ".join(failures[:3]))
        else:
            self.ok(f"Imported {added} curve(s).")

    def _load_boreholes(self) -> None:
        """Read borehole control points and put them on the map.

        Two shapes are accepted, and which one a file is gets decided by
        reading it: a table with one row per hole (name, coordinates, depth to
        bedrock), or ProTO-style logs of ``lithology thickness`` pairs, one
        file per hole, where the depth is the cumulative thickness above the
        first layer whose name marks basement.
        """
        paths_chosen, _ = QFileDialog.getOpenFileNames(
            self, "Borehole table or logs", self.project.raw_dir or str(Path.home()),
            "Text files (*.txt *.csv *.dat);;All files (*)")
        if not paths_chosen:
            return

        added, failures = 0, []
        for path in paths_chosen:
            try:
                wells = well_io.read_table(path)
                if not wells:                       # not a table — try a log
                    wells = [well_io.read_log(path)]
            except Exception as exc:                # noqa: BLE001
                failures.append(f"{Path(path).name}: {exc}")
                continue
            known = {w.name for w in self.project.wells}
            for well in wells:
                if well.name in known:
                    continue
                self.project.wells.append(well)
                known.add(well.name)
                added += 1

        placed = self.ws.locate_wells()
        link_note = self._link_wells()
        self.ws.touch()
        self.ws.sitesChanged.emit()
        self._draw_map()

        message = f"Loaded {added} borehole(s); {placed} have coordinates."
        if failures:
            self.fail(message + "  " + "; ".join(failures[:2]))
        elif added and placed < added:
            self.warn(message + link_note + " Those without coordinates can "
                      "still be used by linking them to a site on the Bedrock "
                      "page.")
        else:
            self.ok(message + link_note)

    def _link_wells(self) -> str:
        """Tie the boreholes to their nearest sites, and say how far that was."""
        touched = self.ws.link_wells()
        if not touched:
            return ""
        distances = np.array([d for _, _, d in touched])
        far = int((distances > FAR_LINK_M).sum())
        note = (f" Linked to the nearest site: median "
                f"{np.median(distances):,.0f} m, furthest "
                f"{distances.max():,.0f} m.")
        if far:
            note += (f" {far} are over {FAR_LINK_M:,.0f} m away — check them on "
                     f"the Bedrock page before fitting.")
        return note

    # -- table -------------------------------------------------------------
    def _fill_table(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        sites = [s for s in self.project.sites
                 if not needle or needle in s.label().lower()]
        self._rows = [s.sid for s in sites]

        self.table.blockSignals(True)
        self.table.setRowCount(len(sites))
        for row, site in enumerate(sites):
            rec = self.ws.recording(site.sid)
            comps = "".join(rec.components) if rec else \
                "".join(sorted(site.files)) or "—"
            record = f"{rec.duration_days:.1f} d" if rec else "—"
            window = _window_label(site)
            status_colour = {"active": None, "locked": "#fbbf24",
                             "excluded": "#f87171"}.get(site.status)

            values = [
                (site.label(), None, False),
                (_num(site.lat, 5), None, True),
                (_num(site.lon, 5), None, True),
                (_num(site.elev, 1), None, True),
                (comps, None, False),
                (record, None, True),
                (window, None, False),
                (_num(site.f0, 3), "#38bdf8", True),
                (_num(site.a0, 2), None, True),
                (site.sesame_score or "—", _sesame_colour(site.sesame_score), False),
                (site.status, status_colour, False),
            ]
            for col, (text, colour, right) in enumerate(values):
                self.table.setItem(row, col, table_item(
                    text, colour=colour, align_right=right,
                    bold=(col == 0 and site.sid == self.ws.current_sid)))
        self.table.blockSignals(False)

        if self.ws.current_sid in self._rows:
            self.table.selectRow(self._rows.index(self.ws.current_sid))

    def _selection_changed(self) -> None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) == 1:
            row = rows.pop()
            if 0 <= row < len(self._rows):
                self.ws.set_current(self._rows[row])
                self._draw_map()

    def _selected_sids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self._rows[r] for r in rows if 0 <= r < len(self._rows)]

    def _context_menu(self, point) -> None:
        sids = self._selected_sids()
        if not sids:
            return
        menu = QMenu(self)
        menu.addAction("Open in H/V Analysis", self._open_analysis)
        menu.addSeparator()
        for status, label in (("active", "Set active"),
                              ("locked", "Lock (keep results, skip recompute)"),
                              ("excluded", "Exclude from the survey")):
            menu.addAction(label, lambda s=status: self._set_status(s))
        menu.addSeparator()
        menu.addAction("Clear computed result", self._clear_results)
        menu.addAction("Remove from project", self._remove_sites)
        menu.exec_(self.table.viewport().mapToGlobal(point))

    def _set_status(self, status: str) -> None:
        for sid in self._selected_sids():
            site = self.project.site(sid)
            if site is not None:
                site.status = status
        self.ws.touch()
        self.ws.sitesChanged.emit()

    def _clear_results(self) -> None:
        for sid in self._selected_sids():
            self.ws.drop_result(sid)
        self.ws.sitesChanged.emit()

    def _remove_sites(self) -> None:
        doomed = set(self._selected_sids())
        self.project.sites = [s for s in self.project.sites if s.sid not in doomed]
        self.ws.touch()
        self.ws.sitesChanged.emit()

    def _open_analysis(self) -> None:
        window = self.window()
        if hasattr(window, "go_to_page"):
            window.go_to_page("H/V Analysis")

    # -- map ---------------------------------------------------------------
    def _draw_map(self) -> None:
        x, y, labels = self.ws.coordinates()
        if x.size == 0 or not np.isfinite(x).any():
            self.map.message(
                "No coordinates yet.\nChoose a station file and press "
                "“Match coordinates”.")
            return
        sites = [s for s in self.project.sites if s.status != "excluded"]
        f0 = np.array([s.f0 for s in sites], dtype=float)
        highlight = next((i for i, s in enumerate(sites)
                          if s.sid == self.ws.current_sid), -1)
        self.map.plot(x, y, f0 if np.isfinite(f0).any() else None,
                      labels=labels, title="f₀ (Hz)" if np.isfinite(f0).any()
                      else "site layout", unit="f₀ (Hz)", highlight=highlight,
                      show_labels=x.size <= 40,
                      wells=(self.project.wells
                             if self.wells_check.isChecked() else None))

    def _map_clicked(self, mx: float, my: float, _button: int) -> None:
        x, y, _ = self.ws.coordinates()
        if x.size == 0:
            return
        sites = [s for s in self.project.sites if s.status != "excluded"]
        d = np.hypot(x - mx, y - my)
        i = int(np.nanargmin(d))
        self.ws.set_current(sites[i].sid)
        self._fill_table()
        self._draw_map()

    def _fill_stats(self) -> None:
        sites = self.project.sites
        active = [s for s in sites if s.is_active]
        computed = [s for s in sites if np.isfinite(s.f0)]
        coords = [s for s in sites if np.isfinite(s.lat)]
        volume = sum(r.bytes for r in self.ws.recordings.values())

        self.tiles["sites"].set(str(len(sites)))
        self.tiles["active"].set(str(len(active)))
        self.tiles["computed"].set(
            str(len(computed)), "#34d399" if computed else None)
        self.tiles["coords"].set(
            str(len(coords)), "#f87171" if coords and len(coords) < len(sites) else None)
        self.tiles["data"].set(f"{volume / 1e9:.0f} GB" if volume else "—")
        self.tiles["wells"].set(str(len(self.project.wells))
                                if self.project.wells else "—")
        self.source_note.setText(
            f"{len(self.ws.recordings)} recording(s) catalogued"
            if self.ws.recordings else "not scanned yet")


def _path_row(label: str, edit: QLineEdit, browse) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    caption = QLabel(label)
    caption.setMinimumWidth(110)
    row.addWidget(caption)
    row.addWidget(edit, 1)
    row.addWidget(button("Browse…", browse))
    return row


def _num(value: float, digits: int) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _window_label(site: Site) -> str:
    if not site.time.is_set():
        return "—"
    mark = {"auto": "auto", "manual": "manual", "all": "all"}.get(site.time.mode, "")
    return f"{site.time.start[5:16]} · {site.time.hours:g} h ({mark})"


def _sesame_colour(score: str) -> str | None:
    if not score:
        return None
    try:
        reliability, clarity = score.split("·")
        r = int(reliability.strip().split("/")[0])
        c = int(clarity.strip().split("/")[0])
    except (ValueError, IndexError):
        return None
    if r == 3 and c >= 5:
        return "#34d399"
    if r == 3:
        return "#fbbf24"
    return "#f87171"
