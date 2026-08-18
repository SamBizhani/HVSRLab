"""Maps & Profiles — the survey seen from above, and in section."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QInputDialog, QListWidget, QListWidgetItem, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...core import bedrock, grids
from ...project import Profile
from ..plots import MapView, SectionView
from ..widgets import (
    Card, ParamForm, button, hint, scroll_column, section_label, table_item)
from .base import Page

QUANTITIES = [
    ("f0", "f₀ — resonance frequency (Hz)"),
    ("a0", "A₀ — peak amplitude"),
    ("depth", "depth to bedrock (m)"),
    ("f0_std", "σ f₀ — window scatter (Hz)"),
    ("slice", "H/V at a chosen frequency"),
    ("clarity", "SESAME clarity score"),
]

COLORMAPS = ["viridis", "magma", "plasma", "cividis", "turbo", "RdYlBu_r",
             "coolwarm", "terrain", "jet"]


class MapsPage(Page):
    title = "Maps & Profiles"
    subtitle = ("Interpolate site results into maps, and cut sections through "
                "the survey.")

    def build(self) -> None:
        split = QSplitter(Qt.Horizontal)

        # ================= controls =======================================
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        map_card = Card("Map")
        self.quantity = QComboBox()
        for key, label in QUANTITIES:
            self.quantity.addItem(label, key)
        self.quantity.currentIndexChanged.connect(self._draw_map)
        map_card.add(self.quantity)

        self.slice_form = ParamForm()
        self.slice_freq = self.slice_form.number(
            "slice_frequency", "Frequency", minimum=0.01, maximum=200.0,
            step=0.5, decimals=3, suffix="Hz")
        self.slice_freq.setValue(1.0)
        map_card.add(self.slice_form)

        self.method = QComboBox()
        for value, label in (("linear", "Linear"), ("cubic", "Cubic"),
                             ("nearest", "Nearest"),
                             ("rbf", "Thin-plate spline")):
            self.method.addItem(label, value)
        self.method.currentIndexChanged.connect(self._draw_map)
        map_card.add(_row("Interpolation", self.method))

        self.mask = QComboBox()
        for value, label in (("hull", "Convex hull of sites"),
                             ("radius", "Near a site"), ("none", "None")):
            self.mask.addItem(label, value)
        self.mask.currentIndexChanged.connect(self._draw_map)
        map_card.add(_row("Trust region", self.mask))

        self.style = QComboBox()
        self.style.addItems(["filled", "lines"])
        self.style.currentIndexChanged.connect(self._draw_map)
        map_card.add(_row("Contours", self.style))

        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        self.cmap.currentIndexChanged.connect(self._draw_map)
        map_card.add(_row("Colour map", self.cmap))

        self.labels_check = QCheckBox("station labels")
        self.labels_check.stateChanged.connect(self._draw_map)
        map_card.add(self.labels_check)
        self.wells_check = QCheckBox("wells")
        self.wells_check.setChecked(True)
        self.wells_check.stateChanged.connect(self._draw_map)
        map_card.add(self.wells_check)
        map_card.add(hint(
            "Outside the sites there are no measurements. The trust region "
            "keeps the interpolator from drawing contours where nothing was "
            "recorded — leave it on for anything that leaves this screen."))

        profile_card = Card("Profiles")
        profile_card.add(hint(
            "Click the map twice to draw a profile — once for each end. Then "
            "use “Pick on map” to click sites in or out of it."))
        self.profile_list = QListWidget()
        self.profile_list.setMaximumHeight(100)
        self.profile_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.profile_list.currentRowChanged.connect(self._profile_selected)
        profile_card.add(self.profile_list)
        row = QHBoxLayout()
        self.redraw_button = button("Redraw ends", self._start_redraw)
        self.redraw_button.setCheckable(True)
        row.addWidget(self.redraw_button)
        row.addWidget(button("Rename", self._rename_profile))
        row.addWidget(button("Delete", self._delete_profile))
        row.addWidget(button("Clear all", self._clear_profiles))
        profile_card.add_layout(row)

        profile_card.add(section_label("sites on this profile"))
        self.member_table = QTableWidget(0, 4)
        self.member_table.setHorizontalHeaderLabels(
            ["", "Site", "Along (m)", "Offset (m)"])
        self.member_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.member_table.verticalHeader().setVisible(False)
        self.member_table.setAlternatingRowColors(True)
        self.member_table.setMaximumHeight(190)
        header = self.member_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in (0, 2, 3):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.member_table.itemChanged.connect(self._member_toggled)
        profile_card.add(self.member_table)

        row = QHBoxLayout()
        self.pick_button = button(
            "Pick on map", self._toggle_pick_mode,
            tooltip="Click sites on the map to add or remove them from this "
                    "profile. Ctrl+click does the same without arming this.")
        self.pick_button.setCheckable(True)
        row.addWidget(self.pick_button)
        row.addWidget(button("All", lambda: self._set_all_members(True)))
        row.addWidget(button("None", lambda: self._set_all_members(False)))
        row.addWidget(button("Reset", self._reset_members,
                             tooltip="Forget the manual picks and go back to "
                                     "the corridor"))
        profile_card.add_layout(row)
        self.member_note = QLabel("")
        self.member_note.setObjectName("Hint")
        self.member_note.setWordWrap(True)
        profile_card.add(self.member_note)
        profile_card.add(hint(
            "The section is built from the ticked sites and nothing else. "
            "A new profile starts with the sites its line actually passes — "
            "within half the typical station spacing — and the corridor below "
            "adjusts that. Ticking overrides the corridor and survives changes "
            "to it. Offset is the perpendicular distance from the line: a large "
            "one means the site is being projected a long way to reach it."))

        profile_card.add(section_label("section"))
        self.section_form = ParamForm()
        self.section_form.choice("axis", "Vertical axis",
                                 [("depth", "Pseudo-depth"),
                                  ("frequency", "Frequency")])
        self.section_form.choice("smoothing", "Smoothing",
                                 [("off", "Off"), ("layer", "Layer"),
                                  ("broad_layer", "Broad layer"),
                                  ("bubble", "Bubble")])
        self.section_form.integer("radius", "Radius", minimum=1, maximum=20)
        self.section_form.choice("normalisation", "Normalisation",
                                 [("off", "Off"),
                                  ("at_main_peak", "At each site's peak"),
                                  ("max_all_stations", "Max over all sites"),
                                  ("max_in_profile", "Max in this profile")])
        self.section_form.number("width", "Corridor half-width", minimum=0.0,
                                 maximum=100000.0, step=100.0, decimals=0,
                                 suffix="m",
                                 tooltip="How far off the line a site may sit "
                                         "and still be picked up. Belongs to "
                                         "this profile; 0 means every site in "
                                         "the survey. Your ticks override it.")
        self.section_form.changed.connect(self._settings_changed)
        profile_card.add(self.section_form)
        profile_card.add(hint(
            "Pseudo-depth maps every frequency through the active bedrock law, "
            "not just the peak. Away from f₀ that mapping has no physical "
            "warrant — read the deep part as pattern, not structure."))

        layout.addWidget(map_card)
        layout.addWidget(profile_card)
        layout.addStretch(1)
        split.addWidget(scroll_column(left, 336))

        # ================= views ==========================================
        self.tabs = QTabWidget()
        self.map = MapView(height=4.4)
        self.map.clicked.connect(self._map_clicked)
        self.tabs.addTab(self.map, "Map")
        self.section = SectionView(height=4.4)
        self.tabs.addTab(self.section, "Section")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        self.add(split, 1)

        self._pending_point: tuple[float, float] | None = None
        self._slice_cache: dict[str, np.ndarray] = {}
        self._redrawing = False
        self._filling_members = False
        self._loading_profile = False
        self._filling_profiles = False

    # -- wiring ------------------------------------------------------------
    def connect_signals(self) -> None:
        self.ws.sitesChanged.connect(self.refresh)
        self.ws.projectChanged.connect(self.refresh)
        self.ws.resultChanged.connect(lambda _: self._invalidate_slices())

    def refresh(self) -> None:
        self._fill_profiles()
        self._fill_members()
        self._draw_map()
        self._draw_section()

    def _invalidate_slices(self) -> None:
        self._slice_cache.clear()
        if self.isVisible():
            self._draw_map()

    # -- values ------------------------------------------------------------
    def _sites(self):
        return [s for s in self.project.sites
                if s.status != "excluded" and np.isfinite(s.x)]

    def _values(self, sites) -> tuple[np.ndarray, str, str]:
        key = self.quantity.currentData()
        law = self.project.regression

        if key == "f0":
            return (np.array([s.f0 for s in sites]), "f₀", "Hz")
        if key == "a0":
            return (np.array([s.a0 for s in sites]), "A₀", "")
        if key == "f0_std":
            return (np.array([s.f0_std for s in sites]), "σ f₀", "Hz")
        if key == "depth":
            f0 = np.array([s.f0 for s in sites])
            return (bedrock.depth_from_f0(f0, law.a, law.b),
                    f"depth (H = {law.a:.4g}·f^{law.b:.3f})", "m")
        if key == "clarity":
            return (np.array([_clarity(s.sesame_score) for s in sites]),
                    "SESAME clarity", "of 6")
        return (self._slice_values(sites), "H/V", "")

    def _slice_values(self, sites) -> np.ndarray:
        """H/V amplitude at one frequency, read from each site's stored result."""
        target = float(self.slice_freq.value())
        cache_key = f"{target:.6f}"
        cached = self._slice_cache.get(cache_key)
        if cached is not None and cached.size == len(sites):
            return cached

        values = np.full(len(sites), np.nan)
        for i, site in enumerate(sites):
            result = self.ws.result(site.sid)
            if result is None or result.freq.size == 0:
                continue
            j = int(np.argmin(np.abs(result.freq - target)))
            values[i] = result.hv[j]
        self._slice_cache[cache_key] = values
        return values

    # -- map ---------------------------------------------------------------
    def _draw_map(self) -> None:
        self.slice_form.setVisible(self.quantity.currentData() == "slice")
        sites = self._sites()
        if len(sites) < 1:
            self.map.message("No sites with coordinates.\n"
                             "Match a station file on the Sites page.")
            return

        x = np.array([s.x for s in sites])
        y = np.array([s.y for s in sites])
        values, title, unit = self._values(sites)

        grid = None
        if np.isfinite(values).sum() >= 3:
            try:
                grid = grids.interpolate(
                    x, y, values, method=self.method.currentData(),
                    mask=self.mask.currentData())
            except Exception as exc:                   # noqa: BLE001
                self.warn(f"Could not interpolate: {exc}")

        highlight = next((i for i, s in enumerate(sites)
                          if s.sid == self.ws.current_sid), -1)

        members: list[int] = []
        profile = self._current_profile()
        if profile is not None:
            in_corridor, _, _ = self._projection(profile, sites)
            chosen = set(profile.members([s.sid for s in sites], in_corridor))
            members = [i for i, s in enumerate(sites) if s.sid in chosen]

        self.map.plot(
            x, y, values, labels=[s.label() for s in sites], grid=grid,
            style=self.style.currentText(), cmap=self.cmap.currentText(),
            title=title, unit=unit, profiles=self.project.profiles,
            highlight=highlight, show_labels=self.labels_check.isChecked(),
            wells=self.project.wells if self.wells_check.isChecked() else None,
            members=members)

    def _map_clicked(self, x: float, y: float, button_id: int) -> None:
        modifier = getattr(self.map, "modifier", "") or ""
        picking = (self.pick_button.isChecked() or "control" in modifier)

        if button_id == 1 and picking and self._current_profile() is not None:
            self._toggle_nearest(x, y)
            return

        if button_id == 3:                     # right click selects a site
            sites = self._sites()
            if sites:
                d = np.hypot(np.array([s.x for s in sites]) - x,
                             np.array([s.y for s in sites]) - y)
                self.ws.set_current(sites[int(np.argmin(d))].sid)
                self._draw_map()
            return

        if self.pick_button.isChecked():
            self.pick_button.setChecked(False)     # drawing wins over picking

        if self._pending_point is None:
            self._pending_point = (x, y)
            self.ws.notify.emit(
                "Profile start set — click the other end." if not self._redrawing
                else "New start set — click the other end.", "info")
            return

        x1, y1 = self._pending_point
        self._pending_point = None

        if self._redrawing:
            profile = self._current_profile()
            self._redrawing = False
            self.redraw_button.setChecked(False)
            if profile is not None:
                profile.x1, profile.y1, profile.x2, profile.y2 = x1, y1, x, y
                self.ws.touch()
                self._fill_profiles()
                self._draw_map()
                self._draw_section()
                self.ws.notify.emit(f"{profile.name} moved.", "good")
                return

        profile = Profile(name=f"P{len(self.project.profiles) + 1}",
                          x1=x1, y1=y1, x2=x, y2=y,
                          width=self._default_corridor())
        self.project.profiles.append(profile)
        self.ws.touch()
        self._fill_profiles()
        self.profile_list.setCurrentRow(len(self.project.profiles) - 1)
        self._draw_map()
        self.tabs.setCurrentIndex(1)

    def _default_corridor(self) -> float:
        """A starting corridor: half the typical station separation.

        A site belongs to a profile when the line passes closer to it than half
        the distance to its own nearest neighbour — that is, when it is the
        station the line goes past. Starting from every site in the survey, as
        ProTO does, produces a "section" that is really the whole survey
        smeared along an arbitrary line.
        """
        sites = self._sites()
        if len(sites) < 2:
            return 0.0
        from scipy.spatial import cKDTree

        points = np.column_stack([[s.x for s in sites], [s.y for s in sites]])
        distance, _ = cKDTree(points).query(points, k=2)
        return float(max(100.0, np.median(distance[:, 1]) / 2.0))

    def _current_profile(self):
        row = self.profile_list.currentRow()
        if 0 <= row < len(self.project.profiles):
            return self.project.profiles[row]
        return None

    def _profile_selected(self, _row: int) -> None:
        if self._filling_profiles:
            return
        if self._current_profile() is None:
            self.pick_button.setChecked(False)
        profile = self._current_profile()
        if profile is not None:
            self._loading_profile = True
            try:
                widget = self.section_form.widget("width")
                if widget is not None:
                    widget.setValue(float(profile.width))
            finally:
                self._loading_profile = False
        self._fill_members()
        self._draw_map()
        self._draw_section()

    def _settings_changed(self) -> None:
        if self._loading_profile:
            return
        profile = self._current_profile()
        if profile is not None:
            settings = _SectionSettings()
            self.section_form.apply(settings)
            profile.width = float(settings.width)
            self.ws.touch()
        self._fill_members()
        self._draw_section()

    def _fill_profiles(self) -> None:
        """Rebuild the list without disturbing what is selected.

        Clearing a QListWidget emits currentRowChanged(-1), which any handler
        reads as "no profile selected". Rebuilding after every membership edit
        would therefore disarm map picking on each click -- the exact tedium
        picking on the map exists to avoid.
        """
        current = self.profile_list.currentRow()
        self._filling_profiles = True
        try:
            self.profile_list.clear()
            for p in self.project.profiles:
                marks = len(p.include) + len(p.exclude)
                suffix = f"   ({marks} edited)" if marks else ""
                self.profile_list.addItem(
                    f"{p.name}   {p.length / 1000:.2f} km{suffix}")
            if 0 <= current < self.profile_list.count():
                self.profile_list.setCurrentRow(current)
            elif self.profile_list.count():
                self.profile_list.setCurrentRow(0)
        finally:
            self._filling_profiles = False

        if not self.project.profiles:
            self.pick_button.setChecked(False)
            self._fill_members()

    def _start_redraw(self) -> None:
        if self._current_profile() is None:
            self.redraw_button.setChecked(False)
            self.warn("Select a profile first.")
            return
        self._redrawing = self.redraw_button.isChecked()
        self._pending_point = None
        if self._redrawing:
            self.tabs.setCurrentIndex(0)
            self.ws.notify.emit(
                "Click the map twice to place the new ends of this profile.",
                "info")

    def _rename_profile(self) -> None:
        profile = self._current_profile()
        if profile is None:
            return
        name, ok = QInputDialog.getText(self, "Rename profile", "Name:",
                                        text=profile.name)
        if ok and name.strip():
            profile.name = name.strip()
            self.ws.touch()
            self._fill_profiles()
            self._draw_map()
            self._draw_section()

    def _delete_profile(self) -> None:
        row = self.profile_list.currentRow()
        if 0 <= row < len(self.project.profiles):
            del self.project.profiles[row]
            self.ws.touch()
            self._fill_profiles()
            self._draw_map()
            self._draw_section()

    def _clear_profiles(self) -> None:
        self.project.profiles = []
        self.ws.touch()
        self._fill_profiles()
        self._draw_map()
        self.section.message("No profiles defined.")

    # -- membership ---------------------------------------------------------
    def _projection(self, profile, sites):
        """Corridor selection and geometry for *sites* against *profile*."""
        settings = _SectionSettings()
        self.section_form.apply(settings)
        settings.width = float(profile.width)      # the profile owns this

        # Between the two clicks that place a profile, both ends sit on the
        # same point and the segment has no direction to project onto. That is
        # a half-finished edit, not a bad profile, so select nothing and let
        # the second click complete it.
        if profile.x1 == profile.x2 and profile.y1 == profile.y2:
            return set(), {}, settings

        x = np.array([s.x for s in sites])
        y = np.array([s.y for s in sites])
        projection = grids.project_to_profile(
            profile.x1, profile.y1, profile.x2, profile.y2, x, y,
            width=settings.width)
        in_corridor = {sites[i].sid for i in projection.indices}

        # Distances for every site, not only those in the corridor, so an
        # excluded one still shows where it would land if ticked back on.
        full = grids.project_to_profile(profile.x1, profile.y1,
                                        profile.x2, profile.y2, x, y, width=0.0)
        geometry = {sites[i].sid: (float(full.distance[k]), float(full.offset[k]))
                    for k, i in enumerate(full.indices)}
        return in_corridor, geometry, settings

    def _fill_members(self) -> None:
        profile = self._current_profile()
        sites = self._sites()
        self._filling_members = True
        try:
            self.member_table.setRowCount(0)
            if profile is None or not sites:
                self.member_note.setText("")
                return
            in_corridor, geometry, _ = self._projection(profile, sites)
            members = set(profile.members([s.sid for s in sites], in_corridor))

            self.member_note.setText(
                f"{len(members)} of {len(sites)} sites on {profile.name} "
                f"({profile.length / 1000:.2f} km, corridor "
                f"±{profile.width:,.0f} m)")

            ordered = sorted(sites, key=lambda s: geometry.get(s.sid, (1e18, 0))[0])
            self.member_table.setRowCount(len(ordered))
            for row, site in enumerate(ordered):
                along, offset = geometry.get(site.sid, (float("nan"),) * 2)
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                check.setCheckState(Qt.Checked if site.sid in members
                                    else Qt.Unchecked)
                check.setData(Qt.UserRole, site.sid)
                self.member_table.setItem(row, 0, check)

                overridden = site.sid in profile.include or site.sid in profile.exclude
                label = site.label() + (" *" if overridden else "")
                colour = "#fbbf24" if overridden else (
                    None if np.isfinite(site.f0) else "#64748b")
                self.member_table.setItem(row, 1, table_item(
                    label, colour=colour,
                    tooltip="no H/V computed yet" if not np.isfinite(site.f0)
                    else ("manually set" if overridden else "")))
                self.member_table.setItem(row, 2, table_item(
                    f"{along:,.0f}" if np.isfinite(along) else "—",
                    align_right=True))
                self.member_table.setItem(row, 3, table_item(
                    f"{offset:,.0f}" if np.isfinite(offset) else "—",
                    align_right=True,
                    colour="#fbbf24" if abs(offset) > max(500.0, profile.length * 0.2)
                    else None))
        finally:
            self._filling_members = False

    def _member_toggled(self, item) -> None:
        if self._filling_members or item.column() != 0:
            return
        profile = self._current_profile()
        if profile is None:
            return
        sid = item.data(Qt.UserRole)
        sites = self._sites()
        in_corridor, _, _ = self._projection(profile, sites)
        profile.set_member(sid, item.checkState() == Qt.Checked,
                           in_corridor=sid in in_corridor)
        self.ws.touch()
        self._fill_profiles()
        self._draw_section()
        self._draw_map()

    def _toggle_pick_mode(self) -> None:
        if self.pick_button.isChecked() and self._current_profile() is None:
            self.pick_button.setChecked(False)
            self.warn("Draw or select a profile first.")
            return
        if self.pick_button.isChecked():
            self.tabs.setCurrentIndex(0)
            self.ws.notify.emit(
                "Click sites on the map to add or remove them from this "
                "profile. Green ring = on the section, red cross = left out.",
                "info")

    def _toggle_nearest(self, x: float, y: float) -> None:
        """Flip the membership of the site nearest the click.

        A click further from any site than a quarter of the typical station
        spacing is ignored, so a stray click on empty map does not silently
        change the section.
        """
        profile = self._current_profile()
        sites = self._sites()
        if profile is None or not sites:
            return

        distance = np.hypot(np.array([s.x for s in sites]) - x,
                            np.array([s.y for s in sites]) - y)
        i = int(np.argmin(distance))
        tolerance = max(self._default_corridor(), 1.0) * 1.5
        if distance[i] > tolerance:
            return

        site = sites[i]
        in_corridor, _, _ = self._projection(profile, sites)
        members = set(profile.members([s.sid for s in sites], in_corridor))
        now_member = site.sid not in members
        profile.set_member(site.sid, now_member,
                           in_corridor=site.sid in in_corridor)

        self.ws.touch()
        self._fill_profiles()
        self._fill_members()
        self._draw_map()
        self._draw_section()
        self.ws.notify.emit(
            f"{site.label()} {'added to' if now_member else 'removed from'} "
            f"{profile.name}.", "good" if now_member else "warn")

    def _set_all_members(self, member: bool) -> None:
        profile = self._current_profile()
        sites = self._sites()
        if profile is None or not sites:
            return
        in_corridor, _, _ = self._projection(profile, sites)
        for site in sites:
            profile.set_member(site.sid, member,
                               in_corridor=site.sid in in_corridor)
        self.ws.touch()
        self._fill_profiles()
        self._fill_members()
        self._draw_section()
        self._draw_map()

    def _reset_members(self) -> None:
        profile = self._current_profile()
        if profile is None:
            return
        profile.clear_overrides()
        self.ws.touch()
        self._fill_profiles()
        self._fill_members()
        self._draw_section()
        self._draw_map()

    # -- section -----------------------------------------------------------
    def _draw_section(self) -> None:
        profile = self._current_profile()
        if profile is None:
            self.section.message("Click the map twice to draw a profile.")
            return

        all_sites = self._sites()
        if not all_sites:
            self.section.message("No sites with coordinates.")
            return

        in_corridor, geometry, settings = self._projection(profile, all_sites)
        members = profile.members([s.sid for s in all_sites], in_corridor)
        sites = [s for s in all_sites
                 if s.sid in set(members) and np.isfinite(s.f0)]
        if len(sites) < 2:
            self.section.message(
                "Fewer than two computed sites are on this profile.\n"
                "Tick more of them in the list, or widen the corridor.")
            return
        sites.sort(key=lambda s: geometry.get(s.sid, (0.0, 0.0))[0])

        curves, freq = [], None
        keep, distance = [], []
        for site in sites:
            d = geometry.get(site.sid, (0.0, 0.0))[0]
            result = self.ws.result(site.sid)
            if result is None or result.freq.size == 0:
                continue
            if freq is None:
                freq = result.freq
                curves.append(result.hv)
            else:
                curves.append(np.interp(freq, result.freq, result.hv,
                                        left=np.nan, right=np.nan))
            keep.append(site)
            distance.append(d)

        if len(curves) < 2:
            self.section.message(
                "The sites on this profile have no stored H/V results.\n"
                "Compute them, or run a batch on the Batch & Export page.")
            return

        law = self.project.regression
        depth_axis = None
        if settings.axis == "depth":
            depth_axis = grids.frequency_to_depth(freq, law.a, law.b)

        peaks = np.array([s.a0 for s in keep])
        global_max = float(np.nanmax([s.a0 for s in self._sites()
                                      if np.isfinite(s.a0)] or [np.nan]))
        d, v, section = grids.build_section(
            np.asarray(distance), freq, curves, depth_axis=depth_axis,
            smoothing=settings.smoothing, smoothing_radius=settings.radius,
            normalisation=settings.normalisation, peak_amplitudes=peaks,
            global_max=global_max)

        bedrock_line = None
        if settings.axis == "depth":
            f0 = np.array([s.f0 for s in keep])
            bedrock_line = (np.asarray(distance),
                            bedrock.depth_from_f0(f0, law.a, law.b))

        self.section.plot(
            d, v, section,
            sites=[(dd, s.label()) for dd, s in zip(distance, keep)],
            depth=(settings.axis == "depth"), cmap=self.cmap.currentText(),
            title=f"{profile.name} — {len(keep)} of {len(all_sites)} sites",
            bedrock=bedrock_line)


class _SectionSettings:
    axis = "depth"
    smoothing = "off"
    radius = 2
    normalisation = "off"
    width = 0.0


def _row(label: str, widget: QWidget) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    caption = QLabel(label)
    caption.setMinimumWidth(96)
    row.addWidget(caption)
    row.addWidget(widget, 1)
    return holder


def _clarity(score: str) -> float:
    try:
        return float(score.split("·")[1].strip().split("/")[0])
    except (ValueError, IndexError, AttributeError):
        return float("nan")
