"""1D Model — what soil column would resonate where this site does."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...core import bedrock, model1d
from ..plots import ModelView
from ..widgets import (
    Card, ParamForm, StatTile, button, hint, scroll_column, section_label)
from .base import Page

HEADERS = ["Thickness (m)", "Vs (m/s)", "Density (kg/m³)", "Damping"]


class ModelPage(Page):
    title = "1D Model"
    subtitle = ("Fit a layered column whose SH resonance matches the measured "
                "f₀ — depth from physics rather than from another basin's "
                "regression.")

    def build(self) -> None:
        self.site_label = QLabel("no site selected")
        self.site_label.setStyleSheet("font-weight: 600;")
        self.header.add_action(self.site_label)

        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        model_card = Card("Layers")
        self.table = QTableWidget(2, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumHeight(190)
        model_card.add(self.table)
        model_card.add(hint("The last row is the half-space; its thickness is "
                            "ignored."))

        row = QHBoxLayout()
        row.addWidget(button("Add layer", self._add_layer))
        row.addWidget(button("Remove", self._remove_layer))
        row.addWidget(button("Forward", self._forward, primary=True))
        model_card.add_layout(row)

        tiles = QHBoxLayout()
        tiles.setSpacing(6)
        self.tile_f0 = StatTile("model f₀ (Hz)")
        self.tile_obs = StatTile("observed f₀")
        self.tile_h = StatTile("cover (m)")
        for tile in (self.tile_f0, self.tile_obs, self.tile_h):
            tiles.addWidget(tile)
        model_card.add_layout(tiles)

        invert_card = Card("Fit to the observed peak")
        self.form = ParamForm()
        self.form.integer("n_layers", "Layers above bedrock", minimum=1,
                          maximum=4)
        self.form.number("vs_min", "Vs from", minimum=30.0, maximum=3000.0,
                         step=10.0, decimals=0, suffix="m/s")
        self.form.number("vs_max", "to", minimum=50.0, maximum=4000.0,
                         step=50.0, decimals=0, suffix="m/s")
        self.form.number("h_min", "Thickness from", minimum=0.5, maximum=1000.0,
                         step=1.0, decimals=1, suffix="m")
        self.form.number("h_max", "to", minimum=1.0, maximum=5000.0, step=10.0,
                         decimals=0, suffix="m")
        self.form.number("vs_halfspace", "Half-space Vs", minimum=200.0,
                         maximum=6000.0, step=100.0, decimals=0, suffix="m/s")
        self.form.number("weight_shape", "Weight on curve shape", minimum=0.0,
                         maximum=1.0, step=0.05, decimals=2,
                         tooltip="Leave at 0. Any weight here asks the fit to "
                                 "match H/V amplitude with an SH transfer "
                                 "function — two different quantities — and it "
                                 "pays for the mismatch by inventing a weak "
                                 "impedance contrast, which drives the cover "
                                 "velocity to its upper bound.")
        invert_card.add(self.form)
        invert_card.add(button("Invert", self._invert, primary=True))

        self.note = QLabel("")
        self.note.setObjectName("Hint")
        self.note.setWordWrap(True)
        invert_card.add(self.note)
        invert_card.add(hint(
            "Thickness and velocity trade off exactly against f₀: only their "
            "ratio is resolved. Fix one from a borehole or from the "
            "ambient-noise Vs model and read the other — the returned model is "
            "one member of that family, not the answer."))

        qw_card = Card("Quarter-wavelength check")
        self.qw_form = ParamForm()
        self.qw_form.number("vs", "Average Vs", minimum=30.0, maximum=3000.0,
                            step=10.0, decimals=0, suffix="m/s")
        self.qw_form.changed.connect(self._quarter_wavelength)
        qw_card.add(self.qw_form)
        self.qw_label = QLabel("—")
        self.qw_label.setWordWrap(True)
        qw_card.add(self.qw_label)
        qw_card.add(hint("H = Vs / (4·f₀). Exact for one uniform layer on a "
                         "rigid half-space; an under-estimate where velocity "
                         "increases gradually with depth."))

        layout.addWidget(model_card)
        layout.addWidget(invert_card)
        layout.addWidget(qw_card)
        layout.addStretch(1)
        split.addWidget(scroll_column(left, 386))

        self.view = ModelView(height=4.6)
        split.addWidget(self.view)
        split.setStretchFactor(1, 1)
        self.add(split, 1)

        self._settings = _Settings()
        self.form.load(self._settings)
        self.qw_form.load(_QW())
        self._set_model(model1d.Model.two_layer(40.0, 300.0))

    # -- wiring ------------------------------------------------------------
    def connect_signals(self) -> None:
        self.ws.currentChanged.connect(lambda _: self.refresh())
        self.ws.resultChanged.connect(lambda _: self.refresh())

    def refresh(self) -> None:
        site = self.site
        self.site_label.setText(site.label() if site else "no site selected")
        if site is not None and np.isfinite(site.f0):
            self.tile_obs.set(f"{site.f0:.3f}")
        else:
            self.tile_obs.set("—")
        if site is not None and np.isfinite(site.f0):
            self.note.setText(self._tradeoff_text(site.f0))
        self._quarter_wavelength()
        self._draw()

    # -- model table -------------------------------------------------------
    def _set_model(self, model: model1d.Model) -> None:
        self.table.setRowCount(model.n)
        for row, layer in enumerate(model.layers):
            for col, value in enumerate((layer.thickness, layer.vs,
                                         layer.density, layer.damping)):
                self.table.setItem(row, col, QTableWidgetItem(f"{value:g}"))

    def _model(self) -> model1d.Model | None:
        layers = []
        for row in range(self.table.rowCount()):
            try:
                values = [float(self.table.item(row, c).text())
                          for c in range(len(HEADERS))]
            except (AttributeError, ValueError):
                self.fail(f"Row {row + 1} has a value that is not a number.")
                return None
            layers.append(model1d.Layer(*values))
        if len(layers) < 2:
            self.fail("A model needs at least one layer over a half-space.")
            return None
        return model1d.Model(layers)

    def _add_layer(self) -> None:
        row = max(0, self.table.rowCount() - 1)
        self.table.insertRow(row)
        for col, value in enumerate((20.0, 300.0, 1900.0, 0.02)):
            self.table.setItem(row, col, QTableWidgetItem(f"{value:g}"))

    def _remove_layer(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        for row in rows:
            if self.table.rowCount() > 2:
                self.table.removeRow(row)

    # -- actions -----------------------------------------------------------
    def _forward(self) -> None:
        model = self._model()
        if model is None:
            return
        self._draw(model)

    def _invert(self) -> None:
        site = self.site
        result = self.ws.result(site.sid) if site else None
        if result is None:
            self.warn("Compute this site's H/V first.")
            return
        self.form.apply(self._settings)
        s = self._settings
        params = self.ws.params_for(site)

        def work(job):
            job.progress_to(0.1, "searching")
            return model1d.invert(
                result.freq, result.hv, n_layers=s.n_layers,
                vs_bounds=(s.vs_min, s.vs_max),
                thickness_bounds=(s.h_min, s.h_max),
                vs_halfspace=s.vs_halfspace,
                fmin=params.freq_min, fmax=params.freq_max,
                weight_shape=s.weight_shape)

        self.ws.submit(f"Invert {site.label()}", work, on_done=self._invert_done)

    def _invert_done(self, job) -> None:
        if job.state.value != "succeeded":
            self.fail(f"Inversion failed: {job.error}")
            return
        inv = job.result
        self._set_model(inv.model)
        self._draw(inv.model, inv)

        s = self._settings
        at_bounds = [
            f"{name} is at its bound"
            for name, value, low, high in (
                ("Vs", inv.model.layers[0].vs, s.vs_min, s.vs_max),
                ("thickness", inv.model.layers[0].thickness, s.h_min, s.h_max))
            if abs(value - low) / low < 0.02 or abs(value - high) / high < 0.02
        ]
        message = (f"f₀ model {inv.f0_model:.3f} Hz vs observed "
                   f"{inv.f0_observed:.3f} Hz, misfit {inv.misfit:.4f}, "
                   f"{inv.n_evaluations} forward runs.\n\n"
                   + self._tradeoff_text(inv.f0_observed))
        if at_bounds:
            message += ("\n\n" + "; ".join(at_bounds)
                        + " — the fit is pressed against the edge of the search "
                          "range, so this is a limit, not a result. Widen the "
                          "range, or set “Weight on curve shape” to 0 if it is "
                          "above it.")
        self.note.setText(message)
        self.ok("Inversion finished.")

    def _tradeoff_text(self, f0: float) -> str:
        """Spell out the family of models that fit this f₀ equally well.

        One answer from an f₀-only fit is misleading: the search returns a
        member of a family, not the member. Showing the family is the honest
        output, and it is what makes the case for bringing in a borehole or a
        velocity from somewhere else.
        """
        if not np.isfinite(f0) or f0 <= 0:
            return ""
        ratio = 4.0 * f0
        pairs = "   ".join(f"{vs:.0f} m/s → {vs / ratio:.0f} m"
                           for vs in (200, 300, 500, 900))
        return (f"f₀ alone fixes only the ratio Vs/H = {ratio:.2f} s⁻¹, so "
                f"these all fit equally well:\n{pairs}\n"
                f"Fix Vs or H from a borehole or a velocity model and read the "
                f"other; the layer table above is one member of that family.")

    def _quarter_wavelength(self) -> None:
        site = self.site
        qw = _QW()
        self.qw_form.apply(qw)
        if site is None or not np.isfinite(site.f0):
            self.qw_label.setText("—")
            return
        depth = bedrock.quarter_wavelength_depth(site.f0, qw.vs)
        law = self.project.regression
        empirical = bedrock.depth_from_f0(site.f0, law.a, law.b)
        self.qw_label.setText(
            f"At f₀ = {site.f0:.3f} Hz with Vs = {qw.vs:g} m/s:  "
            f"H = {depth:.1f} m.\n"
            f"The active regression gives {empirical:.1f} m, which implies an "
            f"average Vs of {4 * site.f0 * empirical:.0f} m/s.")

    # -- drawing -----------------------------------------------------------
    def _draw(self, model=None, inversion=None) -> None:
        site = self.site
        result = self.ws.result(site.sid) if site else None
        model = model or self._model()

        if model is not None:
            f0_model = model1d.fundamental(model)
            self.tile_f0.set(f"{f0_model:.3f}" if np.isfinite(f0_model) else "—")
            self.tile_h.set(f"{model.total_thickness:.1f}")
        else:
            f0_model = float("nan")

        if result is None or result.freq.size == 0:
            self.view.plot(model, f0_model=f0_model)
            return

        freq = result.freq
        predicted = (model1d.transfer_function(model, freq)
                     if model is not None else None)
        self.view.plot(model, freq=freq, observed=result.hv,
                       predicted=predicted,
                       f0_obs=site.f0 if np.isfinite(site.f0) else float("nan"),
                       f0_model=f0_model)


class _Settings:
    n_layers = 1
    vs_min = 80.0
    vs_max = 900.0
    h_min = 1.0
    h_max = 400.0
    vs_halfspace = 1500.0
    #: Zero on purpose. Weighting the curve shape makes the fit match H/V
    #: amplitude against an SH transfer function; the only way it can do that
    #: is by choosing a weak impedance contrast, which pushes the cover
    #: velocity to whatever upper bound it is given. Measured on a basin test
    #: site: weight 0 gives Vs 510 m/s at 63 m with exact f0; weight 0.1
    #: or more gives 840 m/s at 104 m, still exact in f0 but driven there by an
    #: amplitude comparison that does not hold.
    weight_shape = 0.0


class _QW:
    vs = 300.0
