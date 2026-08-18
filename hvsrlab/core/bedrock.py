"""Turning f0 into depth: published power laws, and a regression on your own wells.

Ibs-von Seht & Wohlenberg (1999) observed that over a sedimentary basin the
resonance frequency and the depth to bedrock follow ``H = a · f0^b``, and
calibrated ``a = 96, b = -1.388`` against boreholes in the Lower Rhine
Embayment. Many authors have since refitted the same form to their own basins;
the coefficients are strongly site-dependent, which is exactly why ProTO —
and this module — let you refit them against local control.

Two things are worth keeping in view when using this:

* A power law fitted in one basin carries that basin's velocity structure.
  Applying it elsewhere is an assumption, not a measurement. Prefer
  :func:`fit_regression` on local boreholes whenever you have three or more.
* The published coefficients below are transcribed from the literature and are
  offered as starting points. Check them against the original paper before a
  number derived from one goes into a report.

The quarter-wavelength relation ``H = Vs / (4·f0)`` is also provided: it is not
a regression but the physics the regressions approximate, and it is the right
tool when an independent Vs estimate exists — from the ambient-noise tomography
running alongside this survey, for instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Law:
    """A published ``H = a · f0^b`` calibration."""

    name: str
    a: float
    b: float
    region: str = ""
    reference: str = ""

    def depth(self, f0):
        return depth_from_f0(f0, self.a, self.b)


#: Published calibrations, for use as starting points. Verify against the
#: source before quoting a derived depth.
LAWS: tuple[Law, ...] = (
    Law("Ibs-von Seht & Wohlenberg (1999)", 96.0, -1.388,
        "Lower Rhine Embayment, Germany",
        "Bull. Seismol. Soc. Am. 89(1), 250–259"),
    Law("Delgado et al. (2000)", 55.11, -1.256,
        "Bajo Segura basin, Spain", "J. Appl. Geophys. 45, 19–32"),
    Law("Parolai et al. (2002)", 108.0, -1.551,
        "Cologne area, Germany", "Bull. Seismol. Soc. Am. 92(6), 2521–2527"),
    Law("Hinzen et al. (2004)", 137.0, -1.190,
        "Lower Rhine Embayment, Germany", "Netherlands J. Geosci. 83(4)"),
    Law("D'Amico et al. (2008)", 53.461, -1.010,
        "Florence, Italy", "Bull. Seismol. Soc. Am. 98(3)"),
    Law("Birgören et al. (2009)", 150.99, -1.153,
        "Istanbul, Turkey", "J. Seismol. 13, 249–261"),
)


def law_names() -> list[str]:
    return [law.name for law in LAWS]


def get_law(name: str) -> Law | None:
    for law in LAWS:
        if law.name == name:
            return law
    return None


def depth_from_f0(f0, a: float, b: float):
    """``H = a · f0^b``, NaN-safe and array-safe."""
    f = np.asarray(f0, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = a * np.power(f, b)
    h = np.where(np.isfinite(f) & (f > 0), h, np.nan)
    return float(h) if np.ndim(f0) == 0 else h


def f0_from_depth(depth, a: float, b: float):
    """The inverse relation, for plotting a law against borehole control."""
    h = np.asarray(depth, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.power(h / a, 1.0 / b)
    f = np.where(np.isfinite(h) & (h > 0), f, np.nan)
    return float(f) if np.ndim(depth) == 0 else f


def quarter_wavelength_depth(f0, vs):
    """``H = Vs / (4·f0)`` — the resonance condition for a single soft layer.

    *vs* is the average shear-wave velocity of the cover. Exact for one
    homogeneous layer on a rigid half-space; for a gradient it under-estimates
    depth, which is the physical reason the fitted exponent *b* comes out
    steeper than −1.
    """
    f = np.asarray(f0, dtype=float)
    v = np.asarray(vs, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = v / (4.0 * f)
    h = np.where(np.isfinite(f) & (f > 0), h, np.nan)
    return float(h) if np.ndim(f0) == 0 and np.ndim(vs) == 0 else h


def average_vs_from_pairs(f0, depth):
    """Back out the implied average Vs from paired f0 and known depth."""
    f = np.asarray(f0, dtype=float)
    h = np.asarray(depth, dtype=float)
    return 4.0 * f * h


@dataclass
class FitResult:
    a: float
    b: float
    n: int
    rms: float                      # RMS depth residual, in the units of H
    r2: float                       # coefficient of determination, in log space
    residuals: np.ndarray
    method: str = "log-linear"

    def depth(self, f0):
        return depth_from_f0(f0, self.a, self.b)


def fit_regression(f0, depth, *, method: str = "log-linear") -> FitResult:
    """Fit ``H = a · f0^b`` to paired resonance frequencies and known depths.

    ``log-linear`` (the default) regresses ``log H`` on ``log f0``, which is
    where the power law is straight and where the residuals are closest to
    homoscedastic. ``nonlinear`` reproduces ProTO's ``fitnlm`` — least squares
    on depth itself, seeded from Ibs-von Seht's coefficients — which weights
    deep control points far more heavily.

    Needs at least three usable pairs; two would fit exactly and report a
    meaningless zero residual.
    """
    f = np.asarray(f0, dtype=float)
    h = np.asarray(depth, dtype=float)
    good = np.isfinite(f) & np.isfinite(h) & (f > 0) & (h > 0)
    f, h = f[good], h[good]
    if f.size < 3:
        raise ValueError(
            f"need at least 3 usable (f0, depth) pairs, got {f.size}")

    if method == "nonlinear":
        from scipy.optimize import curve_fit
        model = lambda x, a, b: a * np.power(x, b)      # noqa: E731
        (a, b), _ = curve_fit(model, f, h, p0=(96.0, -1.388), maxfev=20000)
    else:
        b, log_a = np.polyfit(np.log(f), np.log(h), 1)
        a = float(np.exp(log_a))

    predicted = depth_from_f0(f, a, b)
    residuals = h - predicted
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    log_res = np.log(h) - np.log(np.maximum(predicted, 1e-12))
    ss_res = float(np.sum(log_res ** 2))
    ss_tot = float(np.sum((np.log(h) - np.mean(np.log(h))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitResult(a=float(a), b=float(b), n=int(f.size), rms=rms, r2=r2,
                     residuals=residuals, method=method)


def pairs_from_wells(sites, wells, *, max_distance: float = 250.0
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Match wells to sites and return the ``(f0, depth)`` pairs for a fit.

    A well is used when it names a site explicitly, or when exactly one site
    lies within *max_distance* metres of it. Ambiguous wells are skipped and
    named in the returned notes, because silently pairing a borehole with the
    wrong station is the kind of error that survives into a depth map.
    """
    f0s: list[float] = []
    depths: list[float] = []
    notes: list[str] = []

    by_id = {s.sid: s for s in sites}
    for well in wells:
        if not np.isfinite(well.bedrock_depth) or well.bedrock_depth <= 0:
            notes.append(f"{well.name}: no usable bedrock depth")
            continue

        site = by_id.get(well.site) if well.site else None
        if site is None:
            near = [
                (np.hypot(s.x - well.x, s.y - well.y), s) for s in sites
                if np.isfinite(s.x) and np.isfinite(well.x)
            ]
            near = [(d, s) for d, s in near if d <= max_distance]
            if len(near) != 1:
                notes.append(
                    f"{well.name}: {len(near)} sites within {max_distance:g} m "
                    "— link it to a site explicitly")
                continue
            site = near[0][1]

        if not np.isfinite(site.f0):
            notes.append(f"{well.name}: site {site.label()} has no f0 yet")
            continue
        f0s.append(float(site.f0))
        depths.append(float(well.bedrock_depth))

    return np.asarray(f0s), np.asarray(depths), notes
