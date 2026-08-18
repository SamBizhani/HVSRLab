"""SESAME (2004) criteria for a reliable and clear H/V peak.

From *Guidelines for the implementation of the H/V spectral ratio technique on
ambient vibrations*, SESAME European research project WP12, deliverable
D23.12, Annex A — the standard the whole community judges an f0 pick against,
and the one thing OpenHVSR-ProTO leaves entirely to the operator's eye.

Three conditions establish that the *curve* is reliable:

1. ``f0 > 10 / Lw`` — the peak is resolved by the window length.
2. ``nc = Lw · nw · f0 > 200`` — enough significant cycles were averaged.
3. the amplitude scatter σ_A stays below 2 (below 3 if f0 < 0.5 Hz) over
   ``[0.5·f0, 2·f0]``.

Six further criteria establish that the *peak* is clear; SESAME asks for at
least five. They test that the curve falls to half amplitude on both flanks,
that the peak exceeds 2, that adding and subtracting one standard deviation
does not move it, and that the frequency and amplitude scatter stay inside
f0-dependent thresholds.

σ_A here is the multiplicative (log-normal) standard deviation — the factor by
which the curve is multiplied and divided to give the ±1σ band — which is what
SESAME's thresholds of 1.58/1.78/2.0/2.5/3.0 are expressed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import picking


@dataclass
class Criterion:
    """One test, its measured value, its threshold, and the verdict."""

    key: str
    text: str
    passed: bool
    value: float = float("nan")
    threshold: float = float("nan")
    detail: str = ""

    @property
    def mark(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class SesameReport:
    f0: float = float("nan")
    a0: float = float("nan")
    reliability: list[Criterion] = field(default_factory=list)
    clarity: list[Criterion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_reliability(self) -> int:
        return sum(c.passed for c in self.reliability)

    @property
    def n_clarity(self) -> int:
        return sum(c.passed for c in self.clarity)

    @property
    def reliable(self) -> bool:
        """SESAME requires all three reliability conditions."""
        return len(self.reliability) == 3 and self.n_reliability == 3

    @property
    def clear(self) -> bool:
        """SESAME requires at least five of the six clarity criteria."""
        return self.n_clarity >= 5

    @property
    def summary(self) -> str:
        return f"{self.n_reliability}/3 · {self.n_clarity}/6"

    @property
    def verdict(self) -> str:
        if self.reliable and self.clear:
            return "reliable and clear"
        if self.reliable:
            return "reliable, peak not clear"
        if self.clear:
            return "clear peak, curve not reliable"
        return "not reliable"

    def all_criteria(self) -> list[Criterion]:
        return list(self.reliability) + list(self.clarity)


def epsilon_threshold(f0: float) -> float:
    """ε(f0): the allowed standard deviation of the peak frequency, in Hz."""
    if f0 < 0.2:
        return 0.25 * f0
    if f0 < 0.5:
        return 0.20 * f0
    if f0 < 1.0:
        return 0.15 * f0
    if f0 < 2.0:
        return 0.10 * f0
    return 0.05 * f0


def theta_threshold(f0: float) -> float:
    """θ(f0): the allowed multiplicative amplitude scatter at f0."""
    if f0 < 0.2:
        return 3.0
    if f0 < 0.5:
        return 2.5
    if f0 < 1.0:
        return 2.0
    if f0 < 2.0:
        return 1.78
    return 1.58


def evaluate(result, f0: float | None = None) -> SesameReport:
    """Apply all nine tests to a computed :class:`~hvsrlab.core.hvsr.HVSRResult`.

    *f0* defaults to the automatic peak of the mean curve; pass the user's pick
    to judge that instead.
    """
    report = SesameReport()

    freq = np.asarray(result.freq, dtype=float)
    curve = np.asarray(result.hv, dtype=float)
    if freq.size == 0 or curve.size == 0:
        report.notes.append("no curve to evaluate")
        return report

    if f0 is None or not np.isfinite(f0):
        peak = picking.main_peak(freq, curve)
        if peak is None:
            report.notes.append("no peak found")
            return report
    else:
        peak = picking.pick_nearest(freq, curve, float(f0))

    report.f0, report.a0 = peak.frequency, peak.amplitude
    f0, a0 = peak.frequency, peak.amplitude

    lw = float(result.params.get("window_width_s", 0.0)) or _window_seconds(result)
    nw = int(result.n_ok)

    # σ_A as a multiplicative factor. `hv_std` already holds the standard
    # deviation of log(H/V) when statistics are log-normal; convert an additive
    # spread to the equivalent factor otherwise.
    sigma_a = _sigma_amplitude(result, curve)

    # -- reliability -------------------------------------------------------
    limit = 10.0 / lw if lw > 0 else np.inf
    report.reliability.append(Criterion(
        "R1", "f0 > 10 / window length", bool(f0 > limit), f0, limit,
        f"window {lw:g} s"))

    nc = lw * nw * f0
    report.reliability.append(Criterion(
        "R2", "significant cycles nc = Lw·nw·f0 > 200", bool(nc > 200.0),
        nc, 200.0, f"{nw} windows"))

    band = (freq > 0.5 * f0) & (freq < 2.0 * f0)
    sigma_limit = 3.0 if f0 < 0.5 else 2.0
    worst = float(np.nanmax(sigma_a[band])) if band.any() else float("nan")
    report.reliability.append(Criterion(
        "R3", f"σ_A(f) < {sigma_limit:g} over [0.5·f0, 2·f0]",
        bool(np.isfinite(worst) and worst < sigma_limit), worst, sigma_limit))

    # -- clarity ------------------------------------------------------------
    lower = (freq >= f0 / 4.0) & (freq <= f0)
    ok_i = bool(lower.any() and np.nanmin(curve[lower]) < a0 / 2.0)
    report.clarity.append(Criterion(
        "C1", "H/V drops below A0/2 somewhere in [f0/4, f0]", ok_i,
        float(np.nanmin(curve[lower])) if lower.any() else float("nan"),
        a0 / 2.0, f"f⁻ = {_fmt(peak.f_minus)} Hz"))

    upper = (freq >= f0) & (freq <= 4.0 * f0)
    ok_ii = bool(upper.any() and np.nanmin(curve[upper]) < a0 / 2.0)
    report.clarity.append(Criterion(
        "C2", "H/V drops below A0/2 somewhere in [f0, 4·f0]", ok_ii,
        float(np.nanmin(curve[upper])) if upper.any() else float("nan"),
        a0 / 2.0, f"f⁺ = {_fmt(peak.f_plus)} Hz"))

    report.clarity.append(Criterion(
        "C3", "A0 > 2", bool(a0 > 2.0), a0, 2.0))

    shift = _peak_shift(freq, curve, sigma_a, f0)
    report.clarity.append(Criterion(
        "C4", "peak of A ± σ_A stays within 5 % of f0",
        bool(np.isfinite(shift) and shift <= 0.05), shift, 0.05,
        f"worst shift {100 * shift:.1f} %" if np.isfinite(shift) else ""))

    stats = picking.window_peak_statistics(result.window_f0, result.ok)
    sigma_f = stats.get("std", float("nan"))
    eps = epsilon_threshold(f0)
    report.clarity.append(Criterion(
        "C5", "σ_f < ε(f0)", bool(np.isfinite(sigma_f) and sigma_f < eps),
        sigma_f, eps, f"over {int(stats.get('n', 0))} windows"))

    i0 = int(np.argmin(np.abs(freq - f0)))
    sigma_at_f0 = float(sigma_a[i0]) if sigma_a.size else float("nan")
    theta = theta_threshold(f0)
    report.clarity.append(Criterion(
        "C6", "σ_A(f0) < θ(f0)",
        bool(np.isfinite(sigma_at_f0) and sigma_at_f0 < theta),
        sigma_at_f0, theta))

    if nw < 10:
        report.notes.append(
            f"only {nw} windows kept — the scatter-based tests (R3, C5, C6) "
            "are weakly constrained")
    return report


def _window_seconds(result) -> float:
    ws = getattr(result, "windows", None)
    if ws is not None and ws.fs:
        return ws.width_samples / ws.fs
    return 0.0


def _sigma_amplitude(result, curve: np.ndarray) -> np.ndarray:
    """σ_A as the multiplicative factor SESAME's thresholds assume."""
    std = np.asarray(result.hv_std, dtype=float)
    if std.size != curve.size:
        return np.full(curve.shape, np.nan)
    if str(result.params.get("statistics", "lognormal")) == "lognormal":
        return np.exp(std)              # std is already of log(H/V)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(curve > 0, 1.0 + std / curve, np.nan)


def _peak_shift(freq: np.ndarray, curve: np.ndarray, sigma_a: np.ndarray,
                f0: float) -> float:
    """Largest fractional move of the peak when ±1σ is applied to the curve."""
    if not np.isfinite(sigma_a).any():
        return float("nan")
    shifts = []
    for band in (curve * sigma_a, curve / sigma_a):
        finite = np.isfinite(band)
        if not finite.any():
            continue
        i = int(np.nanargmax(np.where(finite, band, -np.inf)))
        shifts.append(abs(freq[i] - f0) / f0)
    return max(shifts) if shifts else float("nan")


def _fmt(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.3g}"
