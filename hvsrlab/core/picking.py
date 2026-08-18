"""Finding the resonance: automatic peak detection and user-guided picking.

ProTO picks the largest maximum in the analysis band and lets the user override
it by clicking. That is reproduced here, with two additions the SESAME tests
need: the half-amplitude frequencies either side of the peak (f⁻ and f⁺), and a
prominence filter so that ripple on a smoothed curve is not reported as a
secondary resonance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Peak:
    """One maximum of an H/V curve."""

    frequency: float
    amplitude: float
    index: int
    prominence: float = float("nan")
    f_minus: float = float("nan")     # where the curve last fell below A0/2 below f0
    f_plus: float = float("nan")      # …and first falls below A0/2 above f0

    @property
    def width(self) -> float:
        if np.isfinite(self.f_minus) and np.isfinite(self.f_plus):
            return self.f_plus - self.f_minus
        return float("nan")

    def as_tuple(self) -> tuple[float, float]:
        return (self.frequency, self.amplitude)


def find_peaks(freq: np.ndarray, curve: np.ndarray, *,
               fmin: float = 0.0, fmax: float = np.inf,
               min_amplitude: float = 0.0,
               prominence_ratio: float = 0.05) -> list[Peak]:
    """All local maxima of *curve* inside the band, strongest first.

    *prominence_ratio* is relative to the curve's own peak-to-trough range, so
    the same setting behaves sensibly whether H/V tops out at 2 or at 12.
    """
    from scipy.signal import find_peaks as _sp_find_peaks, peak_prominences

    freq = np.asarray(freq, dtype=float)
    curve = np.asarray(curve, dtype=float)
    finite = np.isfinite(curve)
    if not finite.any():
        return []

    work = np.where(finite, curve, -np.inf)
    span = float(np.nanmax(curve[finite]) - np.nanmin(curve[finite]))
    prominence = max(1e-12, prominence_ratio * span) if span > 0 else None

    idx, _ = _sp_find_peaks(work, prominence=prominence)
    if idx.size == 0:                    # a monotone or single-sample band
        best = int(np.argmax(work))
        idx = np.array([best])
    proms = peak_prominences(work, idx)[0] if idx.size else np.zeros(0)

    peaks: list[Peak] = []
    for i, p in zip(idx, proms):
        f, a = float(freq[i]), float(curve[i])
        if not (fmin <= f <= fmax) or a < min_amplitude or not np.isfinite(a):
            continue
        fm, fp = half_amplitude_bounds(freq, curve, int(i))
        peaks.append(Peak(frequency=f, amplitude=a, index=int(i),
                          prominence=float(p), f_minus=fm, f_plus=fp))
    peaks.sort(key=lambda pk: pk.amplitude, reverse=True)
    return peaks


def main_peak(freq: np.ndarray, curve: np.ndarray, *,
              fmin: float = 0.0, fmax: float = np.inf) -> Peak | None:
    """The largest maximum in the band — ProTO's automatic pick."""
    peaks = find_peaks(freq, curve, fmin=fmin, fmax=fmax)
    return peaks[0] if peaks else None


def pick_nearest(freq: np.ndarray, curve: np.ndarray, f_click: float, *,
                 snap: bool = True, window: float = 0.15) -> Peak:
    """Turn a click at *f_click* into a pick.

    With *snap* the pick moves to the largest sample within ±*window* (as a
    fraction of the clicked frequency, so the tolerance is constant on a log
    axis), which is what a user means when they click near a crest.
    """
    freq = np.asarray(freq, dtype=float)
    curve = np.asarray(curve, dtype=float)
    i = int(np.argmin(np.abs(freq - f_click)))
    if snap:
        lo, hi = f_click * (1 - window), f_click * (1 + window)
        sel = np.where((freq >= lo) & (freq <= hi) & np.isfinite(curve))[0]
        if sel.size:
            i = int(sel[np.argmax(curve[sel])])
    fm, fp = half_amplitude_bounds(freq, curve, i)
    return Peak(frequency=float(freq[i]), amplitude=float(curve[i]), index=i,
                f_minus=fm, f_plus=fp)


def half_amplitude_bounds(freq: np.ndarray, curve: np.ndarray, i: int
                          ) -> tuple[float, float]:
    """Frequencies either side of sample *i* where the curve drops to A0/2.

    Linearly interpolated between samples. Returns NaN on a side where the
    curve never falls that far inside the computed band — which is itself
    diagnostic, and is what SESAME's criteria (v) and (vi) test.
    """
    curve = np.asarray(curve, dtype=float)
    freq = np.asarray(freq, dtype=float)
    if not 0 <= i < curve.size or not np.isfinite(curve[i]):
        return (float("nan"), float("nan"))
    half = curve[i] / 2.0

    f_minus = float("nan")
    for j in range(i, 0, -1):
        if np.isfinite(curve[j - 1]) and curve[j - 1] <= half:
            f_minus = _interp(freq[j - 1], curve[j - 1], freq[j], curve[j], half)
            break
    f_plus = float("nan")
    for j in range(i, curve.size - 1):
        if np.isfinite(curve[j + 1]) and curve[j + 1] <= half:
            f_plus = _interp(freq[j], curve[j], freq[j + 1], curve[j + 1], half)
            break
    return (f_minus, f_plus)


def _interp(f0: float, a0: float, f1: float, a1: float, target: float) -> float:
    if a1 == a0:
        return float(f1)
    return float(f0 + (target - a0) * (f1 - f0) / (a1 - a0))


def window_peak_statistics(window_f0: np.ndarray, ok: np.ndarray | None = None,
                           kind: str = "lognormal") -> dict[str, float]:
    """Scatter of the per-window peak frequency.

    SESAME expresses σ(f0) as a standard deviation of f0 itself, but the
    quantity is positive and multiplicative, so the log-normal form is also
    reported: ``sigma_log`` multiplies and divides f0 to give the band.
    """
    f = np.asarray(window_f0, dtype=float)
    if ok is not None and ok.size == f.size:
        f = f[ok]
    f = f[np.isfinite(f) & (f > 0)]
    if f.size < 2:
        return {"n": float(f.size), "mean": float(f[0]) if f.size else float("nan"),
                "std": float("nan"), "sigma_log": float("nan"),
                "median": float(f[0]) if f.size else float("nan")}

    logs = np.log(f)
    sigma_log = float(np.std(logs, ddof=1))
    return {
        "n": float(f.size),
        "mean": float(np.exp(np.mean(logs))) if kind == "lognormal"
        else float(np.mean(f)),
        "median": float(np.median(f)),
        "std": float(np.std(f, ddof=1)),
        "sigma_log": sigma_log,
    }
