"""Fourier amplitude spectra and their smoothing.

Two departures from OpenHVSR-ProTO, both about cost rather than definition:

* **Smoothing is a matrix.** Konno-Ohmachi weights depend only on the
  frequency axis and the bandwidth, so the operator is built once and applied
  to every window with a single BLAS call. ProTO rebuilds the weights inside a
  loop over frequencies, for every window, on every parameter change.
* **The output grid may be logarithmic.** Smoothed spectra are band-limited by
  construction, so carrying 2000 linearly spaced points buys nothing; a few
  hundred log-spaced points sample a Konno-Ohmachi curve just as faithfully,
  make the plots honest, and cut the smoothing cost by an order of magnitude.
  Set ``grid="linear"`` for the ProTO frequency axis.

Amplitude convention follows ``samfft.m``: ``2·|FFT(x)| / L``, i.e. a one-sided
amplitude spectrum normalised by the *un-padded* window length, so zero padding
interpolates the spectrum without changing its level.

The smoothing window is that of Konno, K. & Ohmachi, T. (1998), *Ground-motion
characteristics estimated from spectral ratio between horizontal and vertical
components of microtremor*, Bull. Seismol. Soc. Am. 88(1), 228–241.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def frequency_axis(npad: int, fs: float) -> np.ndarray:
    """The one-sided frequency axis of an ``npad``-point real FFT."""
    return np.fft.rfftfreq(int(npad), d=1.0 / float(fs))


def amplitude_spectra(windows: np.ndarray, npad: int) -> np.ndarray:
    """One-sided amplitude spectra of column-wise *windows*.

    ``windows`` is ``(width, nw)``; the result is ``(npad//2 + 1, nw)``.
    """
    width = windows.shape[0]
    spec = np.fft.rfft(windows, n=int(npad), axis=0) / width
    return 2.0 * np.abs(spec)


def complex_spectra(windows: np.ndarray, npad: int) -> np.ndarray:
    """The same transform, kept complex.

    Rotation of the horizontals is linear in the transform, so the azimuthal
    analysis synthesises every azimuth from one pair of complex spectra instead
    of re-transforming the rotated traces at each angle.
    """
    width = windows.shape[0]
    return np.fft.rfft(windows, n=int(npad), axis=0) * (2.0 / width)


def output_grid(freq: np.ndarray, fmin: float, fmax: float, *,
                grid: str = "log", n: int = 512) -> np.ndarray:
    """Frequencies the smoothed spectra are reported on.

    ``linear`` returns the FFT bins inside the band (ProTO's behaviour);
    ``log`` returns *n* logarithmically spaced points, never finer than the
    FFT resolution at the low end.
    """
    fmin = max(float(fmin), float(freq[1]) if len(freq) > 1 else 0.0)
    fmax = min(float(fmax), float(freq[-1]))
    if fmax <= fmin:
        raise ValueError(f"empty frequency band: {fmin:g}–{fmax:g} Hz")

    if grid == "linear":
        sel = (freq >= fmin) & (freq <= fmax)
        return freq[sel]
    n = max(16, int(n))
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


@lru_cache(maxsize=8)
def _konno_ohmachi_cached(f_in: bytes, n_in: int, f_out: bytes, n_out: int,
                          b: float, cutoff: float) -> np.ndarray:
    fin = np.frombuffer(f_in, dtype=np.float64, count=n_in)
    fout = np.frombuffer(f_out, dtype=np.float64, count=n_out)
    return _konno_ohmachi_build(fin, fout, b, cutoff)


def _konno_ohmachi_build(f_in: np.ndarray, f_out: np.ndarray, b: float,
                         cutoff: float) -> np.ndarray:
    """The (n_out, n_in) Konno-Ohmachi operator, rows normalised to unit sum."""
    K = np.zeros((f_out.size, f_in.size), dtype=np.float64)
    positive = f_in > 0
    fin = f_in[positive]

    for i, fc in enumerate(f_out):
        if fc <= 0:
            continue
        lo = b * np.log10(fin / fc)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = (np.sin(lo) / lo) ** 4
        w[~np.isfinite(w)] = 1.0        # the fc bin itself: sin(x)/x -> 1
        if cutoff > 0:
            w[w < cutoff] = 0.0
        total = w.sum()
        if total > 0:
            K[i, positive] = w / total
    return K


def konno_ohmachi_matrix(f_in: np.ndarray, f_out: np.ndarray | None = None,
                         b: float = 40.0, cutoff: float = 1e-6) -> np.ndarray:
    """Smoothing operator taking spectra on *f_in* to smoothed values on *f_out*.

    The Konno-Ohmachi (1998) window ``[sin(b·log10(f/fc)) / (b·log10(f/fc))]⁴``
    has constant bandwidth on a logarithmic axis: small *b* smooths hard, large
    *b* barely at all, and 40 is the usual choice. Weights below *cutoff* of
    the peak are dropped, which costs nothing measurable and keeps far-field
    leakage out of the sum.
    """
    f_in = np.ascontiguousarray(np.asarray(f_in, dtype=np.float64))
    f_out = f_in if f_out is None else np.ascontiguousarray(
        np.asarray(f_out, dtype=np.float64))
    try:
        return _konno_ohmachi_cached(f_in.tobytes(), f_in.size,
                                     f_out.tobytes(), f_out.size,
                                     float(b), float(cutoff))
    except TypeError:                    # unhashable — build without caching
        return _konno_ohmachi_build(f_in, f_out, float(b), float(cutoff))


def moving_average_matrix(f_in: np.ndarray, f_out: np.ndarray | None = None,
                          width: float = 40.0) -> np.ndarray:
    """ProTO's plain "Average" smoothing, as an operator.

    *width* is read as a percentage of a decade, so that the same slider
    position gives a visually comparable amount of smoothing to Konno-Ohmachi's
    bandwidth parameter.
    """
    f_in = np.asarray(f_in, dtype=float)
    f_out = f_in if f_out is None else np.asarray(f_out, dtype=float)
    half = max(1e-3, 1.0 / max(1e-6, float(width)))     # half-width in decades

    K = np.zeros((f_out.size, f_in.size))
    positive = f_in > 0
    log_in = np.full(f_in.shape, -np.inf)
    log_in[positive] = np.log10(f_in[positive])
    for i, fc in enumerate(f_out):
        if fc <= 0:
            continue
        w = (np.abs(log_in - np.log10(fc)) <= half).astype(float)
        total = w.sum()
        if total > 0:
            K[i] = w / total
    return K


def smoothing_matrix(f_in: np.ndarray, f_out: np.ndarray | None, kind: str,
                     b: float) -> np.ndarray | None:
    """Dispatch to the requested smoothing operator; ``None`` means no smoothing."""
    if kind == "konno_ohmachi":
        return konno_ohmachi_matrix(f_in, f_out, b)
    if kind == "moving_average":
        return moving_average_matrix(f_in, f_out, b)
    if f_out is None or (f_out.shape == f_in.shape and np.allclose(f_in, f_out)):
        return None
    return _interpolation_matrix(f_in, f_out)


def _interpolation_matrix(f_in: np.ndarray, f_out: np.ndarray) -> np.ndarray:
    """Linear resampling as a matrix, for "no smoothing" onto a log grid."""
    K = np.zeros((f_out.size, f_in.size))
    idx = np.clip(np.searchsorted(f_in, f_out), 1, f_in.size - 1)
    f0, f1 = f_in[idx - 1], f_in[idx]
    span = np.where(f1 > f0, f1 - f0, 1.0)
    upper = (f_out - f0) / span
    rows = np.arange(f_out.size)
    K[rows, idx - 1] = 1.0 - upper
    K[rows, idx] = upper
    return K


def apply_smoothing(spec: np.ndarray, K: np.ndarray | None) -> np.ndarray:
    """Smooth ``(nf, nw)`` spectra with operator *K*."""
    return spec if K is None else K @ spec
