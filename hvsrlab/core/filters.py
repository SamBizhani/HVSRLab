"""Butterworth pre-filters, matching OpenHVSR-ProTO's filter panel.

ProTO designs an IIR Butterworth with MATLAB's ``fdesign``/``design`` and
applies it with ``filter`` — a one-pass, causal filter that shifts phase. Here
the default is zero-phase (``filtfilt``), which does not move the arrival times
that the STA/LTA anti-trigger is judging and does not bias the spectral
amplitudes. Pass ``zero_phase=False`` for bit-comparable ProTO behaviour.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, sosfilt, sosfiltfilt, tf2sos


def design(kind: str, order: int, fs: float, fmin: float = 0.0,
           fmax: float = 0.0):
    """Return second-order sections for the requested filter.

    ``kind`` is one of ``bandpass``, ``lowpass``, ``highpass``. Corner
    frequencies are clipped just inside the Nyquist so a badly chosen ``fmax``
    degrades to a sensible filter instead of raising.
    """
    nyq = 0.5 * float(fs)
    if nyq <= 0:
        raise ValueError("sampling rate must be positive")

    order = max(1, int(order))
    lo = float(fmin) / nyq
    hi = float(fmax) / nyq
    eps = 1e-6

    if kind == "bandpass":
        lo = min(max(lo, eps), 1 - 2 * eps)
        hi = min(max(hi, lo + eps), 1 - eps)
        return butter(order, [lo, hi], btype="bandpass", output="sos")
    if kind == "lowpass":
        hi = min(max(hi if hi > 0 else lo, eps), 1 - eps)
        return butter(order, hi, btype="lowpass", output="sos")
    if kind == "highpass":
        lo = min(max(lo, eps), 1 - eps)
        return butter(order, lo, btype="highpass", output="sos")
    raise ValueError(f"unknown filter kind {kind!r}")


def apply(data: np.ndarray, kind: str, order: int, fs: float,
          fmin: float = 0.0, fmax: float = 0.0, *,
          zero_phase: bool = True) -> np.ndarray:
    """Filter a 1D signal (or the columns of a 2D array)."""
    if kind in ("off", "", None):
        return data
    sos = design(kind, order, fs, fmin, fmax)
    x = np.asarray(data, dtype=float)
    axis = 0 if x.ndim > 1 else -1
    padlen = 3 * (sos.shape[0] * 2)
    if zero_phase and x.shape[axis] > padlen:
        return sosfiltfilt(sos, x, axis=axis)
    return sosfilt(sos, x, axis=axis)


def apply_all(data: dict[str, np.ndarray], kind: str, order: int, fs: float,
              fmin: float = 0.0, fmax: float = 0.0, *,
              zero_phase: bool = True) -> dict[str, np.ndarray]:
    """Filter every component of a recording, returning a new dict."""
    if kind in ("off", "", None):
        return data
    return {c: apply(v, kind, order, fs, fmin, fmax, zero_phase=zero_phase)
            for c, v in data.items()}


def response(kind: str, order: int, fs: float, fmin: float, fmax: float,
             n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude response of the designed filter, for the GUI's filter preview."""
    from scipy.signal import sosfreqz

    sos = design(kind, order, fs, fmin, fmax)
    w, h = sosfreqz(sos, worN=n, fs=fs)
    return w, np.abs(h)


__all__ = ["design", "apply", "apply_all", "response", "tf2sos", "filtfilt"]
