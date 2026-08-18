"""Cutting a recording into windows, and deciding which ones to keep.

The anti-trigger is OpenHVSR-ProTO's, restated in vectorised form: a window is
rejected when the loudest short-term average inside it exceeds the long-term
average by more than the threshold, judged on the noisiest of the three
components. ProTO's per-window Python-level loop becomes two cumulative sums
here, which matters when 8 hours of data yields several hundred windows.

The taper is applied. In ProTO, ``cosine_taper(wdat, tapervalue)`` is called
without assigning the result, so the taper never reaches the FFT — a bug that
leaves the spectra with the leakage the taper was meant to suppress.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WindowSet:
    """Window geometry plus the keep/reject decision for each window."""

    idx: np.ndarray                    # (nw, 2) int, [start, stop) sample indices
    ok: np.ndarray                     # (nw,) bool
    fs: float = 0.0
    width_s: float = 0.0
    overlap_pc: float = 0.0
    start: float = float("nan")        # epoch seconds of sample 0
    ratio: np.ndarray = field(default_factory=lambda: np.zeros(0))  # STA/LTA per window
    manual: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def n(self) -> int:
        return int(self.idx.shape[0])

    @property
    def n_ok(self) -> int:
        return int(np.count_nonzero(self.ok))

    @property
    def width_samples(self) -> int:
        return int(self.idx[0, 1] - self.idx[0, 0]) if self.n else 0

    def centre_times(self) -> np.ndarray:
        """Window centres in seconds from the start of the recording."""
        if not self.n:
            return np.zeros(0)
        return (self.idx[:, 0] + self.idx[:, 1]) * 0.5 / self.fs

    def centre_epochs(self) -> np.ndarray:
        """Window centres as epoch seconds, for time-of-day analysis."""
        base = 0.0 if not np.isfinite(self.start) else self.start
        return base + self.centre_times()

    def toggle(self, w: int) -> None:
        """Flip one window's keep/reject state and remember it was manual."""
        if not 0 <= w < self.n:
            return
        if self.manual.size != self.n:
            self.manual = np.zeros(self.n, dtype=bool)
        self.ok[w] = not self.ok[w]
        self.manual[w] = True

    def set_all(self, value: bool) -> None:
        self.ok[:] = bool(value)
        self.manual = np.ones(self.n, dtype=bool)


def make_windows(npts: int, fs: float, width_s: float, overlap_pc: float,
                 start: float = float("nan")) -> WindowSet:
    """Lay out equal-length windows over ``npts`` samples.

    Follows ProTO's geometry: the step is ``width - overlap`` samples and any
    trailing partial window is dropped, so every window has identical length
    and the FFTs stack into one array.
    """
    fs = float(fs)
    width = int(width_s * fs)
    if width < 2:
        raise ValueError("window width must cover at least 2 samples")
    if width > npts:
        raise ValueError(
            f"window of {width_s:g} s ({width} samples) is longer than the "
            f"{npts / fs:.1f} s of data available")

    overlap = int(0.01 * width * float(overlap_pc))
    step = max(1, width - overlap)
    starts = np.arange(0, npts - width + 1, step, dtype=np.int64)
    idx = np.column_stack([starts, starts + width])
    return WindowSet(idx=idx, ok=np.ones(len(idx), dtype=bool), fs=fs,
                     width_s=width_s, overlap_pc=overlap_pc, start=start,
                     ratio=np.zeros(len(idx)),
                     manual=np.zeros(len(idx), dtype=bool))


def sta_lta_mask(data: dict[str, np.ndarray], ws: WindowSet, *,
                 sta_s: float, lta_s: float, threshold: float) -> WindowSet:
    """Apply ProTO's STA/LTA anti-trigger, in place, returning *ws*.

    For each window the long-term average is the mean of ``|x|`` over the
    ``lta_s`` seconds ending at the window's end; the short-term average is the
    largest mean of ``|x|`` over the consecutive ``sta_s`` blocks inside the
    window. The window is rejected when the worst component's ratio exceeds
    *threshold*. Windows the user has toggled by hand are left alone.
    """
    if not ws.n:
        return ws
    fs = ws.fs
    npts = int(max(len(v) for v in data.values()))
    ns_sta = max(1, int(sta_s * fs))
    ns_lta = max(ns_sta + 1, int(lta_s * fs))
    if ns_lta > npts // 2:
        ns_lta = max(ns_sta + 1, npts // 2 - 1)
    if ns_sta >= ns_lta:
        # ProTO warns and skips the test rather than inventing a decision.
        ws.ratio = np.zeros(ws.n)
        return ws

    ratio = np.zeros(ws.n)
    for comp, x in data.items():
        a = np.abs(np.asarray(x, dtype=float))
        csum = np.concatenate([[0.0], np.cumsum(a)])

        # LTA: the ns_lta samples ending at each window's end, clamped to the
        # start of the record for the first windows (as ProTO does).
        lta_b = np.minimum(ws.idx[:, 1], npts)
        lta_a = lta_b - ns_lta
        short = lta_a < 0
        lta_a[short] = ws.idx[short, 0]
        lta_b[short] = np.minimum(lta_a[short] + ns_lta, npts)
        lta = (csum[lta_b] - csum[lta_a]) / np.maximum(1, lta_b - lta_a)

        # STA: the maximum over consecutive ns_sta blocks tiling each window.
        width = ws.width_samples
        n_blocks = max(1, width // ns_sta)
        offsets = np.arange(n_blocks) * ns_sta
        block_a = ws.idx[:, 0][:, None] + offsets[None, :]
        block_b = np.minimum(block_a + ns_sta, npts)
        block_a = np.minimum(block_a, npts)
        block_mean = (csum[block_b] - csum[block_a]) / np.maximum(1, block_b - block_a)
        sta = block_mean.max(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(lta > 0, sta / lta, 0.0)
        ratio = np.maximum(ratio, np.nan_to_num(r))

    ws.ratio = ratio
    auto = ratio <= float(threshold)
    if ws.manual.size == ws.n:
        ws.ok = np.where(ws.manual, ws.ok, auto)
    else:
        ws.ok = auto
    return ws


def cosine_taper(n: int, percent: float) -> np.ndarray:
    """A cosine (Tukey) taper of length *n* tapering *percent* of each end.

    Same shape as ProTO's ``cosine_taper.m``: a raised cosine rising over the
    first ``n·percent/100`` samples, flat through the middle, mirrored at the
    end.
    """
    taper = np.ones(n)
    k = int(n * (float(percent) / 100.0))
    if k > 1:
        ramp = 0.5 * (1.0 - np.cos(np.arange(k) * np.pi / (k - 1)))
        taper[:k] = ramp
        taper[n - k:] = ramp[::-1]
    return taper


def extract(x: np.ndarray, ws: WindowSet, *, taper_pc: float = 0.0,
            demean: bool = True) -> np.ndarray:
    """Cut *x* into a ``(width, nw)`` array, demeaned and tapered per window.

    Windows are columns, matching ProTO's layout, so the FFT runs down axis 0.
    """
    x = np.asarray(x, dtype=float)
    width = ws.width_samples
    out = np.empty((width, ws.n), dtype=float)
    for w, (a, b) in enumerate(ws.idx):
        out[:, w] = x[a:b]
    if demean:
        out -= out.mean(axis=0, keepdims=True)
    if taper_pc:
        out *= cosine_taper(width, taper_pc)[:, None]
    return out


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def pad_length(width: int, pad_to: str | int | float) -> int:
    """Resolve ProTO's "pad windows" setting to an FFT length.

    ``"off"`` gives the next power of two above the window; anything numeric is
    rounded up to a power of two and never allowed below that floor.
    """
    floor = next_pow2(width)
    if pad_to in (None, "", "off", "Off", "OFF"):
        return floor
    try:
        requested = next_pow2(int(float(pad_to)))
    except (TypeError, ValueError):
        return floor
    return max(floor, requested)
