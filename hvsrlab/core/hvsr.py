"""The H/V computation itself, and the result object everything else reads.

The chain, in order:

1. optional Butterworth pre-filter
2. windowing, with the STA/LTA anti-trigger
3. per window: demean, cosine taper, zero-pad, FFT, amplitude
4. Konno-Ohmachi smoothing of each component separately
5. combine the horizontals, divide by the vertical, one curve per window
6. statistics across the kept windows

Step 4 before step 5 is deliberate and matches ProTO's "OPTION-B": smoothing
the components and then dividing is not the same as dividing and then
smoothing, and the former is what the SESAME guidelines describe.

The three ways to combine the horizontals are ProTO's, and they differ only by
a constant factor — ``total_energy`` is ``squared_average`` times √2 — but that
factor moves the amplitude past SESAME's A₀ > 2 threshold, so it is worth being
explicit about which one produced a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import filters, spectra as spec_mod, windows as win_mod
from .windows import WindowSet

Progress = Callable[[float, str], None]


def combine_horizontals(e: np.ndarray, n: np.ndarray, strategy: str) -> np.ndarray:
    """Collapse the two horizontal spectra into one, ProTO's three ways."""
    if strategy == "simple_average":
        return 0.5 * (e + n)
    if strategy == "total_energy":
        return np.sqrt(e * e + n * n)
    # squared_average — the default: sqrt[(E² + N²)/2]
    return np.sqrt(0.5) * np.sqrt(e * e + n * n)


@dataclass
class HVSRResult:
    """Everything one site's computation produced."""

    sid: str = ""
    freq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hv_windows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    ok: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    # Mean curve and its spread, on the convention named by ``statistics``.
    hv: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hv_std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hv_lo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hv_hi: np.ndarray = field(default_factory=lambda: np.zeros(0))

    ev: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nv: np.ndarray = field(default_factory=lambda: np.zeros(0))
    amp: dict[str, np.ndarray] = field(default_factory=dict)   # mean component spectra

    # Per-window peak, which drives both the σ(f0) of the SESAME test and the
    # time-stability panel.
    window_f0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    window_a0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    window_times: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Azimuthal analysis (empty when it was not requested).
    azimuths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hv_azimuth: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    windows: WindowSet | None = None
    fs: float = 0.0
    start: float = float("nan")
    params: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def n_windows(self) -> int:
        return int(self.hv_windows.shape[1]) if self.hv_windows.size else 0

    @property
    def n_ok(self) -> int:
        return int(np.count_nonzero(self.ok))

    @property
    def statistics(self) -> str:
        return str(self.params.get("statistics", "lognormal"))

    def kept(self) -> np.ndarray:
        """The kept windows' curves, ``(nf, n_ok)``."""
        return self.hv_windows[:, self.ok]

    def band(self, fmin: float = 0.0, fmax: float = np.inf) -> np.ndarray:
        return (self.freq >= fmin) & (self.freq <= fmax)

    def recompute_statistics(self) -> None:
        """Redo the mean curve after the kept-window set changed."""
        kept = self.kept()
        self.hv, self.hv_std, self.hv_lo, self.hv_hi = _statistics(
            kept, self.statistics)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "sid": np.array(self.sid),
            "freq": self.freq.astype(np.float64),
            "hv_windows": self.hv_windows.astype(np.float32),
            "ok": self.ok,
            "hv": self.hv, "hv_std": self.hv_std,
            "hv_lo": self.hv_lo, "hv_hi": self.hv_hi,
            "ev": self.ev, "nv": self.nv,
            "window_f0": self.window_f0, "window_a0": self.window_a0,
            "window_times": self.window_times,
            "azimuths": self.azimuths,
            "hv_azimuth": self.hv_azimuth.astype(np.float32),
            "fs": np.array(self.fs), "start": np.array(self.start),
            "params": np.array(json.dumps(self.params)),
            "meta": np.array(json.dumps(self.meta, default=str)),
        }
        for comp, arr in self.amp.items():
            payload[f"amp_{comp}"] = arr.astype(np.float32)
        if self.windows is not None:
            payload["win_idx"] = self.windows.idx
            payload["win_ratio"] = self.windows.ratio
            payload["win_manual"] = self.windows.manual
            payload["win_meta"] = np.array(json.dumps({
                "fs": self.windows.fs, "width_s": self.windows.width_s,
                "overlap_pc": self.windows.overlap_pc, "start": self.windows.start,
            }))
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HVSRResult":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            res = cls(
                sid=str(z["sid"]),
                freq=z["freq"],
                hv_windows=z["hv_windows"].astype(np.float64),
                ok=z["ok"].astype(bool),
                hv=z["hv"], hv_std=z["hv_std"], hv_lo=z["hv_lo"], hv_hi=z["hv_hi"],
                ev=z["ev"], nv=z["nv"],
                window_f0=z["window_f0"], window_a0=z["window_a0"],
                window_times=z["window_times"],
                azimuths=z["azimuths"],
                hv_azimuth=z["hv_azimuth"].astype(np.float64),
                fs=float(z["fs"]), start=float(z["start"]),
                params=json.loads(str(z["params"])),
                meta=json.loads(str(z["meta"])),
            )
            res.amp = {k[4:]: z[k].astype(np.float64)
                       for k in z.files if k.startswith("amp_")}
            if "win_idx" in z.files:
                wm = json.loads(str(z["win_meta"]))
                res.windows = WindowSet(
                    idx=z["win_idx"], ok=res.ok.copy(), fs=wm["fs"],
                    width_s=wm["width_s"], overlap_pc=wm["overlap_pc"],
                    start=wm["start"], ratio=z["win_ratio"],
                    manual=z["win_manual"].astype(bool))
        return res


def load_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read just the mean curve and its band from a stored result.

    ``np.load`` on an npz is lazy per array, so this touches a few kilobytes
    where :meth:`HVSRResult.load` would decompress the whole per-window matrix.
    That is the difference between drawing a 102-site gallery in a second and
    in a minute.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        return z["freq"], z["hv"], z["hv_std"]


def compute(data: dict[str, np.ndarray], fs: float, params, *,
            sid: str = "", start: float = float("nan"),
            windows: WindowSet | None = None,
            progress: Progress | None = None) -> HVSRResult:
    """Run the full chain on one three-component recording.

    *params* is a :class:`~hvsrlab.project.ProcParams`. *data* holds the raw
    ``Z``/``N``/``E`` traces at *fs*. Pass *windows* to reuse an existing
    window set — including the user's manual keep/reject edits — instead of
    rebuilding it.
    """
    def report(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, msg)

    for comp in ("Z", "N", "E"):
        if comp not in data:
            raise ValueError(f"component {comp} missing")
    npts = int(min(len(v) for v in data.values()))
    if npts < 2:
        raise ValueError("recording is empty")

    # 1 — pre-filter -------------------------------------------------------
    report(0.02, "filtering")
    working = data
    if params.filter_kind != "off":
        filtered = filters.apply_all(
            data, params.filter_kind, params.filter_order, fs,
            params.filter_fmin, params.filter_fmax)
        working = filtered if params.filter_target == "hvsr" else data
        antitrigger_data = filtered
    else:
        antitrigger_data = data

    # 2 — windows ----------------------------------------------------------
    report(0.08, "windowing")
    if windows is None:
        windows = win_mod.make_windows(
            npts, fs, params.window_width_s, params.window_overlap_pc, start)
        if params.antitrigger:
            win_mod.sta_lta_mask(
                antitrigger_data, windows, sta_s=params.sta_s,
                lta_s=params.lta_s, threshold=params.sta_lta_ratio)
    if windows.n_ok < max(1, int(params.min_windows)):
        raise ValueError(
            f"only {windows.n_ok} window(s) survived the anti-trigger; "
            f"{params.min_windows} required — relax STA/LTA or widen the record")

    # 3 — spectra ----------------------------------------------------------
    npad = win_mod.pad_length(windows.width_samples, params.pad_to)
    f_full = spec_mod.frequency_axis(npad, fs)
    fmax = min(params.freq_max, 0.5 * fs)
    f_out = spec_mod.output_grid(f_full, params.freq_min, fmax,
                                 grid=params.freq_grid, n=params.n_freq)
    K = spec_mod.smoothing_matrix(f_full, f_out, params.smoothing_kind,
                                  params.smoothing_b)

    amp: dict[str, np.ndarray] = {}
    complex_h: dict[str, np.ndarray] = {}
    want_azimuth = float(params.azimuth_step_deg) > 0
    for i, comp in enumerate(("Z", "N", "E")):
        report(0.15 + 0.15 * i, f"spectra {comp}")
        w = win_mod.extract(working[comp][:npts], windows,
                            taper_pc=params.taper_pc)
        if want_azimuth and comp in ("N", "E"):
            ft = spec_mod.complex_spectra(w, npad)
            complex_h[comp] = ft
            amp[comp] = spec_mod.apply_smoothing(np.abs(ft), K)
        else:
            amp[comp] = spec_mod.apply_smoothing(
                spec_mod.amplitude_spectra(w, npad), K)

    # 4 — ratios -----------------------------------------------------------
    report(0.62, "spectral ratios")
    v = amp["Z"]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = combine_horizontals(amp["E"], amp["N"], params.hvsr_strategy)
        hv_windows = np.where(v > 0, h / v, np.nan)
        ev_windows = np.where(v > 0, amp["E"] / v, np.nan)
        nv_windows = np.where(v > 0, amp["N"] / v, np.nan)

    ok = windows.ok.copy()
    stat = params.statistics
    hv, hv_std, hv_lo, hv_hi = _statistics(hv_windows[:, ok], stat)
    ev, _, _, _ = _statistics(ev_windows[:, ok], stat)
    nv, _, _, _ = _statistics(nv_windows[:, ok], stat)

    # 5 — per-window peaks --------------------------------------------------
    report(0.75, "window peaks")
    wf0, wa0 = _window_peaks(f_out, hv_windows)

    result = HVSRResult(
        sid=sid, freq=f_out, hv_windows=hv_windows, ok=ok,
        hv=hv, hv_std=hv_std, hv_lo=hv_lo, hv_hi=hv_hi,
        ev=ev, nv=nv,
        amp={c: _mean_curve(a[:, ok], stat) for c, a in amp.items()},
        window_f0=wf0, window_a0=wa0,
        window_times=windows.centre_epochs(),
        windows=windows, fs=float(fs), start=float(start),
        params=_params_dict(params),
        meta={"npad": int(npad), "df": float(f_full[1] - f_full[0]),
              "npts": npts, "n_windows": windows.n, "n_ok": int(ok.sum())},
    )

    # 6 — azimuthal ---------------------------------------------------------
    if want_azimuth:
        report(0.85, "azimuthal analysis")
        result.azimuths, result.hv_azimuth = _azimuthal(
            complex_h, v, K, float(params.azimuth_step_deg), ok, stat)

    report(1.0, "done")
    return result


def _params_dict(params) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass
    return asdict(params) if is_dataclass(params) else dict(params)


def _statistics(curves: np.ndarray, kind: str
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean curve, spread, and the ±1σ band, over the window axis.

    ``lognormal`` returns the geometric mean and a multiplicative band — the
    right choice for a ratio of positive quantities, and the convention the
    SESAME guidelines use for σ. ``linear`` returns ProTO's arithmetic mean
    with an additive band.
    """
    if curves.size == 0:
        empty = np.zeros(0)
        return empty, empty, empty, empty

    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "lognormal":
            logs = np.log(np.where(curves > 0, curves, np.nan))
            mu = np.nanmean(logs, axis=1)
            sd = np.nanstd(logs, axis=1, ddof=1) if curves.shape[1] > 1 \
                else np.zeros(curves.shape[0])
            mean = np.exp(mu)
            return mean, sd, mean * np.exp(-sd), mean * np.exp(sd)

        mean = np.nanmean(curves, axis=1)
        sd = np.nanstd(curves, axis=1, ddof=1) if curves.shape[1] > 1 \
            else np.zeros(curves.shape[0])
        return mean, sd, mean - sd, mean + sd


def _mean_curve(curves: np.ndarray, kind: str) -> np.ndarray:
    return _statistics(curves, kind)[0]


def _window_peaks(freq: np.ndarray, curves: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Peak frequency and amplitude of every window's own H/V curve."""
    if curves.size == 0:
        return np.zeros(0), np.zeros(0)
    filled = np.where(np.isfinite(curves), curves, -np.inf)
    idx = np.argmax(filled, axis=0)
    a0 = filled[idx, np.arange(curves.shape[1])]
    f0 = freq[idx]
    bad = ~np.isfinite(a0)
    f0 = np.where(bad, np.nan, f0)
    a0 = np.where(bad, np.nan, a0)
    return f0, a0


def _azimuthal(complex_h: dict[str, np.ndarray], v: np.ndarray,
               K: np.ndarray | None, step_deg: float, ok: np.ndarray,
               stat: str) -> tuple[np.ndarray, np.ndarray]:
    """H/V as a function of horizontal azimuth.

    Azimuth is measured clockwise from north, so 0° is N–S and 90° is E–W, and
    only 0–180° is computed because the rotated horizontal at θ and θ+180°
    differ by a sign that the amplitude spectrum discards.

    The rotation is applied to the *complex* spectra rather than the traces:
    ``F(N·cosθ + E·sinθ) = cosθ·F(N) + sinθ·F(E)``, so all azimuths come from
    one pair of transforms.
    """
    angles = np.arange(0.0, 180.0, max(1.0, float(step_deg)))
    fn, fe = complex_h["N"], complex_h["E"]
    out = np.zeros((v.shape[0], angles.size))
    for i, theta in enumerate(angles):
        rad = np.radians(theta)
        rotated = np.abs(np.cos(rad) * fn + np.sin(rad) * fe)
        if K is not None:
            rotated = K @ rotated
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(v > 0, rotated / v, np.nan)
        out[:, i] = _mean_curve(ratio[:, ok], stat)
    return angles, out


def preferred_direction(result: HVSRResult) -> dict[str, float]:
    """Azimuth of maximum H/V at the picked peak, and how directional it is.

    ``contrast`` is (max − min)/mean across azimuth at f0: below roughly 0.2
    the resonance is effectively isotropic, and a "preferred direction" read
    off the polar plot would be noise.
    """
    if result.hv_azimuth.size == 0 or not result.azimuths.size:
        return {}
    f0 = float(result.meta.get("f0", np.nan))
    if not np.isfinite(f0):
        i = int(np.nanargmax(result.hv))
    else:
        i = int(np.argmin(np.abs(result.freq - f0)))
    line = result.hv_azimuth[i]
    if not np.isfinite(line).any():
        return {}
    imax = int(np.nanargmax(line))
    imin = int(np.nanargmin(line))
    mean = float(np.nanmean(line))
    return {
        "frequency": float(result.freq[i]),
        "azimuth_max": float(result.azimuths[imax]),
        "amplitude_max": float(line[imax]),
        "azimuth_min": float(result.azimuths[imin]),
        "amplitude_min": float(line[imin]),
        "contrast": float((line[imax] - line[imin]) / mean) if mean else float("nan"),
    }
