"""1D layered-earth modelling of the resonance.

The forward operator is the classical SH transfer function of a stack of
horizontal layers over a half-space for vertically incident shear waves — the
Haskell–Thomson recursion in the form given by Kramer (1996, §7.2). Its poles
are the resonance frequencies of the soil column, so it predicts **where** the
H/V peak sits from a Vs/thickness model, and inverting it turns a measured f0
into a depth without borrowing another basin's empirical power law.

What it does not do is predict H/V **amplitude**. H/V is a property of the
ambient wavefield — dominated by Rayleigh-wave ellipticity, with a
body-wave contribution whose share is unknown at any given site — while this is
the response of the column to a plane SH wave. The two share their peak
frequency, which is the part the interpretation rests on, and differ in
amplitude, sometimes by a factor of two or more. So: fit the frequency, read
the amplitude as indicative, and say so in the report.

Everything here is in SI: thickness in metres, velocity in m/s, density in
kg/m³, damping as a fraction of critical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Layer:
    thickness: float          # metres; ignored for the half-space
    vs: float                 # m/s
    density: float = 1900.0   # kg/m³
    damping: float = 0.02     # fraction of critical

    def copy(self) -> "Layer":
        return Layer(self.thickness, self.vs, self.density, self.damping)


@dataclass
class Model:
    """A soil column: one or more layers over a half-space.

    The last entry is the half-space; its thickness is not used.
    """

    layers: list[Layer] = field(default_factory=list)
    name: str = ""

    @classmethod
    def two_layer(cls, thickness: float, vs_soil: float,
                  vs_rock: float = 1200.0) -> "Model":
        return cls([Layer(thickness, vs_soil), Layer(0.0, vs_rock, 2400.0, 0.005)],
                   name="two-layer")

    @property
    def n(self) -> int:
        return len(self.layers)

    @property
    def total_thickness(self) -> float:
        return float(sum(l.thickness for l in self.layers[:-1]))

    @property
    def average_vs(self) -> float:
        """Travel-time (slowness-weighted) average Vs through the cover.

        The harmonic mean, not the arithmetic one: it is the average that
        preserves the vertical travel time, and therefore the resonance.
        """
        cover = self.layers[:-1]
        total = sum(l.thickness for l in cover)
        if total <= 0:
            return float("nan")
        time = sum(l.thickness / l.vs for l in cover if l.vs > 0)
        return total / time if time > 0 else float("nan")

    @property
    def f0_quarter_wavelength(self) -> float:
        """``Vs_avg / (4·H)`` — exact for one layer, a good guess for a few."""
        h = self.total_thickness
        v = self.average_vs
        return v / (4.0 * h) if h > 0 and np.isfinite(v) else float("nan")

    def copy(self) -> "Model":
        return Model([l.copy() for l in self.layers], self.name)

    def as_table(self) -> np.ndarray:
        return np.array([[l.thickness, l.vs, l.density, l.damping]
                         for l in self.layers], dtype=float)

    @classmethod
    def from_table(cls, table, name: str = "") -> "Model":
        rows = np.atleast_2d(np.asarray(table, dtype=float))
        return cls([Layer(*row[:4]) for row in rows], name)


def transfer_function(model: Model, freq: np.ndarray, *,
                      reference: str = "outcrop") -> np.ndarray:
    """Amplitude of the SH transfer function of *model* at *freq*.

    ``reference="outcrop"`` gives surface motion relative to the motion that
    would be recorded on the outcropping half-space — the amplification an
    engineer wants. ``reference="within"`` gives it relative to the motion at
    the top of the half-space beneath the column, which is larger by the
    downgoing wave.
    """
    freq = np.asarray(freq, dtype=float)
    omega = 2.0 * np.pi * np.maximum(freq, 1e-12)

    # Complex velocity carries the damping: Vs* = Vs·(1 + iξ) to first order.
    vs = np.array([l.vs for l in model.layers], dtype=float)
    xi = np.array([l.damping for l in model.layers], dtype=float)
    rho = np.array([l.density for l in model.layers], dtype=float)
    h = np.array([l.thickness for l in model.layers], dtype=float)
    vstar = vs * (1.0 + 1j * xi)

    a = np.ones_like(omega, dtype=complex)     # upgoing amplitude at layer top
    b = np.ones_like(omega, dtype=complex)     # downgoing
    for j in range(model.n - 1):
        alpha = (rho[j] * vstar[j]) / (rho[j + 1] * vstar[j + 1])
        k = omega / vstar[j]
        phase = 1j * k * h[j]
        ep, em = np.exp(phase), np.exp(-phase)
        a, b = (0.5 * a * (1.0 + alpha) * ep + 0.5 * b * (1.0 - alpha) * em,
                0.5 * a * (1.0 - alpha) * ep + 0.5 * b * (1.0 + alpha) * em)

    if reference == "within":
        return np.abs(2.0 / (a + b))
    return np.abs(1.0 / a)


def resonances(model: Model, fmin: float = 0.1, fmax: float = 25.0,
               n: int = 4000) -> list[tuple[float, float]]:
    """Peak frequencies and amplifications of the transfer function."""
    from scipy.signal import find_peaks

    freq = np.linspace(max(fmin, 1e-3), fmax, n)
    tf = transfer_function(model, freq)
    idx, _ = find_peaks(tf)
    return [(float(freq[i]), float(tf[i])) for i in idx]


def fundamental(model: Model, fmin: float = 0.05, fmax: float = 50.0) -> float:
    """The lowest resonance of the column."""
    peaks = resonances(model, fmin, fmax)
    return peaks[0][0] if peaks else float("nan")


def depth_for_f0(f0: float, vs_profile: Model) -> float:
    """Scale the cover thickness of *vs_profile* so its f0 matches the target.

    The fundamental of a layered column scales exactly inversely with a uniform
    stretch of all thicknesses, so one division does what a search would.
    """
    current = fundamental(vs_profile)
    if not (np.isfinite(current) and current > 0 and f0 > 0):
        return float("nan")
    return vs_profile.total_thickness * (current / f0)


@dataclass
class InversionResult:
    model: Model
    misfit: float
    f0_model: float
    f0_observed: float
    frequencies: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    n_evaluations: int = 0
    message: str = ""


def invert(freq: np.ndarray, hv: np.ndarray, *, n_layers: int = 1,
           vs_bounds: tuple[float, float] = (80.0, 900.0),
           thickness_bounds: tuple[float, float] = (1.0, 400.0),
           vs_halfspace: float = 1500.0,
           fmin: float = 0.0, fmax: float = np.inf,
           weight_shape: float = 0.3,
           seed: int = 0, maxiter: int = 120) -> InversionResult:
    """Fit a layered model whose SH resonance reproduces the observed H/V peak.

    The misfit is dominated by the peak frequency — a relative error on f0 —
    with *weight_shape* of it coming from the normalised curve shape over the
    band. Frequency is what the transfer function is entitled to predict; the
    shape term only breaks ties between models that fit f0 equally well, which
    is why its weight is small by default. Set ``weight_shape=0`` to fit the
    peak alone.

    An f0-only fit is fundamentally non-unique: thickness and velocity trade
    off exactly, and only their ratio is resolved. Fix one from independent
    information — a borehole, or the Vs model from ambient-noise tomography —
    and read the other. The returned model is one member of that family, not
    the answer.
    """
    from scipy.optimize import differential_evolution

    freq = np.asarray(freq, dtype=float)
    hv = np.asarray(hv, dtype=float)
    band = (freq >= fmin) & (freq <= fmax) & np.isfinite(hv) & (hv > 0)
    if band.sum() < 8:
        raise ValueError("not enough finite H/V samples in the band to invert")
    f = freq[band]
    obs = hv[band]

    i0 = int(np.argmax(obs))
    f0_obs = float(f[i0])
    obs_norm = obs / obs[i0]

    n_layers = max(1, int(n_layers))
    bounds = []
    for _ in range(n_layers):
        bounds.append((np.log(thickness_bounds[0]), np.log(thickness_bounds[1])))
        bounds.append((np.log(vs_bounds[0]), np.log(vs_bounds[1])))

    evaluations = 0

    def build(x: np.ndarray) -> Model:
        layers = []
        for i in range(n_layers):
            layers.append(Layer(float(np.exp(x[2 * i])),
                                float(np.exp(x[2 * i + 1]))))
        layers.append(Layer(0.0, vs_halfspace, 2400.0, 0.005))
        return Model(layers)

    def misfit(x: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        model = build(x)
        # Velocity must increase downward, or the "resonance" is spurious.
        vs = [l.vs for l in model.layers]
        if any(v2 < v1 for v1, v2 in zip(vs, vs[1:])):
            return 1e6

        tf = transfer_function(model, f)
        j = int(np.argmax(tf))
        f0_mod = float(f[j])
        err_f0 = abs(np.log(f0_mod / f0_obs))
        if weight_shape <= 0:
            return err_f0
        pred = tf / tf[j]
        err_shape = float(np.sqrt(np.mean((np.log(pred) - np.log(obs_norm)) ** 2)))
        return (1.0 - weight_shape) * err_f0 + weight_shape * err_shape

    result = differential_evolution(
        misfit, bounds, seed=seed, maxiter=maxiter, tol=1e-6,
        polish=True, init="sobol", updating="deferred")

    model = build(result.x)
    tf = transfer_function(model, f)
    return InversionResult(
        model=model, misfit=float(result.fun),
        f0_model=float(f[int(np.argmax(tf))]), f0_observed=f0_obs,
        frequencies=f, predicted=tf, observed=obs,
        n_evaluations=evaluations, message=str(result.message))


def from_vs_profile(depths, vs, *, density: float = 1900.0,
                    damping: float = 0.02, vs_halfspace: float | None = None
                    ) -> Model:
    """Build a model from a sampled Vs(z) profile — e.g. an ANT inversion result.

    *depths* are layer-interface depths (increasing) and *vs* the velocity of
    the layer above each. The half-space takes the deepest velocity unless
    *vs_halfspace* says otherwise.
    """
    depths = np.asarray(depths, dtype=float)
    vs = np.asarray(vs, dtype=float)
    if depths.size != vs.size:
        raise ValueError("depths and vs must have the same length")

    thicknesses = np.diff(np.concatenate([[0.0], depths]))
    layers = [Layer(float(t), float(v), density, damping)
              for t, v in zip(thicknesses, vs) if t > 0]
    layers.append(Layer(0.0, float(vs_halfspace if vs_halfspace else vs[-1]),
                        2400.0, 0.005))
    return Model(layers, name="from Vs profile")
