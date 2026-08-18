"""Turning scattered site results into maps, sections and volumes.

Three things happen here:

* **Gridding** — scattered (x, y, value) triples become a regular mesh, with a
  mask so the interpolator does not draw confident contours out where there are
  no measurements. That masking is the honest part: ``griddata`` will happily
  extrapolate across a 5 km gap, and a bedrock map that does so is a drawing,
  not data.
* **Sections** — a profile line is defined on the map, nearby sites are
  projected onto it, and each site's whole H/V curve becomes a vertical column.
  The frequency axis maps to a pseudo-depth axis through the same power law the
  bedrock depths use, which is what makes the section a section rather than a
  stack of curves.
* **Volumes** — the same construction on a 3D mesh, for the isosurface views.

The section smoothing and normalisation strategies are ports of ProTO's
``prfsmoothing.m`` and its profile normalisation menu, kept name-for-name so
figures made with either program are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Section smoothing strategies, in ProTO's menu order.
SMOOTHING_STRATEGIES = ("off", "layer", "broad_layer", "bubble")

#: Section normalisation strategies, in ProTO's menu order.
NORMALISATIONS = ("off", "at_main_peak", "max_all_stations", "max_in_profile")


# ---------------------------------------------------------------------------
# Gridding
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    """A regular 2D mesh of one quantity."""

    x: np.ndarray          # (ny, nx)
    y: np.ndarray          # (ny, nx)
    z: np.ndarray          # (ny, nx), NaN outside the mask
    name: str = ""

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (float(self.x.min()), float(self.x.max()),
                float(self.y.min()), float(self.y.max()))

    def finite(self) -> np.ndarray:
        return self.z[np.isfinite(self.z)]


def interpolate(x, y, values, *, nx: int = 160, ny: int = 160,
                method: str = "linear", pad: float = 0.05,
                mask: str = "hull", mask_radius: float = 0.0,
                hull_expand: float = 0.0) -> Grid:
    """Grid scattered values.

    *method* is ``linear``, ``cubic``, ``nearest`` or ``rbf`` (a thin-plate
    spline, which extrapolates smoothly and is the right choice for sparse but
    well-distributed sites).

    *mask* controls where the result is trusted:

    ``hull``      inside the convex hull of the sites, optionally grown by
                  *hull_expand* in map units — the default, and the honest one
    ``radius``    within *mask_radius* of at least one site
    ``none``      no mask; the interpolator's full extrapolation is shown
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[good], y[good], v[good]
    if x.size < 3:
        raise ValueError(f"need at least 3 sites with values, got {x.size}")

    dx = (x.max() - x.min()) or 1.0
    dy = (y.max() - y.min()) or 1.0
    gx = np.linspace(x.min() - pad * dx, x.max() + pad * dx, int(nx))
    gy = np.linspace(y.min() - pad * dy, y.max() + pad * dy, int(ny))
    X, Y = np.meshgrid(gx, gy)

    Z = _interpolate_to(X, Y, x, y, v, method)

    if mask == "hull":
        keep = hull_mask(X, Y, x, y, expand=hull_expand)
        Z = np.where(keep, Z, np.nan)
    elif mask == "radius":
        radius = mask_radius or _typical_spacing(x, y) * 1.5
        Z = np.where(distance_mask(X, Y, x, y, radius), Z, np.nan)

    return Grid(x=X, y=Y, z=Z)


def _interpolate_to(X, Y, x, y, v, method: str) -> np.ndarray:
    from scipy.interpolate import griddata

    if method == "rbf":
        from scipy.interpolate import RBFInterpolator
        rbf = RBFInterpolator(np.column_stack([x, y]), v, kernel="thin_plate_spline",
                             smoothing=0.0)
        return rbf(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)

    Z = griddata((x, y), v, (X, Y), method=method)
    if method in ("linear", "cubic"):
        # griddata leaves NaN outside the triangulation; nearest fills it so a
        # subsequent mask decides what is shown, not the triangulation's edge.
        holes = ~np.isfinite(Z)
        if holes.any():
            Z[holes] = griddata((x, y), v, (X[holes], Y[holes]), method="nearest")
    return Z


def hull_mask(X, Y, x, y, expand: float = 0.0) -> np.ndarray:
    """True inside the convex hull of the sites, grown by *expand* map units."""
    from scipy.spatial import ConvexHull, Delaunay

    points = np.column_stack([x, y])
    if points.shape[0] < 3:
        return np.ones(X.shape, dtype=bool)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.ones(X.shape, dtype=bool)

    vertices = points[hull.vertices]
    if expand:
        centre = vertices.mean(axis=0)
        offsets = vertices - centre
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        vertices = vertices + offsets / np.maximum(norms, 1e-12) * expand

    tri = Delaunay(vertices)
    flat = np.column_stack([X.ravel(), Y.ravel()])
    return (tri.find_simplex(flat) >= 0).reshape(X.shape)


def distance_mask(X, Y, x, y, radius: float) -> np.ndarray:
    """True where a site lies within *radius*."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([x, y]))
    d, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
    return (d.reshape(X.shape) <= radius)


def _typical_spacing(x, y) -> float:
    from scipy.spatial import cKDTree

    points = np.column_stack([x, y])
    if points.shape[0] < 2:
        return 1.0
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2)
    return float(np.median(d[:, 1]))


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass
class ProfileProjection:
    """Sites projected onto a profile line."""

    indices: np.ndarray        # index into the site list, ordered along the line
    distance: np.ndarray       # metres along the profile from its first point
    offset: np.ndarray         # perpendicular distance, signed
    length: float = 0.0


def project_to_profile(x1: float, y1: float, x2: float, y2: float,
                       x, y, *, width: float = 0.0) -> ProfileProjection:
    """Project sites onto the segment (x1,y1)–(x2,y2).

    *width* is the half-corridor: sites further than this from the line are
    dropped. Zero keeps everything, which is ProTO's behaviour, and is usually
    what you want for a sparse survey where no site sits exactly on the line.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        raise ValueError("profile endpoints coincide")
    ux, uy = dx / length, dy / length

    rx, ry = x - x1, y - y1
    along = rx * ux + ry * uy
    across = -rx * uy + ry * ux

    keep = np.isfinite(along)
    if width > 0:
        keep &= np.abs(across) <= width
    idx = np.where(keep)[0]
    order = np.argsort(along[idx])
    idx = idx[order]
    return ProfileProjection(indices=idx, distance=along[idx],
                             offset=across[idx], length=length)


def build_section(distance: np.ndarray, freq: np.ndarray,
                  curves: list[np.ndarray], *, depth_axis: np.ndarray | None = None,
                  n_nodes: int = 200, n_depth: int = 240,
                  smoothing: str = "off", smoothing_radius: int = 2,
                  normalisation: str = "off",
                  peak_amplitudes: np.ndarray | None = None,
                  global_max: float = float("nan")) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble a pseudo-section from the H/V curves of projected sites.

    Each site contributes one column: its H/V amplitude against *depth_axis*
    (or against frequency, when no depth axis is given). Columns are then
    interpolated laterally onto *n_nodes* evenly spaced positions.

    Returns ``(distances, vertical_axis, section)`` with ``section`` shaped
    ``(n_vertical, n_nodes)``.
    """
    if not curves:
        raise ValueError("no curves to section")

    vertical = depth_axis if depth_axis is not None else freq
    n_v = int(n_depth if depth_axis is not None else min(len(freq), n_depth))
    grid_v = (np.linspace(vertical.min(), vertical.max(), n_v)
              if depth_axis is not None else np.asarray(freq, dtype=float))

    columns = np.full((grid_v.size, len(curves)), np.nan)
    for j, curve in enumerate(curves):
        c = np.asarray(curve, dtype=float)
        if depth_axis is None:
            columns[:, j] = np.interp(grid_v, freq, c, left=np.nan, right=np.nan)
        else:
            # depth increases as frequency decreases: sort before interpolating
            order = np.argsort(vertical)
            columns[:, j] = np.interp(grid_v, vertical[order], c[order],
                                      left=np.nan, right=np.nan)

    columns = _normalise(columns, normalisation, peak_amplitudes, global_max)

    grid_d = np.linspace(float(distance.min()), float(distance.max()), int(n_nodes))
    section = np.full((grid_v.size, grid_d.size), np.nan)
    for i in range(grid_v.size):
        row = columns[i]
        finite = np.isfinite(row)
        if finite.sum() >= 2:
            section[i] = np.interp(grid_d, distance[finite], row[finite])
        elif finite.sum() == 1:
            section[i] = row[finite][0]

    if smoothing != "off":
        section = smooth_section(section, smoothing, smoothing_radius)
    return grid_d, grid_v, section


def _normalise(columns: np.ndarray, kind: str,
               peak_amplitudes: np.ndarray | None,
               global_max: float) -> np.ndarray:
    """ProTO's four profile normalisations."""
    if kind == "off":
        return columns
    out = columns.copy()
    if kind == "at_main_peak" and peak_amplitudes is not None:
        scale = np.asarray(peak_amplitudes, dtype=float)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
        return out / scale[None, :]
    if kind == "max_all_stations" and np.isfinite(global_max) and global_max > 0:
        return out / global_max
    if kind == "max_in_profile":
        peak = np.nanmax(out) if np.isfinite(out).any() else np.nan
        if np.isfinite(peak) and peak > 0:
            return out / peak
    return out


def smooth_section(section: np.ndarray, strategy: str, radius: int = 2
                   ) -> np.ndarray:
    """Port of ProTO's ``prfsmoothing.m``.

    ``layer`` averages horizontally within each depth row, ``broad_layer`` adds
    one row either side, ``bubble`` uses a square neighbourhood. The averages
    ignore NaNs, so an edge column does not pull the section towards zero.
    """
    if strategy == "off" or radius < 1:
        return section

    nz, nx = section.shape
    out = np.full_like(section, np.nan)
    kz = {"layer": 0, "broad_layer": 1, "bubble": int(radius)}.get(strategy, 0)

    for k in range(nz):
        k0, k1 = max(0, k - kz), min(nz, k + kz + 1)
        for i in range(nx):
            i0, i1 = max(0, i - radius), min(nx, i + radius + 1)
            block = section[k0:k1, i0:i1]
            if np.isfinite(block).any():
                out[k, i] = np.nanmean(block)
    return out


def frequency_to_depth(freq: np.ndarray, a: float, b: float) -> np.ndarray:
    """Map a frequency axis to pseudo-depth with ``H = a·f^b``.

    Applying the bedrock law to *every* frequency, not just the peak, is what
    turns an H/V curve into a depth column. It is a coordinate change, not a
    velocity model: away from f0 the mapping has no physical warrant, and deep
    parts of a section built this way should be read as pattern, not structure.
    """
    f = np.asarray(freq, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(f > 0, a * np.power(f, b), np.nan)


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

def build_volume(x, y, curves: list[np.ndarray], depths: list[np.ndarray], *,
                 nx: int = 40, ny: int = 40, nz: int = 60,
                 zmax: float = 0.0, method: str = "linear",
                 mask: str = "hull") -> tuple[np.ndarray, ...]:
    """Interpolate site H/V columns into a 3D block.

    Each site supplies amplitude against depth; the block is filled depth slice
    by depth slice, each slice gridded and masked exactly as a map would be.
    Returns ``(X, Y, Z, V)`` on a ``(nz, ny, nx)`` mesh.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(curves) != x.size:
        raise ValueError("one curve per site is required")

    top = 0.0
    bottom = zmax or float(np.nanmax([np.nanmax(d) for d in depths]))
    zs = np.linspace(top, bottom, int(nz))

    resampled = np.full((zs.size, x.size), np.nan)
    for j, (curve, depth) in enumerate(zip(curves, depths)):
        d = np.asarray(depth, dtype=float)
        c = np.asarray(curve, dtype=float)
        good = np.isfinite(d) & np.isfinite(c)
        if good.sum() < 2:
            continue
        order = np.argsort(d[good])
        resampled[:, j] = np.interp(zs, d[good][order], c[good][order],
                                    left=np.nan, right=np.nan)

    X = Y = None
    V = None
    for k in range(zs.size):
        row = resampled[k]
        finite = np.isfinite(row)
        if finite.sum() < 3:
            continue
        grid = interpolate(x[finite], y[finite], row[finite],
                           nx=nx, ny=ny, method=method, mask=mask)
        if V is None:
            X, Y = grid.x, grid.y
            V = np.full((zs.size,) + grid.z.shape, np.nan)
        V[k] = grid.z

    if V is None:
        raise ValueError("no depth slice had enough sites to interpolate")
    Z = np.repeat(zs[:, None, None], X.shape[0], axis=1)
    Z = np.repeat(Z, X.shape[1], axis=2)
    return X, Y, Z, V
