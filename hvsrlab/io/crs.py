"""Map projection: geographic coordinates to UTM metres and back.

Everything downstream of this module — distances between sites, profile
lengths, interpolation grids, the ProTO export — works in metres on a plane.
UTM is the right plane for a passive-seismic survey: it is metric, it is what
field GPS and GIS already speak, and its distortion over the few tens of
kilometres a deployment spans is under 1 m/km.

The zone is chosen from the survey's own centroid rather than asked for, and
recorded on the project so every later session lands on the same grid. It can
be overridden, which matters for a survey that straddles a zone boundary:
forcing one zone across the join keeps the geometry continuous, at the cost of
a little more distortion on the far side.

``pyproj`` does the work when it is installed. When it is not, a direct
implementation of the Karney/USGS transverse-Mercator series takes over — good
to a few millimetres within a zone, which is far below the precision of a
handheld GPS fix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: WGS 84.
_A = 6378137.0
_F = 1.0 / 298.257223563
_K0 = 0.9996
_FALSE_EASTING = 500000.0
_FALSE_NORTHING = 10000000.0        # southern hemisphere only


@dataclass(frozen=True)
class UTM:
    """A UTM zone."""

    zone: int
    south: bool

    @property
    def epsg(self) -> int:
        return (32700 if self.south else 32600) + self.zone

    @property
    def name(self) -> str:
        return f"UTM zone {self.zone}{'S' if self.south else 'N'}"

    @property
    def label(self) -> str:
        return f"{self.name} (EPSG:{self.epsg})"

    def to_dict(self) -> dict:
        return {"kind": "utm", "zone": self.zone, "south": self.south,
                "epsg": self.epsg}

    @classmethod
    def from_dict(cls, data: dict | None) -> "UTM | None":
        if not data or data.get("kind") != "utm":
            return None
        try:
            return cls(int(data["zone"]), bool(data["south"]))
        except (KeyError, TypeError, ValueError):
            return None


def zone_for(lat: float, lon: float) -> UTM:
    """The standard zone for one position, including the two exceptions.

    Norway widened zone 32 and Svalbard rearranged 31–37; both are in every
    conforming implementation, and a survey that ignored them would disagree
    with the local mapping by hundreds of kilometres.
    """
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    lat = float(lat)
    zone = int((lon + 180.0) / 6.0) + 1

    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32                                    # south-west Norway
    elif 72.0 <= lat < 84.0:                         # Svalbard
        if 0.0 <= lon < 9.0:
            zone = 31
        elif 9.0 <= lon < 21.0:
            zone = 33
        elif 21.0 <= lon < 33.0:
            zone = 35
        elif 33.0 <= lon < 42.0:
            zone = 37
    return UTM(zone=max(1, min(60, zone)), south=lat < 0.0)


def zone_for_survey(lats, lons) -> UTM | None:
    """The zone for a set of positions, from their centroid.

    Sites in several zones all go into the centroid's, so the survey stays on
    one continuous grid. Crossing more than about one zone that way starts to
    cost real accuracy, and :func:`zone_span` reports when that has happened.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    good = np.isfinite(lats) & np.isfinite(lons)
    if not good.any():
        return None
    return zone_for(float(np.mean(lats[good])), float(np.mean(lons[good])))


def zone_span(lats, lons) -> list[int]:
    """Every zone the sites would fall in naturally, sorted."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    good = np.isfinite(lats) & np.isfinite(lons)
    return sorted({zone_for(la, lo).zone
                   for la, lo in zip(lats[good], lons[good])})


# ---------------------------------------------------------------------------
# Forward and inverse
# ---------------------------------------------------------------------------

def to_utm(lats, lons, zone: UTM | None = None) -> tuple[np.ndarray, np.ndarray, UTM]:
    """Project to ``(easting, northing)`` metres in *zone*.

    The zone defaults to the one the positions themselves imply. Positions
    outside it are still projected into it — that is the point of naming one —
    so the result stays a single consistent grid.
    """
    lats = np.atleast_1d(np.asarray(lats, dtype=float))
    lons = np.atleast_1d(np.asarray(lons, dtype=float))
    if zone is None:
        zone = zone_for_survey(lats, lons)
    if zone is None:
        raise ValueError("no usable coordinates to choose a UTM zone from")

    easting = np.full(lats.shape, np.nan)
    northing = np.full(lats.shape, np.nan)
    good = np.isfinite(lats) & np.isfinite(lons)
    if good.any():
        e, n = _forward(lats[good], lons[good], zone)
        easting[good], northing[good] = e, n
    return easting, northing, zone


def from_utm(easting, northing, zone: UTM) -> tuple[np.ndarray, np.ndarray]:
    """Back to ``(latitude, longitude)`` degrees."""
    easting = np.atleast_1d(np.asarray(easting, dtype=float))
    northing = np.atleast_1d(np.asarray(northing, dtype=float))
    lats = np.full(easting.shape, np.nan)
    lons = np.full(easting.shape, np.nan)
    good = np.isfinite(easting) & np.isfinite(northing)
    if good.any():
        la, lo = _inverse(easting[good], northing[good], zone)
        lats[good], lons[good] = la, lo
    return lats, lons


def _transformer(zone: UTM, inverse: bool = False):
    try:
        from pyproj import Transformer
    except ImportError:
        return None
    source, target = "EPSG:4326", f"EPSG:{zone.epsg}"
    if inverse:
        source, target = target, source
    return Transformer.from_crs(source, target, always_xy=True)


def _forward(lats: np.ndarray, lons: np.ndarray, zone: UTM):
    transformer = _transformer(zone)
    if transformer is not None:
        e, n = transformer.transform(lons, lats)
        return np.asarray(e, dtype=float), np.asarray(n, dtype=float)
    return _forward_series(lats, lons, zone)


def _inverse(easting: np.ndarray, northing: np.ndarray, zone: UTM):
    transformer = _transformer(zone, inverse=True)
    if transformer is not None:
        lon, lat = transformer.transform(easting, northing)
        return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return _inverse_series(easting, northing, zone)


# ---------------------------------------------------------------------------
# Fallback: transverse Mercator by series, used only without pyproj
# ---------------------------------------------------------------------------

def _central_meridian(zone: int) -> float:
    return (zone - 1) * 6.0 - 180.0 + 3.0


def _forward_series(lats, lons, zone: UTM):
    e2 = _F * (2 - _F)
    ep2 = e2 / (1 - e2)
    phi = np.radians(lats)
    lam = np.radians(lons - _central_meridian(zone.zone))

    n = _A / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    t = np.tan(phi)
    c = ep2 * np.cos(phi) ** 2
    a = np.cos(phi) * lam

    m = _A * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi)
        - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))

    easting = _K0 * n * (
        a + (1 - t ** 2 + c) * a ** 3 / 6
        + (5 - 18 * t ** 2 + t ** 4 + 72 * c - 58 * ep2) * a ** 5 / 120
    ) + _FALSE_EASTING

    northing = _K0 * (m + n * t * (
        a ** 2 / 2 + (5 - t ** 2 + 9 * c + 4 * c ** 2) * a ** 4 / 24
        + (61 - 58 * t ** 2 + t ** 4 + 600 * c - 330 * ep2) * a ** 6 / 720))
    if zone.south:
        northing = northing + _FALSE_NORTHING
    return easting, northing


def _inverse_series(easting, northing, zone: UTM):
    e2 = _F * (2 - _F)
    ep2 = e2 / (1 - e2)
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    x = easting - _FALSE_EASTING
    y = northing - (_FALSE_NORTHING if zone.south else 0.0)

    m = y / _K0
    mu = m / (_A * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))

    c1 = ep2 * np.cos(phi1) ** 2
    t1 = np.tan(phi1) ** 2
    n1 = _A / np.sqrt(1 - e2 * np.sin(phi1) ** 2)
    r1 = _A * (1 - e2) / (1 - e2 * np.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * _K0)

    lat = phi1 - (n1 * np.tan(phi1) / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2)
        * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2)
           * d ** 5 / 120) / np.cos(phi1)
    return np.degrees(lat), np.degrees(lon) + _central_meridian(zone.zone)


def looks_projected(x: float, y: float) -> bool:
    """True when a coordinate pair is plainly metres, not degrees.

    UTM eastings live in 160 000–840 000 m and northings in 0–10 000 000 m, so
    anything past the ±180 that a longitude can reach is projected already.
    """
    return abs(float(x)) > 180.0 or abs(float(y)) > 180.0


def available() -> str:
    """Which implementation is in use, for the diagnostics report."""
    try:
        import pyproj
        return f"pyproj {pyproj.__version__}"
    except ImportError:
        return "built-in series (pyproj not installed)"
