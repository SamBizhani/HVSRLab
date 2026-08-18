"""Station coordinate lists, plus projection to local plan coordinates.

The reader sniffs the layout rather than assuming one, because the same survey
turns up in several forms. All of these parse::

    SS S001 150.123456 -27.654321 120.00       EGFAnalysisTimeFreq station.txt
    sta lon lat                                 common minimal form
    lon lat                                     coordinate-only
    450000123,2024-01-05,00:00:00,2024-02-15,00:00:00,-27.6543,150.1235,120.00

Separator (comma, tab or spaces) and column order are both detected. Latitude
and longitude are found by plausibility, not position: a numeric token beyond
±180 cannot be a coordinate, which is what distinguishes a station serial like
``450000123`` from a longitude.

The parsing strategy is shared with the sibling ANTAgent application so the
same survey files feed both.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np

#: Splits on commas, tabs or runs of spaces, so CSV and whitespace layouts
#: both work without the caller having to say which it is.
_SPLIT = re.compile(r"[,\t]|\s+")

#: Mean Earth radius, metres — good to 0.5 % for the local tangent plane below.
_R_EARTH = 6371008.8


@dataclass(frozen=True)
class Station:
    name: str
    lat: float
    lon: float
    elev: float = 0.0
    network: str = ""
    #: Set instead of lat/lon when the file was already in metres.
    easting: float = float("nan")
    northing: float = float("nan")

    @property
    def code(self) -> str:
        return f"{self.network}.{self.name}" if self.network else self.name

    @property
    def projected(self) -> bool:
        return math.isnan(self.lat) and not math.isnan(self.easting)


def read(path: str | Path) -> list[Station]:
    """Read every station row that can be recognised in *path*."""
    path = Path(path)
    stations: dict[str, Station] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "%")):
            continue
        parts = [p for p in _SPLIT.split(line) if p]
        station = _parse(parts)
        if station is not None:
            stations.setdefault(station.code, station)  # files often repeat rows
    return list(stations.values())


def _parse(parts: list[str]) -> Station | None:
    """Pull a station out of one row, whatever the column layout."""
    numeric = [(i, float(t)) for i, t in enumerate(parts) if _is_number(t)]
    if len(numeric) < 2:          # header rows have no numbers at all
        return None

    coords = [(i, v) for i, v in numeric if abs(v) <= 180.0]
    pair = None
    for (i1, v1), (i2, v2) in zip(coords, coords[1:]):
        if abs(v1) <= 90.0 and abs(v2) <= 180.0:
            pair = ((i1, v1), (i2, v2), False)   # lat then lon
            break
        if abs(v1) <= 180.0 and abs(v2) <= 90.0:
            pair = ((i1, v1), (i2, v2), True)    # lon then lat
            break

    easting = northing = float("nan")
    if pair is None:
        # No degrees in the row. A file may already be in metres, so look for
        # a pair in the ranges UTM occupies: eastings are 160-840 km from the
        # false origin, northings 0-10 000 km. A station serial number does
        # not sit in both of those at once, which is what makes this safe.
        for (i1, v1), (i2, v2) in zip(numeric, numeric[1:]):
            if 1e5 <= v1 <= 9.99e5 and 0.0 <= v2 <= 1.001e7:
                easting, northing = v1, v2
                i1_, i2_ = i1, i2
                break
        else:
            return None
        elev = next((v for i, v in numeric if i > i2_), 0.0)
        network, name = _identify(parts, i1_, f"{easting:.0f}_{northing:.0f}")
        return Station(name=name, lat=float("nan"), lon=float("nan"),
                       elev=elev, network=network,
                       easting=easting, northing=northing)

    (i1, v1), (i2, v2), lon_first = pair
    lat, lon = (v2, v1) if lon_first else (v1, v2)

    # Elevation is the next numeric after the pair, whatever its magnitude --
    # it is legitimately allowed to exceed 180.
    elev = 0.0
    for i, v in numeric:
        if i > i2:
            elev = v
            break

    network, name = _identify(parts, i1, f"{lat:.4f}_{lon:.4f}")
    return Station(name=name, lat=lat, lon=lon, elev=elev, network=network)


def _identify(parts: list[str], stop: int, fallback: str) -> tuple[str, str]:
    """Split the tokens before the coordinates into a network and a name.

    Numeric tokens count as labels here. A station whose name *is* a number —
    ``SS 450000123 512345 6543210`` — would otherwise lose it to the "that is a
    serial, not a coordinate" rule and every row in the file would come back
    named after its network, collapsing the whole survey into one station.

    The last label is the name and the first is the network, which reads all
    the layouts seen in the field: ``NET NAME …``, ``NAME …``, and bare
    coordinates.
    """
    tokens = [t for t in parts[:stop] if not _looks_temporal(t)]
    if len(tokens) >= 2:
        return tokens[0], tokens[-1]
    if len(tokens) == 1:
        return "", tokens[0]
    return "", fallback


def _looks_temporal(token: str) -> bool:
    """True for ``2024-06-17`` or ``22:49:39`` style tokens."""
    return bool(re.fullmatch(
        r"\d{2,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}:\d{2}(:\d{2})?", token))


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def write(stations: list[Station], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# network station longitude latitude elevation\n")
        for s in stations:
            fh.write(f"{s.network or '--'} {s.name} {s.lon:.6f} "
                     f"{s.lat:.6f} {s.elev:.3f}\n")
    return path


def match(stations: list[Station], key: str) -> Station | None:
    """Find the station whose name best matches *key*.

    Field loggers write the same instrument several ways -- ``450000123`` in
    the deployment sheet, ``00123`` in the MiniSEED header, ``S007`` on the
    map. Exact match wins; otherwise one name being a suffix of the other is
    accepted, which is the relationship those forms actually have.
    """
    key = str(key).strip()
    for s in stations:
        if s.name == key or s.code == key:
            return s
    candidates = [
        s for s in stations
        if (s.name and (key.endswith(s.name) or s.name.endswith(key)))
    ]
    if len(candidates) == 1:
        return candidates[0]
    # Ambiguity is resolved by the longest common suffix, which prefers
    # 00123 -> 450000123 over 00123 -> 123.
    if candidates:
        return max(candidates, key=lambda s: len(s.name))
    return None


def to_local_xy(lats, lons, origin: tuple[float, float] | None = None):
    """Project geographic coordinates onto a local tangent plane, in metres.

    An equirectangular projection about the survey centroid. Over the few tens
    of kilometres a passive-seismic survey spans, its distortion is well under
    the coordinate precision, and unlike a UTM conversion it needs no extra
    dependency and never straddles a zone boundary.

    Returns ``(x, y, origin)`` with x east and y north.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    finite = np.isfinite(lats) & np.isfinite(lons)
    if origin is None:
        if not finite.any():
            origin = (0.0, 0.0)
        else:
            origin = (float(np.mean(lats[finite])), float(np.mean(lons[finite])))
    lat0, lon0 = origin
    coslat = math.cos(math.radians(lat0))
    x = np.radians(lons - lon0) * _R_EARTH * coslat
    y = np.radians(lats - lat0) * _R_EARTH
    return x, y, origin


def coordinates(stations: list[Station]) -> tuple[np.ndarray, np.ndarray]:
    lats = np.array([s.lat for s in stations], dtype=float)
    lons = np.array([s.lon for s in stations], dtype=float)
    return lats, lons
