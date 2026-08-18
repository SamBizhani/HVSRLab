"""Borehole control points used to calibrate the f0 → depth regression.

Two layouts are read:

* **ProTO well files** — ``lithology  thickness`` pairs, one layer per line.
  Depth to bedrock is the cumulative thickness above the first layer whose
  name marks basement (or the total, if none does).
* **A survey table** — one row per well with a name, coordinates and a depth,
  which is all the regression actually needs.

Both end up as :class:`~hvsrlab.project.Well` records.
"""

from __future__ import annotations

from pathlib import Path
import re

from ..project import Well

#: Layer names that mean "this is the half-space". Matched case-insensitively
#: as a substring, so "weathered bedrock" and "BEDROCK" both count.
BASEMENT_WORDS = ("bedrock", "basement", "rock", "granite", "basalt",
                  "limestone", "refusal")

_SPLIT = re.compile(r"[,\t]|\s{2,}|\s+")


def read_log(path: str | Path, *, name: str = "") -> Well:
    """Read one ProTO-style well log: ``lithology thickness`` per line."""
    path = Path(path)
    layers: list[list] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "%")):
            continue
        parts = [p for p in _SPLIT.split(line) if p]
        if len(parts) < 2:
            continue
        try:
            thickness = float(parts[-1])
        except ValueError:
            continue
        layers.append([" ".join(parts[:-1]), thickness])

    well = Well(name=name or path.stem, layers=layers)
    well.bedrock_depth = bedrock_depth(layers)
    return well


def bedrock_depth(layers: list[list]) -> float:
    """Cumulative thickness above the first basement layer.

    Falls back to the total logged thickness when no layer name matches, which
    is the right answer for logs that stop at refusal.
    """
    total = 0.0
    for lithology, thickness in layers:
        if any(word in str(lithology).lower() for word in BASEMENT_WORDS):
            return total
        try:
            total += float(thickness)
        except (TypeError, ValueError):
            continue
    return total if total > 0 else float("nan")


def read_table(path: str | Path) -> list[Well]:
    """Read a table of wells: ``name  lat  lon  depth`` — or in metres.

    Column order is detected the same way station files are, and by the same
    code, so a borehole table written in UTM easting/northing is read as
    readily as one in degrees. The projected pair goes straight to the well's
    plan coordinates; the caller's projection then fills in the degrees.
    """
    from . import stations as station_io

    path = Path(path)
    out: list[Well] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "%")):
            continue
        parts = [p for p in _SPLIT.split(line) if p]
        st = station_io._parse(parts)
        if st is None:
            continue
        # `_parse` treats the first numeric after the coordinates as elevation;
        # in a well table that column is the depth to bedrock.
        well = Well(name=st.name, bedrock_depth=st.elev)
        if st.projected:
            well.x, well.y = st.easting, st.northing
        else:
            well.lat, well.lon = st.lat, st.lon
        out.append(well)
    return out
