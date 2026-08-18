"""Non-SEED inputs: SAF and ASCII microtremor recordings, and H/V curve files.

OpenHVSR-ProTO reads three things this module reproduces:

* **SAF** — SESAME ASCII data format, as written by MAE, Grilla and Pasi
  instruments. A ``KEY = value`` header, a ``####----`` separator, then three
  numeric columns whose meaning is declared by ``CH0_ID``/``CH1_ID``/``CH2_ID``.
* **ASCII with a header** — free-form header text, a user-declared separator
  line, then three columns in a user-declared order.
* **ASCII data only** — the columns with nothing above them.

Rather than fingerprinting every vendor dialect the way ProTO does, the SAF
reader keys off the structure all of them share: the header is ``KEY = value``
lines, the data starts after the last ``#``-run separator, and the channel
identifiers say which column is which. That reads all three vendors, and
anything else that follows the published format.

It also reads and writes plain H/V curve files, which is how results move
between HVSRLab, OpenHVSR-Inversion and Geopsy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

#: Channel identifiers seen in SAF headers, mapped to canonical components.
_CH_ALIASES = {
    "UD": "Z", "V": "Z", "Z": "Z", "VERT": "Z", "UP": "Z",
    "NS": "N", "N": "N", "NORTH": "N", "H1": "N", "Y": "N",
    "EW": "E", "E": "E", "EAST": "E", "H2": "E", "X": "E",
}

#: A run of '#' followed by dashes: every SAF dialect ends its header this way.
_SEPARATOR_RE = re.compile(r"^#{2,}\s*-{2,}")

_KEYVALUE_RE = re.compile(r"^\s*([A-Za-z0-9_ ()]+?)\s*=\s*(.*?)\s*$")


@dataclass
class Recording:
    """A three-component microtremor recording read from a text file."""

    data: dict[str, np.ndarray] = field(default_factory=dict)
    fs: float = 0.0
    start: float = float("nan")        # epoch seconds, if the file says
    units: str = ""
    fmt: str = ""
    header: dict[str, str] = field(default_factory=dict)
    lat: float = float("nan")
    lon: float = float("nan")
    elev: float = float("nan")

    @property
    def npts(self) -> int:
        return len(self.data["Z"]) if "Z" in self.data else 0


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def read_recording(path: str | Path, *, fs: float = 0.0,
                   separator: str = "", columns: tuple[int, int, int] = (1, 2, 3)
                   ) -> Recording:
    """Read a microtremor recording from SAF or ASCII.

    ``columns`` is 1-based and ordered ``(vertical, east, north)`` — ProTO's
    convention — and applies only to the non-SAF paths, where the file itself
    does not say which column is which. ``separator`` is the line that ends a
    free-form header; when empty, the first line that parses as numbers wins.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    if _is_saf(text):
        return _read_saf(text, path)
    return _read_ascii(text, path, fs=fs, separator=separator, columns=columns)


def _is_saf(lines: list[str]) -> bool:
    head = "\n".join(lines[:5]).upper()
    return "SESAME ASCII DATA FORMAT" in head or "SAF" in head.split("\n")[0].upper()


def _read_saf(lines: list[str], path: Path) -> Recording:
    header: dict[str, str] = {}
    data_start = 0
    for i, line in enumerate(lines):
        if _SEPARATOR_RE.match(line):
            data_start = i + 1          # the last such line wins
            continue
        m = _KEYVALUE_RE.match(line)
        if m and not line.lstrip().startswith("#"):
            header[m.group(1).strip().upper()] = m.group(2).strip()

    if not data_start:
        data_start = _first_numeric_line(lines)

    table = _read_table(lines[data_start:])
    if table.size == 0:
        raise ValueError(f"{path.name}: no numeric data found")

    # Column order is declared by CHn_ID; default to the SAF standard UD/NS/EW.
    order = []
    for n in range(3):
        cid = header.get(f"CH{n}_ID", "").strip().upper()
        order.append(_CH_ALIASES.get(cid, ""))
    if sorted(c for c in order if c) != ["E", "N", "Z"]:
        order = ["Z", "N", "E"]

    ncols = table.shape[1]
    conv = _as_float(header.get("CONV_FACTOR", "1"), 1.0) or 1.0
    data = {}
    for i, comp in enumerate(order[:ncols]):
        data[comp] = table[:, i] * conv

    rec = Recording(
        data=data,
        fs=_as_float(header.get("SAMP_FREQ", ""), 0.0),
        units=header.get("UNITS", ""),
        fmt="saf",
        header=header,
        start=_saf_start(header),
    )
    rec.lon = _as_float(header.get("EVT_X", ""), float("nan"))
    rec.lat = _as_float(header.get("EVT_Y", ""), float("nan"))
    rec.elev = _as_float(header.get("EVT_Z", ""), float("nan"))
    if not rec.fs:
        raise ValueError(f"{path.name}: SAMP_FREQ missing from the SAF header")
    return rec


def _read_ascii(lines: list[str], path: Path, *, fs: float, separator: str,
                columns: tuple[int, int, int]) -> Recording:
    start = 0
    if separator:
        needle = separator.strip()
        for i, line in enumerate(lines):
            if line.strip().startswith(needle):
                start = i + 1
                break
        else:
            start = _first_numeric_line(lines)
    else:
        start = _first_numeric_line(lines)

    table = _read_table(lines[start:])
    if table.size == 0:
        raise ValueError(f"{path.name}: no numeric data found")
    if table.shape[1] < 3:
        raise ValueError(f"{path.name}: expected 3 data columns, found "
                         f"{table.shape[1]}")

    cv, ce, cn = (int(c) - 1 for c in columns)
    for c in (cv, ce, cn):
        if not 0 <= c < table.shape[1]:
            raise ValueError(f"{path.name}: column {c + 1} is out of range "
                             f"({table.shape[1]} columns present)")
    data = {"Z": table[:, cv], "E": table[:, ce], "N": table[:, cn]}
    return Recording(data=data, fs=float(fs), fmt="ascii")


def _first_numeric_line(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        parts = line.replace(",", " ").split()
        if len(parts) >= 2 and all(_is_number(p) for p in parts):
            return i
    return len(lines)


def _read_table(lines: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    width = 0
    for line in lines:
        parts = line.replace(",", " ").split()
        if not parts or not all(_is_number(p) for p in parts):
            if rows:                    # trailing footer ends the table
                break
            continue
        values = [float(p) for p in parts]
        width = width or len(values)
        if len(values) != width:
            break
        rows.append(values)
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 0))


def _saf_start(header: dict[str, str]) -> float:
    raw = header.get("START_TIME", "").strip()
    parts = raw.replace("/", " ").replace(":", " ").split()
    if len(parts) < 6:
        return float("nan")
    try:
        y, mo, d, h, mi = (int(float(p)) for p in parts[:5])
        s = float(parts[5])
        return datetime(y, mo, d, h, mi, int(s),
                        int((s % 1) * 1e6), tzinfo=timezone.utc).timestamp()
    except (ValueError, OverflowError):
        return float("nan")


# ---------------------------------------------------------------------------
# H/V curve files
# ---------------------------------------------------------------------------

def read_curve(path: str | Path, columns: tuple[int, int, int] = (1, 2, 3)
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``frequency, H/V, standard deviation`` from a text file.

    ``columns`` is 1-based; the third is optional and yields NaNs when the file
    has only two columns.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    table = _read_table(lines[_first_numeric_line(lines):])
    if table.size == 0:
        raise ValueError(f"{path.name}: no numeric data found")

    cf, ch, cs = (int(c) - 1 for c in columns)
    freq = table[:, cf]
    hv = table[:, ch]
    std = table[:, cs] if 0 <= cs < table.shape[1] else np.full(freq.shape, np.nan)
    return freq, hv, std


def write_curve(path: str | Path, freq, hv, std=None, *, header: str = "") -> Path:
    """Write an H/V curve as three whitespace-separated columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(freq, dtype=float)
    hv = np.asarray(hv, dtype=float)
    std = np.full(freq.shape, np.nan) if std is None else np.asarray(std, dtype=float)
    with open(path, "w", encoding="utf-8") as fh:
        if header:
            for line in header.splitlines():
                fh.write(f"# {line}\n")
        fh.write("# frequency_Hz   HVSR   std\n")
        for f, h, s in zip(freq, hv, std):
            fh.write(f"{f:12.6f} {h:12.6f} {s:12.6f}\n")
    return path


def _as_float(text: str, default: float) -> float:
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return default


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True
