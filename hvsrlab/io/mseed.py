"""MiniSEED ingestion: cataloguing a survey, and pulling one time slice out of it.

A passive-seismic deployment leaves behind a lot of data — a hundred sites
× six weeks × 3 components at 250 Hz runs to some 12 000 files and a few
hundred gigabytes. H/V needs a few hours of it. So nothing here ever reads a
whole recording:

* :func:`scan` builds the catalogue from file *names* where it can, falling
  back to record headers only where it must. A 12 000-file survey catalogues in
  seconds rather than minutes.
* :func:`load_segment` uses ObsPy's record-level selection to lift one window
  out of a 22 MB day file in about 50 ms, then decimates before anything else
  touches the samples.

Layouts handled:

* one folder per site (``<root>/450000123/*.miniseed``)
* every file in one flat folder
* arbitrary nesting — the walk is recursive either way
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence

import numpy as np

#: Extensions treated as MiniSEED.
EXTENSIONS = (".miniseed", ".mseed", ".msd", ".seed", ".ms")

#: Canonical component order used everywhere downstream.
COMPONENTS = ("Z", "N", "E")

#: Channel-code / filename suffix to canonical component. Vertical is Z or 3;
#: the horizontals are N/E or 1/2 depending on the digitiser's vintage.
_COMPONENT_ALIASES = {
    "Z": "Z", "3": "Z", "V": "Z", "UD": "Z",
    "N": "N", "1": "N", "NS": "N",
    "E": "E", "2": "E", "EW": "E",
}

#: ``450000123.0002.2024.01.05.00.00.00.000.E.miniseed`` — the layout written
#: by the survey's dataloggers. Everything the catalogue needs is in the name.
_NAME_RE = re.compile(
    r"^(?P<site>[A-Za-z0-9_\-]+?)"
    r"\.(?P<seq>\d+)"
    r"\.(?P<Y>\d{4})\.(?P<M>\d{2})\.(?P<D>\d{2})"
    r"\.(?P<h>\d{2})\.(?P<m>\d{2})\.(?P<s>\d{2})(?:\.(?P<ms>\d{1,3}))?"
    r"\.(?P<comp>[A-Za-z0-9]{1,2})$"
)

Progress = Callable[[int, int, str], None]


@dataclass
class TraceFile:
    """One MiniSEED file, one component."""

    path: str
    component: str
    start: float                 # epoch seconds, UTC
    end: float                   # epoch seconds, UTC
    fs: float = 0.0
    exact: bool = False          # True when start/end came from record headers
    size: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, t0: float, t1: float) -> bool:
        return self.start < t1 and self.end > t0


@dataclass
class SiteRecording:
    """Every file belonging to one measurement point."""

    sid: str
    station: str = ""
    network: str = ""
    location: str = ""
    fs: float = 0.0
    files: dict[str, list[TraceFile]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- coverage ----------------------------------------------------------
    @property
    def components(self) -> tuple[str, ...]:
        return tuple(c for c in COMPONENTS if self.files.get(c))

    @property
    def is_three_component(self) -> bool:
        return len(self.components) == 3

    @property
    def start(self) -> float:
        starts = [f.start for fs in self.files.values() for f in fs]
        return min(starts) if starts else float("nan")

    @property
    def end(self) -> float:
        ends = [f.end for fs in self.files.values() for f in fs]
        return max(ends) if ends else float("nan")

    @property
    def duration_days(self) -> float:
        s, e = self.start, self.end
        return (e - s) / 86400.0 if np.isfinite(s) and np.isfinite(e) else float("nan")

    @property
    def n_files(self) -> int:
        return sum(len(v) for v in self.files.values())

    @property
    def bytes(self) -> int:
        return sum(f.size for v in self.files.values() for f in v)

    def common_span(self) -> tuple[float, float]:
        """The interval covered by *all* present components."""
        if not self.components:
            return (float("nan"), float("nan"))
        starts = [min(f.start for f in self.files[c]) for c in self.components]
        ends = [max(f.end for f in self.files[c]) for c in self.components]
        return (max(starts), min(ends))

    def files_in(self, comp: str, t0: float, t1: float) -> list[TraceFile]:
        return [f for f in self.files.get(comp, []) if f.contains(t0, t1)]

    def paths_in(self, t0: float, t1: float) -> list[str]:
        out: list[str] = []
        for comp in self.components:
            out.extend(f.path for f in self.files_in(comp, t0, t1))
        return out


@dataclass
class ScanResult:
    sites: list[SiteRecording] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return sum(s.n_files for s in self.sites)

    @property
    def total_bytes(self) -> int:
        return sum(s.bytes for s in self.sites)


@dataclass
class Segment:
    """A loaded, decimated, three-component time slice, ready for windowing."""

    sid: str
    fs: float
    start: float                       # epoch seconds of the first sample
    data: dict[str, np.ndarray]        # {"Z": …, "N": …, "E": …}, equal lengths
    gaps: int = 0                      # gap/overlap count reported by the merge
    source_fs: float = 0.0

    @property
    def npts(self) -> int:
        return len(self.data["Z"]) if "Z" in self.data else 0

    @property
    def duration(self) -> float:
        return self.npts / self.fs if self.fs else 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration

    def as_matrix(self) -> np.ndarray:
        """(npts, 3) array in Z, N, E order."""
        return np.column_stack([self.data[c] for c in COMPONENTS])

    def times(self) -> np.ndarray:
        return np.arange(self.npts, dtype=float) / self.fs


# ---------------------------------------------------------------------------
# Cataloguing
# ---------------------------------------------------------------------------

def scan(root: str | Path, *, refine: bool = False,
         progress: Progress | None = None,
         limit: int | None = None) -> ScanResult:
    """Catalogue every MiniSEED file under *root*.

    With ``refine=False`` (the default) start times and component labels come
    from file names, and each file's end time is inferred from the start of the
    next file in its own component sequence — only the last file of each
    sequence costs a header read. Set ``refine=True`` to read the true span of
    every file, which is slower but exact in the presence of gaps.

    Files whose names do not follow a recognised pattern always fall back to
    header reads, so unusual naming costs speed but never correctness.
    """
    root = Path(root)
    result = ScanResult()
    if not root.exists():
        result.warnings.append(f"{root} does not exist")
        return result

    paths = _find_files(root, limit=limit)
    if not paths:
        result.warnings.append(f"no MiniSEED files found under {root}")
        return result

    # Site identity comes from the folder when the survey is organised one
    # folder per site, and from the file name otherwise.
    folder_per_site = _looks_folder_per_site(root, paths)

    by_site: dict[str, SiteRecording] = {}
    total = len(paths)
    for i, path in enumerate(paths):
        if progress is not None and (i % 200 == 0 or i == total - 1):
            progress(i + 1, total, path.name)

        info = _parse_name(path)
        if info is None:
            info = _read_header(path)
            if info is None:
                result.skipped.append(str(path))
                continue

        sid = path.parent.name if folder_per_site else info["site"]
        rec = by_site.get(sid)
        if rec is None:
            rec = by_site[sid] = SiteRecording(sid=sid, station=info.get("station", ""))
        comp = info["component"]
        tf = TraceFile(
            path=str(path),
            component=comp,
            start=info["start"],
            end=info.get("end", info["start"]),
            fs=info.get("fs", 0.0),
            exact=bool(info.get("exact")),
            size=_size(path),
        )
        rec.files.setdefault(comp, []).append(tf)

    for rec in by_site.values():
        _finalise(rec, refine=refine)
        if not rec.is_three_component:
            missing = [c for c in COMPONENTS if c not in rec.components]
            rec.warnings.append("missing component(s): " + ", ".join(missing))

    result.sites = sorted(by_site.values(), key=lambda r: r.sid)
    if result.skipped:
        result.warnings.append(f"{len(result.skipped)} file(s) not recognised")
    return result


def _find_files(root: Path, limit: int | None = None) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            out.append(path)
            if limit and len(out) >= limit:
                break
    return out


def _looks_folder_per_site(root: Path, paths: Sequence[Path]) -> bool:
    """True when the files sit in per-site subfolders rather than one flat dir."""
    parents = {p.parent for p in paths}
    if len(parents) <= 1:
        return False
    return root not in parents or len(parents) > 2


def _parse_name(path: Path) -> dict | None:
    """Extract site, component and start time from the file name."""
    m = _NAME_RE.match(path.stem)
    if not m:
        return None
    comp = _COMPONENT_ALIASES.get(m.group("comp").upper())
    if comp is None:
        return None
    ms = int((m.group("ms") or "0").ljust(3, "0"))
    try:
        t = datetime(
            int(m.group("Y")), int(m.group("M")), int(m.group("D")),
            int(m.group("h")), int(m.group("m")), int(m.group("s")),
            ms * 1000, tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    return {"site": m.group("site"), "component": comp,
            "start": t.timestamp(), "exact": False}


def _read_header(path: Path) -> dict | None:
    """Fall back to the record headers for files with unfamiliar names."""
    try:
        from obspy.io.mseed.util import get_record_information, get_start_and_end_time
        info = get_record_information(str(path))
        start, end = get_start_and_end_time(str(path))
    except Exception:
        return None

    channel = str(info.get("channel", "") or "")
    comp = _COMPONENT_ALIASES.get(channel[-1:].upper()) if channel else None
    if comp is None:
        comp = _COMPONENT_ALIASES.get(path.stem.split(".")[-1].upper())
    if comp is None:
        return None
    return {
        "site": str(info.get("station", "") or path.stem),
        "station": str(info.get("station", "") or ""),
        "component": comp,
        "start": float(start.timestamp),
        "end": float(end.timestamp),
        "fs": float(info.get("samp_rate", 0.0) or 0.0),
        "exact": True,
    }


def _finalise(rec: SiteRecording, *, refine: bool) -> None:
    """Sort each component's files and fill in the end times."""
    for comp, files in rec.files.items():
        files.sort(key=lambda f: f.start)
        for i, tf in enumerate(files):
            if tf.exact and not refine:
                continue
            if refine or i == len(files) - 1:
                _refine_one(tf)
            else:
                # A day file ends where the next one begins. True unless the
                # logger stopped mid-sequence, which `refine=True` catches.
                tf.end = files[i + 1].start
        if files and not rec.fs:
            rec.fs = next((f.fs for f in files if f.fs), 0.0)

    if not rec.fs:
        for comp in rec.components:
            _refine_one(rec.files[comp][0])
            if rec.files[comp][0].fs:
                rec.fs = rec.files[comp][0].fs
                break


def _refine_one(tf: TraceFile) -> None:
    try:
        from obspy.io.mseed.util import get_record_information, get_start_and_end_time
        start, end = get_start_and_end_time(tf.path)
        tf.start = float(start.timestamp)
        tf.end = float(end.timestamp)
        tf.exact = True
        if not tf.fs:
            tf.fs = float(get_record_information(tf.path).get("samp_rate", 0.0) or 0.0)
    except Exception:
        if tf.end <= tf.start:
            tf.end = tf.start + 86400.0     # assume a day file; better than zero


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_segment(rec: SiteRecording, start: float, end: float, *,
                 target_fs: float = 0.0,
                 detrend: str = "demean",
                 components: Iterable[str] = COMPONENTS) -> Segment:
    """Read ``[start, end)`` of *rec* and return a decimated three-component slice.

    Only the records overlapping the interval are decoded. Gaps are merged with
    zeros after detrending, which keeps window indexing uniform; the count is
    reported on the returned :class:`Segment` so the caller can warn.

    Raises ``ValueError`` if a requested component has no data in the interval.
    """
    from obspy import UTCDateTime, Stream, read

    t0, t1 = UTCDateTime(start), UTCDateTime(end)
    data: dict[str, np.ndarray] = {}
    fs_out = 0.0
    gaps = 0
    source_fs = rec.fs
    actual_start = start

    for comp in components:
        files = rec.files_in(comp, start, end)
        if not files:
            raise ValueError(f"{rec.sid}: no {comp} data between "
                             f"{_iso(start)} and {_iso(end)}")
        st = Stream()
        for tf in files:
            try:
                st += read(tf.path, starttime=t0, endtime=t1, format="MSEED")
            except Exception as exc:                       # corrupt chunk
                rec.warnings.append(f"{Path(tf.path).name}: {exc}")
        if not st:
            raise ValueError(f"{rec.sid}: {comp} unreadable in the requested window")

        st.merge(method=1, fill_value=0, interpolation_samples=0)
        gaps += max(0, len(st) - 1)
        st.trim(t0, t1)
        tr = st[0]
        source_fs = float(tr.stats.sampling_rate)

        if detrend and detrend != "none":
            tr.detrend("demean" if detrend == "demean" else "linear")
        if target_fs and target_fs < source_fs:
            _downsample(tr, target_fs)

        data[comp] = np.ascontiguousarray(tr.data, dtype=np.float64)
        fs_out = float(tr.stats.sampling_rate)
        actual_start = float(tr.stats.starttime.timestamp)

    n = min(len(v) for v in data.values())
    for comp in list(data):
        data[comp] = data[comp][:n]

    return Segment(sid=rec.sid, fs=fs_out, start=actual_start, data=data,
                   gaps=gaps, source_fs=source_fs)


def _downsample(tr, target_fs: float) -> None:
    """Decimate to *target_fs*, anti-aliasing on the way.

    An integer factor uses ObsPy's decimate (FIR lowpass then take every nth
    sample). A non-integer ratio falls back to Fourier resampling, which also
    applies an anti-alias filter but costs an FFT of the whole trace.
    """
    ratio = tr.stats.sampling_rate / target_fs
    factor = int(round(ratio))
    if factor >= 2 and abs(ratio - factor) < 1e-6:
        # Decimating by more than ~16 at once strains the FIR design; step it.
        for f in _factorise(factor):
            tr.decimate(f, strict_length=False, no_filter=False)
    elif ratio > 1.0:
        tr.resample(target_fs, window="hann", no_filter=False)


def _factorise(n: int, limit: int = 8) -> list[int]:
    """Split a decimation factor into stages no larger than *limit*."""
    out: list[int] = []
    remaining = n
    for f in (7, 5, 4, 3, 2):
        while remaining % f == 0 and remaining > 1:
            out.append(f)
            remaining //= f
    if remaining > 1:                      # prime and large: take it in one go
        out.append(remaining)
    return out or [n]


def probe_rms(rec: SiteRecording, start: float, duration: float,
              components: Iterable[str] = COMPONENTS) -> dict[str, float]:
    """Cheap amplitude probe: RMS of a short slice, per component.

    Used by the reconnaissance scan that ranks hours of a multi-day recording.
    Reads only the records it needs, so a 60 s probe out of a 22 MB day file
    costs tens of milliseconds.
    """
    from obspy import UTCDateTime, read

    t0 = UTCDateTime(start)
    t1 = UTCDateTime(start + duration)
    out: dict[str, float] = {}
    for comp in components:
        files = rec.files_in(comp, start, start + duration)
        if not files:
            out[comp] = float("nan")
            continue
        try:
            st = read(files[0].path, starttime=t0, endtime=t1, format="MSEED")
            for tf in files[1:]:
                st += read(tf.path, starttime=t0, endtime=t1, format="MSEED")
            st.merge(method=1, fill_value=0)
            tr = st[0]
            x = np.asarray(tr.data, dtype=float)
            x -= x.mean()
            out[comp] = float(np.sqrt(np.mean(x * x))) if x.size else float("nan")
        except Exception:
            out[comp] = float("nan")
    return out


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def iso(epoch: float) -> str:
    """UTC ``YYYY-MM-DD HH:MM:SS`` for an epoch time (public helper)."""
    return _iso(epoch)


def to_epoch(text: str) -> float:
    """Parse an ISO-ish UTC timestamp into epoch seconds."""
    text = text.strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"cannot parse time {text!r}")
