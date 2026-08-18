"""Project model: sites, processing parameters, and on-disk persistence.

A project is one JSON file (``project.json``) plus a directory tree. The JSON
holds everything cheap — site coordinates, file paths, processing parameters,
picks — while the expensive per-site arrays (per-window spectra, H/V curves,
azimuthal grids) live in ``results/<site_id>.npz`` and are loaded on demand.
That split is what makes a 102-site survey openable in under a second.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from . import paths

SCHEMA = 1

#: The three ways ProTO can collapse the two horizontals into one H.
HVSR_STRATEGIES = ("squared_average", "simple_average", "total_energy")

HVSR_STRATEGY_LABELS = {
    "squared_average": "Average Squared   H = sqrt[(E² + N²)/2]",
    "simple_average": "Simple Average    H = (E + N)/2",
    "total_energy": "Total Energy      H = sqrt(E² + N²)",
}

SMOOTHING_KINDS = ("konno_ohmachi", "moving_average", "none")

FILTER_KINDS = ("off", "bandpass", "lowpass", "highpass")

#: Where the filtered traces are used: for H/V itself, or only to drive the
#: STA/LTA window selection (ProTO's "data to use" switch).
FILTER_TARGETS = ("hvsr", "antitrigger_only")

STATISTICS_KINDS = ("lognormal", "linear")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _from_dict(cls, data: dict[str, Any]):
    """Build a dataclass from a dict, ignoring keys the class does not know.

    Projects written by an older build stay loadable, and unknown future keys
    do not crash this one.
    """
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ProcParams:
    """Everything that controls one H/V computation.

    Defaults follow ``DEFAULT_VALUES.m`` from OpenHVSR-ProTO except where the
    comment says otherwise.
    """

    # -- windowing ---------------------------------------------------------
    window_width_s: float = 60.0        # ProTO: 30. 60 s resolves lower f0.
    window_overlap_pc: float = 50.0
    taper_pc: float = 5.0
    pad_to: str = "off"                 # "off" or a sample count; rounded up to 2^n

    # -- STA/LTA anti-trigger ---------------------------------------------
    antitrigger: bool = True
    sta_s: float = 1.0
    lta_s: float = 30.0
    sta_lta_ratio: float = 4.0

    # -- spectra -----------------------------------------------------------
    freq_min: float = 0.2               # ProTO: 0.5
    freq_max: float = 25.0              # ProTO: 50
    smoothing_kind: str = "konno_ohmachi"
    smoothing_b: float = 40.0
    hvsr_strategy: str = "squared_average"
    statistics: str = "lognormal"       # SESAME reports log-normal std of f0
    freq_grid: str = "log"              # "log" | "linear" (linear = ProTO's axis)
    n_freq: int = 512                   # points on the log grid

    # -- pre-filter --------------------------------------------------------
    filter_kind: str = "off"
    filter_order: int = 4
    filter_fmin: float = 0.5
    filter_fmax: float = 25.0
    filter_target: str = "hvsr"

    # -- ingestion ---------------------------------------------------------
    target_fs: float = 0.0              # 0 = keep native; else decimate to this
    detrend: str = "demean"             # "none" | "demean" | "linear"

    # -- azimuthal analysis -------------------------------------------------
    azimuth_step_deg: float = 0.0       # 0 = off; 10 or 15 are typical

    # -- acceptance --------------------------------------------------------
    min_windows: int = 5                # ProTO excludes a site below 5

    def copy(self) -> "ProcParams":
        return _from_dict(ProcParams, asdict(self))

    @property
    def nyquist_ok_max(self) -> float:
        """Upper frequency that survives the requested decimation."""
        return self.target_fs / 2.0 if self.target_fs else float("inf")


@dataclass
class TimeSelection:
    """Which slice of a multi-day recording feeds the H/V of one site.

    ``start``/``end`` are ISO-8601 UTC strings. ``mode`` records how they were
    chosen so the GUI can show provenance and a batch run can re-derive them.
    """

    mode: str = "auto"          # "auto" | "manual" | "all"
    start: str = ""
    end: str = ""
    hours: float = 8.0          # requested duration for "auto"
    score: float = float("nan")  # noise score of the chosen block, if scanned

    def is_set(self) -> bool:
        return bool(self.start and self.end)


@dataclass
class Site:
    """One measurement point."""

    sid: str                     # stable identifier, unique in the project
    name: str = ""
    lat: float = float("nan")
    lon: float = float("nan")
    elev: float = 0.0
    # Projected / local plan coordinates used by maps, profiles and 3D views.
    x: float = float("nan")
    y: float = float("nan")
    z: float = 0.0

    source: str = "mseed"        # "mseed" | "curve"
    files: dict[str, str] = field(default_factory=dict)   # {"Z":…, "N":…, "E":…}
    curve_file: str = ""
    fs: float = 0.0              # native sampling rate, Hz

    status: str = "active"       # "active" | "locked" | "excluded"
    note: str = ""

    time: TimeSelection = field(default_factory=TimeSelection)
    params: dict[str, Any] = field(default_factory=dict)  # per-site overrides

    # Picks and headline results, kept in the JSON so maps draw without
    # touching the .npz files.
    f0: float = float("nan")
    a0: float = float("nan")
    f0_std: float = float("nan")
    f0_source: str = ""          # "auto" | "user"
    extra_peaks: list[list[float]] = field(default_factory=list)  # [[f, A], …]
    sesame_score: str = ""       # e.g. "3/3 · 5/6"
    depth: float = float("nan")  # bedrock depth from the active regression
    n_windows: int = 0
    n_windows_ok: int = 0
    computed: str = ""           # ISO timestamp of the last successful run

    def label(self) -> str:
        return self.name or self.sid

    def effective_params(self, base: ProcParams) -> ProcParams:
        """Project defaults with this site's overrides applied on top."""
        merged = asdict(base)
        merged.update(self.params or {})
        return _from_dict(ProcParams, merged)

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass
class Well:
    """A borehole used to calibrate the f0 -> depth regression.

    Drilling rarely lands on a seismic station, so a well is tied to the
    *nearest* site automatically and the separation is kept alongside it. That
    distance is the assumption the calibration rests on — that the basin does
    not change between the hole and the station — so it is carried with the
    data rather than being forgotten once the link is made.
    """

    name: str = ""
    lat: float = float("nan")
    lon: float = float("nan")
    x: float = float("nan")
    y: float = float("nan")
    z: float = 0.0
    bedrock_depth: float = float("nan")
    layers: list[list[Any]] = field(default_factory=list)  # [[lithology, thickness], …]
    site: str = ""               # sid of the associated H/V site, if any
    link_mode: str = "auto"      # "auto" = follow the nearest site, "user" = pinned
    link_distance: float = float("nan")   # metres from the well to that site


@dataclass
class Profile:
    """A 2D section defined by two map points.

    Membership starts from the corridor rule — every site within ``width`` of
    the line, or all of them when ``width`` is zero — and the two override
    lists then edit that set by hand. Keeping the rule and the overrides
    separate means a profile still picks up sites added later, while the
    decisions you made explicitly are never quietly undone.
    """

    name: str = ""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    width: float = 0.0           # capture half-width for station projection
    n_nodes: int = 200
    include: list[str] = field(default_factory=list)   # sids forced in
    exclude: list[str] = field(default_factory=list)   # sids forced out

    @property
    def length(self) -> float:
        import math
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def members(self, sids: Iterable[str], within: Iterable[str]) -> list[str]:
        """Which of *sids* belong, given the corridor's own selection *within*."""
        auto = set(within)
        forced_in, forced_out = set(self.include), set(self.exclude)
        return [s for s in sids
                if (s in forced_in) or (s in auto and s not in forced_out)]

    def set_member(self, sid: str, member: bool, *, in_corridor: bool) -> None:
        """Record an explicit decision about one site.

        A choice that agrees with the corridor rule clears the override rather
        than storing it, so the lists only ever hold genuine exceptions.
        """
        self.include = [s for s in self.include if s != sid]
        self.exclude = [s for s in self.exclude if s != sid]
        if member and not in_corridor:
            self.include.append(sid)
        elif not member and in_corridor:
            self.exclude.append(sid)

    def clear_overrides(self) -> None:
        self.include = []
        self.exclude = []


@dataclass
class Regression:
    """Bedrock depth law  H = a · f0^b  (Ibs-von Seht & Wohlenberg, 1999)."""

    name: str = "Ibs-von Seht & Wohlenberg (1999)"
    a: float = 96.0
    b: float = -1.388
    fitted: bool = False
    n_points: int = 0
    rms: float = float("nan")


class Project:
    """A survey: its sites, its parameters, and where its outputs go."""

    def __init__(self, root: str | Path, name: str = "") -> None:
        self.root = Path(root)
        self.name = name or self.root.name
        self.created = _utcnow()
        self.modified = self.created
        self.notes = ""

        # Data source
        self.source_kind = "mseed"          # "mseed" | "curve"
        self.raw_dir = ""
        self.station_file = ""
        self.topography_file = ""

        #: The plan coordinate system, as ``UTM.to_dict()``. Chosen from the
        #: survey's own centroid the first time coordinates arrive, then kept,
        #: so every session lands on the same grid.
        self.crs: dict[str, Any] = {}

        #: Hours to add to UTC for display only. MiniSEED is UTC and stays
        #: UTC internally; this exists so "the quiet hours" read as the local
        #: night they actually are.
        self.utc_offset = 0.0

        self.params = ProcParams()
        self.sites: list[Site] = []
        self.wells: list[Well] = []
        self.profiles: list[Profile] = []
        self.regression = Regression()
        self.history: list[dict[str, Any]] = []

    # -- paths -------------------------------------------------------------
    @property
    def json_path(self) -> Path:
        return self.root / "project.json"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    def result_path(self, sid: str) -> Path:
        return self.results_dir / f"{_safe(sid)}.npz"

    def ensure_tree(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in paths.SUBDIRS:
            (self.root / sub).mkdir(exist_ok=True)

    # -- site access -------------------------------------------------------
    def site(self, sid: str) -> Site | None:
        for s in self.sites:
            if s.sid == sid:
                return s
        return None

    def index_of(self, sid: str) -> int:
        for i, s in enumerate(self.sites):
            if s.sid == sid:
                return i
        return -1

    def active_sites(self) -> list[Site]:
        return [s for s in self.sites if s.is_active]

    def computed_sites(self) -> list[Site]:
        import math
        return [s for s in self.sites if s.status != "excluded"
                and not math.isnan(s.f0)]

    def add_sites(self, sites: Iterable[Site]) -> int:
        """Add sites, skipping ones whose sid is already present."""
        seen = {s.sid for s in self.sites}
        added = 0
        for s in sites:
            if s.sid in seen:
                continue
            self.sites.append(s)
            seen.add(s.sid)
            added += 1
        return added

    def log(self, action: str, **detail: Any) -> None:
        self.history.append({"time": _utcnow(), "action": action, **detail})
        if len(self.history) > 500:
            del self.history[:-500]

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "root": str(self.root),
            "created": self.created,
            "modified": _utcnow(),
            "notes": self.notes,
            "utc_offset": self.utc_offset,
            "crs": self.crs,
            "source": {
                "kind": self.source_kind,
                "raw_dir": self.raw_dir,
                "station_file": self.station_file,
                "topography_file": self.topography_file,
            },
            "params": asdict(self.params),
            "regression": asdict(self.regression),
            "sites": [_site_to_dict(s) for s in self.sites],
            "wells": [asdict(w) for w in self.wells],
            "profiles": [asdict(p) for p in self.profiles],
            "history": self.history,
        }

    def save(self) -> Path:
        self.ensure_tree()
        self.modified = _utcnow()
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.json_path)
        return self.json_path

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        if path.is_dir():
            path = path / "project.json"
        data = json.loads(path.read_text(encoding="utf-8"))

        proj = cls(path.parent, data.get("name", path.parent.name))
        proj.created = data.get("created", proj.created)
        proj.modified = data.get("modified", proj.modified)
        proj.notes = data.get("notes", "")
        proj.utc_offset = float(data.get("utc_offset", 0.0))
        proj.crs = data.get("crs", {}) or {}

        src = data.get("source", {})
        proj.source_kind = src.get("kind", "mseed")
        proj.raw_dir = src.get("raw_dir", "")
        proj.station_file = src.get("station_file", "")
        proj.topography_file = src.get("topography_file", "")

        proj.params = _from_dict(ProcParams, data.get("params", {}))
        proj.regression = _from_dict(Regression, data.get("regression", {}))
        proj.sites = [_site_from_dict(d) for d in data.get("sites", [])]
        proj.wells = [_from_dict(Well, d) for d in data.get("wells", [])]
        proj.profiles = [_from_dict(Profile, d) for d in data.get("profiles", [])]
        proj.history = data.get("history", [])
        return proj


def _site_to_dict(site: Site) -> dict[str, Any]:
    d = asdict(site)
    d["time"] = asdict(site.time)
    return d


def _site_from_dict(data: dict[str, Any]) -> Site:
    time = _from_dict(TimeSelection, data.get("time", {}) or {})
    site = _from_dict(Site, {k: v for k, v in data.items() if k != "time"})
    site.time = time
    return site


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))
