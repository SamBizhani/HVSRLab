"""Shared application state.

Every page reads and writes one :class:`Workspace`. It owns the project, the
job queue, and the caches that keep the interface responsive: the MiniSEED
catalogue, the loaded segment for the site being worked on, and the computed
results. Pages never talk to each other — they change the workspace and listen
to its signals.

Caches are bounded. A 102-site survey's results are ~2 MB each in memory, so
holding all of them would be a few hundred megabytes for no benefit: what is on
screen is one site, and the maps need only the headline numbers, which live in
``project.json``.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from .. import paths
from ..core import hvsr as hvsr_core
from ..core import timeselect
from ..io import crs as crs_io
from ..io import mseed, stations as station_io
from ..jobs import Job, JobQueue
from ..project import Project, ProcParams, Site
from .widgets import JobBridge

#: How many computed results to keep in memory at once.
RESULT_CACHE = 8

#: How many loaded waveform segments to keep. These are the expensive ones —
#: eight hours at 50 Hz is 1.4 M samples per component.
SEGMENT_CACHE = 3


class Workspace(QObject):
    """The application's single source of truth."""

    projectChanged = pyqtSignal()          # a different project was opened
    sitesChanged = pyqtSignal()            # the site list or its values changed
    currentChanged = pyqtSignal(str)       # a different site is selected
    resultChanged = pyqtSignal(str)        # a site's H/V result changed
    segmentChanged = pyqtSignal(str)       # a site's loaded waveform changed
    notify = pyqtSignal(str, str)          # message, tone
    logged = pyqtSignal(str, str)          # text, level — straight to Activity
    jobFinished = pyqtSignal(object)       # emitted from the worker thread

    def __init__(self, project: Project | None = None) -> None:
        super().__init__()
        self.project = project or Project(paths.projects_dir() / "untitled")
        self.queue = JobQueue()
        self.bridge = JobBridge()
        self._completions: dict[int, Callable[[Job], None]] = {}
        self.jobFinished.connect(self._dispatch_completion)

        self.recordings: dict[str, mseed.SiteRecording] = {}
        self.scans: dict[str, tuple] = {}          # sid -> (coarse, fine, block)
        self._results: "OrderedDict[str, hvsr_core.HVSRResult]" = OrderedDict()
        self._segments: "OrderedDict[str, mseed.Segment]" = OrderedDict()
        self.current_sid: str = (self.project.sites[0].sid
                                 if self.project.sites else "")
        self.dirty = False
        self.assign_plan_coordinates()

    # -- project -----------------------------------------------------------
    def set_project(self, project: Project) -> None:
        self.project = project
        self.recordings.clear()
        self.scans.clear()
        self._results.clear()
        self._segments.clear()
        self.current_sid = project.sites[0].sid if project.sites else ""
        self.dirty = False
        # Plan coordinates are derived, so re-deriving them on open is both
        # correct and idempotent -- and it migrates a project written before
        # the grid was recorded.
        self.assign_plan_coordinates()
        self.projectChanged.emit()
        self.sitesChanged.emit()
        self.currentChanged.emit(self.current_sid)

    def save(self) -> Path:
        path = self.project.save()
        self.dirty = False
        self.notify.emit(f"Saved {self.project.name}", "good")
        return path

    def touch(self) -> None:
        """Mark the project as having unsaved changes."""
        self.dirty = True

    def log(self, text: str, level: str = "info") -> None:
        """Put a line in the Activity panel without disturbing the status bar.

        Safe to call from a worker thread: the signal is delivered queued.
        """
        self.logged.emit(str(text), level)

    @property
    def utc_offset(self) -> float:
        return float(getattr(self.project, "utc_offset", 0.0))

    # -- sites -------------------------------------------------------------
    @property
    def site(self) -> Site | None:
        return self.project.site(self.current_sid) if self.current_sid else None

    def set_current(self, sid: str) -> None:
        if sid == self.current_sid:
            return
        self.current_sid = sid
        self.currentChanged.emit(sid)

    def params_for(self, site: Site | None = None) -> ProcParams:
        site = site or self.site
        if site is None:
            return self.project.params
        return site.effective_params(self.project.params)

    def coordinates(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Plan coordinates of every non-excluded site, plus their labels."""
        sites = [s for s in self.project.sites if s.status != "excluded"]
        x = np.array([s.x for s in sites], dtype=float)
        y = np.array([s.y for s in sites], dtype=float)
        return x, y, [s.label() for s in sites]

    # -- coordinates -------------------------------------------------------
    @property
    def zone(self):
        """The project's UTM zone, or None before coordinates arrive."""
        return crs_io.UTM.from_dict(self.project.crs)

    def set_zone(self, zone) -> None:
        self.project.crs = zone.to_dict() if zone is not None else {}
        self.assign_plan_coordinates()
        self.touch()

    def ensure_zone(self):
        """Pick a zone from the sites if the project has none yet."""
        zone = self.zone
        if zone is not None:
            return zone
        lats = np.array([s.lat for s in self.project.sites], dtype=float)
        lons = np.array([s.lon for s in self.project.sites], dtype=float)
        zone = crs_io.zone_for_survey(lats, lons)
        if zone is not None:
            self.project.crs = zone.to_dict()
        return zone

    def assign_plan_coordinates(self) -> None:
        """Fill every site's x/y with UTM easting/northing.

        One zone for the whole project, chosen from its centroid: sites in a
        neighbouring zone are projected into it rather than getting their own,
        because a profile whose two ends are on different grids has no
        meaningful length.
        """
        sites = self.project.sites
        if not sites:
            return
        zone = self.ensure_zone()
        if zone is None:
            return

        lats = np.array([s.lat for s in sites], dtype=float)
        lons = np.array([s.lon for s in sites], dtype=float)
        if np.isfinite(lats).any():
            easting, northing, _ = crs_io.to_utm(lats, lons, zone)
            for site, e, n in zip(sites, easting, northing):
                if np.isfinite(e):
                    site.x, site.y = float(e), float(n)
                if not np.isfinite(site.z):
                    site.z = site.elev

        # A site read from a file that was already in metres has no degrees;
        # give it some, so it can be shown on a web map or exported as WGS 84.
        orphans = [s for s in sites
                   if not np.isfinite(s.lat) and np.isfinite(s.x)]
        if orphans:
            lat, lon = crs_io.from_utm([s.x for s in orphans],
                                       [s.y for s in orphans], zone)
            for site, la, lo in zip(orphans, lat, lon):
                site.lat, site.lon = float(la), float(lo)

        self.locate_wells()

    #: Kept under the old name so existing callers still work.
    assign_local_coordinates = assign_plan_coordinates

    def zone_warning(self) -> str:
        """A note when the survey straddles more than one natural zone."""
        zone = self.zone
        if zone is None:
            return ""
        lats = np.array([s.lat for s in self.project.sites], dtype=float)
        lons = np.array([s.lon for s in self.project.sites], dtype=float)
        spans = crs_io.zone_span(lats, lons)
        if len(spans) > 1:
            others = ", ".join(str(z) for z in spans if z != zone.zone)
            return (f"Some sites fall naturally in zone(s) {others}; they are "
                    f"projected into {zone.name} so the survey stays on one "
                    f"grid. Distortion grows with distance from the zone.")
        return ""

    def locate_wells(self) -> int:
        """Put every well on the project's grid. Returns how many were placed.

        Wells and sites must share a projection or a borehole 200 m from a
        station plots kilometres away. A well given only easting/northing is
        taken as already being on the grid, and its latitude and longitude are
        filled in from it.
        """
        wells = self.project.wells
        if not wells:
            return 0
        zone = self.ensure_zone()
        if zone is None:
            return 0

        placed = 0
        for well in wells:
            if np.isfinite(well.lat) and np.isfinite(well.lon):
                e, n, _ = crs_io.to_utm([well.lat], [well.lon], zone)
                well.x, well.y = float(e[0]), float(n[0])
                placed += 1
            elif np.isfinite(well.x) and np.isfinite(well.y):
                lat, lon = crs_io.from_utm([well.x], [well.y], zone)
                well.lat, well.lon = float(lat[0]), float(lon[0])
                placed += 1
        self.link_wells()
        return placed

    def nearest_site(self, x: float, y: float, limit: float = 1e9):
        """The site closest to a plan position, within *limit* metres."""
        best, best_distance = None, limit
        for site in self.project.sites:
            if site.status == "excluded" or not np.isfinite(site.x):
                continue
            d = float(np.hypot(site.x - x, site.y - y))
            if d < best_distance:
                best, best_distance = site, d
        return best, best_distance

    def link_wells(self, *, force: bool = False) -> list[tuple]:
        """Tie every borehole to its nearest site and record the separation.

        Boreholes are drilled near stations, not on them, so the link is made
        by proximity with no distance cut: refusing to link a hole 400 m from
        the nearest station would just leave the operator to do the same sum by
        hand. What matters is that the distance travels with the link, and it
        does — the table shows it, and a large one is flagged, because tying a
        depth to an f₀ a kilometre away assumes the basin between them is flat.

        Wells whose site was chosen by hand keep it unless *force*. Returns
        ``(well, site, distance)`` for everything it touched.
        """
        touched: list[tuple] = []
        for well in self.project.wells:
            if well.link_mode == "user" and not force and well.site:
                site = self.project.site(well.site)
                well.link_distance = (
                    float(np.hypot(site.x - well.x, site.y - well.y))
                    if site is not None and np.isfinite(site.x)
                    and np.isfinite(well.x) else float("nan"))
                continue
            if not (np.isfinite(well.x) and np.isfinite(well.y)):
                continue
            site, distance = self.nearest_site(well.x, well.y)
            if site is None:
                continue
            well.site = site.sid
            well.link_distance = distance
            if force:
                well.link_mode = "auto"
            touched.append((well, site, distance))
        return touched

    # -- recordings --------------------------------------------------------
    def recording(self, sid: str) -> mseed.SiteRecording | None:
        return self.recordings.get(sid)

    def set_recordings(self, records: list[mseed.SiteRecording]) -> None:
        self.recordings = {r.sid: r for r in records}

    # -- segments ----------------------------------------------------------
    def segment(self, sid: str) -> mseed.Segment | None:
        seg = self._segments.get(sid)
        if seg is not None:
            self._segments.move_to_end(sid)
        return seg

    def set_segment(self, sid: str, segment: mseed.Segment) -> None:
        self._segments[sid] = segment
        self._segments.move_to_end(sid)
        while len(self._segments) > SEGMENT_CACHE:
            self._segments.popitem(last=False)
        self.segmentChanged.emit(sid)

    def load_segment(self, site: Site, job: Job | None = None
                     ) -> mseed.Segment:
        """Read the site's selected time window, decimated per its parameters."""
        rec = self.recordings.get(site.sid)
        if rec is None:
            raise ValueError(
                f"{site.label()}: no MiniSEED catalogue — scan the raw data "
                "directory on the Sites page first")

        params = self.params_for(site)
        if not site.time.is_set():
            block = timeselect.fallback_block(rec, site.time.hours or 8.0)
            site.time.start = mseed.iso(block.start)
            site.time.end = mseed.iso(block.end)
            site.time.mode = "auto"
            if job is not None:
                job.log_line(f"{site.label()}: no window chosen; using the first "
                             f"night in the record ({block.label()})")

        t0 = mseed.to_epoch(site.time.start)
        t1 = mseed.to_epoch(site.time.end)
        segment = mseed.load_segment(rec, t0, t1, target_fs=params.target_fs,
                                     detrend=params.detrend)
        self.set_segment(site.sid, segment)
        return segment

    # -- results -----------------------------------------------------------
    def result(self, sid: str, *, load: bool = True) -> hvsr_core.HVSRResult | None:
        res = self._results.get(sid)
        if res is not None:
            self._results.move_to_end(sid)
            return res
        if not load:
            return None
        path = self.project.result_path(sid)
        if path.exists():
            try:
                res = hvsr_core.HVSRResult.load(path)
            except Exception as exc:                       # noqa: BLE001
                self.notify.emit(f"Could not read {path.name}: {exc}", "bad")
                return None
            self._cache_result(sid, res)
            return res
        return None

    def set_result(self, sid: str, result: hvsr_core.HVSRResult, *,
                   persist: bool = True) -> None:
        self._cache_result(sid, result)
        if persist:
            result.save(self.project.result_path(sid))
        self._sync_site_from_result(sid, result)
        self.touch()
        self.resultChanged.emit(sid)

    def _cache_result(self, sid: str, result: hvsr_core.HVSRResult) -> None:
        self._results[sid] = result
        self._results.move_to_end(sid)
        while len(self._results) > RESULT_CACHE:
            self._results.popitem(last=False)

    def drop_result(self, sid: str) -> None:
        self._results.pop(sid, None)
        path = self.project.result_path(sid)
        if path.exists():
            path.unlink()
        site = self.project.site(sid)
        if site is not None:
            site.f0 = site.a0 = site.f0_std = float("nan")
            site.sesame_score = ""
            site.computed = ""
        self.touch()
        self.resultChanged.emit(sid)

    def _sync_site_from_result(self, sid: str, result) -> None:
        """Copy the headline numbers into the project so maps can draw fast.

        Shared with the batch runner so a site looks identical however it was
        computed.
        """
        from ..batch import apply_result_to_site

        site = self.project.site(sid)
        if site is not None:
            apply_result_to_site(site, result, self.params_for(site))

    def computed_count(self) -> int:
        return sum(1 for s in self.project.sites if np.isfinite(s.f0))

    # -- jobs --------------------------------------------------------------
    def submit(self, name: str, target: Callable[[Job], object], *,
               on_done: Callable[[Job], None] | None = None) -> Job:
        """Queue background work, wiring its callbacks to the Qt event loop.

        ``on_done`` runs on the *main* thread once the job finishes, so it may
        touch widgets. It gets there by way of :attr:`jobFinished`, which is
        emitted from the worker thread and delivered queued — the one safe way
        to cross that boundary.
        """
        job = Job(name=name, target=target)
        self.bridge.attach(job)
        if on_done is not None:
            self._completions[id(job)] = on_done
        job.on_state(lambda j: self.jobFinished.emit(j) if j.state.finished else None)
        return self.queue.submit(job)

    def _dispatch_completion(self, job: Job) -> None:
        callback = self._completions.pop(id(job), None)
        if callback is not None:
            callback(job)
