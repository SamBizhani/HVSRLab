"""Running the whole survey.

One site at a time is the interactive case; a hundred and two is the real one.
This module does that with a thread pool.

Threads rather than processes, deliberately. Almost all the time is spent in
code that releases the GIL — ObsPy decoding MiniSEED records, SciPy's
anti-alias filter, NumPy's FFT, and the BLAS call behind the Konno-Ohmachi
matrix product — so threads scale nearly as well as processes here while
keeping one address space, one progress stream, and no pickling of results
across a boundary. The default worker count is deliberately modest: past four
or so the survey disk, not the CPU, is the limit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Callable, Iterable

import numpy as np

from .core import hvsr as hvsr_core
from .core import picking, sesame, timeselect
from .io import mseed
from .jobs import Cancelled, Job
from .project import Project, Site

DEFAULT_WORKERS = 4


@dataclass
class SiteOutcome:
    sid: str
    ok: bool = False
    f0: float = float("nan")
    a0: float = float("nan")
    sesame: str = ""
    n_ok: int = 0
    seconds: float = 0.0
    error: str = ""


@dataclass
class BatchReport:
    outcomes: list[SiteOutcome] = field(default_factory=list)
    started: str = ""
    finished: str = ""

    @property
    def succeeded(self) -> list[SiteOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[SiteOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def summary(self) -> str:
        return (f"{len(self.succeeded)} of {len(self.outcomes)} sites computed"
                + (f"; {len(self.failed)} failed" if self.failed else ""))


def apply_result_to_site(site: Site, result, params, *, keep_user_pick: bool = True
                         ) -> None:
    """Copy a result's headline numbers onto its site.

    Shared by the interactive path and the batch so a site looks the same
    however it was computed. An f₀ the user picked by hand is preserved unless
    the caller says otherwise — a batch re-run should not silently discard
    an hour of picking.
    """
    f0 = site.f0 if (keep_user_pick and site.f0_source == "user"
                     and np.isfinite(site.f0)) else None
    peak = (picking.pick_nearest(result.freq, result.hv, f0) if f0
            else picking.main_peak(result.freq, result.hv,
                                   fmin=params.freq_min, fmax=params.freq_max))
    if peak is not None:
        site.f0, site.a0 = peak.frequency, peak.amplitude
        if not site.f0_source:
            site.f0_source = "auto"
        report = sesame.evaluate(result, peak.frequency)
        site.sesame_score = report.summary
        stats = picking.window_peak_statistics(result.window_f0, result.ok)
        site.f0_std = stats.get("std", float("nan"))
        result.meta["f0"] = site.f0

    site.n_windows = result.n_windows
    site.n_windows_ok = result.n_ok
    site.computed = datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_site(project: Project, site: Site,
                 recording: mseed.SiteRecording, *,
                 choose_window: bool = False,
                 scan_budget: int = 120,
                 log: Callable[[str], None] | None = None) -> SiteOutcome:
    """Load, compute and save one site. Returns what happened, never raises."""
    import time

    started = time.time()
    outcome = SiteOutcome(sid=site.sid)

    def say(text: str) -> None:
        if log is not None:
            log(text)

    try:
        params = site.effective_params(project.params)

        if choose_window or not site.time.is_set():
            hours = site.time.hours or 8.0
            if choose_window:
                block, _, _ = timeselect.find_window(recording, hours=hours,
                                                     budget=scan_budget)
            else:
                block = timeselect.fallback_block(recording, hours)
            site.time.start = mseed.iso(block.start)
            site.time.end = mseed.iso(block.end)
            site.time.hours = hours
            site.time.mode = "auto" if choose_window else "manual"
            site.time.score = block.score
            say(f"{site.label()}: window {block.label()}")

        segment = mseed.load_segment(
            recording, mseed.to_epoch(site.time.start),
            mseed.to_epoch(site.time.end),
            target_fs=params.target_fs, detrend=params.detrend)

        result = hvsr_core.compute(segment.data, segment.fs, params,
                                   sid=site.sid, start=segment.start)
        result.save(project.result_path(site.sid))
        apply_result_to_site(site, result, params)

        outcome.ok = True
        outcome.f0, outcome.a0 = site.f0, site.a0
        outcome.sesame = site.sesame_score
        outcome.n_ok = result.n_ok
    except Exception as exc:                           # noqa: BLE001
        outcome.error = f"{type(exc).__name__}: {exc}"
        say(f"{site.label()}: FAILED — {outcome.error}")
    finally:
        outcome.seconds = time.time() - started
    return outcome


def run(project: Project, sites: Iterable[Site],
        recordings: dict[str, mseed.SiteRecording], *,
        workers: int = DEFAULT_WORKERS,
        choose_window: bool = False,
        scan_budget: int = 120,
        job: Job | None = None) -> BatchReport:
    """Compute every site in *sites*, in parallel.

    Results are written to the project's ``results`` directory as they finish,
    so a run that is cancelled or killed keeps everything it had already done.
    """
    sites = [s for s in sites if s.status != "excluded"]
    report = BatchReport(started=datetime.now(timezone.utc).isoformat(
        timespec="seconds"))
    if not sites:
        return report

    lock = threading.Lock()
    done = 0
    total = len(sites)

    def log(text: str) -> None:
        if job is not None:
            with lock:
                job.log_line(text)

    def one(site: Site) -> SiteOutcome:
        if job is not None:
            job.check_cancel()
        recording = recordings.get(site.sid)
        if recording is None:
            return SiteOutcome(sid=site.sid,
                               error="no MiniSEED catalogue for this site")
        return compute_site(project, site, recording,
                            choose_window=choose_window,
                            scan_budget=scan_budget, log=log)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(one, site): site for site in sites}
        try:
            for future in as_completed(futures):
                site = futures[future]
                try:
                    outcome = future.result()
                except Cancelled:
                    break
                except Exception as exc:               # noqa: BLE001
                    outcome = SiteOutcome(sid=site.sid, error=str(exc))
                report.outcomes.append(outcome)

                done += 1
                if job is not None:
                    status = (f"f₀ = {outcome.f0:.3g} Hz  {outcome.sesame}"
                              if outcome.ok else outcome.error)
                    job.log_line(f"[{done}/{total}] {site.label()}  {status}")
                    job.counted(done, total, "computing sites")
                    if job.cancelled:
                        for pending in futures:
                            pending.cancel()
                        break
        finally:
            report.finished = datetime.now(timezone.utc).isoformat(
                timespec="seconds")

    project.log("batch", sites=total, succeeded=len(report.succeeded),
                failed=len(report.failed))
    return report
