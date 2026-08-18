"""Choosing which hours of a long deployment to run H/V on.

An ambient-noise tomography deployment leaves weeks of continuous recording per
site. H/V does not want weeks. The number of significant cycles behind a peak
is ``Lw · nw · f0``; SESAME asks for more than 200, and at f0 = 1 Hz with 60 s
windows that is satisfied by four minutes of clean data. Everything past a few
hours buys precision that the site-to-site variability swamps anyway — while
costing a factor of a hundred in I/O and in the operator's patience.

What *does* matter is which hours. Ambient noise is strongly non-stationary on
a daily cycle: traffic, machinery and wind raise the amplitude by an order of
magnitude and, worse, make it transient rather than diffuse. The H/V peak
frequency is usually robust to this, but the amplitude and the window-to-window
scatter are not — and the scatter is what SESAME's reliability tests judge.

So this module runs a cheap reconnaissance over the whole recording — short
probes at wide spacing, tens of milliseconds each — builds the diurnal
amplitude pattern, and ranks candidate blocks by how quiet *and* how steady
they are. The steadiness term matters as much as the level: a block whose
amplitude is low but lurching is worse for H/V than one slightly louder and
stationary.

The result is a proposal, not a verdict. The GUI shows the whole scan with the
chosen block marked, so the choice can be overruled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from ..io import mseed

Progress = Callable[[int, int, str], None]


@dataclass
class NoiseSurvey:
    """Amplitude reconnaissance over a whole recording."""

    sid: str = ""
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))   # epoch
    rms: dict[str, np.ndarray] = field(default_factory=dict)
    probe_s: float = 60.0
    every_s: float = 1800.0

    @property
    def n(self) -> int:
        return int(self.times.size)

    def level(self) -> np.ndarray:
        """One amplitude per probe: the geometric mean over components.

        Geometric, because component amplitudes span orders of magnitude and a
        single loud horizontal should not by itself condemn a probe.
        """
        if not self.rms:
            return np.zeros(0)
        stack = np.vstack([v for v in self.rms.values()])
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log(np.where(stack > 0, stack, np.nan))
        return np.exp(np.nanmean(logs, axis=0))

    def hours_of_day(self) -> np.ndarray:
        if not self.n:
            return np.zeros(0)
        return np.array([
            datetime.fromtimestamp(t, timezone.utc).hour
            + datetime.fromtimestamp(t, timezone.utc).minute / 60.0
            for t in self.times])

    def diurnal(self, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """Median amplitude against hour of day (UTC)."""
        level = self.level()
        hours = self.hours_of_day()
        centres = np.arange(bins) * (24.0 / bins) + 12.0 / bins
        out = np.full(bins, np.nan)
        if not level.size:
            return centres, out
        edges = np.linspace(0, 24, bins + 1)
        for i in range(bins):
            sel = (hours >= edges[i]) & (hours < edges[i + 1]) & np.isfinite(level)
            if sel.any():
                out[i] = float(np.median(level[sel]))
        return centres, out


@dataclass
class Block:
    """A candidate stretch of recording."""

    start: float
    end: float
    score: float = float("nan")
    level: float = float("nan")        # median amplitude
    steadiness: float = float("nan")   # spread of log amplitude — lower is steadier
    n_probes: int = 0
    coverage: float = 1.0              # fraction of expected probes that returned data

    @property
    def hours(self) -> float:
        return (self.end - self.start) / 3600.0

    def label(self) -> str:
        return (f"{mseed.iso(self.start)} -> {mseed.iso(self.end)} "
                f"({self.hours:.1f} h)")


def survey(rec: mseed.SiteRecording, *, probe_s: float = 30.0,
           every_s: float = 0.0, start: float | None = None,
           end: float | None = None, max_probes: int = 160,
           components: tuple[str, ...] = ("Z",),
           progress: Progress | None = None) -> NoiseSurvey:
    """Probe the recording at intervals and measure the amplitude of each probe.

    ``every_s=0`` spreads *max_probes* evenly over the whole span, which is the
    useful default: the cost is then fixed and predictable regardless of
    whether the deployment ran for two days or six weeks.

    Probing the vertical alone is enough to rank hours by quietness and costs a
    third of the reads; pass ``components=mseed.COMPONENTS`` when the
    horizontals matter — near a road, where traffic loads the horizontals far
    more than the vertical, they do.
    """
    t0, t1 = rec.common_span()
    if start is not None:
        t0 = max(t0, start)
    if end is not None:
        t1 = min(t1, end)
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        raise ValueError(f"{rec.sid}: no common three-component coverage to scan")

    span = t1 - t0
    if every_s <= 0:
        every_s = max(probe_s, span / max(1, max_probes))
    elif span / every_s > max_probes:
        every_s = span / max_probes
    times = np.arange(t0, max(t0 + probe_s, t1 - probe_s), every_s)

    out = NoiseSurvey(sid=rec.sid, times=times, probe_s=probe_s,
                      every_s=every_s,
                      rms={c: np.full(times.size, np.nan) for c in components})
    total = times.size
    for i, t in enumerate(times):
        if progress is not None and (i % 5 == 0 or i == total - 1):
            progress(i + 1, total, mseed.iso(t))
        values = mseed.probe_rms(rec, float(t), probe_s, components=components)
        for comp, v in values.items():
            if comp in out.rms:
                out.rms[comp][i] = v
    return out


def find_window(rec: mseed.SiteRecording, hours: float = 8.0, *,
                budget: int = 160, probe_s: float = 30.0,
                components: tuple[str, ...] = ("Z",),
                steadiness_weight: float = 0.5,
                progress: Progress | None = None
                ) -> tuple[Block, NoiseSurvey, NoiseSurvey]:
    """Two-stage search for the best *hours*-long block, on a fixed probe budget.

    A single-pass scan cannot do both jobs: to compare blocks you need several
    probes inside each one, and to cover six weeks at that spacing costs
    thousands of reads. So the budget is split — half to a coarse pass that
    finds the quietest *day*, half to a fine pass inside a window around it that
    resolves individual blocks.

    Returns ``(block, coarse, fine)``; both scans come back so the GUI can draw
    the whole record with the chosen block marked on it.
    """
    half = max(20, budget // 2)

    def stage(tag: str):
        if progress is None:
            return None
        return lambda i, n, msg: progress(i, n, f"{tag}: {msg}")

    coarse = survey(rec, probe_s=probe_s, max_probes=half,
                    components=components, progress=stage("scanning record"))

    span_start, span_end = rec.common_span()
    duration = hours * 3600.0
    focus_width = max(3.0 * duration, 86400.0)
    centre = _quietest_centre(coarse, focus_width)
    f0 = min(max(span_start, centre - focus_width / 2), max(span_start, span_end - focus_width))
    f1 = min(span_end, f0 + focus_width)

    fine = survey(rec, probe_s=probe_s, start=f0, end=f1, max_probes=half,
                  components=components, progress=stage("refining"))

    block = best_block(fine, hours, steadiness_weight=steadiness_weight)
    if block is None:
        block = best_block(coarse, hours, steadiness_weight=steadiness_weight)
    if block is None:
        block = fallback_block(rec, hours)
    return clip_to_coverage(rec, block), coarse, fine


def _quietest_centre(scan: NoiseSurvey, width: float) -> float:
    """Centre of the *width*-long stretch with the lowest median amplitude."""
    level = scan.level()
    finite = np.isfinite(level)
    if not finite.any():
        return float(np.mean(scan.times)) if scan.n else float("nan")

    best_score, best_centre = np.inf, float(scan.times[0] + width / 2)
    for t in scan.times:
        sel = (scan.times >= t) & (scan.times < t + width) & finite
        if sel.sum() < 2:
            continue
        score = float(np.median(np.log(level[sel])))
        if score < best_score:
            best_score, best_centre = score, float(t + width / 2)
    return best_centre


def rank_blocks(scan: NoiseSurvey, hours: float, *, step_s: float = 3600.0,
                steadiness_weight: float = 0.5,
                min_coverage: float = 0.9) -> list[Block]:
    """Score every candidate block of *hours*, quietest and steadiest first.

    The score combines two standardised terms, both in log amplitude: the
    median level, and the interquartile spread. A block scores well by being
    quiet *and* stationary. ``steadiness_weight`` sets the balance — raise it
    when the survey is dominated by intermittent machinery, lower it when the
    background is steady and only the level varies.

    Blocks missing more than ``1 - min_coverage`` of their probes are dropped:
    a gap in the recording would otherwise read as beautiful silence.
    """
    level = scan.level()
    if scan.n < 3 or not np.isfinite(level).any():
        return []

    with np.errstate(divide="ignore", invalid="ignore"):
        log_level = np.log(np.where(level > 0, level, np.nan))

    duration = hours * 3600.0
    expected = max(1.0, duration / scan.every_s)
    step_s = min(step_s, max(scan.every_s, duration / 8.0))
    starts = np.arange(scan.times[0], scan.times[-1] - duration + 1.0, step_s)
    if starts.size == 0:
        starts = np.array([scan.times[0]])

    blocks: list[Block] = []
    for s in starts:
        sel = (scan.times >= s) & (scan.times < s + duration)
        values = log_level[sel]
        finite = values[np.isfinite(values)]
        coverage = finite.size / expected
        if coverage < min_coverage or finite.size < 2:
            continue
        # With only two probes in a block the spread is not an estimate of
        # anything; score it on level alone rather than on noise.
        q75, q25 = (np.percentile(finite, [75, 25]) if finite.size >= 4
                    else (np.max(finite), np.min(finite)))
        blocks.append(Block(
            start=float(s), end=float(s + duration),
            level=float(np.exp(np.median(finite))),
            steadiness=float(q75 - q25) if finite.size >= 3 else 0.0,
            n_probes=int(finite.size), coverage=float(min(1.0, coverage))))

    if not blocks:
        return []

    med = np.array([np.log(b.level) for b in blocks])
    spread = np.array([b.steadiness for b in blocks])
    for b, m, s in zip(blocks, _standardise(med), _standardise(spread)):
        b.score = float((1.0 - steadiness_weight) * m + steadiness_weight * s)
    blocks.sort(key=lambda b: b.score)
    return blocks


def _standardise(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-spread, robust to the one pathological block."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return np.zeros_like(values)
    centre = float(np.median(finite))
    scale = float(np.percentile(finite, 75) - np.percentile(finite, 25)) or 1.0
    return (values - centre) / scale


def best_block(scan: NoiseSurvey, hours: float, **kwargs) -> Block | None:
    blocks = rank_blocks(scan, hours, **kwargs)
    return blocks[0] if blocks else None


def fallback_block(rec: mseed.SiteRecording, hours: float) -> Block:
    """A block chosen without scanning: the first full local night in the record.

    Used when a survey would cost more than it is worth — a short recording, or
    a batch run in a hurry. Local midnight to 04:00 is the quietest stretch at
    most sites; without a scan this is an assumption, and it is labelled as one
    on the returned block.
    """
    t0, t1 = rec.common_span()
    if not (np.isfinite(t0) and np.isfinite(t1)):
        raise ValueError(f"{rec.sid}: no coverage")

    duration = hours * 3600.0
    day = datetime.fromtimestamp(t0, timezone.utc)
    midnight = datetime(day.year, day.month, day.day, 0, 0,
                        tzinfo=timezone.utc).timestamp()
    candidate = midnight
    while candidate < t0:
        candidate += 86400.0
    if candidate + duration > t1:
        candidate = max(t0, t1 - duration)
    return Block(start=float(candidate), end=float(candidate + duration),
                 score=float("nan"), coverage=float("nan"))


def clip_to_coverage(rec: mseed.SiteRecording, block: Block) -> Block:
    """Pull a block inside the recording's actual three-component coverage."""
    t0, t1 = rec.common_span()
    duration = block.end - block.start
    start = min(max(block.start, t0), max(t0, t1 - duration))
    return Block(start=float(start), end=float(min(start + duration, t1)),
                 score=block.score, level=block.level,
                 steadiness=block.steadiness, n_probes=block.n_probes,
                 coverage=block.coverage)
