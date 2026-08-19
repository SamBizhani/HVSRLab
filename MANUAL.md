# HVSRLab — User Manual

Version 1.0.0

A working guide to computing H/V spectral ratios from passive seismic
recordings, judging the result against SESAME 2004, and turning it into a
bedrock map.

If you have never used the program before, read [First run](#first-run) and
[The workflow](#the-workflow) and ignore the rest until you need it.

---

## Contents

1. [Installing](#installing)
2. [First run](#first-run)
3. [How the program is organised](#how-the-program-is-organised)
4. [The workflow](#the-workflow)
   - [Overview](#1-overview)
   - [Sites](#2-sites)
   - [Data & Windows](#3-data--windows)
   - [H/V Analysis](#4-hv-analysis)
   - [Maps & Profiles](#5-maps--profiles)
   - [3D Views](#6-3d-views)
   - [Bedrock](#7-bedrock)
   - [1D Model](#8-1d-model)
   - [Batch & Export](#9-batch--export)
5. [Parameter reference](#parameter-reference)
6. [Command line](#command-line)
7. [Files on disk](#files-on-disk)
8. [Troubleshooting](#troubleshooting)
9. [Reading your results honestly](#reading-your-results-honestly)

---

## Installing

**Requirements:** Python 3.9 or newer, and `numpy`, `scipy`, `matplotlib`,
`obspy`, `PyQt5`.

`pyproj` is optional. Without it a built-in transverse-Mercator series does the
UTM conversion instead, agreeing with `pyproj` to 0.03 mm.

```bash
pip install numpy scipy matplotlib obspy PyQt5
```

Check the environment before anything else:

```bash
python run.py --check
```

This prints every dependency with its version, the projection backend in use,
the projects directory and the CPU count, and it names anything missing. If it
reports `obspy MISSING`, install it and run the check again.

One version constraint: if your PyQt5 ships Qt 5.9, matplotlib must stay below
3.6. Newer Qt has no such limit. `--check` warns you if this applies.

---

## First run

```bash
cd path/to/HVSRLab
python run.py
```

With no arguments the program opens the most recently modified project, or
creates an empty one called `untitled` if there are none.

To open a specific project:

```bash
python run.py path/to/HVSRLab_Projects/<name>
```

Projects live in `HVSRLab_Projects/` beside the application directory by
default. Set the `HVSRLAB_PROJECTS` environment variable to put them elsewhere.

**Save often.** There is a **Save project** button on the Overview page and
`Ctrl+S` works everywhere. Computation results are written to disk as they
finish, but site status, picks, profiles and boreholes live in `project.json`
and are only written when you save.

---

## How the program is organised

Three ideas carry through the whole interface.

**A project** is one survey: its raw data location, its processing parameters,
its sites, its boreholes and its profiles. It is a folder containing
`project.json` and the results computed from it.

**A site** is one measurement point. It has coordinates, a status, a time
selection saying which hours of its recording to analyse, and — once computed —
an f₀, an A₀ and a SESAME verdict.

**A result** is the per-window arrays behind one site's curve, stored as
`results/<site>.npz`. Results are self-contained: once a site is computed you
can inspect, re-pick and export it without the raw data being present.

The nine pages in the left sidebar run roughly in workflow order. You can move
between them freely; the program does not force a sequence.

---

## The workflow

### 1. Overview

Survey status, the layout of the sites, and the distribution of whatever has
been computed so far. Nothing is set here — it is where you look to see where
things stand.

**Save project** lives in the header.

---

### 2. Sites

*Where the survey is defined: raw data in, measurement points out.*

This page is the starting point for a new project.

**Data source**

| Control | What it does |
|---|---|
| **Scan raw data** | Point it at the root folder of your MiniSEED. It catalogues every file and creates one site per recording. |
| **Import H/V curves** | Use instead of scanning when you already have computed curves — SAF, Geopsy or ProTO format. |
| **Load boreholes** | Depth control for the Bedrock page. Can also be done there. |
| **Browse…** | Sets the raw data root directly. |

The scan reads file *names* where it can and record headers only where it must,
so a twelve-thousand-file survey catalogues in seconds. Folder-per-site, flat
folders and arbitrary nesting are all handled.

**Measurement points**

The site table. Each row is one site: name, coordinates, status, and results
once computed.

- **Match coordinates** — load a station list and attach coordinates to the
  scanned recordings. The reader detects the separator and column order itself,
  and identifies latitude and longitude by plausibility rather than position,
  so a station serial is never mistaken for a coordinate.
- **Set time zone…** — the UTC offset for the survey. This affects how times
  are displayed and how "local night" is reported; it does not change any
  computation.
- **Grid…** — the interpolation grid used by Maps and 3D Views.

Set a site's status here to exclude bad stations from every later step.

**Coordinates.** Latitude and longitude are converted automatically to the UTM
zone of the survey's own centroid, and that zone is recorded on the project so
every later session lands on the same grid. Files already in metres are
recognised and read directly. A survey straddling a zone boundary is projected
into one zone with a warning, because a profile whose ends sit on different
grids has no meaningful length.

---

### 3. Data & Windows

*Which hours to analyse, and which windows inside them.*

Two decisions happen here, and the first matters more than most people expect.

**Recording window — which hours**

H/V needs a few hours out of weeks of recording, and *which* hours changes the
answer. Cultural noise contaminates daytime records; the resonance you want is
clearest when the site is quiet.

- **Find the quietest window** runs a two-stage reconnaissance: short probes
  across the whole deployment, then a finer pass around the quietest day.
  Candidate blocks are ranked by how quiet *and* how steady they are. Six weeks
  is characterised in roughly 160 reads. On real data this finds local night
  without being told the time zone.
- Choose the window length first: **4**, **8**, **12** or **24** hours.
- **Use these times** accepts the suggestion. **Load** fetches that slice.
- **Apply this window to every site** propagates the choice across the survey —
  useful when sites were deployed together and share a noise regime.

The **Noise through the deployment** plot shows what the scan found, so you can
see whether the chosen block is genuinely representative or a lucky quiet hour.

**Windowing — which windows inside those hours**

The loaded segment is cut into windows (default 60 s, 50 % overlap) and each is
tested by the STA/LTA anti-trigger, which drops windows containing transients —
a vehicle, a footstep, an instrument glitch.

**Keep all** and **Drop all** override the automatic decision. Inspect what was
dropped before overriding: if the anti-trigger is rejecting most of the record,
the site is noisy and the honest response is usually a different time window,
not a looser threshold.

A site needs at least `min_windows` (default 5) surviving windows to produce a
result.

---

### 4. H/V Analysis

*The curve, the pick, and whether either can be trusted.*

**Processing** and **Pre-filter** hold the parameters for the computation — see
the [parameter reference](#parameter-reference). The pre-filter is off by
default and rarely needed; Konno-Ohmachi smoothing already handles what most
people reach for a filter to do.

**Compute this site** runs the chain: optional filter → windowing with
anti-trigger → per window demean, taper, zero-pad, FFT, amplitude →
Konno-Ohmachi smoothing of each component → combine horizontals, divide by
vertical → statistics across windows.

Smoothing happens *before* the division. This is deliberate, matches ProTO's
"OPTION-B", and is what the SESAME guidelines describe. Smoothing after
dividing gives a different curve.

**The pick.** f₀ is picked automatically. **Re-pick automatically** redoes it
after a parameter change; you can also pick by hand on the curve. Secondary
peaks can be marked, and **Clear secondary peaks** removes them.

**The SESAME checklist** is the point of this page. Three reliability
conditions establish that the *curve* is trustworthy:

1. f₀ > 10 / L<sub>w</sub> — the peak is resolved by the window length
2. n<sub>c</sub> = L<sub>w</sub> · n<sub>w</sub> · f₀ > 200 — enough significant cycles were averaged
3. amplitude scatter σ<sub>A</sub> < 2 over [0.5·f₀, 2·f₀] (< 3 if f₀ < 0.5 Hz)

Six further criteria establish that the *peak* is clear; SESAME asks for at
least five. They test that the curve falls to half amplitude on both flanks,
that the peak exceeds 2, that adding and subtracting one standard deviation
does not move it, and that frequency and amplitude scatter stay inside
f₀-dependent thresholds.

Each shows its measured value, its threshold and its verdict. **A site that
fails reliability should not have its f₀ used**, however convincing the curve
looks.

**The other tabs**

- **Component spectra** — the three smoothed spectra before division. The first
  place to look when a curve is odd: a vertical-component resonance produces a
  spurious H/V trough, and a flat vertical with a peaked horizontal is what a
  real site response looks like.
- **Azimuth** — H/V against rotation azimuth as a heat map and a polar rose,
  computed by rotating the complex spectra rather than re-transforming at each
  angle. A site with a genuinely 1D structure is round; strong directionality
  means a 2D or 3D structure and the 1D interpretation weakens.
- **Stability** — H/V against time through the recording, with the per-window
  peak below it. This is where diurnal contamination shows itself: a peak that
  wanders with the working day is not a site resonance.
- **Gallery** — every computed site side by side, for spotting outliers.

Set `azimuth_step_deg` (10 or 15 are typical) to enable the azimuth analysis;
0 turns it off.

---

### 5. Maps & Profiles

*The survey seen from above, and in section.*

**Map** — interpolated over the survey grid. Choose the quantity:

| Quantity | |
|---|---|
| `f0` | resonance frequency (Hz) |
| `a0` | peak amplitude |
| `depth` | depth to bedrock (m) — needs a law set on the Bedrock page |
| `f0_std` | σ f₀, the window scatter — effectively a reliability map |
| `slice` | H/V at a chosen frequency |
| `clarity` | SESAME clarity score |

Contour style, colour map, site labels and borehole overlay are all switchable.

Map `f0_std` alongside `f0` before believing any structure in the latter. Where
scatter is high, the f₀ contours are drawing noise.

**Profiles**

Draw a line with two clicks. It starts with the sites it actually passes —
those within half the typical station spacing — rather than the whole survey
smeared along an arbitrary line.

- **Redraw ends** — place the two ends again. *Between the two clicks the
  profile is momentarily degenerate; the program handles this and waits for the
  second click.*
- **Rename**, **Delete**, **Clear all**.
- **Sites on this profile** — arm **Pick on map** (or Ctrl+click) and click
  sites in and out. A green ring means on the section, a red cross means left
  out. **All**, **None** and **Reset** act on the whole list.

The corridor belongs to the profile, so two profiles can select differently,
and hand-picked choices survive changes to the line.

**Section** — the pseudo-section built from the chosen sites, with ProTO's four
smoothing and four normalisation strategies.

---

### 6. 3D Views

Surfaces over the survey, and H/V columns standing in place.

Surface options: bedrock depth below surface, bedrock elevation, f₀, A₀, or
topography. **Redraw** rebuilds after a change.

The H/V tile view — every site's curve standing at its own location — is the
most honest 3D display of an H/V survey, because it shows the data rather than
an interpolation of a picked number.

---

### 7. Bedrock

*Turning f₀ into depth, and calibrating the law that does it.*

Depth follows **H = a · f₀<sup>b</sup>**.

**Active law** — choose a published calibration or enter your own `a` and `b`.
Six published laws ship with the program, listed in
[REFERENCES.md](REFERENCES.md). **Apply** sets it; **Reset to Ibs-von Seht**
returns to the 1999 original (`a = 96, b = -1.388`).

**A published law carries the velocity structure of the basin it was fitted
in.** Using one elsewhere is an assumption, not a measurement.

**Borehole control** — the alternative, and the better one where you have it.

- **Add** — type a borehole in by hand.
- **Load table…** — a coordinate/depth table.
- **Load log…** — ProTO-style `lithology thickness` logs.
- **Fit a and b to these wells** — regression on your own control.

Each borehole links itself to the closest site — on load, on manual entry, and
again whenever coordinates change — and the separation is kept and shown. That
distance *is* the assumption the calibration rests on: that the basin is the
same at the hole and at the station. **Anything past 500 m is flagged.** The
fit logs the separations it relied on, and a link can be pinned by hand when
local knowledge beats proximity.

Three wells is the practical minimum for a two-parameter fit, and three
widely-separated wells will underdetermine it. Check the RMS *and* the
separations before trusting the coefficients.

---

### 8. 1D Model

*What soil column would resonate where this site does.*

A Haskell–Thomson SH transfer function for a layered column over a half-space.
Depth from physics rather than from another basin's regression.

**Layers** — thickness (m), Vs (m/s), density (kg/m³), damping (fraction of
critical). **Add layer** and **Remove** edit the stack.

- **Forward** — compute the transfer function of the layers as entered and
  compare its peak against the observed f₀.
- **Invert** — fit Vs and thickness to the observed f₀.
- **Quarter-wavelength check** — H = Vs / (4·f₀), the physics the empirical
  laws approximate.

**An f₀-only fit is non-unique.** Thickness and velocity trade off exactly: one
equation, two unknowns. Fix one from a borehole or from an independent Vs model
— ambient-noise tomography over the same deployment is the natural source — and
read the other. This is the single most useful thing on this page.

The model predicts *frequency*, not amplitude. Its amplitude is indicative
only: the transfer function is the SH response of the soil column, while H/V is
a property of the ambient wavefield, dominated by Rayleigh-wave ellipticity.

---

### 9. Batch & Export

**Run** — compute many sites without stepping through them.

Scope: **sites not computed yet**, **all active sites**, or **every site except
excluded**. Set the worker count to suit your machine. Results are written as
each site finishes, so an interrupted run keeps what it had. **Stop** ends it
cleanly.

**Export**

| Button | Output |
|---|---|
| **H/V curves (one file per site)** | The curves as text |
| **Summary table (CSV)** | One row per site: f₀, A₀, σ, SESAME verdict, depth |
| **Full output set** | Everything, including per-window arrays |
| **OpenHVSR-ProTO project** | Readable by the MATLAB toolkit |
| **HTML report** | Self-contained: survey summary, distributions, and a page per site with its curve, SESAME table and window statistics, every figure embedded |

Every export records the parameters used and names the coordinate system. The
HTML report has no external dependencies — it can be emailed and will still
render years from now.

---

## Parameter reference

Defaults follow `DEFAULT_VALUES.m` from OpenHVSR-ProTO except where noted.

### Windowing

| Parameter | Default | Notes |
|---|---|---|
| `window_width_s` | 60.0 | ProTO uses 30. 60 s resolves lower f₀. |
| `window_overlap_pc` | 50.0 | |
| `taper_pc` | 5.0 | Raised-cosine, applied to each window |
| `pad_to` | `off` | Or a sample count, rounded up to a power of two |

Window length sets the lowest usable frequency: SESAME's first reliability
condition is f₀ > 10 / L<sub>w</sub>, so 60 s windows are trustworthy only
above about 0.17 Hz.

### STA/LTA anti-trigger

| Parameter | Default |
|---|---|
| `antitrigger` | `True` |
| `sta_s` | 1.0 |
| `lta_s` | 30.0 |
| `sta_lta_ratio` | 4.0 |

### Spectra

| Parameter | Default | Notes |
|---|---|---|
| `freq_min` | 0.2 | ProTO: 0.5 |
| `freq_max` | 25.0 | ProTO: 50 |
| `smoothing_kind` | `konno_ohmachi` | or moving average |
| `smoothing_b` | 40.0 | Lower = smoother |
| `hvsr_strategy` | `squared_average` | see below |
| `statistics` | `lognormal` | SESAME reports log-normal σ of f₀ |
| `freq_grid` | `log` | `linear` restores ProTO's axis |
| `n_freq` | 512 | points on the log grid |

**`hvsr_strategy`** — the three ways to collapse two horizontals into one H.
They differ only by a constant factor, but that factor matters: **Total Energy
is exactly √2 larger than Average Squared**, which is enough to move a site
across SESAME's A₀ > 2 threshold. Every export records which rule was used.

### Pre-filter

| Parameter | Default |
|---|---|
| `filter_kind` | `off` |
| `filter_order` | 4 |
| `filter_fmin` | 0.5 |
| `filter_fmax` | 25.0 |
| `filter_target` | `hvsr` |

Zero-phase by default. Set `zero_phase=False` for bit-comparable ProTO
behaviour.

### Ingestion, azimuth, acceptance

| Parameter | Default | Notes |
|---|---|---|
| `target_fs` | 0.0 | 0 keeps native; else decimate to this |
| `detrend` | `demean` | `none`, `demean` or `linear` |
| `azimuth_step_deg` | 0.0 | 0 = off; 10 or 15 typical |
| `min_windows` | 5 | ProTO excludes a site below this |

Decimating costs nothing in accuracy below the new Nyquist and saves a great
deal of time. With `freq_max` at 25 Hz, decimating to 125 Hz is safe.

### Reproducing ProTO exactly

Set `freq_grid = "linear"` and `statistics = "linear"`. This restores ProTO's
frequency axis and arithmetic mean.

Two differences remain by design, both documented at the point of departure in
the source:

1. **The cosine taper is applied.** ProTO calls `cosine_taper(wdat, tapervalue)`
   without keeping the return value, so the taper never reaches the FFT.
2. **Smoothing is a matrix.** Konno-Ohmachi weights depend only on the frequency
   axis and the bandwidth, so the operator is built once and applied with one
   BLAS call rather than rebuilt per frequency per window.

---

## Command line

```bash
python run.py                      # open the most recent project
python run.py <project>            # open a specific project (folder or json)
python run.py --check              # environment report, then exit
python run.py --batch <project> --workers 6 --scan-windows
```

`--batch` computes every pending active site without the interface.
`--workers` sets parallelism (default 4). `--scan-windows` runs the quiet-window
reconnaissance for each recording rather than using the stored selection.

Results are written as each site finishes. Exit status is 0 on success, 1 if
the project has no raw data directory, 2 if any site failed.

---

## Files on disk

```
HVSRLab_Projects/<name>/
  project.json        sites, parameters, picks, wells, profiles, history
  results/<site>.npz  per-window arrays
  figures/            saved figures
  exports/            everything you write out
  logs/session.log    the Activity panel, mirrored
  cache/              scratch
```

`project.json` is plain JSON and readable. Results are self-contained, so a
project folder can be moved or shared without the raw data — though note it
carries absolute paths to the raw data, which will need updating if the data
moves or the drive letter changes.

---

## Troubleshooting

**Start here: the Activity panel.** It catches job output, page messages,
Python warnings, the standard library's logging, Qt's message stream, and —
most importantly — unhandled exceptions. Without that last one, an error inside
a Qt slot goes to a stderr nobody is reading and the interface simply appears
to do nothing. Filter by level, and use **Copy diagnostics** to put the
environment report and the log on the clipboard as one block.

A clean startup shows zero warnings, so any warning you see is real.

| Symptom | Cause |
|---|---|
| `obspy MISSING` at startup | Run `python run.py --check` and install what it names |
| Site produces no result | Fewer than `min_windows` windows survived the anti-trigger — try a different time window |
| Curve is spiky and unstable | Too few windows, or `smoothing_b` too high |
| Peak at the very edge of the band | Widen `freq_min` / `freq_max`; a peak at the edge is usually an artefact |
| f₀ fails SESAME condition 1 | Window too short for that frequency — increase `window_width_s` |
| Sites all at one location on the map | Coordinates not matched, or read as metres when they are degrees |
| Profile section is empty | No sites inside the corridor — widen it or pick sites by hand |
| Depths look implausible | Check which law is active and how far the boreholes sit from their stations |
| Interface freezes on a big scan | Scanning is one pass over file names; a very large survey takes a minute |

---

## Reading your results honestly

**The peak frequency is robust.** It is what the depth interpretation rests on,
and it is reproducible across reasonable parameter choices. If f₀ moves when
you change the window length or the smoothing, you do not yet have a result.

**The amplitude is not a site amplification factor.** It depends on the ambient
wavefield and on which horizontal-combination rule produced it. Quote A₀ only
alongside the rule used.

**Directionality invalidates the 1D reading.** Check the azimuth tab before
converting f₀ to depth. A strongly directional site is not a layered column.

**Depth conversion is the weakest link in the chain.** An empirical law
imported from another basin can be wrong by tens of percent. A regression on
local boreholes is better, but only as good as the separations between those
holes and their stations. The quarter-wavelength relation with an independent
Vs is better still, and is the reason the 1D Model page exists.

**Report what you did.** Every export records the parameters, the
horizontal-combination rule and the coordinate system, because an H/V number
without them is not reproducible.

---

## Further reading

[REFERENCES.md](REFERENCES.md) — the papers behind each method, including the
SESAME guidelines, the six bedrock power laws, and the software this is
built on.
