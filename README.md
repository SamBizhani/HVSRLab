# HVSRLab

Horizontal-to-vertical spectral ratio processing for passive seismic surveys.

A Python re-implementation of Samuel Bignardi's **OpenHVSR Processing Toolkit**
(GPL-3, MATLAB) with a Qt interface, native MiniSEED ingestion, SESAME 2004
quality criteria, azimuthal and temporal stability analysis, and 1D forward
modelling of the resonance.

```bash
cd path/to/HVSRLab && python run.py
```

```bash
python run.py --check
```

---

## Why this exists

An ambient-noise tomography deployment leaves weeks of three-component
recording at every site — a hundred sites, six weeks each at 250 Hz, runs to
some twelve thousand files and a few hundred gigabytes. H/V needs a few hours
of that, and it matters *which*
hours. HVSRLab is built around those two facts: it never reads more than it
needs, and it helps you choose.

## What it does

| Page | What happens there |
|---|---|
| **Overview** | Survey status, layout, distribution of results |
| **Sites** | Catalogue the raw MiniSEED, match station coordinates, set each site's status |
| **Data & Windows** | Find the quietest hours; window them; drop windows with transients |
| **H/V Analysis** | Compute the curve, pick f₀, judge it against SESAME; azimuth, time-stability, and every site side by side. Runs one site or the whole survey |
| **Maps & Profiles** | Interpolated maps of f₀, A₀, depth, σf₀ or H/V at a chosen frequency; hand-drawn profiles whose sites you click in and out on the map, and pseudo-sections |
| **3D Views** | The bedrock surface, or every site's H/V curve standing in place |
| **Bedrock** | H = a·f₀^b — published laws, or a regression fitted to boreholes you load or type in |
| **1D Model** | Layered SH transfer function; invert f₀ for a Vs/thickness model |
| **Batch & Export** | Run the survey in parallel; curves, CSV, ProTO project, HTML report |

## Everything OpenHVSR-ProTO does

Windowing with STA/LTA anti-trigger · cosine taper · zero-padding to a power of
two · Butterworth pre-filters · Konno-Ohmachi and moving-average smoothing ·
the three horizontal-combination rules (Average Squared, Simple Average, Total
Energy) · per-window and average H/V with E/V and N/V · directional H/V ·
manual and automatic peak picking with secondary peaks · maps with contour
styles and colour maps · profiles with ProTO's four smoothing and four
normalisation strategies · pseudo-depth sections · 3D surfaces and H/V tiles ·
Ibs-von Seht & Wohlenberg bedrock depth with a custom regression from wells ·
SAF and ASCII readers · export as an OpenHVSR-ProTO project.

## And what it adds

**SESAME 2004 criteria.** The three reliability conditions and six clarity
criteria, evaluated automatically for every site and shown as a live checklist.
This is the standard the community judges an f₀ pick against, and it is the one
thing ProTO leaves entirely to the operator's eye.

**Smart window selection.** A two-stage reconnaissance — short probes across
the whole deployment, then a finer pass around the quietest day — ranks
candidate blocks by how quiet *and* how steady they are. Six weeks of recording
is characterised in about 160 reads. On real deployment data it picks local
night without being told where the site is or what time zone it is in.

**Azimuthal and temporal analysis.** H/V against rotation azimuth as a heat map
and a polar rose, computed by rotating the complex spectra rather than
re-transforming at each angle. H/V against time through the recording, with the
per-window peak below it, which is where diurnal contamination of a peak shows
itself.

**1D forward modelling and inversion.** Haskell–Thomson SH transfer function
for a layered column; fit Vs and thickness to the observed f₀, or read the
quarter-wavelength depth from an independent Vs. Depth from physics rather than
from another basin's regression.

**Profiles you edit on the map.** Draw a line with two clicks and it starts
with the sites it actually passes — those within half the typical station
spacing, not the whole survey smeared along an arbitrary line. Then arm
**Pick on map** (or Ctrl+click) and click sites in and out of it: a green ring
means on the section, a red cross means left out, and the section is built from
the ringed sites and nothing else. The corridor belongs to the profile, so two
profiles can select differently, and hand-picked choices survive changes to it.

**Boreholes tied to the nearest site, with the distance kept.** Drilling
rarely lands on a seismic station, so a borehole links itself to whichever site
is genuinely closest — on load, on manual entry, and again whenever its
coordinates change — and the separation is carried alongside the link and shown
in the table. That distance *is* the assumption the calibration rests on: that
the basin is the same at the hole and at the station. Anything past 500 m is
flagged, the fit logs the separations it relied on, and a link can be pinned by
hand when local knowledge beats proximity.

**UTM plan coordinates.** Latitude and longitude are converted automatically
to the UTM zone of the survey's own centroid, and that zone is
recorded on the project so every later session lands on the same grid. Station
or borehole files that are *already* in metres are recognised as such and read
directly. Everything metric downstream — site spacing, profile lengths,
interpolation grids, the ProTO export, the CSV — is on that grid, and each
export names it. A survey straddling a zone boundary is projected into one zone
rather than two, with a warning saying so, because a profile whose ends sit on
different grids has no meaningful length. `pyproj` does the conversion when
present; a built-in transverse-Mercator series agrees with it to 0.03 mm and
takes over when it is not.

**An Activity panel that catches everything.** Job output, page messages,
Python warnings, the standard library's logging, Qt's own message stream, and —
most importantly — unhandled exceptions. Without that last one an error inside
a Qt slot goes to a stderr nobody is reading and the interface simply appears
to do nothing, which is the least debuggable failure there is. Filterable by
level, mirrored to `<project>/logs/session.log`, with a **Copy diagnostics**
button that puts the environment report and the log on the clipboard as one
block. Consecutive duplicate lines collapse, and Qt's unactionable complaints
are demoted to debug, so a clean startup shows zero warnings — which means any
warning you do see is real.

**Reports.** A self-contained HTML report — survey summary, distributions, and
a page per site with its curve, its SESAME table and its window statistics —
with every figure embedded, so it can be emailed and will still render years
from now.

## Two deliberate departures from the MATLAB original

1. **The cosine taper is applied.** ProTO calls `cosine_taper(wdat, tapervalue)`
   without keeping the return value, so the taper never reaches the FFT and the
   spectra carry the leakage it was meant to suppress.
2. **Smoothing is a matrix.** Konno-Ohmachi weights depend only on the frequency
   axis and the bandwidth, so the operator is built once and applied to every
   window with one BLAS call, rather than rebuilt inside a loop over
   frequencies for every window on every parameter change.

Both are documented at the point of departure in the source. For bit-comparable
behaviour set `freq_grid = "linear"` and `statistics = "linear"`, which restores
ProTO's frequency axis and arithmetic mean.

## What H/V does and does not tell you

The **peak frequency** is robust, and it is what the depth interpretation rests
on. The **amplitude** is not a site amplification factor: it depends on the
ambient wavefield and on which of the three horizontal-combination rules
produced it — Total Energy is exactly √2 larger than Average Squared, which is
enough to move a site across SESAME's A₀ > 2 threshold. Every export records
which rule was used.

The 1D model fits the frequency. Its amplitude is indicative only: the transfer
function is the SH response of the soil column, while H/V is a property of the
wavefield, dominated by Rayleigh-wave ellipticity. And an f₀-only fit is
non-unique — thickness and velocity trade off exactly, so fix one from a
borehole or from the ambient-noise Vs model and read the other.

## Layout

```
HVSRLab/
  run.py                  launcher; --check, --batch
  hvsrlab/
    project.py            sites, parameters, JSON persistence
    batch.py              parallel survey runs
    export.py             curves, CSV, ProTO project, HTML report
    jobs.py               background work, Qt-free
    io/
      crs.py              UTM projection, zone choice, geographic round trip
      mseed.py            catalogue, time-slice loading, amplitude probes
      curves.py           SAF and ASCII recordings, H/V curve files
      stations.py         coordinate lists, local projection
      wells.py            borehole control
    core/
      windows.py          windowing, taper, STA/LTA
      spectra.py          FFT, Konno-Ohmachi operator
      hvsr.py             the computation and its result object
      picking.py          peak detection and picking
      sesame.py           SESAME 2004 criteria
      timeselect.py       reconnaissance and window choice
      bedrock.py          f₀ → depth laws and regression
      model1d.py          SH transfer function, inversion
      grids.py            maps, profiles, sections, volumes
      filters.py          Butterworth
    gui/
      theme.py            palette, stylesheet, matplotlib rc
      plots.py            every embedded figure
      state.py            shared workspace
      main_window.py      navigation, menus, activity log
      pages/              one module per page
```

Projects live in `..\HVSRLab_Projects\<name>\` by default: `project.json` holds
sites, parameters and picks; `results/<site>.npz` holds the per-window arrays;
`exports/` gets everything you write out.

## Requirements

Python 3.9+, numpy, scipy, matplotlib, obspy, PyQt5. `pyproj` is optional — a
built-in transverse-Mercator series stands in when it is absent and agrees with
it to 0.03 mm. If your PyQt5 ships Qt 5.9, matplotlib must stay below 3.6;
newer Qt has no such limit. Run

```bash
python run.py --check
```

to see what is installed and whether it is enough.

## Command line

```bash
python run.py --batch path/to/HVSRLab_Projects/<name> --workers 6 --scan-windows
```

Computes every pending site without the interface, scanning each recording for
its quietest window. Results are written as each site finishes, so an
interrupted run keeps what it had.

## Credit

OpenHVSR Processing Toolkit © 2017 Samuel Bignardi, GPL-3
(<https://github.com/sedysen/OpenHVSR-Processing-Toolkit>). The processing
chain, the parameter vocabulary and the profile smoothing and normalisation
strategies follow that program so results are comparable.

SESAME (2004), *Guidelines for the implementation of the H/V spectral ratio
technique on ambient vibrations*, European Commission research project
EVG1-CT-2000-00026, deliverable D23.12.

The technique itself is Nakamura (1989); the smoothing window is Konno &
Ohmachi (1998); the layered transfer function follows Haskell (1953) and
Thomson (1950) in the form given by Kramer (1996). Full details, including the
six published bedrock power laws and the software this is built on, are in
[REFERENCES.md](REFERENCES.md).

## Licence

GPL-3. See [LICENSE](LICENSE).

HVSRLab is a derivative work of OpenHVSR Processing Toolkit (© 2017 Samuel
Bignardi, GPL-3) and is released under the same terms. PyQt5 is used under its
GPL-3 licence.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

No survey data is included in this repository.
