# References

The work HVSRLab rests on, grouped by where it is used. Entries marked
**[in code]** are also cited at the point of use in the source.

---

## The method

**Nakamura, Y. (1989).** A method for dynamic characteristics estimation of
subsurface using microtremor on the ground surface. *Quarterly Report of the
Railway Technical Research Institute*, 30(1), 25–33.

> The origin of the horizontal-to-vertical spectral ratio technique.

**SESAME (2004).** *Guidelines for the implementation of the H/V spectral ratio
technique on ambient vibrations: measurements, processing and interpretation.*
SESAME European research project, WP12, deliverable D23.12, European Commission
Research General Directorate, project EVG1-CT-2000-00026. **[in code:
`core/sesame.py`]**

> The reliability and clarity criteria implemented in `core/sesame.py`, and the
> processing order (smooth the components, then divide) followed in
> `core/hvsr.py`.

---

## The program this one re-implements

**Bignardi, S. (2017).** *OpenHVSR Processing Toolkit.* GPL-3.
<https://github.com/sedysen/OpenHVSR-Processing-Toolkit> **[in code:
throughout]**

> The processing chain, the parameter vocabulary, the horizontal-combination
> rules, the profile smoothing and normalisation strategies, and the project
> export format follow this program so that results are comparable. HVSRLab is
> a derivative work and is released under the same licence.

Related publications by the same author on OpenHVSR should be cited alongside
the toolkit where the method rather than the software is meant.

---

## Spectral estimation

**Konno, K. & Ohmachi, T. (1998).** Ground-motion characteristics estimated
from spectral ratio between horizontal and vertical components of microtremor.
*Bulletin of the Seismological Society of America*, 88(1), 228–241. **[in code:
`core/spectra.py`]**

> The smoothing window applied to every amplitude spectrum in the program.

---

## Depth to bedrock — empirical power laws

All of the form `H = a · f0^b`, implemented in `core/bedrock.py`. The
coefficients are strongly site-dependent; prefer a regression on local
boreholes where three or more are available.

| Calibration | Region | Reference |
|---|---|---|
| **Ibs-von Seht, M. & Wohlenberg, J. (1999)** | Lower Rhine Embayment, Germany | *BSSA*, 89(1), 250–259 |
| **Delgado, J. et al. (2000)** | Bajo Segura basin, Spain | *J. Applied Geophysics*, 45, 19–32 |
| **Parolai, S. et al. (2002)** | Cologne area, Germany | *BSSA*, 92(6), 2521–2527 |
| **Hinzen, K.-G. et al. (2004)** | Lower Rhine Embayment, Germany | *Netherlands J. of Geosciences*, 83(4) |
| **D'Amico, V. et al. (2008)** | Florence, Italy | *BSSA*, 98(3) |
| **Birgören, G. et al. (2009)** | Istanbul, Turkey | *J. Seismology*, 13, 249–261 |

**[in code: `core/bedrock.py`, with journal references carried on each `Law`]**

---

## 1D forward modelling

**Haskell, N. A. (1953).** The dispersion of surface waves on multilayered
media. *Bulletin of the Seismological Society of America*, 43(1), 17–34.

**Thomson, W. T. (1950).** Transmission of elastic waves through a stratified
solid medium. *Journal of Applied Physics*, 21(2), 89–93.

**Kramer, S. L. (1996).** *Geotechnical Earthquake Engineering*, §7.2.
Prentice Hall. **[in code: `core/model1d.py`]**

> The SH transfer function of a layered column over a half-space, in the
> recursion form given by Kramer.

---

## File formats read

**SAF** — SESAME ASCII format, as specified in the SESAME deliverables.
**Geopsy** — <http://www.geopsy.org>, for H/V curve exchange.
**[in code: `io/curves.py`]**

---

## Software

HVSRLab is built on these; each asks to be cited in work that uses it.

**ObsPy** — Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y. &
Wassermann, J. (2010). ObsPy: A Python toolbox for seismology. *Seismological
Research Letters*, 81(3), 530–533.

**NumPy** — Harris, C. R. et al. (2020). Array programming with NumPy.
*Nature*, 585, 357–362.

**SciPy** — Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for
scientific computing in Python. *Nature Methods*, 17, 261–272.

**Matplotlib** — Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
*Computing in Science & Engineering*, 9(3), 90–95.

**PROJ / pyproj** — PROJ contributors. *PROJ coordinate transformation
software library.* Open Source Geospatial Foundation. Used for the UTM
conversion in `io/crs.py` when installed.

**Qt / PyQt5** — Riverbank Computing. PyQt5 is used under its GPL-3 licence.

---

## A note on accuracy

Bibliographic details here are transcribed and should be checked against the
originals before any of them goes into a thesis or a paper — the same caution
`core/bedrock.py` gives about the power-law coefficients themselves.
