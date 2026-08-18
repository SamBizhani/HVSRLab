"""HVSRLab — horizontal-to-vertical spectral ratio processing for passive seismic surveys.

A Python re-implementation of Samuel Bignardi's OpenHVSR Processing Toolkit
(GPL-3, MATLAB) with a modern Qt interface, native MiniSEED ingestion, SESAME
2004 quality criteria, azimuthal and temporal stability analysis, and 1D
forward modelling of the resonance.

The processing chain mirrors OpenHVSR-ProTO so results are comparable:

    raw traces -> optional Butterworth filter -> windowing (+ STA/LTA anti-trigger)
    -> demean -> cosine taper -> FFT -> Konno-Ohmachi smoothing
    -> H/V per window -> statistics over windows -> peak picking

Two departures from the MATLAB original are deliberate and documented in
``core.hvsr``: the cosine taper is actually applied (ProTO discards the return
value of ``cosine_taper``), and smoothing is a single pre-computed matrix
product rather than a per-window loop.

The technique itself is Nakamura, Y. (1989), *A method for dynamic
characteristics estimation of subsurface using microtremor on the ground
surface*, QR of RTRI 30(1), 25–33; the criteria used to judge a peak are those
of SESAME (2004), deliverable D23.12. See ``REFERENCES.md``.

Copyright (C) 2026 the HVSRLab authors.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License (``LICENSE``) for more details.

Derived from Samuel Bignardi's OpenHVSR Processing Toolkit (C) 2017, GPL-3.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
