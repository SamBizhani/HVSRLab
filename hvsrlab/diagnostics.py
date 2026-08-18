"""What is installed, what is loaded, and where things are.

One report, used by ``run.py --check``, by the Activity panel's "Copy
diagnostics" button, and by the header of any saved log — so a problem
reported from any of the three arrives with the same context attached.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import os
import platform
import sys

#: Everything the application will not start without.
REQUIRED = ("numpy", "scipy", "matplotlib", "obspy", "PyQt5")

#: Present or absent, either is fine.
OPTIONAL = {
    "pandas": "faster table exports",
    "numba": "reserved for large surveys",
}


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in list(REQUIRED) + list(OPTIONAL):
        try:
            module = importlib.import_module(name)
            out[name] = str(getattr(module, "__version__", "installed"))
        except ImportError:
            out[name] = "MISSING" if name in REQUIRED else "absent"
    try:
        from PyQt5.QtCore import QT_VERSION_STR
        out["Qt"] = QT_VERSION_STR
    except ImportError:
        pass
    return out


def compatibility_warnings() -> list[str]:
    """Version pairings known to break, checked rather than assumed."""
    notes: list[str] = []
    try:
        import matplotlib
        from PyQt5.QtCore import QT_VERSION_STR

        qt = tuple(int(p) for p in QT_VERSION_STR.split(".")[:2])
        mpl = tuple(int(p) for p in matplotlib.__version__.split(".")[:2])
        if qt < (5, 10) and mpl >= (3, 6):
            notes.append(
                f"matplotlib {matplotlib.__version__} needs Qt >= 5.10 but Qt "
                f"is {QT_VERSION_STR}; pin matplotlib to 3.5.x or upgrade PyQt5")
    except (ImportError, ValueError):
        pass
    return notes


def environment_report(project=None, *, include_project: bool = True) -> str:
    """A block of text safe to paste anywhere."""
    from . import __version__, paths

    lines = [
        f"HVSRLab {__version__}",
        f"time        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"platform    {platform.platform()}",
        f"python      {sys.version.split()[0]}  ({sys.executable})",
    ]
    for name, version in package_versions().items():
        lines.append(f"{name:<11} {version}")
    try:
        from .io import crs as crs_io
        lines.append(f"projection  {crs_io.available()}")
    except Exception:                                  # noqa: BLE001
        pass
    lines.append(f"projects    {paths.PROJECTS_DIR}")
    lines.append(f"cpus        {os.cpu_count()}")

    for note in compatibility_warnings():
        lines.append(f"! {note}")

    if include_project and project is not None:
        import numpy as np

        computed = sum(1 for s in project.sites if np.isfinite(s.f0))
        lines += [
            "",
            f"project     {project.name}",
            f"root        {project.root}",
            f"raw data    {project.raw_dir or '—'}",
            f"stations    {project.station_file or '—'}",
            f"sites       {len(project.sites)} ({computed} computed)",
            f"plan grid   {(project.crs or {}).get('epsg', 'not set')}",
            f"band        {project.params.freq_min:g}–{project.params.freq_max:g} Hz",
            f"windows     {project.params.window_width_s:g} s, "
            f"{project.params.window_overlap_pc:g} % overlap",
            f"decimate to {project.params.target_fs or 'native'} Hz",
            f"horizontals {project.params.hvsr_strategy}",
        ]
    return "\n".join(lines)


def print_check(project=None) -> int:
    """``run.py --check``: report, and return a shell exit code."""
    print(environment_report(project))
    missing = [n for n in REQUIRED if package_versions()[n] == "MISSING"]
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print(f"  pip install --user {' '.join(missing)}")
        return 1
    print("\nReady.")
    return 0
