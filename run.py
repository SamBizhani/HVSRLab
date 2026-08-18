#!/usr/bin/env python
"""Start HVSRLab.

    python run.py                     open the last project, or a new one
    python run.py <project.json>      open a specific project
    python run.py --check             report on the environment and exit
    python run.py --batch <project>   compute every pending site, no interface
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _utf8_console() -> None:
    """Windows consoles default to cp1252, which cannot print σ or ·."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def check() -> int:
    """Print what is installed and whether it is enough."""
    from hvsrlab import diagnostics

    return diagnostics.print_check()


def batch(project_path: str, workers: int, choose_window: bool) -> int:
    """Run the survey without the interface."""
    from hvsrlab import batch as batch_mod
    from hvsrlab.io import mseed
    from hvsrlab.jobs import Job
    from hvsrlab.project import Project

    project = Project.load(project_path)
    print(f"{project.name}: {len(project.sites)} sites")
    if not project.raw_dir:
        print("This project has no raw data directory.")
        return 1

    print(f"Cataloguing {project.raw_dir} …")
    scan = mseed.scan(project.raw_dir)
    recordings = {r.sid: r for r in scan.sites}
    print(f"  {len(recordings)} recordings, {scan.n_files} files")

    import math
    pending = [s for s in project.sites
               if s.is_active and math.isnan(s.f0)]
    if not pending:
        print("Nothing pending.")
        return 0

    job = Job(name="batch")
    job.on_line(print)
    report = batch_mod.run(project, pending, recordings, workers=workers,
                           choose_window=choose_window, job=job)
    project.save()
    print(report.summary())
    return 0 if not report.failed else 2


def main() -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(
        description="HVSRLab — H/V spectral ratio processing")
    parser.add_argument("project", nargs="?", default=None,
                        help="project.json (or its folder) to open")
    parser.add_argument("--check", action="store_true",
                        help="report on the environment and exit")
    parser.add_argument("--batch", metavar="PROJECT",
                        help="compute every pending site without the interface")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel sites in a batch run (default 4)")
    parser.add_argument("--scan-windows", action="store_true",
                        help="in a batch run, scan each recording for its "
                             "quietest window")
    args = parser.parse_args()

    if args.check:
        return check()
    if args.batch:
        return batch(args.batch, args.workers, args.scan_windows)

    from PyQt5.QtWidgets import QApplication

    from hvsrlab import paths
    from hvsrlab.gui import MainWindow
    from hvsrlab.gui import theme
    from hvsrlab.project import Project

    app = QApplication(sys.argv)
    app.setApplicationName("HVSRLab")
    theme.apply(app)
    theme.install_mpl()

    project = None
    if args.project:
        project = Project.load(args.project)
    else:
        latest = _most_recent_project(paths.projects_dir())
        if latest is not None:
            try:
                project = Project.load(latest)
            except Exception:                          # noqa: BLE001
                project = None
    if project is None:
        root = paths.unique_project_dir("untitled")
        project = Project(root, "untitled")
        project.ensure_tree()

    window = MainWindow(project)
    window.show()
    return app.exec_()


def _most_recent_project(root: Path):
    candidates = sorted(root.glob("*/project.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


if __name__ == "__main__":
    raise SystemExit(main())
