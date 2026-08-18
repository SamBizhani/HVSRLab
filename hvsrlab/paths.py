"""Where things live on disk.

Projects default to a sibling of the application directory so that HVSRLab and
ANTAgent projects sit next to each other, but nothing here is enforced: a
project carries its own absolute root.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Package directory (…/HVSRLab/hvsrlab).
PACKAGE_DIR = Path(__file__).resolve().parent

#: Application directory (…/HVSRLab).
APP_DIR = PACKAGE_DIR.parent

#: Default parent directory for projects.
PROJECTS_DIR = Path(
    os.environ.get("HVSRLAB_PROJECTS", APP_DIR.parent / "HVSRLab_Projects")
)

#: Per-project sub-directories, created on demand by :meth:`Project.ensure_tree`.
SUBDIRS = ("results", "figures", "exports", "logs", "cache")


def projects_dir() -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def unique_project_dir(name: str) -> Path:
    """A project directory named *name*, suffixed if that name is taken."""
    base = projects_dir() / _slug(name)
    if not base.exists():
        return base
    for n in range(2, 1000):
        candidate = base.with_name(f"{base.name}_{n}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot find a free project directory for {name!r}")


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in name.strip()]
    slug = "".join(keep).strip("_")
    return slug or "untitled"
