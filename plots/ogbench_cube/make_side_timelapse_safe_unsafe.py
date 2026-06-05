#!/usr/bin/env python3
"""Compatibility entrypoint for the OGBench cube safe/unsafe timelapse renderer."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots.ogbench_cube.make_ogbench_cube_timelapse_safe_unsafe import main


if __name__ == "__main__":
    main()
