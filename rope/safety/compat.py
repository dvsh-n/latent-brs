"""Compatibility shims for old rope checkpoint module names."""

from __future__ import annotations

import sys
import types

import rope.shared.models as rope_shared_models


def register_legacy_checkpoint_aliases() -> None:
    """Resolve older object checkpoints pickled under pre-merge module names."""

    ogbench_cube_pkg = sys.modules.setdefault("ogbench_cube", types.ModuleType("ogbench_cube"))
    ogbench_cube_pkg.__path__ = []  # type: ignore[attr-defined]
    ogbench_cube_shared_pkg = sys.modules.setdefault("ogbench_cube.shared", types.ModuleType("ogbench_cube.shared"))
    ogbench_cube_shared_pkg.__path__ = []  # type: ignore[attr-defined]
    setattr(ogbench_cube_pkg, "shared", ogbench_cube_shared_pkg)
    setattr(ogbench_cube_shared_pkg, "models", rope_shared_models)
    sys.modules.setdefault("ogbench_cube.shared.models", rope_shared_models)
