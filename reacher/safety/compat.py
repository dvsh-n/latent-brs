"""Compatibility shims for old Reacher checkpoint module names."""

from __future__ import annotations

import sys
import types

import reacher.shared.models as reacher_shared_models


def register_legacy_checkpoint_aliases() -> None:
    """Resolve older object checkpoints pickled under pre-merge module names."""

    for package_name in ("reacher", "ogbench_cube"):
        pkg = sys.modules.setdefault(package_name, types.ModuleType(package_name))
        if not hasattr(pkg, "__path__"):
            pkg.__path__ = []  # type: ignore[attr-defined]
        shared_pkg = sys.modules.setdefault(f"{package_name}.shared", types.ModuleType(f"{package_name}.shared"))
        shared_pkg.__path__ = []  # type: ignore[attr-defined]
        setattr(pkg, "shared", shared_pkg)
        setattr(shared_pkg, "models", reacher_shared_models)
        sys.modules.setdefault(f"{package_name}.shared.models", reacher_shared_models)
