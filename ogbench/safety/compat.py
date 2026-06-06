"""Compatibility shims for old OGBench cube checkpoint module names."""

from __future__ import annotations

import sys
import types

import reacher.shared.models as shared_models
from ogbench.train import lewm_train_mlp_markov as cube_train


def register_legacy_checkpoint_aliases() -> None:
    """Resolve object checkpoints pickled under the old ``ogbench_cube`` package."""

    ogbench_cube_pkg = sys.modules.setdefault("ogbench_cube", types.ModuleType("ogbench_cube"))
    ogbench_cube_pkg.__path__ = []  # type: ignore[attr-defined]
    train_pkg = sys.modules.setdefault("ogbench_cube.train", types.ModuleType("ogbench_cube.train"))
    train_pkg.__path__ = []  # type: ignore[attr-defined]
    shared_pkg = sys.modules.setdefault("ogbench_cube.shared", types.ModuleType("ogbench_cube.shared"))
    shared_pkg.__path__ = []  # type: ignore[attr-defined]

    setattr(ogbench_cube_pkg, "train", train_pkg)
    setattr(ogbench_cube_pkg, "shared", shared_pkg)
    setattr(train_pkg, "mlpdyn_train", cube_train)
    setattr(shared_pkg, "models", shared_models)
    sys.modules.setdefault("ogbench_cube.train.mlpdyn_train", cube_train)
    sys.modules.setdefault("ogbench_cube.shared.models", shared_models)
    if hasattr(cube_train, "LeWMOGBenchDataset") and not hasattr(cube_train, "LeWMOGBenchCubeDataset"):
        cube_train.LeWMOGBenchCubeDataset = cube_train.LeWMOGBenchDataset
