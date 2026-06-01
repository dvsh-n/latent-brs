#!/usr/bin/env python3
"""Launch robust PushT training runs for 0th, 1st, and 2nd order Markov states."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET_PATH = Path("pusht_rboust/data/pusht_robust_train.h5")
DEFAULT_RUN_ROOT = Path("pusht_rboust/models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--output-model-name", default="mlpdyn")
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Additional args passed after -- to pusht_rboust.train.mlpdyn_train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_args = args.trainer_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    for order in args.orders:
        if order < 0:
            raise ValueError("Markov orders must be non-negative.")
        run_dir = args.run_root / f"mlpdyn_embd_{args.embed_dim}_md_{order}"
        cmd = [
            sys.executable,
            "-m",
            "pusht_rboust.train.mlpdyn_train",
            "--dataset-path",
            str(args.dataset_path),
            "--run-dir",
            str(run_dir),
            "--output-model-name",
            args.output_model_name,
            "--embed-dim",
            str(args.embed_dim),
            "--markov-deriv",
            str(order),
            *extra_args,
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
