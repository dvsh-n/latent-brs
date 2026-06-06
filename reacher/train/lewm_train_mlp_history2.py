#!/usr/bin/env python3
"""Compatibility launcher for the Reacher non-Markov history-2 MLP ablation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reacher.train.mlpdyn_train_history2 import main


if __name__ == "__main__":
    main()
