# OGBench Cube Safety

This package contains the cube latent-safety code. The LPB-specific code is now isolated in:

```text
ogbench/safety/lpb/
  __init__.py
  barrier.py
  build_bank_cube.py
  README.md
```

The remaining files are shared safety infrastructure used by PyHJ/HJ filtering and, in a few places, reused by LPB:

- `obstacle_classifier.py`: trains the latent obstacle classifier.
- `classifier_oracle.py`: loads a classifier checkpoint as a signed margin oracle.
- `constraints.py`: analytic cube obstacle rule and signed margin.
- `latent_cache.py`: encodes cube frames, builds Markov states, normalizes actions, and computes margins.
- `latent_env.py`: Gymnasium-style latent safety environment over learned dynamics.
- `train_pyhj_cube.py`: trains the PyHJ avoid-DDPG value/policy.
- `eval_pyhj_cube.py`: evaluates the trained PyHJ policy/value.
- `hj_filter.py`: online HJ safety filter used during closed-loop cube rollout.
- `benchmark_hj_filter_cube.py`: benchmark harness for nominal, HJ, LPB, and HJ+LPB comparisons.
- `select_geometry_unsafe_cube.py`: selects held-out test episodes that cross the analytic obstacle.
- `run_smoke.py`: quick latent-env smoke rollout.
- `compat.py`: checkpoint alias shim for legacy `ogbench_cube.*` pickle paths.

For the LPB implementation and previous-run analysis, see `ogbench/safety/lpb/README.md`.
