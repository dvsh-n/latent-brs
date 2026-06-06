# Reacher LPB

This folder contains the Reacher Latent Policy Barrier implementation. It
mirrors the cube and rope LPB path: build a shuffled prototype bank from
encoded trusted Reacher states, then add the nearest-prototype penalty to iLQR.

By default, the bank filters safe states with the same joint-space obstacle rule
used by `reacher/Haoran_obs_data/obs_data_collect_new.py`: obstacle/unsafe means
`qpos` lies inside the configured box, default `q1 in [0, 3.1415]` and
`q2 in [-2.88, -2.45]`; safe means outside that box.

Build a small smoke bank:

```bash
python3 -m reacher.safety.lpb.build_bank_reacher \
  --dataset-path reacher/data/expert_data_50hz/reacher_expert.h5 \
  --model-dir reacher/models/mlpdyn_ft_1 \
  --output-path reacher/safety/runs/lpb_reacher/lpb_bank_smoke.pt \
  --max-frames 512 \
  --num-prototypes 256 \
  --overwrite
```

Run Reacher iLQR with LPB:

```bash
python3 -m reacher.plan.benchmark_reacher_hard \
  --method ilqr \
  --dataset-path reacher/data/test_data_50hz/reacher_test.h5 \
  --model-dir reacher/models/mlpdyn_ft_1 \
  --stats-dataset-path reacher/data/expert_data_50hz/reacher_expert.h5 \
  --lpb-bank-path reacher/safety/runs/lpb_reacher/lpb_bank_smoke.pt \
  --lpb-weight 1.0
```

The HJ benchmark also supports `--mode lpb` and `--mode all` when
`--lpb-bank-path` is provided. `--mode all` runs the three reported comparisons:
nominal, HJ-filtered, and LPB-only.
