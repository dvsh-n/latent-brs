# Rope LPB

This is the rope Latent Policy Barrier implementation. It mirrors the cube LPB
path while using rope's Markov latent dynamics, rope HDF5 keys, and the same
obstacle-zone distribution as `rope/data/obs_data_collect.py`.

Build a small smoke bank:

```bash
./latent_brs_venv/bin/python -m rope.safety.lpb.build_bank_rope \
  --dataset-path rope/data/train_data_noshadow.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --output-path rope/safety/runs/lpb_rope/lpb_bank_smoke.pt \
  --device cuda \
  --max-frames 4096 \
  --num-prototypes 1024 \
  --margin-source analytic \
  --threshold-quantile 0.95 \
  --overwrite
```

Run rope iLQR with LPB:

```bash
./latent_brs_venv/bin/python -m rope.plan.benchmark_rope_hard \
  --method ilqr \
  --dataset-path rope/data/obstacle_crossing/rope_obstacle_crossing.h5 \
  --stats-dataset-path rope/data/train_data_noshadow.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --lpb-bank-path rope/safety/runs/lpb_rope/lpb_bank_smoke.pt \
  --lpb-weight 1.0 \
  --num-eval 1 \
  --no-videos \
  --device cuda
```
