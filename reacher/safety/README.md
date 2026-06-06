# Reacher Latent-Safety Pipeline

This is the Reacher copy of the rope latent-safety/PyHJ flow. It defaults to
the artifacts in `reacher/Haoran_obs_data`.

```text
Reacher HDF5
  -> reacher/safety/latent_cache.py
  -> reacher/safety/train_pyhj_reacher.py
  -> reacher/safety/eval_pyhj_reacher.py
  -> reacher/safety/benchmark_hj_filter_reacher.py
```

## 1. Optional Classifier Refresh

The existing classifier is used by default:

```bash
reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt
```

To train a fresh copy:

```bash
./latent_brs_venv/bin/python -m reacher.safety.obstacle_classifier \
  --model-dir reacher/Haoran_obs_data/mlpdyn_ft_6 \
  --data-path reacher/Haoran_obs_data/obstacle_data/obstacle_classifier_data.pt \
  --out-dir "reacher/safety/obs_net_$(date +%Y%m%d_%H%M%S)" \
  --device cuda
```

## 2. Build A Latent Cache

Before building the cache, compare classifier hits against the same fingertip
circle geometry used by `reacher/Haoran_obs_data/obs_data_collect.py`:

```bash
./latent_brs_venv/bin/python -m reacher.safety.check_obstacle_geometry \
  --dataset reacher/data/expert_data_50hz/reacher_expert.h5 \
  --obstacle-model "$CLS" \
  --device cuda \
  --max-frames all \
  --output reacher/safety/runs/geometry_check_reacher_expert.json
```

Smoke cache:

```bash
./latent_brs_venv/bin/python -m reacher.safety.latent_cache \
  --dataset-path reacher/data/test_data_50hz/reacher_test.h5 \
  --model-dir reacher/Haoran_obs_data/mlpdyn_ft_6 \
  --output-path reacher/safety/cache/reacher_latent_safety_smoke.pt \
  --max-frames 64 \
  --margin-source classifier \
  --classifier-checkpoint reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt \
  --overwrite
```

Full training cache:

```bash
./latent_brs_venv/bin/python -m reacher.safety.latent_cache \
  --dataset-path reacher/data/expert_data_50hz/reacher_expert.h5 \
  --model-dir reacher/Haoran_obs_data/mlpdyn_ft_6 \
  --output-path reacher/safety/cache/reacher_latent_safety_classifier_train_tanh.pt \
  --margin-source classifier \
  --classifier-checkpoint reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt \
  --margin-transform tanh \
  --device cuda \
  --max-frames all \
  --overwrite
```

## 3. Dry-Run PyHJ

```bash
./latent_brs_venv/bin/python -m reacher.safety.train_pyhj_reacher \
  --cache-path reacher/safety/cache/reacher_latent_safety_smoke.pt \
  --run-dir reacher/safety/runs/pyhj_reacher_smoke \
  --device cpu \
  --dry-run
```

## 4. Train PyHJ

```bash
./latent_brs_venv/bin/python -m reacher.safety.train_pyhj_reacher \
  --cache-path reacher/safety/cache/reacher_latent_safety_classifier_train_tanh.pt \
  --run-dir reacher/safety/runs/pyhj_train_tanh \
  --device cuda \
  --classifier-checkpoint reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt
```

## 5. Evaluate PyHJ

```bash
./latent_brs_venv/bin/python -m reacher.safety.eval_pyhj_reacher \
  --cache-path reacher/safety/cache/reacher_latent_safety_classifier_train_tanh.pt \
  --policy-path reacher/safety/runs/pyhj_train_tanh/policy_latest.pth \
  --classifier-checkpoint reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt \
  --device cuda \
  --episodes 1000 \
  --output reacher/safety/runs/pyhj_train_tanh/eval_1000ep.json
```

## 6. Closed-Loop HJ Safety Filter

```bash
./latent_brs_venv/bin/python -m reacher.safety.benchmark_hj_filter_reacher \
  --dataset-path reacher/data/test_data_50hz/reacher_test.h5 \
  --model-dir reacher/Haoran_obs_data/mlpdyn_ft_6 \
  --base-method ilqr \
  --hj-cache-path reacher/safety/cache/reacher_latent_safety_classifier_train_tanh.pt \
  --hj-policy-path reacher/safety/runs/pyhj_train_tanh/policy_latest.pth \
  --classifier-checkpoint reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt \
  --epsilon 0.0 \
  --num-eval 50 \
  --eval-budget 120 \
  --device cuda \
  --out-dir reacher/safety/runs/closed_loop_hj_filter
```
