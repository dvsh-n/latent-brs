# Rope Latent-Safety Pipeline

This is the rope version of the latent-safety/PyHJ scaffold. The intended flow is:

```text
rope HDF5
  -> rope/safety/latent_cache.py
  -> rope/safety/train_pyhj_rope.py
  -> rope/safety/eval_pyhj_rope.py
  -> rope/safety/benchmark_hj_filter_rope.py
```

The LPB path lives in `rope/safety/lpb`:

```text
trusted rope frames -> encode with the rope world model -> Markov latent states
  -> safe prototype bank -> iLQR latent nearest-neighbor barrier
```

Use `rope/safety/lpb/build_bank_rope.py` to build a bank, then pass
`--lpb-bank-path` to `rope.plan.benchmark_rope_hard`.

The obstacle classifier now lives here too:

- `collect_obstacle_data.py`: builds the rendered obstacle dataset.
- `obstacle_classifier.py`: trains/conformalizes the classifier.
- `classifier_oracle.py`: adapts the trained classifier into the positive-safe margin expected by PyHJ.

## Safety Rule

This is an obstacle-zone safety task, not a global "all low rope is unsafe"
task. The obstacle-data labels are:

```text
unsafe iff reach is inside [obstacle_reach_lower, obstacle_reach_upper]
          and estimated_low_rope_height <= low_rope_cutoff
```

With the current data artifact, the obstacle reach band is `[0.05, 0.15]` and
the low-rope cutoff is `0.925`. Low-rope configurations outside that reach band
are intentionally labeled safe/non-obstacle.

For `--margin-source classifier` or `--oracle classifier`, the trained
classifier score is the safety value that latent-safety sees:

```text
safety_margin = classifier_score - threshold
safe if safety_margin > 0
unsafe/failure if safety_margin <= 0
```

For the paper-faithful HJ run, build the cache with:

```text
safety_margin = tanh(classifier_score - threshold)
```

The raw margin is still saved as `raw_safety_margin`; PyHJ training/eval/deploy
should all use the transformed `safety_margin`.

The analytic rule is only a temporary/debug fallback and a way to generate
labels. It should not override classifier margins once the classifier oracle is
selected.

## Classifier Compatibility

The existing classifier artifact at `rope/safety/obs_net/da270d7d1050f110/model.pt`
expects `input_dim=12` and was trained with `rope/models/mlpdyn_noshadow_ft`.
The matching rope world model has `embed_dim=12` and `markov_state_dim=24`, so
this pair can be used without `--allow-classifier-latent-slice`.

For the real experiment, keep this invariant:

```text
classifier input_dim == world-model embed_dim
```

The older/default non-shadowless `rope/models/mlpdyn` and `rope/models/mlpdyn_ft`
models use `embed_dim=32`, so they do not match this classifier.

## 1. Build Or Refresh Obstacle Classifier

The moved existing artifact can be used for interface checks:

```bash
rope/safety/obs_net/da270d7d1050f110/model.pt
```

To train a fresh classifier against the current default rope WM without
overwriting the existing artifact, use a new timestamped output root. This uses
the exact latent-safety DINO failure-head loss:

```text
mean(relu(gamma - safe_scores))
+ mean(relu(gamma + unsafe_scores))
+ mean(relu(weak_unsafe_scores))
```

with `gamma = 0.75`, `0 = safe`, `1 = unsafe`, and optional `2 = weak unsafe`.

```bash
./latent_brs_venv/bin/python -m rope.safety.obstacle_classifier \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --data-path rope/safety/obstacle_data/obstacle_classifier_data-001.pt \
  --out-dir "rope/safety/obs_net_latent_safety_dino_$(date +%Y%m%d_%H%M%S)" \
  --loss latent-safety-dino \
  --margin 0.75 \
  --device cuda
```

## 2. Build A Latent Cache

Analytic temporary margins:

```bash
./latent_brs_venv/bin/python -m rope.safety.latent_cache \
  --dataset-path rope/data/rope_random_cubic_spline_test.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --output-path rope/safety/cache/rope_latent_safety_smoke.pt \
  --max-frames 64 \
  --margin-source analytic \
  --overwrite
```

Classifier margins:

```bash
./latent_brs_venv/bin/python -m rope.safety.latent_cache \
  --dataset-path rope/data/rope_random_cubic_spline_test.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --output-path rope/safety/cache/rope_latent_safety_classifier_smoke.pt \
  --max-frames 64 \
  --margin-source classifier \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --overwrite
```

Full training-dataset classifier pass:

```bash
./latent_brs_venv/bin/python -m rope.safety.latent_cache \
  --dataset-path rope/data/train_data_noshadow.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --output-path rope/safety/cache/rope_latent_safety_classifier_train_noshadow_tanh.pt \
  --margin-source classifier \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --classifier-threshold conformal \
  --margin-transform tanh \
  --device cuda \
  --max-frames all \
  --overwrite
```

Before building a cache, you can run the same direct frame classifier pass used
by the original obstacle-counting script:

```bash
./latent_brs_venv/bin/python -m rope.safety.count_obstacle_hits \
  --dataset rope/data/train_data_noshadow.h5 \
  --obstacle-model rope/safety/obs_net/da270d7d1050f110/model.pt \
  --device cuda \
  --max-frames all
```

## 3. Dry-Run PyHJ Setup

```bash
./latent_brs_venv/bin/python -m rope.safety.train_pyhj_rope \
  --cache-path rope/safety/cache/rope_latent_safety_smoke.pt \
  --run-dir rope/safety/runs/pyhj_smoke \
  --device cpu \
  --oracle knn \
  --dry-run
```

Classifier oracle wiring check:

```bash
./latent_brs_venv/bin/python -m rope.safety.train_pyhj_rope \
  --cache-path rope/safety/cache/rope_latent_safety_smoke.pt \
  --run-dir rope/safety/runs/pyhj_classifier_smoke \
  --device cpu \
  --oracle classifier \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --dry-run
```

## 4. Train PyHJ Avoid-DDPG

Paper-style training command. Do not run unless you are ready to launch the
actual HJ value/policy training.

```bash
./latent_brs_venv/bin/python -m rope.safety.train_pyhj_rope \
  --cache-path rope/safety/cache/rope_latent_safety_classifier_train_noshadow_tanh.pt \
  --run-dir rope/safety/runs/pyhj_train_noshadow_tanh \
  --device cuda \
  --oracle classifier \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --actor-hidden 512 512 512 512 \
  --critic-hidden 512 512 512 512 \
  --epoch 50 \
  --step-per-epoch 40000 \
  --step-per-collect 8 \
  --batch-size 512 \
  --buffer-size 40000 \
  --gamma 0.9999
```

## 5. Evaluate PyHJ Value/Policy

After training, compare the learned PyHJ policy against random rollouts and
check whether the critic/value separates cached safe and unsafe states:

```bash
./latent_brs_venv/bin/python -m rope.safety.eval_pyhj_rope \
  --cache-path rope/safety/cache/rope_latent_safety_classifier_train_noshadow_tanh.pt \
  --policy-path rope/safety/runs/pyhj_train_noshadow_tanh/policy_latest.pth \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --device cuda \
  --episodes 1000 \
  --value-samples 20000 \
  --output rope/safety/runs/pyhj_train_noshadow_tanh/eval_1000ep.json
```

Useful signals:

```text
policy failure_rate < random failure_rate
policy mean_min_margin > random mean_min_margin
critic_value_safe_auc > 0.5
critic_value_margin_corr > 0
critic_value_safe_mean > critic_value_unsafe_mean
```

`run_smoke.py` is still random rollout only. It does not plan and does not train.

## 6. Closed-Loop Rope HJ Safety Filter

This is the actual LatentSafe deployment layer: run the normal rope controller,
predict the nominal action's next latent state, evaluate `B = min(l, V)`, and
execute the PyHJ safety actor when `B <= epsilon`.

```bash
./latent_brs_venv/bin/python -m rope.safety.benchmark_hj_filter_rope \
  --dataset-path rope/data/train_data_noshadow.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --base-method ilqr \
  --hj-cache-path rope/safety/cache/rope_latent_safety_classifier_train_noshadow_tanh.pt \
  --hj-policy-path rope/safety/runs/pyhj_train_noshadow_tanh/policy_latest.pth \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --epsilon 0.0 \
  --num-eval 50 \
  --eval-budget 120 \
  --device cuda \
  --out-dir rope/safety/runs/closed_loop_hj_filter
```

For a fast plumbing check:

```bash
./latent_brs_venv/bin/python -m rope.safety.benchmark_hj_filter_rope \
  --dataset-path rope/data/train_data_noshadow.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --base-method ilqr \
  --hj-cache-path rope/safety/cache/rope_latent_safety_classifier_train_noshadow_tanh.pt \
  --hj-policy-path rope/safety/runs/pyhj_train_noshadow_tanh/policy_latest.pth \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --num-eval 1 \
  --eval-budget 10 \
  --no-videos \
  --device cuda
```

## 7. Unsafe-Trajectory Steering Diagnostic

This script replays dataset actions from unsafe episodes and compares raw replay
against HJ-filtered replay. It is the most direct check that `V` goes unsafe
before `l` and that the safety actor can steer away.

```bash
./latent_brs_venv/bin/python -m rope.safety.steer_unsafe_trajectory \
  --dataset-path rope/data/train_data_noshadow.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --hj-cache-path rope/safety/cache/rope_latent_safety_classifier_train_noshadow_tanh.pt \
  --hj-policy-path rope/safety/runs/pyhj_train_noshadow_tanh/policy_latest.pth \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --num-episodes 5 \
  --device cuda
```

## 8. Replay, HJ Recover, Then iLQR

This diagnostic starts from the beginning of an unsafe recorded episode, replays
the dataset action sequence until the HJ filter warns, executes the PyHJ safety
actor until the barrier is safe again, and then hands off to iLQR to reach the
episode goal.

Each run also saves the paired nominal iLQR rollout without HJ steering under
the same timestamped directory:

```text
<run>/nominal_ilqr/case_.../rollout.mp4
<run>/replay_hj_recover_ilqr/case_.../rollout.mp4
```

During the replay phase, the default `--replay-hj-source dataset` uses exact
HDF5 frames for the HJ state. This is intentional: the unsafe cache/classifier
labels were generated from those stored frames, while live MuJoCo re-renders can
shift classifier scores enough to hide the recorded violation. After the first
override, the HJ filter switches to live simulator frames.

Current existing-checkpoint command:

```bash
./latent_brs_venv/bin/python -m rope.safety.replay_hj_recover_ilqr \
  --dataset-path rope/data/train_data_noshadow.h5 \
  --model-dir rope/models/mlpdyn_noshadow_ft \
  --hj-cache-path rope/safety/cache/rope_latent_safety_classifier_train_noshadow.pt \
  --hj-policy-path rope/safety/runs/pyhj_train_noshadow/policy_latest.pth \
  --classifier-checkpoint rope/safety/obs_net/da270d7d1050f110/model.pt \
  --margin-transform identity \
  --actor-hidden 128 128 128 \
  --critic-hidden 128 128 128 \
  --num-episodes 5 \
  --goal-tolerance 0.05 \
  --device cuda \
  --out-dir rope/safety/runs/replay_hj_recover_ilqr_existing
```
