# OGBench Cube LPB

This folder contains the OGBench Cube implementation of LPB, short for Latent Policy Barrier. The upstream reference lives at `third_party/lpb`.

## What LPB Is

LPB is a test-time action-optimization method for visuomotor policies. Its core idea is simple: plans should stay close to the latent state distribution represented by expert or otherwise trusted rollouts. In the upstream repository, this is implemented by training a visual dynamics model, then optimizing candidate actions with a cost that includes a learned latent in-distribution barrier. The barrier discourages trajectories whose predicted latent states drift away from the support of the reference dataset.

For cube, we keep the same code idea but adapt the policy class and dynamics interface to this repository:

```text
trusted cube data -> encode frames -> Markov latent states -> safe prototype bank
current rollout -> iLQR predicts Markov latent states -> nearest-prototype LPB penalty
```

This is code-faithful to the LPB mechanism, not an experiment reproduction. The cube implementation does not reproduce the paper's diffusion-policy tasks, datasets, or reported numbers.

## Faithfulness To `third_party/lpb`

The upstream LPB pipeline has three relevant code phases:

1. Train a base visuomotor policy.
2. Train a visual dynamics model in the policy's latent/image space.
3. At inference, optimize actions through that dynamics model while penalizing predicted latent states that go out of distribution.

The cube implementation mirrors phase 3 and the barrier data structure:

- Upstream uses a learned visual dynamics model for rollout prediction; cube uses the existing OGBench DINO-WM/MLP dynamics model loaded by `ogbench.plan.benchmark_cube_hard`.
- Upstream builds a latent reference distribution from demonstrations/rollouts; cube builds a prototype bank from encoded cube expert data.
- Upstream optimizes actions at test time with an LPB cost; cube adds the LPB cost directly into the iLQR stage/terminal derivative path.
- Upstream treats the barrier as a distribution-support regularizer, independent of the environment's true safety rule; cube optionally filters the bank to analytically/classifier-safe states, then still applies the LPB as a latent support penalty.

The cube implementation intentionally avoids copying the full `third_party/lpb/diffusion_policy` stack because OGBench already has a trained world model, iLQR planner, data loader, and benchmark loop. The faithful part is the LPB mechanism: encode trusted states, normalize them, compare predicted latent states to a saved reference set, and penalize out-of-bank states during action optimization.

## Files

- `barrier.py`: runtime `CubeLPBBarrier`. It loads a saved prototype bank, whitens query Markov states with saved mean/std, computes nearest-prototype distances, returns squared excess penalties, and reports LPB diagnostics.
- `build_bank_cube.py`: CLI for building a cube LPB bank. It encodes dataset pixels, creates Markov latent states, optionally filters them by analytic or classifier margin, samples prototypes, calibrates a nearest-neighbor threshold, and writes a `.pt` bank.
- `__init__.py`: exports `CubeLPBBarrier`.

Related LPB integration outside this folder:

- `ogbench/plan/benchmark_cube_hard.py`: adds LPB CLI flags, loads `CubeLPBBarrier`, adds LPB penalties and Hessian/gradient terms inside iLQR, and logs diagnostics.
- `ogbench/safety/benchmark_hj_filter_cube.py`: has comparison modes for `lpb`, `hj_lpb`, and `all`.
- `ogbench/safety/latent_cache.py`, `constraints.py`, and `compat.py`: shared helpers reused by the bank builder.

## Pipeline

1. Build or choose a cube world model.

The LPB bank dimension must match the planner model's Markov state dimension. The existing LPB bank under `ogbench/safety/runs/lpb_cube` was built for the current default planner model, `ogbench/models/mlpdyn`, with `latent_dim=24`, `markov_deriv=1`, and `markov_state_dim=48`. The older PyHJ/HJ-filter run used a separate 8-D model, so do not mix its cache or policy with this 48-D LPB bank.

2. Build the LPB prototype bank.

```bash
./latent_brs_venv/bin/python -m ogbench.safety.lpb.build_bank_cube \
  --dataset-path ogbench/data/expert_data/ogbench_cube_expert.h5 \
  --model-dir ogbench/models/mlpdyn \
  --output-path ogbench/safety/runs/lpb_cube/lpb_bank_4096_analytic.pt \
  --device cuda \
  --max-frames 4096 \
  --num-prototypes 1024 \
  --margin-source analytic \
  --threshold-quantile 0.95 \
  --overwrite
```

What happens:

- Load the same model config/checkpoint used by the cube planner.
- Encode dataset pixels with the model.
- Build Markov states by stacking the current and derivative/history latent states.
- Compute margins if `--margin-source` is `analytic` or `classifier`.
- Keep states whose margin is above `--safe-margin-min`.
- Whiten states by saved mean/std.
- Sample prototypes and calibrate the distance threshold from held-out non-prototype states.
- Save `prototypes`, `state_mean`, `state_std`, `threshold`, and metadata.

3. Run cube iLQR with LPB enabled.

```bash
./latent_brs_venv/bin/python -m ogbench.plan.benchmark_cube_hard \
  --method ilqr \
  --dataset-path ogbench/data/test_data/ogbench_cube_test.h5 \
  --model-dir ogbench/models/mlpdyn \
  --lpb-bank-path ogbench/safety/runs/lpb_cube/lpb_bank_4096_analytic.pt \
  --lpb-weight 1.0 \
  --lpb-threshold-scale 1.0 \
  --num-eval 10 \
  --device cuda
```

What happens inside iLQR:

- Each candidate trajectory is predicted in Markov latent space.
- `CubeLPBBarrier.state_penalty(x)` computes `relu(nearest_distance(x) - threshold)^2`.
- `trajectory_penalty` averages that value across stage states by default.
- iLQR adds the weighted LPB gradient/Hessian to the normal goal/control objective.
- Step records include `lpb_threshold`, distance statistics, violation rate, penalty, and weight.

4. Compare LPB with HJ filtering.

Use `ogbench.safety.benchmark_hj_filter_cube --mode all` only when the HJ cache/policy/classifier and the LPB bank were built from the same cube world model. The current checked-in LPB artifact is 48-D, while the previous HJ artifacts are 16-D, so those exact artifacts are not a valid HJ+LPB combination. For LPB-only evaluation with the current artifacts, use `ogbench.plan.benchmark_cube_hard` as shown above.

## Previous Run Analysis

Artifacts analyzed:

- `ogbench/safety/runs/cube_embd8_full_20260524_175429`
- `ogbench/safety/runs/cube_embd8_full_20260524_175436/pipeline.log`
- `ogbench/safety/runs/lpb_cube/lpb_bank_4096_analytic.pt`
- `ogbench/plan/ogbench_cube_hard_eval/logs`

There were two model tracks:

- LPB bank/eval used the default 24-D cube model at `ogbench/models/mlpdyn`, with `markov_state_dim=48`.
- PyHJ/HJ-filter safety used the older 8-D cube model at `ogbench/experiments/cube_obstacle/mlpdyn_embd_8`, with `markov_state_dim=16`.

This matters because LPB banks, latent caches, policies, and planners are dimension-locked to the world model that produced their latents.

LPB bank and small comparison:

- Bank: `lpb_bank_4096_analytic.pt`.
- Model: `ogbench/models/mlpdyn/lewm_epoch_15_object.ckpt`.
- Bank metadata: `latent_dim=24`, `markov_state_dim=48`, `total_frames=4096`, `safe_state_count=4020`, `num_prototypes=1024`.
- Margin source: analytic half-ellipse obstacle rule.
- Calibrated LPB threshold: 4.3175 at quantile 0.95.
- Calibration distances: mean 1.8435, max 9.6489.
- Tiny 3-case comparison: nominal iLQR success 100%; LPB-guided iLQR success 100%.
- Mean final cube distance was essentially unchanged: about 0.0381 nominal vs 0.0385 LPB.
- The LPB comparison is only 3 episodes, so it proves the code path runs but is too small to claim a performance effect.

Hard cube planning baseline context:

- 5-seed iLQR success: 91.5% +/- 3.74.
- 5-seed DINO-WM/CEM success: 69.0% +/- 2.55.
- 5-seed LeWM/CEM success: 28.5% +/- 3.39.
- PLDM/CEM success: 28.75% +/- 5.73, but the seed set is incomplete.

Classifier:

- Model: `ogbench/experiments/cube_obstacle/mlpdyn_embd_8/lewm_epoch_2_object.ckpt`.
- Data: `ogbench/experiments/cube_obstacle/obstacle_data/obstacle_classifier_data-002.pt`.
- Size: 16,384 balanced samples.
- Train/val/cal accuracy: 0.9940 / 0.9946 / 0.9933.
- Recall on obstacle class was 1.0 for train, validation, and calibration.

Latent cache and smoke:

- Cache: `cube_latent_safety_embd8_train_tanh2.pt`.
- Frames: 761,258; valid transitions: 750,031.
- Unsafe fraction: 0.0573.
- Five-episode smoke rollout had 0 failures and worst min margin about 0.9798.

PyHJ training and evaluation:

- Training ran 2,000,000 environment steps over 50 epochs.
- Training duration was about 2,153 seconds.
- Best test reward was 25.0.
- In 1,000 evaluation episodes, random rollout failure rate was 29.6%; the learned policy failure rate was 6.9%.
- Among safe starts, random rollout failure was 24.54%; learned policy rollout failure was 0.214%.
- Critic value safe AUC was 0.986; clipped value safe AUC was 1.0.
- Value/margin correlation was strong: 0.803 for critic value and 0.994 for clipped value.

Closed-loop HJ benchmark on held-out geometry-unsafe episodes:

- Episodes: `[0, 3, 5, 8, 16, 17, 24, 26, 36, 37]`.
- Nominal iLQR success: 90%.
- Nominal safety violation: 100%.
- HJ-filtered success: 60%.
- HJ-filtered safety violation: 100%.
- Mean override count: 10.3; mean override rate: 15.17%.
- HJ filtering improved mean minimum analytic/classifier margin only slightly, from about -0.975 to -0.935, and reduced task success.

Interpretation:

The latent PyHJ policy learned a meaningful recovery behavior in its own latent environment, but the closed-loop filter did not solve the held-out geometry-unsafe cube cases. The filter was active, but the barrier definition `min(classifier_margin, critic_value)` was dominated by the margin crossing; once the nominal trajectory entered the geometry-unsafe region, the learned recovery was not enough to prevent violations in the real rollout. Treat the prior closed-loop result as a negative safety result with a useful diagnostic: latent value quality was high offline, but online intervention strength/triggering was insufficient for this held-out set.

The LPB bank artifact exists separately at `ogbench/safety/runs/lpb_cube/lpb_bank_4096_analytic.pt`. It should be evaluated with the matching 24-D default model and on a larger held-out set before drawing conclusions about LPB itself.

## Quick Checks

```bash
./latent_brs_venv/bin/python -m compileall -q ogbench/safety/lpb ogbench/plan/benchmark_cube_hard.py
./latent_brs_venv/bin/python -m ogbench.safety.lpb.build_bank_cube --help
```
