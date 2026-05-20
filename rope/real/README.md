# Real Rope Data Collection

`rope/data/rope_real_data_gen.py` collects real-hardware rope trajectories into
the same HDF5 schema as the MuJoCo collector.

The hardware path is:

```text
random rope task policy
  -> MuJoCo LabEnv IK helper
  -> Drake LCM bimanual iiwa station
  -> real KUKA arms
  -> OpenCV camera + measured iiwa state
  -> HDF5 dataset
```

MuJoCo is used only for IK/FK helper math. It does not simulate the real data.

The standalone Cartesian motion scripts use `iiwa_cartesian_ik.py` and Drake's
single-arm iiwa model. They are useful for hardware sanity checks, but the rope
dataset collector still uses `LabEnv` because it already encodes the shared
two-arm/table/rope task frame.

## Files That Matter

- `rope/data/iiwa_hardware.py`: Drake LCM station for the two iiwas.
- `rope/data/iiwa_cartesian_ik.py`: Drake single-arm Cartesian IK helper for manual tests.
- `rope/data/home_bimanual_7d.py`: conservative homing script to the mirrored rope-task home pose.
- `rope/data/test_bimanual_cartesian_motion.py`: small Cartesian hardware sanity test.
- `rope/data/motion_bimanual_position.py`: larger manual Cartesian waypoint motion.
- `rope/data/rope_real_data_gen.py`: real dataset collector.
- `rope/real/drake_lcm_backend.py`: backend that connects the collector to the Drake LCM station.

## Arm Mapping

Default mapping:

```text
IIWA_STATUS / IIWA_COMMAND      -> robot0 -> left / positive-y rope arm
IIWA_STATUS_2 / IIWA_COMMAND_2  -> robot1 -> right / negative-y rope arm
```

The HDF5 `qpos`, `qvel`, and `control` order is:

```text
left_q[0:7], right_q[0:7]
```

Use `--arm-mapping robot1-left` only if the physical setup is swapped.

## Lab Run Sequence

Run everything from the repo root on the lab machine that has Drake and can see
the iiwa LCM channels.

1. Home the arms:

```bash
python3 rope/data/home_bimanual_7d.py
```

This reads the current iiwa joint positions, computes the mirrored rope home
pose from `LabEnv`, checks the full two-arm path for arm-arm collision, and then
moves slowly to that pose. The two attachment sites should have nearly equal
`x/z` and opposite-sign `y`.

2. Optional small Cartesian sanity test:

```bash
python3 rope/data/test_bimanual_cartesian_motion.py
```

Skip this if homing and LCM status are already known-good. The real collector
does not depend on this script.

3. Do one short collection dry run with collision guard enabled:

```bash
python3 rope/data/rope_real_data_gen.py \
  --robot-backend drake-lcm \
  --num-trajectories 1 \
  --max-episode-steps 20 \
  --save-mp4 \
  --camera-index 0 \
  --i-understand-this-moves-real-robots
```

4. Inspect the generated MP4 and HDF5 before collecting more:

```bash
python3 - <<'PY'
import h5py
import numpy as np
from pathlib import Path

path = Path("rope/data/real_data/rope_random_cubic_spline.h5")
with h5py.File(path, "r") as h5:
    print("episodes:", h5.attrs["num_episodes"])
    print("frames:", h5["pixels"].shape)
    print("actions:", h5["action"].shape)
    print("qpos:", h5["qpos"].shape)
    print("finite actions:", (~np.isnan(h5["action"][:]).any(axis=1)).sum())
PY
```

5. Collect a larger run after the dry run looks sane:

```bash
python3 rope/data/rope_real_data_gen.py \
  --robot-backend drake-lcm \
  --outdir rope/data/real_data \
  --output-name rope_random_cubic_spline_real.h5 \
  --num-trajectories 100 \
  --save-mp4 \
  --camera-index 0 \
  --i-understand-this-moves-real-robots
```

## Safety Flags

Defaults are intentionally conservative:

```text
--control-timestep 0.05
--reset-duration 3.0
--drake-publish-period 0.005
--status-timeout 5.0
--max-control-joint-step-deg 5.0
--max-reset-joint-move-deg 90.0
--arm-arm-min-distance 0.06
--collision-control-samples 5
--collision-reset-samples 25
```

The collector checks the interpolated joint path in the two-arm MuJoCo model
before every hardware command. The homing and optional Cartesian scripts run
the same arm-arm path check before sending their planned motion. These checks
reject arm-arm contacts and any sampled arm-arm collision geometry distance
below `--arm-arm-min-distance`. Keep this enabled unless you are deliberately
debugging the guard itself.

For the first hardware test, keep `--max-episode-steps` small and keep someone
near the E-stop.
