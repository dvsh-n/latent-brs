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

## Files That Matter

- `rope/data/iiwa_hardware.py`: Drake LCM station for the two iiwas.
- `rope/data/home_bimanual_7d.py`: conservative homing script.
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

2. Do one short collection dry run:

```bash
python3 rope/data/rope_real_data_gen.py \
  --robot-backend drake-lcm \
  --num-trajectories 1 \
  --max-episode-steps 20 \
  --save-mp4 \
  --camera-index 0 \
  --i-understand-this-moves-real-robots
```

3. Inspect the generated MP4 and HDF5 before collecting more:

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

4. Collect a larger run after the dry run looks sane:

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
```

For the first hardware test, keep `--max-episode-steps` small and keep someone
near the E-stop.
