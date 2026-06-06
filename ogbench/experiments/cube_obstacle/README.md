# Cube Obstacle Experiments

Legacy OGBench cube obstacle-collection, classifier-training, and planning
experiments live here with their local artifacts.

Some scripts still import modules from the former `ogbench_cube` package. They
are preserved for reproducibility, but require that legacy package or a future
port to the current `ogbench` package before they can run.

Primary scripts:

- `collect_height_obstacle_data.py`
- `train_obstacle_classifier.py`
- `plan_ilqr_mpc.py`
- `plan_endpoint_ilqr_mpc.py`
- `plan_sls_mpc_mppi.py`
