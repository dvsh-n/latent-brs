# Latent BRS

Research code for learning latent dynamics and evaluating planning and safety
methods across several control environments.

## Repository Layout

- `reacher/`, `pusht/`, `rope/`, and `ogbench/`: environment-specific data,
  training, evaluation, planning, and safety code.
- Task-local `experiments/` directories: preserved experimental workflows that
  may depend on legacy code or local artifacts.
- `shared/`: code intended for reuse across environments.
- `scripts/`: visualization utilities and third-party maintenance helpers.
- `third_party/`: external repositories managed primarily as Git submodules.
- `data/`, task-local `data/` directories, `models/`, and output directories:
  local datasets, checkpoints, and generated experiment artifacts.

Large datasets, checkpoints, rendered media, logs, and experiment outputs are
ignored by Git. Keep their locations stable locally, but do not add them to the
repository.

## Setup

Clone the repository and initialize its submodules:

```bash
git clone --recurse-submodules https://github.com/dvsh-n/latent-brs.git
cd latent-brs
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Create an environment and install the project:

```bash
python -m venv latent_brs_venv
source latent_brs_venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

The pinned `requirements.txt` represents the current research environment and
includes GPU-specific packages. Adjust CUDA-dependent packages for the target
machine when necessary.

Optional development tools can be installed with:

```bash
python -m pip install -e ".[dev]"
```

## Development Checks

Run lightweight static checks:

```bash
ruff check .
```

Run a targeted test or smoke test rather than collecting every research script:

```bash
pytest path/to/test_file.py
```

Many experiments require local datasets, checkpoints, GPUs, simulators, or
hardware. Verify those prerequisites before running task scripts.

## Working With Submodules

The parent repository records only the commit checked out by each submodule.
Local modifications inside a submodule are not included in parent-repository
commits.

Inspect submodule state with:

```bash
git submodule status
git status --short
```

After pulling parent-repository updates, synchronize submodules with:

```bash
git submodule update --init --recursive
```

Local third-party code changes are preserved as parent-repository patches
instead of being pushed to upstream projects. See
[`scripts/third_party/README.md`](scripts/third_party/README.md) for the
push-protection and patch save/apply workflow.
