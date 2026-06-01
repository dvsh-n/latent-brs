#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

print_usage() {
  cat <<'EOF'
Usage:
  bash run_constrained_ilqr_episode_sweeps.sh
  bash run_constrained_ilqr_episode_sweeps.sh --episodes 0 1 2 3
  bash run_constrained_ilqr_episode_sweeps.sh --start 0 --end 9
  bash run_constrained_ilqr_episode_sweeps.sh --smoke --foreground

Options:
  --episodes IDX [IDX ...]  Explicit episode indices to sweep.
  --start N                 Inclusive start of an integer episode range.
  --end N                   Inclusive end of an integer episode range.
  --smoke                   Run tiny one-step checks instead of the full planner settings.
  --foreground              Run in the current shell instead of relaunching under nohup.
  -h, --help                Show this help message.

Default behavior with no episode arguments: sweep episode_idx 0..34.

For every episode, this runs each planner in two variants:
  1. latent_ellipsoid_on
  2. latent_ellipsoid_off with state bounds widened to [-1000, 1000].
EOF
}

ORIGINAL_ARGS=("$@")
RUN_FOREGROUND=0
RUN_SMOKE=0
START_IDX=""
END_IDX=""
EPISODES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EPISODES+=("$1")
        shift
      done
      ;;
    --start)
      START_IDX="${2:-}"
      shift 2
      ;;
    --end)
      END_IDX="${2:-}"
      shift 2
      ;;
    --smoke)
      RUN_SMOKE=1
      shift
      ;;
    --foreground)
      RUN_FOREGROUND=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#EPISODES[@]} -eq 0 ]]; then
  if [[ -n "${START_IDX}" && -n "${END_IDX}" ]]; then
    if (( START_IDX > END_IDX )); then
      echo "--start must be <= --end" >&2
      exit 1
    fi
    for ((idx=START_IDX; idx<=END_IDX; idx++)); do
      EPISODES+=("${idx}")
    done
  elif [[ ${RUN_SMOKE} -eq 1 ]]; then
    EPISODES=(0)
  else
    for ((idx=0; idx<=34; idx++)); do
      EPISODES+=("${idx}")
    done
  fi
fi

LOG_DIR="${SCRIPT_DIR}/sweep_logs"
mkdir -p "${LOG_DIR}"
SWEEP_ROOT="${SCRIPT_DIR}/sweeps_constrained_ilqr"
mkdir -p "${SWEEP_ROOT}"

if [[ ${RUN_FOREGROUND} -eq 0 && -z "${CONSTRAINED_ILQR_SWEEP_DETACHED:-}" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_PATH="${LOG_DIR}/constrained_ilqr_sweep_${TIMESTAMP}.log"
  nohup env CONSTRAINED_ILQR_SWEEP_DETACHED=1 bash "$0" --foreground "${ORIGINAL_ARGS[@]}" > "${LOG_PATH}" 2>&1 &
  echo "Sweep relaunched under nohup."
  echo "PID: $!"
  echo "Log: ${LOG_PATH}"
  exit 0
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
elif [[ -d "${HOME}/miniconda3" ]]; then
  CONDA_BASE="${HOME}/miniconda3"
elif [[ -d "${HOME}/anaconda3" ]]; then
  CONDA_BASE="${HOME}/anaconda3"
else
  echo "Could not locate a conda installation." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate wm_env

echo "Activated conda environment: $(conda info --json | python -c 'import json,sys; print(json.load(sys.stdin)["active_prefix_name"])')"
echo "Running constrained iLQR sweeps from ${SCRIPT_DIR}"
echo "Episode indices: ${EPISODES[*]}"
echo "Sweep root: ${SWEEP_ROOT}"
if [[ ${RUN_SMOKE} -eq 1 ]]; then
  echo "Smoke mode enabled: max_mpc_steps=1, horizon=2, mppi_horizon=2, mppi_samples=8, mppi_update_iter=1"
fi

ROPE_SCRIPT="rope/plan/plan_constrained_ilqr_mpc_mppi.py"
ROPE_CONFIG="rope/plan/sample_ilqr_config_mppi.yaml"
REACHER_SCRIPT="reacher/plan/plan_constrained_ilqr_mpc_mppi.py"
REACHER_CONFIG="reacher/plan/sample_config_ilqr_mppi.yaml"
OGBENCH_SCRIPT="ogbench_cube/plan/plan_constrained_ilqr_mppi_fixed_grasp.py"
OGBENCH_CONFIG="ogbench_cube/plan/sample_ilqr_config_mppi.yaml"

SNAPSHOT_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_DIR="${SWEEP_ROOT}/launch_snapshot_${SNAPSHOT_TIMESTAMP}"
GENERATED_CONFIG_DIR="${SNAPSHOT_DIR}/generated_configs"
mkdir -p "${SNAPSHOT_DIR}/rope" "${SNAPSHOT_DIR}/reacher" "${SNAPSHOT_DIR}/ogbench_cube" "${GENERATED_CONFIG_DIR}"

cp "${ROPE_SCRIPT}" "${SNAPSHOT_DIR}/rope/"
cp "${ROPE_CONFIG}" "${SNAPSHOT_DIR}/rope/"
cp "${REACHER_SCRIPT}" "${SNAPSHOT_DIR}/reacher/"
cp "${REACHER_CONFIG}" "${SNAPSHOT_DIR}/reacher/"
cp "reacher/plan/constrained_ilqr_common.py" "${SNAPSHOT_DIR}/reacher/"
cp "${OGBENCH_SCRIPT}" "${SNAPSHOT_DIR}/ogbench_cube/"
cp "${OGBENCH_CONFIG}" "${SNAPSHOT_DIR}/ogbench_cube/"

write_variant_config() {
  local domain="$1"
  local variant="$2"
  local src_config="$3"
  local dst_config="$4"

  python - "$domain" "$variant" "$src_config" "$dst_config" <<'PY'
import sys
from pathlib import Path
import yaml

domain, variant, src, dst = sys.argv[1:]
with open(src, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

if variant == "latent_ellipsoid_on":
    if domain == "ogbench_cube":
        cfg.setdefault("state_region_path", "ogbench_cube/eval/latent_ellipsoid")
        cfg["state_abs_limit"] = None
    else:
        cfg["use_latent_ellipsoid_constraint"] = True
elif variant == "latent_ellipsoid_off":
    if domain == "ogbench_cube":
        cfg["state_region_path"] = None
        cfg["state_abs_limit"] = 1000.0
    else:
        cfg["use_latent_ellipsoid_constraint"] = False
        cfg["state_abs_limit"] = 1000.0
        cfg["state_delta_abs_limit"] = 1000.0
else:
    raise ValueError(f"Unknown variant: {variant}")

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
}

DOMAINS=(rope reacher ogbench_cube)
VARIANTS=(latent_ellipsoid_on latent_ellipsoid_off)

for domain in "${DOMAINS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    mkdir -p "${SWEEP_ROOT}/${domain}/${variant}"
  done
done

write_variant_config "rope" "latent_ellipsoid_on" "${ROPE_CONFIG}" "${GENERATED_CONFIG_DIR}/rope_latent_ellipsoid_on.yaml"
write_variant_config "rope" "latent_ellipsoid_off" "${ROPE_CONFIG}" "${GENERATED_CONFIG_DIR}/rope_latent_ellipsoid_off.yaml"
write_variant_config "reacher" "latent_ellipsoid_on" "${REACHER_CONFIG}" "${GENERATED_CONFIG_DIR}/reacher_latent_ellipsoid_on.yaml"
write_variant_config "reacher" "latent_ellipsoid_off" "${REACHER_CONFIG}" "${GENERATED_CONFIG_DIR}/reacher_latent_ellipsoid_off.yaml"
write_variant_config "ogbench_cube" "latent_ellipsoid_on" "${OGBENCH_CONFIG}" "${GENERATED_CONFIG_DIR}/ogbench_cube_latent_ellipsoid_on.yaml"
write_variant_config "ogbench_cube" "latent_ellipsoid_off" "${OGBENCH_CONFIG}" "${GENERATED_CONFIG_DIR}/ogbench_cube_latent_ellipsoid_off.yaml"

printf '%s\n' "${EPISODES[@]}" > "${SNAPSHOT_DIR}/episode_indices.txt"
{
  echo "snapshot_timestamp=${SNAPSHOT_TIMESTAMP}"
  echo "smoke=${RUN_SMOKE}"
  echo "rope_script=${ROPE_SCRIPT}"
  echo "rope_config=${ROPE_CONFIG}"
  echo "reacher_script=${REACHER_SCRIPT}"
  echo "reacher_config=${REACHER_CONFIG}"
  echo "ogbench_cube_script=${OGBENCH_SCRIPT}"
  echo "ogbench_cube_config=${OGBENCH_CONFIG}"
  echo "sweep_root=${SWEEP_ROOT}"
  echo "generated_config_dir=${GENERATED_CONFIG_DIR}"
} > "${SNAPSHOT_DIR}/run_manifest.txt"

echo "Saved launch snapshot to ${SNAPSHOT_DIR}"

run_planner() {
  local planner_label="$1"
  local script_path="$2"
  local config_path="$3"
  local episode_idx="$4"
  local out_dir="$5"
  shift 5
  local extra_args=("$@")

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${planner_label}: episode_idx=${episode_idx}, out_dir=${out_dir}"
  if python "${script_path}" --config_path="${config_path}" --episode_idx="${episode_idx}" --out_dir="${out_dir}" "${extra_args[@]}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${planner_label}: episode_idx=${episode_idx} completed successfully"
  else
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${planner_label}: episode_idx=${episode_idx} failed with exit code ${exit_code}" >&2
    return 0
  fi
}

for episode_idx in "${EPISODES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    COMMON_EXTRA=()
    OGBENCH_EXTRA=()
    if [[ ${RUN_SMOKE} -eq 1 ]]; then
      COMMON_EXTRA=(--max_mpc_steps=1 --horizon=2 --mppi_horizon=2 --mppi_samples=8 --mppi_update_iter=1)
      OGBENCH_EXTRA=(--max_mpc_steps=1 --max_oracle_steps=80 --horizon=2 --mppi_horizon=2 --mppi_samples=8 --mppi_update_iter=1)
    fi

    run_planner \
      "rope/${variant}" \
      "${ROPE_SCRIPT}" \
      "${GENERATED_CONFIG_DIR}/rope_${variant}.yaml" \
      "${episode_idx}" \
      "${SWEEP_ROOT}/rope/${variant}" \
      "${COMMON_EXTRA[@]}"

    run_planner \
      "reacher/${variant}" \
      "${REACHER_SCRIPT}" \
      "${GENERATED_CONFIG_DIR}/reacher_${variant}.yaml" \
      "${episode_idx}" \
      "${SWEEP_ROOT}/reacher/${variant}" \
      "${COMMON_EXTRA[@]}"

    run_planner \
      "ogbench_cube/${variant}" \
      "${OGBENCH_SCRIPT}" \
      "${GENERATED_CONFIG_DIR}/ogbench_cube_${variant}.yaml" \
      "${episode_idx}" \
      "${SWEEP_ROOT}/ogbench_cube/${variant}" \
      "${OGBENCH_EXTRA[@]}"
  done
done

echo
echo "All constrained iLQR sweeps completed."
