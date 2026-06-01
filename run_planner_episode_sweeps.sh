#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

print_usage() {
  cat <<'EOF'
Usage:
  bash run_planner_episode_sweeps.sh
  bash run_planner_episode_sweeps.sh --episodes 0 1 2 3
  bash run_planner_episode_sweeps.sh --start 0 --end 9

Options:
  --episodes IDX [IDX ...]  Explicit episode indices to sweep.
  --start N                 Inclusive start of an integer episode range.
  --end N                   Inclusive end of an integer episode range.
  --foreground              Run in the current shell instead of relaunching under nohup.
  -h, --help                Show this help message.

The script runs planners in this order for every episode index:
  1. rope
  2. reacher
  3. ogbench_cube

Default behavior with no episode arguments: sweep episode_idx 0..34.
EOF
}

ORIGINAL_ARGS=("$@")
RUN_FOREGROUND=0
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
  else
    for ((idx=0; idx<=34; idx++)); do
      EPISODES+=("${idx}")
    done
  fi
fi

LOG_DIR="${SCRIPT_DIR}/sweep_logs"
mkdir -p "${LOG_DIR}"
SWEEP_ROOT="${SCRIPT_DIR}/sweeps"
mkdir -p "${SWEEP_ROOT}"

if [[ ${RUN_FOREGROUND} -eq 0 && -z "${PLANNER_SWEEP_DETACHED:-}" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_PATH="${LOG_DIR}/planner_sweep_${TIMESTAMP}.log"
  nohup env PLANNER_SWEEP_DETACHED=1 bash "$0" --foreground "${ORIGINAL_ARGS[@]}" > "${LOG_PATH}" 2>&1 &
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

echo "Activated conda environment: $(conda info --json | python -c 'import json,sys; print(json.load(sys.stdin)[\"active_prefix_name\"])')"
echo "Running planner sweeps from ${SCRIPT_DIR}"
echo "Episode indices: ${EPISODES[*]}"
echo "Sweep root: ${SWEEP_ROOT}"

ROPE_CONFIG="rope/plan/sample_config_mppi.yaml"
REACHER_CONFIG="reacher/plan/sample_config_sls_mppi.yaml"
OGBENCH_SCRIPT="ogbench_cube/plan/plan_sls_mpc_mppi_fixed_grasp.py"
OGBENCH_CONFIG="ogbench_cube/plan/sample_config_mppi.yaml"

ROPE_OUT_DIR="${SWEEP_ROOT}/rope"
REACHER_OUT_DIR="${SWEEP_ROOT}/reacher"
OGBENCH_OUT_DIR="${SWEEP_ROOT}/ogbench_cube"

mkdir -p "${ROPE_OUT_DIR}" "${REACHER_OUT_DIR}" "${OGBENCH_OUT_DIR}"

SNAPSHOT_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_DIR="${SWEEP_ROOT}/launch_snapshot_${SNAPSHOT_TIMESTAMP}"
mkdir -p "${SNAPSHOT_DIR}/rope" "${SNAPSHOT_DIR}/reacher" "${SNAPSHOT_DIR}/ogbench_cube"

cp "rope/plan/plan_sls_mpc_mppi.py" "${SNAPSHOT_DIR}/rope/"
cp "${ROPE_CONFIG}" "${SNAPSHOT_DIR}/rope/"
cp "reacher/plan/plan_sls_mpc_mppi.py" "${SNAPSHOT_DIR}/reacher/"
cp "${REACHER_CONFIG}" "${SNAPSHOT_DIR}/reacher/"
cp "${OGBENCH_SCRIPT}" "${SNAPSHOT_DIR}/ogbench_cube/"
cp "${OGBENCH_CONFIG}" "${SNAPSHOT_DIR}/ogbench_cube/"

printf '%s\n' "${EPISODES[@]}" > "${SNAPSHOT_DIR}/episode_indices.txt"
{
  echo "snapshot_timestamp=${SNAPSHOT_TIMESTAMP}"
  echo "rope_script=rope/plan/plan_sls_mpc_mppi.py"
  echo "rope_config=${ROPE_CONFIG}"
  echo "reacher_script=reacher/plan/plan_sls_mpc_mppi.py"
  echo "reacher_config=${REACHER_CONFIG}"
  echo "ogbench_cube_script=${OGBENCH_SCRIPT}"
  echo "ogbench_cube_config=${OGBENCH_CONFIG}"
  echo "rope_out_dir=${ROPE_OUT_DIR}"
  echo "reacher_out_dir=${REACHER_OUT_DIR}"
  echo "ogbench_cube_out_dir=${OGBENCH_OUT_DIR}"
} > "${SNAPSHOT_DIR}/run_manifest.txt"

echo "Saved launch snapshot to ${SNAPSHOT_DIR}"

run_planner() {
  local planner_label="$1"
  local script_path="$2"
  local config_path="$3"
  local episode_idx="$4"
  local out_dir="$5"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${planner_label}: episode_idx=${episode_idx}, out_dir=${out_dir}"
  if python "${script_path}" --config_path="${config_path}" --episode_idx="${episode_idx}" --out_dir="${out_dir}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${planner_label}: episode_idx=${episode_idx} completed successfully"
  else
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${planner_label}: episode_idx=${episode_idx} failed with exit code ${exit_code}" >&2
    return 0
  fi
}

for episode_idx in "${EPISODES[@]}"; do
  run_planner \
    "rope" \
    "rope/plan/plan_sls_mpc_mppi.py" \
    "${ROPE_CONFIG}" \
    "${episode_idx}" \
    "${ROPE_OUT_DIR}"

  run_planner \
    "reacher" \
    "reacher/plan/plan_sls_mpc_mppi.py" \
    "${REACHER_CONFIG}" \
    "${episode_idx}" \
    "${REACHER_OUT_DIR}"

  run_planner \
    "ogbench_cube" \
    "${OGBENCH_SCRIPT}" \
    "${OGBENCH_CONFIG}" \
    "${episode_idx}" \
    "${OGBENCH_OUT_DIR}"
done

echo
echo "All planner sweeps completed."
