#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

config="${script_dir}/rope_real_chained_smoke_cl.yaml"
python_bin="/usr/bin/python"
generator="${script_dir}/rope_real_chained_data_gen_cl.py"
outdir="${script_dir}/real_data_new"
dest_dir="/media/daniel/Anutam SSD/real_rope_data"
count=""
seed=""
seed_step="20"
stable_seconds="1"
dry_run="0"
generator_args=()

usage() {
  cat >&2 <<EOF
usage: $0 --count COUNT [options] [-- extra generator args]

Runs the real chained data generator COUNT times.
After each run, async camera side files are moved to external storage.

Required:
  --count COUNT             Number of full collection+move cycles to run.

Options:
  --config PATH             YAML config. Default: ${config}
  --python PATH             Python interpreter. Default: ${python_bin}
  --generator PATH          Generator script. Default: ${generator}
  --outdir DIR              Data output directory. Default: ${outdir}
  --dest DIR                Async-file destination. Default: ${dest_dir}
  --seed N                  Starting seed. Default: read from --config.
  --seed-step N             Amount to add after each run. Default: ${seed_step}
  --stable-seconds N        File stability wait before moving. Default: ${stable_seconds}
  --dry-run                 Print commands without running collection or moving files.
  -h, --help                Show this help.

Examples:
  $0 --count 1
  $0 --count 5 --seed 7000 --seed-step 20
  $0 --count 3 -- --num-trajectories 2
EOF
}

read_seed_from_config() {
  local path="$1"
  "${python_bin}" - "$path" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1]).expanduser()
try:
    import yaml
except ImportError:
    yaml = None

if yaml is not None:
    data = yaml.safe_load(path.read_text()) or {}
    if "seed" in data:
        print(int(data["seed"]))
        raise SystemExit(0)

for line in path.read_text().splitlines():
    match = re.match(r"^\s*seed\s*:\s*([-+]?\d+)\s*(?:#.*)?$", line)
    if match:
        print(int(match.group(1)))
        raise SystemExit(0)

raise SystemExit(f"could not find seed in {path}")
PY
}

move_async_files() {
  if [[ "${dry_run}" == "1" ]]; then
    "${python_bin}" "${script_dir}/move_real_rope_async_data.py" \
      --source "${outdir}" \
      --dest "${dest_dir}" \
      --stable-seconds "${stable_seconds}" \
      --once \
      --dry-run
  else
    "${python_bin}" "${script_dir}/move_real_rope_async_data.py" \
      --source "${outdir}" \
      --dest "${dest_dir}" \
      --stable-seconds "${stable_seconds}" \
      --once
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count)
      count="${2:-}"
      shift 2
      ;;
    --config)
      config="${2:-}"
      shift 2
      ;;
    --python)
      python_bin="${2:-}"
      shift 2
      ;;
    --generator)
      generator="${2:-}"
      shift 2
      ;;
    --outdir)
      outdir="${2:-}"
      shift 2
      ;;
    --dest)
      dest_dir="${2:-}"
      shift 2
      ;;
    --seed)
      seed="${2:-}"
      shift 2
      ;;
    --seed-step)
      seed_step="${2:-}"
      shift 2
      ;;
    --stable-seconds)
      stable_seconds="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      generator_args=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${count}" ]]; then
  echo "--count is required" >&2
  usage
  exit 2
fi

if [[ ! -f "${config}" ]]; then
  echo "config does not exist: ${config}" >&2
  exit 1
fi
config="$(cd -- "$(dirname -- "${config}")" && pwd)/$(basename -- "${config}")"

if [[ ! -x "${python_bin}" ]]; then
  echo "python interpreter does not exist or is not executable: ${python_bin}" >&2
  exit 1
fi

if [[ ! -f "${generator}" ]]; then
  echo "generator does not exist: ${generator}" >&2
  exit 1
fi
generator="$(cd -- "$(dirname -- "${generator}")" && pwd)/$(basename -- "${generator}")"
mkdir -p -- "${outdir}"
outdir="$(cd -- "${outdir}" && pwd)"

if ! [[ "${count}" =~ ^[0-9]+$ ]]; then
  echo "--count must be a positive integer: ${count}" >&2
  exit 2
fi

if [[ "${count}" -le 0 ]]; then
  echo "--count must be positive" >&2
  exit 2
fi

if [[ -z "${seed}" ]]; then
  seed="$(read_seed_from_config "${config}")"
fi

if ! [[ "${seed}" =~ ^[-+]?[0-9]+$ ]]; then
  echo "--seed must be an integer: ${seed}" >&2
  exit 2
fi

if ! [[ "${seed_step}" =~ ^[-+]?[0-9]+$ ]]; then
  echo "--seed-step must be an integer: ${seed_step}" >&2
  exit 2
fi

run_index=0

cd "${repo_root}"

echo "loop started at $(date)"
echo "count: ${count}"
echo "config: ${config}"
echo "python: ${python_bin}"
echo "generator: ${generator}"
echo "outdir: ${outdir}"
echo "async destination: ${dest_dir}"
echo "starting seed: ${seed}"
echo "seed step: ${seed_step}"

while [[ "${run_index}" -lt "${count}" ]]; do
  run_index="$((run_index + 1))"
  echo
  echo "=== run ${run_index}/${count}: seed ${seed} ==="

  cmd=(
    "${python_bin}" "${generator}"
    --config "${config}"
    --outdir "${outdir}"
    --seed "${seed}"
    --shard-id "${seed}"
    --i-understand-this-moves-real-robots
  )
  if [[ "${#generator_args[@]}" -gt 0 ]]; then
    cmd+=("${generator_args[@]}")
  fi

  if [[ "${dry_run}" == "1" ]]; then
    printf 'would run:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi

  move_async_files
  seed="$((seed + seed_step))"
done

echo
echo "completed runs: ${run_index}"
echo "next seed would be: ${seed}"
