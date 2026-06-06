#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PATCH_DIR="$ROOT/third_party_patches"
mkdir -p "$PATCH_DIR"

save_patch() {
  local name="$1"
  local path="$2"
  shift 2

  local patch="$PATCH_DIR/$name.patch"
  local base="$PATCH_DIR/$name.base"
  local parent_base

  parent_base="$(git ls-files -s "$path" | awk '{print $2}')"
  if [[ -z "$parent_base" ]]; then
    echo "error: parent repository does not track submodule path: $path" >&2
    return 1
  fi

  printf '%s\n' "$parent_base" > "$base"
  git -C "$path" diff --binary --no-ext-diff "$parent_base" -- > "$patch"

  local rel
  for rel in "$@"; do
    if [[ ! -f "$path/$rel" ]]; then
      echo "warning: reviewed untracked file is missing: $path/$rel" >&2
      continue
    fi
    git -C "$path" diff --no-index --binary -- /dev/null "$rel" >> "$patch" || {
      status=$?
      if [[ "$status" -ne 1 ]]; then
        return "$status"
      fi
    }
  done

  echo "saved: $patch"
}

cd "$ROOT"

save_patch gpu_sls third_party/gpu_sls \
  src/gpu_sls/mppi_planner.py

save_patch latent-safety third_party/latent-safety

save_patch le-wm third_party/le-wm \
  config/eval/reacher_latentbrs.yaml \
  convert_hf_pusht_ckpt.py \
  convert_velocity_ckpt.py

save_patch stable-pretraining third_party/stable-pretraining \
  stable_pretraining/_version.py

save_patch stable-worldmodel third_party/stable-worldmodel \
  experiments/lewm-pusht/config.json \
  experiments/lewm-pusht/README.md \
  scripts/expert/train_pusht.py \
  scripts/expert/viz_diffusion_pusht_expert.py \
  scripts/expert/viz_pusht_expert.py \
  stable_worldmodel/envs/pusht/diffusion_expert_policy.py \
  stable_worldmodel/envs/pusht/sb3_wrapper.py
