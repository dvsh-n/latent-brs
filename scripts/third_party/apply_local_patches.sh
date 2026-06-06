#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PATCH_DIR="$ROOT/third_party_patches"

apply_patch() {
  local name="$1"
  local path="$2"
  local patch="$PATCH_DIR/$name.patch"
  local base="$PATCH_DIR/$name.base"

  if [[ ! -e "$path/.git" ]]; then
    echo "skip (not initialized): $path"
    return
  fi
  if [[ ! -s "$patch" ]]; then
    echo "skip (empty patch): $path"
    return
  fi

  local expected actual
  expected="$(cat "$base")"
  actual="$(git -C "$path" rev-parse HEAD)"
  if [[ "$expected" != "$actual" ]]; then
    echo "warning: $path is at $actual; patch was saved from $expected" >&2
  fi

  git -C "$path" apply --3way "$patch"
  echo "applied: $path"
}

cd "$ROOT"

apply_patch gpu_sls third_party/gpu_sls
apply_patch latent-safety third_party/latent-safety
apply_patch le-wm third_party/le-wm
apply_patch stable-pretraining third_party/stable-pretraining
apply_patch stable-worldmodel third_party/stable-worldmodel
