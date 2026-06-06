#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

while read -r _ path; do
  if [[ ! -e "$path/.git" ]]; then
    echo "skip (not initialized): $path"
    continue
  fi
  if ! git -C "$path" config --get remote.origin.url >/dev/null; then
    echo "skip (no origin): $path"
    continue
  fi
  git -C "$path" config remote.origin.pushurl "disabled://push-protected"
  echo "protected: $path"
done < <(git config --file .gitmodules --get-regexp '^submodule\..*\.path$')
