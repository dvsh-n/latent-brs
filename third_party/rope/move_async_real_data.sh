#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DEST_DIR" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${script_dir}/real_data"
dest_dir="$1"

if [[ ! -d "${source_dir}" ]]; then
  echo "source directory does not exist: ${source_dir}" >&2
  exit 1
fi

mkdir -p -- "${dest_dir}"

find "${source_dir}" -maxdepth 1 -type f -name '*async*' -print0 |
  while IFS= read -r -d '' file; do
    echo "moving ${file} -> ${dest_dir}/"
    mv -n -- "${file}" "${dest_dir}/"
  done
