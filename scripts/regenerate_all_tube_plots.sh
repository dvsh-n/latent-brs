#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/regenerate_ogbench_tube_plots.sh"
"$(dirname "$0")/regenerate_reacher_tube_plots.sh"
