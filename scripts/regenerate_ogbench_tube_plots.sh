#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-ogbench_cube/plan/sls_mppi_conformal_tube_vis/1780011278_mppi_sls_cube_episode_00028}"
PLAN_STRIDE="${PLAN_STRIDE:-5}"
MAX_PLANS="${MAX_PLANS:-7}"
START_STEP="${START_STEP:-1}"
ALPHA="${ALPHA:-0.5}"
HORIZON_ALPHA_DECAY="${HORIZON_ALPHA_DECAY:-0.89}"
CONDA_ENV="${CONDA_ENV:-wm_env}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

mkdir -p "${MPLCONFIGDIR}"

conda run -n "${CONDA_ENV}" python ogbench_cube/plan/plot_saved_latent_tubes.py \
  "${RUN_DIR}" \
  --start-step "${START_STEP}" \
  --plan-stride "${PLAN_STRIDE}" \
  --max-plans "${MAX_PLANS}" \
  --alpha "${ALPHA}" \
  --horizon-alpha-decay "${HORIZON_ALPHA_DECAY}"
