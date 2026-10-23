#!/usr/bin/env bash
set -euo pipefail

# Batch-size sensitivity sweep for the circle_trio PiFM experiment.
# Runs 8 batch sizes × 5 seeds = 40 independent training runs via Hydra multirun.
#
# Examples:
#   bash scripts/run_circle_trio_batchsize.sh
#   bash scripts/run_circle_trio_batchsize.sh trainer.max_epochs=100

EXTRA_OVERRIDES=("$@")
SWEEP_ID="$(date +%Y-%m-%d_%H-%M-%S)"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs}"
LOGS_DIR="${LOGS_DIR%/}"
SWEEP_DIR="${LOGS_DIR}/circle_trio_pifm/sweeps/batchsize/${SWEEP_ID}"

echo "Starting circle_trio batch-size sweep → ${SWEEP_DIR}"

python src/train.py -m \
  "experiment=circle_trio_pifm" \
  "hparams_search=circle_trio_batchsize" \
  "hydra.sweep.dir=${SWEEP_DIR}" \
  "${EXTRA_OVERRIDES[@]}"

echo "Sweep complete. Results in: ${SWEEP_DIR}"
