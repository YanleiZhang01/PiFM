#!/usr/bin/env bash
set -euo pipefail

# Train the curly-flow PiFM schematic experiment, then evaluate the best checkpoint.
#
# Examples:
#   bash scripts/run_curly_flow_pifm.sh
#   bash scripts/run_curly_flow_pifm.sh model.use_lie_bracket=false
#   bash scripts/run_curly_flow_pifm.sh trainer.max_epochs=100 data.batch_size=256

EXTRA_OVERRIDES=("$@")
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs}"
LOGS_DIR="${LOGS_DIR%/}"
RUN_DIR="${LOGS_DIR}/curly_flow_pifm/runs/${RUN_ID}"
CKPT_PATH="${RUN_DIR}/checkpoints/best.ckpt"

COMMON_OVERRIDES=(
  "experiment=curly_flow_pifm"
  "trainer=default.yaml"
  "hydra.run.dir=${RUN_DIR}"
)

echo "Training Curly Flow PiFM:"

python src/train.py \
  "${COMMON_OVERRIDES[@]}" \
  "${EXTRA_OVERRIDES[@]}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Expected checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

echo "Evaluating checkpoint: ${CKPT_PATH}"
python src/eval.py \
  "experiment=curly_flow_pifm" \
  "trainer=default.yaml" \
  "ckpt_path=${CKPT_PATH}" \
  "hydra.run.dir=${RUN_DIR}/eval" \
  "${EXTRA_OVERRIDES[@]}"

echo "Done. Run directory: ${RUN_DIR}"
