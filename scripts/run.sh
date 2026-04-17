#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EPISODE_ID="${EPISODE_ID:-all}"
DATA_PARALLEL="${DATA_PARALLEL:-false}"

ARGS=(
  run
  "${DATASET_ID}"
  --episode-id "${EPISODE_ID}"
  --export-dir "${EXPORT_DIR}"
  --cosmos-root "${COSMOS_ROOT}"
  --cosmos-python "${COSMOS_PYTHON}"
  --prompt-path "${PROMPT_PATH}"
  --hf-home "${HF_HOME}"
  --cosmos-model "${COSMOS_MODEL}"
  --guidance "${GUIDANCE}"
  --num-steps "${NUM_STEPS}"
  --seed "${SEED}"
  --num-trajectories "${NUM_TRAJECTORIES}"
  --master-port "${MASTER_PORT}"
)

if [[ "${DATA_PARALLEL}" == "true" ]]; then
  ARGS+=(--data-parallel --gpu-ids "${GPU_IDS}" --poll-interval "${POLL_INTERVAL:-2.0}")
else
  ARGS+=(--nproc-per-node "${NPROC_PER_NODE}")
fi

if [[ "${DISABLE_GUARDRAILS}" == "true" ]]; then
  ARGS+=(--disable-guardrails)
fi

run_edt "${ARGS[@]}"
