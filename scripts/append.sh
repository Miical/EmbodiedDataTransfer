#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EPISODE_ID="${EPISODE_ID:-all}"
HF_REPO="${HF_REPO:-Miical/so101-30episodes-augmented}"
HF_TOKEN_ENV="${HF_TOKEN_ENV:-HF_TOKEN}"
UPLOAD="${UPLOAD:-false}"

ARGS=(
  append
  "${DATASET_ID}"
  --episode-id "${EPISODE_ID}"
  --export-dir "${EXPORT_DIR}"
  --raw-dir "${RAW_DIR}"
  --target-dir "${TARGET_DIR}"
  --cosmos-model "${COSMOS_MODEL}"
)

if [[ "${UPLOAD}" == "true" ]]; then
  ARGS+=(--upload --hf-repo "${HF_REPO}" --hf-token-env "${HF_TOKEN_ENV}")
fi

run_edt "${ARGS[@]}"
