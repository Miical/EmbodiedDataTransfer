#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EPISODE_ID="${1:-all}"

"${SCRIPT_DIR}/process_dataset.sh"
EPISODE_ID="${EPISODE_ID}" "${SCRIPT_DIR}/run.sh"
EPISODE_ID="${EPISODE_ID}" UPLOAD="${UPLOAD:-false}" "${SCRIPT_DIR}/append.sh"
