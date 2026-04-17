#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_edt process \
  "${DATASET_ID}" \
  --split "${SPLIT}" \
  --cache-dir "${CACHE_DIR}" \
  --raw-dir "${RAW_DIR}" \
  --export-dir "${EXPORT_DIR}"
