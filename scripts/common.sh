#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_ID="${DATASET_ID:-Miical/so101-30episodes}"
SPLIT="${SPLIT:-train}"

EXPORT_DIR="${EXPORT_DIR:-data/episode_exports}"
RAW_DIR="${RAW_DIR:-data/hf_raw}"
CACHE_DIR="${CACHE_DIR:-data/huggingface}"
TARGET_DIR="${TARGET_DIR:-data/augmented_datasets}"

COSMOS_ROOT="${COSMOS_ROOT:-/root/code/cosmos-transfer2.5}"
COSMOS_PYTHON="${COSMOS_PYTHON:-/root/code/cosmos-transfer2.5/.venv/bin/python}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_ROOT}/prompts/single_arm_scene_tuning_en.txt}"
HF_HOME="${HF_HOME:-/file_system/liujincheng/models/cosmos_model_cache}"

COSMOS_MODEL="${COSMOS_MODEL:-edge/distilled}"
GUIDANCE="${GUIDANCE:-3}"
NUM_STEPS="${NUM_STEPS:-4}"
SEED="${SEED:-1}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-12341}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DISABLE_GUARDRAILS="${DISABLE_GUARDRAILS:-true}"

run_edt() {
  PYTHONPATH=src "${PYTHON_BIN}" -m embodied_data_transfer "$@"
}
