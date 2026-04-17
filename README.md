# EmbodiedDataTransfer

EmbodiedDataTransfer is a small workflow project for:

- downloading and exporting LeRobot-style datasets episode by episode
- running NVIDIA Cosmos `edge/distilled` generation on exported robot videos
- generating multiple trajectory variants per episode with different seeds
- appending generated variants back into a local augmented dataset
- optionally uploading the augmented dataset to Hugging Face

The repository now exposes two ways to use the workflow:

- high-level shell scripts in [scripts/README.md](/file_system/liujincheng/Projects/EmbodiedDataTransfer/scripts/README.md)
- lower-level Python CLI commands in [cli.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/cli.py)

## Quick Start

If you want one command that runs the whole workflow on the full dataset, use:

```bash
DATA_PARALLEL=true NUM_TRAJECTORIES=4 GPU_IDS=0,1,2,3,4,5,6,7 UPLOAD=true HF_TOKEN=hf_xxx HF_REPO=Miical/so101-30episodes-augmented ./scripts/full_pipeline_episode.sh
```

That command will:

1. download and export the dataset into `data/episode_exports`
2. run Cosmos generation for all exported episodes
3. append all generated variants into the local augmented dataset
4. upload the augmented dataset to Hugging Face

If you prefer the lower-level CLI instead of wrapper scripts, the same workflow is built from:

```bash
PYTHONPATH=src python3 -m embodied_data_transfer process ...
PYTHONPATH=src python3 -m embodied_data_transfer run ...
PYTHONPATH=src python3 -m embodied_data_transfer append ...
```

## Setup

Typical setup:

```bash
cd /file_system/liujincheng/Projects/EmbodiedDataTransfer
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If your environment already uses the project venv, the scripts in `scripts/` can be run directly.

## Project Layout

```text
EmbodiedDataTransfer/
├── prompts/
├── scripts/
├── src/
│   └── embodied_data_transfer/
│       ├── __main__.py
│       ├── cli.py
│       ├── common.py
│       ├── dataset_processing.py
│       ├── cosmos_workflow.py
│       ├── augmentation.py
│       └── dataset_workflow.py
├── tests/
├── pyproject.toml
└── README.md
```

## Code Structure

- [cli.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/cli.py): command-line entrypoints and argument parsing
- [common.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/common.py): shared path, naming, and metadata helpers
- [dataset_processing.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/dataset_processing.py): dataset download, inspection, and episode export
- [cosmos_workflow.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/cosmos_workflow.py): Cosmos spec generation, single-run execution, and data-parallel scheduling
- [augmentation.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/augmentation.py): appending generated trajectories into LeRobot datasets and optional upload
- [dataset_workflow.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/dataset_workflow.py): compatibility facade that re-exports the main workflow functions

## Script Usage

The recommended entrypoints are:

- [scripts/process_dataset.sh](/file_system/liujincheng/Projects/EmbodiedDataTransfer/scripts/process_dataset.sh)
- [scripts/run.sh](/file_system/liujincheng/Projects/EmbodiedDataTransfer/scripts/run.sh)
- [scripts/append.sh](/file_system/liujincheng/Projects/EmbodiedDataTransfer/scripts/append.sh)
- [scripts/full_pipeline_episode.sh](/file_system/liujincheng/Projects/EmbodiedDataTransfer/scripts/full_pipeline_episode.sh)

Shared defaults live in [scripts/common.sh](/file_system/liujincheng/Projects/EmbodiedDataTransfer/scripts/common.sh).

Common environment variables:

```bash
DATASET_ID=Miical/so101-30episodes
PROMPT_PATH=/file_system/liujincheng/Projects/EmbodiedDataTransfer/prompts/single_arm_scene_tuning_en.txt
HF_HOME=/file_system/liujincheng/models/cosmos_model_cache
NUM_TRAJECTORIES=4
GPU_IDS=0,1,2,3,4,5,6,7
HF_REPO=Miical/so101-30episodes-augmented
```

### `process_dataset.sh`

Downloads dataset metadata and videos, then exports one directory per episode.

```bash
./scripts/process_dataset.sh
```

Result:

- raw dataset cache under `data/hf_raw`
- exported episodes under `data/episode_exports/<dataset_name>/episode_XXX`

### `run.sh`

Runs Cosmos generation.

Default behavior:

- `EPISODE_ID=all`
- `DATA_PARALLEL=false`
- `NUM_TRAJECTORIES=1`

Run the full dataset:

```bash
NUM_TRAJECTORIES=4 ./scripts/run.sh
```

Run one episode:

```bash
EPISODE_ID=3 NUM_TRAJECTORIES=4 ./scripts/run.sh
```

Run the full dataset with data parallel scheduling across GPUs:

```bash
DATA_PARALLEL=true NUM_TRAJECTORIES=4 GPU_IDS=0,1,2,3,4,5,6,7 ./scripts/run.sh
```

What it does:

- uses `seed`, `seed + 1`, `seed + 2`, ... for different trajectory variants
- writes outputs into `cosmos_edge_distilled/variants/variant_XXX`

### `append.sh`

Appends generated variants into the local augmented dataset, with optional upload.

Default behavior:

- `EPISODE_ID=all`
- `UPLOAD=false`

Append everything locally:

```bash
./scripts/append.sh
```

Append one episode only:

```bash
EPISODE_ID=3 ./scripts/append.sh
```

Append everything and upload:

```bash
UPLOAD=true HF_TOKEN=hf_xxx HF_REPO=Miical/so101-30episodes-augmented ./scripts/append.sh
```

### `full_pipeline_episode.sh`

Runs the whole workflow:

```bash
./scripts/process_dataset.sh
EPISODE_ID=... ./scripts/run.sh
EPISODE_ID=... UPLOAD=... ./scripts/append.sh
```

Default behavior:

- if you pass no argument, `EPISODE_ID=all`
- if you pass one argument, it is treated as the episode id

Run everything for the whole dataset:

```bash
DATA_PARALLEL=true NUM_TRAJECTORIES=4 GPU_IDS=0,1,2,3,4,5,6,7 ./scripts/full_pipeline_episode.sh
```

Run everything for one episode:

```bash
DATA_PARALLEL=true NUM_TRAJECTORIES=4 GPU_IDS=0,1,2,3,4,5,6,7 ./scripts/full_pipeline_episode.sh 3
```

Run everything and upload at the end:

```bash
DATA_PARALLEL=true NUM_TRAJECTORIES=4 GPU_IDS=0,1,2,3,4,5,6,7 UPLOAD=true HF_TOKEN=hf_xxx HF_REPO=Miical/so101-30episodes-augmented ./scripts/full_pipeline_episode.sh
```

## CLI Usage

If you want to bypass the shell scripts, the Python CLI exposes these commands:

- `inspect`
- `process`
- `run`
- `append`

The CLI entrypoints live in [cli.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/cli.py), while the workflow logic is split across the modules listed above.

### `inspect`

Print rows grouped by episode from the source dataset.

```bash
PYTHONPATH=src python3 -m embodied_data_transfer inspect \
  Miical/so101-30episodes \
  --split train \
  --cache-dir data/huggingface
```

### `process`

Download the dataset and export episode directories.

```bash
PYTHONPATH=src python3 -m embodied_data_transfer process \
  Miical/so101-30episodes \
  --split train \
  --cache-dir data/huggingface \
  --raw-dir data/hf_raw \
  --export-dir data/episode_exports
```

### `run`

Unified generation entrypoint.

Important parameters:

- `--episode-id 3` runs one episode
- `--episode-id all` runs all exported episodes
- `--data-parallel` enables multi-GPU scheduling
- `--gpu-ids 0,1,2,3` selects available GPUs
- `--num-trajectories 4` generates four variants per episode

Run one episode:

```bash
PYTHONPATH=src python3 -m embodied_data_transfer run \
  Miical/so101-30episodes \
  --episode-id 3 \
  --export-dir data/episode_exports \
  --cosmos-root /root/code/cosmos-transfer2.5 \
  --cosmos-python /root/code/cosmos-transfer2.5/.venv/bin/python \
  --prompt-path /file_system/liujincheng/Projects/EmbodiedDataTransfer/prompts/single_arm_scene_tuning_en.txt \
  --hf-home /file_system/liujincheng/models/cosmos_model_cache \
  --cosmos-model edge/distilled \
  --num-steps 4 \
  --seed 1 \
  --num-trajectories 4 \
  --nproc-per-node 1 \
  --master-port 12341
```

Run the full dataset with data parallel scheduling:

```bash
PYTHONPATH=src python3 -m embodied_data_transfer run \
  Miical/so101-30episodes \
  --episode-id all \
  --export-dir data/episode_exports \
  --cosmos-root /root/code/cosmos-transfer2.5 \
  --cosmos-python /root/code/cosmos-transfer2.5/.venv/bin/python \
  --prompt-path /file_system/liujincheng/Projects/EmbodiedDataTransfer/prompts/single_arm_scene_tuning_en.txt \
  --hf-home /file_system/liujincheng/models/cosmos_model_cache \
  --cosmos-model edge/distilled \
  --num-steps 4 \
  --seed 1 \
  --num-trajectories 4 \
  --data-parallel \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --master-port 12341
```

### `append`

Unified append entrypoint.

Important parameters:

- `--episode-id 3` appends one episode
- `--episode-id all` appends all exported episodes
- `--upload` uploads after append
- `--hf-repo` selects the target Hugging Face dataset repo

Append everything locally:

```bash
PYTHONPATH=src python3 -m embodied_data_transfer append \
  Miical/so101-30episodes \
  --episode-id all \
  --export-dir data/episode_exports \
  --raw-dir data/hf_raw \
  --target-dir data/augmented_datasets \
  --cosmos-model edge/distilled
```

Append everything and upload:

```bash
PYTHONPATH=src python3 -m embodied_data_transfer append \
  Miical/so101-30episodes \
  --episode-id all \
  --export-dir data/episode_exports \
  --raw-dir data/hf_raw \
  --target-dir data/augmented_datasets \
  --cosmos-model edge/distilled \
  --upload \
  --hf-repo Miical/so101-30episodes-augmented \
  --hf-token-env HF_TOKEN
```

Append only one episode and upload:

```bash
PYTHONPATH=src python3 -m embodied_data_transfer append \
  Miical/so101-30episodes \
  --episode-id 3 \
  --export-dir data/episode_exports \
  --raw-dir data/hf_raw \
  --target-dir data/augmented_datasets \
  --cosmos-model edge/distilled \
  --upload \
  --hf-repo Miical/so101-30episodes-augmented \
  --hf-token-env HF_TOKEN
```

## Notes

- `run` and `append` both default to `episode-id=all` in the wrapper scripts.
- multi-trajectory generation uses incrementing seeds to make variants different.
- generated Cosmos outputs are stored per episode and per variant.
- if you use a SOCKS proxy with Cosmos downloads, make sure your Cosmos checkout includes the `socksio` fix in its `checkpoint_db.py`.

## Where To Look Next

- high-level script examples: [scripts/README.md](/file_system/liujincheng/Projects/EmbodiedDataTransfer/scripts/README.md)
- prompt templates: [prompts/README.md](/file_system/liujincheng/Projects/EmbodiedDataTransfer/prompts/README.md)
- CLI implementation: [cli.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/cli.py)
- dataset download and export: [dataset_processing.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/dataset_processing.py)
- Cosmos generation workflow: [cosmos_workflow.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/cosmos_workflow.py)
- dataset append and upload: [augmentation.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/augmentation.py)
- compatibility re-exports: [dataset_workflow.py](/file_system/liujincheng/Projects/EmbodiedDataTransfer/src/embodied_data_transfer/dataset_workflow.py)
