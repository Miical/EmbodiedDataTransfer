# Scripts

These scripts wrap the main `embodied_data_transfer` commands we have been using.

## Defaults

All scripts read environment variables from `scripts/common.sh`. You can override them inline.

Common overrides:

```bash
DATASET_ID=Miical/so101-30episodes
PROMPT_PATH=/file_system/liujincheng/Projects/EmbodiedDataTransfer/prompts/single_arm_scene_tuning_en.txt
HF_HOME=/file_system/liujincheng/models/cosmos_model_cache
NUM_TRAJECTORIES=4
GPU_IDS=0,1,2,3,4,5,6,7
```

## Scripts

Process and export the dataset:

```bash
./scripts/process_dataset.sh
```

Run one episode through Cosmos:

```bash
EPISODE_ID=0 NUM_TRAJECTORIES=4 ./scripts/run.sh
```

Run the full exported dataset:

```bash
NUM_TRAJECTORIES=4 ./scripts/run.sh
```

Run the full exported dataset in data-parallel mode across GPUs:

```bash
DATA_PARALLEL=true NUM_TRAJECTORIES=4 GPU_IDS=0,1,2,3,4,5,6,7 ./scripts/run.sh
```

Run one selected episode in data-parallel mode:

```bash
EPISODE_ID=3 DATA_PARALLEL=true NUM_TRAJECTORIES=4 GPU_IDS=0,1,2,3,4,5,6,7 ./scripts/run.sh
```

Append one episode's generated variants into the local augmented dataset:

```bash
EPISODE_ID=0 ./scripts/append.sh
```

Append one episode and upload:

```bash
EPISODE_ID=0 UPLOAD=true HF_TOKEN=hf_xxx HF_REPO=Miical/so101-30episodes-augmented ./scripts/append.sh
```

Append all generated episodes locally:

```bash
./scripts/append.sh
```

Append all generated episodes and upload once:

```bash
UPLOAD=true HF_TOKEN=hf_xxx HF_REPO=Miical/so101-30episodes-augmented ./scripts/append.sh
```

Run process -> generate -> append for the full dataset:

```bash
NUM_TRAJECTORIES=4 ./scripts/full_pipeline_episode.sh
```

Run process -> generate -> append for one episode:

```bash
NUM_TRAJECTORIES=4 ./scripts/full_pipeline_episode.sh 0
```
