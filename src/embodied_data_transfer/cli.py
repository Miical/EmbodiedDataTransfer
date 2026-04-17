from __future__ import annotations

import argparse
from pathlib import Path

from embodied_data_transfer.dataset_workflow import (
    append_generated_episode_to_dataset,
    append_generated_episode_and_upload,
    augmented_repo_id,
    inspect_dataset,
    process_dataset,
    run_cosmos_depth_inference_for_all_episodes,
    run_cosmos_depth_inference_for_episode,
    run_cosmos_depth_inference_parallel_single_gpu,
)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and process Hugging Face / LeRobot datasets.")
    subparsers = parser.add_subparsers(dest="command")

    def add_cosmos_args(target_parser: argparse.ArgumentParser) -> None:
        target_parser.add_argument("--export-dir", type=Path, default=Path("data/episode_exports"))
        target_parser.add_argument("--cosmos-root", type=Path, default=Path("/root/code/cosmos-transfer2.5"))
        target_parser.add_argument(
            "--cosmos-python",
            type=Path,
            default=Path("/root/code/cosmos-transfer2.5/.venv/bin/python"),
        )
        target_parser.add_argument(
            "--prompt-path",
            type=Path,
            default=Path("/root/code/cosmos-transfer2.5/assets/robot_example/robot_prompt.txt"),
        )
        target_parser.add_argument("--guidance", type=int, default=3)
        target_parser.add_argument("--cosmos-model", default="edge/distilled")
        target_parser.add_argument(
            "--num-steps",
            type=int,
            default=None,
            help="Sampling steps written into each Cosmos spec. Defaults to 4 for distilled models and 35 otherwise.",
        )
        target_parser.add_argument("--seed", type=int, default=1)
        target_parser.add_argument(
            "--num-trajectories",
            type=int,
            default=1,
            help="How many trajectory variants to generate per source episode. Each variant uses seed+index.",
        )
        target_parser.add_argument(
            "--hf-home",
            type=Path,
            default=Path("/file_system/liujincheng/models/cosmos_model_cache"),
        )
        target_parser.add_argument("--nproc-per-node", type=int, default=8)
        target_parser.add_argument("--master-port", type=int, default=12341)
        target_parser.add_argument(
            "--disable-experimental-checkpoints",
            action="store_true",
            help="Do not set COSMOS_EXPERIMENTAL_CHECKPOINTS=1 for the inference subprocess.",
        )

    inspect_parser = subparsers.add_parser("inspect", help="Print dataset rows grouped by episode.")
    inspect_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    inspect_parser.add_argument("--split", default="train")
    inspect_parser.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))

    process_parser = subparsers.add_parser(
        "process",
        help="Download dataset assets and export one directory per episode.",
    )
    process_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    process_parser.add_argument("--split", default="train")
    process_parser.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))
    process_parser.add_argument("--raw-dir", type=Path, default=Path("data/hf_raw"))
    process_parser.add_argument("--export-dir", type=Path, default=Path("data/episode_exports"))

    infer_parser = subparsers.add_parser(
        "infer-episode",
        help="Run Cosmos inference for every video in a processed episode directory.",
    )
    infer_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    infer_parser.add_argument("--episode-id", type=int, required=True)
    add_cosmos_args(infer_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="One-command entrypoint for running Cosmos inference on a prepared episode.",
    )
    run_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    run_parser.add_argument("--episode-id", type=int, required=True)
    add_cosmos_args(run_parser)

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run Cosmos inference for every prepared episode in the dataset export directory.",
    )
    run_all_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    add_cosmos_args(run_all_parser)

    run_parallel_parser = subparsers.add_parser(
        "run-parallel",
        help="Schedule multiple prepared episodes across multiple GPUs, one single-GPU Cosmos worker per episode.",
    )
    run_parallel_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    add_cosmos_args(run_parallel_parser)
    run_parallel_parser.add_argument(
        "--gpu-ids",
        type=parse_int_list,
        required=True,
        help="Comma-separated GPU ids, for example: 0,1,2,3",
    )
    run_parallel_parser.add_argument(
        "--episode-ids",
        type=parse_int_list,
        default=None,
        help="Optional comma-separated episode ids to run. Defaults to all exported episodes.",
    )
    run_parallel_parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between worker status polls.",
    )

    append_parser = subparsers.add_parser(
        "append-episode",
        help="Append one generated episode into a new local augmented LeRobot-style dataset.",
    )
    append_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    append_parser.add_argument("--episode-id", type=int, required=True)
    append_parser.add_argument("--export-dir", type=Path, default=Path("data/episode_exports"))
    append_parser.add_argument("--raw-dir", type=Path, default=Path("data/hf_raw"))
    append_parser.add_argument("--target-dir", type=Path, default=Path("data/augmented_datasets"))
    append_parser.add_argument("--cosmos-model", default="edge/distilled")

    append_upload_parser = subparsers.add_parser(
        "append-and-upload",
        help="Append one generated episode into a new local augmented dataset and upload it to Hugging Face.",
    )
    append_upload_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    append_upload_parser.add_argument("--episode-id", type=int, required=True)
    append_upload_parser.add_argument("--export-dir", type=Path, default=Path("data/episode_exports"))
    append_upload_parser.add_argument("--raw-dir", type=Path, default=Path("data/hf_raw"))
    append_upload_parser.add_argument("--target-dir", type=Path, default=Path("data/augmented_datasets"))
    append_upload_parser.add_argument("--hf-repo", default=None)
    append_upload_parser.add_argument("--hf-token-env", default="HF_TOKEN")
    append_upload_parser.add_argument("--cosmos-model", default="edge/distilled")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    command = args.command or "inspect"
    if command == "inspect":
        inspect_dataset(
            dataset_id=args.dataset,
            split=args.split,
            cache_dir=args.cache_dir,
        )
        return

    if command == "process":
        process_dataset(
            dataset_id=args.dataset,
            split=args.split,
            cache_dir=args.cache_dir,
            raw_dir=args.raw_dir,
            export_dir=args.export_dir,
        )
        return

    if command == "infer-episode":
        run_dir = run_cosmos_depth_inference_for_episode(
            dataset_id=args.dataset,
            episode_id=args.episode_id,
            export_dir=args.export_dir,
            cosmos_root=args.cosmos_root,
            cosmos_python=args.cosmos_python,
            cosmos_prompt_path=args.prompt_path,
            guidance=args.guidance,
            cosmos_model=args.cosmos_model,
            num_steps=args.num_steps,
            seed=args.seed,
            num_trajectories=args.num_trajectories,
            hf_home=args.hf_home,
            cosmos_experimental_checkpoints=not args.disable_experimental_checkpoints,
            nproc_per_node=args.nproc_per_node,
            master_port=args.master_port,
        )
        print(f"Cosmos outputs saved under: {run_dir}")
        return

    if command == "run":
        run_dir = run_cosmos_depth_inference_for_episode(
            dataset_id=args.dataset,
            episode_id=args.episode_id,
            export_dir=args.export_dir,
            cosmos_root=args.cosmos_root,
            cosmos_python=args.cosmos_python,
            cosmos_prompt_path=args.prompt_path,
            guidance=args.guidance,
            cosmos_model=args.cosmos_model,
            num_steps=args.num_steps,
            seed=args.seed,
            num_trajectories=args.num_trajectories,
            hf_home=args.hf_home,
            cosmos_experimental_checkpoints=not args.disable_experimental_checkpoints,
            nproc_per_node=args.nproc_per_node,
            master_port=args.master_port,
        )
        print(f"Cosmos outputs saved under: {run_dir}")
        return

    if command == "run-all":
        run_dirs = run_cosmos_depth_inference_for_all_episodes(
            dataset_id=args.dataset,
            export_dir=args.export_dir,
            cosmos_root=args.cosmos_root,
            cosmos_python=args.cosmos_python,
            cosmos_prompt_path=args.prompt_path,
            guidance=args.guidance,
            cosmos_model=args.cosmos_model,
            num_steps=args.num_steps,
            seed=args.seed,
            num_trajectories=args.num_trajectories,
            hf_home=args.hf_home,
            cosmos_experimental_checkpoints=not args.disable_experimental_checkpoints,
            nproc_per_node=args.nproc_per_node,
            master_port=args.master_port,
        )
        print(f"Completed {len(run_dirs)} episode runs.")
        return

    if command == "run-parallel":
        run_dirs = run_cosmos_depth_inference_parallel_single_gpu(
            dataset_id=args.dataset,
            export_dir=args.export_dir,
            cosmos_root=args.cosmos_root,
            cosmos_python=args.cosmos_python,
            cosmos_prompt_path=args.prompt_path,
            gpu_ids=args.gpu_ids,
            guidance=args.guidance,
            cosmos_model=args.cosmos_model,
            num_steps=args.num_steps,
            seed=args.seed,
            num_trajectories=args.num_trajectories,
            hf_home=args.hf_home,
            cosmos_experimental_checkpoints=not args.disable_experimental_checkpoints,
            master_port_start=args.master_port,
            episode_ids=args.episode_ids,
            poll_interval_seconds=args.poll_interval,
        )
        print(f"Completed {len(run_dirs)} episode runs.")
        return

    if command == "append-episode":
        target_dir = append_generated_episode_to_dataset(
            dataset_id=args.dataset,
            episode_id=args.episode_id,
            export_dir=args.export_dir,
            raw_dir=args.raw_dir,
            augmented_root=args.target_dir,
            cosmos_model=args.cosmos_model,
        )
        print(f"Augmented dataset saved under: {target_dir}")
        return

    if command == "append-and-upload":
        hf_repo = args.hf_repo or augmented_repo_id(args.dataset)
        target_dir, commit_url = append_generated_episode_and_upload(
            dataset_id=args.dataset,
            episode_id=args.episode_id,
            export_dir=args.export_dir,
            raw_dir=args.raw_dir,
            augmented_root=args.target_dir,
            hf_repo_id=hf_repo,
            hf_token_env_var=args.hf_token_env,
            cosmos_model=args.cosmos_model,
        )
        print(f"Augmented dataset saved under: {target_dir}")
        print(f"Uploaded to Hugging Face dataset repo: {hf_repo}")
        print(f"Upload result: {commit_url}")
        return

    raise ValueError(f"Unsupported command: {command}")
