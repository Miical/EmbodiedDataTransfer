from __future__ import annotations

import argparse
from pathlib import Path

from embodied_data_transfer.dataset_workflow import (
    append_all_generated_episodes_and_upload,
    append_all_generated_episodes_to_dataset,
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


def parse_episode_selector(value: str) -> str | int:
    stripped = value.strip().lower()
    if stripped == "all":
        return "all"
    return int(value)


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
        target_parser.add_argument(
            "--disable-guardrails",
            dest="disable_guardrails",
            action="store_true",
            default=True,
            help="Disable Cosmos prompt/video guardrails. This is the default behavior in this workflow.",
        )
        target_parser.add_argument(
            "--enable-guardrails",
            dest="disable_guardrails",
            action="store_false",
            help="Enable Cosmos prompt/video guardrails.",
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
        help="Run Cosmos inference for one episode or all exported episodes, with optional data parallel scheduling.",
    )
    run_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    run_parser.add_argument(
        "--episode-id",
        type=parse_episode_selector,
        default="all",
        help="Episode id to run, or 'all' to run every exported episode.",
    )
    add_cosmos_args(run_parser)
    run_parser.add_argument(
        "--gpu-ids",
        type=parse_int_list,
        default=parse_int_list("0,1,2,3,4,5,6,7"),
        help="Comma-separated GPU ids used when --data-parallel is enabled.",
    )
    run_parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Schedule episodes across multiple GPUs, one single-GPU worker per episode/variant.",
    )
    run_parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between worker status polls when --data-parallel is enabled.",
    )

    append_parser = subparsers.add_parser(
        "append",
        help="Append generated episodes into a local augmented dataset, with optional upload.",
    )
    append_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    append_parser.add_argument(
        "--episode-id",
        type=parse_episode_selector,
        default="all",
        help="Episode id to append, or 'all' to append every exported episode.",
    )
    append_parser.add_argument("--export-dir", type=Path, default=Path("data/episode_exports"))
    append_parser.add_argument("--raw-dir", type=Path, default=Path("data/hf_raw"))
    append_parser.add_argument("--target-dir", type=Path, default=Path("data/augmented_datasets"))
    append_parser.add_argument("--cosmos-model", default="edge/distilled")
    append_parser.add_argument("--upload", action="store_true", help="Upload to Hugging Face after append.")
    append_parser.add_argument("--hf-repo", default=None)
    append_parser.add_argument("--hf-token-env", default="HF_TOKEN")

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
            disable_guardrails=args.disable_guardrails,
            hf_home=args.hf_home,
            cosmos_experimental_checkpoints=not args.disable_experimental_checkpoints,
            nproc_per_node=args.nproc_per_node,
            master_port=args.master_port,
        )
        print(f"Cosmos outputs saved under: {run_dir}")
        return

    if command == "run":
        episode_selector = args.episode_id
        if args.data_parallel:
            if episode_selector == "all":
                episode_ids = None
            else:
                episode_ids = [episode_selector]
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
                disable_guardrails=args.disable_guardrails,
                hf_home=args.hf_home,
                cosmos_experimental_checkpoints=not args.disable_experimental_checkpoints,
                master_port_start=args.master_port,
                episode_ids=episode_ids,
                poll_interval_seconds=args.poll_interval,
            )
            print(f"Completed {len(run_dirs)} episode runs.")
            return

        if episode_selector == "all":
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
                disable_guardrails=args.disable_guardrails,
                hf_home=args.hf_home,
                cosmos_experimental_checkpoints=not args.disable_experimental_checkpoints,
                nproc_per_node=args.nproc_per_node,
                master_port=args.master_port,
            )
            print(f"Completed {len(run_dirs)} episode runs.")
            return

        run_dir = run_cosmos_depth_inference_for_episode(
            dataset_id=args.dataset,
            episode_id=episode_selector,
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

    if command == "append":
        hf_repo = args.hf_repo or augmented_repo_id(args.dataset)
        episode_selector = args.episode_id
        if episode_selector == "all":
            if args.upload:
                target_dir, commit_url = append_all_generated_episodes_and_upload(
                    dataset_id=args.dataset,
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
            else:
                target_dir = append_all_generated_episodes_to_dataset(
                    dataset_id=args.dataset,
                    export_dir=args.export_dir,
                    raw_dir=args.raw_dir,
                    augmented_root=args.target_dir,
                    cosmos_model=args.cosmos_model,
                )
                print(f"Augmented dataset saved under: {target_dir}")
            return

        if args.upload:
            target_dir, commit_url = append_generated_episode_and_upload(
                dataset_id=args.dataset,
                episode_id=episode_selector,
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
        else:
            target_dir = append_generated_episode_to_dataset(
                dataset_id=args.dataset,
                episode_id=episode_selector,
                export_dir=args.export_dir,
                raw_dir=args.raw_dir,
                augmented_root=args.target_dir,
                cosmos_model=args.cosmos_model,
            )
            print(f"Augmented dataset saved under: {target_dir}")
        return

    raise ValueError(f"Unsupported command: {command}")
