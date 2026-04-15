from __future__ import annotations

import argparse
from pathlib import Path

from embodied_data_transfer.dataset_workflow import (
    inspect_dataset,
    process_dataset,
    run_cosmos_depth_inference_for_episode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and process Hugging Face / LeRobot datasets.")
    subparsers = parser.add_subparsers(dest="command")

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
        help="Run Cosmos depth inference for every video in a processed episode directory.",
    )
    infer_parser.add_argument("dataset", nargs="?", default="Miical/record-test-2")
    infer_parser.add_argument("--episode-id", type=int, required=True)
    infer_parser.add_argument("--export-dir", type=Path, default=Path("data/episode_exports"))
    infer_parser.add_argument("--cosmos-root", type=Path, default=Path("/workspace/cosmos-transfer2.5"))
    infer_parser.add_argument(
        "--cosmos-python",
        type=Path,
        default=Path("/workspace/cosmos-transfer2.5/.venv/bin/python"),
    )
    infer_parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("/workspace/cosmos-transfer2.5/assets/robot_example/robot_prompt.txt"),
    )
    infer_parser.add_argument("--guidance", type=int, default=3)

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
        )
        print(f"Cosmos outputs saved under: {run_dir}")
        return

    raise ValueError(f"Unsupported command: {command}")
