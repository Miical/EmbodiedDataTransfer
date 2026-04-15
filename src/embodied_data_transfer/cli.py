from __future__ import annotations

import argparse
from pathlib import Path

from embodied_data_transfer.dataset_workflow import inspect_dataset, process_dataset


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

    raise ValueError(f"Unsupported command: {command}")
