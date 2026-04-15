from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face / LeRobot dataset and print its contents grouped by episode."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="Miical/record-test-2",
        help="Hugging Face dataset id, for example 'Miical/record-test-2'.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/huggingface"),
        help="Local cache directory for downloaded dataset artifacts.",
    )
    return parser


def _to_pretty_json(sample: dict[str, Any]) -> str:
    return json.dumps(sample, ensure_ascii=False, indent=2, default=str)


def _serialize(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _print_lerobot_dataset(dataset: Any, dataset_id: str, split: str) -> None:
    episodes = dataset.meta.episodes
    from_indices = list(episodes["dataset_from_index"])
    to_indices = list(episodes["dataset_to_index"])

    print(f"Loaded dataset via LeRobotDataset: {dataset_id}")
    print(f"Split: {split}")
    print(f"Total frames: {len(dataset)}")
    print(f"Total episodes: {len(from_indices)}")
    print(f"Features: {list(dataset.features.keys())}")

    for episode_index, (from_idx, to_idx) in enumerate(zip(from_indices, to_indices, strict=True)):
        print("=" * 80)
        print(f"Episode {episode_index}")
        print(f"Frame range: [{from_idx}, {to_idx})")
        print(f"Number of frames: {to_idx - from_idx}")
        print("-" * 80)

        for frame_index in range(from_idx, to_idx):
            sample = _serialize(dict(dataset[frame_index]))
            print(f"Frame {frame_index}")
            print(_to_pretty_json(sample))


def _print_hf_dataset(dataset: Any, dataset_id: str, split: str) -> None:
    print(f"Loaded dataset via datasets.load_dataset: {dataset_id}")
    print(f"Split: {split}")
    print(f"Number of rows: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")

    if "episode_index" not in dataset.features:
        print("This dataset does not expose an 'episode_index' column, so rows are printed without episode grouping.")
        for row_index, sample in enumerate(dataset):
            print("=" * 80)
            print(f"Row {row_index}")
            print(_to_pretty_json(_serialize(sample)))
        return

    episodes: dict[int, list[dict[str, Any]]] = {}
    for sample in dataset:
        episode_index = int(sample["episode_index"])
        episodes.setdefault(episode_index, []).append(sample)

    for episode_index in sorted(episodes):
        print("=" * 80)
        print(f"Episode {episode_index}")
        print("-" * 80)
        episode_rows = episodes[episode_index]
        print(f"Number of rows: {len(episode_rows)}")

        for row_index, sample in enumerate(episode_rows):
            print(f"Row {row_index}")
            print(_to_pretty_json(_serialize(sample)))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(args.dataset, root=args.cache_dir)
        _print_lerobot_dataset(dataset, args.dataset, args.split)
        return
    except Exception as exc:
        print(f"LeRobotDataset load failed, falling back to datasets.load_dataset: {exc}")

    dataset = load_dataset(args.dataset, split=args.split, cache_dir=str(args.cache_dir))
    _print_hf_dataset(dataset, args.dataset, args.split)
