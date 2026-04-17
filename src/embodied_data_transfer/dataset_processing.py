from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from embodied_data_transfer.common import dataset_dir_name, to_pretty_json, to_serializable


def list_dataset_repo_paths(dataset_id: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{dataset_id}/tree/main?recursive=true"
    with urllib.request.urlopen(url) as response:
        payload = json.load(response)
    return [item.get("path", "") for item in payload]


def download_dataset_file(dataset_id: str, filename: str, local_dir: Path) -> Path:
    local_path = hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_dir),
    )
    return Path(local_path)


def load_tabular_dataset(dataset_id: str, split: str, cache_dir: Path):
    return load_dataset(dataset_id, split=split, cache_dir=str(cache_dir))


def group_rows_by_episode(dataset: Any) -> dict[int, list[dict[str, Any]]]:
    episodes: dict[int, list[dict[str, Any]]] = {}
    for sample in dataset:
        episode_index = int(sample["episode_index"])
        episodes.setdefault(episode_index, []).append(to_serializable(sample))
    return episodes


def read_dataset_info(dataset_id: str, raw_dir: Path) -> dict[str, Any]:
    info_path = download_dataset_file(dataset_id, "meta/info.json", raw_dir)
    return json.loads(info_path.read_text(encoding="utf-8"))


def download_root_repo_files(dataset_id: str, local_dir: Path) -> None:
    for filename in ["README.md", ".gitattributes"]:
        try:
            download_dataset_file(dataset_id, filename, local_dir)
        except Exception:
            pass


def download_episode_metadata(dataset_id: str, raw_dir: Path, repo_paths: list[str]) -> list[Path]:
    meta_paths = sorted(
        path for path in repo_paths if path.startswith("meta/episodes/") and path.endswith(".parquet")
    )
    return [download_dataset_file(dataset_id, path, raw_dir) for path in meta_paths]


def load_episode_metadata(metadata_files: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(path) for path in metadata_files], ignore_index=True).sort_values(
        "episode_index"
    )


def discover_video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return [key for key, spec in features.items() if spec.get("dtype") == "video"]


def download_video_assets(dataset_id: str, raw_dir: Path, repo_paths: list[str]) -> list[Path]:
    video_paths = sorted(path for path in repo_paths if path.startswith("videos/") and path.endswith(".mp4"))
    return [download_dataset_file(dataset_id, path, raw_dir) for path in video_paths]


def export_episode_directory(
    episode: dict[str, Any],
    frames_by_episode: dict[int, list[dict[str, Any]]],
    video_keys: list[str],
    info: dict[str, Any],
    raw_dir: Path,
    export_dir: Path,
) -> None:
    episode_index = int(episode["episode_index"])
    episode_dir = export_dir / f"episode_{episode_index:03d}"
    if episode_dir.exists():
        import shutil

        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)

    episode_meta = {key: to_serializable(value) for key, value in episode.items()}
    (episode_dir / "episode_meta.json").write_text(
        json.dumps(episode_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (episode_dir / "frames.json").write_text(
        json.dumps(frames_by_episode.get(episode_index, []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    video_template = info["video_path"]
    for video_key in video_keys:
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        start_key = f"videos/{video_key}/from_timestamp"
        end_key = f"videos/{video_key}/to_timestamp"
        if not all(key in episode for key in (chunk_key, file_key, start_key, end_key)):
            continue

        src_rel = video_template.format(
            video_key=video_key,
            chunk_index=int(episode[chunk_key]),
            file_index=int(episode[file_key]),
        )
        src = raw_dir / src_rel
        dst = episode_dir / f"{video_key}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{float(episode[start_key]):.6f}",
            "-to",
            f"{float(episode[end_key]):.6f}",
            "-i",
            str(src),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def inspect_dataset(dataset_id: str, split: str, cache_dir: Path) -> None:
    dataset = load_tabular_dataset(dataset_id, split, cache_dir)
    print(f"Loaded dataset: {dataset_id}")
    print(f"Split: {split}")
    print(f"Number of rows: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")

    if "episode_index" not in dataset.features:
        print("This dataset does not expose an 'episode_index' column, so rows are printed without episode grouping.")
        for row_index, sample in enumerate(dataset):
            print("=" * 80)
            print(f"Row {row_index}")
            print(to_pretty_json(to_serializable(sample)))
        return

    rows_by_episode = group_rows_by_episode(dataset)
    for episode_index in sorted(rows_by_episode):
        print("=" * 80)
        print(f"Episode {episode_index}")
        print("-" * 80)
        print(f"Number of rows: {len(rows_by_episode[episode_index])}")
        for row_index, sample in enumerate(rows_by_episode[episode_index]):
            print(f"Row {row_index}")
            print(to_pretty_json(sample))


def process_dataset(
    dataset_id: str,
    split: str,
    cache_dir: Path,
    raw_dir: Path,
    export_dir: Path,
) -> Path:
    repo_paths = list_dataset_repo_paths(dataset_id)
    dataset = load_tabular_dataset(dataset_id, split, cache_dir)
    frames_by_episode = group_rows_by_episode(dataset)

    info = read_dataset_info(dataset_id, raw_dir)
    metadata_files = download_episode_metadata(dataset_id, raw_dir, repo_paths)
    _ = download_video_assets(dataset_id, raw_dir, repo_paths)

    meta_df = load_episode_metadata(metadata_files).reset_index(drop=True)
    video_keys = discover_video_keys(info)

    dataset_export_dir = export_dir / dataset_dir_name(dataset_id)
    dataset_export_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": dataset_id,
        "split": split,
        "total_rows": len(dataset),
        "total_episodes": len(meta_df),
        "video_keys": video_keys,
        "export_dir": str(dataset_export_dir),
    }
    (dataset_export_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for episode in meta_df.to_dict(orient="records"):
        export_episode_directory(
            episode=episode,
            frames_by_episode=frames_by_episode,
            video_keys=video_keys,
            info=info,
            raw_dir=raw_dir,
            export_dir=dataset_export_dir,
        )

    print(f"Processed dataset: {dataset_id}")
    print(f"Split: {split}")
    print(f"Total rows: {len(dataset)}")
    print(f"Total episodes: {len(meta_df)}")
    print(f"Video keys: {video_keys}")
    print(f"Export dir: {dataset_export_dir}")
    for episode_index in sorted(frames_by_episode):
        print(f"episode_{episode_index:03d}: {len(frames_by_episode[episode_index])} frames")

    return dataset_export_dir


def find_episode_dir(export_dir: Path, dataset_id: str, episode_id: int) -> Path:
    dataset_export_dir = export_dir / dataset_dir_name(dataset_id)
    episode_dir = dataset_export_dir / f"episode_{episode_id:03d}"
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    return episode_dir


def list_available_episode_ids(export_dir: Path, dataset_id: str) -> list[int]:
    dataset_export_dir = export_dir / dataset_dir_name(dataset_id)
    if not dataset_export_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {dataset_export_dir}")

    episode_ids: list[int] = []
    for episode_dir in sorted(dataset_export_dir.glob("episode_*")):
        try:
            episode_ids.append(int(episode_dir.name.split("_")[-1]))
        except ValueError:
            continue
    return episode_ids
