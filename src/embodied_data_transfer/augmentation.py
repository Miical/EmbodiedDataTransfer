from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from huggingface_hub import snapshot_download
from lerobot.datasets import LeRobotDataset

from embodied_data_transfer.common import (
    augmented_dataset_dir_name,
    load_json_file,
    write_json_file,
)
from embodied_data_transfer.cosmos_workflow import cosmos_run_dir_name
from embodied_data_transfer.dataset_processing import (
    download_root_repo_files,
    find_episode_dir,
    list_available_episode_ids,
)


def has_complete_dataset_snapshot(root: Path) -> bool:
    return (
        (root / "meta" / "info.json").exists()
        and (root / "meta" / "stats.json").exists()
        and (root / "meta" / "tasks.parquet").exists()
        and any((root / "data").rglob("file-*.parquet"))
        and any((root / "videos").rglob("file-*.mp4"))
    )


def initialize_augmented_dataset(raw_dir: Path, dataset_id: str, augmented_root: Path) -> Path:
    target_dir = augmented_root / augmented_dataset_dir_name(dataset_id)
    if target_dir.exists() and has_complete_dataset_snapshot(target_dir):
        return target_dir

    if target_dir.exists():
        shutil.rmtree(target_dir)

    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=[".cache/*"],
    )
    download_root_repo_files(dataset_id, target_dir)
    return target_dir


def next_file_index(pattern: str, root: Path) -> int:
    files = sorted(root.glob(pattern))
    if not files:
        return 0
    return max(int(path.stem.split("-")[-1]) for path in files) + 1


def directory_size_mb(path: Path) -> int:
    total_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return int((total_bytes + 1024 * 1024 - 1) // (1024 * 1024))


def ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def load_video_frames(path: Path) -> list[Any]:
    return [frame for frame in iio.imiter(path)]


def list_generated_variant_dirs(source_episode_dir: Path, cosmos_model: str) -> list[Path]:
    variants_root = source_episode_dir / cosmos_run_dir_name(cosmos_model) / "variants"
    if not variants_root.exists():
        raise FileNotFoundError(f"Generated variants directory not found: {variants_root}")
    variant_dirs = sorted(path for path in variants_root.glob("variant_*") if path.is_dir())
    if not variant_dirs:
        raise FileNotFoundError(f"No generated variants found in {variants_root}")
    return variant_dirs


def append_generated_episode_to_dataset(
    *,
    dataset_id: str,
    episode_id: int,
    export_dir: Path,
    raw_dir: Path,
    augmented_root: Path,
    cosmos_model: str = "edge/distilled",
) -> Path:
    source_episode_dir = find_episode_dir(export_dir=export_dir, dataset_id=dataset_id, episode_id=episode_id)
    variant_dirs = list_generated_variant_dirs(source_episode_dir=source_episode_dir, cosmos_model=cosmos_model)

    target_dir = initialize_augmented_dataset(raw_dir=raw_dir, dataset_id=dataset_id, augmented_root=augmented_root)
    manifest_path = target_dir / "meta" / "augmentation_manifest.json"
    manifest = load_json_file(manifest_path) if manifest_path.exists() else {"source_dataset": dataset_id, "appended": []}

    frames = load_json_file(source_episode_dir / "frames.json")
    source_episode_meta = load_json_file(source_episode_dir / "episode_meta.json")
    task = source_episode_meta["tasks"][0]

    dataset = LeRobotDataset.resume(dataset_id, root=target_dir)
    video_keys = list(dataset.meta.video_keys)
    appended_entries: list[dict[str, Any]] = []

    for variant_dir in variant_dirs:
        run_meta_path = variant_dir / "run_meta.json"
        run_meta = load_json_file(run_meta_path) if run_meta_path.exists() else {}
        variant_index = int(run_meta.get("variant_index", variant_dir.name.split("_")[-1]))
        if any(
            item["source_episode_id"] == episode_id and item["variant_index"] == variant_index
            for item in manifest["appended"]
        ):
            raise ValueError(
                f"Episode {episode_id} variant {variant_index} has already been appended to {target_dir}"
            )

        generated_dir = variant_dir / "generated"
        if not generated_dir.exists():
            raise FileNotFoundError(f"Generated video directory not found: {generated_dir}")

        new_episode_index = dataset.num_episodes
        video_frames: dict[str, list[Any]] = {}
        for video_key in video_keys:
            source_video = generated_dir / f"{video_key}_generated.mp4"
            if not source_video.exists():
                raise FileNotFoundError(f"Generated video missing for {video_key}: {source_video}")
            video_frames[video_key] = load_video_frames(source_video)
            if len(video_frames[video_key]) != len(frames):
                raise ValueError(
                    f"Frame count mismatch for {video_key}: "
                    f"{len(video_frames[video_key])} video frames vs {len(frames)} metadata frames"
                )

        for frame_index, frame in enumerate(frames):
            frame_payload = {
                "action": np.asarray(frame["action"], dtype=np.float32),
                "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
                "task": task,
            }
            for video_key in video_keys:
                frame_payload[video_key] = video_frames[video_key][frame_index]
            dataset.add_frame(frame_payload)

        dataset.save_episode()
        appended_entries.append(
            {
                "source_episode_id": episode_id,
                "variant_index": variant_index,
                "seed": run_meta.get("seed"),
                "new_episode_index": new_episode_index,
            }
        )
        print(
            f"Appended source episode {episode_id} variant {variant_index} "
            f"as episode {new_episode_index}"
        )

    dataset.finalize()

    manifest["appended"].extend(appended_entries)
    write_json_file(manifest_path, manifest)

    print(f"Target dataset: {target_dir}")
    print(f"Frames written per variant: {len(frames)}")
    print(f"Variants appended: {len(appended_entries)}")
    return target_dir


def append_all_generated_episodes_to_dataset(
    *,
    dataset_id: str,
    export_dir: Path,
    raw_dir: Path,
    augmented_root: Path,
    cosmos_model: str = "edge/distilled",
    episode_ids: list[int] | None = None,
) -> Path:
    selected_episode_ids = episode_ids or list_available_episode_ids(export_dir=export_dir, dataset_id=dataset_id)
    if not selected_episode_ids:
        raise ValueError(f"No episode directories found for dataset {dataset_id}")

    target_dir: Path | None = None
    for episode_id in selected_episode_ids:
        print("=" * 80)
        print(f"Appending generated episode {episode_id}")
        target_dir = append_generated_episode_to_dataset(
            dataset_id=dataset_id,
            episode_id=episode_id,
            export_dir=export_dir,
            raw_dir=raw_dir,
            augmented_root=augmented_root,
            cosmos_model=cosmos_model,
        )

    if target_dir is None:
        raise ValueError(f"No generated episodes were appended for dataset {dataset_id}")
    return target_dir


def upload_dataset_to_hf(target_dir: Path, repo_id: str, token: str, message: str) -> str:
    os.environ["HF_TOKEN"] = token
    dataset = LeRobotDataset(repo_id=repo_id, root=target_dir, download_videos=False)
    dataset.push_to_hub()
    return f"https://huggingface.co/datasets/{repo_id}"


def append_generated_episode_and_upload(
    *,
    dataset_id: str,
    episode_id: int,
    export_dir: Path,
    raw_dir: Path,
    augmented_root: Path,
    hf_repo_id: str,
    hf_token_env_var: str = "HF_TOKEN",
    cosmos_model: str = "edge/distilled",
) -> tuple[Path, str]:
    token = os.environ.get(hf_token_env_var)
    if not token:
        raise ValueError(f"Missing Hugging Face token in environment variable: {hf_token_env_var}")
    target_dir = append_generated_episode_to_dataset(
        dataset_id=dataset_id,
        episode_id=episode_id,
        export_dir=export_dir,
        raw_dir=raw_dir,
        augmented_root=augmented_root,
        cosmos_model=cosmos_model,
    )
    commit_url = upload_dataset_to_hf(
        target_dir=target_dir,
        repo_id=hf_repo_id,
        token=token,
        message=f"Append generated episode {episode_id} from {dataset_id}",
    )
    return target_dir, commit_url


def append_all_generated_episodes_and_upload(
    *,
    dataset_id: str,
    export_dir: Path,
    raw_dir: Path,
    augmented_root: Path,
    hf_repo_id: str,
    hf_token_env_var: str = "HF_TOKEN",
    cosmos_model: str = "edge/distilled",
    episode_ids: list[int] | None = None,
) -> tuple[Path, str]:
    token = os.environ.get(hf_token_env_var)
    if not token:
        raise ValueError(f"Missing Hugging Face token in environment variable: {hf_token_env_var}")
    target_dir = append_all_generated_episodes_to_dataset(
        dataset_id=dataset_id,
        export_dir=export_dir,
        raw_dir=raw_dir,
        augmented_root=augmented_root,
        cosmos_model=cosmos_model,
        episode_ids=episode_ids,
    )
    commit_url = upload_dataset_to_hf(
        target_dir=target_dir,
        repo_id=hf_repo_id,
        token=token,
        message=f"Append all generated episodes from {dataset_id}",
    )
    return target_dir, commit_url
