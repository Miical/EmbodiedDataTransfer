from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return to_serializable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def to_pretty_json(sample: dict[str, Any]) -> str:
    return json.dumps(sample, ensure_ascii=False, indent=2, default=str)


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
            # Some datasets may not expose both files; keep initialization permissive.
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

    dataset_export_dir = export_dir / dataset_id.replace("/", "_")
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
    dataset_export_dir = export_dir / dataset_id.replace("/", "_")
    episode_dir = dataset_export_dir / f"episode_{episode_id:03d}"
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    return episode_dir


def build_cosmos_depth_spec(
    *,
    name: str,
    video_path: Path,
    prompt_path: Path,
    spec_path: Path,
    guidance: int,
) -> Path:
    spec = {
        "name": name,
        "prompt_path": str(prompt_path),
        "video_path": str(video_path),
        "guidance": guidance,
        "depth": {
            "control_weight": 1.0,
        },
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec_path


def run_cosmos_depth_inference_for_episode(
    *,
    dataset_id: str,
    episode_id: int,
    export_dir: Path,
    cosmos_root: Path,
    cosmos_python: Path,
    cosmos_prompt_path: Path,
    guidance: int = 3,
) -> Path:
    episode_dir = find_episode_dir(export_dir=export_dir, dataset_id=dataset_id, episode_id=episode_id)
    input_videos = sorted(episode_dir.glob("*.mp4"))
    if not input_videos:
        raise FileNotFoundError(f"No input videos found in {episode_dir}")

    cosmos_run_dir = episode_dir / "cosmos_depth"
    specs_dir = cosmos_run_dir / "specs"
    outputs_dir = cosmos_run_dir / "outputs"
    batch_output_dir = outputs_dir / "batch"
    generated_dir = cosmos_run_dir / "generated"
    specs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    for video_path in input_videos:
        video_stem = video_path.stem
        job_name = f"episode_{episode_id:03d}_{video_stem}_depth"
        spec_path = build_cosmos_depth_spec(
            name=job_name,
            video_path=video_path.resolve(),
            prompt_path=cosmos_prompt_path.resolve(),
            spec_path=specs_dir / f"{job_name}.json",
            guidance=guidance,
        )
        jobs.append((video_stem, spec_path.resolve()))

    cmd = [
        str(cosmos_python),
        "examples/inference.py",
        "--model",
        "depth",
    ]
    for _, spec_path in jobs:
        cmd.extend(["-i", str(spec_path)])
    cmd.extend(["-o", str(batch_output_dir.resolve())])
    subprocess.run(cmd, check=True, cwd=str(cosmos_root))

    for video_stem, _ in jobs:
        job_name = f"episode_{episode_id:03d}_{video_stem}_depth"
        generated_video = batch_output_dir / f"{job_name}.mp4"
        if not generated_video.exists():
            raise FileNotFoundError(f"Expected generated video not found: {generated_video}")

        shutil.copy2(generated_video, generated_dir / f"{video_stem}_generated.mp4")

    return cosmos_run_dir


def dataset_dir_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "_")


def augmented_dataset_dir_name(dataset_id: str) -> str:
    return f"{dataset_dir_name(dataset_id)}_augmented"


def augmented_repo_id(dataset_id: str) -> str:
    owner, name = dataset_id.split("/", maxsplit=1)
    return f"{owner}/{name}-augmented"


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def initialize_augmented_dataset(raw_dir: Path, dataset_id: str, augmented_root: Path) -> Path:
    target_dir = augmented_root / augmented_dataset_dir_name(dataset_id)
    if target_dir.exists():
        return target_dir
    shutil.copytree(raw_dir, target_dir, ignore=shutil.ignore_patterns(".cache"))
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


def append_generated_episode_to_dataset(
    *,
    dataset_id: str,
    episode_id: int,
    export_dir: Path,
    raw_dir: Path,
    augmented_root: Path,
) -> Path:
    source_episode_dir = find_episode_dir(export_dir=export_dir, dataset_id=dataset_id, episode_id=episode_id)
    generated_dir = source_episode_dir / "cosmos_depth" / "generated"
    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated video directory not found: {generated_dir}")

    target_dir = initialize_augmented_dataset(raw_dir=raw_dir, dataset_id=dataset_id, augmented_root=augmented_root)
    manifest_path = target_dir / "meta" / "augmentation_manifest.json"
    manifest = load_json_file(manifest_path) if manifest_path.exists() else {"source_dataset": dataset_id, "appended": []}
    if any(item["source_episode_id"] == episode_id for item in manifest["appended"]):
        raise ValueError(f"Episode {episode_id} has already been appended to {target_dir}")

    info_path = target_dir / "meta" / "info.json"
    info = load_json_file(info_path)

    new_episode_index = int(info["total_episodes"])
    new_frame_start_index = int(info["total_frames"])

    data_chunk_dir = target_dir / "data" / "chunk-000"
    meta_chunk_dir = target_dir / "meta" / "episodes" / "chunk-000"
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    meta_chunk_dir.mkdir(parents=True, exist_ok=True)

    data_file_index = next_file_index("file-*.parquet", data_chunk_dir)
    meta_file_index = next_file_index("file-*.parquet", meta_chunk_dir)

    frames = load_json_file(source_episode_dir / "frames.json")
    updated_frames: list[dict[str, Any]] = []
    for offset, frame in enumerate(frames):
        frame = dict(frame)
        frame["episode_index"] = new_episode_index
        frame["index"] = new_frame_start_index + offset
        updated_frames.append(frame)

    frame_df = pd.DataFrame(updated_frames)
    data_file_path = data_chunk_dir / f"file-{data_file_index:03d}.parquet"
    frame_df.to_parquet(data_file_path, index=False)

    source_episode_meta = load_json_file(source_episode_dir / "episode_meta.json")
    new_episode_meta = dict(source_episode_meta)
    new_episode_meta["episode_index"] = new_episode_index
    new_episode_meta["data/chunk_index"] = 0
    new_episode_meta["data/file_index"] = data_file_index
    new_episode_meta["dataset_from_index"] = new_frame_start_index
    new_episode_meta["dataset_to_index"] = new_frame_start_index + len(updated_frames)

    video_keys = [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]
    for video_key in video_keys:
        source_video = generated_dir / f"{video_key}_generated.mp4"
        if not source_video.exists():
            raise FileNotFoundError(f"Generated video missing for {video_key}: {source_video}")
        target_video_dir = target_dir / "videos" / video_key / "chunk-000"
        target_video_dir.mkdir(parents=True, exist_ok=True)
        video_file_index = next_file_index("file-*.mp4", target_video_dir)
        target_video = target_video_dir / f"file-{video_file_index:03d}.mp4"
        shutil.copy2(source_video, target_video)

        duration = ffprobe_duration_seconds(target_video)
        new_episode_meta[f"videos/{video_key}/chunk_index"] = 0
        new_episode_meta[f"videos/{video_key}/file_index"] = video_file_index
        new_episode_meta[f"videos/{video_key}/from_timestamp"] = 0.0
        new_episode_meta[f"videos/{video_key}/to_timestamp"] = duration

    meta_file_path = meta_chunk_dir / f"file-{meta_file_index:03d}.parquet"
    pd.DataFrame([new_episode_meta]).to_parquet(meta_file_path, index=False)

    info["total_episodes"] = new_episode_index + 1
    info["total_frames"] = new_frame_start_index + len(updated_frames)
    info["splits"]["train"] = f"0:{info['total_episodes']}"
    info["data_files_size_in_mb"] = directory_size_mb(target_dir / "data")
    info["video_files_size_in_mb"] = directory_size_mb(target_dir / "videos")
    write_json_file(info_path, info)

    manifest["appended"].append(
        {
            "source_episode_id": episode_id,
            "new_episode_index": new_episode_index,
            "data_file": str(data_file_path.relative_to(target_dir)),
            "meta_file": str(meta_file_path.relative_to(target_dir)),
        }
    )
    write_json_file(manifest_path, manifest)

    print(f"Appended source episode {episode_id} as episode {new_episode_index}")
    print(f"Target dataset: {target_dir}")
    print(f"Frames written: {len(updated_frames)}")
    return target_dir


def upload_dataset_to_hf(target_dir: Path, repo_id: str, token: str, message: str) -> str:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(target_dir),
        commit_message=message,
        ignore_patterns=[".cache/*"],
    )


def append_generated_episode_and_upload(
    *,
    dataset_id: str,
    episode_id: int,
    export_dir: Path,
    raw_dir: Path,
    augmented_root: Path,
    hf_repo_id: str,
    hf_token_env_var: str = "HF_TOKEN",
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
    )
    commit_url = upload_dataset_to_hf(
        target_dir=target_dir,
        repo_id=hf_repo_id,
        token=token,
        message=f"Append generated episode {episode_id} from {dataset_id}",
    )
    return target_dir, commit_url
