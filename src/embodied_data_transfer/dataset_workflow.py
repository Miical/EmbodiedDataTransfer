from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from lerobot.datasets import LeRobotDataset


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


def build_cosmos_edge_spec(
    *,
    name: str,
    video_path: Path,
    prompt_path: Path,
    spec_path: Path,
    guidance: int,
    num_steps: int,
    seed: int,
) -> Path:
    spec = {
        "name": name,
        "prompt_path": str(prompt_path),
        "video_path": str(video_path),
        "guidance": guidance,
        "num_steps": num_steps,
        "seed": seed,
        "edge": {
            "control_weight": 1.0,
        },
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec_path


def cosmos_run_dir_name(cosmos_model: str) -> str:
    return f"cosmos_{cosmos_model.replace('/', '_')}"


def default_num_steps_for_model(cosmos_model: str) -> int:
    return 4 if "distilled" in cosmos_model else 35


def variant_dir_name(variant_index: int) -> str:
    return f"variant_{variant_index:03d}"


def variant_seed(base_seed: int, variant_index: int) -> int:
    return base_seed + variant_index


def prepare_cosmos_edge_jobs(
    *,
    dataset_id: str,
    episode_id: int,
    export_dir: Path,
    cosmos_prompt_path: Path,
    guidance: int,
    cosmos_model: str,
    num_steps: int | None,
    seed: int,
    variant_index: int,
) -> tuple[Path, Path, Path, list[tuple[str, Path]]]:
    episode_dir = find_episode_dir(export_dir=export_dir, dataset_id=dataset_id, episode_id=episode_id)
    input_videos = sorted(episode_dir.glob("*.mp4"))
    if not input_videos:
        raise FileNotFoundError(f"No input videos found in {episode_dir}")

    cosmos_run_dir = episode_dir / cosmos_run_dir_name(cosmos_model)
    variant_dir = cosmos_run_dir / "variants" / variant_dir_name(variant_index)
    specs_dir = variant_dir / "specs"
    outputs_dir = variant_dir / "outputs"
    batch_output_dir = outputs_dir / "batch"
    generated_dir = variant_dir / "generated"
    specs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    resolved_num_steps = num_steps if num_steps is not None else default_num_steps_for_model(cosmos_model)

    jobs: list[tuple[str, Path]] = []
    for video_path in input_videos:
        video_stem = video_path.stem
        job_name = f"episode_{episode_id:03d}_{video_stem}_edge"
        spec_path = build_cosmos_edge_spec(
            name=job_name,
            video_path=video_path.resolve(),
            prompt_path=cosmos_prompt_path.resolve(),
            spec_path=specs_dir / f"{job_name}.json",
            guidance=guidance,
            num_steps=resolved_num_steps,
            seed=seed,
        )
        jobs.append((video_stem, spec_path.resolve()))

    write_json_file(
        variant_dir / "run_meta.json",
        {
            "episode_id": episode_id,
            "variant_index": variant_index,
            "seed": seed,
            "num_steps": resolved_num_steps,
            "guidance": guidance,
            "cosmos_model": cosmos_model,
        },
    )

    return cosmos_run_dir, variant_dir, batch_output_dir, jobs


def build_cosmos_inference_command(
    *,
    cosmos_python: Path,
    cosmos_model: str,
    spec_paths: list[Path],
    output_dir: Path,
    nproc_per_node: int,
    master_port: int,
) -> list[str]:
    if nproc_per_node <= 1:
        cmd = [
            str(cosmos_python),
            "examples/inference.py",
            "--model",
            cosmos_model,
            "-i",
        ]
    else:
        cmd = [
            str(cosmos_python),
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={master_port}",
            "examples/inference.py",
            "--model",
            cosmos_model,
            "-i",
        ]
    cmd.extend(str(path) for path in spec_paths)
    cmd.extend(["-o", str(output_dir.resolve())])
    return cmd


def build_cosmos_inference_env(
    *,
    hf_home: Path | None,
    cosmos_experimental_checkpoints: bool,
    gpu_id: int | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)
    if cosmos_experimental_checkpoints:
        env["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def collect_generated_videos(
    *,
    episode_id: int,
    jobs: list[tuple[str, Path]],
    batch_output_dir: Path,
    generated_dir: Path,
) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    for video_stem, _ in jobs:
        job_name = f"episode_{episode_id:03d}_{video_stem}_edge"
        generated_video = batch_output_dir / f"{job_name}.mp4"
        if not generated_video.exists():
            raise FileNotFoundError(f"Expected generated video not found: {generated_video}")
        shutil.copy2(generated_video, generated_dir / f"{video_stem}_generated.mp4")


def run_cosmos_depth_inference_for_episode(
    *,
    dataset_id: str,
    episode_id: int,
    export_dir: Path,
    cosmos_root: Path,
    cosmos_python: Path,
    cosmos_prompt_path: Path,
    guidance: int = 3,
    cosmos_model: str = "edge/distilled",
    num_steps: int | None = None,
    seed: int = 1,
    num_trajectories: int = 1,
    hf_home: Path | None = None,
    cosmos_experimental_checkpoints: bool = True,
    nproc_per_node: int = 8,
    master_port: int = 12341,
) -> Path:
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")

    cosmos_run_dir = find_episode_dir(export_dir=export_dir, dataset_id=dataset_id, episode_id=episode_id) / cosmos_run_dir_name(cosmos_model)
    for variant_index in range(num_trajectories):
        current_seed = variant_seed(seed, variant_index)
        _, variant_dir, batch_output_dir, jobs = prepare_cosmos_edge_jobs(
            dataset_id=dataset_id,
            episode_id=episode_id,
            export_dir=export_dir,
            cosmos_prompt_path=cosmos_prompt_path,
            guidance=guidance,
            cosmos_model=cosmos_model,
            num_steps=num_steps,
            seed=current_seed,
            variant_index=variant_index,
        )
        cmd = build_cosmos_inference_command(
            cosmos_python=cosmos_python,
            cosmos_model=cosmos_model,
            spec_paths=[spec_path for _, spec_path in jobs],
            output_dir=batch_output_dir,
            nproc_per_node=nproc_per_node,
            master_port=master_port + variant_index,
        )
        env = build_cosmos_inference_env(
            hf_home=hf_home,
            cosmos_experimental_checkpoints=cosmos_experimental_checkpoints,
        )

        subprocess.run(cmd, check=True, cwd=str(cosmos_root), env=env)

        collect_generated_videos(
            episode_id=episode_id,
            jobs=jobs,
            batch_output_dir=batch_output_dir,
            generated_dir=variant_dir / "generated",
        )

    return cosmos_run_dir


def run_cosmos_depth_inference_parallel_single_gpu(
    *,
    dataset_id: str,
    export_dir: Path,
    cosmos_root: Path,
    cosmos_python: Path,
    cosmos_prompt_path: Path,
    gpu_ids: list[int],
    guidance: int = 3,
    cosmos_model: str = "edge/distilled",
    num_steps: int | None = None,
    seed: int = 1,
    num_trajectories: int = 1,
    hf_home: Path | None = None,
    cosmos_experimental_checkpoints: bool = True,
    master_port_start: int = 12341,
    episode_ids: list[int] | None = None,
    poll_interval_seconds: float = 2.0,
) -> list[Path]:
    selected_episode_ids = episode_ids or list_available_episode_ids(export_dir=export_dir, dataset_id=dataset_id)
    if not selected_episode_ids:
        raise ValueError(f"No episode directories found for dataset {dataset_id}")
    if not gpu_ids:
        raise ValueError("At least one GPU id is required for parallel single-GPU scheduling")
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")

    pending_jobs = [
        (episode_id, variant_index)
        for episode_id in selected_episode_ids
        for variant_index in range(num_trajectories)
    ]
    available_gpu_ids = list(gpu_ids)
    active_jobs: dict[int, dict[str, Any]] = {}
    completed_run_dirs: list[Path] = []
    failures: list[tuple[int, int, int]] = []
    worker_offset = 0

    while pending_jobs or active_jobs:
        while pending_jobs and available_gpu_ids:
            episode_id, variant_index = pending_jobs.pop(0)
            gpu_id = available_gpu_ids.pop(0)
            master_port = master_port_start + worker_offset
            worker_offset += 1
            current_seed = variant_seed(seed, variant_index)

            cosmos_run_dir, variant_dir, batch_output_dir, jobs = prepare_cosmos_edge_jobs(
                dataset_id=dataset_id,
                episode_id=episode_id,
                export_dir=export_dir,
                cosmos_prompt_path=cosmos_prompt_path,
                guidance=guidance,
                cosmos_model=cosmos_model,
                num_steps=num_steps,
                seed=current_seed,
                variant_index=variant_index,
            )
            cmd = build_cosmos_inference_command(
                cosmos_python=cosmos_python,
                cosmos_model=cosmos_model,
                spec_paths=[spec_path for _, spec_path in jobs],
                output_dir=batch_output_dir,
                nproc_per_node=1,
                master_port=master_port,
            )
            env = build_cosmos_inference_env(
                hf_home=hf_home,
                cosmos_experimental_checkpoints=cosmos_experimental_checkpoints,
                gpu_id=gpu_id,
            )
            log_path = variant_dir / "outputs" / f"worker_gpu{gpu_id}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                cwd=str(cosmos_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active_jobs[process.pid] = {
                "process": process,
                "gpu_id": gpu_id,
                "episode_id": episode_id,
                "variant_index": variant_index,
                "seed": current_seed,
                "cosmos_run_dir": cosmos_run_dir,
                "variant_dir": variant_dir,
                "batch_output_dir": batch_output_dir,
                "jobs": jobs,
                "log_handle": log_handle,
                "log_path": log_path,
            }
            print(
                f"Started episode {episode_id} variant {variant_index} seed {current_seed} on GPU {gpu_id} "
                f"(pid={process.pid}, log={log_path})"
            )

        if not active_jobs:
            break

        finished_pids: list[int] = []
        for pid, job in active_jobs.items():
            process: subprocess.Popen[str] = job["process"]
            return_code = process.poll()
            if return_code is None:
                continue

            finished_pids.append(pid)
            job["log_handle"].close()
            available_gpu_ids.append(job["gpu_id"])

            if return_code == 0:
                collect_generated_videos(
                    episode_id=job["episode_id"],
                    jobs=job["jobs"],
                    batch_output_dir=job["batch_output_dir"],
                    generated_dir=job["variant_dir"] / "generated",
                )
                completed_run_dirs.append(job["variant_dir"])
                print(
                    f"Finished episode {job['episode_id']} variant {job['variant_index']} "
                    f"seed {job['seed']} on GPU {job['gpu_id']} "
                    f"(log={job['log_path']})"
                )
            else:
                failures.append((job["episode_id"], job["variant_index"], return_code))
                print(
                    f"Episode {job['episode_id']} variant {job['variant_index']} "
                    f"failed on GPU {job['gpu_id']} "
                    f"with exit code {return_code} (log={job['log_path']})"
                )

        for pid in finished_pids:
            active_jobs.pop(pid, None)

        available_gpu_ids.sort()
        if active_jobs:
            time.sleep(poll_interval_seconds)

    if failures:
        failure_summary = ", ".join(
            f"episode {episode_id} variant {variant_index} (exit {code})"
            for episode_id, variant_index, code in failures
        )
        raise RuntimeError(f"Parallel Cosmos inference failed for {failure_summary}")

    return completed_run_dirs


def list_available_episode_ids(export_dir: Path, dataset_id: str) -> list[int]:
    dataset_export_dir = export_dir / dataset_id.replace("/", "_")
    if not dataset_export_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {dataset_export_dir}")

    episode_ids: list[int] = []
    for episode_dir in sorted(dataset_export_dir.glob("episode_*")):
        try:
            episode_ids.append(int(episode_dir.name.split("_")[-1]))
        except ValueError:
            continue
    return episode_ids


def run_cosmos_depth_inference_for_all_episodes(
    *,
    dataset_id: str,
    export_dir: Path,
    cosmos_root: Path,
    cosmos_python: Path,
    cosmos_prompt_path: Path,
    guidance: int = 3,
    cosmos_model: str = "edge/distilled",
    num_steps: int | None = None,
    seed: int = 1,
    num_trajectories: int = 1,
    hf_home: Path | None = None,
    cosmos_experimental_checkpoints: bool = True,
    nproc_per_node: int = 8,
    master_port: int = 12341,
    episode_ids: list[int] | None = None,
) -> list[Path]:
    selected_episode_ids = episode_ids or list_available_episode_ids(export_dir=export_dir, dataset_id=dataset_id)
    if not selected_episode_ids:
        raise ValueError(f"No episode directories found for dataset {dataset_id}")

    run_dirs: list[Path] = []
    for episode_id in selected_episode_ids:
        print("=" * 80)
        print(f"Running Cosmos depth inference for episode {episode_id}")
        run_dir = run_cosmos_depth_inference_for_episode(
            dataset_id=dataset_id,
            episode_id=episode_id,
            export_dir=export_dir,
            cosmos_root=cosmos_root,
            cosmos_python=cosmos_python,
            cosmos_prompt_path=cosmos_prompt_path,
            guidance=guidance,
            cosmos_model=cosmos_model,
            num_steps=num_steps,
            seed=seed,
            num_trajectories=num_trajectories,
            hf_home=hf_home,
            cosmos_experimental_checkpoints=cosmos_experimental_checkpoints,
            nproc_per_node=nproc_per_node,
            master_port=master_port,
        )
        run_dirs.append(run_dir)
    return run_dirs


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
