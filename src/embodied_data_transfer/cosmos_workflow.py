from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from embodied_data_transfer.common import write_json_file
from embodied_data_transfer.dataset_processing import find_episode_dir, list_available_episode_ids


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
    disable_guardrails: bool = False,
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
    if disable_guardrails:
        cmd.append("--disable-guardrails")
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
    debug_log_path = batch_output_dir / "debug.log"
    debug_log_text = debug_log_path.read_text(encoding="utf-8", errors="ignore") if debug_log_path.exists() else ""
    for video_stem, _ in jobs:
        job_name = f"episode_{episode_id:03d}_{video_stem}_edge"
        generated_video = batch_output_dir / f"{job_name}.mp4"
        if not generated_video.exists():
            if "GUARDRAIL BLOCKED" in debug_log_text:
                raise RuntimeError(
                    "Cosmos guardrails blocked the generated video, so the final output mp4 was not saved. "
                    f"Missing file: {generated_video}. "
                    "You can retry with guardrails disabled in this workflow."
                )
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
    disable_guardrails: bool = False,
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
            disable_guardrails=disable_guardrails,
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
    disable_guardrails: bool = False,
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
                disable_guardrails=disable_guardrails,
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
    disable_guardrails: bool = False,
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
            disable_guardrails=disable_guardrails,
            hf_home=hf_home,
            cosmos_experimental_checkpoints=cosmos_experimental_checkpoints,
            nproc_per_node=nproc_per_node,
            master_port=master_port,
        )
        run_dirs.append(run_dir)
    return run_dirs
