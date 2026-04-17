from __future__ import annotations

"""Compatibility facade for the refactored workflow modules.

The implementation now lives in:
- embodied_data_transfer.common
- embodied_data_transfer.dataset_processing
- embodied_data_transfer.cosmos_workflow
- embodied_data_transfer.augmentation
"""

from embodied_data_transfer.augmentation import (
    append_all_generated_episodes_and_upload,
    append_all_generated_episodes_to_dataset,
    append_generated_episode_and_upload,
    append_generated_episode_to_dataset,
    directory_size_mb,
    ffprobe_duration_seconds,
    has_complete_dataset_snapshot,
    initialize_augmented_dataset,
    list_generated_variant_dirs,
    load_video_frames,
    next_file_index,
    upload_dataset_to_hf,
)
from embodied_data_transfer.common import (
    augmented_dataset_dir_name,
    augmented_repo_id,
    dataset_dir_name,
    load_json_file,
    to_pretty_json,
    to_serializable,
    write_json_file,
)
from embodied_data_transfer.cosmos_workflow import (
    build_cosmos_edge_spec,
    build_cosmos_inference_command,
    build_cosmos_inference_env,
    collect_generated_videos,
    cosmos_run_dir_name,
    default_num_steps_for_model,
    prepare_cosmos_edge_jobs,
    run_cosmos_depth_inference_for_all_episodes,
    run_cosmos_depth_inference_for_episode,
    run_cosmos_depth_inference_parallel_single_gpu,
    variant_dir_name,
    variant_seed,
)
from embodied_data_transfer.dataset_processing import (
    discover_video_keys,
    download_dataset_file,
    download_episode_metadata,
    download_root_repo_files,
    download_video_assets,
    export_episode_directory,
    find_episode_dir,
    group_rows_by_episode,
    inspect_dataset,
    list_available_episode_ids,
    list_dataset_repo_paths,
    load_episode_metadata,
    load_tabular_dataset,
    process_dataset,
    read_dataset_info,
)
