from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
