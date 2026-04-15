# EmbodiedDataTransfer

Minimal Python project skeleton.

## Quick Start

```bash
cd /file_system/liujincheng/Projects/EmbodiedDataTransfer
python -m venv .venv
source .venv/bin/activate
pip install -e .
embodied-data-transfer
```

## Layout

```text
EmbodiedDataTransfer/
├── .gitignore
├── pyproject.toml
├── README.md
├── src/
│   └── embodied_data_transfer/
│       ├── __about__.py
│       ├── __init__.py
│       ├── __main__.py
│       └── cli.py
└── tests/
```

*** Update File: /file_system/liujincheng/Projects/EmbodiedDataTransfer/.gitignore
@@
+ .venv/
+ __pycache__/
+ *.py[cod]
+ .pytest_cache/
+ .mypy_cache/
+ .ruff_cache/
+ .coverage
+ coverage.xml
+ build/
+ dist/
+ *.egg-info/
+ .DS_Store

