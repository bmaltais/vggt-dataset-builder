# CLAUDE.md — vggt-dataset-builder

Project-level instructions for Claude Code. Supplements `~/.claude/CLAUDE.md`.

## Project Overview

VGGT Dataset Builder: ComfyUI custom nodes + CLI scripts for building 3D datasets using
the VGGT (Video/Image-to-Geometry-Gaussian-Transform) model. Outputs point clouds (.ply),
depth maps, and Warp-format datasets.

Key files:
- [vggt_comfy_nodes.py](vggt_comfy_nodes.py) — ComfyUI node definitions
- [build_warp_dataset.py](build_warp_dataset.py) — CLI dataset builder
- [hole_filling_renderer.py](hole_filling_renderer.py) — GPU-based hole-filling renderer
- [dataset_utils.py](dataset_utils.py) — shared dataset utilities
- [vggt_point_cloud_viewer.py](vggt_point_cloud_viewer.py) — point cloud viewer

## Development Conventions

### Python Environment
- **Always use `uv`** — never `pip` directly
- Run scripts: `uv run python script.py`
- Install deps: `uv pip install <package>`

### Code Quality
- Format with `black` before every commit: `uv run black .`
- Pylint score tracked; avoid introducing new lint warnings
- Remove unused imports promptly

### Testing
- Run tests: `uv run pytest`
- Markers: `unit` (fast, no GPU) and `integration` (may need GPU/model)
- Test images: `tests/input-img/` (PNG files in numbered subdirs like `02/`)
- Model downloads cached at `.cache/` (gitignored)
- Smoke tests exist for viewer and renderer — run them to catch regressions

### Commit Style
Prefix commit messages with the pattern used in this repo:
- `⚡ Bolt:` — performance optimization
- `🏜️ Dryer:` — DRY/deduplication refactor
- `🏛️ Classer:` — extract class from procedural code
- `🧪 Testify:` — add/fix tests
- `🔍 Pylint:` — lint/import cleanup

## Before Committing

1. `uv run black .`
2. `uv run pytest` (at minimum the unit tests)
3. Verify no new pylint warnings on changed files

## Architecture Notes

- GPU operations use CUDA via PyTorch; device selection via shared utility in `dataset_utils.py`
- Frame caching managed by `FrameCacheManager` class
- Point cloud filtering by `PointCloudFilter` class (color/sky/black-background)
- VGGT model path setup centralized — see `dataset_utils.py` `setup_vggt_path()`
