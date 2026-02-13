# VGGT Dataset Builder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-VGGT-lightgrey.svg)](vggt/LICENSE.txt)

Build warping datasets by rendering VGGT depth point clouds into the next view. This tool processes input images through the VGGT model to generate depth maps and camera poses, then renders point clouds into adjacent views to create training pairs for novel view synthesis.

## ✨ Key Features

- 🕳️ **Hole-Filling Renderer**: Advanced GPU-accelerated pipeline for filling occlusions in rendered views.
- 🎨 **ComfyUI Integration**: Custom nodes for VGGT model inference and point cloud generation.
- 📐 **Smart Resolution Handling**: Automatic rescaling and dimension matching for mixed-resolution datasets.
- 🔄 **Bidirectional Generation**: Create warping pairs in both directions between consecutive frames.
- ☁️ **Point Cloud Filtering**: Semantic sky removal and background filtering for cleaner datasets.
- 📦 **Multi-Format Export**: Support for high-quality JPG/PNG images and binary PLY point clouds.
- 🤖 **Dataset Integration**: One-click preparation for ModelScope and AI Toolkit training.

## 📋 Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [ModelScope LoRA Training Dataset](#modelscope-lora-training-dataset)
- [AI Toolkit Dataset](#ai-toolkit-dataset)
- [Command-Line Options](#command-line-options)
- [Input Structure](#input-structure)
- [Output Structure](#output-structure)
- [Resolution Handling](#resolution-handling)

## Setup

1. Clone this repo with submodules:

   ```bash
   git clone --recurse-submodules <repo-url>
   cd vggt-dataset-builder
   ```

2. Create and activate a virtual environment using uv:

   ```bash
   uv venv --python 3.10 --seed
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. Install PyTorch with CUDA 12.8 support:

   ```bash
   uv pip install torch==2.8.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
   ```

4. Install remaining dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

5. Log in to Hugging Face (required for gated model access):
   ```bash
   uv run python -c "from huggingface_hub import login; login()"
   ```

## Usage

Run the dataset builder:

```bash
uv run python build_warp_dataset.py
```

## ComfyUI Node (VGGT Model Inference)

This repo includes a ComfyUI node for running VGGT directly and exporting a GaussianViewer-compatible PLY plus camera matrices.

**Node name:** `VGGT Model Inference`

**Inputs (required)**

- `image_1`: Primary input image.
- `device`: `cuda` or `cpu`.

**Inputs (optional)**

- `image_2`, `image_3`, `image_4`: Additional frames for multi-image inference (1–4 total). **All images are automatically resized to the same dimensions** before processing to handle different aspect ratios.
- `depth_conf_threshold` (default: `50.0`): Percentile threshold for confidence filtering (0–100). Example: 50 keeps the top 50% of points.
- `gaussian_scale_multiplier` (default: `0.1`): Splat size multiplier for the exported Gaussian PLY.
- `preprocess_mode` (default: `crop`):
  - `crop`: Demo-style resize to width=518 and center-crop height if needed (can lose top/bottom on tall images).
  - `pad_white`: Preserve all pixels; pad to square with white borders (demo-style padding).
  - `pad_black`: Preserve all pixels; pad to square with black borders.
  - `tile`: Keep width=518 and full height (no crop); experimental for tall images.
- `mask_black_bg` (default: `false`): Filter black background pixels (RGB sum < 16).
- `mask_white_bg` (default: `false`): Filter white background pixels (RGB > 240).
- `boundary_threshold` (default: `0`): Exclude points within N pixels of the image boundary.
- `max_depth` (default: `-1`): Max depth distance; `-1` disables the filter.

**Outputs**

- `ply_path`: Path to the exported PLY (GaussianViewer compatible).
- `extrinsics`: First camera extrinsic matrix (3×4).
- `intrinsics`: First camera intrinsic matrix (3×3).

**Notes**

- For the best alignment, `crop` usually matches the demo behavior most closely.
- Use `pad_white`/`pad_black` when you need full image coverage (no cropping) at the cost of slightly less stable alignment.

**Multi-Image Aspect Ratio Handling**

The node automatically handles images with different aspect ratios when processing multiple frames (image_2, image_3, image_4):

1. **Each image is preprocessed independently** based on its own aspect ratio
2. **The first image sets the target dimensions** (based on the selected `preprocess_mode`)
3. **All subsequent images are resized to match** those target dimensions
4. **Then all images are concatenated** for batch processing

This means you can safely pass images like:

- Image 1: 1920x1440 (tall portrait)
- Image 2: 1600x900 (landscape)
- Image 3: 2048x1536 (different ratio)

The preprocessing will ensure they're all converted to the same size before being fed to the model, preventing "Sizes of tensors must match" errors.

**Example**: With `preprocess_mode=pad_white`:

- Image 1 (600x800): Resized to 392x518 (preserving aspect ratio)
- Image 2 (400x800): Resized to 252x518
- Image 3 (500x600): Resized to 434x518
- All then resized to Image 1's target: 392x518
- Then concatenated for inference

### Example Settings

Basic usage with memory constraints:

```bash
uv run python build_warp_dataset.py --upsample-depth --auto-s0 --max-megapixels 2.0
```

Higher resolution with explicit dimensions:

```bash
uv run python build_warp_dataset.py --resize-width 1216 --resize-height 832 --sigma 12 --upsample-depth --auto-s0 --max-megapixels 2.0
```

For DL3DV datasets with automatic frame selection:

```bash
uv run python build_warp_dataset.py --upsample-depth --auto-s0 --auto-skip --target-overlap 0.5 --limit 10
```

Export point clouds for visualization:

```bash
uv run python build_warp_dataset.py --upsample-depth --auto-s0 --save-ply
```

Filter sky and background for cleaner point clouds:

```bash
uv run python build_warp_dataset.py --upsample-depth --auto-s0 --filter-sky --filter-black-bg
```

Generate bidirectional pairs for enhanced training:

```bash
uv run python build_warp_dataset.py --upsample-depth --auto-s0 --bidirectional
```

Resume interrupted runs or add bidirectional pairs to existing datasets:

```bash
# Run again with --bidirectional - only missing files will be generated
uv run python build_warp_dataset.py --upsample-depth --auto-s0 --bidirectional
```

**Note**: `--filter-sky` requires additional dependencies:

```bash
uv pip install opencv-python onnxruntime
```

## ModelScope LoRA Training Dataset

To prepare the generated warping dataset for LoRA training, use the `modelscope.py` script to reorganize files into the expected format:

```bash
uv run python modelscope.py
```

This creates a `modelscope-dataset/` folder with training triplets:

- `<N>_start_1.<ext>`: Rendered splat image from reference view
- `<N>_start_2.<ext>`: Reference image
- `<N>_end.<ext>`: Target image
- `<N>.txt`: Training prompt (if `--prompt` provided)

### Example with prompt:

```bash
uv run python modelscope.py --prompt "a beautiful landscape with mountains and trees"
```

### Options:

- `--output-dir <path>`: Input directory (default: output)
- `--modelscope-dir <path>`: Output directory (default: modelscope-dataset)
- `--prompt <text>`: Prompt text to save for each triplet as `<N>.txt`
- `--quiet`: Suppress progress output

## AI Toolkit Dataset

AI Toolkit expects three folders: `control1`, `control2`, and `target`. Use `aitoolkit.py` to prepare this structure:

```bash
uv run python aitoolkit.py
```

This creates an `aitoolkit-dataset/` folder:

- `control1/<N>.<ext>`: Rendered splat image
- `control2/<N>.<ext>`: Reference image
- `target/<N>.<ext>`: Target image
- `target/<N>.txt`: Prompt text (if `--prompt` provided)

### Example with prompt:

```bash
uv run python aitoolkit.py --prompt "refer to image 2, fix the distortion and blank areas in image 1"
```

### Options:

- `--output-dir <path>`: Input directory (default: output)
- `--aitoolkit-dir <path>`: Output directory (default: aitoolkit-dataset)
- `--prompt <text>`: Prompt text to save as `target/<N>.txt`

## Viewer

Run the interactive viewer (if `pyglet` is installed) or render a single image to disk:

```bash
uv run python vggt_point_cloud_viewer.py input/01/image1.jpg --output out.png
```

Supported options (subset): `--viewport-width`, `--viewport-height`, `--resize-width`, `--resize-height`, `--sigma`, `--auto-s0`, `--s0`, `--occlusion-threshold`, `--coarse-level`, `--conf-threshold`.

The viewer will try to open an interactive window using `pyglet`. If `pyglet` is not installed or an interactive context cannot be created, the script falls back to rendering a single frame to an image using `HoleFillingRenderer`.

To install the optional viewer dependencies:

```bash
uv pip install pyglet moderngl glcontext
```
- `--quiet`: Suppress progress output

### Interactive viewer — Controls & Features

- **Right-click menu (pyglet custom UI)**: right-click in the viewer opens a small context menu with actions like "Copy image to clipboard" and "Save image (E)". The menu is drawn with an opaque background and highlights items under the pointer (supports hover while dragging).
- **Copy to clipboard**: press `Ctrl+C` (or `Cmd+C` on macOS) to render the current frame and copy the pixel image to the system clipboard. On Windows this uses `pywin32` to place a DIB-format image on the clipboard.
- **Save current frame**: press `E` to export the currently rendered frame to the `--output` path you supplied.
- **On-screen confirmation**: copy/save operations display a short on-screen toast message (bottom-center) confirming success or showing an error.
- **Camera reset**: press `F` to reset the camera to the initial view (including clearing any roll). If you prefer restoring the exact initial camera pose captured at startup, that can be enabled.
- **Input capture while menu visible**: when the context menu is open it captures input — left-click activates the highlighted menu item, right-click dismisses the menu.

Notes:
- On Windows you must have `pywin32` installed for clipboard image copy to work; `pywin32` has been added to `requirements.txt`.
- The viewer implements these features using `pyglet` primitives (labels, shapes, and batches). If you prefer native OS context menus instead of a drawn menu, that can be implemented separately per-platform.

## Command-Line Options

### Input/Output

- `--input-dir <path>`: Directory with input images organized in subdirectories (default: `input`)
- `--output-dir <path>`: Directory for output image pairs (default: `output`)
- `--output-format <format>`: Output format - `jpg` (default), `jpeg`, or `png`. JPG saves with quality=95 and optimization enabled. Confidence maps (when enabled) are always saved as PNG.

### Resolution Control

- `--max-megapixels <float>`: Maximum resolution in megapixels (default: 1.0). **Input images are automatically rescaled in-memory** to this limit before processing, constraining memory usage throughout the entire pipeline. Temporary rescaled images are cleaned up after each scene.
- `--resize-width <int>`: Explicit output width in pixels (default: 0, disabled). Capped to `--max-megapixels` if specified.
- `--resize-height <int>`: Explicit output height in pixels (default: 0, disabled). Capped to `--max-megapixels` if specified.

### Preprocessing

- `--preprocess-mode <mode>`: Image preprocessing mode (default: `crop`)
  - `crop`: Resize width to 518px, center-crop height if needed
  - `pad`: Resize largest dimension to 518px, pad smaller dimension to square

### Depth & Rendering

- `--upsample-depth`: Upsample VGGT depth/confidence maps to output resolution before rendering (disabled by default)
- `--depth-conf-threshold <float>`: Filter depth points with confidence below this value (default: 1.01, keeps all points). Lower values filter more aggressively.
- `--sigma <float>`: Gaussian splatting sigma parameter controlling splat size (default: 20.0)
- `--auto-s0`: Automatically estimate per-frame Gaussian splat size (s0) from depth and intrinsics (disabled by default)
- `--save-confidence`: Save depth confidence maps as PNG files (disabled by default)
- `--save-ply`: Save point clouds as PLY files for reference frames (disabled by default). Creates viewable 3D point clouds with RGB colors and optional confidence values.

### Point Cloud Filtering

- `--filter-sky`: Filter sky points using semantic segmentation (disabled by default). Requires `opencv-python` and `onnxruntime`. Downloads skyseg.onnx model on first use.
- `--filter-black-bg`: Filter black background points (RGB sum < 16, disabled by default). Useful for images with black borders or backgrounds.
- `--filter-white-bg`: Filter white background points (RGB > 240, disabled by default). Useful for images with white backgrounds or overexposed regions.

### Frame Selection

- `--skip-every <int>`: Use every Nth image to increase view spacing (default: 1, uses all images)
- `--auto-skip`: Automatically select frames based on `transforms.json` view overlap (disabled by default)
- `--target-overlap <float>`: Target view overlap (0-1) for auto-skip (default: 0.5)
- `--limit <int>`: Process only first N images per scene after filtering (default: 0, no limit)
- `--bidirectional`: Generate bidirectional pairs (A→B and B→A) for each consecutive frame pair (disabled by default). This doubles the dataset by creating warping pairs in both directions between consecutive frames. **Smart file detection**: Automatically detects existing triplet files and only creates missing ones, allowing you to resume interrupted runs or add bidirectional pairs to existing datasets without re-processing.

### System

- `--device <device>`: Force device selection - `cuda` or `cpu` (default: auto-detects CUDA availability)
- `--nocache`: Disable reading/writing the on-disk per-frame cache for this run (cached `.npz` files live under `.cache/`).
- `--clear-cache`: Delete the repository `.cache/` directory before running (useful to drop all precomputed frame data).
- `--force_output`: Force recalculation and overwrite existing output files (does not by itself delete the on-disk cache).

## Input Structure

Organize input images in subdirectories under `--input-dir`:

```
input/
  scene1/
    image1.jpg
    image2.jpg
    ...
  scene2/
    transforms.json  (optional, for --auto-skip)
    images/
      image1.png
      ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.heic`, `.heif`

## Output Structure

For each scene, generates triplets (or quadruplets with `--save-ply`) of images in `--output-dir`:

```
output/
  scene1/
    image2_splats.jpg      # Rendered from previous view (image1→image2)
    image2_target.jpg      # Ground truth current view
    image2_reference.jpg   # Previous view (reference)
    image2_confidence.png  # Depth confidence map (if --save-confidence)
    image2_reference.ply   # Point cloud (if --save-ply)
```

With `--bidirectional`, also generates reverse pairs:

```
output/
  scene1/
    image1_splats.jpg      # Rendered from next view (image2→image1)
    image1_target.jpg      # Ground truth previous view
    image1_reference.jpg   # Next view (reference)
    image1_confidence.png  # Depth confidence map (if --save-confidence)
    image1_reference.ply   # Point cloud (if --save-ply)
```

### PLY Files

When `--save-ply` is enabled, point cloud files are saved in binary PLY format containing:

- **Vertices**: 3D world coordinates (X, Y, Z) for each valid depth point
- **Colors**: RGB values (0-255) from the reference image
- **Confidence**: Depth confidence values (if `--save-confidence` is also enabled)

Binary format provides ~80% smaller file sizes compared to ASCII. PLY files can be viewed in tools like MeshLab, CloudCompare, or Blender.

## Resolution Handling

### Automatic Rescaling

All input images are **automatically rescaled in-memory** to the `--max-megapixels` limit before any processing. Rescaled images are saved to temporary directories and cleaned up after each scene is processed, avoiding persistent disk usage.

### Mixed Resolutions

When `--upsample-depth` is enabled and images within a scene have different resolutions (after rescaling), the script:

1. Automatically uses the **minimum dimensions** across all images in that scene
2. Resizes all images to match these dimensions
3. Proceeds with depth upsampling

This avoids upscaling artifacts while ensuring consistent output resolution per scene.

### Explicit Dimensions

If `--resize-width` and `--resize-height` are specified, they are capped to honor `--max-megapixels`. For example, with `--max-megapixels 2.0` and `--resize-width 2000 --resize-height 1500` (3.0MP), the dimensions will be scaled down proportionally to ~1633×1225 (2.0MP).

## ComfyUI Node (VGGT Model Inference)

This repository includes a ComfyUI node implementation (`VGGT Model Inference`) that runs the VGGT model and exports a GaussianViewer-compatible PLY plus camera matrices. The node is implemented in `vggt_comfy_nodes.py` and mirrors the demo processing pipeline used elsewhere in the project.

**Node name:** `VGGT Model Inference`

**Inputs (required)**

- `device`: `cuda` or `cpu` (default: `cuda`)

**Inputs (optional)**

- `image_1`..`image_20`: Zero or more connected image inputs. The node will accept up to 20 image inputs and will process all connected images as a sequence. `image_1` is commonly used as the primary frame.
- `depth_conf_threshold` (default: `50.0`): Percentile threshold (0–100) for confidence filtering; e.g. `50.0` keeps the top 50% of points.
- `gaussian_scale_multiplier` (default: `0.1`): Multiplier for Gaussian splat size written to the exported PLY.
- `preprocess_mode` (default: `crop`): How each image is preprocessed prior to inference. Options:
  - `crop`: Resize to fit within 518px then center-crop height when necessary (matches demo behavior).
  - `pad_white`: Resize preserving aspect ratio; pad to square with white borders.
  - `pad_black`: Resize preserving aspect ratio; pad to square with black borders.
  - `tile`: Keep width = 518px and allow full height (experimental for tall images).
- `mask_black_bg` (default: `false`): Remove nearly-black background pixels.
- `mask_white_bg` (default: `false`): Remove nearly-white background pixels.
- `mask_sky` (default: `false`): Use semantic sky segmentation to filter sky points (requires `opencv-python` and `onnxruntime`).
- `boundary_threshold` (default: `0`): Exclude points within N pixels of image edges.
- `max_depth` (default: `-1.0`): Maximum depth distance to keep; `-1.0` disables this filter.
- `upsample_depth` (default: `true`): Upsample VGGT depth and confidence maps to each image's original resolution prior to unprojection. When enabled and all input images share the same original resolution, depth/conf maps are bicubically upsampled and intrinsics adjusted accordingly.

**Outputs**

- `ply_path` (STRING): Path to the exported PLY file (GaussianViewer-compatible).
- `extrinsics` (EXTRINSICS): First camera's extrinsic matrix (3×4).
- `intrinsics` (INTRINSICS): First camera's intrinsic matrix (3×3).

**Behavioral notes**

- Multi-image processing: each connected image is preprocessed independently, then the first processed image's dimensions are used as the target size — subsequent images are resized to match before concatenation. This avoids tensor size mismatch errors when supplying mixed-aspect images.
- Tiling (`preprocess_mode=tile`) is experimental: the node will keep width=518 and allow larger heights, using an internal tile flag to indicate special handling for very tall images.
- When `upsample_depth` is enabled, intrinsics are scaled to match the upsampled resolution so unprojection yields correctly scaled world points.
- Sky filtering downloads a small ONNX model on first use and requires `opencv-python` and `onnxruntime`.

**Example (basic)**

Use the node with `preprocess_mode=crop` and default filtering to generate a PLY and camera matrices for downstream nodes.

---
