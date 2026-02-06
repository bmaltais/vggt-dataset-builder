# VGGT Dataset Builder

Build warping datasets by rendering VGGT depth point clouds into the next view. This tool processes input images through the VGGT model to generate depth maps and camera poses, then renders point clouds into adjacent views to create training pairs for novel view synthesis.

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
- `--quiet`: Suppress progress output

## Command-Line Options

### Input/Output
- `--input-dir <path>`: Directory with input images organized in subdirectories (default: `input`)
- `--output-dir <path>`: Directory for output image pairs (default: `output`)
- `--output-format <format>`: Output format - `jpg` (default), `jpeg`, or `png`. JPG saves with quality=95 and optimization enabled. Confidence maps are always saved as PNG.

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
- `--no-confidence`: Skip saving depth confidence maps (saves confidence by default)
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

### System
- `--device <device>`: Force device selection - `cuda` or `cpu` (default: auto-detects CUDA availability)

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
    image2_splats.jpg      # Rendered from previous view
    image2_target.jpg      # Ground truth current view
    image2_reference.jpg   # Previous view (reference)
    image2_confidence.png  # Depth confidence map (if not --no-confidence)
    image2_reference.ply   # Point cloud (if --save-ply)
```

### PLY Files
When `--save-ply` is enabled, point cloud files are saved in binary PLY format containing:
- **Vertices**: 3D world coordinates (X, Y, Z) for each valid depth point
- **Colors**: RGB values (0-255) from the reference image
- **Confidence**: Depth confidence values (if not `--no-confidence`)

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

