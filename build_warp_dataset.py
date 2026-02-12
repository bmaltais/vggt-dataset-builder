import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add the vggt submodule to the path so its internal imports work
sys.path.insert(0, str(Path(__file__).parent / "vggt"))

import numpy as np
import torch
import torch.nn.functional as torch_nn
from PIL import Image
import pillow_heif

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from hole_filling_renderer import HoleFillingRenderer

try:
    import cv2
    import onnxruntime
    from vggt.visual_util import segment_sky, download_file_from_url
    SKY_FILTER_AVAILABLE = True
except ImportError:
    SKY_FILTER_AVAILABLE = False
    cv2 = None
    onnxruntime = None

pillow_heif.register_heif_opener()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a warping dataset by rendering VGGT depth point clouds into the next view."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory with input images (default: input).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write train/test image pairs (default: output).",
    )
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default="crop",
        choices=["crop", "pad"],
        help="Preprocessing mode used before VGGT inference (default: crop).",
    )
    parser.add_argument(
        "--depth-conf-threshold",
        type=float,
        default=1.01,
        help="Filter depth points with confidence below this value (default: 1.01).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=20.0,
        help="Sigma for fake Gaussian splatting (default: 20.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Force a device selection; defaults to cuda if available.",
    )
    parser.add_argument(
        "--skip-every",
        type=int,
        default=1,
        help=(
            "Use every Nth image in each subfolder to increase view spacing "
            "(default: 1, which uses every image)."
        ),
    )
    parser.add_argument(
        "--auto-skip",
        action="store_true",
        help=(
            "Automatically select frames per scene based on transforms.json "
            "view overlap (default: off)."
        ),
    )
    parser.add_argument(
        "--target-overlap",
        type=float,
        default=0.5,
        help=(
            "Target view overlap (0-1) between selected frames when auto-skipping "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Resize input images to this width before preprocessing (default: 0).",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=0,
        help="Resize input images to this height before preprocessing (default: 0).",
    )
    parser.add_argument(
        "--upsample-depth",
        action="store_true",
        help=(
            "Upsample VGGT depth/confidence maps to the output resolution before rendering "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--auto-s0",
        action="store_true",
        help=("Estimate s0 per frame from depth and intrinsics (default: off)."),
    )
    parser.add_argument(
        "--save-confidence",
        action="store_true",
        help=("Save the depth confidence map (default: off)."),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=("Limit to the first N images after filtering (default: 0, no limit)."),
    )
    parser.add_argument(
        "--max-megapixels",
        type=float,
        default=1.0,
        help=(
            "Maximum output resolution in megapixels when upsampling depth "
            "(default: 1.0). Images exceeding this will be scaled down proportionally."
        ),
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Output image format (default: jpg).",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Save point clouds as PLY files for the reference frames (default: off).",
    )
    parser.add_argument(
        "--filter-sky",
        action="store_true",
        help="Filter sky points using segmentation model (requires onnxruntime, default: off).",
    )
    parser.add_argument(
        "--filter-black-bg",
        action="store_true",
        help="Filter black background points (RGB sum < 16, default: off).",
    )
    parser.add_argument(
        "--filter-white-bg",
        action="store_true",
        help="Filter white background points (RGB > 240, default: off).",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Generate bidirectional pairs (A->B and B->A) for each consecutive frame pair (default: off).",
    )
    return parser.parse_args()


def write_ply(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    confidences: np.ndarray | None = None,
) -> None:
    """Write point cloud to binary PLY file.
    
    Args:
        output_path: Path to output PLY file
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-1 range)
        confidences: Optional Nx1 array of confidence values
    """
    num_points = points.shape[0]
    
    # Convert colors from 0-1 to 0-255 range
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # Write header as text
    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {num_points}\n"
    header += "property float x\n"
    header += "property float y\n"
    header += "property float z\n"
    header += "property uchar red\n"
    header += "property uchar green\n"
    header += "property uchar blue\n"
    if confidences is not None:
        header += "property float confidence\n"
    header += "end_header\n"
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(header.encode('ascii'))
        
        # ⚡ Bolt: Bulk binary writing with NumPy structured array
        # This is significantly faster than a Python loop with struct.pack
        # for large point clouds (e.g. 1M points).
        dtype = [
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        if confidences is not None:
            dtype.append(('confidence', '<f4'))

        vertex_data = np.empty(num_points, dtype=dtype)
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['red'] = colors_uint8[:, 0]
        vertex_data['green'] = colors_uint8[:, 1]
        vertex_data['blue'] = colors_uint8[:, 2]
        if confidences is not None:
            # Handle both (N,) and (N, 1) confidence arrays
            vertex_data['confidence'] = confidences.ravel()

        f.write(vertex_data.tobytes())


def apply_sky_filter(
    conf_frame: np.ndarray,
    image_path: Path,
    skyseg_session,
    sky_masks_dir: Path,
) -> np.ndarray:
    """Apply sky segmentation to filter confidence scores."""
    if not SKY_FILTER_AVAILABLE:
        print("Warning: Sky filtering requires opencv-python and onnxruntime. Skipping.")
        return conf_frame
    
    sky_masks_dir.mkdir(parents=True, exist_ok=True)
    image_name = image_path.name
    mask_path = sky_masks_dir / image_name
    
    if mask_path.exists():
        sky_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        sky_mask = segment_sky(str(image_path), skyseg_session, str(mask_path))
    
    # Resize mask to match confidence map if needed
    H, W = conf_frame.shape
    if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
        sky_mask = cv2.resize(sky_mask, (W, H))
    
    # Apply mask (segment_sky returns 255 for non-sky, 0 for sky)
    sky_mask_binary = (sky_mask > 128).astype(np.float32)
    return conf_frame * sky_mask_binary


def apply_background_filters(
    colors: np.ndarray,
    filter_black: bool,
    filter_white: bool,
) -> np.ndarray:
    """Apply black and/or white background filtering.
    
    Args:
        colors: Nx3 array of RGB colors (0-1 range)
        filter_black: Filter black background
        filter_white: Filter white background
    
    Returns:
        Boolean mask of valid points
    """
    mask = np.ones(colors.shape[0], dtype=bool)
    if not (filter_black or filter_white):
        return mask

    # ⚡ Bolt: Use float comparisons directly to avoid expensive uint8 conversions
    # and redundant computations for black/white filtering.
    if filter_black:
        # RGB sum < 16/255 approx 0.0627 in 0-1 float range
        mask &= (colors.sum(axis=1) >= (16 / 255.0))
    
    if filter_white:
        # ⚡ Bolt: floor(c * 255) > 240  <=>  c * 255 >= 241  <=>  c >= 241/255
        # This avoids creating an expensive floor copy and multiple mask arrays.
        threshold = 241.0 / 255.0
        mask &= ~((colors[:, 0] >= threshold) & (colors[:, 1] >= threshold) & (colors[:, 2] >= threshold))
    
    return mask


def sorted_image_paths(input_dir: Path, skip_every: int) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if skip_every < 1:
        raise ValueError("skip_every must be >= 1")
    image_paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if skip_every > 1:
        image_paths = image_paths[::skip_every]
    if len(image_paths) < 2:
        raise ValueError(f"Need at least two images after skipping in {input_dir}.")
    return image_paths


def rescale_image_to_max_megapixels(
    path: Path,
    max_megapixels: float,
    temp_dir: Path,
) -> Path:
    """Rescale image if it exceeds max megapixels, saving to temporary directory."""
    if max_megapixels <= 0:
        return path
    with Image.open(path) as img:
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        width, height = img.size
        megapixels = (width * height) / 1_000_000
        if megapixels <= max_megapixels:
            return path
        scale = (max_megapixels / megapixels) ** 0.5
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if new_width == width and new_height == height:
            return path
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{path.stem}_rescaled{path.suffix}"
        resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        resized.save(temp_path)
        return temp_path


def rescale_scene_images_to_max_megapixels(
    image_paths: list[Path],
    max_megapixels: float,
    temp_dir: Path,
) -> tuple[list[Path], str | None]:
    """Rescale all images in a scene to max megapixels limit."""
    if max_megapixels <= 0:
        return image_paths, None
    resized_paths: list[Path] = []
    resized_count = 0
    for path in image_paths:
        resized_path = rescale_image_to_max_megapixels(
            path, max_megapixels, temp_dir
        )
        if resized_path != path:
            resized_count += 1
        resized_paths.append(resized_path)
    note = None
    if resized_count > 0:
        note = (
            f"Rescaled {resized_count}/{len(image_paths)} images to "
            f"<= {max_megapixels:.2f}MP (in-memory)"
        )
    return resized_paths, note


def resolve_images_dir(scene_dir: Path) -> Path | None:
    for candidate in sorted(scene_dir.iterdir()):
        if candidate.is_dir() and candidate.name.startswith("images"):
            return candidate
    return None


def load_transforms(scene_dir: Path) -> tuple[list[dict], dict[str, float]]:
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.exists():
        return [], {}
    with transforms_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        frames = []
    metadata = {}
    for key in ("camera_angle_x", "camera_angle_y", "fl_x", "fl_y", "w", "h"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            metadata[key] = float(value)
    return frames, metadata


def build_frame_paths(scene_dir: Path, frames: list[dict]) -> list[tuple[dict, Path]]:
    images_dir = resolve_images_dir(scene_dir)
    frame_paths: list[tuple[dict, Path]] = []
    for frame in frames:
        file_path = frame.get("file_path")
        if not isinstance(file_path, str):
            continue
        filename = Path(file_path).name
        if images_dir is not None:
            candidate = images_dir / filename
        else:
            candidate = scene_dir / file_path
        if not candidate.exists() and images_dir is not None:
            fallback = scene_dir / file_path
            if fallback.exists():
                candidate = fallback
        if candidate.exists() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            frame_paths.append((frame, candidate))
    return frame_paths


def compute_overlap_scores(
    frames: list[dict], metadata: dict[str, float]
) -> np.ndarray:
    if len(frames) < 2:
        return np.array([], dtype=np.float64)
    positions = []
    rotations = []
    for frame in frames:
        matrix = frame.get("transform_matrix")
        if matrix is None:
            continue
        transform = np.array(matrix, dtype=np.float64)
        if transform.shape != (4, 4):
            continue
        rotations.append(transform[:3, :3])
        positions.append(transform[:3, 3])
    if len(positions) < 2:
        return np.array([], dtype=np.float64)

    positions_arr = np.stack(positions, axis=0)
    dists = np.linalg.norm(np.diff(positions_arr, axis=0), axis=1)
    angles = []
    for idx in range(len(rotations) - 1):
        r_rel = rotations[idx].T @ rotations[idx + 1]
        trace = np.trace(r_rel)
        cos_angle = (trace - 1.0) * 0.5
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles.append(float(np.arccos(cos_angle)))
    if not angles:
        return np.array([], dtype=np.float64)

    angles_arr = np.array(angles, dtype=np.float64)
    eps = 1e-6
    default_fov = np.deg2rad(60.0)
    fov_candidates = []
    if "camera_angle_x" in metadata:
        fov_candidates.append(metadata["camera_angle_x"])
    if "camera_angle_y" in metadata:
        fov_candidates.append(metadata["camera_angle_y"])
    if "fl_x" in metadata and "w" in metadata and metadata["fl_x"] > eps:
        fov_candidates.append(2.0 * np.arctan(metadata["w"] / (2.0 * metadata["fl_x"])))
    if "fl_y" in metadata and "h" in metadata and metadata["fl_y"] > eps:
        fov_candidates.append(2.0 * np.arctan(metadata["h"] / (2.0 * metadata["fl_y"])))
    fov_ref = min(fov_candidates) if fov_candidates else default_fov
    fov_ref = max(float(fov_ref), eps)

    center = positions_arr.mean(axis=0)
    depth_candidates = []
    dist_to_center = np.linalg.norm(positions_arr - center, axis=1)
    if dist_to_center.size > 0:
        depth_candidates.append(float(np.median(dist_to_center)))
    if dists.size > 0:
        depth_candidates.append(float(np.median(dists)))
    depth_ref = max(max(depth_candidates, default=1.0), eps)

    tan_half_fov = np.tan(fov_ref * 0.5)
    trans_scale = 2.0 * depth_ref * tan_half_fov
    overlaps = []
    for dist, angle in zip(dists, angles_arr):
        rot_overlap = max(0.0, 1.0 - float(angle) / fov_ref)
        if trans_scale > eps:
            trans_overlap = max(0.0, 1.0 - float(dist) / trans_scale)
        else:
            trans_overlap = 0.0
        overlaps.append(min(rot_overlap, trans_overlap))
    return np.array(overlaps, dtype=np.float64)


def select_frame_indices_by_overlap(
    overlaps: np.ndarray, target_overlap: float
) -> list[int]:
    if overlaps.size == 0:
        return [0]
    target_overlap = float(np.clip(target_overlap, 0.0, 1.0))
    target_loss = 1.0 - target_overlap
    selected = [0]
    accumulated = 0.0
    for idx, overlap in enumerate(overlaps, start=1):
        accumulated += 1.0 - float(np.clip(overlap, 0.0, 1.0))
        if accumulated >= target_loss:
            selected.append(idx)
            accumulated = 0.0
    if selected[-1] != len(overlaps):
        selected.append(len(overlaps))
    return selected


def scene_image_paths(
    scene_dir: Path,
    skip_every: int,
    auto_skip: bool,
    target_overlap: float,
    limit: int,
) -> tuple[list[Path], str | None]:
    frames, metadata = load_transforms(scene_dir) if auto_skip else ([], {})
    frame_paths = build_frame_paths(scene_dir, frames) if frames else []
    if frame_paths:
        image_paths = [path for _, path in frame_paths]
        auto_skip_note = None
        if auto_skip:
            overlaps = compute_overlap_scores(
                [frame for frame, _ in frame_paths], metadata
            )
            indices = select_frame_indices_by_overlap(overlaps, target_overlap)
            image_paths = [image_paths[idx] for idx in indices]
            auto_skip_note = (
                f"auto-selected {len(image_paths)} of {len(frame_paths)} frames"
            )
        elif skip_every > 1:
            image_paths = image_paths[::skip_every]
        if limit > 0:
            image_paths = image_paths[:limit]
        if len(image_paths) < 2:
            raise ValueError(f"Need at least two images after skipping in {scene_dir}.")
        return image_paths, auto_skip_note

    image_paths = sorted_image_paths(scene_dir, skip_every)
    if limit > 0:
        image_paths = image_paths[:limit]
    return image_paths, None


def list_scene_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    scene_dirs = [path for path in sorted(input_dir.iterdir()) if path.is_dir()]
    if not scene_dirs:
        raise ValueError(f"No subdirectories found in {input_dir}.")
    return scene_dirs


def load_model(device: torch.device) -> VGGT:
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model


def build_preprocess_metadata(
    image_paths: list[Path],
    mode: str,
    target_size: int,
    model_height: int,
    model_width: int,
) -> list[dict[str, float]]:
    metas: list[dict[str, float]] = []
    effective_sizes: list[tuple[int, int]] = []

    for path in image_paths:
        with Image.open(path) as img:
            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            img = img.convert("RGB")
            width, height = img.size

        if mode == "pad":
            if width >= height:
                resized_width = target_size
                resized_height = round(height * (resized_width / width) / 14) * 14
            else:
                resized_height = target_size
                resized_width = round(width * (resized_height / height) / 14) * 14

            scale_x = resized_width / width
            scale_y = resized_height / height
            pad_left = (target_size - resized_width) // 2
            pad_top = (target_size - resized_height) // 2
            crop_top = 0
            # Content dimensions (before padding to target_size)
            effective_width = resized_width
            effective_height = resized_height
            # In pad mode, all images are padded to target_size x target_size,
            # so model space dimensions equal target_size
            model_space_height = target_size
            model_space_width = target_size
        else:
            resized_width = target_size
            resized_height = round(height * (resized_width / width) / 14) * 14
            scale_x = resized_width / width
            scale_y = resized_height / height
            crop_top = max((resized_height - target_size) // 2, 0)
            pad_left = 0
            pad_top = 0
            effective_width = resized_width
            effective_height = min(resized_height, target_size)
            # In crop mode, model space dimensions equal effective dimensions
            model_space_height = effective_height
            model_space_width = effective_width

        effective_sizes.append((model_space_height, model_space_width))
        metas.append(
            {
                "orig_width": width,
                "orig_height": height,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "pad_left": pad_left,
                "pad_top": pad_top,
                "crop_top": crop_top,
                "effective_width": effective_width,
                "effective_height": effective_height,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "model_space_width": model_space_width,
                "model_space_height": model_space_height,
            }
        )

    max_height = max(model_height, max(h for h, _ in effective_sizes))
    max_width = max(model_width, max(w for _, w in effective_sizes))

    for meta in metas:
        extra_pad_top = (max_height - meta["model_space_height"]) // 2
        extra_pad_left = (max_width - meta["model_space_width"]) // 2
        meta["total_pad_top"] = meta["pad_top"] + extra_pad_top
        meta["total_pad_left"] = meta["pad_left"] + extra_pad_left
        meta["model_height"] = max_height
        meta["model_width"] = max_width

    return metas


def restore_to_original_resolution(
    model_rgb: np.ndarray,
    meta: dict[str, float],
    mode: str,
) -> np.ndarray:
    image = Image.fromarray(model_rgb)
    left = int(meta["total_pad_left"])
    top = int(meta["total_pad_top"])
    right = left + int(meta["effective_width"])
    bottom = top + int(meta["effective_height"])
    image = image.crop((left, top, right, bottom))

    if mode == "crop":
        resized_width = int(meta["resized_width"])
        resized_height = int(meta["resized_height"])
        canvas = Image.new("RGB", (resized_width, resized_height), (0, 0, 0))
        canvas.paste(image, (0, int(meta["crop_top"])))
        image = canvas

    image = image.resize(
        (int(meta["orig_width"]), int(meta["orig_height"])), Image.Resampling.BICUBIC
    )
    return np.array(image)


def restore_map_to_original_resolution(
    model_map: np.ndarray,
    meta: dict[str, float],
    mode: str,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    map_tensor = torch.from_numpy(model_map).unsqueeze(0).unsqueeze(0)
    left = int(meta["total_pad_left"])
    top = int(meta["total_pad_top"])
    right = left + int(meta["effective_width"])
    bottom = top + int(meta["effective_height"])
    map_tensor = map_tensor[:, :, top:bottom, left:right]

    if mode == "crop":
        resized_width = int(meta["resized_width"])
        resized_height = int(meta["resized_height"])
        canvas = torch.full(
            (1, 1, resized_height, resized_width),
            fill_value,
            dtype=map_tensor.dtype,
        )
        crop_top = int(meta["crop_top"])
        cropped_height = map_tensor.shape[2]
        cropped_width = map_tensor.shape[3]
        canvas[:, :, crop_top : crop_top + cropped_height, :cropped_width] = map_tensor
        map_tensor = canvas

    map_tensor = torch_nn.interpolate(
        map_tensor,
        size=(int(meta["orig_height"]), int(meta["orig_width"])),
        mode="bilinear",
        align_corners=False,
    )
    return map_tensor.squeeze(0).squeeze(0).numpy()


def resize_map_to_output(
    map_data: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
    output_width, output_height = output_size
    if map_data.shape == (output_height, output_width):
        return map_data
    map_tensor = torch.from_numpy(map_data).unsqueeze(0).unsqueeze(0)
    map_tensor = torch_nn.interpolate(
        map_tensor,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )
    return map_tensor.squeeze(0).squeeze(0).numpy()


def build_view_matrix(extrinsic: np.ndarray) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = extrinsic[:3, :3]
    view[:3, 3] = extrinsic[:3, 3]
    conversion = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return conversion @ view


def ensure_depth_channel(depth_map: np.ndarray) -> np.ndarray:
    if depth_map.ndim == 3:
        return depth_map[..., None]
    return depth_map


def estimate_s0_from_depth(
    depth_map: np.ndarray, intrinsic: np.ndarray, eps: float = 1e-6
) -> float:
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(-1)
    valid = depth_map > eps
    if not np.any(valid):
        return 0.0

    # Bolt: Only compute spacing for valid points to avoid unnecessary calculations
    # on the entire depth map.
    valid_depths = depth_map[valid]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    dx = valid_depths / max(fx, eps)
    dy = valid_depths / max(fy, eps)
    spacing = np.sqrt(dx * dx + dy * dy)
    return float(np.median(spacing))


def build_projection_matrix(
    intrinsic: np.ndarray,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 1000.0,
) -> np.ndarray:
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height
    proj[0, 2] = 2.0 * cx / width - 1.0
    proj[1, 2] = 1.0 - 2.0 * cy / height
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0
    return proj


def get_triplet_paths(
    scene_output_dir: Path,
    stem: str,
    output_ext: str,
) -> dict[str, Path]:
    """Generate paths for all output files in a triplet.

    Args:
        scene_output_dir: Path to scene output directory
        stem: Base name for the target frame
        output_ext: Extension for image files (jpg, png, etc.)

    Returns:
        Dictionary mapping file types to Path objects
    """
    return {
        "splats": scene_output_dir / f"{stem}_splats.{output_ext}",
        "target": scene_output_dir / f"{stem}_target.{output_ext}",
        "reference": scene_output_dir / f"{stem}_reference.{output_ext}",
        "confidence": scene_output_dir / f"{stem}_confidence.png",
        "ply": scene_output_dir / f"{stem}_reference.ply",
    }


def check_scene_needs_processing(
    scene_dir: Path,
    output_dir: Path,
    output_ext: str,
    bidirectional: bool,
    save_confidence: bool,
    save_ply: bool,
    image_paths: list[Path],
) -> bool:
    """Check if this scene needs any processing.
    
    Returns True if any pair is missing files, False if all pairs complete.
    """
    if len(image_paths) < 2:
        return False
    
    scene_output_dir = output_dir / scene_dir.name
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in range(len(image_paths) - 1):
        next_idx = idx + 1
        next_name = image_paths[next_idx].stem
        curr_name = image_paths[idx].stem
        
        # Check both original and rescaled versions of the filename
        def files_exist_for_name(name: str) -> bool:
            for stem in [name, f"{name}_rescaled"]:
                paths = get_triplet_paths(scene_output_dir, stem, output_ext)
                files_ok = (
                    paths["splats"].exists() and
                    paths["target"].exists() and
                    paths["reference"].exists() and
                    (not save_confidence or paths["confidence"].exists()) and
                    (not save_ply or paths["ply"].exists())
                )
                if files_ok:
                    return True
            return False
        
        # Check forward direction files
        if not files_exist_for_name(next_name):
            return True  # Forward direction needs work
        
        # Check reverse direction if bidirectional
        if bidirectional:
            if not files_exist_for_name(curr_name):
                return True  # Reverse direction needs work
    
    return False  # All pairs complete


def intrinsic_for_output(
    model_intrinsic: np.ndarray,
    meta: dict[str, float],
    output_size: tuple[int, int] | None,
) -> np.ndarray:
    if output_size is None:
        return model_intrinsic
    offset_x = float(meta["total_pad_left"])
    offset_y = float(meta["total_pad_top"]) - float(meta["crop_top"])
    scale_x = float(meta["scale_x"])
    scale_y = float(meta["scale_y"])

    fx_model = float(model_intrinsic[0, 0])
    fy_model = float(model_intrinsic[1, 1])
    cx_model = float(model_intrinsic[0, 2])
    cy_model = float(model_intrinsic[1, 2])

    fx_orig = fx_model / scale_x
    fy_orig = fy_model / scale_y
    cx_orig = (cx_model - offset_x) / scale_x
    cy_orig = (cy_model - offset_y) / scale_y

    output_width, output_height = output_size
    scale_out_x = output_width / float(meta["orig_width"])
    scale_out_y = output_height / float(meta["orig_height"])

    intrinsic_out = np.array(model_intrinsic, copy=True)
    intrinsic_out[0, 0] = fx_orig * scale_out_x
    intrinsic_out[1, 1] = fy_orig * scale_out_y
    intrinsic_out[0, 2] = cx_orig * scale_out_x
    intrinsic_out[1, 2] = cy_orig * scale_out_y
    return intrinsic_out


def render_and_save_pair(
    source_idx: int,
    target_idx: int,
    image_paths: list[Path],
    scene_output_dir: Path,
    output_ext: str,
    save_kwargs: dict,
    args: argparse.Namespace,
    depth: np.ndarray,
    depth_conf: np.ndarray,
    world_points: np.ndarray,
    images_np: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    preprocess_metas: list[dict],
    renderer: HoleFillingRenderer,
    render_width: int,
    render_height: int,
    resize_size: tuple[int, int] | None,
    skyseg_session,
) -> None:
    """Render and save a single image pair (source -> target)."""
    target_name = image_paths[target_idx].stem
    paths = get_triplet_paths(scene_output_dir, target_name, output_ext)

    splats_path = paths["splats"]
    target_path = paths["target"]
    reference_path = paths["reference"]
    conf_path = paths["confidence"]
    ply_path = paths["ply"]

    missing = (
        not splats_path.exists()
        or not target_path.exists()
        or not reference_path.exists()
        or (args.save_confidence and not conf_path.exists())
        or (args.save_ply and not ply_path.exists())
    )

    if not missing:
        print(
            f"Skipped {scene_output_dir.name}/{target_name}_* (direction already complete)"
        )
        return

    depth_frame = depth[source_idx]
    conf_frame = depth_conf[source_idx]
    if depth_frame.ndim == 3:
        depth_frame = depth_frame.squeeze(-1)
    if conf_frame.ndim == 3:
        conf_frame = conf_frame.squeeze(-1)

    # Apply sky filtering if enabled
    if args.filter_sky and skyseg_session is not None:
        sky_masks_dir = scene_output_dir / "sky_masks"
        conf_frame = apply_sky_filter(
            conf_frame, image_paths[source_idx], skyseg_session, sky_masks_dir
        )

    valid_mask = depth_frame > 1e-6

    points = world_points[source_idx][valid_mask]
    colors = images_np[source_idx][valid_mask]
    confidences = conf_frame[valid_mask]

    # Apply background filtering if enabled
    if args.filter_black_bg or args.filter_white_bg:
        bg_mask = apply_background_filters(
            colors, args.filter_black_bg, args.filter_white_bg
        )
        points = points[bg_mask]
        colors = colors[bg_mask]
        confidences = confidences[bg_mask]

    if args.auto_s0:
        s0 = estimate_s0_from_depth(depth_frame, intrinsic[source_idx])
        if s0 > 0.0:
            renderer.s0 = s0

    view_mat = build_view_matrix(extrinsic[target_idx])
    if args.upsample_depth:
        target_intrinsic = intrinsic[target_idx]
    else:
        target_intrinsic = intrinsic_for_output(
            intrinsic[target_idx], preprocess_metas[target_idx], resize_size
        )
    proj_mat = build_projection_matrix(target_intrinsic, render_width, render_height)
    fov_y = 2.0 * np.arctan(0.5 * render_height / target_intrinsic[1, 1])

    model_render = renderer.render(
        points, colors, confidences, view_mat, proj_mat, fov_y
    )
    if args.upsample_depth or resize_size is not None:
        splats_image = model_render
    else:
        splats_image = restore_to_original_resolution(
            model_render, preprocess_metas[target_idx], args.preprocess_mode
        )

    if not splats_path.exists():
        Image.fromarray(splats_image).save(splats_path, **save_kwargs)

    if args.save_confidence and not conf_path.exists():
        # Save confidence map for the source frame (source_idx) as grayscale
        conf_for_save = conf_frame.copy()
        # Normalize to 0-255 range
        conf_min = conf_for_save.min()
        conf_max = conf_for_save.max()
        if conf_max > conf_min:
            conf_normalized = (conf_for_save - conf_min) / (conf_max - conf_min)
        else:
            conf_normalized = np.zeros_like(conf_for_save)
        conf_uint8 = (conf_normalized * 255).astype(np.uint8)
        if not args.upsample_depth and resize_size is None:
            # Restore to original resolution
            conf_image = Image.fromarray(conf_uint8, mode="L")
            meta = preprocess_metas[source_idx]
            left = int(meta["total_pad_left"])
            top = int(meta["total_pad_top"])
            right = left + int(meta["effective_width"])
            bottom = top + int(meta["effective_height"])
            conf_image = conf_image.crop((left, top, right, bottom))
            conf_image = conf_image.resize(
                (int(meta["orig_width"]), int(meta["orig_height"])),
                Image.Resampling.BICUBIC,
            )
        else:
            conf_image = Image.fromarray(conf_uint8, mode="L")
            if resize_size is not None:
                conf_image = conf_image.resize(resize_size, Image.Resampling.BICUBIC)
        # Always save confidence as PNG for lossless grayscale
        conf_image.save(conf_path)

    if not target_path.exists() or not reference_path.exists():
        target_img_obj = Image.open(image_paths[target_idx])
        reference_img_obj = Image.open(image_paths[source_idx])
        try:
            target_img_obj = target_img_obj.convert("RGB")
            reference_img_obj = reference_img_obj.convert("RGB")
            if resize_size is not None:
                target_img_obj = target_img_obj.resize(
                    resize_size, Image.Resampling.BICUBIC
                )
                reference_img_obj = reference_img_obj.resize(
                    resize_size, Image.Resampling.BICUBIC
                )
            if not target_path.exists():
                target_img_obj.save(target_path, **save_kwargs)
            if not reference_path.exists():
                reference_img_obj.save(reference_path, **save_kwargs)
        finally:
            target_img_obj.close()
            reference_img_obj.close()

    # Save PLY file if requested
    if args.save_ply and not ply_path.exists():
        write_ply(
            ply_path, points, colors, confidences if args.save_confidence else None
        )

    ply_note = f" and {target_name}_reference.ply" if args.save_ply else ""
    print(
        f"Wrote {scene_output_dir.name}/{splats_path.name}, {target_path.name}, "
        f"and {reference_path.name}{ply_note}"
    )


def process_scene(
    scene_dir: Path,
    args: argparse.Namespace,
    model: VGGT,
    device: torch.device,
    dtype: torch.dtype,
    renderer_info: dict,
    skyseg_session,
    output_dir: Path,
    resize_size: tuple[int, int] | None,
) -> None:
    """Process all image pairs in a single scene."""
    try:
        image_paths, auto_skip_note = scene_image_paths(
            scene_dir,
            args.skip_every,
            args.auto_skip,
            args.target_overlap,
            args.limit,
        )
    except ValueError as exc:
        print(f"Skipping {scene_dir}: {exc}")
        return

    output_ext = "jpg" if args.output_format in ["jpg", "jpeg"] else args.output_format

    scene_output_dir = output_dir / scene_dir.name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    # Check upfront if this scene needs any processing (before rescaling to save time)
    scene_needs_work = check_scene_needs_processing(
        scene_dir,
        output_dir,
        output_ext,
        args.bidirectional,
        args.save_confidence,
        args.save_ply,
        image_paths,
    )

    if not scene_needs_work:
        print(f"Skipping {scene_dir.name}: all triplets already complete")
        return

    print(f"Processing {scene_dir.name}...")

    if auto_skip_note is not None:
        print(
            f"Auto skip for {scene_dir.name}: {auto_skip_note} "
            f"(target overlap {args.target_overlap})."
        )

    # Create temporary directory for rescaled images (cleaned up later)
    temp_resized_dir = None
    if args.max_megapixels > 0:
        temp_resized_dir = Path(tempfile.mkdtemp(prefix=f"rescale_{scene_dir.name}_"))
        image_paths, resize_note = rescale_scene_images_to_max_megapixels(
            image_paths, args.max_megapixels, temp_resized_dir
        )
        if resize_note is not None:
            print(f"{scene_dir.name}: {resize_note}")

    try:
        print(f"Loading {len(image_paths)} images from {scene_dir}...")
        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=args.preprocess_mode,
        )
        images = images.to(device)
        model_height, model_width = images.shape[-2:]

        preprocess_metas = build_preprocess_metadata(
            image_paths,
            args.preprocess_mode,
            target_size=518,
            model_height=model_height,
            model_width=model_width,
        )

        if args.upsample_depth:
            if resize_size is not None:
                output_size = resize_size
            else:
                output_size = (
                    int(preprocess_metas[0]["orig_width"]),
                    int(preprocess_metas[0]["orig_height"]),
                )
                # Check if all images have the same resolution
                mismatched_resolutions = False
                for meta in preprocess_metas[1:]:
                    if (
                        int(meta["orig_width"]) != output_size[0]
                        or int(meta["orig_height"]) != output_size[1]
                    ):
                        mismatched_resolutions = True
                        break

                # If resolutions don't match, use the minimum dimensions to avoid upscaling
                if mismatched_resolutions:
                    print(
                        f"Images in {scene_dir.name} have different resolutions. "
                        f"Using minimum dimensions for output size."
                    )
                    min_width = min(
                        int(meta["orig_width"]) for meta in preprocess_metas
                    )
                    min_height = min(
                        int(meta["orig_height"]) for meta in preprocess_metas
                    )
                    output_size = (min_width, min_height)

            # Apply megapixel limit
            width, height = output_size
            megapixels = (width * height) / 1_000_000
            if megapixels > args.max_megapixels:
                scale = (args.max_megapixels / megapixels) ** 0.5
                width = int(width * scale)
                height = int(height * scale)
                output_size = (width, height)
                print(
                    f"Scaling output from {megapixels:.2f}MP to {args.max_megapixels:.2f}MP: "
                    f"{output_size[0]}x{output_size[1]}"
                )

            render_width, render_height = output_size
        else:
            output_size = None
            render_width = resize_size[0] if resize_size is not None else model_width
            render_height = resize_size[1] if resize_size is not None else model_height

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype, enabled=device.type == "cuda"):
                predictions = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        if extrinsic is None or intrinsic is None:
            raise ValueError("Camera predictions are missing from VGGT output.")

        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        if depth is None or depth_conf is None:
            raise ValueError("Depth predictions are missing from VGGT output.")
        depth = depth.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()

        if args.upsample_depth:
            if output_size is None:
                raise ValueError("Output size is required when upsampling depth.")
            images_np = []
            for path in image_paths:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if img.size != output_size:
                        img = img.resize(output_size, Image.Resampling.BICUBIC)
                    images_np.append(np.array(img, dtype=np.float32) / 255.0)
            images_np = np.stack(images_np, axis=0)

            depth_frames = []
            conf_frames = []
            intrinsic_frames = []
            for idx, meta in enumerate(preprocess_metas):
                depth_frame_map = depth[idx]
                if depth_frame_map.ndim == 3:
                    depth_frame_map = depth_frame_map.squeeze(-1)
                conf_frame_map = depth_conf[idx]
                if conf_frame_map.ndim == 3:
                    conf_frame_map = conf_frame_map.squeeze(-1)

                depth_frame_map = restore_map_to_original_resolution(
                    depth_frame_map, meta, args.preprocess_mode
                )
                conf_frame_map = restore_map_to_original_resolution(
                    conf_frame_map, meta, args.preprocess_mode, fill_value=0.0
                )
                depth_frame_map = resize_map_to_output(depth_frame_map, output_size)
                conf_frame_map = resize_map_to_output(conf_frame_map, output_size)
                depth_frames.append(depth_frame_map)
                conf_frames.append(conf_frame_map)
                intrinsic_frames.append(
                    intrinsic_for_output(intrinsic[idx], meta, output_size)
                )

            depth = np.stack(depth_frames, axis=0)
            depth_conf = np.stack(conf_frames, axis=0)
            intrinsic = np.stack(intrinsic_frames, axis=0)
            depth_for_unproject = ensure_depth_channel(depth)
        else:
            images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
            depth_for_unproject = ensure_depth_channel(depth)

        # Renderer management
        renderer = renderer_info["renderer"]
        renderer_size = renderer_info["renderer_size"]
        shaders_dir = renderer_info["shaders_dir"]

        if renderer is None or renderer_size != (render_height, render_width):
            renderer = HoleFillingRenderer(
                render_width,
                render_height,
                shaders_dir=shaders_dir,
                confidence_threshold=args.depth_conf_threshold,
                jfa_mask_sigma=args.sigma,
            )
            renderer_info["renderer"] = renderer
            renderer_info["renderer_size"] = (render_height, render_width)
        else:
            renderer.confidence_threshold = args.depth_conf_threshold

        world_points = unproject_depth_map_to_point_map(
            depth_for_unproject, extrinsic, intrinsic
        )

        save_kwargs = {"quality": 95, "optimize": True} if output_ext == "jpg" else {}

        for idx in range(len(image_paths) - 1):
            next_idx = idx + 1

            # Forward: idx -> next_idx
            render_and_save_pair(
                idx,
                next_idx,
                image_paths,
                scene_output_dir,
                output_ext,
                save_kwargs,
                args,
                depth,
                depth_conf,
                world_points,
                images_np,
                extrinsic,
                intrinsic,
                preprocess_metas,
                renderer,
                render_width,
                render_height,
                resize_size,
                skyseg_session,
            )

            # Reverse: next_idx -> idx
            if args.bidirectional:
                render_and_save_pair(
                    next_idx,
                    idx,
                    image_paths,
                    scene_output_dir,
                    output_ext,
                    save_kwargs,
                    args,
                    depth,
                    depth_conf,
                    world_points,
                    images_np,
                    extrinsic,
                    intrinsic,
                    preprocess_metas,
                    renderer,
                    render_width,
                    render_height,
                    resize_size,
                    skyseg_session,
                )

        # Explicit cleanup
        del images
        del predictions
        del depth
        del depth_for_unproject
        del depth_conf
        del extrinsic
        del intrinsic
        del images_np
        del world_points
        del preprocess_metas
        if device.type == "cuda":
            torch.cuda.empty_cache()

    finally:
        # Clean up temporary rescaled images
        if temp_resized_dir is not None and temp_resized_dir.exists():
            shutil.rmtree(temp_resized_dir)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if (args.resize_width == 0) != (args.resize_height == 0):
        raise ValueError("Both resize width and height must be set or both 0.")
    resize_size = None
    if args.resize_width > 0 and args.resize_height > 0:
        resize_size = (args.resize_width, args.resize_height)
        resize_megapixels = (resize_size[0] * resize_size[1]) / 1_000_000
        if args.max_megapixels > 0 and resize_megapixels > args.max_megapixels:
            scale = (args.max_megapixels / resize_megapixels) ** 0.5
            resize_size = (
                max(1, int(round(resize_size[0] * scale))),
                max(1, int(round(resize_size[1] * scale))),
            )
            print(
                f"Rescaling resize-size to honor max megapixels: {resize_size[0]}x{resize_size[1]}"
            )

    scene_dirs = list_scene_dirs(input_dir)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    model = load_model(device)
    renderer_info = {
        "renderer": None,
        "renderer_size": None,
        "shaders_dir": Path(__file__).resolve().parent / "shaders",
    }

    # Initialize sky segmentation if needed
    skyseg_session = None
    if args.filter_sky:
        if not SKY_FILTER_AVAILABLE:
            print(
                "Warning: --filter-sky requires opencv-python and onnxruntime. Install with:"
            )
            print("  uv pip install opencv-python onnxruntime")
            print("Sky filtering will be disabled.")
        else:
            skyseg_path = Path("skyseg.onnx")
            if not skyseg_path.exists():
                print("Downloading sky segmentation model...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                    "skyseg.onnx",
                )
            skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")

    for scene_dir in scene_dirs:
        process_scene(
            scene_dir,
            args,
            model,
            device,
            dtype,
            renderer_info,
            skyseg_session,
            output_dir,
            resize_size,
        )


if __name__ == "__main__":
    main()
