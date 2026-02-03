import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from hole_filling_renderer import HoleFillingRenderer


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
        default=2.0,
        help="Filter depth points with confidence below this value (default: 2.0).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=32.0,
        help="Sigma for fake Gaussian splatting (default: 32.0)",
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
            "camera motion (default: off)."
        ),
    )
    parser.add_argument(
        "--target-motion",
        type=float,
        default=45,
        help=(
            "Target normalized motion between selected frames when auto-skipping "
            "(default: 45)."
        ),
    )
    parser.add_argument(
        "--rotation-weight",
        type=float,
        default=2.0,
        help="Weight to apply to rotation motion (default: 2.0).",
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
        "--limit",
        type=int,
        default=0,
        help=("Limit to the first N images after filtering (default: 0, no limit)."),
    )
    return parser.parse_args()


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


def resolve_images_dir(scene_dir: Path) -> Path | None:
    for candidate in sorted(scene_dir.iterdir()):
        if candidate.is_dir() and candidate.name.startswith("images"):
            return candidate
    return None


def load_transforms(scene_dir: Path) -> list[dict]:
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.exists():
        return []
    with transforms_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        return []
    return frames


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


def compute_motion_scores(frames: list[dict], rotation_weight: float) -> np.ndarray:
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
    dist_ref = float(np.median(dists))
    angle_ref = float(np.median(angles_arr))
    eps = 1e-6
    if dist_ref < eps and angle_ref < eps:
        return np.zeros_like(angles_arr, dtype=np.float64)
    dist_scale = dist_ref if dist_ref >= eps else 1.0
    angle_scale = angle_ref if angle_ref >= eps else 1.0
    return np.sqrt(
        (dists / dist_scale) ** 2 + (rotation_weight * angles_arr / angle_scale) ** 2
    )


def select_frame_indices_by_motion(
    scores: np.ndarray, target_motion: float
) -> list[int]:
    if scores.size == 0:
        return [0]
    selected = [0]
    accumulated = 0.0
    for idx, score in enumerate(scores, start=1):
        accumulated += float(score)
        if accumulated >= target_motion:
            selected.append(idx)
            accumulated = 0.0
    if selected[-1] != len(scores):
        selected.append(len(scores))
    return selected


def scene_image_paths(
    scene_dir: Path,
    skip_every: int,
    auto_skip: bool,
    target_motion: float,
    rotation_weight: float,
    limit: int,
) -> tuple[list[Path], str | None]:
    frames = load_transforms(scene_dir) if auto_skip else []
    frame_paths = build_frame_paths(scene_dir, frames) if frames else []
    if frame_paths:
        image_paths = [path for _, path in frame_paths]
        auto_skip_note = None
        if auto_skip:
            scores = compute_motion_scores(
                [frame for frame, _ in frame_paths], rotation_weight
            )
            indices = select_frame_indices_by_motion(scores, target_motion)
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
            effective_width = resized_width
            effective_height = resized_height
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

        effective_sizes.append((effective_height, effective_width))
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
            }
        )

    max_height = max(model_height, max(h for h, _ in effective_sizes))
    max_width = max(model_width, max(w for _, w in effective_sizes))

    for meta in metas:
        extra_pad_top = (max_height - meta["effective_height"]) // 2
        extra_pad_left = (max_width - meta["effective_width"]) // 2
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


def build_view_matrix(extrinsic: np.ndarray) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = extrinsic[:3, :3]
    view[:3, 3] = extrinsic[:3, 3]
    conversion = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return conversion @ view


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

    scene_dirs = list_scene_dirs(input_dir)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    model = load_model(device)
    renderer: HoleFillingRenderer | None = None
    renderer_size: tuple[int, int] | None = None
    shaders_dir = Path(__file__).resolve().parent / "shaders"

    for scene_dir in scene_dirs:
        try:
            image_paths, auto_skip_note = scene_image_paths(
                scene_dir,
                args.skip_every,
                args.auto_skip,
                args.target_motion,
                args.rotation_weight,
                args.limit,
            )
        except ValueError as exc:
            print(f"Skipping {scene_dir}: {exc}")
            continue

        scene_output_dir = output_dir / scene_dir.name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        if auto_skip_note is not None:
            print(
                f"Auto skip for {scene_dir.name}: {auto_skip_note} "
                f"(target motion {args.target_motion})."
            )
        print(f"Loading {len(image_paths)} images from {scene_dir}...")
        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=args.preprocess_mode,
        )
        images = images.to(device)
        model_height, model_width = images.shape[-2:]
        render_width = resize_size[0] if resize_size is not None else model_width
        render_height = resize_size[1] if resize_size is not None else model_height

        if renderer is None:
            renderer = HoleFillingRenderer(
                render_width,
                render_height,
                shaders_dir=shaders_dir,
                confidence_threshold=args.depth_conf_threshold,
                jfa_mask_sigma=args.sigma,
            )
            renderer_size = (render_height, render_width)
        elif renderer_size != (render_height, render_width):
            raise ValueError(
                "All scenes must share the same model resolution to reuse the renderer. "
                f"Expected {renderer_size}, got {(render_height, render_width)} for {scene_dir}."
            )

        preprocess_metas = build_preprocess_metadata(
            image_paths,
            args.preprocess_mode,
            target_size=518,
            model_height=model_height,
            model_width=model_width,
        )

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
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)

        world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

        for idx in range(len(image_paths) - 1):
            next_idx = idx + 1
            next_name = image_paths[next_idx].stem

            depth_frame = depth[idx]
            if depth_frame.ndim == 3:
                depth_frame = depth_frame.squeeze(-1)
            conf_frame = depth_conf[idx]
            valid_mask = depth_frame > 1e-6

            points = world_points[idx][valid_mask]
            colors = images_np[idx][valid_mask]
            confidences = conf_frame[valid_mask]

            view_mat = build_view_matrix(extrinsic[next_idx])
            target_intrinsic = intrinsic_for_output(
                intrinsic[next_idx], preprocess_metas[next_idx], resize_size
            )
            proj_mat = build_projection_matrix(
                target_intrinsic, render_width, render_height
            )
            fov_y = 2.0 * np.arctan(0.5 * render_height / target_intrinsic[1, 1])

            model_render = renderer.render(
                points, colors, confidences, view_mat, proj_mat, fov_y
            )
            if resize_size is None:
                train_image = restore_to_original_resolution(
                    model_render, preprocess_metas[next_idx], args.preprocess_mode
                )
            else:
                train_image = model_render

            train_path = scene_output_dir / f"{next_name}_train.png"
            test_path = scene_output_dir / f"{next_name}_test.png"
            reference_path = scene_output_dir / f"{next_name}_reference.png"

            Image.fromarray(train_image).save(train_path)
            test_image = Image.open(image_paths[next_idx])
            reference_image = Image.open(image_paths[idx])
            try:
                test_image = test_image.convert("RGB")
                reference_image = reference_image.convert("RGB")
                if resize_size is not None:
                    test_image = test_image.resize(
                        resize_size, Image.Resampling.BICUBIC
                    )
                    reference_image = reference_image.resize(
                        resize_size, Image.Resampling.BICUBIC
                    )
                test_image.save(test_path)
                reference_image.save(reference_path)
            finally:
                test_image.close()
                reference_image.close()

            print(
                f"Wrote {scene_dir.name}/{train_path.name}, {test_path.name}, "
                f"and {reference_path.name}"
            )

        del images
        del predictions
        del depth
        del depth_conf
        del extrinsic
        del intrinsic
        del images_np
        del world_points
        del preprocess_metas
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
