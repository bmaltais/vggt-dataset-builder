import argparse
import math
from pathlib import Path

import moderngl
import numpy as np
import torch
import torch.nn.functional as torch_nn
from PIL import Image
import sys
from pathlib import Path as _Path

# Ensure local `vggt/` package is importable when running the script directly
_repo_root = _Path(__file__).resolve().parent
_vggt_path = str(_repo_root / "vggt")
if _vggt_path not in sys.path:
    sys.path.insert(0, _vggt_path)

from hole_filling_renderer import HoleFillingRenderer
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT on one or more images and view the point cloud in OpenGL."
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="One or more input image paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write exported render image.",
    )
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default="crop",
        choices=["crop", "pad"],
        help="Preprocessing mode before VGGT inference (default: crop).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Force a device selection; defaults to cuda if available.",
    )
    parser.add_argument(
        "--conf-threshold",
        "--depth-conf-threshold",
        dest="conf_threshold",
        type=float,
        default=1.01,
        help="Filter depth points with confidence below this value (default: 1.01).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride when sampling pixels (default: 1).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="OpenGL point size (default: 1.5).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Resize depth/confidence/color maps to this width (default: 0).",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=0,
        help="Resize depth/confidence/color maps to this height (default: 0).",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=0,
        help="Initial viewer window width (default: map width).",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=0,
        help="Initial viewer window height (default: map height).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=20.0,
        help="Sigma for fake Gaussian splatting (default: 20.0).",
    )
    parser.add_argument(
        "--auto-s0",
        action="store_true",
        help="Estimate s0 from depth and intrinsics (default: off).",
    )
    parser.add_argument(
        "--s0",
        type=float,
        default=0.0,
        help="Override s0 manually when --auto-s0 is off (default: 0.0).",
    )
    parser.add_argument(
        "--occlusion-threshold",
        type=float,
        default=0.1,
        help="HPR occlusion threshold (default: 0.1).",
    )
    parser.add_argument(
        "--coarse-level",
        type=int,
        default=4,
        help="HPR coarse mip level (default: 4).",
    )
    return parser.parse_args()


def load_model(device: torch.device) -> VGGT:
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model


def run_vggt(image_paths: list[Path], device: torch.device, preprocess_mode: str):
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode=preprocess_mode
    )
    images = images.to(device)
    model_height, model_width = images.shape[-2:]
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    model = load_model(device)
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

    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4:
        raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")

    depth = depth.cpu().numpy()
    depth_conf = depth_conf.cpu().numpy()
    extrinsic = extrinsic.cpu().numpy()
    intrinsic = intrinsic.cpu().numpy()

    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth_conf.ndim == 5 and depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]
    if (
        depth_conf.ndim == 4
        and depth_conf.shape[0] == 1
        and depth_conf.shape[1] == depth.shape[0]
    ):
        depth_conf = depth_conf[0]
    if extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    if intrinsic.ndim == 4 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]

    colors = images.permute(0, 2, 3, 1).cpu().numpy()

    return depth, depth_conf, extrinsic, intrinsic, colors, (model_width, model_height)


def build_point_cloud_batch(
    depth: np.ndarray,
    depth_conf: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    colors: np.ndarray,
    conf_threshold: float,
    stride: int,
):
    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth_conf.ndim == 5 and depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]
    if (
        depth_conf.ndim == 4
        and depth_conf.shape[0] == 1
        and depth_conf.shape[1] == depth.shape[0]
    ):
        depth_conf = depth_conf[0]
    if extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    if intrinsic.ndim == 4 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]
    if colors.ndim == 5 and colors.shape[0] == 1:
        colors = colors[0]

    count = int(depth.shape[0])
    points_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []

    for idx in range(count):
        pts, cols, confs = build_point_cloud(
            depth[idx],
            depth_conf[idx],
            extrinsic[idx],
            intrinsic[idx],
            colors[idx],
            conf_threshold=conf_threshold,
            stride=stride,
        )
        if pts.size == 0:
            continue
        points_list.append(pts)
        colors_list.append(cols)
        conf_list.append(confs)

    if not points_list:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    points = np.concatenate(points_list, axis=0)
    point_colors = np.concatenate(colors_list, axis=0)
    conf_values = np.concatenate(conf_list, axis=0)

    return (
        points.astype(np.float32),
        point_colors.astype(np.float32),
        conf_values.astype(np.float32),
    )


def estimate_s0_batch(depth: np.ndarray, intrinsic: np.ndarray) -> float:
    values: list[float] = []
    for idx in range(int(depth.shape[0])):
        cur_s0 = estimate_s0_from_depth(depth[idx], intrinsic[idx])
        if cur_s0 > 0.0:
            values.append(cur_s0)
    if not values:
        return 0.0
    return float(np.median(np.array(values, dtype=np.float32)))


def resize_depth_conf_colors(
    depth: np.ndarray,
    depth_conf: np.ndarray,
    colors: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_width, target_height = target_size

    depth_tensor = torch.from_numpy(np.asarray(depth, dtype=np.float32)).permute(
        0, 3, 1, 2
    )
    depth_tensor = torch_nn.interpolate(
        depth_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    depth_out = depth_tensor.permute(0, 2, 3, 1).cpu().numpy()

    conf_tensor = torch.from_numpy(np.asarray(depth_conf, dtype=np.float32))[
        :, None, :, :
    ]
    conf_tensor = torch_nn.interpolate(
        conf_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    conf_out = conf_tensor[:, 0, :, :].cpu().numpy()

    color_tensor = torch.from_numpy(np.asarray(colors, dtype=np.float32)).permute(
        0, 3, 1, 2
    )
    color_tensor = torch_nn.interpolate(
        color_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    color_out = color_tensor.permute(0, 2, 3, 1).cpu().numpy()

    return depth_out, conf_out, color_out


def scale_intrinsic_batch(
    intrinsic: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    scaled = np.asarray(intrinsic, dtype=np.float32).copy()
    for idx in range(scaled.shape[0]):
        scaled[idx] = scale_intrinsic(scaled[idx], src_size, dst_size)
    return scaled


def build_point_cloud(
    depth: np.ndarray,
    depth_conf: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    colors: np.ndarray,
    conf_threshold: float,
    stride: int,
):
    if depth.ndim == 4 and depth.shape[0] == 1 and depth.shape[-1] == 1:
        depth_2d = depth[0, :, :, 0]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth_2d = depth[:, :, 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth_2d = depth[0]
    elif depth.ndim == 2:
        depth_2d = depth
    else:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    depth_map = depth_2d[:, :, None]

    if depth_conf.ndim == 4 and depth_conf.shape[0] == 1 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[0, :, :, 0]
    elif depth_conf.ndim == 3 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[:, :, 0]
    elif depth_conf.ndim == 3 and depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]

    if stride > 1:
        depth_map = depth_map[::stride, ::stride]
        depth_conf = depth_conf[::stride, ::stride]
        colors = colors[::stride, ::stride]

    if extrinsic.ndim == 3 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    if intrinsic.ndim == 3 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]

    world_points = unproject_depth_map_to_point_map(
        depth_map[None, ...], extrinsic[None, ...], intrinsic[None, ...]
    )[0]

    valid_mask = (depth_map.squeeze(-1) > 1e-6) & (depth_conf >= conf_threshold)
    points = world_points[valid_mask]
    point_colors = colors[valid_mask]
    conf_values = depth_conf[valid_mask]

    return points.astype(np.float32), point_colors.astype(np.float32), conf_values


def compute_depth_colors(points: np.ndarray) -> np.ndarray:
    depth = points[:, 2]
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    depth_min = float(np.percentile(depth, 5))
    depth_max = float(np.percentile(depth, 95))
    span = max(depth_max - depth_min, 1e-6)
    t = np.clip((depth - depth_min) / span, 0.0, 1.0)
    return np.stack([t, 1.0 - t, np.full_like(t, 0.1)], axis=-1).astype(np.float32)


def estimate_s0_from_depth(
    depth_map: np.ndarray, intrinsic: np.ndarray, eps: float = 1e-6
) -> float:
    if depth_map.ndim == 4 and depth_map.shape[0] == 1 and depth_map.shape[-1] == 1:
        depth_map = depth_map[0, :, :, 0]
    elif depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[:, :, 0]
    elif depth_map.ndim == 3 and depth_map.shape[0] == 1:
        depth_map = depth_map[0]
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(-1)
    if intrinsic.ndim > 2:
        intrinsic = intrinsic.squeeze()
    valid = depth_map > eps
    if not np.any(valid):
        return 0.0
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    dx = depth_map / max(fx, eps)
    dy = depth_map / max(fy, eps)
    spacing = np.sqrt(dx * dx + dy * dy)
    return float(np.median(spacing[valid]))


def scale_intrinsic(
    intrinsic: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    if intrinsic.ndim > 2:
        intrinsic = intrinsic.squeeze()
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic 3x3, got {intrinsic.shape}")
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale_x = dst_w / max(src_w, 1)
    scale_y = dst_h / max(src_h, 1)
    scaled = np.array(intrinsic, copy=True)
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled


def projection_from_intrinsic(
    intrinsic: np.ndarray, width: int, height: int, near: float, far: float
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


def build_view_matrix(extrinsic: np.ndarray) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = extrinsic[:3, :3]
    view[:3, 3] = extrinsic[:3, 3]
    conversion = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return conversion @ view


def main() -> None:
    args = parse_args()
    if (args.resize_width == 0) != (args.resize_height == 0):
        raise ValueError("Both resize width and height must be set or both 0.")
    if (args.viewport_width == 0) != (args.viewport_height == 0):
        raise ValueError("Both viewport width and height must be set or both 0.")

    resize_size = None
    if args.resize_width > 0 and args.resize_height > 0:
        resize_size = (args.resize_width, args.resize_height)

    missing = [path for path in args.images if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Image(s) not found: {missing_text}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    depth, depth_conf, extrinsic, intrinsic, colors, model_size = run_vggt(
        args.images, device, args.preprocess_mode
    )
    point_map_size = model_size
    if resize_size is not None:
        depth, depth_conf, colors = resize_depth_conf_colors(
            depth, depth_conf, colors, resize_size
        )
        intrinsic = scale_intrinsic_batch(intrinsic, model_size, resize_size)
        point_map_size = resize_size

    print(
        f"Point-map resolution: {point_map_size[0]}x{point_map_size[1]} "
        f"(model output {model_size[0]}x{model_size[1]})"
    )

    points, point_colors, conf_values = build_point_cloud_batch(
        depth,
        depth_conf,
        extrinsic,
        intrinsic,
        colors,
        conf_threshold=args.conf_threshold,
        stride=max(args.stride, 1),
    )
    conf_threshold = args.conf_threshold

    if points.size == 0 and args.conf_threshold > 0.0:
        print("No points after confidence filter; retrying with --conf-threshold 0.0")
        points, point_colors, conf_values = build_point_cloud_batch(
            depth,
            depth_conf,
            extrinsic,
            intrinsic,
            colors,
            conf_threshold=0.0,
            stride=max(args.stride, 1),
        )
        conf_threshold = 0.0

    if conf_values.size > 0 and conf_threshold > conf_values.max():
        conf_threshold = 0.0

    if points.size == 0:
        raise ValueError(
            "No points to display. Try lowering --stride or check depth output."
        )

    print(f"Rendering {points.shape[0]} points")

    depth_colors = compute_depth_colors(points)
    if args.auto_s0:
        s0 = estimate_s0_batch(depth, intrinsic)
    else:
        s0 = max(float(args.s0), 0.0)

    try:
        import pyglet
        from pyglet import gl
    except ImportError as exc:
        raise ImportError(
            "pyglet is required for viewing. Install with: pip install pyglet"
        ) from exc

    def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        forward = target - eye
        forward = forward / max(np.linalg.norm(forward), 1e-6)
        right = np.cross(forward, up)
        right = right / max(np.linalg.norm(right), 1e-6)
        true_up = np.cross(right, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = true_up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(true_up, eye)
        view[2, 3] = np.dot(forward, eye)
        return view

    class PointCloudViewer(pyglet.window.Window):
        def __init__(
            self,
            vertices,
            rgb_colors,
            depth_colors,
            confidences,
            point_size,
            intrinsic,
            model_size,
            window_size,
            extrinsic,
            shaders_dir,
            output_path,
            confidence_threshold,
            s0,
            sigma,
            occlusion_threshold,
            coarse_level,
            resizable: bool = True,
        ):
            if window_size is None:
                width, height = int(model_size[0]), int(model_size[1])
            else:
                width, height = int(window_size[0]), int(window_size[1])
            super().__init__(
                width=width,
                height=height,
                caption="VGGT Point Cloud Viewer",
                resizable=resizable,
            )
            self.vertices = vertices
            self.rgb_colors = rgb_colors
            self.depth_colors = depth_colors
            self.confidences = confidences
            self.use_depth_colors = False
            self.point_size = point_size
            self.intrinsic = intrinsic
            self.model_size = model_size
            self.fixed_render_size = window_size
            self.extrinsic = extrinsic
            self.shaders_dir = Path(shaders_dir)
            self.output_path = Path(output_path)
            self.confidence_threshold = float(confidence_threshold)
            self.s0_base = float(s0)
            self.jfa_mask_sigma = float(sigma)
            self.occlusion_threshold = float(occlusion_threshold)
            self.coarse_level = max(int(coarse_level), 0)
            self.world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

            self.center = self.vertices.mean(axis=0)
            radius = np.linalg.norm(self.vertices - self.center, axis=1).max()
            self.distance = max(radius * 2.5, 0.1)
            self.yaw = 0.0
            self.pitch = 0.0
            self.camera_pos = self.center + np.array(
                [0.0, 0.0, self.distance], dtype=np.float32
            )
            self.mouse_look = False
            self.move_speed = max(radius * 0.5, 0.05)
            self.look_sensitivity = 0.15
            self.use_vggt_pose = True
            self.move_state = {
                "forward": False,
                "backward": False,
                "left": False,
                "right": False,
                "up": False,
                "down": False,
                "boost": False,
            }

            self.renderer: HoleFillingRenderer | None = None
            self.render_size: tuple[int, int] | None = None
            self.ctx: moderngl.Context | None = None

            # Source aspect ratio (point-map width / height) — keep this fixed
            self.source_aspect = float(self.model_size[0]) / max(float(self.model_size[1]), 1.0)
            # Current full-window size (framebuffer) and computed sub-viewport (x,y,w,h)
            self.window_size_current: tuple[int, int] = (width, height)
            self._viewport: tuple[int, int, int, int] = (0, 0, width, height)

            self._reset_camera()
            self._init_gl()
            # initialize viewport/projection for current framebuffer size
            fb_w, fb_h = self.get_framebuffer_size()
            self._update_viewport_and_projection(fb_w, fb_h)
            pyglet.clock.schedule_interval(self._update_camera, 1.0 / 60.0)

        def _save_current_frame(self):
            self._ensure_renderer()
            self._render_frame()
            if self.renderer is None:
                raise RuntimeError("Renderer not initialized")
            frame = self.renderer.read_final_color()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(self.output_path)
            print(f"Saved render to {self.output_path}")

        def _reset_camera(self):
            if self.use_vggt_pose and self.extrinsic is not None:
                self._init_camera_from_extrinsic(self.extrinsic)
            else:
                self.yaw = 0.0
                self.pitch = 0.0
                self.camera_pos = self.center + np.array(
                    [0.0, 0.0, -self.distance], dtype=np.float32
                )

        def _init_camera_from_extrinsic(self, extrinsic):
            if extrinsic is None:
                return
            if extrinsic.ndim == 3 and extrinsic.shape[0] == 1:
                extr = extrinsic[0]
            else:
                extr = extrinsic
            if extr.shape == (3, 4):
                extr_4 = np.eye(4, dtype=np.float32)
                extr_4[:3, :] = extr
            else:
                extr_4 = extr.astype(np.float32)

            r = extr_4[:3, :3]
            t = extr_4[:3, 3]
            cam_to_world_r = r.T
            cam_to_world_t = -r.T @ t

            forward = cam_to_world_r @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
            yaw = math.degrees(math.atan2(forward[0], forward[2]))
            pitch = -math.degrees(math.asin(np.clip(forward[1], -1.0, 1.0)))
            self.yaw = yaw
            self.pitch = pitch
            self.camera_pos = cam_to_world_t.astype(np.float32)

        def _init_gl(self):
            gl.glClearColor(0.05, 0.06, 0.07, 1.0)

        def _compute_intrinsic_preserve_vertical(self, intrinsic: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
            # Keep vertical FOV stable: scale by dst_h / src_h for fx and fy,
            # adjust principal point relative to center so the view expands horizontally when wider.
            if intrinsic is None:
                # create a reasonable default pinhole with centered principal point
                src_w, src_h = src_size
                dst_w, dst_h = dst_size
                fx = dst_h
                fy = dst_h
                cx = dst_w * 0.5
                cy = dst_h * 0.5
                K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
                return K
            K = intrinsic.copy().astype(np.float32)
            if K.ndim > 2:
                K = K.squeeze()
            src_w, src_h = int(src_size[0]), int(src_size[1])
            dst_w, dst_h = int(dst_size[0]), int(dst_size[1])
            scale = dst_h / max(src_h, 1)
            fx = float(K[0, 0]) * scale
            fy = float(K[1, 1]) * scale
            # principal point offset from source center, then scaled
            ox = float(K[0, 2]) - (src_w * 0.5)
            oy = float(K[1, 2]) - (src_h * 0.5)
            cx = (dst_w * 0.5) + ox * scale
            cy = (dst_h * 0.5) + oy * scale
            K2 = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            return K2

        def _compute_viewport(self, win_w: int, win_h: int) -> tuple[int, int, int, int]:
            # For expand-to-fill behavior we use the full window as the viewport.
            return (0, 0, max(int(win_w), 1), max(int(win_h), 1))

        def _update_viewport_and_projection(self, win_w: int, win_h: int) -> None:
            self.window_size_current = (win_w, win_h)
            # For expand-to-fill, viewport is full framebuffer
            vp = self._compute_viewport(win_w, win_h)
            self._viewport = vp
            # Mark renderer for recreation if framebuffer size changed
            _, _, w, h = vp
            if self.render_size != (w, h):
                self.render_size = None
            try:
                gl.glViewport(vp[0], vp[1], vp[2], vp[3])
            except Exception:
                pass

        def on_resize(self, width: int, height: int) -> None:
            self._update_viewport_and_projection(width, height)
            # Recreate renderer immediately to match new size
            try:
                self._ensure_renderer()
            except Exception:
                pass
            return super().on_resize(width, height)

        def _ensure_renderer(self):
            self.switch_to()
            if self.ctx is None:
                self.ctx = moderngl.create_context()
            # Determine full framebuffer size or fixed render size
            if self.fixed_render_size is not None:
                full_size = (int(self.fixed_render_size[0]), int(self.fixed_render_size[1]))
            else:
                full_size = self.get_framebuffer_size()
            # Compute a centered sub-viewport that preserves the source aspect
            vp = self._compute_viewport(full_size[0], full_size[1])
            x, y, eff_w, eff_h = vp
            try:
                gl.glViewport(x, y, eff_w, eff_h)
            except Exception:
                pass

            if self.renderer is None or self.render_size != (eff_w, eff_h):
                self.renderer = HoleFillingRenderer(
                    eff_w,
                    eff_h,
                    shaders_dir=self.shaders_dir,
                    confidence_threshold=self.confidence_threshold,
                    jfa_mask_sigma=self.jfa_mask_sigma,
                    occlusion_threshold=self.occlusion_threshold,
                    coarse_level=self.coarse_level,
                    ctx=self.ctx,
                )
                if self.s0_base > 0.0:
                    self.renderer.s0 = self.s0_base
                self.render_size = (eff_w, eff_h)
            if self.renderer is not None:
                self.renderer.ctx.point_size = float(self.point_size)

        def _render_frame(self) -> None:
            colors = self.depth_colors if self.use_depth_colors else self.rgb_colors
            near = 0.01
            far = 10000.0
            # Use the computed effective render size (preserves source aspect)
            if self.render_size is None:
                # ensure renderer and viewport are set
                self._ensure_renderer()
            render_width, render_height = self.render_size if self.render_size is not None else self.get_framebuffer_size()
            vp_x, vp_y, vp_w, vp_h = self._viewport

            # Preserve vertical FOV: compute intrinsic scaled by render height
            base_intrinsic = self._compute_intrinsic_preserve_vertical(
                self.intrinsic.squeeze() if self.intrinsic is not None else None,
                self.model_size,
                (render_width, render_height),
            )

            proj = projection_from_intrinsic(
                base_intrinsic if base_intrinsic is not None else None, render_width, render_height, near, far
            )
            # compute vertical fov using fy
            if base_intrinsic is None:
                fov_y = 2.0 * math.atan(0.5 * render_height / max(1.0, render_height))
            else:
                fov_y = 2.0 * math.atan(0.5 * render_height / base_intrinsic[1, 1])

            yaw_rad = math.radians(self.yaw)
            pitch_rad = math.radians(self.pitch)
            forward = np.array(
                [
                    math.cos(pitch_rad) * math.sin(yaw_rad),
                    -math.sin(pitch_rad),
                    math.cos(pitch_rad) * math.cos(yaw_rad),
                ],
                dtype=np.float32,
            )
            view = look_at(
                self.camera_pos,
                self.camera_pos + forward,
                self.world_up,
            )

            if self.renderer is None:
                raise RuntimeError("Renderer not initialized")
            # Prefer direct GPU rendering when the renderer supports it
            if hasattr(self.renderer, "render_to_screen"):
                try:
                    self.renderer.render_to_screen(self.vertices, colors, self.confidences, view, proj, fov_y)
                except Exception as e:
                    print(f"[viewer] render_to_screen failed: {e}")
            else:
                # Fallback to CPU render+blit
                frame = self.renderer.render(self.vertices, colors, self.confidences, view, proj, fov_y)
                try:
                    frame = np.ascontiguousarray(frame)
                    # Diagnostics to help debug blank/flash: log min/max/mean
                    try:
                        fmin = int(frame.min())
                        fmax = int(frame.max())
                        fmean = float(frame.mean())
                        print(f"[viewer] rendered frame shape={frame.shape} min={fmin} max={fmax} mean={fmean:.2f}")
                    except Exception:
                        print(f"[viewer] rendered frame shape={getattr(frame, 'shape', None)}; cannot compute stats")
                    # Flip vertically for correct orientation when blitting
                    frame_flip = np.ascontiguousarray(np.flipud(frame))
                    img = pyglet.image.ImageData(vp_w, vp_h, 'RGB', frame_flip.tobytes(), pitch=vp_w * 3)
                    # Blit into the computed viewport origin to keep letterboxing/pillarboxing
                    img.blit(vp_x, vp_y)
                except Exception as e:
                    print(f"[viewer] frame blit failed: {e}")

        def on_draw(self):
            self.clear()
            self._ensure_renderer()
            self._render_frame()

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            if self.mouse_look:
                self.yaw += dx * self.look_sensitivity
                self.pitch += dy * self.look_sensitivity
                self.pitch = max(min(self.pitch, 89.0), -89.0)

        def on_mouse_press(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self.mouse_look = True

        def on_mouse_release(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self.mouse_look = False

        def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
            self.move_speed *= math.pow(1.1, scroll_y)

        def on_key_press(self, symbol, modifiers):
            if symbol == pyglet.window.key.F:
                self.yaw = 0.0
                self.pitch = 0.0
                self.camera_pos = self.center + np.array(
                    [0.0, 0.0, -self.distance], dtype=np.float32
                )
            elif symbol == pyglet.window.key.E:
                self._save_current_frame()
            elif symbol == pyglet.window.key.W:
                self.move_state["forward"] = True
            elif symbol == pyglet.window.key.S:
                self.move_state["backward"] = True
            elif symbol == pyglet.window.key.A:
                self.move_state["left"] = True
            elif symbol == pyglet.window.key.D:
                self.move_state["right"] = True
            elif symbol == pyglet.window.key.Q:
                self.move_state["down"] = True
            elif symbol == pyglet.window.key.SPACE:
                self.move_state["up"] = True
            elif symbol == pyglet.window.key.LSHIFT:
                self.move_state["boost"] = True

        def on_key_release(self, symbol, modifiers):
            if symbol == pyglet.window.key.W:
                self.move_state["forward"] = False
            elif symbol == pyglet.window.key.S:
                self.move_state["backward"] = False
            elif symbol == pyglet.window.key.A:
                self.move_state["left"] = False
            elif symbol == pyglet.window.key.D:
                self.move_state["right"] = False
            elif symbol == pyglet.window.key.Q:
                self.move_state["down"] = False
            elif symbol == pyglet.window.key.SPACE:
                self.move_state["up"] = False
            elif symbol == pyglet.window.key.LSHIFT:
                self.move_state["boost"] = False

        def on_close(self):
            self._save_current_frame()
            return super().on_close()

        def _update_camera(self, dt):
            yaw_rad = math.radians(self.yaw)
            pitch_rad = math.radians(self.pitch)
            forward = np.array(
                [
                    math.cos(pitch_rad) * math.sin(yaw_rad),
                    -math.sin(pitch_rad),
                    math.cos(pitch_rad) * math.cos(yaw_rad),
                ],
                dtype=np.float32,
            )
            right = np.cross(forward, self.world_up)
            norm_right = np.linalg.norm(right)
            if norm_right > 1e-6:
                right = right / norm_right

            speed = self.move_speed * (3.0 if self.move_state["boost"] else 1.0)
            velocity = np.zeros(3, dtype=np.float32)
            if self.move_state["forward"]:
                velocity += forward
            if self.move_state["backward"]:
                velocity -= forward
            if self.move_state["left"]:
                velocity -= right
            if self.move_state["right"]:
                velocity += right
            if self.move_state["up"]:
                velocity += self.world_up
            if self.move_state["down"]:
                velocity -= self.world_up

            norm = np.linalg.norm(velocity)
            if norm > 1e-6:
                velocity = velocity / norm
                self.camera_pos += velocity * speed * dt

    print("Controls: left-drag look, scroll adjust speed")
    print("Keys: WASD move, Q/Space down/up, Shift boost, F front view, E export")

    shaders_dir = Path(__file__).resolve().parent / "shaders"
    window_size = None
    if args.viewport_width > 0 and args.viewport_height > 0:
        window_size = (args.viewport_width, args.viewport_height)

    viewer = PointCloudViewer(
        points,
        point_colors,
        depth_colors,
        conf_values,
        args.point_size,
        intrinsic[0],
        point_map_size,
        window_size,
        extrinsic[0],
        shaders_dir,
        args.output,
        conf_threshold,
        s0,
        args.sigma,
        args.occlusion_threshold,
        args.coarse_level,
        resizable=True,
    )
    pyglet.app.run()


if __name__ == "__main__":
    main()
