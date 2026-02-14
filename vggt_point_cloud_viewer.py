import argparse
import math
from pathlib import Path

import moderngl
import numpy as np
import torch
import torch.nn.functional as torch_nn
from PIL import Image
import sys
import io
import platform

# Ensure local `vggt/` package is importable when running the script directly
_repo_root = Path(__file__).resolve().parent
_vggt_path = str(_repo_root / "vggt")
if _vggt_path not in sys.path:
    sys.path.insert(0, _vggt_path)

from hole_filling_renderer import HoleFillingRenderer
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
try:
    import win32clipboard
    import win32con
except Exception:
    win32clipboard = None
    win32con = None


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


def _pil_image_to_clipboard(img: Image.Image) -> None:
    """Copy a PIL Image to the Windows clipboard as a DIB (works on Windows).

    Raises RuntimeError if not running on Windows or pywin32 not available.
    """
    if platform.system() != "Windows":
        raise RuntimeError("Clipboard image copy only implemented on Windows")
    if win32clipboard is None or win32con is None:
        raise RuntimeError("pywin32 is required for clipboard support (install pywin32)")

    output = io.BytesIO()
    # Save as BMP to get a DIB-compatible byte stream, then strip the 14-byte BMP header
    img.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_DIB, data)
    finally:
        win32clipboard.CloseClipboard()


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
            # roll angle (degrees) and roll state
            self.roll = 0.0
            self.roll_speed = 45.0
            self.roll_state = {"left": False, "right": False}
            self.mouse_look = False
            self.shift_drag = False
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

            # Context menu state for right-click actions
            self.context_menu_visible = False
            self.context_menu_origin = (0, 0)
            self.context_menu_hover_index = -1
            self.context_menu_item_height = 22
            self.context_menu_padding = 8
            self.context_menu_items = [
                ("Copy image to clipboard", self._copy_current_frame_to_clipboard),
                ("Save image (E)", self._save_current_frame),
            ]
            # cached menu box (ox, oy, w, h)
            self.context_menu_box = None
            # message toast
            self._message_text = ""
            self._message_duration = 1.5
            self._message_label = None

        def _save_current_frame(self):
            self._ensure_renderer()
            self._render_frame()
            if self.renderer is None:
                raise RuntimeError("Renderer not initialized")
            frame = self.renderer.read_final_color()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(self.output_path)
            print(f"Saved render to {self.output_path}")

        def _copy_current_frame_to_clipboard(self):
            self._ensure_renderer()
            self._render_frame()
            if self.renderer is None:
                print("Renderer not initialized; cannot copy to clipboard")
                return
            frame = self.renderer.read_final_color()
            try:
                img = Image.fromarray(frame)
                _pil_image_to_clipboard(img)
                print("Copied render to clipboard")
                try:
                    self._show_message("Copied render to clipboard", duration=1.6)
                except Exception:
                    pass
            except Exception as e:
                print(f"Failed to copy to clipboard: {e}")
                try:
                    self._show_message(f"Copy failed: {e}", duration=2.5)
                except Exception:
                    pass

        def _update_context_hover(self, x: int, y: int) -> None:
            # Compute menu box and hover index, clamp to framebuffer
            ox, oy = self.context_menu_origin
            items = getattr(self, "context_menu_items", [])
            item_h = getattr(self, "context_menu_item_height", 22)
            pad = getattr(self, "context_menu_padding", 8)

            # compute width using text metrics
            maxw = 0
            for txt, _ in items:
                lbl = pyglet.text.Label(txt)
                w = getattr(lbl, "content_width", len(txt) * 8)
                maxw = max(maxw, w)
            box_w = maxw + pad * 2
            box_h = item_h * len(items)

            fb_w, fb_h = self.get_framebuffer_size()
            # clamp origin so menu fits in framebuffer
            ox = min(max(0, ox), max(0, fb_w - 4))
            oy = min(max(0, oy), max(0, fb_h))
            if ox + box_w > fb_w:
                ox = max(0, fb_w - box_w)
            if oy - box_h < 0:
                oy = box_h
            self.context_menu_origin = (int(ox), int(oy))
            self.context_menu_box = (int(ox), int(oy), int(box_w), int(box_h))

            left = ox
            right = ox + box_w
            top = oy
            bottom = oy - box_h

            if x < left or x > right or y < bottom or y > top:
                self.context_menu_hover_index = -1
                return
            idx = int((top - y) // item_h)
            if idx < 0 or idx >= len(items):
                self.context_menu_hover_index = -1
            else:
                self.context_menu_hover_index = idx

        def _show_message(self, text: str, duration: float = 1.5) -> None:
            try:
                pyglet.clock.unschedule(self._clear_message)
            except Exception:
                pass
            self._message_text = str(text)
            self._message_duration = float(duration)
            # create centered label near bottom
            try:
                self._message_label = pyglet.text.Label(
                    self._message_text,
                    x=self.get_framebuffer_size()[0] // 2,
                    y=20,
                    anchor_x="center",
                    anchor_y="bottom",
                    color=(255, 255, 255, 255),
                    font_size=12,
                )
            except Exception:
                self._message_label = None
            try:
                pyglet.clock.schedule_once(self._clear_message, self._message_duration)
            except Exception:
                pass

        def _clear_message(self, dt: float) -> None:
            self._message_text = ""
            try:
                self._message_label = None
            except Exception:
                pass

        def _draw_message(self) -> None:
            try:
                if getattr(self, "_message_label", None) is not None:
                    self._message_label.draw()
            except Exception:
                pass

        def on_mouse_motion(self, x, y, dx, dy):
            # Update hover index for context menu when visible
            if not getattr(self, "context_menu_visible", False):
                return
            try:
                self._update_context_hover(x, y)
            except Exception:
                pass

        def _reset_camera(self):
            if self.use_vggt_pose and self.extrinsic is not None:
                self._init_camera_from_extrinsic(self.extrinsic)
            else:
                self.yaw = 0.0
                self.pitch = 0.0
                self.camera_pos = self.center + np.array(
                    [0.0, 0.0, -self.distance], dtype=np.float32
                )
            # Reset roll state so 'F' restores the original upright view
            try:
                self.roll = 0.0
            except Exception:
                pass
            try:
                if hasattr(self, "roll_state") and isinstance(self.roll_state, dict):
                    self.roll_state["left"] = False
                    self.roll_state["right"] = False
            except Exception:
                pass

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

        def _rotate_vector_around_axis(self, v: np.ndarray, k: np.ndarray, angle_rad: float) -> np.ndarray:
            # Rodrigues' rotation formula: rotate vector v around axis k (unit) by angle
            k = np.asarray(k, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32)
            k_norm = np.linalg.norm(k)
            if k_norm < 1e-9:
                return v
            k = k / k_norm
            c = math.cos(angle_rad)
            s = math.sin(angle_rad)
            return (v * c) + (np.cross(k, v) * s) + (k * (np.dot(k, v) * (1.0 - c)))

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
            # compute up vector with roll applied around the forward axis
            up = self.world_up
            if abs(self.roll) > 1e-6:
                up = self._rotate_vector_around_axis(up, forward, math.radians(self.roll))
            view = look_at(
                self.camera_pos,
                self.camera_pos + forward,
                up,
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
            # Draw context menu overlay if visible
            try:
                if getattr(self, "context_menu_visible", False):
                    self._draw_context_menu()
            except Exception:
                pass
            # Draw toast / message overlays if any
            self._draw_message()

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            # If context menu visible, update hover on drag and consume input
            if getattr(self, "context_menu_visible", False):
                try:
                    self._update_context_hover(x, y)
                except Exception:
                    pass
                return

            # If shift-dragging with left button, map drag to strafing/up-down movement
            if getattr(self, "shift_drag", False):
                # small deadzone to avoid jitter
                dead = 2.0
                if dx < -dead:
                    self.move_state["left"] = True
                    self.move_state["right"] = False
                elif dx > dead:
                    self.move_state["right"] = True
                    self.move_state["left"] = False
                else:
                    self.move_state["left"] = False
                    self.move_state["right"] = False

                if dy > dead:
                    # moving mouse up -> move up (Space)
                    self.move_state["up"] = True
                    self.move_state["down"] = False
                elif dy < -dead:
                    # moving mouse down -> move down (Q)
                    self.move_state["down"] = True
                    self.move_state["up"] = False
                else:
                    self.move_state["up"] = False
                    self.move_state["down"] = False
                return

            if self.mouse_look:
                self.yaw += dx * self.look_sensitivity
                self.pitch += dy * self.look_sensitivity
                self.pitch = max(min(self.pitch, 89.0), -89.0)

        def on_mouse_press(self, x, y, button, modifiers):
            # If a context menu is visible, intercept clicks for menu interaction
            if getattr(self, "context_menu_visible", False):
                if button == pyglet.window.mouse.LEFT:
                    idx = getattr(self, "context_menu_hover_index", -1)
                    if idx is None:
                        idx = -1
                    items = getattr(self, "context_menu_items", [])
                    if idx >= 0 and idx < len(items):
                        try:
                            items[idx][1]()
                        except Exception as e:
                            print(f"Context menu item failed: {e}")
                    self.context_menu_visible = False
                    return
                elif button == pyglet.window.mouse.RIGHT:
                    # right click while menu open dismisses it
                    self.context_menu_visible = False
                    return

            if button == pyglet.window.mouse.LEFT:
                # If Shift is held while pressing left, enter shift-drag mode (strafe/vertical)
                try:
                    shift_held = (modifiers & pyglet.window.key.MOD_SHIFT) != 0
                except Exception:
                    shift_held = False
                if shift_held:
                    self.shift_drag = True
                    self.mouse_look = False
                else:
                    self.mouse_look = True

        def on_mouse_release(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                # Clear either mouse look or shift-drag state
                if getattr(self, "shift_drag", False):
                    self.shift_drag = False
                    # clear movement keys set by shift-drag
                    self.move_state["left"] = False
                    self.move_state["right"] = False
                    self.move_state["up"] = False
                    self.move_state["down"] = False
                self.mouse_look = False
            elif button == pyglet.window.mouse.RIGHT:
                # Open context menu on right-button release (platform-consistent)
                # If menu already visible this was handled in on_mouse_press above
                try:
                    self.context_menu_origin = (int(x), int(y))
                    # compute hover and clamp box
                    self.context_menu_visible = True
                    self.context_menu_hover_index = -1
                    self._update_context_hover(x, y)
                except Exception:
                    self.context_menu_visible = False

        def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
            # keep existing behavior (adjust move speed)
            self.move_speed *= math.pow(1.1, scroll_y)
            # Map scroll wheel to forward/backward motion like W/S.
            # For a natural feel, set the move state briefly and clear it after a short duration.
            try:
                if scroll_y > 0:
                    self.move_state["forward"] = True
                    pyglet.clock.schedule_once(lambda dt: self._clear_move_state("forward"), 0.12 * float(scroll_y))
                elif scroll_y < 0:
                    self.move_state["backward"] = True
                    pyglet.clock.schedule_once(lambda dt: self._clear_move_state("backward"), 0.12 * float(abs(scroll_y)))
            except Exception:
                pass

        def _clear_move_state(self, key: str, dt: float) -> None:
            # Helper used by scheduled callbacks to clear transient move state
            try:
                if key in self.move_state:
                    self.move_state[key] = False
            except Exception:
                pass

        def on_key_press(self, symbol, modifiers):
            if symbol == pyglet.window.key.F:
                # Reset camera to the configured initial pose
                try:
                    self._reset_camera()
                except Exception:
                    # Fallback to sensible default
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

            # Ctrl+C (or Cmd+C on macOS) -> copy current render to clipboard
            try:
                is_copy = (
                    symbol == pyglet.window.key.C
                    and (
                        (modifiers & pyglet.window.key.MOD_CTRL) != 0
                        or (modifiers & getattr(pyglet.window.key, "MOD_COMMAND", 0)) != 0
                    )
                )
            except Exception:
                is_copy = False
            if is_copy:
                try:
                    self._copy_current_frame_to_clipboard()
                except Exception as e:
                    print(f"Copy to clipboard failed: {e}")
                # Consume the key so the plain 'C' behavior (roll right) does not run
                return

            if symbol == pyglet.window.key.Z:
                # roll left
                self.roll_state["left"] = True
            if symbol == pyglet.window.key.C:
                # roll right
                self.roll_state["right"] = True

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
            elif symbol == pyglet.window.key.Z:
                self.roll_state["left"] = False
            elif symbol == pyglet.window.key.C:
                self.roll_state["right"] = False

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
            # update roll from key state
            if self.roll_state["left"]:
                self.roll += self.roll_speed * dt
            if self.roll_state["right"]:
                self.roll -= self.roll_speed * dt
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

        def _draw_context_menu(self) -> None:
            # Simple pyglet-drawn context menu at the stored origin (top-left)
            if not getattr(self, "context_menu_visible", False):
                return
            ox, oy = self.context_menu_origin
            items = getattr(self, "context_menu_items", [])
            item_h = getattr(self, "context_menu_item_height", 22)
            pad = getattr(self, "context_menu_padding", 8)
            # compute widths using labels
            maxw = 0
            labels_info = []
            for idx, (txt, _) in enumerate(items):
                lbl = pyglet.text.Label(txt, anchor_x="left", anchor_y="top", font_size=12)
                w = getattr(lbl, "content_width", len(txt) * 8)
                maxw = max(maxw, w)
                labels_info.append((txt, w))

            box_w = maxw + pad * 2
            box_h = item_h * len(items)

            # clamp origin to framebuffer
            fb_w, fb_h = self.get_framebuffer_size()
            ox = min(max(0, ox), max(0, fb_w - 4))
            oy = min(max(0, oy), max(0, fb_h))
            if ox + box_w > fb_w:
                ox = max(0, fb_w - box_w)
            if oy - box_h < 0:
                oy = box_h
            self.context_menu_origin = (int(ox), int(oy))
            self.context_menu_box = (int(ox), int(oy), int(box_w), int(box_h))

            batch = pyglet.graphics.Batch()
            from pyglet import shapes

            # Background (pyglet shapes use bottom-left origin)
            bg = shapes.Rectangle(ox, oy - box_h, box_w, box_h, color=(40, 40, 40), batch=batch)

            # Hover highlight
            hover_idx = getattr(self, "context_menu_hover_index", -1)
            if 0 <= hover_idx < len(items):
                hy_y = oy - (hover_idx + 1) * item_h
                hover_rect = shapes.Rectangle(ox, hy_y, box_w, item_h, color=(70, 70, 70), batch=batch)

            # Labels
            # vertical padding inside each item
            pad_v = max(2, (item_h - 12) // 2)
            labels = []
            for idx, (txt, _) in enumerate(items):
                lbl = pyglet.text.Label(
                    txt,
                    x=ox + pad,
                    y=oy - pad_v - idx * item_h,
                    anchor_x="left",
                    anchor_y="top",
                    color=(255, 255, 255, 255),
                    batch=batch,
                    font_size=12,
                )
                labels.append(lbl)

            try:
                batch.draw()
            except Exception:
                # fallback to drawing labels individually
                for lbl in labels:
                    try:
                        lbl.draw()
                    except Exception:
                        pass

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
