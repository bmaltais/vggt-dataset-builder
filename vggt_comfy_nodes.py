import os
import struct
import numpy as np
import torch
import sys
try:
    from .hole_filling_renderer import HoleFillingRenderer
except ImportError:
    from hole_filling_renderer import HoleFillingRenderer
from pathlib import Path
import json

# Add vggt to path
vggt_path = str(Path(__file__).parent / "vggt")
if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

# Lazy loading of VGGT model
_vggt_model = None

def get_vggt_model(device_name):
    global _vggt_model
    if _vggt_model is None:
        from vggt.models.vggt import VGGT
        device = torch.device(device_name)
        _vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        _vggt_model.eval()
    return _vggt_model

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

class VGGT_Model_Inference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("VGGT_POINTS", "VGGT_CAMERAS")
    FUNCTION = "infer"
    CATEGORY = "VGGT"

    def infer(self, images, device):
        # images is (B, H, W, C), range 0-1
        batch_size, height, width, channels = images.shape

        # Prepare images for model (B, C, H, W)
        model_input = images.permute(0, 3, 1, 2).to(device)

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        model = get_vggt_model(device)

        dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype, enabled=device == "cuda"):
                predictions = model(model_input)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], (height, width)
        )

        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]

        # Convert to numpy
        extrinsic = extrinsic.cpu().numpy()
        intrinsic = intrinsic.cpu().numpy()
        depth = depth.cpu().numpy()
        depth_conf = depth_conf.cpu().numpy()
        images_np = images.cpu().numpy()

        # Unproject points for all frames
        world_points_batch = unproject_depth_map_to_point_map(
            depth, extrinsic, intrinsic
        )

        cameras = []
        for i in range(batch_size):
            view_mat = build_view_matrix(extrinsic[i])
            proj_mat = build_projection_matrix(intrinsic[i], width, height)
            fov_y = 2.0 * np.arctan(0.5 * height / intrinsic[i, 1, 1])
            cameras.append({
                "view_matrix": view_mat,
                "proj_matrix": proj_mat,
                "fov_y": fov_y,
                "extrinsic": extrinsic[i],
                "intrinsic": intrinsic[i]
            })

        # For VGGT_POINTS, we'll just take the points from the first frame for now
        # OR we could merge them all. Let's provide points from the first frame
        # and allow the user to select which frame's points to use if needed.
        # But typically we want a single point cloud.

        # For simplicity, return the first frame's points but all cameras
        vggt_points = {
            "points": world_points_batch[0].reshape(-1, 3),
            "colors": images_np[0].reshape(-1, 3),
            "confidences": depth_conf[0].reshape(-1),
            "all_points": world_points_batch,
            "all_colors": images_np,
            "all_confidences": depth_conf,
            "width": width,
            "height": height,
            "cameras": cameras
        }

        # Optionally save a temporary PLY for the viewer
        import folder_paths
        import uuid
        output_dir = folder_paths.get_output_directory()
        temp_id = str(uuid.uuid4())[:8]
        ply_filename = f"vggt_temp_{temp_id}.ply"
        ply_path = os.path.join(output_dir, ply_filename)

        # Reuse write_ply from build_warp_dataset.py logic
        def write_ply_basic(path, points, colors, confs):
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
            header += "property float x\nproperty float y\nproperty float z\n"
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            header += "property float confidence\nend_header\n"
            with open(path, 'wb') as f:
                f.write(header.encode('ascii'))
                colors_u8 = (colors * 255).astype(np.uint8)
                for i in range(len(points)):
                    f.write(struct.pack('fffBBBf', *points[i], *colors_u8[i], confs[i]))

        write_ply_basic(ply_path, vggt_points["points"], vggt_points["colors"], vggt_points["confidences"])
        vggt_points["ply_path"] = ply_filename # ComfyUI prefers relative to output/input

        return (vggt_points, cameras)

class VGGT_PLY_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {"default": "output/scene1/image2_reference.ply"}),
            }
        }

    RETURN_TYPES = ("VGGT_POINTS", "VGGT_CAMERAS")
    FUNCTION = "load_ply"
    CATEGORY = "VGGT"

    def load_ply(self, ply_path):
        if not os.path.exists(ply_path):
            # Try relative to repo root if not found
            root_ply_path = Path(__file__).parent / ply_path
            if root_ply_path.exists():
                ply_path = str(root_ply_path)
            else:
                raise FileNotFoundError(f"PLY file not found: {ply_path}")

        with open(ply_path, "rb") as f:
            header = ""
            num_points = None
            while "end_header" not in header:
                line_bytes = f.readline()
                if not line_bytes:
                    raise ValueError(f"Unexpected EOF while reading PLY header: 'end_header' not found in file '{ply_path}'")
                line = line_bytes.decode("ascii")
                header += line
                if "element vertex" in line:
                    num_points = int(line.split()[-1])

            if num_points is None:
                raise ValueError(f"PLY header is missing required 'element vertex' line in file '{ply_path}'")

            # Validate format
            if "format binary_little_endian" not in header:
                raise ValueError(f"PLY file must be in binary_little_endian format. File: '{ply_path}'")

            # Check for confidence property
            has_confidence = "property float confidence" in header

            # Read binary data
            # fffBBB (12 + 3 = 15 bytes) or fffBBBf (15 + 4 = 19 bytes)
            point_size = 19 if has_confidence else 15
            data = f.read(num_points * point_size)

            # Validate data length
            if len(data) != num_points * point_size:
                raise ValueError(
                    f"PLY data size mismatch: expected {num_points * point_size} bytes, "
                    f"got {len(data)} bytes in file '{ply_path}'"
                )

            # Use numpy for faster loading if possible
            if has_confidence:
                dt = np.dtype([
                    ('pos', 'f4', 3),
                    ('color', 'u1', 3),
                    ('conf', 'f4', 1)
                ])
            else:
                dt = np.dtype([
                    ('pos', 'f4', 3),
                    ('color', 'u1', 3)
                ])

            array = np.frombuffer(data, dtype=dt)
            points = array['pos'].astype(np.float32)
            colors = array['color'].astype(np.float32) / 255.0
            if has_confidence:
                confidences = array['conf'].astype(np.float32).flatten()
            else:
                confidences = np.ones(num_points, dtype=np.float32)

            # Check for sidecar JSON with camera info
            json_path = ply_path.replace(".ply", ".json")
            cameras = []
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        cam_data = json.load(f)
                        if isinstance(cam_data, dict):
                            cam_data = [cam_data]

                        for cam in cam_data:
                            extrinsic = np.array(cam["extrinsic"])
                            intrinsic = np.array(cam["intrinsic"])
                            width = cam.get("width", 512)
                            height = cam.get("height", 512)
                            view_mat = build_view_matrix(extrinsic)
                            proj_mat = build_projection_matrix(intrinsic, width, height)
                            fov_y = 2.0 * np.arctan(0.5 * height / intrinsic[1, 1])
                            cameras.append({
                                "view_matrix": view_mat,
                                "proj_matrix": proj_mat,
                                "fov_y": fov_y,
                                "extrinsic": extrinsic,
                                "intrinsic": intrinsic
                            })
                except Exception as e:
                    print(f"Warning: Failed to load camera sidecar {json_path}: {e}")

            vggt_points = {
                "points": points,
                "colors": colors,
                "confidences": confidences,
                "ply_path": ply_path,
                "cameras": cameras
            }
            return (vggt_points, cameras)

class VGGT_PLY_Viewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vggt_points": ("VGGT_POINTS",),
                "camera_state": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "vggt_cameras": ("VGGT_CAMERAS",),
            }
        }

    RETURN_TYPES = ("VGGT_CAMERA",)
    FUNCTION = "get_camera"
    CATEGORY = "VGGT"
    OUTPUT_NODE = True

    def get_camera(self, vggt_points, camera_state, vggt_cameras=None):
        # Notify the UI about the PLY path and cameras
        ui_data = {
            "ply_path": vggt_points.get("ply_path", ""),
            "cameras": []
        }

        all_cams = vggt_cameras if vggt_cameras is not None else vggt_points.get("cameras", [])
        for cam in all_cams:
            ui_data["cameras"].append({
                "view_matrix": cam["view_matrix"].tolist(),
                "proj_matrix": cam["proj_matrix"].tolist(),
                "fov_y": cam["fov_y"]
            })

        if camera_state:
            try:
                state = json.loads(camera_state)
                view_mat = np.array(state["view_matrix"], dtype=np.float32).reshape(4, 4)
                proj_mat = np.array(state["proj_matrix"], dtype=np.float32).reshape(4, 4)
                fov_y = float(state["fov_y"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    "Invalid camera_state: expected JSON with keys "
                    "'view_matrix', 'proj_matrix', and 'fov_y', with 4x4 matrices."
                ) from e
        elif all_cams:
            view_mat = all_cams[0]["view_matrix"]
            proj_mat = all_cams[0]["proj_matrix"]
            fov_y = all_cams[0]["fov_y"]
        else:
            # Default camera: look at the center of the point cloud
            points = vggt_points["points"]
            center = points.mean(axis=0)
            extent = points.max(axis=0) - points.min(axis=0)
            max_extent = extent.max()

            # Simple default view matrix (looking from some distance)
            view_mat = np.eye(4, dtype=np.float32)
            view_mat[2, 3] = -max_extent * 2 # Move back
            view_mat[:3, 3] += np.dot(view_mat[:3, :3], -center) # Look at center

            # Default projection matrix
            fov_y = 0.785 # 45 degrees
            aspect = 1.0
            near = 0.01
            far = max_extent * 100

            f = 1.0 / np.tan(fov_y / 2.0)
            proj_mat = np.zeros((4, 4), dtype=np.float32)
            proj_mat[0, 0] = f / aspect
            proj_mat[1, 1] = f
            proj_mat[2, 2] = -(far + near) / (far - near)
            proj_mat[2, 3] = -(2.0 * far * near) / (far - near)
            proj_mat[3, 2] = -1.0

        return {"ui": ui_data, "result": ({"view_matrix": view_mat, "proj_matrix": proj_mat, "fov_y": fov_y},)}

class VGGT_PLY_Renderer:
    def __init__(self):
        self.renderer = None
        self.last_config = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vggt_points": ("VGGT_POINTS",),
                "vggt_camera": ("VGGT_CAMERA",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "confidence_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "sigma": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "VGGT"

    def render(self, vggt_points, vggt_camera, width, height, confidence_threshold, sigma):
        # Use the directory of this file to find shaders
        shaders_dir = Path(__file__).parent / "shaders"
        current_config = (width, height)

        if self.renderer is None or self.last_config != current_config:
            self.renderer = HoleFillingRenderer(
                width=width,
                height=height,
                shaders_dir=shaders_dir,
            )
            self.last_config = current_config
        
        # Update confidence threshold and sigma (these can change without recreating renderer)
        self.renderer.confidence_threshold = confidence_threshold
        self.renderer.jfa_mask_sigma = sigma

        img = self.renderer.render(
            points=vggt_points["points"],
            colors=vggt_points["colors"],
            confidences=vggt_points["confidences"],
            view_mat=vggt_camera["view_matrix"],
            proj_mat=vggt_camera["proj_matrix"],
            fov_y=vggt_camera["fov_y"]
        )

        # Convert to ComfyUI image tensor (B, H, W, C) range 0-1
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0) # Add batch dimension

        return (img_tensor,)
