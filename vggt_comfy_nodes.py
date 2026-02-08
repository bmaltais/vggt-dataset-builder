import os
import struct
import numpy as np
import torch
try:
    from .hole_filling_renderer import HoleFillingRenderer
except ImportError:
    from hole_filling_renderer import HoleFillingRenderer
from pathlib import Path
import json

class VGGT_PLY_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ply_path": ("STRING", {"default": "output/scene1/image2_reference.ply"}),
            }
        }

    RETURN_TYPES = ("VGGT_POINTS",)
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
            while "end_header" not in header:
                line = f.readline().decode("ascii")
                header += line
                if "element vertex" in line:
                    num_points = int(line.split()[-1])

            # Check for confidence property
            has_confidence = "property float confidence" in header

            # Read binary data
            # fffBBB (12 + 3 = 15 bytes) or fffBBBf (15 + 4 = 19 bytes)
            point_size = 19 if has_confidence else 15
            data = f.read(num_points * point_size)

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

            return ({"points": points, "colors": colors, "confidences": confidences, "ply_path": ply_path},)

class VGGT_PLY_Viewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vggt_points": ("VGGT_POINTS",),
                "camera_state": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("VGGT_CAMERA",)
    FUNCTION = "get_camera"
    CATEGORY = "VGGT"
    OUTPUT_NODE = True

    def get_camera(self, vggt_points, camera_state):
        # Notify the UI about the PLY path
        ui_data = {"ply_path": vggt_points.get("ply_path", "")}

        if not camera_state:
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
        else:
            state = json.loads(camera_state)
            view_mat = np.array(state["view_matrix"], dtype=np.float32).reshape(4, 4)
            proj_mat = np.array(state["proj_matrix"], dtype=np.float32).reshape(4, 4)
            fov_y = float(state["fov_y"])

        return {"ui": ui_data, "result": ({"view_matrix": view_mat, "proj_matrix": proj_mat, "fov_y": fov_y},)}

class VGGT_PLY_Renderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vggt_points": ("VGGT_POINTS",),
                "vggt_camera": ("VGGT_CAMERA",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "confidence_threshold": ("FLOAT", {"default": 1.01, "min": 0.0, "max": 2.0}),
                "sigma": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "VGGT"

    def render(self, vggt_points, vggt_camera, width, height, confidence_threshold, sigma):
        # Use the directory of this file to find shaders
        shaders_dir = Path(__file__).parent / "shaders"

        renderer = HoleFillingRenderer(
            width=width,
            height=height,
            shaders_dir=shaders_dir,
            confidence_threshold=confidence_threshold,
            jfa_mask_sigma=sigma
        )

        img = renderer.render(
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
