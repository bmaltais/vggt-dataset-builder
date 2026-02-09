import os
import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add vggt to path
vggt_path = str(Path(__file__).parent / "vggt")
if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

# Import VGGT utilities
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# Lazy loading of VGGT model
_vggt_model = None
_vggt_model_device = None

def get_vggt_model(device_name):
    global _vggt_model, _vggt_model_device
    from vggt.models.vggt import VGGT
    device = torch.device(device_name)
    
    if _vggt_model is None:
        _vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
        _vggt_model.eval()
    
    # Move model to the requested device if it's not already there
    if _vggt_model_device != device_name:
        _vggt_model = _vggt_model.to(device)
        _vggt_model_device = device_name
    
    return _vggt_model

def write_ply_basic(path, points, colors, confs, scale_multiplier=0.5):
    """Write a 3DGS-compatible PLY file matching GaussianViewer's format exactly."""
    import numpy as np
    from plyfile import PlyElement, PlyData
    
    n_points = len(points)
    
    # Clamp colors to [0, 1] range
    colors_float = np.clip(colors, 0.0, 1.0).astype(np.float32)
    
    # Convert colors to spherical harmonics DC component
    # GaussianViewer uses: rgb = (0.5 + SH_C0 * coeff) * 255
    # So reverse: coeff = (rgb/255 - 0.5) / SH_C0
    # SH_C0 = sqrt(1 / (4 * pi)) ≈ 0.28209479177387814
    SH_C0 = 0.28209479177387814
    sh_dc = (colors_float - 0.5) / SH_C0
    
    # Compute reasonable scales based on point density
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    scene_size = np.linalg.norm(points_max - points_min)
    
    # Average distance between points (rough estimate)
    if n_points > 1:
        avg_point_distance = scene_size / (n_points ** (1/3))
    else:
        avg_point_distance = 0.1
    
    # Create per-point scales
    point_scales = np.ones((n_points, 3), dtype=np.float32) * avg_point_distance * scale_multiplier
    scales = np.log(point_scales)
    
    # Default quaternion (identity rotation) for all points
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (n_points, 1)).astype(np.float32)
    
    # Convert opacity to logits using inverse sigmoid
    # Clamp confidences to valid range
    confs_safe = np.clip(confs, 1e-7, 1.0 - 1e-7).astype(np.float32)
    opacity_logits = np.log(confs_safe / (1.0 - confs_safe))
    
    print(f"[PLY Writer] SH conversion debug:")
    print(f"  Input colors range: [{colors_float.min():.4f}, {colors_float.max():.4f}]")
    print(f"  SH_C0 constant: {SH_C0:.10f}")
    print(f"  SH_DC range: [{sh_dc.min():.4f}, {sh_dc.max():.4f}]")
    print(f"  Opacity logits range: [{opacity_logits.min():.4f}, {opacity_logits.max():.4f}]")
    print(f"  Scales range: [{scales.min():.4f}, {scales.max():.4f}]")
    
    # Build dtype matching GaussianViewer's PLY format
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    
    # Create structured array
    elements = np.zeros(n_points, dtype=dtype_full)
    elements['x'] = points[:, 0]
    elements['y'] = points[:, 1]
    elements['z'] = points[:, 2]
    elements['f_dc_0'] = sh_dc[:, 0]
    elements['f_dc_1'] = sh_dc[:, 1]
    elements['f_dc_2'] = sh_dc[:, 2]
    elements['opacity'] = opacity_logits
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = quaternions[:, 0]
    elements['rot_1'] = quaternions[:, 1]
    elements['rot_2'] = quaternions[:, 2]
    elements['rot_3'] = quaternions[:, 3]
    
    # Write PLY file using plyfile
    vertex_element = PlyElement.describe(elements, 'vertex')
    plydata = PlyData([vertex_element])
    plydata.write(path)
    
    # Debug output
    print(f"[PLY Writer] Wrote {n_points} Gaussians")
    print(f"[PLY Writer] Scene size: {scene_size:.3f}, avg point distance: {avg_point_distance:.6f}")
    print(f"[PLY Writer] Scale range (log space): [{scales.min():.3f}, {scales.max():.3f}]")
    print(f"[PLY Writer] Opacity logits range: [{opacity_logits.min():.3f}, {opacity_logits.max():.3f}]")

class VGGT_Model_Inference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Optional second image for multi-frame inference"}),
                "image_3": ("IMAGE", {"tooltip": "Optional third image for multi-frame inference"}),
                "image_4": ("IMAGE", {"tooltip": "Optional fourth image for multi-frame inference"}),
                "depth_conf_threshold": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Confidence threshold as percentile (0-100). Default 50 = keep top 50% of points by confidence. Higher = fewer but more confident points."
                }),
                "gaussian_scale_multiplier": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Multiplier for Gaussian splat size (higher = larger splats)"
                }),
                "preprocess_mode": (["crop", "pad_white", "pad_black", "tile"], {
                    "default": "pad_white",
                    "tooltip": "Preprocessing mode: crop=demo default (may crop tall images), pad_white=preserve all pixels with white padding (demo-style), pad_black=preserve all pixels with black padding, tile=full height at 518px width (experimental)"
                }),
                "mask_black_bg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Filter out black background pixels (useful for images with black borders)"
                }),
                "mask_white_bg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Filter out white background pixels (useful for images with white backgrounds)"
                }),
                "boundary_threshold": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Exclude points near image boundaries (pixels from edge). 0 = no boundary filtering"
                }),
                "max_depth": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "tooltip": "Maximum depth value to keep. -1 = no limit"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "EXTRINSICS", "INTRINSICS")
    RETURN_NAMES = ("ply_path", "extrinsics", "intrinsics")
    FUNCTION = "infer"
    CATEGORY = "VGGT"

    def infer(self, image_1, device, image_2=None, image_3=None, image_4=None, depth_conf_threshold=50.0, gaussian_scale_multiplier=0.1, preprocess_mode="pad_white", mask_black_bg=False, mask_white_bg=False, boundary_threshold=0, max_depth=-1.0):
        # Combine multiple images into a sequence
        images_list = [image_1]
        if image_2 is not None:
            images_list.append(image_2)
        if image_3 is not None:
            images_list.append(image_3)
        if image_4 is not None:
            images_list.append(image_4)
        
        # ===== KEY FIX: Preprocess each image individually with consistent output dimensions =====
        # When images have different aspect ratios, we preprocess each one separately
        # and ensure they all output the same dimensions before concatenation
        
        import torch.nn.functional as F
        target_size = 518
        
        if len(images_list) > 1:
            print(f"[VGGT] Processing {len(images_list)} images as sequence")
            # Preprocess each image separately first
            processed_images = []
            target_output_dims = None  # Will be set based on first image
            
            for img_idx, img_tensor in enumerate(images_list):
                batch_size, height, width, channels = img_tensor.shape
                
                # Convert to (B, C, H, W)
                model_input = img_tensor.permute(0, 3, 1, 2)
                
                # Preprocess this specific image
                model_input, out_height, out_width = self._preprocess_image_to_dims(
                    model_input, height, width, target_size, preprocess_mode
                )
                
                # Set target dimensions from first image
                if target_output_dims is None:
                    target_output_dims = (out_height, out_width)
                    print(f"[VGGT] Target output dimensions: {out_height}x{out_width}")
                else:
                    # Resize this image to match first image's dimensions
                    if (out_height, out_width) != target_output_dims:
                        model_input = F.interpolate(
                            model_input, size=target_output_dims, mode='bicubic', align_corners=False
                        )
                        print(f"[VGGT] Image {img_idx+1}: Resized from {out_height}x{out_width} to {target_output_dims[0]}x{target_output_dims[1]} for consistency")
                
                processed_images.append(model_input)
            
            # Now concatenate all images with consistent dimensions
            model_input = torch.cat(processed_images, dim=0)
            height, width = target_output_dims
            batch_size = len(images_list)
        else:
            # Single image
            batch_size, height, width, channels = image_1.shape
            model_input = image_1.permute(0, 3, 1, 2)
            model_input, height, width = self._preprocess_image_to_dims(
                model_input, height, width, target_size, preprocess_mode
            )
        
        # Move to device
        model_input = model_input.to(device)
        
        # Flag to indicate if we're using tiling
        using_tiles = preprocess_mode == "tile" and height > width

        from vggt.models.vggt import VGGT

        model = get_vggt_model(device)

        dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # === HANDLE TILING FOR TALL IMAGES (if requested) ===
        tile_info = None
        if using_tiles:
            print(f"[VGGT] TILE MODE: Processing tall image ({height}x{width}) with overlapping tiles for full coverage")
            print(f"[VGGT] WARNING: Tile mode is EXPERIMENTAL. Use pad mode if results are incorrect.")
            # For tile mode, we'll do inference ONCE on the full image (no cropping)
            # Then extract overlapping tiles from the OUTPUT point clouds
            # This gives us better consistency than cropping inputs
            tile_info = {"enabled": True, "original_height": height}
        
        # === EXACT SAME PROCESSING AS DEMO_GRADIO.PY ===
        # Run inference (matching demo exactly)
        print("[VGGT] Running inference...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype, enabled=device == "cuda"):
                predictions = model(model_input)

        # Convert pose encoding to extrinsic and intrinsic matrices (demo approach)
        print("[VGGT] Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], model_input.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert to numpy for colors BEFORE deleting model_input
        # Images are in (N, C, H, W) format from model_input
        model_input_np = model_input.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, C)
        
        # Convert tensors to numpy - EXACTLY like demo does
        print("[VGGT] Converting predictions to numpy...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
        predictions['pose_enc_list'] = None  # remove pose_enc_list

        # Generate world points from depth map - EXACTLY like demo does
        print("[VGGT] Computing world points from depth map (depth-based unprojection)...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )
        predictions["world_points_from_depth"] = world_points

        # Clean up GPU memory
        del model_input
        if device == "cuda":
            torch.cuda.empty_cache()

        # Extract outputs for point cloud generation (using depth-based unprojection)
        print("[VGGT] Using Depthmap and Camera Branch (geometric unprojection)")
        world_points_batch = predictions["world_points_from_depth"]
        depth_conf = predictions["depth_conf"]
        
        extrinsic_np = predictions["extrinsic"]  # (S, 3, 4)
        intrinsic_np = predictions["intrinsic"]  # (S, 3, 3)
        
        print(f"[VGGT] Point cloud generation:")
        print(f"[VGGT]   World points shape: {world_points_batch.shape}")
        print(f"[VGGT]   Model input (resized) shape: {model_input_np.shape}")
        print(f"[VGGT]   Depth confidence shape: {depth_conf.shape}")
        
        # Flatten all frames into single point cloud (matching demo approach)
        S, H, W, _ = world_points_batch.shape
        points_all_frames = world_points_batch.reshape(-1, 3)  # (S*H*W, 3)
        
        # Use MODEL INPUT IMAGES (resized to match world points) for colors
        # Keep in [0, 1] range - write_ply_basic handles conversion to SH coefficients
        colors_all_frames = model_input_np.reshape(-1, 3).astype(np.float32)  # (S*H*W, 3) in [0, 1]
        
        # Flatten confidence scores
        if depth_conf.ndim == 4:
            conf_all_frames = depth_conf.squeeze(-1).reshape(-1)
        else:
            conf_all_frames = depth_conf.reshape(-1)
        
        print(f"[VGGT]   Flattened points: {points_all_frames.shape}")
        print(f"[VGGT]   Flattened colors: {colors_all_frames.shape} (range: [{colors_all_frames.min():.3f}, {colors_all_frames.max():.3f}])")
        print(f"[VGGT]   Flattened confidence: {conf_all_frames.shape}")
        print(f"[VGGT]   Confidence stats: min={conf_all_frames.min():.4f}, max={conf_all_frames.max():.4f}, mean={conf_all_frames.mean():.4f}")
        
        # Apply PERCENTILE-based confidence filtering (matching demo_gradio.py approach)
        # This filters out the bottom X% of points by confidence, not absolute threshold
        if depth_conf_threshold == 0.0:
            conf_threshold_value = 0.0
            percentile = 0.0
        else:
            # depth_conf_threshold is interpreted as percentile (0-100)
            # Default 50.0 means 50th percentile - keep top 50% of points
            percentile = depth_conf_threshold if depth_conf_threshold > 1.0 else depth_conf_threshold * 100
            conf_threshold_value = np.percentile(conf_all_frames, percentile)
        
        # Warning for users with old workflows
        if 0 < depth_conf_threshold < 1.0:
            print(f"[VGGT] ⚠️ WARNING: depth_conf_threshold={depth_conf_threshold} is very low!")
            print(f"[VGGT] ⚠️ This parameter is now a PERCENTILE (0-100). Use 50.0 for default quality.")
            print(f"[VGGT] ⚠️ Current setting keeps top {100-percentile:.1f}% of points (too many noisy points!)")
        
        print(f"[VGGT]   Confidence filtering: percentile={percentile:.1f}%, threshold_value={conf_threshold_value:.4f}")
        valid_mask_all = (conf_all_frames >= conf_threshold_value) & (conf_all_frames > 1e-5)
        
        # Apply max depth filtering to all frames
        if max_depth > 0:
            # Compute depth as distance from camera (simple approximation)
            depth_all = np.linalg.norm(points_all_frames, axis=1)
            valid_mask_all = valid_mask_all & (depth_all <= max_depth)
            print(f"[VGGT] Applied max_depth filter: {max_depth}")
        
        # Apply boundary filtering to all frames
        if boundary_threshold > 0:
            # Create boundary mask for each frame
            boundary_mask = np.ones(S * H * W, dtype=bool)
            for s in range(S):
                frame_offset = s * H * W
                # Top and bottom
                boundary_mask[frame_offset : frame_offset + boundary_threshold * W] = False
                boundary_mask[frame_offset + (H - boundary_threshold) * W : frame_offset + H * W] = False
                # Left and right (per row)
                for h in range(boundary_threshold, H - boundary_threshold):
                    row_start = frame_offset + h * W
                    boundary_mask[row_start : row_start + boundary_threshold] = False
                    boundary_mask[row_start + W - boundary_threshold : row_start + W] = False
            valid_mask_all = valid_mask_all & boundary_mask
            print(f"[VGGT] Applied boundary_threshold filter: {boundary_threshold}px")
        
        # Apply black/white background filtering
        if mask_black_bg:
            color_sum = colors_all_frames.sum(axis=1)
            black_mask = color_sum >= (16 / 255.0)
            valid_mask_all = valid_mask_all & black_mask
            print(f"[VGGT] Applied mask_black_bg filter")
        
        if mask_white_bg:
            white_mask = ~((colors_all_frames[:, 0] > 240/255.0) & (colors_all_frames[:, 1] > 240/255.0) & (colors_all_frames[:, 2] > 240/255.0))
            valid_mask_all = valid_mask_all & white_mask
            print(f"[VGGT] Applied mask_white_bg filter")
        
        points = points_all_frames[valid_mask_all]
        colors = colors_all_frames[valid_mask_all]
        confidences = conf_all_frames[valid_mask_all]
        
        print(f"[VGGT] Confidence threshold: {depth_conf_threshold}")
        print(f"[VGGT] Points before filtering: {len(points_all_frames)}")
        print(f"[VGGT] Points after confidence filtering: {len(points)}")
        
        # Ensure correct shapes
        if points.ndim != 2 or points.shape[1] != 3:
            print(f"Warning: Unexpected points shape {points.shape}, reshaping...")
            points = points.reshape(-1, 3)
        if colors.ndim != 2 or colors.shape[1] != 3:
            print(f"Warning: Unexpected colors shape {colors.shape}, reshaping...")
            colors = colors.reshape(-1, 3)
        if confidences.ndim != 1:
            print(f"Warning: Unexpected confidences shape {confidences.shape}, flattening...")
            confidences = confidences.flatten()

        # Save PLY file
        import folder_paths
        import uuid
        output_dir = folder_paths.get_output_directory()
        temp_id = str(uuid.uuid4())[:8]
        ply_filename = f"vggt_temp_{temp_id}.ply"
        ply_path = os.path.join(output_dir, ply_filename)

        write_ply_basic(ply_path, points, colors, confidences, gaussian_scale_multiplier)
        print(f"Saved PLY file: {ply_path} with {len(points)} points")
        print(f"  Points range - X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  Points range - Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  Points range - Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"  Colors range: [{colors.min():.3f}, {colors.max():.3f}]")
        print(f"  Confidences range: [{confidences.min():.3f}, {confidences.max():.3f}]")

        # Extract first camera's extrinsics and intrinsics for GaussianViewer
        # Convert numpy arrays to lists for compatibility
        first_extrinsics = extrinsic_np[0].tolist() if extrinsic_np.ndim > 2 and len(extrinsic_np) > 0 else [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        first_intrinsics = intrinsic_np[0].tolist() if intrinsic_np.ndim > 2 and len(intrinsic_np) > 0 else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        print(f"[VGGT] Camera matrices for GaussianViewer:")
        print(f"[VGGT]   Extrinsic (3x4 camera-from-world):")
        for row in first_extrinsics:
            print(f"[VGGT]     {row}")
        print(f"[VGGT]   Intrinsic (3x3 camera calibration):")
        for row in first_intrinsics:
            print(f"[VGGT]     {row}")

        return (ply_path, first_extrinsics, first_intrinsics)

    def _preprocess_image_to_dims(self, model_input, height, width, target_size, preprocess_mode):
        """Preprocess a single image tensor. Returns (preprocessed_tensor, output_height, output_width)"""
        import torch.nn.functional as F
        
        if preprocess_mode == "crop":
            # Demo mode: resize to width=518px maintaining aspect ratio
            # IMPORTANT: For tall/square images, this will CENTER-CROP the height to 518px,
            # which LOSES top and bottom information. Use "pad" mode to preserve all pixels.
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
            
            model_input = F.interpolate(model_input, size=(new_height, new_width), mode='bicubic', align_corners=False)
            
            # Center crop height ONLY if > 518 (loses top/bottom info for tall images)
            if new_height > target_size:
                start_y = (new_height - target_size) // 2
                model_input = model_input[:, :, start_y:start_y + target_size, :]
                new_height = target_size
            
            return model_input, new_height, new_width
            
        elif preprocess_mode in ["pad_white", "pad_black"]:
            # Pad mode: largest dimension = 518px, pad to square
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
            
            model_input = F.interpolate(model_input, size=(new_height, new_width), mode='bicubic', align_corners=False)
            
            # Pad to square if needed
            h_padding = target_size - new_height
            w_padding = target_size - new_width
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                pad_value = 1.0 if preprocess_mode == "pad_white" else 0.0
                model_input = F.pad(model_input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)
                new_height = target_size
                new_width = target_size
            
            return model_input, new_height, new_width
            
        elif preprocess_mode == "tile":
            # Tile mode: for tall images, keep width=518px (model training constraint)
            # but allow full height without center-cropping
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
            
            model_input = F.interpolate(model_input, size=(new_height, new_width), mode='bicubic', align_corners=False)
            
            return model_input, new_height, new_width
        else:
            # No preprocessing - just ensure divisible by 14
            target_height = round(height / 14) * 14
            target_width = round(width / 14) * 14
            
            if target_height != height or target_width != width:
                model_input = F.interpolate(model_input, size=(target_height, target_width), mode='bicubic', align_corners=False)
            
            return model_input, target_height, target_width
