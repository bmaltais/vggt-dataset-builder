#!/usr/bin/env python3
"""
Shared utilities for dataset extraction and model inference scripts.

This module contains common functions used by multiple scripts including:
- aitoolkit-dataset.py, modelscope-dataset.py (dataset extraction)
- build_warp_dataset.py, vggt_point_cloud_viewer.py (model inference)

Functions are organized to eliminate code duplication across the codebase.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None
    HAS_TORCH = False

try:
    from vggt.models.vggt import VGGT

    HAS_VGGT = True
except (ImportError, OSError):
    VGGT = None
    HAS_VGGT = False


class FrameCacheManager:
    """Manages frame caching for VGGT dataset processing.

    Centralizes all cache-related operations: manifest validation, frame data
    loading/saving, and cache invalidation on processing args changes.

    This class encapsulates the previously scattered caching logic in build_warp_dataset.py,
    providing a clean API for cache operations and improving testability.

    Attributes:
        scene_cache_dir: Path to the scene's cache directory.
        args_hash: SHA1 hash of current processing args for cache invalidation.
        manifest: Dict mapping image stems to their SHA1 signatures (for validation).
    """

    def __init__(self, scene_cache_dir: Path, args_hash: str) -> None:
        """Initialize cache manager for a scene.

        Args:
            scene_cache_dir: Path to create/use for scene cache.
            args_hash: SHA1 hash of current processing args. Cache is invalidated
                       if args change when reloading manifest.

        Returns:
            None
        """
        self.scene_cache_dir = scene_cache_dir
        self.args_hash = args_hash
        self.manifest_path = scene_cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load manifest from disk, invalidating cache if args_hash changed.

        Returns:
            Dict mapping image stems to SHA1 signatures, or empty dict if invalid/missing.
        """
        if not self.manifest_path.exists():
            return {}

        try:
            with self.manifest_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

            # If args_hash changed, invalidate entire cache
            if data.get("args_hash") != self.args_hash:
                self.clear()
                return {}

            return data.get("images", {}) or {}
        except Exception:
            return {}

    def _save_manifest(self) -> None:
        """Save manifest (args_hash + image signatures) to disk."""
        try:
            self.scene_cache_dir.mkdir(parents=True, exist_ok=True)
            with self.manifest_path.open("w", encoding="utf-8") as fh:
                json.dump({"args_hash": self.args_hash, "images": self.manifest}, fh)
        except Exception:
            pass

    def _image_signature(self, path: Path) -> str:
        """Compute SHA1 signature of image file contents for change detection.

        Args:
            path: Path to the image file.

        Returns:
            SHA1 hexdigest, or empty string on error (treated as invalid).
        """
        try:
            h = hashlib.sha1()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    def get_cache_path(self, image_path: Path) -> Path:
        """Get the NPZ cache file path for an image.

        Args:
            image_path: Path to the source image.

        Returns:
            Path to the corresponding .npz cache file in scene_cache_dir.
        """
        return self.scene_cache_dir / f"{image_path.stem}.npz"

    def is_frame_cached(self, image_path: Path, validate_file: bool = True) -> bool:
        """Check if frame is cached and valid (file exists + signature matches).

        Args:
            image_path: Path to the image file.
            validate_file: If True, validates both manifest entry and file existence.
                          If False, only checks if manifest entry exists.

        Returns:
            True if frame is cached and valid, False otherwise.
        """
        cache_path = self.get_cache_path(image_path)

        if not cache_path.exists():
            return False

        if not validate_file:
            return True

        # Verify signature matches to ensure image hasn't changed
        sig = self._image_signature(image_path)
        if not sig:
            return False

        cached_sig = self.manifest.get(image_path.stem)
        return cached_sig == sig

    def load_frame_data(self, image_path: Path) -> dict | None:
        """Load frame data from cache.

        Args:
            image_path: Path to the image file.

        Returns:
            Frame data dict with keys 'points', 'colors', 'confidences' (required),
            and optionally 's0' and 'conf_image'. Returns None if not cached or invalid.

        Raises:
            None: Errors are caught and None is returned.
        """
        if not self.is_frame_cached(image_path):
            return None

        cache_path = self.get_cache_path(image_path)
        try:
            data = np.load(cache_path, allow_pickle=False)
            frame_data = {
                "points": data["points"],
                "colors": data["colors"],
                "confidences": data["confidences"],
            }
            if "s0" in data:
                frame_data["s0"] = float(data["s0"].tolist())
            if "conf_image" in data:
                frame_data["conf_image"] = Image.fromarray(
                    data["conf_image"].astype(np.uint8)
                )
            return frame_data
        except Exception:
            return None

    def save_frame_data(self, image_path: Path, frame_data: dict) -> None:
        """Save frame data to cache and update manifest.

        Args:
            image_path: Path to the image file.
            frame_data: Dict with keys 'points', 'colors', 'confidences' (required),
                       and optionally 's0' (float) and 'conf_image' (PIL Image).

        Returns:
            None

        Raises:
            None: Errors are caught and silently ignored (cache not critical).
        """
        cache_path = self.get_cache_path(image_path)

        try:
            self.scene_cache_dir.mkdir(parents=True, exist_ok=True)

            arrs = {
                "points": frame_data["points"],
                "colors": frame_data["colors"],
                "confidences": frame_data["confidences"],
            }
            if "s0" in frame_data:
                arrs["s0"] = np.array(float(frame_data["s0"]))
            if "conf_image" in frame_data:
                arrs["conf_image"] = np.array(frame_data["conf_image"], dtype=np.uint8)

            np.savez_compressed(cache_path, **arrs)

            # Update manifest with image signature
            sig = self._image_signature(image_path)
            if sig:
                self.manifest[image_path.stem] = sig
                self._save_manifest()
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all cached data for this scene.

        Returns:
            None
        """
        try:
            if self.scene_cache_dir.exists():
                shutil.rmtree(self.scene_cache_dir)
        except Exception:
            pass
        self.manifest = {}


class PointCloudFilter:
    """Manages point cloud filtering operations for depth-based reconstruction.

    This class encapsulates multiple filtering strategies to clean up point clouds
    generated from depth maps. Filtering can occur at two stages:
    1. Pre-unprojection: Applying masks to confidence maps (e.g., sky segmentation)
    2. Post-unprojection: Filtering 3D points based on color (e.g., background removal)

    The class manages both the filtering configuration and runtime state (sky
    segmentation session, mask caching directory), eliminating the need for
    scattered procedural functions and external state management.

    Attributes:
        filter_sky: Enable sky segmentation filtering.
        filter_black_bg: Filter black background points (RGB sum < 16).
        filter_white_bg: Filter white background points (RGB > 240 per channel).
        sky_masks_dir: Directory for caching sky segmentation masks.
        skyseg_session: ONNX session for sky segmentation model (optional).

    Example:
        >>> filter_cfg = PointCloudFilter(
        ...     filter_sky=True,
        ...     filter_black_bg=True,
        ...     sky_masks_dir=Path("output/scene/sky_masks"),
        ...     skyseg_session=onnx_session
        ... )
        >>> # Apply sky filter to confidence map
        >>> conf_filtered = filter_cfg.apply_confidence_filter(conf_map, image_path)
        >>> # Later, filter colors after unprojection
        >>> color_mask = filter_cfg.apply_color_filter(point_colors)
    """

    def __init__(
        self,
        filter_sky: bool = False,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        sky_masks_dir: Optional[Path] = None,
        skyseg_session=None,
    ) -> None:
        """Initialize the point cloud filter with configuration and resources.

        Args:
            filter_sky: If True, use sky segmentation to filter sky pixels.
            filter_black_bg: If True, filter black background points (RGB sum < 16).
            filter_white_bg: If True, filter white background points (RGB > 240).
            sky_masks_dir: Directory to cache sky segmentation masks (required if filter_sky=True).
            skyseg_session: ONNX runtime session for sky segmentation (required if filter_sky=True).

        Returns:
            None
        """
        self.filter_sky = bool(filter_sky)
        self.filter_black_bg = bool(filter_black_bg)
        self.filter_white_bg = bool(filter_white_bg)
        self.sky_masks_dir = sky_masks_dir
        self.skyseg_session = skyseg_session

        # Validate configuration
        if self.filter_sky and self.sky_masks_dir is not None:
            self.sky_masks_dir.mkdir(parents=True, exist_ok=True)

    def has_confidence_filters(self) -> bool:
        """Check if any confidence-stage filters are enabled.

        Returns:
            True if sky filtering is enabled, False otherwise.
        """
        return self.filter_sky

    def has_color_filters(self) -> bool:
        """Check if any color-stage filters are enabled.

        Returns:
            True if black or white background filtering is enabled.
        """
        return self.filter_black_bg or self.filter_white_bg

    def apply_confidence_filter(
        self, conf_frame: "np.ndarray", image_path: Path
    ) -> "np.ndarray":
        """Apply filtering to confidence map (pre-unprojection stage).

        Currently applies sky segmentation if enabled. The confidence map is
        multiplied by a binary mask where 0 = sky (filtered) and 1 = valid.

        Args:
            conf_frame: 2D confidence map (H x W), float32.
            image_path: Path to the source image (used for loading/caching sky masks).

        Returns:
            Filtered confidence map with same shape as input. Sky pixels
            will have confidence set to 0.

        Raises:
            ImportError: If sky filtering is enabled but opencv or onnxruntime unavailable.
        """
        if not self.filter_sky:
            return conf_frame

        if not HAS_NUMPY:
            raise ImportError("numpy is required for confidence filtering")

        # Import cv2 and segment_sky on-demand to avoid hard dependency
        try:
            import cv2
            from vggt.visual_util import segment_sky
        except ImportError as exc:
            print(
                "Warning: Sky filtering requires opencv-python and onnxruntime. Skipping."
            )
            return conf_frame

        if self.skyseg_session is None or self.sky_masks_dir is None:
            print("Warning: Sky filtering enabled but session/directory missing.")
            return conf_frame

        # Load or generate sky mask
        image_name = image_path.name
        mask_path = self.sky_masks_dir / image_name

        if mask_path.exists():
            sky_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(str(image_path), self.skyseg_session, str(mask_path))

        # Resize mask to match confidence map if needed
        H, W = conf_frame.shape
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        # Apply mask (segment_sky returns 255 for non-sky, 0 for sky)
        sky_mask_binary = (sky_mask > 128).astype(np.float32)
        return conf_frame * sky_mask_binary

    def apply_color_filter(self, colors: "np.ndarray") -> "np.ndarray":
        """Apply filtering to point colors (post-unprojection stage).

        Filters black and/or white background points based on RGB values.
        This is applied after points are unprojected to 3D space.

        Args:
            colors: Nx3 array of RGB colors in [0, 1] range (float32).

        Returns:
            Boolean mask of shape (N,) where True = keep point, False = filter.
            All True if no color filters are enabled.

        Raises:
            ImportError: If numpy is not available.
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for color filtering")

        mask = np.ones(colors.shape[0], dtype=bool)

        if not (self.filter_black_bg or self.filter_white_bg):
            return mask

        # Filter black background: RGB sum < 16/255 approx 0.0627 in 0-1 float range
        if self.filter_black_bg:
            mask &= colors.sum(axis=1) >= (16 / 255.0)

        # Filter white background: All channels >= 241/255
        if self.filter_white_bg:
            threshold = 241.0 / 255.0
            mask &= ~(
                (colors[:, 0] >= threshold)
                & (colors[:, 1] >= threshold)
                & (colors[:, 2] >= threshold)
            )

        return mask


def setup_vggt_path() -> None:
    """Add the local vggt submodule to sys.path for imports to work.

    Ensures that when a script imports VGGT components (e.g., 'from vggt.models.vggt import VGGT'),
    the local vggt/ submodule directory is in the search path. This function is idempotent
    (safe to call multiple times) and thread-safe.

    This utility eliminates code duplication across scripts that all need to set up
    the vggt module path:
    - build_warp_dataset.py
    - vggt_point_cloud_viewer.py
    - vggt_comfy_nodes.py

    Returns:
        None

    Example:
        >>> setup_vggt_path()
        >>> from vggt.models.vggt import VGGT  # This will now work
    """
    vggt_path = str(Path(__file__).parent / "vggt")
    if vggt_path not in sys.path:
        sys.path.insert(0, vggt_path)


def get_file_extension(pattern_path: Optional[Path]) -> str:
    """Get file extension from a path, or return empty string if None.

    Args:
        pattern_path: Path object to extract extension from, or None.

    Returns:
        File extension (e.g., '.jpg', '.png') in lowercase, or empty string if path is None.

    Example:
        >>> path = Path("image.jpg")
        >>> get_file_extension(path)
        '.jpg'
    """
    if pattern_path is None:
        return ""
    return pattern_path.suffix.lower()


def find_triplets_in_scene(scene_dir: Path) -> list[dict]:
    """Find all triplets in a scene directory.

    Scans a directory for matching sets of three image files that form a training triplet:
    - *_splats.<ext>: Splat rendering
    - *_reference.<ext>: Reference image
    - *_target.<ext>: Target image

    All three files must share the same stem (base name before the suffix).
    Only .jpg and .png files are matched; .ply, .txt, and other file types are ignored.

    Args:
        scene_dir: Path to the scene directory to scan for triplets.

    Returns:
        A list of dicts, each containing:
            - 'splats': Path to *_splats.<ext> file
            - 'reference': Path to *_reference.<ext> file (excluding .ply)
            - 'target': Path to *_target.<ext> file
            - 'stem': Base name without suffix (e.g., 'image1')

        Only complete triplets (all three files present) are returned.
        The list is sorted by stem name for consistent ordering.

    Example:
        Given a directory with:
            - image1_splats.jpg
            - image1_reference.jpg
            - image1_target.jpg
            - image2_splats.png  (incomplete, missing reference/target)

        Returns:
            [{'splats': Path('image1_splats.jpg'),
              'reference': Path('image1_reference.jpg'),
              'target': Path('image1_target.jpg'),
              'stem': 'image1'}]
    """
    triplets = {}

    for file_path in scene_dir.iterdir():
        if not file_path.is_file():
            continue

        name = file_path.name

        # Extract the stem (e.g., "image1" from "image1_splats.jpg")
        # Only match specific suffixes to avoid metadata files like _reference_intrinsics.txt
        if name.endswith("_splats.jpg") or name.endswith("_splats.png"):
            stem = name.replace("_splats.jpg", "").replace("_splats.png", "")
            if stem not in triplets:
                triplets[stem] = {}
            triplets[stem]["splats"] = file_path
            triplets[stem]["stem"] = stem
        elif name.endswith("_reference.jpg") or name.endswith("_reference.png"):
            # Only match JPG and PNG, NOT PLY or TXT files
            stem = name.replace("_reference.jpg", "").replace("_reference.png", "")
            if stem not in triplets:
                triplets[stem] = {}
            triplets[stem]["reference"] = file_path
            triplets[stem]["stem"] = stem
        elif name.endswith("_target.jpg") or name.endswith("_target.png"):
            stem = name.replace("_target.jpg", "").replace("_target.png", "")
            if stem not in triplets:
                triplets[stem] = {}
            triplets[stem]["target"] = file_path
            triplets[stem]["stem"] = stem

    # Filter to only complete triplets and return as list
    complete_triplets = []
    for stem, files in sorted(triplets.items()):
        if "splats" in files and "reference" in files and "target" in files:
            complete_triplets.append(files)

    return complete_triplets


def validate_image_file(file_path: Path, file_type: str = "image") -> None:
    """Validate that an image file is readable using PIL.

    Args:
        file_path: Path to the image file to validate.
        file_type: Descriptive name for the file type (used in error messages).

    Raises:
        ImportError: If PIL is not installed.
        ValueError: If the image cannot be read or is invalid.

    Example:
        >>> validate_image_file(Path("image.jpg"), "splats")
    """
    if not HAS_PIL:
        raise ImportError("PIL (Pillow) is required for image validation")

    try:
        with Image.open(file_path) as img:
            # Force load to verify the file is valid
            img.load()
    except Exception as e:
        raise ValueError(f"Cannot read {file_type} image: {file_path} - {e}") from e


def extract_dataset_generic(
    output_dir: Path,
    target_base_dir: Path,
    naming_fn: Callable[[int, str], dict[str, Path]],
    prompt: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Generic dataset extraction function for dataset preparation scripts.

    Eliminates code duplication across aitoolkit-dataset.py and modelscope-dataset.py
    by providing a reusable extraction pipeline that:

    1. Iterates through scene directories
    2. Finds triplets of images in each scene
    3. Validates image files are readable
    4. Copies files to output locations with custom naming
    5. Optionally writes prompt files
    6. Tracks progress with verbose output

    Args:
        output_dir: Input directory with scene outputs from build_warp_dataset.py
        target_base_dir: Base output directory for the dataset
        naming_fn: Callable that takes (folder_counter, file_extension) and returns
                   a dict with keys 'splats', 'reference', 'target' (required) and
                   optionally 'prompt'. Each value should be a Path object where
                   the file should be written.
        prompt: Optional prompt text to save for each training triplet (if naming_fn
                provides a 'prompt' key, the prompt will be written to that path).
        verbose: Print progress information to stdout.

    Returns:
        Total number of training triplets extracted.

    Raises:
        ValueError: If an image file fails validation.

    Example:
        >>> def naming_fn(counter, ext):
        ...     return {
        ...         'splats': target_dir / f'{counter}{ext}',
        ...         'reference': target_dir / f'{counter}{ext}',
        ...         'target': target_dir / f'{counter}{ext}',
        ...     }
        >>> extract_dataset_generic(output_dir, target_dir, naming_fn, verbose=True)
    """
    folder_counter = 1
    total_triplets = 0

    if verbose:
        print(f"Extracting dataset to {target_base_dir}")

    # Iterate through scene folders
    scene_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name

        # Skip hidden directories
        if scene_name.startswith("."):
            continue

        if verbose:
            print(f"Processing {scene_name}...")

        triplets = find_triplets_in_scene(scene_dir)

        if not triplets:
            if verbose:
                print(
                    f"  Warning: No complete triplets found in {scene_name}, skipping"
                )
            continue

        # Process each triplet
        for files in triplets:
            ext = get_file_extension(files["splats"])

            # Get output paths from the naming function
            paths = naming_fn(folder_counter, ext)

            # Create parent directories as needed
            for path_dict_key, path in paths.items():
                if path_dict_key != "prompt":  # Don't create dir for prompt yet
                    path.parent.mkdir(parents=True, exist_ok=True)

            # Validate image files are readable before copying
            if HAS_PIL:
                for src_file, dest_name in [
                    (files["splats"], "splats"),
                    (files["reference"], "reference"),
                    (files["target"], "target"),
                ]:
                    try:
                        validate_image_file(src_file, dest_name)
                    except ValueError as e:
                        print(f"  ERROR: Invalid image file {src_file.name}: {e}")
                        raise

            # Copy files to output locations
            shutil.copy2(files["splats"], paths["splats"])
            shutil.copy2(files["reference"], paths["reference"])
            shutil.copy2(files["target"], paths["target"])

            # Write prompt file if provided and naming_fn includes 'prompt' key
            if prompt is not None and "prompt" in paths:
                prompt_path = paths["prompt"]
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt)

            if verbose:
                print(f"  Created triplet {folder_counter} from {files['stem']}:")
                for key in ["splats", "reference", "target", "prompt"]:
                    if key in paths:
                        print(f"    - {paths[key].name}")

            folder_counter += 1
            total_triplets += 1

    if verbose:
        print(f"\nExtraction complete!")
        print(f"Created {total_triplets} training triplets in {target_base_dir}")

    return total_triplets


def load_model(device: "torch.device") -> "VGGT":
    """Load VGGT model for depth estimation.

    Loads the facebook/VGGT-1B model, moves it to the specified device,
    and sets it to evaluation mode. This function is used by both
    build_warp_dataset.py and vggt_point_cloud_viewer.py.

    Args:
        device: torch.device to load the model on (e.g., torch.device('cuda') or 'cpu').

    Returns:
        VGGT model in evaluation mode on the specified device.

    Raises:
        ImportError: If torch or VGGT is not installed.

    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = load_model(device)
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for load_model()")

    # Lazy import VGGT in case the module-level import failed due to path issues
    # This allows setup_vggt_path() to be called first by the user script
    try:
        from vggt.models.vggt import VGGT as VGGT_Model
    except (ImportError, OSError) as e:
        raise ImportError(
            f"VGGT model unavailable. Make sure setup_vggt_path() was called first "
            f"and VGGT library is installed. Error: {e}"
        ) from e

    model = VGGT_Model.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model


def build_view_matrix(extrinsic: "np.ndarray") -> "np.ndarray":
    """Build a 4x4 view matrix from extrinsic (camera pose) matrix.

    Converts an extrinsic camera matrix (rotation + translation) to a view matrix
    using OpenGL coordinate system conventions. This function is used by both
    build_warp_dataset.py and vggt_point_cloud_viewer.py for point cloud rendering.

    The conversion applies:
    1. Identity + extrinsic orientation and position
    2. Coordinate system conversion: [1, -1, -1, 1] (RDF to RUB convention)

    Args:
        extrinsic: 4x4 extrinsic matrix or 3x4 camera pose matrix.
                  Should contain rotation in [:3, :3] and translation in [:3, 3].

    Returns:
        4x4 view matrix in OpenGL-compatible format (float32).

    Raises:
        ImportError: If numpy is not installed.

    Example:
        >>> extrinsic = np.eye(4)  # Identity pose
        >>> view = build_view_matrix(extrinsic)
        >>> view.shape
        (4, 4)
    """
    if not HAS_NUMPY:
        raise ImportError("numpy is required for build_view_matrix()")

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = extrinsic[:3, :3]
    view[:3, 3] = extrinsic[:3, 3]
    conversion = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return conversion @ view


def select_device(device_arg: Optional[str] = None) -> "torch.device":
    """Select torch device based on argument and CUDA availability.

    Provides consistent device selection logic across scripts. If device_arg is None,
    automatically detects CUDA availability. This function eliminates duplication
    between build_warp_dataset.py and vggt_point_cloud_viewer.py.

    Args:
        device_arg: Device string from command-line argument ('cuda', 'cpu', or None).
                   If None, auto-detects CUDA availability.

    Returns:
        torch.device object for the selected device.

    Raises:
        ImportError: If torch is not installed.

    Example:
        >>> device = select_device(None)  # Auto-detect
        >>> device = select_device('cuda')  # Force CUDA
        >>> device = select_device('cpu')  # Force CPU
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for select_device()")

    device_name = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def select_dtype(device: "torch.device") -> "torch.dtype":
    """Select appropriate torch dtype based on device capability.

    Selects bfloat16 for CUDA devices with compute capability >= 8.0 (e.g., A100, H100),
    otherwise uses float16. This provides optimal performance while maintaining
    compatibility across different GPU generations. Eliminates duplication between
    build_warp_dataset.py and vggt_point_cloud_viewer.py.

    Args:
        device: torch.device to check capabilities for.

    Returns:
        torch.bfloat16 for modern CUDA devices (compute capability >= 8),
        torch.float16 otherwise.

    Raises:
        ImportError: If torch is not installed.

    Example:
        >>> device = torch.device('cuda')
        >>> dtype = select_dtype(device)
        >>> # Returns torch.bfloat16 on A100/H100, torch.float16 on older GPUs
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for select_dtype()")

    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16
