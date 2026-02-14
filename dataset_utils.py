#!/usr/bin/env python3
"""
Shared utilities for dataset extraction scripts.

This module contains common functions used by multiple dataset extraction
scripts (aitoolkit-dataset.py, modelscope-dataset.py) to eliminate code duplication.
"""

from pathlib import Path
from typing import Optional

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False


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
        elif (name.endswith("_reference.jpg") or name.endswith("_reference.png")):
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
