#!/usr/bin/env python3
"""
Extract dataset from output folder for LoRA training.

This script reorganizes the output from build_warp_dataset.py into a format
suitable for training LoRA models with ModelScope.

Structure:
  output/
    scene1/
      image1_splats.jpg
      image1_reference.jpg
      image1_target.jpg
      ...
    scene2/
      ...

Output:
  modelscope-dataset/
    1_start_1.jpg
    1_start_2.jpg
    1_end.jpg
    2_start_1.jpg
    2_start_2.jpg
    2_end.jpg
    ...
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

from dataset_utils import (
    find_triplets_in_scene,
    get_file_extension,
    HAS_PIL,
    validate_image_file,
)


def extract_dataset(
    output_dir: Path,
    modelscope_dir: Path,
    prompt: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Extract dataset from output folder for ModelScope LoRA training.
    
    Args:
        output_dir: Path to output folder from build_warp_dataset.py
        modelscope_dir: Path to output modelscope-dataset folder
        prompt: Optional prompt text to save for each triplet
        verbose: Print progress information
    """
    # Create output directory
    modelscope_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Extracting dataset to {modelscope_dir}")
    
    folder_counter = 1
    total_triplets = 0
    
    # Iterate through scene folders
    scene_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        
        # Skip hidden directories and cache directories
        if scene_name.startswith("."):
            continue
        
        if verbose:
            print(f"Processing {scene_name}...")
        
        triplets = find_triplets_in_scene(scene_dir)
        
        # Skip if no complete triplets found
        if not triplets:
            if verbose:
                print(f"  Warning: No complete triplets found in {scene_name}, skipping")
            continue
        
        # Process each triplet
        for files in triplets:
            # Get file extension (use the first available file's extension)
            ext = get_file_extension(files["splats"])
            
            # Copy files with new names
            splats_dest = modelscope_dir / f"{folder_counter}_start_1{ext}"
            reference_dest = modelscope_dir / f"{folder_counter}_start_2{ext}"
            target_dest = modelscope_dir / f"{folder_counter}_end{ext}"
            
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
            
            shutil.copy2(files["splats"], splats_dest)
            shutil.copy2(files["reference"], reference_dest)
            shutil.copy2(files["target"], target_dest)
            
            # Write prompt file if provided
            if prompt is not None:
                prompt_dest = modelscope_dir / f"{folder_counter}.txt"
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    f.write(prompt)
            
            if verbose:
                print(f"  Created triplet {folder_counter} from {files['stem']}:")
                print(f"    - {splats_dest.name}")
                print(f"    - {reference_dest.name}")
                print(f"    - {target_dest.name}")
                if prompt is not None:
                    print(f"    - {prompt_dest.name}")
            
            folder_counter += 1
            total_triplets += 1
    
    if verbose:
        print(f"\nExtraction complete!")
        print(f"Created {total_triplets} training triplets in {modelscope_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract dataset from build_warp_dataset.py output for ModelScope LoRA training."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory with scene outputs from build_warp_dataset.py (default: output)",
    )
    parser.add_argument(
        "--modelscope-dir",
        type=Path,
        default=Path("modelscope-dataset"),
        help="Output directory for ModelScope dataset (default: modelscope-dataset)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to save for each training triplet (saved as <N>.txt)",
    )
    
    args = parser.parse_args()
    
    # Validate output directory
    if not args.output_dir.exists():
        print(f"Error: Output directory not found: {args.output_dir}")
        return 1
    
    if not args.output_dir.is_dir():
        print(f"Error: Output path is not a directory: {args.output_dir}")
        return 1
    
    extract_dataset(
        args.output_dir,
        args.modelscope_dir,
        prompt=args.prompt,
        verbose=not args.quiet,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
