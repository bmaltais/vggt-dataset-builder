#!/usr/bin/env python3
"""
Prepare dataset for AI Toolkit.

AI Toolkit expects three separate folders:
  - control1: splats images
  - control2: reference images
  - target: target images and prompts

Input structure (from build_warp_dataset.py):
  output/
    scene1/
      image1_splats.jpg
      image1_reference.jpg
      image1_target.jpg
      ...

Output structure:
  aitoolkit-dataset/
    control1/
      1.jpg
      2.jpg
      ...
    control2/
      1.jpg
      2.jpg
      ...
    target/
      1.jpg
      1.txt
      2.jpg
      2.txt
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
    aitoolkit_dir: Path,
    prompt: Optional[str] = None,
    verbose: bool = True,
) -> None:
    control1_dir = aitoolkit_dir / "control1"
    control2_dir = aitoolkit_dir / "control2"
    target_dir = aitoolkit_dir / "target"

    control1_dir.mkdir(parents=True, exist_ok=True)
    control2_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Extracting dataset to {aitoolkit_dir}")

    folder_counter = 1
    total_triplets = 0

    scene_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        if scene_name.startswith("."):
            continue

        if verbose:
            print(f"Processing {scene_name}...")

        triplets = find_triplets_in_scene(scene_dir)

        if not triplets:
            if verbose:
                print(f"  Warning: No complete triplets found in {scene_name}, skipping")
            continue

        for files in triplets:
            ext = get_file_extension(files["splats"])

            control1_dest = control1_dir / f"{folder_counter}{ext}"
            control2_dest = control2_dir / f"{folder_counter}{ext}"
            target_dest = target_dir / f"{folder_counter}{ext}"

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

            shutil.copy2(files["splats"], control1_dest)
            shutil.copy2(files["reference"], control2_dest)
            shutil.copy2(files["target"], target_dest)

            if prompt is not None:
                prompt_dest = target_dir / f"{folder_counter}.txt"
                with open(prompt_dest, "w", encoding="utf-8") as f:
                    f.write(prompt)

            if verbose:
                print(f"  Created triplet {folder_counter} from {files['stem']}:")
                print(f"    - control1/{control1_dest.name}")
                print(f"    - control2/{control2_dest.name}")
                print(f"    - target/{target_dest.name}")
                if prompt is not None:
                    print(f"    - target/{prompt_dest.name}")

            folder_counter += 1
            total_triplets += 1

    if verbose:
        print(f"\nExtraction complete!")
        print(f"Created {total_triplets} training triplets in {aitoolkit_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract dataset from build_warp_dataset.py output for AI Toolkit."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory with scene outputs from build_warp_dataset.py (default: output)",
    )
    parser.add_argument(
        "--aitoolkit-dir",
        type=Path,
        default=Path("aitoolkit-dataset"),
        help="Output directory for AI Toolkit dataset (default: aitoolkit-dataset)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to save for each training triplet (saved as target/<N>.txt)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: Output directory not found: {args.output_dir}")
        return 1

    if not args.output_dir.is_dir():
        print(f"Error: Output path is not a directory: {args.output_dir}")
        return 1

    extract_dataset(
        args.output_dir,
        args.aitoolkit_dir,
        prompt=args.prompt,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
