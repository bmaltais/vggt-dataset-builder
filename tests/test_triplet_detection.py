#!/usr/bin/env python3
"""Test script to verify triplet detection in aitoolkit and modelscope scripts."""

from pathlib import Path

def find_triplets_in_scene(scene_dir: Path) -> list[dict]:
    """Find all triplets in a scene directory."""
    triplets = {}
    
    for file_path in scene_dir.iterdir():
        if not file_path.is_file():
            continue
        
        name = file_path.name
        
        # Extract the stem (e.g., "image1" from "image1_splats.jpg")
        if "_splats" in name and file_path.suffix.lower() != ".ply":
            stem = name.split("_splats")[0]
            if stem not in triplets:
                triplets[stem] = {}
            triplets[stem]["splats"] = file_path
            triplets[stem]["stem"] = stem
        elif "_reference" in name and file_path.suffix.lower() != ".ply":
            stem = name.split("_reference")[0]
            if stem not in triplets:
                triplets[stem] = {}
            triplets[stem]["reference"] = file_path
            triplets[stem]["stem"] = stem
        elif "_target" in name and file_path.suffix.lower() != ".ply":
            stem = name.split("_target")[0]
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

def test_triplet_detection():
    """Test the triplet detection logic."""
    
    scene_dir = Path("output/01")
    
    print("Testing Triplet Detection")
    print("=" * 70)
    print(f"Scene: {scene_dir}")
    print()
    
    # List all files in the scene
    print("Files in scene:")
    for file_path in sorted(scene_dir.iterdir()):
        if file_path.is_file():
            print(f"  {file_path.name}")
    
    print()
    print("=" * 70)
    
    # Find triplets
    triplets = find_triplets_in_scene(scene_dir)
    
    print(f"\nFound {len(triplets)} complete triplet(s):")
    print()
    
    for i, files in enumerate(triplets, 1):
        print(f"Triplet {i} (stem: {files['stem']}):")
        print(f"  Splats:    {files['splats'].name}")
        print(f"  Reference: {files['reference'].name}")
        print(f"  Target:    {files['target'].name}")
        print()
    
    print("=" * 70)
    print("\nExpected behavior:")
    print("  - With current forward-only data: Should find 1 triplet")
    print("  - With bidirectional data: Should find 2 triplets")
    print(f"\nActual result: Found {len(triplets)} triplet(s)")
    
    if len(triplets) == 1:
        print("✓ Correct! (forward-only dataset)")
    elif len(triplets) == 2:
        print("✓ Correct! (bidirectional dataset)")
    else:
        print("✗ Unexpected number of triplets")

if __name__ == "__main__":
    test_triplet_detection()
