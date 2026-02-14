#!/usr/bin/env python3
"""Test script to verify triplet detection in aitoolkit and modelscope scripts."""

from pathlib import Path
import pytest

from dataset_utils import find_triplets_in_scene

def test_triplet_detection(tmp_path):
    """Test the triplet detection logic using a temporary directory."""
    
    scene_dir = tmp_path / "scene_01"
    scene_dir.mkdir()
    
    # 1. Create a complete triplet
    (scene_dir / "img1_splats.jpg").write_text("dummy")
    (scene_dir / "img1_reference.jpg").write_text("dummy")
    (scene_dir / "img1_target.jpg").write_text("dummy")
    
    # 2. Create an incomplete triplet (missing target)
    (scene_dir / "img2_splats.jpg").write_text("dummy")
    (scene_dir / "img2_reference.jpg").write_text("dummy")
    
    # 3. Create another complete triplet with different extension
    (scene_dir / "img3_splats.png").write_text("dummy")
    (scene_dir / "img3_reference.png").write_text("dummy")
    (scene_dir / "img3_target.png").write_text("dummy")

    # 4. Create distractor files
    (scene_dir / "img1_reference.ply").write_text("dummy")
    (scene_dir / "notes.txt").write_text("dummy")
    
    # Find triplets
    triplets = find_triplets_in_scene(scene_dir)
    
    # Assertions
    assert len(triplets) == 2
    
    # Verify first triplet
    assert triplets[0]["stem"] == "img1"
    assert triplets[0]["splats"].name == "img1_splats.jpg"
    assert triplets[0]["reference"].name == "img1_reference.jpg"
    assert triplets[0]["target"].name == "img1_target.jpg"
    
    # Verify second triplet
    assert triplets[1]["stem"] == "img3"
    assert triplets[1]["splats"].name == "img3_splats.png"
    assert triplets[1]["reference"].name == "img3_reference.png"
    assert triplets[1]["target"].name == "img3_target.png"

if __name__ == "__main__":
    # Allow running as a script
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_triplet_detection(Path(tmpdir))
        print("✓ test_triplet_detection passed!")
