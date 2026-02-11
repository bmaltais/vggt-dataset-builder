#!/usr/bin/env python3
"""Test script to verify bidirectional file detection logic."""

from pathlib import Path
import pytest

def check_files_missing(scene_dir, name, output_ext, no_confidence, save_ply):
    """Helper function mimicking the logic in build_warp_dataset.py."""
    splats_path = scene_dir / f"{name}_splats.{output_ext}"
    target_path = scene_dir / f"{name}_target.{output_ext}"
    reference_path = scene_dir / f"{name}_reference.{output_ext}"
    conf_path = scene_dir / f"{name}_confidence.png"
    ply_path = scene_dir / f"{name}_reference.ply"

    return (
        not splats_path.exists() or
        not target_path.exists() or
        not reference_path.exists() or
        (not no_confidence and not conf_path.exists()) or
        (save_ply and not ply_path.exists())
    )

def test_file_detection(tmp_path):
    """Test the file detection logic for bidirectional processing."""
    
    scene_dir = tmp_path / "01"
    scene_dir.mkdir()

    next_name = "IMG_6636_rescaled"
    curr_name = "IMG_6635_rescaled"
    output_ext = "jpg"
    
    # 1. Initially, all should be missing
    assert check_files_missing(scene_dir, next_name, output_ext, no_confidence=False, save_ply=True) is True
    assert check_files_missing(scene_dir, curr_name, output_ext, no_confidence=False, save_ply=True) is True
    
    # 2. Create forward files
    (scene_dir / f"{next_name}_splats.{output_ext}").write_text("dummy")
    (scene_dir / f"{next_name}_target.{output_ext}").write_text("dummy")
    (scene_dir / f"{next_name}_reference.{output_ext}").write_text("dummy")
    (scene_dir / f"{next_name}_confidence.png").write_text("dummy")
    (scene_dir / f"{next_name}_reference.ply").write_text("dummy")
    
    # Now forward should NOT be missing, but reverse should still be missing
    assert check_files_missing(scene_dir, next_name, output_ext, no_confidence=False, save_ply=True) is False
    assert check_files_missing(scene_dir, curr_name, output_ext, no_confidence=False, save_ply=True) is True
    
    # 3. Test with no_confidence=True
    # If we remove the confidence file, it should be missing if no_confidence=False
    (scene_dir / f"{next_name}_confidence.png").unlink()
    assert check_files_missing(scene_dir, next_name, output_ext, no_confidence=False, save_ply=True) is True
    # But NOT missing if no_confidence=True
    assert check_files_missing(scene_dir, next_name, output_ext, no_confidence=True, save_ply=True) is False

if __name__ == "__main__":
    # Allow running as a script
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file_detection(Path(tmpdir))
        print("✓ test_file_detection passed!")
