#!/usr/bin/env python3
"""Test script to verify bidirectional file detection logic."""

from pathlib import Path

def test_file_detection():
    """Test the file detection logic for bidirectional processing."""
    
    # Test parameters
    scene_dir = Path("output/01")
    next_name = "IMG_6636_rescaled"
    curr_name = "IMG_6635_rescaled"
    output_ext = "jpg"
    no_confidence = False
    save_ply = True
    
    print("Testing Forward Direction File Detection")
    print("=" * 50)
    
    # Forward direction files
    forward_splats_path = scene_dir / f"{next_name}_splats.{output_ext}"
    forward_target_path = scene_dir / f"{next_name}_target.{output_ext}"
    forward_reference_path = scene_dir / f"{next_name}_reference.{output_ext}"
    forward_conf_path = scene_dir / f"{next_name}_confidence.png"
    forward_ply_path = scene_dir / f"{next_name}_reference.ply"
    
    print(f"Splats:     {forward_splats_path.exists()} - {forward_splats_path}")
    print(f"Target:     {forward_target_path.exists()} - {forward_target_path}")
    print(f"Reference:  {forward_reference_path.exists()} - {forward_reference_path}")
    print(f"Confidence: {forward_conf_path.exists()} - {forward_conf_path}")
    print(f"PLY:        {forward_ply_path.exists()} - {forward_ply_path}")
    
    forward_missing = (
        not forward_splats_path.exists() or 
        not forward_target_path.exists() or 
        not forward_reference_path.exists() or
        (not no_confidence and not forward_conf_path.exists()) or
        (save_ply and not forward_ply_path.exists())
    )
    
    print(f"\nForward missing: {forward_missing}")
    print(f"Would process forward: {forward_missing}")
    
    print("\n" + "=" * 50)
    print("Testing Reverse Direction File Detection")
    print("=" * 50)
    
    # Reverse direction files
    reverse_splats_path = scene_dir / f"{curr_name}_splats.{output_ext}"
    reverse_target_path = scene_dir / f"{curr_name}_target.{output_ext}"
    reverse_reference_path = scene_dir / f"{curr_name}_reference.{output_ext}"
    reverse_conf_path = scene_dir / f"{curr_name}_confidence.png"
    reverse_ply_path = scene_dir / f"{curr_name}_reference.ply"
    
    print(f"Splats:     {reverse_splats_path.exists()} - {reverse_splats_path}")
    print(f"Target:     {reverse_target_path.exists()} - {reverse_target_path}")
    print(f"Reference:  {reverse_reference_path.exists()} - {reverse_reference_path}")
    print(f"Confidence: {reverse_conf_path.exists()} - {reverse_conf_path}")
    print(f"PLY:        {reverse_ply_path.exists()} - {reverse_ply_path}")
    
    reverse_missing = (
        not reverse_splats_path.exists() or 
        not reverse_target_path.exists() or 
        not reverse_reference_path.exists() or
        (not no_confidence and not reverse_conf_path.exists()) or
        (save_ply and not reverse_ply_path.exists())
    )
    
    print(f"\nReverse missing: {reverse_missing}")
    print(f"Would process reverse: {reverse_missing}")
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Forward direction: {'NEEDS PROCESSING' if forward_missing else 'COMPLETE - WILL SKIP'}")
    print(f"Reverse direction: {'NEEDS PROCESSING' if reverse_missing else 'COMPLETE - WILL SKIP'}")

if __name__ == "__main__":
    test_file_detection()
