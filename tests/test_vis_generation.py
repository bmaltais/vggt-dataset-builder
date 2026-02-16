import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from unittest.mock import MagicMock, patch

from build_warp_dataset import get_triplet_paths, render_and_save_pair

def test_get_triplet_paths_includes_vis():
    scene_dir = Path("output/scene1")
    stem = "image1"
    ext = "jpg"
    paths = get_triplet_paths(scene_dir, stem, ext)

    assert "vis" in paths
    assert paths["vis"] == scene_dir / "image1_vis.jpg"

@patch("build_warp_dataset.Image.open")
@patch("build_warp_dataset.Image.new")
@patch("build_warp_dataset.Image.fromarray")
def test_render_and_save_pair_vis_logic(mock_fromarray, mock_new, mock_open, tmp_path):
    # Setup arguments
    args = argparse.Namespace(
        save_vis=True,
        force_output=False,
        upsample_depth=False,
        save_confidence=False,
        save_ply=False,
        preprocess_mode="crop",
        auto_s0=False
    )

    scene_output_dir = tmp_path / "output"
    scene_output_dir.mkdir()

    image_paths = [Path("input/1.jpg"), Path("input/2.jpg")]

    # Mock source/target frame data
    source_frame_data = {
        "points": np.zeros((10, 3)),
        "colors": np.zeros((10, 3)),
        "confidences": np.zeros(10),
        "img_cached": None
    }
    target_frame_data = {
        "img_cached": None
    }

    # Mock renderer and its output
    mock_renderer = MagicMock()
    # model_render output (numpy array)
    splats_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_renderer.render.return_value = splats_image

    # Mock PIL images
    mock_ref_img = MagicMock(spec=Image.Image)
    mock_ref_img.size = (100, 100)
    mock_ref_img.convert.return_value = mock_ref_img
    mock_tar_img = MagicMock(spec=Image.Image)
    mock_tar_img.size = (100, 100)
    mock_tar_img.convert.return_value = mock_tar_img
    mock_open.side_effect = [mock_tar_img, mock_ref_img, mock_ref_img, mock_tar_img]

    mock_splat_img = MagicMock(spec=Image.Image)
    mock_splat_img.size = (100, 100)
    mock_fromarray.return_value = mock_splat_img

    mock_combined_img = MagicMock(spec=Image.Image)
    mock_new.return_value = mock_combined_img

    # Other args
    extrinsic_batch = np.zeros((2, 3, 4))
    intrinsic_render_batch = [np.eye(3), np.eye(3)]
    preprocess_metas = [{}, {}]

    # We need to mock restore_to_original_resolution because upsample_depth is False
    with patch("build_warp_dataset.restore_to_original_resolution") as mock_restore:
        mock_restore.return_value = splats_image

        # Call the function
        # source_idx=0, target_idx=1
        render_and_save_pair(
            0, 1, image_paths, scene_output_dir, "jpg", {}, args,
            source_frame_data, target_frame_data, extrinsic_batch,
            intrinsic_render_batch, preprocess_metas, mock_renderer,
            100, 100, None
        )

    # Verify visualization was created
    # Image.new should be called with (300, 100)
    mock_new.assert_called_once_with("RGB", (300, 100))

    # combined.paste should be called 3 times
    assert mock_combined_img.paste.call_count == 3

    # combined.save should be called for the vis file
    vis_path = scene_output_dir / "2_vis.jpg"
    mock_combined_img.save.assert_any_call(vis_path, quality=90, optimize=True)
