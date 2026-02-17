from pathlib import Path
import numpy as np
import pytest
from dataset_utils import FrameCacheManager


def test_cache_smoke():
    """Test frame cache save/load functionality with FrameCacheManager."""
    scene_cache = Path(".cache") / "smoke_scene"

    # Clean up cache before test
    import shutil

    if scene_cache.exists():
        shutil.rmtree(scene_cache)

    args_hash = "test_hash_123"
    cache_mgr = FrameCacheManager(scene_cache, args_hash)

    # Create a temp image file for testing
    test_image_path = Path(".cache") / "test_image.tmp"
    test_image_path.parent.mkdir(parents=True, exist_ok=True)
    test_image_path.write_bytes(b"test image data")

    frame = {
        "points": np.random.rand(5, 3).astype(np.float32),
        "colors": np.random.rand(5, 3).astype(np.float32),
        "confidences": np.random.rand(5).astype(np.float32),
        "s0": 0.123,
    }

    print("Saving cache using FrameCacheManager")
    cache_mgr.save_frame_data(test_image_path, frame)

    assert cache_mgr.is_frame_cached(test_image_path), "Frame should be cached"

    loaded = cache_mgr.load_frame_data(test_image_path)
    print("Loaded keys:", list(loaded.keys()) if loaded else "None")
    print("points shape:", loaded["points"].shape)
    print("colors shape:", loaded["colors"].shape)
    print("confidences shape:", loaded["confidences"].shape)
    print("s0:", loaded.get("s0"))

    # Assertions to verify cache works correctly
    assert "points" in loaded
    assert "colors" in loaded
    assert "confidences" in loaded
    assert loaded["points"].shape == (5, 3)
    assert loaded["colors"].shape == (5, 3)
    assert loaded["confidences"].shape == (5,)
    assert loaded["s0"] == 0.123
    # Clean up
    import shutil

    if scene_cache.exists():
        shutil.rmtree(scene_cache)
    if test_image_path.exists():
        test_image_path.unlink()
