import numpy as np
import pytest

def original_boundary_filter(S, H, W, boundary_threshold):
    boundary_mask = np.ones(S * H * W, dtype=bool)
    for s in range(S):
        frame_offset = s * H * W
        # Top and bottom
        boundary_mask[frame_offset : frame_offset + boundary_threshold * W] = (
            False
        )
        boundary_mask[
            frame_offset + (H - boundary_threshold) * W : frame_offset + H * W
        ] = False
        # Left and right (per row)
        for h in range(boundary_threshold, H - boundary_threshold):
            row_start = frame_offset + h * W
            boundary_mask[row_start : row_start + boundary_threshold] = False
            boundary_mask[
                row_start + W - boundary_threshold : row_start + W
            ] = False
    return boundary_mask

def test_boundary_filter_parity():
    S, H, W = 2, 100, 100
    threshold = 10

    # Original logic
    mask_orig = original_boundary_filter(S, H, W, threshold)

    # New logic
    boundary_mask_2d = np.ones((H, W), dtype=bool)
    boundary_mask_2d[:threshold, :] = False
    boundary_mask_2d[-threshold:, :] = False
    boundary_mask_2d[:, :threshold] = False
    boundary_mask_2d[:, -threshold:] = False
    mask_new = np.broadcast_to(boundary_mask_2d, (S, H, W)).reshape(-1)

    assert np.array_equal(mask_orig, mask_new)

def test_depth_filter_parity():
    N = 1000
    points = np.random.randn(N, 3).astype(np.float32)
    max_depth = 5.0

    # Original logic
    mask_orig = np.linalg.norm(points, axis=1) <= max_depth

    # New logic
    mask_new = np.sum(points**2, axis=1) <= (max_depth ** 2)

    # Use small epsilon for float comparison if needed, but here it should be exact or very close
    assert np.array_equal(mask_orig, mask_new)
