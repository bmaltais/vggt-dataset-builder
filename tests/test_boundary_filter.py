import numpy as np
import pytest


def original_boundary_filter(S, H, W, boundary_threshold):
    boundary_mask = np.ones(S * H * W, dtype=bool)
    for s in range(S):
        frame_offset = s * H * W
        # Top and bottom
        boundary_mask[frame_offset : frame_offset + boundary_threshold * W] = False
        boundary_mask[
            frame_offset + (H - boundary_threshold) * W : frame_offset + H * W
        ] = False
        # Left and right (per row)
        for h in range(boundary_threshold, H - boundary_threshold):
            row_start = frame_offset + h * W
            boundary_mask[row_start : row_start + boundary_threshold] = False
            boundary_mask[row_start + W - boundary_threshold : row_start + W] = False
    return boundary_mask


def optimized_boundary_filter(S, H, W, boundary_threshold):
    boundary_mask = np.ones((S, H, W), dtype=bool)
    if boundary_threshold > 0:
        boundary_mask[:, :boundary_threshold, :] = False
        boundary_mask[:, -boundary_threshold:, :] = False
        boundary_mask[:, :, :boundary_threshold] = False
        boundary_mask[:, :, -boundary_threshold:] = False
    return boundary_mask.ravel()


@pytest.mark.parametrize(
    "S, H, W, threshold",
    [
        (1, 10, 10, 2),
        (2, 20, 30, 5),
        (1, 100, 100, 10),
        (3, 50, 50, 0),
        (1, 5, 5, 2),
    ],
)
def test_boundary_filter_equivalence(S, H, W, threshold):
    """Ensure optimized boundary filter matches original nested loop implementation."""
    res_orig = original_boundary_filter(S, H, W, threshold)
    res_opt = optimized_boundary_filter(S, H, W, threshold)

    assert res_orig.shape == res_opt.shape
    assert np.array_equal(res_orig, res_opt)


def test_boundary_filter_all_false():
    """Test case where threshold covers the entire image."""
    S, H, W = 1, 10, 10
    threshold = 5  # 5 from top, 5 from bottom = 10 (all)
    res = optimized_boundary_filter(S, H, W, threshold)
    assert not np.any(res)


def test_boundary_filter_no_op():
    """Test case where threshold is 0."""
    S, H, W = 1, 10, 10
    threshold = 0
    res = optimized_boundary_filter(S, H, W, threshold)
    assert np.all(res)
