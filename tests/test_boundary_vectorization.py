import numpy as np
import pytest

def get_boundary_mask_original(S, H, W, boundary_threshold):
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
            boundary_mask[
                row_start + W - boundary_threshold : row_start + W
            ] = False
    return boundary_mask

def get_boundary_mask_vectorized(S, H, W, boundary_threshold):
    boundary_mask = np.ones(S * H * W, dtype=bool)
    mask_view = boundary_mask.reshape(S, H, W)
    mask_view[:, :boundary_threshold, :] = False
    mask_view[:, H - boundary_threshold :, :] = False
    mask_view[:, :, :boundary_threshold] = False
    mask_view[:, :, W - boundary_threshold :] = False
    return boundary_mask

@pytest.mark.parametrize("S, H, W, threshold", [
    (1, 10, 10, 2),
    (2, 20, 30, 5),
    (3, 100, 100, 10),
    (1, 518, 518, 14),
])
def test_boundary_mask_equivalence(S, H, W, threshold):
    original = get_boundary_mask_original(S, H, W, threshold)
    vectorized = get_boundary_mask_vectorized(S, H, W, threshold)
    assert np.array_equal(original, vectorized)

def test_boundary_mask_zero_threshold():
    S, H, W = 1, 10, 10
    # Vectorized code path for threshold=0 is not exercised in the node if threshold == 0 check is present,
    # but let's see what it does.
    vectorized = get_boundary_mask_vectorized(S, H, W, 0)
    assert np.all(vectorized)

if __name__ == "__main__":
    pytest.main([__file__])
