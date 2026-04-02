import numpy as np
import time

def slow_boundary_filter(S, H, W, boundary_threshold):
    boundary_mask = np.ones(S * H * W, dtype=bool)
    for s in range(S):
        frame_offset = s * H * W
        # Top and bottom
        boundary_mask[frame_offset : frame_offset + boundary_threshold * W] = False
        boundary_mask[frame_offset + (H - boundary_threshold) * W : frame_offset + H * W] = False
        # Left and right (per row)
        for h in range(boundary_threshold, H - boundary_threshold):
            row_start = frame_offset + h * W
            boundary_mask[row_start : row_start + boundary_threshold] = False
            boundary_mask[row_start + W - boundary_threshold : row_start + W] = False
    return boundary_mask

def fast_boundary_filter(S, H, W, boundary_threshold):
    mask_3d = np.ones((S, H, W), dtype=bool)
    if boundary_threshold > 0:
        mask_3d[:, :boundary_threshold, :] = False
        mask_3d[:, -boundary_threshold:, :] = False
        mask_3d[:, :, :boundary_threshold] = False
        mask_3d[:, :, -boundary_threshold:] = False
    return mask_3d.flatten()

# Typical values from VGGT
S = 10
H = 518
W = 518
BT = 20

# Warmup
slow_boundary_filter(1, 100, 100, 5)
fast_boundary_filter(1, 100, 100, 5)

start = time.perf_counter()
res_slow = slow_boundary_filter(S, H, W, BT)
end = time.perf_counter()
print(f"Slow: {end - start:.6f}s")

start = time.perf_counter()
res_fast = fast_boundary_filter(S, H, W, BT)
end = time.perf_counter()
print(f"Fast: {end - start:.6f}s")

assert np.array_equal(res_slow, res_fast)
print("Verification successful!")
