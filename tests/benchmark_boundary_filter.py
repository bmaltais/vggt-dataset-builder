import numpy as np
import time

def original_boundary_filter(valid_mask_all, S, H, W, boundary_threshold):
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
    return valid_mask_all & boundary_mask

def optimized_boundary_filter(valid_mask_all, S, H, W, boundary_threshold):
    if boundary_threshold > 0:
        boundary_mask_2d = np.ones((H, W), dtype=bool)
        boundary_mask_2d[:boundary_threshold, :] = False
        boundary_mask_2d[-boundary_threshold:, :] = False
        boundary_mask_2d[:, :boundary_threshold] = False
        boundary_mask_2d[:, -boundary_threshold:] = False

        # Use a reshaped view to apply the 2D mask to all frames in-place
        valid_mask_all.reshape(S, H, W)[:] &= boundary_mask_2d
    return valid_mask_all

# Setup benchmark
S, H, W = 10, 518, 518
boundary_threshold = 20
N = S * H * W

print(f"Benchmarking boundary filtering for {S} frames of {H}x{W} (Total points: {N})")

# Original
valid_mask_orig = np.ones(N, dtype=bool)
start = time.perf_counter()
res_orig = original_boundary_filter(valid_mask_orig, S, H, W, boundary_threshold)
end = time.perf_counter()
orig_time = end - start
print(f"Original time: {orig_time:.6f}s")

# Optimized
valid_mask_opt = np.ones(N, dtype=bool)
start = time.perf_counter()
res_opt = optimized_boundary_filter(valid_mask_opt, S, H, W, boundary_threshold)
end = time.perf_counter()
opt_time = end - start
print(f"Optimized time: {opt_time:.6f}s")

print(f"Speedup: {orig_time / opt_time:.2f}x")

# Verification
assert np.array_equal(res_orig, res_opt)
print("Verification passed!")
