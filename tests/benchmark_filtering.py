import numpy as np
import time

def original_filtering(S, H, W, points, colors, conf, boundary_threshold, max_depth, mask_black_bg):
    valid_mask_all = conf > 0.5

    # Apply max depth filtering
    if max_depth > 0:
        depth_all = np.linalg.norm(points, axis=1)
        valid_mask_all = valid_mask_all & (depth_all <= max_depth)

    # Apply boundary filtering
    if boundary_threshold > 0:
        boundary_mask = np.ones(S * H * W, dtype=bool)
        for s in range(S):
            frame_offset = s * H * W
            boundary_mask[frame_offset : frame_offset + boundary_threshold * W] = False
            boundary_mask[
                frame_offset + (H - boundary_threshold) * W : frame_offset + H * W
            ] = False
            for h in range(boundary_threshold, H - boundary_threshold):
                row_start = frame_offset + h * W
                boundary_mask[row_start : row_start + boundary_threshold] = False
                boundary_mask[
                    row_start + W - boundary_threshold : row_start + W
                ] = False
        valid_mask_all = valid_mask_all & boundary_mask

    # Apply black background filtering
    if mask_black_bg:
        black_mask = colors.sum(axis=1) >= (16 / 255.0)
        valid_mask_all = valid_mask_all & black_mask

    return valid_mask_all

def optimized_filtering(S, H, W, points, colors, conf, boundary_threshold, max_depth, mask_black_bg):
    valid_mask_all = conf > 0.5

    # Apply max depth filtering
    if max_depth > 0:
        # Use squared distance to avoid np.linalg.norm's square root
        dist_sq = np.einsum('ij,ij->i', points, points)
        valid_mask_all &= (dist_sq <= max_depth**2)

    # Apply boundary filtering (Vectorized in-place)
    if boundary_threshold > 0:
        mask_view = valid_mask_all.reshape(S, H, W)
        mask_view[:, :boundary_threshold, :] = False
        mask_view[:, -boundary_threshold:, :] = False
        mask_view[:, :, :boundary_threshold] = False
        mask_view[:, :, -boundary_threshold:] = False

    # Apply black background filtering
    if mask_black_bg:
        # Explicit channel addition is faster than sum(axis=1)
        color_sum = colors[:, 0] + colors[:, 1] + colors[:, 2]
        valid_mask_all &= (color_sum >= (16 / 255.0))

    return valid_mask_all

def benchmark():
    S, H, W = 10, 1080, 1920
    N = S * H * W
    points = np.random.randn(N, 3).astype(np.float32)
    colors = np.random.rand(N, 3).astype(np.float32)
    conf = np.random.rand(N).astype(np.float32)

    boundary_threshold = 10
    max_depth = 5.0
    mask_black_bg = True

    print(f"Benchmarking with S={S}, H={H}, W={W} ({N} points)")

    start = time.time()
    res1 = original_filtering(S, H, W, points, colors, conf, boundary_threshold, max_depth, mask_black_bg)
    end = time.time()
    print(f"Original: {end - start:.4f}s")

    start = time.time()
    res2 = optimized_filtering(S, H, W, points, colors, conf, boundary_threshold, max_depth, mask_black_bg)
    end = time.time()
    print(f"Optimized: {end - start:.4f}s")

    np.testing.assert_array_equal(res1, res2)
    print("Results match!")

if __name__ == "__main__":
    benchmark()
