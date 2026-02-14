## 2026-02-11 - [Bulk Binary Writing for Point Clouds]
**Learning:** Writing large binary files (like PLY) point-by-point in Python using `struct.pack` is extremely slow due to the overhead of millions of Python function calls. NumPy structured arrays can be used to pack the data in memory and write it in a single `.tobytes()` operation, providing a 75x+ speedup.
**Action:** Always look for loops performing binary packing; replace with NumPy structured arrays for bulk I/O.

## 2026-02-11 - [Vectorized Background Filtering]
**Learning:** Redundant conversions between float (0-1) and uint8 (0-255) for color thresholding add significant overhead and memory pressure when processing millions of points. Float comparisons are faster and avoid extra allocations.
**Action:** Use direct float comparisons for color filtering when the source data is already floating-point.

## 2026-02-12 - [Fast GPU-to-CPU Readback]
**Learning:** Reading float32 textures from GPU to CPU is 4x bandwidth-heavy and requires expensive CPU-side clipping and scaling. Moving alpha premultiplication and color scaling to the fragment shader allows using a 3-channel uint8 texture, providing a ~24x speedup for the readback operation and saving significant CPU cycles.
**Action:** Move final image processing to the GPU and use uint8 textures for final render outputs.

## 2026-02-12 - [Optimized Background Filtering]
**Learning:** `np.floor(colors * 255.0) > 240` can be mathematically simplified to `colors >= 241/255.0`. This avoids creating a full-size float32 copy of the array and multiple intermediate boolean masks, reducing memory pressure and improving execution speed by ~20%.
**Action:** Simplify image thresholding math to avoid redundant array allocations.

## 2026-02-12 - [Redundant Point Cloud Extraction]
**Learning:** In bidirectional rendering pipelines, each frame is often used as a source for multiple target views. Performing point cloud extraction, sky filtering, and background masking repeatedly for the same source frame adds massive redundant overhead in NumPy indexing and I/O.
**Action:** Pre-calculate and cache filtered point cloud data once per frame; delete original large arrays immediately to minimize peak memory usage.

## 2026-02-13 - [Deferred High-Resolution Processing]
**Learning:** Allocating stacks of upsampled depth maps and world point clouds for an entire scene (e.g., 100+ frames) at once leads to massive peak memory spikes that can cause OOM errors. Moving these expensive operations inside a per-frame loop and deferring them until actually needed reduces peak memory by ~10x and avoids work for cached frames.
**Action:** Defer high-resolution allocations and expensive post-processing to the last possible moment; process on a per-frame basis to keep memory footprint flat.

## 2026-02-13 - [Sparse Unprojection]
**Learning:** Unprojecting a full HxW depth map to a (H,W,3) world point array is inefficient when many pixels are invalid or filtered. Vectorizing the unprojection to operate only on masked valid points avoids large meshgrid allocations and redundant matrix multiplications.
**Action:** Use masked indexing to unproject only valid depth points into world space.

## 2026-02-14 - [Redundant Image Loading During Rendering]
**Learning:** When `--upsample-depth` is enabled, images are loaded from disk and resized TWICE: once during pre-calculation for color extraction, and again in `render_and_save_pair()` to save reference/target images. This redundant disk I/O and BICUBIC resizing adds 50%+ overhead. Caching the resized PIL Image object in frame_data eliminates the second load entirely.
**Action:** Cache expensive image operations in intermediate data structures to avoid redundant disk I/O and CPU-intensive operations (resize, color conversion) when the same data is needed later in the pipeline.
