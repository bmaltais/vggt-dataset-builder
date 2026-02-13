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
