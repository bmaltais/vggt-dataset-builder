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

## 2026-02-15 - [Multi-Pass Interpolation & Correct Caching]
**Learning:** Combining multiple spatial transformations (e.g., restore from crop + resize to output) into a single interpolation pass significantly reduces CPU overhead and resampling artifacts. Additionally, caching PIL Image objects across the bidirectional pipeline avoids massive redundant disk I/O and BICUBIC resizing, while fixing a critical bug where source frame data was incorrectly used for target frame outputs.
**Action:** Always look for sequential interpolations that can be merged; ensure caching logic correctly distinguishes between source and target frame metadata in rendering pipelines.

## 2026-02-15 - [Lazy Pre-calculation & Active Memory Management]
**Learning:** In pipelines with dependency chains (like bidirectional rendering), pre-calculating all intermediate states upfront is wasteful if many outputs already exist. Additionally, storing PIL Image objects for the entire scene can lead to OOM; clearing them from the cache immediately after their last use in the loop keeps memory usage constant.
**Action:** Use "needed frames" sets to skip redundant pre-calculation; actively clear large objects from caches during iterative processing once their dependency lifecycle ends.

## 2026-02-15 - [Headless GPU Rendering]
**Learning:** Running ModernGL-based rendering or tests in headless Linux environments will fail with `Exception: (standalone) XOpenDisplay: cannot open display`. Using `xvfb-run` provides the necessary virtual X server for the OpenGL context to initialize correctly.
**Action:** Always wrap Python commands that use `HoleFillingRenderer` in `xvfb-run` when working in a remote or CI environment.

## 2026-02-16 - [GPU-Accelerated Masking & Redundancy Removal]
**Learning:** Moving alpha premultiplication and distance masking to the GPU significantly reduces CPU-side mathematical overhead (avoiding millions of multiplications per frame). While switching to uint8 readback is ideal for bandwidth, preserving the 4-channel float32 format can be necessary for architectural compatibility and precision requirements. Even without changing the texture format, removing redundant CPU-side masking provides a solid performance win.
**Action:** Always check if shaders are already performing operations that are being redundantly repeated on the CPU; offload expensive masking to the GPU whenever possible.

## 2026-02-17 - [Unnecessary Image Mode Conversion for Metadata]
**Learning:** In `build_preprocess_metadata()`, every image was being converted from RGBA → RGB even though only the image dimensions were needed. PIL's `Image.size` property works on any image mode without conversion overhead, so these operations were purely wasteful. Removing the RGBA composite and RGB conversion operations eliminates redundant CPU work for ~O(N) images per scene.
**Action:** Always check if pixel-level operations (color conversions, compositing) are actually needed; dimension queries (`Image.size`) work natively on any image format and should skip conversion overhead.


## 2026-02-18 - [Optimized Color Summation for Background Filtering]
**Learning:** For small, fixed-dimension arrays like RGB (Nx3), NumPy's generalized `sum(axis=1)` reduction is significantly slower (~4x) than explicit channel-wise addition (`c0 + c1 + c2`) due to reduction overhead. However, when working with `uint8` data, direct addition will cause overflow wrapping; explicit casting (e.g., to `uint32`) is required before manual addition to ensure correctness while maintaining performance.
**Action:** Replace `sum(axis=1)` with explicit channel addition for small RGB arrays to improve filtering performance; ensure proper casting for integer types to prevent overflow.

## 2026-02-18 - [Vectorized Boundary Filtering & Early Exit Rescale]
**Learning:** Nested Python loops for boundary filtering on large point clouds can be replaced with NumPy slicing on a reshaped (S, H, W) array for a ~2.2x speedup. Additionally, checking image dimensions before expensive color conversions in rescaling functions provides a massive (~380x) speedup for images already within size limits.
**Action:** Use vectorized slicing for geometric filtering; always perform dimension-based early exits before pixel-level operations.
