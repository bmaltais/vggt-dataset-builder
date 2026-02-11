## 2026-02-11 - [Bulk Binary Writing for Point Clouds]
**Learning:** Writing large binary files (like PLY) point-by-point in Python using `struct.pack` is extremely slow due to the overhead of millions of Python function calls. NumPy structured arrays can be used to pack the data in memory and write it in a single `.tobytes()` operation, providing a 75x+ speedup.
**Action:** Always look for loops performing binary packing; replace with NumPy structured arrays for bulk I/O.

## 2026-02-11 - [Vectorized Background Filtering]
**Learning:** Redundant conversions between float (0-1) and uint8 (0-255) for color thresholding add significant overhead and memory pressure when processing millions of points. Float comparisons are faster and avoid extra allocations.
**Action:** Use direct float comparisons for color filtering when the source data is already floating-point.
