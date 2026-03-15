## 2026-02-18 - [Optimized PLY Scale and Rotation Allocations]
**Learning:** In `write_ply_basic`, allocating large `(N, 3)` and `(N, 4)` arrays for constant values (scales and identity quaternions) and then copying them into a structured array adds significant memory pressure and CPU overhead. Assigning constant scalars directly to the structured array columns is ~120x faster and more memory-efficient.
**Action:** Avoid allocating large intermediate arrays for constant values when filling structured arrays; use direct column assignment instead.

## 2026-02-18 - [Vectorized Boundary Filtering in ComfyUI Nodes]
**Learning:** Using nested Python loops to create boundary masks for multi-frame sequences is extremely slow (O(S*H)). Reshaping the flattened mask to `(S, H, W)` and using NumPy slicing provides a ~25x speedup and improves code readability.
**Action:** Always prefer NumPy slicing and reshaping over Python loops for spatial mask operations, even on flattened arrays.
