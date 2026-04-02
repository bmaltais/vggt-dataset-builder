import numpy as np
import time

def slow_mask_combine(mask_size, num_masks):
    masks = [np.random.choice([True, False], size=mask_size) for _ in range(num_masks)]
    start = time.perf_counter()
    valid_mask = masks[0]
    for i in range(1, num_masks):
        valid_mask = valid_mask & masks[i]
    end = time.perf_counter()
    return end - start

def fast_mask_combine(mask_size, num_masks):
    masks = [np.random.choice([True, False], size=mask_size) for _ in range(num_masks)]
    start = time.perf_counter()
    valid_mask = masks[0]
    for i in range(1, num_masks):
        valid_mask &= masks[i]
    end = time.perf_counter()
    return end - start

mask_size = 10 * 518 * 518
num_masks = 5

t_slow = slow_mask_combine(mask_size, num_masks)
print(f"Slow (reassignment): {t_slow:.6f}s")

t_fast = fast_mask_combine(mask_size, num_masks)
print(f"Fast (in-place): {t_fast:.6f}s")
