import time
import numpy as np
from hole_filling_renderer import HoleFillingRenderer
import moderngl


def benchmark_readback():
    width, height = 1920, 1080
    print(f"Benchmarking readback at {width}x{height}...")

    renderer = HoleFillingRenderer(width, height)

    # Mock data for rendering
    pts = np.random.rand(1000, 3).astype(np.float32)
    cols = np.random.rand(1000, 3).astype(np.float32)
    confs = np.random.rand(1000).astype(np.float32)
    view = np.eye(4, dtype=np.float32)
    proj = np.eye(4, dtype=np.float32)

    # Perform one render to ensure all resources are initialized
    renderer.render(pts, cols, confs, view, proj, 1.0)

    # Benchmark _read_final_color
    n_iters = 50
    start_time = time.perf_counter()
    for _ in range(n_iters):
        renderer._read_final_color()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / n_iters
    print(
        f"Average _read_final_color time over {n_iters} iterations: {avg_time*1000:.2f} ms"
    )


if __name__ == "__main__":
    benchmark_readback()
