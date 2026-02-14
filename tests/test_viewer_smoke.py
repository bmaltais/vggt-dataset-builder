#!/usr/bin/env python3
"""Smoke test: render a synthetic point cloud with HoleFillingRenderer."""
import numpy as np
from pathlib import Path
from PIL import Image

from hole_filling_renderer import HoleFillingRenderer

def test_viewer_smoke():
    """Test rendering a synthetic point cloud with HoleFillingRenderer."""
    out = Path("output") / "viewer_smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Create synthetic points (grid on XY, varying Z)
    xs = np.linspace(-1.0, 1.0, 128)
    ys = np.linspace(-1.0, 1.0, 128)
    xv, yv = np.meshgrid(xs, ys)
    zv = 2.0 + 0.5 * np.sin(3.0 * xv) * np.cos(3.0 * yv)
    points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1).astype(np.float32)

    colors = np.zeros_like(points, dtype=np.float32)
    colors[:, 0] = (xv.flatten() + 1.0) * 0.5
    colors[:, 1] = (yv.flatten() + 1.0) * 0.5
    colors[:, 2] = 0.8

    confs = np.ones((points.shape[0],), dtype=np.float32)

    # Create renderer and render
    width, height = 512, 512
    renderer = HoleFillingRenderer(width, height)

    view = np.eye(4, dtype=np.float32)
    proj = np.eye(4, dtype=np.float32)
    fov_y = 0.8

    frame = renderer.render(points, colors, confs, view, proj, fov_y)
    Image.fromarray(frame).save(out)
    print(f"Saved smoke render to {out}")
